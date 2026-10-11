# DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`)

Support lives in `ucode/dsv4.uc`. Checkpoint used: `DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw`
(16 shards, ~115 GB, EXL3 `mul1` codebook, head 5 bpw).
Reference implementation / oracle: exllamav3's installed wheel
(`exllamav3/architecture/deepseek_v4.py`, `modules/dsv4.py`,
`modules/hyperconnections.py`, `modules/block_sparse_mlp_routing.py`). The reference runs fused Triton /
CUDA kernels (`dsa_attn`, `dsv4_compress`, `dsa_topk`); the semantics below were read out of those.

## Model

43 transformer layers, hidden 4096, vocab 129280, rms eps **1e-20** (fp32 variance), tied nothing.

**Residual: mHC** (identical to GLM-5.3): 4 fp32 streams `(T, 4, D)`; per site
`(post, comb, c) = hc_mix(streams)`, `c` (fp16) is normed (`attn_norm` / `ffn_norm`), the sublayer runs,
`hc_apply(streams, y, post, comb)`. Same `torch.hc_mix` / `torch.hc_apply` kernels. Embedding broadcasts into
the 4 streams; the head collapses them with `HyperHead`
(`pre = sigmoid(rms(flat) . fn^T * scale + base) + eps`, `x = sum_h pre_h * stream_h`), then `norm`, then the
5 bpw `head` linear.

**Attention** (MQA, ONE shared key/value head of 512 dims, **V = K**), 64 query heads:

- `q_res = rms_norm(wq_a(x), q_norm)` (1024), `q = wq_b(q_res)` -> `(T, 64, 512)`, unweighted per-head RMS
  norm of q (fp32), `kv = rms_norm(wkv(x), kv_norm)` (512).
- RoPE: GPT-J (interleaved pairs) on the trailing 64 of the 512 channels, applied to q and kv. Sliding layers use
  `theta = 10000` unscaled; CSA/HCA layers use `compress_rope_theta = 160000` with a yarn ramp
  (`util/rope.py:yarn_inv_freq`, no attention factor).
- Keys of a query at position p: the sliding window (p-127 .. p, raw kv rows) **plus** compressed pool entries.
- Scores `q.k * 512^-0.5`, an attention **sink** per head (`attn_sink`, an extra logit in the softmax
  denominator, contributes no value), softmax, `out = P @ K` (V is K).
- The output's rope slice is **de-rotated** by the query position (eq. 26: rotate by -p*freq).
- Grouped output projection: 8 groups of 8 heads: `wo_a.slice.g` (4096 -> 1024, one EXL3 tensor per group),
  concat -> 8192, `wo_b` (8192 -> 4096).
- Layer types from `compress_ratios[:43]`: `0` sliding (layers 0, 1, and the trailing ones), `4` CSA, `128` HCA.

**Compressor** (CSA m = 4 overlapping, HCA m = 128, plus a second one in the CSA *indexer*, width 128):
`wkv`, `wgate` project x to W (= 512, or 1024 for the overlapping scheme); rows are buffered until a window of
m is complete. `gate += ape` (per position in window), entry = `sum(kv * softmax(gate, over window))`;
overlapping (CSA): window w mixes its own second-half channels with the first-half channels of window w-1
(the first window has none: kv 0, gate -inf; the carried half keeps its `ape`), softmax over 2m rows. Entry ->
fp16 -> `rms_norm(compressor.norm)` -> rope (compress table) at position **w*m** (window start). Entry j is
visible to a query at p iff `(j+1)*m <= p+1`.

**Indexer (CSA only)**: once the pool holds more than `index_topk = 512` entries, scores
`sum_h w_h * relu(q_idx_h . k_idx) * (128^-0.5 * 64^-0.5)` with `q_idx = idx_wq_b(q_res)` roped with the
compress table, `w = idx_weights_proj(x)` (fp16 matrix, 4096 -> 64), over the indexer pool; top-512 entries
(causal-masked) replace the dense pool. Below that the dense pool is exactly the same set.

**MoE**: 256 routed experts (top 6) + 1 shared; expert/shared MLP = `w1` (gate) / `w3` (up) / `w2` (down),
clamped SwiGLU `silu(min(g, 10)) * clamp(u, -10, 10)`. Router: `sqrt(softplus(x . gate^T))` affinities; selection
on `affinity + gate.bias` (noaux_tc); weights = unbiased affinities of the selected experts, normalized,
x `routed_scaling_factor` (1.5). Layers 0-2 (**hash layers**) select by token id from the frozen
`gate.tid2eid` table (the learned gate still weights them). Reused from `glm5.uc`: `moe_forward`
(`torch.moe_coop` decode / `torch.moe_fused` prefill / composite), with a `router: 'sqrtsp'` branch in
`glm5.moe_route`.

## Vision (`ucode/dsv4_vision.uc`)

32-block ViT (dim 1024, 16 heads, patch 14, RMSNorm 1e-6, biased EXL3 q/k/v/wo, gated MLP, NEOX 2-D rotary with
h angles in the first and w angles in the second half, bidirectional, fp32 residual) + aligner (3x3 unfold,
w1 -> exact GELU -> w2). Preprocessing (`torch.image_info`, `torch.image_prepare` with `target_w/target_h/pad`):
aspect clamp (<= 8), upscale to 147456 px, token-budget resize (`safe_resize`), gray (127) PIL `ImageOps.pad`
or stretch for very wide images, [-1, 1] normalization.

The text model sees a BLOCK of embedding rows (`build_image_block`): lead pads (so IMAGE_START sits at
position % 4 == 3, a CSA compressor group boundary), IMAGE_START, aligner rows interleaved in column pairs,
IMAGE_NEW_LINE per grid row, pad rows, IMAGE_END; markers are the root tensors image_start/pad/newline/end.
Rules in the language model (`dsv4.uc`, `cache.mm` spans):
- the image proper (START .. END) is prefilled as exactly ONE chunk and attends bidirectionally inside it
  (window keys of the chunk are all visible; history keys keep each row's own 128-window; compressed pool
  rows stay causal) -- `torch.dsv4_mask(..., ncstart)`;
- every image row (lead pads too) routes its experts on `gate.bias_vl` instead of `gate.bias`, hash layers
  included (`glm5.moe_route`, `M.vlon`);
- token ids of image rows are marker ids (>= 2^30, per-image hash) so prefix reuse cannot confuse images.

**Not implemented**: video, remote image URLs, tower tensor parallelism.

## DSpark drafter (`mtp.*`, config `dspark_*`) — implemented

DeepSeek's own speculative-decoding mechanism (a DFlash-style block drafter, not classic MTP).
Per round: the target's stream mean after layers 40/41/42 (fp16, saturated) of every position goes
through `main_proj` (12288 -> 4096) + `main_norm`; each of the three drafter blocks keeps a main-kv
row per position (`wkv`, weighted kv norm, rope). One forward over `[seed, noise x 4]` (3 ordinary V4
blocks, sliding attention, 256-expert MoE; the 5 queries see the last 128 main rows plus the block,
non-causally) -> `hc_head` + norm -> the TARGET's head -> logits (5, V); greedy chain with the markov
bias `markov_w2(markov_w1[prev token])`. The target verifies `[seed, d1 .. dV]` (V = 4 by default)
and the engine accepts by the one-hot-draft rule (exact match at temperature 0).
`dspark.uc` is the eager reference; `dspark_tp.uc` runs it in the 3 x TP-2 pipeline (the drafter lives
with the last stage) as CUDA graphs: draft / verify / update.

## Execution and performance

`--tp --stages 3` (default for DeepSeek-V4 on >= 2 GPUs) = `dsv4_tp.uc`; `--layer-split` = `dsv4.uc`.
Numbers (six-GPU host, 6 x A5000/A5500, full model): see `OVERVIEW.md` section 5. Open work: prefill is host
bound (launches per layer), a fused prefill attention/compressor, vision.

## Tokenizer

DeepSeek's `tokenizer.json` pre-tokenizer is a Sequence of three `Split(Isolated)` stages (digit runs <= 3,
CJK runs, a letter/punctuation pattern) + a regex-less `ByteLevel`. `src/tokenizer.c` applies all splitting
stages in order (it used to take only the first). `tests/tokenizer_dsv4.uc`.

## Oracle and tests

`tests/oracle/make_dsv4_oracle.py` (exllamav3 wheel, all 6 GPUs, ~10 min incl. 5 min load) writes three
fixtures `dsv4_oracle_{short,medium,long}.safetensors` (16 / 701 / 2601 tokens: the long one has CSA pools
beyond `index_topk`): per block `attn_in/out`, `mlp_in/out`, `x`, hc post/comb, routes, attention internals
(`a{i}_q/kv/out/pool_*/idx` for layers 0-4), logits, and for `short` a greedy continuation.
`tests/dsv4_full.uc` compares block by block, logits and a teacher-forced decode.

## Status / numbers

Composite-op implementation; decode and prefill are host-launch bound until the
attention / compressor / indexer are fused (open work).
