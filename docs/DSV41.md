# DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`)

Support lives in `ucode/dsv41.uc` (model), `ucode/chat_dsv41.uc` + `chat_dsv41.jinja` (chat format).
Checkpoint used: `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` (17 shards, 330 GB). Reference: DeepSeek's
`inference/model.py`, `engram.py`, `kernel.py` of `deepseek-ai/DeepSeek-V4.1-Flash`. There is no exllamav3
implementation of V4.1 (the pack's README asks for "a recent ExLlamaV3 with V4.1 / CED support" that is not
public), so the oracles are DeepSeek's own code run on the CPU (`tests/oracle/make_dsv41_*.py`).

## What is in the pack

| Part | Format | Size |
|---|---|---|
| 40 layers x 384 routed experts (w1, w3, w2) | EXL3 trellis, `mul1` codebook, **K = 1..6 per matrix** (SAGE) | 108.9 GB |
| attention (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`), shared expert, indexer `wq_b`, Engram `wkv` | block-scaled FP8 (e4m3, one e8m0 scale per 32 x 32 block) | ~7 GB |
| compressors, indexer `wk` / `weights_proj`, gates, embed, head | bf16 | 3.6 GB |
| Engram tables (layers 1 and 14): 2 x 384M rows x 256 | FP8 + one e8m0 scale per 32 values | **204 GB** (shards 16, 17) |
| `mtp.*` (DSpark drafter, 3 stages, 128 experts each) | EXL3 + FP8 | 7.4 GB |
| vision tower | EXL3 / bf16 | 0.8 GB |

The Engram tables are read from disk, never loaded (see below). Nothing else of the 330 GB is larger than the
experts; the text model without tables, drafter and tower is 110.6 GiB.

## Architecture (differences from V4, `docs/DSV4.md`)

- hidden 5120, 40 layers, 64 query heads (head dim 512, rope 64), q rank 1280, o rank 1024 / 8 groups, 384
  routed experts (top 6, moe size 2304) + 1 shared, `rms_norm_eps` 1e-20. No hash-routed layers; the router is
  `sqrt(softplus)` with the selection bias, as in V4.
- **mHC one site late.** Every site computes `pre`, `post`, `comb` from the stream, but the collapse that feeds a
  sublayer uses the `pre` of the *previous* site: the attention input of layer l is collapsed with the FFN `pre`
  of layer l-1 (identity for layer 0), the FFN input with the attention `pre` of the same layer. The final
  collapse (no HyperHead) uses the last FFN `pre`. `post` / `comb` apply as in V4.
- **Attention.** Window 128 + compressed keys. `compress_ratios`: layers 0-1 none, 2-19 ratio 2, 20-39 ratio 1.
  - ratio 2: non-overlapping pairs pooled with a softmax over the two gate values per channel (no `ape`),
    RMS-normed, rope at the position of the pair's first token; ratio 1: `norm(wkv(x))` per token.
  - Only the **kv source** layers (2, 8, 14, 20) own a compressor and a pool; every other layer reads the
    pool of the last source before it (cross-GPU consumers keep a mirror).
  - The **index source** layers (2, 8, 14, 20, 24, 28, 32, 36) score the pool with a 32-head, 128-dim
    "lightning" indexer (`sum_h w_h relu(q_h . k)`) and keep the top 512 entries per query; the layers in
    between reuse the last pick. The index keys (`k_norm(wk(latent))`, roped like the pool) belong to the kv
    source (layer 20's serve 24-36).
  - **Candidate blocks.** Layer 20 additionally keeps, per query, the 2048 best blocks of 8 consecutive
    positions (a block scores by its best row, the newest block is always kept); layers 24-36 pick their
    top 512 inside those blocks only. Below 16384 positions nothing is restricted.
  - Queries are not RMS-normed per head (V4 did); the output goes through `wo_a` (8 independent
    4096 -> 1024 blocks) and `wo_b`.
- **Engram** before layers 1 and 14: 2/3/4-grams of the *compressed token ids* (tokens that normalize alike
  collapse: NFKC, accents, case, whitespace) are hashed (`token * multiplier` XORed over the look-back, modulo a
  per-(order, head) prime, plus the bucket offset) into 24 rows of the layer's 384M-row table, projected
  (6144 -> 5120 x 5) into one key per mHC stream and a shared value; each stream gets
  `value * sigmoid(signed_sqrt(normalized_dot(stream, key)))`.
- DSpark block drafter (`mtp.*`, 3 stages, block 5): not implemented in ampstein yet.

## Implementation notes

- **FP8 weights**: `torch.fp8_gemv` (decode, M <= 8, one warp per output row, 16 weights per lane) and
  `torch.fp8_dequant` -> fp16 scratch + cuBLAS for prefill. Activations stay fp16 (the reference's fake
  quantization of activations and KV is not applied; exllamav3's V4 port does not either).
- **Mixed-K experts**: the vendored fused MoE kernels take one bit width per launch. The decode kernel
  (`moe_coop`) got `bits == 0` instances whose blocks switch on the expert's width (the unchanged static
  `gemv_tile<K>` per case, bit-identical to the static kernel), the token-sorted prefill kernel
  (`moe_fused`) reads per-expert widths in its run-time-switch instance. Both take `g_bits / u_bits /
  d_bits` tables (`PATCHES.md` 3, 4; `tests/moe_mixed.uc`).
- **Engram tables**: `stfile.gather` reads the rows with threaded `pread` (the vendored gather pool of the
  Qwen PLE tables) straight from the shard files into pinned memory; the page cache keeps the hot rows. A
  decode token needs 4 x 24 rows, a prefill chunk the unique rows of the chunk. The hash tables (token map,
  multipliers, primes) are derived from `tokenizer.json` with `tests/oracle/make_engram_meta.py`
  (once per model, writes `engram_meta.safetensors` into the model directory).
- **Chat format**: `chat_dsv41.jinja` is a transpiler-friendly rendering half of DeepSeek's
  `encoding/encoding.py` (the upstream `chat_template.jinja` needs `dict(**)`, slices and list `+=`);
  `chat_dsv41.prepare` does the normalization (tool results become blocks of the following user turn, in the
  order of the calls; top-level tools go onto the first system message). Tool calls are
  `<｜DSML｜ calls>` ... with a space after the bar (V4 had none).
- **Decode graphs**: positions live on the device (`st.meta`), the pool a graph reads is a bucket capacity
  (4096, 16384, ... positions), the ratio-2 compressor writes the pooled row of the pair every step (not visible
  until the pair is complete), the indexer masks what the position has not reached. One graph per device
  segment and bucket, a shared allocator pool per device.

## Validation

| Test | Against | Result |
|---|---|---|
| `fp8_ops`, `v41_attn` | composite fp32 references | 2e-4 (fp16 rounding) |
| `moe_mixed` | per-expert EXL3 gemms (decode and prefill kernels, K = 1..4 mixed) | 1.1e-2 / 4e-3 (the composite M = 1 gemv path is the noisier one: 1.3e-2 against the cuBLAS reconstruction, the fused kernel 6e-3) |
| `dsv41_attn` (layers 2, 3, 20, 21, 24: chunked prefill + decode through both paths, candidate blocks active) | DeepSeek's `Attention` on the CPU | ratio 2: 1e-3; ratio 1: 1e-2 (prefill) / 3e-2 (decode) — the reference itself moves by the same amount when its index scores are rounded to fp16 (the top-512 boundary), so that is the noise floor |
| `dsv41_hc` | `Block.hc_mixes` | 1e-7 |
| `dsv41_engram` | `engram.py` hash ids (token map, multipliers, primes) | exact |
| `dsv41_engram_apply` | `Engram.forward` with the real layer-1 projection | 5e-5 |
| `dsv41_graph` | eager decode, same prompt (full model, 6 GPUs, crossing the 4096 bucket) | identical logits |
| `dsv41_pipeline` | sequential chunked prefill | identical logits (the wavefront changes the issue order only) |
| `chat_dsv41` | `encoding.py` on 13 conversations (tools, parallel results out of order, reasoning drop, effort, reminders) | identical strings |

## Placement and memory (jai: 6 x RTX A5000 / A5500, 23.5 GiB each)

`plan_devices` splits the 40 layers over the GPUs with a DP that minimizes the fullest device (checkpoint
bytes of its layers, the embedding / head, the pools it owns and a mirror of every pool its layers read from
another device). Result for any context: layers per GPU 7 / 6 / 6 / 6 / 7 / 8.

| What | Per GPU after load | After cache + graphs |
|---|---|---|
| 110.6 GiB of weights over 6 GPUs | 3.5 - 6.2 GiB free | |
| pools for **1,048,576 positions** (fp16 pool + index keys, owners and mirrors) | | 1.8 - 4.7 GiB free |
| serving (`--chunk 2048`, warm-up at the end of the context) | | 0.45 GiB free on the tightest GPU |

So the full native context of the model (1,048,576 tokens) fits; the pools cost ~1.28 KB per position and
source layer (ratio 1: 1.28 GB per 1M positions, ratio 2: half). The drafter (7.4 GiB) would not fit next to it.

## Speed (one request)

| | |
|---|---|
| decode, graphs, 8k context | 38.6 ms/token (25.9 tok/s); 6 segments of 5.2 - 8.2 ms (the last one has the head) |
| prefill, wavefront, chunk 2048 | 1130 tok/s at 24k tokens, 1060 tok/s at 100k tokens |
| prefill, sequential (before the wavefront) | 239 - 540 tok/s |
| prefix reuse | the 100k-token prompt of the previous request: 0.6 s |
| load | 70 s with a warm page cache, 5 min cold (110 GB of weights) |

Where the time goes: prefill is MoE bound (a 512-token chunk costs ~20 ms per layer, 2048 tokens ~50 ms: the
fixed cost of streaming every expert once dominates, hence the large chunk); decode spends ~0.8 ms per layer, of
which the fused MoE kernels take ~0.24 ms (mixed-width instances, 6 experts), the FP8 GEMVs ~0.3 ms (at the
memory roofline since the conversion fix, L152) and ~100 small kernels the rest.
