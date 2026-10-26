# Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`)

Checkpoint: `/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw` (~100 GB EXL3 +
39 GB n-gram table). Reference: exllamav3 `architecture/qwen4_exp.py`
(installed wheel). Oracle: `tests/oracle/make_qwen4_oracle.py`.

## Architecture (from config.json / exllamav3)

- 48 layers, hidden 2560, every 4th layer full attention (24 q heads, 2 kv heads,
  head_dim 256, interleaved q gate, q/k RMSNorm +1, partial rotary 0.25, mrope),
  the rest GDN (16 k heads, 48 v heads, dim 128, conv 4). Block output gate of the
  GDN norm is **sigmoid** (`output_gate_type`), same math as the KDA out norm.
- MoE on every layer: 512 experts, top 10, intermediate 640, softmax router
  ("std": softmax over all logits, top-k), shared expert (640) times
  `sigmoid(shared_expert_gate(x))`. Expert tensors are per-expert
  (`mlp.experts.N.{gate,up,down}_proj`), like GLM.
- **Gated-residual hyper-connections** (`GatedResidual`), 4 fp32 streams, replace
  the input/post layernorms: per-stream RMSNorm (weight applied as 1+w) ->
  low-rank sigmoid gate (`input_mix_weight_down/up`, rank 320) selects what each
  stream contributes to the **mean** that feeds the sublayer; the sublayer output
  is injected as `x[h] += post[h] * y`, `post = 2*sigmoid(block_inject(flat)/4)`.
  No Sinkhorn / comb matrix. Reference: `GatedResidual._mix_ref`.
  Final `hyper_connection_mixer` = same mix without inject (`use_combine=False`),
  **no final norm**, then lm_head.
- Streams start as the embedding broadcast to 4 fp32 streams (`ExpandStreams`).
- **PLE layer** before block idx 1 (`ple_layer_ids = [2]`, 1-based): hashed n-gram
  embedding (ngram 3, 8 heads per n-gram, 16 rows of 160 -> 2560) -> key_proj (one
  key per stream), value_proj (shared value); signed-sqrt dot gate against the
  normed streams; dilated (3) depthwise causal conv (k=4) over the normed gated
  values; added to all streams. Recurrent state: conv tail
  ((k-1)*dilation columns) + the last ngram-1 token ids.
- **n-gram table** (`ngram_embedding.safetensors`, 39 GB, exl3_ngram_trellis
  rows of 160 values): never resident on a GPU. Per forward the token ids are
  hashed on the host, the *unique* rows (a few hundred per chunk) are gathered
  from the mmap'd file (page cache = RAM, else SSD), uploaded, dequantized with
  `ngram_dequant`. Kernels are already in `vendor/exl3` (`ngram.cu`, `ple.cu`).
- **QSA** indexer on full-attention layers: block (4 tokens) mean-pooled keys,
  top 512 blocks + tail block; dense attention is *exact* up to
  `4*512+3 = 2051` tokens, so the sparse path is only needed beyond that.
- MTP head (`mtp.*`, 1 full-attention block, own embeddings off) and the Qwen3.5
  vision tower (no deepstack) follow after the text path.

## Implementation (ucode/qwen4.uc)

- Layer split over all GPUs (`qwen4.load(dir, { devices: N })`, layers spread evenly);
  one function set serves the oracle test (`forward_chunk`), the engine
  (`prefill_pipeline` wavefront runner, `graphs`, `aux_get/aux_set/aux_reset`) and
  the benchmark (`examples/bench_qwen4.uc`).
- Gated-residual mix: composite ATen ops for prefill rows (the cuBLAS-path
  reference in exllamav3), the vendored fused `gr_mix` kernels for <= 8 rows
  (`torch.gr_mix`; folded `cat(down, inject) * w` and repacked `up` tables built at
  load). The streams are (rows, 4, D) fp32 and `torch.hc_apply` (comb = null)
  injects the sublayer output.
- PLE: `ple_prepare` (host: hash + gather the unique table rows from the mmap'd
  `ngram_embedding.safetensors`, `ngram_dequant` on the GPU) and `ple_apply`
  (device: `torch.ple_forward`, conv tail updated in place) so the second half is
  capturable. The table is never loaded: a chunk touches a few hundred rows.
- MoE: glm5.uc's `moe_begin/moe_finish` (fused decode kernel for <= 8 rows,
  token-sorted fused kernel for prefill) with `router: 'softmax'` (softmax over 512
  logits, top 10, renormalised) and the sigmoid-gated shared expert.
- QSA: per full-attention layer a raw indexer key cache (max_seq, 128) and a pooled
  block cache (max_seq / 4, 128); the KV cache is (max_seq, kvh, hd) so selected
  tokens are a row gather. Prefill uses 64-query slabs (host-computed block counts),
  dense FlashAttention below 2051 tokens. The decode graphs use fixed shapes: the
  selection scans all pooled capacity masked per query, gathers 2052 tokens, and
  every position-dependent index comes from one packed int64 buffer rewritten
  before each replay (`dev_state` / `dev_values`).
- Engine: `model.aux_*` snapshot / restore / reset the PLE state next to the GDN
  states (the conv tail is a persistent tensor, so a snapshot copies it). The RAM
  store and hot reload are off for this model (store layout assumes the (1, kvh, n,
  hd) cache of the dense Qwen).
- MTP (`build_mtp`, `mtp_step`/`mtp_core`, model hooks `mtp_chunk`, `spec_bufs`,
  `aux_ids`): the draft input is the flattened pre-mixer stream stack (4 x 2560 fp16,
  `model.hidden_width`), per-stream tap as in exllamav3 (27.7% vs 42.8% teacher-forced
  with averaged streams, `AMP_QWEN4_MTP_TAP=0`). The verify graph (S = window) keeps the GDN
  states and the PLE conv tail after each row for the rollback; the PLE id context is
  resynced from the accepted history after every round. The async device-id round is
  off (the PLE hash needs host ids), the synchronous round is used.
- Vision: `qwen35_vision` loads this checkpoint unchanged; `cache.mm` feeds the
  embedding overlay and `rope_rows` (MRoPE) for the main attention of every attention
  layer, the indexer uses plain positions; decode graphs carry the rope positions
  (`rpos`) in the packed index buffer.

## Measured (six-GPU host, 6 x RTX A5000/A5500, 6.05 bpw, layer split)

- vs exllamav3 (oracle fixtures): blocks 0-3 rel-rms 2-3e-3; whole model logits
  4-6e-2 (noise amplification, like GLM), argmax 24/24, greedy continuation 12/12;
  4523-token prompt through the sparse path: argmax 62/64, continuation 8/8.
- Decode: eager 44 ms/token (fused mix), graphs 22-25 ms/token (40-45 tok/s), also
  at 4.5k context (23.7 ms; the eager attention was 56 ms there). Profile of a
  graph step: 8-bit projections (exl3 gemm, bandwidth bound) 8 ms, MoE kernels
  5 ms, gated-residual mixes 3.6 ms, router/top-k/elementwise the rest.
- Prefill: 1590 tok/s on a 4.5k prompt (wavefront, chunk 1024); 2.0k tok/s at 29k
  depth; the sparse attention is the fused `torch.qsa_attn` kernel (the composite ATen
  version gave 1.2k at the same depth). Model load 53 s, 57 s to serve.
- Long context (server, ctx 131072, needle-in-a-haystack, 17-20 GiB per GPU): 37k
  tokens and 102k tokens (needle at 15% and at 50%) answered exactly; 102k prefill
  968 tok/s on the first (cold) request, 2.7k tok/s for the 87k new tokens of the
  second; decode 37-43 tok/s at 102k depth.

- MTP head accuracy on the model's own greedy output (depth 1): prose 70%, explanation
  88%, code 98%, list 84%. Server decode, 200 tokens greedy: plain 37-42 tok/s; MTP
  window 2: 46-59; window 3: 51-75 (default); window 4: no better. Greedy output
  differs from plain decode at floating-point near-ties (also between two plain
  kernel configurations), the code prompt is byte-identical.
- Vision: the tower's embeddings vs exllamav3 cosine 0.9998 (rel-rms 1.9e-2); with
  exllamav3's own embeddings injected (AMP_VIS_EMB debug override in the server) the
  description matches in content; short questions ("red", "ucode", "4") are identical.

## Tensor parallel (`--tp`, ucode/qwen4_tp.uc)

Every layer on all ranks: GDN by k-head pairs, MoE expert-parallel (experts split over
the ranks, shared expert and attention on one owner rank per layer, rotating), the
streams and the gated-residual tables replicated, PLE on rank 0 (broadcast), the head
sharded by vocabulary, P2P all-reduces, one multi-GPU CUDA graph per step size and a
draft-step graph for MTP. Streamed straight from the checkpoint (60-77 s). Measured on
the 6 GPUs: plain decode 20.8 ms/token vs 23-25 for the layer split; with MTP window 3
52 / 79 / 93 / 66 tok/s (layer split 51 / 65 / 75 / 53); at 102k depth 57-84 tok/s;
prefill 457-657 tok/s (host-bound, `--chunk 4096` default) vs 1.6-2.7k. See L99 for the
analysis of why the decode gain is modest.

`--tp --stages 3` (3 groups of 2 ranks, layers 0-15 / 16-31 / 32-47): the groups hand the
replicated streams on, decode is one multi-GPU graph through all of them, and prefill
is a wavefront (one chunk per group in flight; the host launches for 2 ranks per layer).
Steady-state prefill 2.0-2.1k tok/s at 32-37k tokens (stages 1: 0.7k, stages 2: 1.3k),
plain decode 23.2 ms/token (stages 2: 21.6, stages 1: 20.8), MTP 52 / 66 / 77 / 56 tok/s,
17.7-19.3 GiB per GPU at ctx 131072. This is the recommended TP setting; plain
`--tp` (one group) decodes ~10% faster but prefills at a third of the speed.

## Measured after the decode work (TP x 3 stages, ctx 262144, 6 GPUs)

Plain decode 21.97 -> 17.3 ms/token at 4k depth (identical outputs), verify S=3 28.8 -> 24.3 ms; server, greedy
code prompt 62.8 -> 77 tok/s (MTP window 3), T=1.0 ~63; needle at 64k depth correct, prefill 1.8-2.1k tok/s. See
LESSONS L167 for what changed (shared expert fork in the graphs, DSA selection kernels, device-side draft chain).

## Open
- (done) Q8 KV: `--cache-mode Q8` (see L168); the 2.05 bpw quant on 48 GB then uses 22.9 of 24 GB per GPU at ctx 131072.
- Attention head-parallel split, fewer gated-residual kernels, stage-staggered decode of several sequences.

## Other checkpoints

- 4.05 bpw: the MTP block's mixer is in the main checkpoint (no `mtp_hyper_connection_mixer_patch` file).
- 2.05 bpw (turboderp `2.05bpw_h4_ng4`, 36.4 GB of weights): the n-gram table is 128 consecutive shards of
  (2,500,012, 41) int16 (K = 4, 26 GB); `build_ple` addresses them as one table (`stfile.gather_async` with
  `total_rows`). Fits two 24 GB GPUs at ctx 131072 (`--gpus 2 --tp --stages 1`): decode 15.0 ms/token, MTP
  ~94-112 tok/s, needle at 38k correct, prefill ~1.2k tok/s.

- Balance the TP attention layers (the owner rank runs the QSA chain alone, ~285 us
  of arrival spread per attention layer) and the expert-parallel MoE.
- RAM store / hot reload for the PLE state, fused router kernel (`routing.cu` is
  vendored but not built), the async device-id MTP round (needs a device-side PLE hash).

## Experts in host memory (`--expert-resident`)

`--tp --expert-resident F --expert-profile FILE` keeps the fraction F of the experts in VRAM (ranked by the routing
profile) and the rest in pinned host RAM, for quants that do not fit in VRAM (4.05 bpw on 2x 24 GB: F = 0.45 at ctx
131072). Prefill stages the cold experts layer by layer behind the compute; decode reads them over PCIe from inside
the kernels. Without the flag nothing changes. See `docs/EXPERT_STREAMING.md` (Qwen3.8-Flash-Next on j7) for numbers,
`tests/qwen4_host_experts.uc` for the test and L171.

`--expert-swaps N` (with `--expert-resident`) lets the resident set follow the requests: up to N experts per decode step
are exchanged with host memory (N = 4: +13-21% decode on 6.05 bpw). 6.05 bpw: `--expert-resident 0.24 --chunk 8192`.


## GGUF (Qwen4exp, IQ3_XXS / IQ2_XS) on 2x RTX 3090

The GGUF loader uses the same `qwen4` layer split (no TP). `qwen4_tp` is not adapted to GGUF. Key details:

- GGUF linear-attention value heads are stored **group-major** (`i = g*16 + h`); the HF/EXL3 layout is
  **head-major** (`j = h*3 + g`). `build_sublayer_at` reorders the value part of `in_proj_qkv`, all rows of
  `in_proj_z`, the rows of `in_proj_b` / `in_proj_a`, the value channels of `conv1d`, and the input activation
  of `out_proj` (the raw `out_proj` columns stay group-major). `A_log` / `ssm_dt.bias` are reordered as
  `[3,16] -> [16,3]`; GGUF `ssm_a` is already `-exp(A_log)`, so it is reordered only.
- The n-gram table (`per_layer_token_embd.weight`, IQ4_NL, ~28.8 GB) stays on disk; `ple_start` gathers the
  unique rows and dequantizes them per chunk.
- KV cache: `AMP_QWEN4_KV=q8` stores int8 codes plus fp16 scales, cutting the 256k full-attention KV from
  ~3.1 GB to ~1.7 GB.
- Expert placement: `AMP_QWEN4_EXPERT_RESIDENT=F` keeps `int(512*F)` routed experts in VRAM and the rest in
  mapped pinned host memory (`torch.host_mapped`); decode reads cold rows over PCIe without staging copies.
  `AMP_QWEN4_EXPERT_PROFILE=FILE` is a JSON object `{layer: [expert ids sorted by routing frequency]}`; the
  resident set is the top `int(512*F)` ids, padded with unused ids if the profile is shorter.
- Pointer tables: `build_moe` builds `pt1/pt2/pt3`, one int64 device pointer per global expert to the start of
  its row in either the resident stack or the mapped cold stack. `torch.gguf_moe_gemv_pt` then runs one grouped
  gemv per matrix for all decode slots, mixing resident and cold experts without a host-side split. The old i8
  types (the Qwen4 GGUF experts are `IQ2_XS`) use the pointer-table i8 kernel; newer types use the fp16 kernel.
- Routing bias is real: replaying three Pi coding sessions gave per-layer top-358 coverage 94.1-99.9%, versus
  62.6-80.1% for a uniform prefix. Across all 512 experts, the per-layer max/min count ratio was ~5.4x; all
  experts were used, but not equally.

Measured on j7 (2x RTX 3090, both PCIe x8), IQ3_XXS, Q8 KV, F = 0.70, chunk 4096, 256k context:

| Workload | Time | Rate |
|---|---:|---:|
| PP 262k tokens | ~241-246 s | ~1.06-1.09k tok/s |
| TG 20 tokens at ~262k, generic profile + pointer tables | ~0.835 s | ~24 tok/s |
| TG 20 tokens at ~262k, Pi-session profile + pointer tables | ~0.616 s | ~32.5 tok/s |

With a routing-frequency profile the cold-hit fraction is ~0.4% on the generic profile and ~0.1-5.8% per
layer on the Pi-session profile, versus ~30% for a uniform prefix. The pointer-table decode path removes the
host `.tolist()` split and the per-token staging copies; it is now one grouped gemv per matrix for all slots.
The remaining decode cost is launch overhead and the layer-split duty cycle: CUDA graphs / fused decode are
the next step.

### MTP from a separate GGUF

The `ISTA-DASLab` GGUF shards have no MTP tensors; `ggml-org/Qwen3.8-Flash-Next-GGUF` publishes it as a separate
file (`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`, 2.2 GB; BF16 / Q8_0 variants exist). Symlink that file into the model
directory next to the backbone shards. `ggufw` maps the file's `blk.48.*` tensors to the engine's
`mtp.layers.0.*` keys and its `blk.48.nextn.*` tensors to `mtp.*`:

| GGUF tensor | engine key |
|---|---|
| `blk.48.nextn.eh_proj.weight` | `mtp.eh_proj.weight` |
| `blk.48.nextn.enorm.weight` / `hnorm.weight` | `mtp.pre_fc_norm_embedding.weight` / `mtp.pre_fc_norm_hidden.weight` |
| `blk.48.nextn.hc_head_{down,up,norm}.weight` | `mtp.hyper_connection_mixer.*` |
| `blk.48.{attn,ffn,hc_*}.weight` | `mtp.layers.0.*` |

GGUF fuses the two EXL3 draft projections (`fc_hidden`, `fc_embedding`) into one `eh_proj` that takes
`[rms(embed); rms(stream)]` per hyper-connection stream and returns one hidden vector per stream. `build_mtp`
uses `ggufq.linear` for `eh_proj`, and `mtp_input` concatenates the broadcast embedding with the normalized
stream stack before the projection. The GGUF `enorm` / `hnorm` already carry the `+1` RMS bias, unlike the
EXL3 patch. The MTP layer's own MoE experts use the same resident/host split as the backbone.

Q4_0 was added to the CUDA dequant/gemv kernels for this file (block 32, 18 bytes, low nibbles first 16 values,
high nibbles next 16). With `--mtp 3` on IQ3_XXS, 8k context, F = 0.75, Q8 KV:

| Request | Plain decode | MTP decode |
|---|---:|---:|
| 29 tokens | 25.0 tok/s | 29.1 tok/s (11 rounds, 73% accepted, 2.64 tok/round) |
| 120 tokens | — | 23.9 tok/s (55 rounds, 50% accepted, 2.18 tok/round) |

The draft head is opt-in (`--mtp`). Its own expert stack can use a different resident fraction:
`AMP_QWEN4_MTP_EXPERT_RESIDENT` (default: 1.0, fully resident; independent of the backbone). At 256k the MTP KV/cache and draft block
still cost ~0.4–0.5 GB of VRAM. With `AMP_QWEN4_EXPERT_RESIDENT=0.66`, `AMP_QWEN4_MTP_EXPERT_RESIDENT=0.0`,
Q8 KV, chunk 4096, a 220k-token prompt gave PP 1196 tok/s and 20-token decode 19.9 tok/s (9 rounds, 56%
accepted, 2.22 tok/round). Plain decode at the same context is faster (~24 tok/s at F = 0.70), so MTP is a
win for short/medium contexts, not for 256k decode. At F = 0.70 with MTP the warm-up OOMs.
