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
- DSpark block drafter (`mtp.*`, 3 stages, block 5, 128 FP4 experts top 3): see the DSpark section below.

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

## DSpark block drafter (`dsv41_spark.uc`)

Per round one forward over a block `[seed, noise x 4]` through the 3 `mtp.*` stages (plain sliding-window blocks, mHC
one site late like the trunk, **128 FP4 experts top 3** + a shared FP8 expert), whose queries attend non-causally over
the last 128 *main kv rows* plus the block itself. The main rows come from the target: the stream mean at the input of
layers 37 / 38 / 39 (after the Engram), `main_norm(main_proj(cat))`, then per stage `rope(kv_norm(wkv(row)))` into a
1024-row ring. Drafts are the greedy chain of `head(norm(state)) + markov_bias(previous token)` (the target's head,
the markov embedding / head of rank 256); a confidence head gives each draft's acceptance estimate.

- **FP4 experts** (`torch.fp4_moe_gemv`, `fp4_dequant`): e2m1 nibbles (low first) + one e8m0 per 32 values, expert
  base addresses from device pointer tables, one warp per 4 output rows, x staged in shared memory with a pad float
  per 32 (L158). 0.20 / 0.25 ms for 15 slots (94 MB). `tests/fp4_ops.uc`.
- **Taps and rows.** `chunk_layers` records the taps of a chunk (prefill pipeline and graph decode alike);
  `model.graphs(cache, [1 .. 5])` also captures S-row steps (`new_dec_rows`, `attn_decode_batch`, L161) and
  `model.rollback(cache, L)` takes back the pool counts and the half-built compressor pairs of rejected rows.
  `tests/dsv41_verify.uc` (rows vs eager steps, rollback, 40 layers at 17k and 65k positions).
- **Round** (`dsv41_spark.run`): draft through 3 stage graphs + the head + a chain graph (~10 ms), pick the draft count
  V (L160), one S = V + 1 row graph, feed the drafter the verified rows (`update_rows`), engine accepts (greedy:
  exact match; sampled: p(x) of the one-hot draft) and rolls back. A draft needs its last 128 rows in the ring
  (`ready`); after a prefix reuse that lost them, one plain step per round refills them.
- **Placement.** `plan_devices(..., spark)` tries every assignment of the 3 stages to GPUs (2.6 GB each) and keeps
  the best split; drafter stages share GPUs with trunk layers.

| Verify step | 1 row | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ms (8k context) | 36.8 | 45.2 | 54.2 | 61.3 | 67.1 |

Measured with the server (256k context, chunk 1536): code, greedy 34 tok/s (2.6 tokens per round, 76 ms per round:
draft 10, verify 64, update 1.4); the same at temperature 1.0 27.5 tok/s (2.1 per round); prose/thinking 29 - 34
tok/s; plain decode 26 tok/s. 256k fits with chunk 2048 (the planner reserves 1 GB on the first GPU: layers 7/7/6/6/7/7, prefill ~900 tok/s, code decode 36 tok/s).

## Tensor parallel: 3 stages x 2 ranks (`--tp`, `dsv41_tp.uc`, `dsv41_spark_tp.uc`)

The layer split runs one GPU at a time. `--tp` makes the three pipeline stages pairs of GPUs (the layout of `dsv4_tp.uc`):
each stage's two ranks run every layer of the stage on their share and sum the partial sublayer outputs with the P2P
all-reduce; the 4 mHC streams stay replicated and bit-identical inside a pair.

- **Sharding.** Attention: query heads (`wq_b` rows, sinks), `o_groups` (`wo_a` groups) and `wo_b` by input columns (a
  partial sum); `wq_a`, `wkv`, the compressors and the indexer are replicated (one shared K = V head). MoE: expert
  parallel (192 + 192, the fused kernels filter the router's picks to the rank's range), the shared expert by rows /
  columns, router replicated. Engram replicated. Head by vocabulary slice over the last stage's ranks.
- **Lanes.** Rank r of every stage forms a lane with its own complete cache (rings, pools, mirrors); `dsv41.uc`'s
  `attn_forward / attn_decode / attn_decode_batch` run unchanged on a config copy with 32 heads / 4 groups.
- **FFN all-reduce.** The expert partials exceed the fp16 range on the massive-activation rows (the single-chain model
  sums them in fp32): they cross the fp16 all-reduce divided by 16 and `hc_apply` multiplies its post weights by 16.
- **Graphs.** One multi-GPU graph per (stage, step size, pool bucket) with a barrier copy around every replay (as
  `dsv4_tp`); the inter-stage hand-off, the cross-stage index / candidate copies and the pool mirrors are host-driven
  between the replays, as in the single-chain model. Defaults: pool buckets 16384 / max_seq, step sizes 1 / 3 / 5
  (`AMP_DSV41_TP_BUCKETS=full`, `AMP_DSV41_TP_SIZES=full` restore the complete sets: +0.4 GiB per GPU, ~5% faster).
- **Drafter.** Each drafter stage runs on the GPU pair of one pipeline stage (attention heads / groups, FP4 experts
  64 + 64, shared expert sharded; the FP4 kernel skips picks with id < 0). The planner chooses which pipeline stage hosts
  which drafter stage (`SPARK_DEVS`).
- **Tests.** `tests/dsv41_tp.uc` (2 stages x 2 ranks, 8 layers, against the single-chain model: prefill, wavefront,
  eager steps, graph step, 5-row step; 1e-3 .. 1e-2 like every comparison of this model).

Measured (jai, 256k context, chunk 1536, greedy code / temperature 1.0 / prose): **45 / 43 / 41 tok/s** with the
drafter (47 / 45 with the full graph sets and chunk 1024), against 40 / 34 / 37 for the single chain. A plain step is
24.5 ms (single chain 31.7), a 5-row verify step 44.5 ms (60.5). Prefill is host-bound under TP (each GPU busy ~30%:
the host issues both ranks' launches): 667 tok/s at 75k tokens (single chain 899), and it grows with the chunk, which
the memory (drafter + graphs + replicated pools) caps at 1536.

## jai alone: the 2.50bpw pack with host-RAM experts

The 2.50bpw pack (`DSV4.1-Flash-1017-exl3-2.50bpw-H5-HQ5bpw-SH`, 158.8 GiB of routed experts) does not fit jai's
6 x 23.5 GiB. A fraction of every stage's experts stays in pinned host memory (`--expert-resident`, chosen by a routing
profile), and the picks of those experts run on the CPU (`--expert-cpu`, the vendored exllamav3 CPU kernels,
`docs/EXPERT_STREAMING.md`). jai's links are PCIe gen3 x4 (~2.9 GB/s measured), which bounds prefill: a prefill chunk
group stages a layer's host experts over the link (`AMP_DSV41_TP_GROUP` chunks share one staging), except the CPU's part.

Launch (from an installed tree; `cmake --install build --prefix ~/ampstein-hx` first), with the DSpark drafter:

```sh
AMP_DSV41_TP_GROUP=3 AMP_EXPERT_CPU_SPLIT=0.25,0.35,0.4 ucode ~/ampstein-hx/bin/ampstein \
	--model /data/models/DSV4.1-Flash-1017-exl3-2.50bpw-H5-HQ5bpw-SH --tp \
	--ctx 262144 --chunk 1536 --cache-ram 16 \
	--expert-profile ~/ampstein-rpc/profile_dsv41.json \
	--expert-resident 0.58,0.65,0.655,0.655,0.62,0.625 --expert-cpu 12 --expert-swaps 0.5 \
	--host 0.0.0.0 --port 8061
```

From a build tree the same, as `ucode -L $PWD/build/lib -L $PWD/src/ucode ./ampstein ...` (absolute `-L` paths). Loading
takes ~7 min (disk-bound: one NVMe, ~0.7 GB/s); SIGTERM is ignored until the model is loaded. The GPUs must be free
before a restart (an old process still releasing its memory makes the new one fail with out of memory).

- `--expert-resident`: the fraction of the routed experts kept in VRAM, one per stage or (as above) one per GPU; the values
  above leave 0.6 - 0.9 GiB free per GPU after the warm-up (GPU 0 holds the embedding, GPU 1 has the most room). The rest
  (~60 GiB) is pinned host memory: the host needs ~75 GiB free.
- `--cache-ram 16`: the host-RAM session store (`cache_export` / `cache_import`: per lane the pools and index keys,
  the aux state, the Engram ids, the drafter's ring). A conversation that another one displaced comes back in ~0.1 s
  instead of a fresh prefill (24k tokens: 1.5 s for the next turn against 49 s); a 24k-token entry takes 0.1 GiB.
- `--expert-profile FILE`: per-layer expert pick counts (`examples/trace_routing_dsv41.uc` over real sessions, then
  `tests/oracle/routing_profile.py`); it chooses the resident set and orders the host experts by popularity.
- `--expert-cpu N`: threads of the CPU expert pool (12 = jai's physical cores; 16 and 24 are slower: one pool, spinning
  surplus workers). Decode picks of host experts always run there.
- `--expert-swaps N`: exchanges between VRAM and host memory per decode step (fractional: 0.5), following the requests'
  routing, plus a settle after each prefill of 512 tokens or more; the CPU's set follows (window miss rate ~2.3 - 3.5%
  against ~5% static). One exchange is one native call (`torch.expert_exchange`, ~0.5 ms of host time); the residency it
  learns persists, so a low rate is enough.
- `AMP_EXPERT_CPU_SPLIT=q0,q1,q2`: per stage, the least popular fraction q of each layer's host experts is not staged for
  prefill: the CPU computes their picks while the link stages the rest (balance: CPU time ~ link time per stage; stage 0's
  early layers route less predictably, hence less). Also shrinks the staging buffers (1.96 -> 1.31 GiB per GPU). Only
  for prompts that fit one group (tunable `split_max`, default group x chunk): with more groups the stages overlap and
  their CPU shares queue on the same cores, so the kernels read those experts zero-copy from the arena instead (a few
  rows each: about one pass over the link, like staging).
- `AMP_DSV41_TP_GROUP=3`: chunks per prefill group (each group stages the host experts once); 3 x 1536 rows is what
  the VRAM holds next to the resident experts (4 runs out of memory, as does chunk 2048 with 2).
- `AMP_EXPERT_CPU_PF_MAX=N` (default 512): prompts of up to N new tokens prefill with every host expert on the CPU (no
  staging; for the short increments of an agent turn). `AMP_EXPERT_STAGE_BUFS=1`: one staging buffer instead of two.
- Prefill copies every stage's first two layers ahead (its GPUs are idle until the first chunk arrives): ~1 - 2 s less
  per fresh prompt.
- Live tunables (`src/ucode/ampstein/tune.uc`): `AMP_TUNE=FILE` names a JSON object re-read when it changes, e.g.
  `{"swaps": 1, "cpu_threads": 10}`. Keys: `cpu_threads`, `cpu_prof` (CPU call statistics every N calls), `cpu_phases`,
  `cpu_pf_max`, `stage_min`, `group`, `prestage`, `settle_min`, `settle_rounds`, `swaps`, `interval`, `margin`, `decay`,
  `pf_steps`, `pf_weight`, `dspark_prof` (draft / verify / update ms per round), `dspark_adapt`, `dspark_costs`.
- Diagnostics at start: `AMP_EXPERT_CPU_PROF=N`, `EXL3_MOE_CPU_PROF=1`, `AMP_EXPERT_DYN_LOG=N`, `AMP_ALLOC_LOG=1`,
  `AMP_DSV41_TP_HOSTPROF=1`, `AMP_NSYS=prefill|decode` (with `nsys --capture-range=cudaProfilerApi`; add
  `--cuda-graph-trace=node` for decode).

Measured (fresh prompts, nothing cached; earlier "~4 s for 1.9k tokens" figures came from a bug — a prefill group id
that matched a layer's stale staging — and are void): 700 tokens 8.6 s, 1650 tokens 10.7 s, 2350 tokens 12 s, 5400
tokens ~21 s, 22.7k tokens 42.7 s (530 tok/s), 55k tokens 89.7 s (613 tok/s). A fresh prompt pays ~8 s of staging over
the x4 links before its length matters (the three stages one after another); long prompts overlap the stages and
approach the host-bound TP prefill of the all-resident model (~670 - 900 tok/s). A prompt increment of 50 - 250 tokens
(an agent turn): 1 - 4 s. Decode with the drafter: ~75 - 80 ms per round (draft 7, verify 65, update 5) of 2.7 - 3.1
tokens, 35 - 42 tok/s warm (26 - 33 right after a start), against ~41 - 45 tok/s with every expert resident. Greedy
reference outputs equal the GPU-only ones.

Tried and dropped: two CPU pools of 6 threads (one per rank; slower: the ranks' calls rarely overlap and a call halves
its threads), 16 / 24 threads, drafter costs measured online (`dspark_adapt`: shorter drafts, same round time), chunk
2048, a cache of decoded experts for the CPU (65% of decode-time CPU experts were decoded among the last 1024, 84%
among 4096; exact copies are 2 bytes per weight and read as slowly as the trellis decodes: only a lossy int8 copy would
pay, ~7% of decode for ~25 GB).

Quality of the CPU experts (`examples/dsv41_tp_nll.uc`, the same resident set computed by the GPU kernels reading the
host experts zero-copy against the CPU kernels; 14k tokens of context, 2048 scored positions in 16-row steps): NLL
1.8784 / 1.8788, top-1 1134 / 1135 of 2048, 71 argmax flips (near-ties). Greedy continuations of agent sessions fork
at such a tie within a few hundred tokens and go on as equally coherent text (tool calls well-formed on both sides):
greedy equality is not the measure here, the likelihood is.

## KV cache quantization (Q8 pools, default)

The compressed pools (the pooled rows and the indexer keys of the kv source layers, replicated per lane under TP) are
stored as int8 codes plus one fp16 scale per 32 elements (`kv_q8_put`), 664 instead of 1280 bytes per entry.
`AMP_DSV41_KV=q8pool` quantizes the rows only, `fp16` restores the old layout. The attention kernels (`v41_attn`, CUDA-core
and tensor-core) and the indexer scoring kernel (`idx_score`) dequantize while staging the rows into shared memory
(`amp_q8x8`): in-kernel and dequantized-then-fp16 results are bit-identical (`tests/v41_attn.uc`, `tests/idx_score.uc`); the
window ring stays fp16. Quality (`examples/dsv41_nll.uc`, teacher-forced NLL of real text, 40k context, last 512 positions):
fp16 0.5214, q8pool 0.5201, q8 0.5184 (top-1 436 / 437 / 438 of 512): no measurable loss. A single layer's attention output
moves by ~1e-2 against the fp16 CPU oracle (`dsv41_attn` tolerances are x6 under Q8). Decode and prefill speed are unchanged;
the memory saved is 0.1 - 0.3 GiB per GPU at 256k (3.7 GiB less pool in total than fp16 at 1M single-chain: the planner counts
the smaller entries).

