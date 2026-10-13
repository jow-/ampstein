# Expert streaming: serving MoE models larger than VRAM

Status: investigation, nothing shipped. Measurements on jai (6x RTX A5000, PCIe
gen3 x4 per GPU), 2026-10-13. The aim is a larger quant of GLM (or DeepSeek-V4)
than fits in VRAM, with the cold experts held in host memory and moved on demand.
The target to try next is j7 (2x RTX 3090, PCIe 4.0 x8).

## Summary

- The kernel side is solved: the fused MoE kernels read expert weights through
  int64 pointer tables, and a pointer may point at pinned host memory (UVA). The
  kernel then runs at link speed with no sync, no graph break, bit-exact.
- The whole problem is the host link. One GLM expert is 9.4 MiB; jai's x4 link
  moves 3 GB/s per GPU, so a miss costs ~3.1 ms against a ~0.2 ms layer.
- Routing locality is moderate. There is no global hot set; a per-layer LRU
  reaches ~0.63 hits at 25% of the experts resident and ~0.82 at 50%.
- The router itself is a usable predictor: applying layer L's router to the hidden
  state of layer L-1 recovers 67% of the real experts with 8 guesses (83% with 16).
  History-based heatmaps (prefill routing, early decode, other prompts) were worse
  than plain LRU.
- Whether prefetch pays depends on the link, not the policy. At 3 GB/s it makes
  things slower (the link is saturated; prefetch adds bytes). At >= 12 GB/s it
  removes essentially all of the streaming penalty at 50% resident.
- Dense models do not stream: every weight is read every token, there is no cold
  part, and compute per layer is far shorter than a layer's transfer.

## Link speed (jai)

H2D copies from pinned memory (`examples/bench_h2d.uc`, `copy_` over
`torch.pinned_empty`):

| Case | Result |
|---|---|
| one GPU, 9 MB .. 512 MB copies | 2.97 GB/s |
| all six GPUs at once | 2.98 GB/s each (links are independent: aggregate 18 GB/s) |
| `nvidia-smi` link state | gen3 x4 under load (cards are x16 capable; idle reports gen1) |

Expert parallelism therefore scales streaming bandwidth with the GPU count: each
rank fetches only its own shard over its own link.

## Zero-copy kernel (`examples/bench_zerocopy.uc`)

64 experts of GLM layer 3; `torch.moe_coop` with the three trellis pointer tables
pointing at VRAM or at `cudaHostAlloc` memory (`suh`/`svh`, a few KB per matrix,
stay on the device). Distinct experts per call so L2 does not help.

| Rows (T) | VRAM | all 8 experts per token on host | half/half |
|---|---|---|---|
| 1 | 0.21 ms | 25.0 ms | 12.5 ms |
| 3 | 0.57 ms | 74.9 ms | 37.5 ms |
| 8 | 2.4 ms | 199.7 ms | 99.9 ms |

25 ms for 8 x 9.4 MB is the link (3 GB/s), not a kernel latency problem: the
kernel streams at line rate. A mixed table whose picks are all VRAM-resident costs
the same as an all-VRAM table (0.214 ms), so a hit is free and a miss costs
exactly its transfer time. Expect 10 GB/s-class numbers on a gen4 x8 link; run
the benchmark first on j7 (`ucode ... examples/bench_zerocopy.uc 0 64`).

## Routing locality

### Method

- `examples/trace_routing.uc` logs the top-8 expert ids per MoE layer per token
  (a `set_trace` hook in `glm5.moe_route`, eager path, one sync per layer).
- Run 1: six synthetic prompts, model's own greedy output, 255 decode tokens each.
- Run 2: eight real agent sessions from `~/.pi` (open-source projects only;
  `tests/oracle/pi_samples.py` picks them): 430..6.5k tokens of history prefilled,
  then 178..528 tokens of the recorded assistant reply teacher-forced as decode.
  2,976 decode tokens x 42 layers. Run 2 also records, per decode row, the top-32
  guess of the router of layer L applied to the router input of layers L-1, L-2 and
  L-4 (`glm5.route_peek`).
- Analysis: `tests/oracle/routing_analysis.py route.txt`.

### Findings

GLM-5.3-Flash 3.05bpw: 42 MoE layers, 288 routed experts, top-8, 9.4 MiB each.
Hit rates are per-layer caches of C experts over the decode tokens.

| Cache (of 288) | LRU (run 1) | LRU (run 2) | static set, oracle (run 1) | profile of other prompts (run 1) |
|---|---|---|---|---|
| 12.5% | 0.53 | | 0.54 | 0.27 |
| 25% | 0.68 | 0.63 | 0.74 | 0.44 |
| 50% | 0.84 | 0.82 | 0.93 | 0.71 |
| 66% | 0.88 | | 0.98 | 0.83 |

- LFU (decayed counts) is within 0.01 of LRU. A fancy heatmap policy buys little
  at these window lengths. The windows are short (<= 528 tokens), so both are
  somewhat pessimistic: caches start cold.
- Per layer, the top 25% of experts cover 40-60% of the accesses; 1-12 experts per
  layer are never used in a few hundred tokens.
- Consecutive tokens share 28% of their experts (2.2 of 8). A 3-token window
  (MTP verify) touches 18.9 distinct experts of 24 slots.
- A prompt's prefill touches ~157 of 288 experts per layer.
- Static set from the prompt's own prefill routing, tested on its decode: 0.74 at
  50% (LRU 0.82). From the first 32 decode tokens: 0.60. From other prompts'
  prefills: 0.68. None beat LRU.

### Router peek as predictor (run 2, recall of the 8 real experts)

| Lookahead | top-8 | top-12 | top-16 | top-24 | top-32 |
|---|---|---|---|---|---|
| d = 1 layer | 0.67 | 0.77 | 0.83 | 0.88 | 0.92 |
| d = 2 | 0.57 | 0.66 | 0.72 | 0.79 | 0.83 |
| d = 4 | 0.45 | 0.53 | 0.58 | 0.66 | 0.71 |
| chance | 0.03 | 0.04 | 0.06 | 0.08 | 0.11 |

The peek is cheap (one fp32 gemv + sigmoid + topk per layer; the router matrix is
288 x 4096).

### LRU plus peek prefetch (per token, all 42 layers; 336 fetches if all miss)

| Config | Stalling misses | Prefetch transfers | Hit rate at access |
|---|---|---|---|
| 25% resident, LRU | 124 | 0 | 0.63 |
| + d=1, top-8 | 54 | 120 | 0.84 |
| + d=1, top-16 | 35 | 266 | 0.90 |
| 50% resident, LRU | 61 | 0 | 0.82 |
| + d=1, top-8 | 28 | 57 | 0.92 |
| + d=1, top-16 | 18 | 122 | 0.95 |
| 50%, d=4, top-8 | 46 | 50 | 0.86 |

Prefetch converts stalls into asynchronous traffic but adds bytes (2.2 spurious
transfers per avoided miss at d=1 top-8 / 25%). Prefetched experts also compete for
the same slots.

## Step-time model

Not measured end to end. `T0` = resident decode step, `t_x` = 9.4 MiB / link
bandwidth, `G` = GPUs sharing the transfers (expert parallel).

- demand only: `T0 + misses * t_x / G` (a miss stalls the kernel)
- prefetch (optimistic, perfect overlap): `max(T0, (misses + prefetches) * t_x / G)`

With T0 = 30 ms (jai's 3-stage step), 50% of the experts resident:

| Link | G=4 demand | G=4 prefetch d1/top-8 | G=2 demand | G=2 prefetch |
|---|---|---|---|---|
| 3 GB/s (jai x4) | 78 ms | 67 ms | 126 ms | 133 ms |
| 12 GB/s | 42 ms | 30 ms | 54 ms | 33 ms |
| 25 GB/s (x16 gen4) | 36 ms | 30 ms | 42 ms | 30 ms |

With 25% resident and G=4 the 12 GB/s case is 54 ms demand / 34 ms prefetch. Two
consequences:

- On a slow link keep the resident fraction as high as possible and do not
  prefetch; shed other VRAM first (KV precision, context).
- On a fast link the router-peek prefetch is what closes the gap.

MTP interacts badly: a 3-token verify round touches ~19 distinct experts per layer
instead of 8, i.e. 2.4x one token's traffic for ~2 emitted tokens (1.2x per token).
Under streaming MTP wins only while the step is compute-bound; measure with
`--mtp 0` against `--mtp 3`.

## Models

| | GLM-5.3-Flash 3.05bpw | DeepSeek-V4-Flash(-Vision) 3.04bpw |
|---|---|---|
| Checkpoint | 113.9 GiB | 110.2 GiB |
| Routed experts | 42 layers x 288, top-8 | 43 layers x 256, top-6 (3 hash-routed layers) |
| One expert | 9.4 MiB | 9.4 MiB (0.753 GiB per matrix family per layer / 256 x 3) |
| Expert bytes | 106.7 GiB (94%) | ~97 GiB (88%) |
| Non-expert (attention, shared, embed, head) | ~7.2 GiB | ~13 GiB (incl. vision) |
| All-miss traffic per token | 336 fetches = 3.2 GB | ~258 fetches = 2.4 GB |
| Routing trace | measured | not measured |

DeepSeek has fewer active experts per token (6 of 256 = 2.3%, vs 8 of 288 = 2.8%),
so the all-miss cost is ~23% lower, and its three hash-routed layers choose their
experts from the token id alone (`tid2eid`): those are known before the forward
pass and need no prediction. The router there is `sqrtsp`; `route_peek` is
sigmoid-only today, extend it and put a `TRACE` call in the DSV4 router path before
trusting any DeepSeek number.

## Plan for j7

Hardware: 2x RTX 3090 (24 GiB each, 48 GiB total), PCIe 4.0 x8 per GPU
(~13-14 GB/s practical expected), 187 GB host RAM (check how much is free: this
machine already runs the Qwen server and its buffers).

Budget for GLM, TP over both GPUs: 48 GiB - ~8 GiB non-expert - ~4 GiB CUDA
context/graphs/activations - ~3 GiB KV (32k) leaves ~33 GiB of experts: ~31% of
106.7 GiB resident, ~74 GiB pinned in host RAM. DeepSeek: ~28 GiB resident (29%),
~69 GiB pinned. Expected (model, G=2, T0 15-30 ms, 31% resident, hit ~0.68): about
55-70 ms/token, i.e. 14-18 tok/s without MTP; with d1/top-8 prefetch it becomes
link-bound at roughly the same figure. Treat these as the upper end; run the
numbers below before committing.

Checklist, in order:

1. `examples/bench_zerocopy.uc 0 64` and the H2D copy loop: real GB/s under load
   (x8 gen4), and whether 2 GPUs transfer concurrently without contention.
2. Pin ~75 GiB: `cudaHostAlloc` of that size, and how long it takes. Do not pin
   from `mmap`'d checkpoint pages (L40: sequential `read()` into pinned buffers).
   The 117 GiB page cache plus a pinned copy doubles the footprint.
3. Trace routing for the models on a x8 host with the real client mix
   (`pi_samples.py` -> `trace_routing.uc` -> `routing_analysis.py`); for DeepSeek,
   add the hook first.
4. Prototype residency manager (below), measure decode with prefetch off, then on.

## Design sketch (not built)

- Per MoE layer: a pool of S device slots (trellis only, 3 per expert) plus the
  full pinned host copy. `suh`/`svh` of all experts stay on the device (a few
  hundred MB total for GLM).
- The pointer tables (`g_trellis` ... in `glm5.build_coop`) already are the
  indirection: entry e points to its slot if resident, else to the pinned copy. A
  miss just reads over the link; no stall logic in the kernel path.
- A host thread keeps per-expert exponentially decayed counts per layer plus the
  LRU order, and between graph replays rewrites table entries and issues async
  H2D copies of promoted experts on a copy stream per GPU (events order the table
  write after the copy). Tables are device int64 tensors updated in place, so the
  captured graphs are unchanged.
- Prefetch hook: after layer L's router, run `route_peek` for layer L+1 on L's
  router input (device-side top-k, one small D2H of 8-16 ids, or a device kernel
  that fills a "wanted" list the copy stream consumes). Only worth it when
  `t_x` x prefetches per layer fits the layer time, i.e. >= ~12 GB/s.
- Prefill is the easy case: a chunk's routing counts select which whole layers /
  experts to stage, and large chunks amortize a fetch over many tokens. Decode and
  prefill should share the same table but may use a different policy.
- Expert parallel shards (`M.e0..e1`) keep their own pools and their own links.
- Failure to avoid (llama.cpp #29887): a pool smaller than the experts one token
  activates thrashes (hit rate 0.7%). With the pinned fallback the table never
  needs a slot for a miss, so this degrades gracefully instead.

## Reproduce

```sh
# link
ucode -L $PWD/build/lib -L $PWD/src/ucode -ltorch=ampstein.torch examples/bench_zerocopy.uc 0 64
# routing trace over real sessions (needs the server stopped: it uses every GPU)
python3 tests/oracle/pi_samples.py samples.json
ucode -L $PWD/build/lib -L $PWD/src/ucode -ltorch=ampstein.torch examples/trace_routing.uc samples.json route.txt 6
python3 tests/oracle/routing_analysis.py route.txt
```

Use absolute `-L` paths: relative ones do not resolve the `ampstein.*` modules.

## Not covered

- End-to-end tokens/s with a real residency manager (everything above the table is
  trace-driven simulation).
- Long sessions: traces are short, so LRU numbers are cold-start pessimistic.
- Batched serving (several sequences share loaded experts).
- Quality: nothing here changes numerics; mixed-precision cold experts (a 2 bpw
  tier) would, and is not considered.
- Other related work: MoE-Infinity, Mira, Pre-gated MoE, llama.cpp #29887
  (LRU slot bank, no prefetch, <= 32-token batches), Strata (hot/cold tiering plus
  1-2 token-ahead prediction; the repository was not inspected).
