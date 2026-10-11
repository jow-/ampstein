# Multi-GPU: uneven tensor parallelism over slow PCIe

Status: largely implemented, with differences from this original design.
Tensor / expert parallel variants exist for every model family
(`qwen35_tp.uc`, `qwen4_tp.uc`, `glm5_tp.uc`, `dsv4_tp.uc`), including stages of
TP groups (`--tp --stages N`) and a custom P2P all-reduce with a scheme chosen
by rank count instead of NCCL; uneven shares are supported for two GPUs
(`--tp-share`). The cost model and reasoning below are the original
analysis; measured results are in `OVERVIEW.md`.
Companion to `MEMORY.md`.

## Goals and constraints

- Use every GPU present. Don't require power-of-two counts or identical
  cards.
- Tensor parallelism (TP) using NCCL over PCIe P2P.
- Plan for PCIe 3.0 x4 between cards: about 3.9 GB/s per direction in theory,
  roughly 3.2 GB/s in practice. Faster links later should only change the
  numbers in the cost model, not the design.
- One user: optimize single-stream latency, decode first. Don't tune for
  throughput across concurrent requests.

Current state: `qwen35.uc` takes a per-layer device map (`opts.devices`). That
is a layer split, i.e. pipeline-style placement with a single stream. Since
`ff9ed22`, the bindings carry device guards and graph capture works per device.

## Cost model: why decode wants TP and prefill wants pipelining

All numbers are estimates to replace with measurements (see step 1 of the
roadmap). The example is the 27B at 5 bpw: about 17 GB of weights, and a hidden
state of 5,120 × fp16 = 10 KiB per token.

**Decode (one token):**

- The step is bound by memory bandwidth: every weight byte is read once.
- On a layer split, GPUs take turns, so decode time is the sum of each GPU's
  share of the bytes divided by its bandwidth. Adding GPUs adds capacity, but
  no speed.
- On TP, every layer is split, so all GPUs read at once. Time is roughly
  17 GB / Σ bandwidth, plus communication.
- TP communication is two all-reduces per layer: after the attention or GDN
  output projection, and after MLP down. That's 128 per token, each only
  10 KiB, so **latency dominates, not bandwidth**. At an assumed 20–30 µs per
  NCCL all-reduce on PCIe, that's 2.5–4 ms per token. A one-shot P2P all-reduce
  kernel (below) should get to single-digit µs, or about 1 ms per token.
- Example: two 3090-class cards (~936 GB/s each). A layer split costs about
  18 ms per token. TP costs about 9 ms + 1–4 ms. **TP wins clearly on decode,
  even over x4 links.**

**Prefill (one chunk of C = 512 tokens):**

- Each all-reduce now carries 512 × 10 KiB = 5 MiB. A ring all-reduce moves
  about 2(N−1)/N of that per GPU, roughly 3 ms at 3.2 GB/s. Times 128, that's
  ~0.4 s per chunk, which is the same order as the chunk's compute.
- **TP prefill over x4 is therefore communication-bound** unless
  communication overlaps with compute. Split each chunk into two micro-chunks
  and all-reduce one while computing the other.
- A layer split moves just one 5 MiB hidden state per stage boundary per chunk
  (~1.6 ms). With chunks pipelined across stages (below), it parallelizes
  prefill with almost no communication.

The conclusion is a **hybrid**: TP within groups of GPUs, and pipeline stages
across groups. A planner picks the shape from measured costs.

## Placement model

Each layer gets a share vector over the GPUs: the fraction of that layer's
channels or heads each GPU holds. This one structure covers every case:

- `[1, 0, 0]` is a layer on a single GPU (layer split).
- `[0.5, 0.25, 0.25]` is uneven TP.
- Layers can differ from each other. For example, GDN layers can be weighted
  towards cards with little VRAM headroom, because their memory is fixed,
  while attention layers, whose KV grows, go where the headroom is.

**Planner objective:** minimize the per-token decode time
Σ over layers of max over GPUs of (share × layer bytes / bandwidth of that GPU),
plus the communication cost of each layer's sharding pattern (all-reduce
within a TP group, a point-to-point copy at a stage boundary). This is subject
to each GPU's memory budget: weights + fixed state + activation envelope + KV
per token × target context (see `MEMORY.md`).

The inputs are measured, not configured: per-GPU free memory and effective
bandwidth, and the P2P bandwidth/latency matrix. A small greedy search
or ILP is enough at 64 layers × a handful of GPUs.

exllamav3's `TPAllocator` (`model/model_tp_alloc.py`) is the reference for the
splittable/replicated storage accounting and the 128-channel granularity. Treat
it as the spec, the way `exl3.py` was treated for the tensor format (L2).

## Placement planner (contiguous layer split)

The base case, and the first thing to build. It needs no kernels: its output
is the per-layer `opts.devices` array that `qwen35.uc` already consumes. TP
groups plug in later as virtual GPUs.

**Units.** The model's layers in order, plus pseudo-units at the ends: the
embedding (if it stays on a GPU), final norm + lm_head (~0.95 GB at 6 bpw on
the 27B), and the MTP head. Each layer is split into its two sub-blocks
(attention/GDN, then MLP). A cut between them costs the same as a cut between
layers (one 10 KiB hidden-state transfer per token), but gives 128 units
instead of 64, so packing gets tighter.

**Per-unit costs**, computed before loading anything:

- Weight bytes: exact from the safetensors headers (`st.info().nbytes`), so
  mixed bitrates are handled automatically.
- Fixed state: ~3 MiB fp32 recurrent state per GDN layer.
- KV per token: 4 KiB per full-attention layer, 0 for GDN.

**Per-GPU inputs:** free VRAM via `cudaMemGetInfo` after CUDA and libtorch
are initialized, minus the activation reserve and a safety margin (the
lm_head GPU also needs the logits buffer); measured bandwidth; the link matrix
from the topology probe.

**Contiguity is a hard constraint:** every GPU gets one contiguous run of
units, so there are exactly (GPUs used − 1) boundaries.

**Objective: balance KV headroom, not just weights.** Context is limited by
the first GPU to run out:
`ctx_g = (free_g − weights_g − fixed_g) / kv_per_token_g`, and the achievable
context is the minimum over g. Only the 16 attention layers grow, so maximizing
the minimum pushes GDN-heavy runs onto tight cards. With uniform layers it
reduces to splitting proportionally to VRAM.

Two modes:

- **max-context** (default): maximize the minimum context, then break ties by
  decode time.
- **target T, then fastest**: among splits that hold T tokens, minimize the
  decode time Σ bytes_g / bandwidth_g plus the link latency at each boundary.
  This moves bytes to faster cards.

**Algorithm:**

1. For a fixed GPU order:
   - max-context: binary-search T. The feasibility check is greedy: fill the
     current GPU until the next unit wouldn't fit `weights + fixed + T·kv`,
     then move on. For a fixed order, this check is exactly optimal.
   - target T: a dynamic program over (unit, GPU) minimizing decode time under
     the capacity limits, O(G·U²). With 8 GPUs and 128 units that's trivial.
2. Over GPU orders: enumerate permutations, which decide the boundary links
   and which card hosts lm_head. 8! = 40,320 cheap checks is fine. Beyond
   that, sort by link locality and improve with local swaps.
3. Allow a GPU to stay empty when its share would be negligible but it would
   still add a boundary.
4. Print the plan at load (unit range, GiB, and per-GPU context, with the
   limiting GPU marked). Allow overrides: pin units to GPUs, exclude GPUs,
   change the margin.

**With TP:** a TP group enters as a virtual GPU, with summed capacity and
summed bandwidth discounted by an efficiency factor from the measured
all-reduce latency. Choosing which cards to pair is an outer search on top.
Contiguity applies between stages; the uneven shares below apply within a
stage.

## Sharding rules (Qwen 3.5 dense)

| Module | Split | Granularity on the 27B | Notes |
| --- | --- | --- | --- |
| Attention q/k/v, gate | Column-parallel by KV-head group | 4 KV heads, so at most 4 shards; 6 q heads per KV head | Uneven works (e.g. 2/1/1), but KV growth then follows the head count. Plan for it. |
| Attention o_proj | Row-parallel, then all-reduce | Matches the head split | |
| GDN projections | Column-parallel by head | 16 key / 48 value heads | Recurrent and conv state split with the heads, so state memory stays fixed per GPU. |
| GDN out_proj | Row-parallel, then all-reduce | | |
| MLP gate/up | Column-parallel | 17,408 / 128 = 136 channels | Finest granularity, so the best place to absorb unevenness. |
| MLP down | Row-parallel, then all-reduce | 128-aligned in k | |
| lm_head | Vocab-parallel | 248,320 / 128 | No logits gather. Each GPU runs a local argmax (generalizing `argmax_advance`) and exchanges one (value, index) pair. Temperature sampling exchanges each shard's top-k plus its logsumexp. |
| Embedding | Not sharded | | Keep it in host memory (`MEMORY.md`), and copy the row to every GPU that needs it. |
| Norms, biases | Replicated, small | | Add biases once, on one rank, before the all-reduce. |

EXL3 specifics: the trellis tiles are 16×16, and the kernel's column blocks
are 128. For row-parallel splits, each shard needs its own slice of `suh`
(per input channel), while `svh` (per output channel) stays whole. exllamav3's
`tp_import_split` shows how it handles both directions. Follow it and
oracle-test against its TP mode.

## Communication backends

| Backend | Use | Notes |
| --- | --- | --- |
| NCCL (P2P transport) | Prefill all-reduces (bandwidth), and as the correctness reference | Use one process driving all GPUs: `ncclCommInitAll` plus `ncclGroupStart/End` around per-device calls. NCCL initializes lazily, so warm it up at load (exllamav3 does too). |
| One-shot P2P all-reduce kernel | Decode all-reduces (latency) | Each GPU writes its 10 KiB to its peers through P2P-mapped buffers, sets a flag, and sums locally after waiting on peer flags. This is the pattern of vLLM's custom all-reduce and TensorRT-LLM's one-shot mode. It's plain kernels, so it captures into CUDA graphs trivially, and it works for any GPU count. |
| exllamav3 native (pinned host + CPU reduce) | Fallback when P2P is unavailable | Vendored under `parallel/`. Slower, but needs no P2P. |

**Spike to run first: single-process, multi-device graph capture with
collectives.** One ucode thread drives every device. Decode needs one graph per
device per step, and the collectives have to work inside those graphs.

- With the custom kernel this is easy: each device's graph waits on flags in
  peer memory.
- With NCCL, grouped calls across devices from one thread inside stream
  capture are the part to verify early. If they don't work, NCCL stays
  prefill-only, and decode uses the custom kernel.

## Prefill pipelining

Chunked prefill (`MEMORY.md`) makes a layer split parallel: while stage 2
processes chunk k, stage 1 is already on chunk k+1. With S stages and many
chunks, prefill approaches S× speedup over one stage, at ~1.6 ms of
communication per chunk boundary.

What stays sequential:

- GDN layers carry recurrent state from chunk to chunk, so one stage can't
  start chunk k+1 before it has finished chunk k. That's already the case
  within a stage, so it doesn't stop pipelining across stages.
- Attention for chunk k+1 needs KV for chunks ≤ k in the same layer, which
  lives on the same stage. Fine.

This is the "async" part that does fit: per-stage queues, one CUDA stream per
stage, and events between stages. Driving it from a single ucode thread works,
because everything is asynchronous launches.

## Async setups: prefill on some GPUs, decode on others

Disaggregated serving gives prefill and decode separate GPU pools (DistServe,
Splitwise). For this setup it **doesn't fit**:

- Both pools need a full copy of the weights, which halves the largest model
  that fits. That contradicts the main goal.
- With one user, prefill and decode of the same request never run at the same
  time, so there's nothing to overlap. Disaggregation pays off as throughput
  across concurrent requests.
- The handoff costs a KV transfer of 64 KiB per token plus ~150 MiB of GDN
  state. That's ~0.65 s for a 32k context at x4 speeds. Tolerable, but pure
  overhead here.

Asymmetric use of spare GPUs that does fit:

- **Pipelined prefill across stages** (above): different GPUs work on
  different chunks at the same time.
- **A draft model on a weak or leftover GPU** for speculative decoding, beside
  or instead of MTP. A small card can't hold a TP shard worth having, but it
  can run a 0.5–1.5B draft. That needs a draft model with the same tokenizer.
- **Hosting the fixed extras** on a card too small for a useful shard: the
  vision tower, the MTP head, or the lm_head shard.

## Prior art

Surveyed 2026-10-07. The exllamav3 entries come from the local checkout (`~/devel/exllamav3` at `d3739fd`, the same commit as `vendor/exl3`); vLLM and llama.cpp entries from their documentation and search results, not their source. "Not found" means not found, not proven absent.

| Idea | vLLM | llama.cpp | exllamav3 |
| --- | --- | --- | --- |
| Tensor parallelism, uneven GPUs | Heads must divide evenly by the TP size | `--split-mode row`; `tensor` mode marked experimental | **Yes:** `TPAllocator` (`model/model_tp_alloc.py`) splits by memory ratio at 128-channel granularity, any GPU count |
| Layer split by proportions | Pipeline parallelism | `--tensor-split` proportions (e.g. `2,1`), default layer split | Autosplit across devices |
| One-shot P2P all-reduce | **Yes** (custom all-reduce), but disabled on more than two PCIe-only GPUs or without P2P | n/a | Own backends: NCCL, or pinned host memory with a CPU reduce (`model/model_tp_backend.py`) |
| Planner balancing KV growth headroom (hybrid models) | Not found | Not found | Not found |
| Bandwidth-aware placement, TP + pipeline hybrid on slow links | Not found | Not found | Not found |

**Study before building:** `model/model_tp_alloc.py` (splittable vs replicated
storage, per-device limits), `model/model_tp_backend.py` (NCCL warmup, the
native backend), and `modules/linear.py` `tp_import_split` (EXL3 row and
column splits). vLLM's custom all-reduce is the reference for the one-shot
kernel; its PCIe restriction is the case this design targets.

## Roadmap

1. **Topology probe** (`examples/topo.uc`): per-GPU free memory and effective
   bandwidth, a P2P bandwidth and latency matrix, and NCCL all-reduce latency
   at 10 KiB and 5 MiB. Every later decision depends on these numbers.
2. **Spike:** single-process NCCL plus multi-device graph capture; a minimal
   one-shot P2P all-reduce kernel; compare their latency.
3. **TP for MLP only** (column gate/up, row down, one all-reduce) on two GPUs,
   oracle-checked against exllamav3's TP mode. The MLP is the largest share of
   the weights, and the simplest sharding.
4. Attention and GDN sharding by heads, including split KV and GDN state.
5. Vocab-parallel lm_head with the distributed argmax.
6. Planner: share vectors from the measured topology and the `MEMORY.md`
   budgets. Uneven and mixed TP/pipeline splits.
7. Pipelined chunked prefill across stages, and micro-chunk overlap inside TP
   groups.

## Open questions

- The actual GPU inventory per setup (models, VRAM, link widths). The planner
  should handle any mix, but the first spike should target the most common one.
- Is P2P open on every pair, or only some? The P2P matrix from step 1 tells
  the planner which pairs can use the one-shot kernel.
- Is an all-reduce payload below fp16 (fp8, int8) worth its accuracy cost for
  prefill? Defer until step 7 measurements show prefill is
  communication-bound in practice.
