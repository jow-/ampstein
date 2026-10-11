# Memory model: no late allocations, growable KV cache

Status: partly implemented. Memory is planned at load (weights, graphs, KV,
state buffers, device slots) and the sparse-attention models keep ring
buffers for raw KV; the growable KV cache, the sealed shape envelope and the
mixed-precision live KV cache described below are still proposals.

## Context and goals

Ampstein runs on a homelab: one user, heterogeneous GPUs, and the aim of
fitting the largest possible model. Failures don't cost money, but they cost
time, so the expensive failure is the one that happens after minutes of
prefill.

Two goals:

1. **No allocations after load.** When the model has loaded and reported that
   it fits, no later request may fail with an out-of-memory error in the middle
   of compute.
2. **A KV cache that grows on demand.** Allocate a configured minimum at load
   and fail the load if it doesn't fit. Beyond that, grow as tokens arrive. When
   growth fails, return a "prompt too large" error, so the client can run its
   context-compaction logic.

Goal 2 is the one deliberate exception to goal 1. It stays safe because growth
is the only allocation site, it runs only at host-side checkpoints between
chunks or replays (never inside compute), and a failed growth is recoverable.

Example numbers below are for Qwen3.8-27B (`model_type: qwen3_5`): 64 layers,
16 of them full attention (4 KV heads × head_dim 256) and 48 GatedDeltaNet,
vocab 248,320, hidden 5,120.

## What allocates late today

| Grows with | Where | Size on the 27B |
| --- | --- | --- |
| Prompt length: full-chunk logits | `forward_chunk` returns `(1, S, vocab)`, and `generate` prefills the whole prompt as one chunk | 248,320 × 2 B ≈ **485 KiB per prompt token**: a 16k prompt allocates ~7.6 GiB just for logits |
| Prompt length: activations | Unchunked prefill: every intermediate is O(S) | Scales with S |
| Context length: attention scratch | `chunk_mask` is `(S, pos+S)`; eager `sdpa` may materialize scores | O(S × ctx) |
| Context length: KV cache | Allocated per `generate` call, sized prompt + max_new | 64 KiB per token (see below) |
| GC timing | ucode frees dead tensors only when the GC runs, so the caching allocator's peak depends on collection timing | Unpredictable |
| Lazy library state | cuBLAS workspaces, SDPA backend choice, the exl3 autotuner, graph instantiation | Small but unbounded in timing |

What doesn't grow: the GDN recurrent state is 48 × 128 × 128 × fp32 ≈ 3 MiB per
layer, ~150 MiB across 48 layers, independent of context. The conv state is
negligible. This is the hybrid architecture's big advantage here.

## Design part 1: a fixed shape envelope, then seal

1. **Chunked prefill with a fixed chunk size C** (a load option, default
   around 512). The last chunk is padded. Activation memory then depends on C
   and nothing else.
2. **Logits only where they're needed:** the last position, or the W+1
   positions an MTP verify step scores.
3. **Attention whose scratch doesn't grow with context:** a flash-style
   chunked prefill kernel over the cache with a causal offset. It's the
   natural generalization of `decode_attn` from 1 query to C queries; check
   first whether the vendored `attention.cu` fits. Masks are never
   materialized.
4. **Reserve, then seal.** After loading weights, run every worst-case shape
   once: a full chunk at maximum context, an MTP verify window, a decode step.
   This makes the caching allocator, cuBLAS, the autotuner and graph
   instantiation all reserve their peak. Then seal:
   - First, detection: a `torch.memory_stats(dev)` binding exposing the
     allocator's reserved bytes and segment count, plus a test that runs a long
     conversation and asserts neither grows after sealing.
   - Later, enforcement: a pluggable-allocator wrapper that raises
     "allocation after seal" instead of calling `cudaMalloc`. That would be a
     bug, not an OOM, so it should be loud.
5. **Capture the prefill chunk as a CUDA graph**, like decode. Replay never
   allocates, so the guarantee becomes structural instead of depending on
   every op following a rule.
6. **Print a budget at load**, for example:
   `weights 17.1 GiB · fixed state 0.15 GiB · activations 0.9 GiB · free for KV now: 61k tokens`.
   Sealing is what makes this number trustworthy.

## Design part 2: growable KV cache

Per token, the 27B needs 16 attention layers × K+V × 4 heads × 256 × fp16 =
**64 KiB**. So 32k context is 2 GiB, and 128k is 8 GiB.

### Mechanism: CUDA virtual memory

- Realloc-and-copy needs twice the memory at exactly the worst moment, and
  moving the buffer breaks captured graphs.
- A paged cache with block tables in every attention kernel is worth doing
  later for batching, but isn't needed here.
- **Use virtual memory instead.** Reserve address space for the theoretical
  maximum context once (`cuMemAddressReserve`). Map physical memory in
  granularity-sized pieces as the cache grows (`cuMemCreate` + `cuMemMap` +
  `cuMemSetAccess`). Addresses never move, so existing kernels and captured
  graphs keep working. PyTorch's expandable segments use the same technique.
- **Use a sequence-major layout, `(max_seq, kvh, hd)`,** so growth is a
  contiguous tail. The current head-major `(kvh, max_seq, hd)` strides each
  head by `max_seq`. `decode_attn` indexes `(pos*kvh + j)*hd`, and the eager
  path uses strided views.
- At the 2 MiB granularity, one layer's K covers 1,024 tokens, which is fine.

### Policy for a single user

| When | Action |
| --- | --- |
| Load | Weights → activation envelope → seal → map `kv_min` tokens. Any failure fails the load. |
| Request arrives | Try to map enough KV for prompt + `max_tokens`, or a configured generation budget, **before any compute**. A failed mapping returns in milliseconds, so a too-large request costs no prefill time. On failure, return `context_length_exceeded` (HTTP 400) with the current limit in the message. |
| During decode | Normally nothing, since everything was mapped up front. If an open-ended budget does run out mid-generation, end with `finish_reason: "length"` and keep the tokens already produced. |
| Ceiling | Not sticky. Try growing again on each request, because a failure is cheap and other servers on the same GPUs free and take VRAM over time. Remember the last known limit only for logging. |
| Release | Never, by default. Mapped KV stays for the life of the process. An optional idle-unmap down to `kv_min` only helps when the cards are shared. |
| Multi-GPU | Each device maps its own layers' (or heads') KV. The first device to fail sets the limit. See `PARALLEL.md`: placement should balance KV growth headroom, not just weights. |
| MTP | The draft cache grows on the same schedule. |

## Mixed-precision KV cache

The cache doesn't need one format everywhere. Besides K vs V, it can be split
along four axes:

| Split by | Example | Fit with the kernels | Use |
| --- | --- | --- | --- |
| **Layer** | Layer 11 at fp16, the rest at 4-bit | Easy: each layer's cache is a separate tensor, and the attention kernel is chosen per layer | Yes; telemetry picks the layers |
| **KV head** | One sensitive head at 6-bit, the others at 4-bit | Natural: `decode_attn` runs one block per KV head (`grid.x`), so each block looks up its head's format. Each head is stored as its own segment. | Yes, if sensitivity is concentrated in a few heads |
| **Token position** | Sink block and recent window at fp16, the middle at 4-bit | Natural: `decode_attn` already splits the slot range and merges the splits with online softmax, so each split can read a different format with no extra pass | **Yes, the main scheme** |
| **Channel** | Outlier channels at fp16 | Poor: breaks `q_cache`'s groups of 32, and the Hadamard rotation already spreads outliers | No |

### Position-based policy

- **Sink block at fp16:** the first 256 tokens (BOS and the start of the
  system prompt). These tokens absorb a large share of attention in nearly
  every head, so their errors matter disproportionately.
- **Recent window at fp16:** e.g. the last 4k tokens, which carry most of the
  attention for the current step.
- **Quantize when a block leaves the window.** New tokens are written as
  fp16. When a 256-token block falls out of the window, it is quantized once
  and never touched again. `q_cache` already works in 256-token pages, the same
  block size as the prefix cache, which then stores each block in its final
  format with no transcoding (`SESSIONS.md`).
- **Combinations:** the middle segment can additionally use per-layer or
  per-head bit widths, e.g. most layers at 4-bit and a few sensitive ones at
  6-bit.

### Size on the 27B (256k session)

| Layout | KV size |
| --- | --- |
| All fp16 | 16 GiB |
| Sink + 4k window fp16, rest 4-bit (4.5 bits effective) | ~0.26 GiB + ~4.5 GiB ≈ 4.8 GiB |
| All 4-bit | ~4.5 GiB |

The window costs about 0.3 GiB over all-4-bit, and protects the tokens that
matter most.

### Kernel work

- A format table per (layer, KV head, segment): fp16, or `q_cache` with its
  bit width.
- `decode_attn` and the planned chunked prefill kernel gain an in-kernel
  dequantization path for quantized segments. Their split-and-merge structure
  already does the rest.
- The VMM growth (above) applies per segment: the fp16 window is a ring of
  fixed size, and only the quantized region grows.

### Choosing the settings

Telemetry (`TELEMETRY.md`) supplies the inputs: the shadow quantization error
per layer and head picks the bit widths, and the attention mass by position
picks the window size. The KLD replay confirms the final setting (mean KLD
< 1e-3).

## Related wins for "largest model, most context"

- **Reuse the cache across turns.** Agentic clients resend the whole
  conversation every turn, and today each turn re-prefills it from scratch.
  Compare token prefixes and prefill only the new suffix. GDN layers can't
  rewind, so snapshot their recurrent and conv state at each turn boundary.
  This is the same mechanism the MTP rollback already uses (`gdn_snap`). For an
  agentic workload this is likely the biggest time saver available.
- **Quantize the KV cache.** The vendored exllamav3 `cache/q_cache*` kernels
  would cut 64 KiB per token to roughly a quarter at 4 bits, about 4× the
  context per GiB.
- **Keep the embedding table in host memory.** On the 27B it's 248,320 × 5,120
  × 2 B ≈ 2.4 GiB on a GPU, while decode reads one 10 KiB row per token. A
  kernel reading a mapped, pinned host buffer by device-side token id stays
  graph-safe and costs microseconds over PCIe. Prefill reads C rows per chunk,
  still small.
- **Upload only what the text model uses.** Skip the vision tower, and the MTP
  weights when MTP is off. `upload()` currently uploads every tensor.
- **Drop the raw copy after a dtype cast.** `gpu16()` currently caches both the
  bf16 upload and its fp16 cast.

## Prior art

Surveyed 2026-10-07. The exllamav3 entries come from the local checkout (`~/devel/exllamav3` at `d3739fd`, the same commit as `vendor/exl3`); vLLM and llama.cpp entries from their documentation and search results, not their source. "Not found" means not found, not proven absent.

| Idea | vLLM | llama.cpp | exllamav3 |
| --- | --- | --- | --- |
| Preallocate, no late allocations | Profiles peak memory with a dummy run at startup, preallocates a fixed KV pool, captures CUDA graphs | `--fit` adjusts unset settings at load to fit device memory | Fixed-size cache allocated up front |
| Sealed allocator, load-time budget | Not found | Not found | Not found |
| KV growing on demand, `context_length_exceeded` when it can't | Fixed pool; prompts over `max_model_len` rejected | Fixed `--ctx-size` | Fixed cache size |
| KV quantization | fp8 natively; lower via add-ons (KIVI) | q8_0, q5_x, q4_x, iq4_nl, separately for K and V (`--cache-type-k/-v`) | `q_cache`, 2–8 bits, Hadamard-rotated groups of 32 |
| Mixed precision by position or per layer | Not found | Not found | Not found |

**Research behind the mixed-precision layout:** KIVI keeps the most recent
tokens in fp16 and quantizes older ones; KVQuant keeps the first token in
fp16 (the attention-sink effect). Neither is part of a mainstream engine.

**Why engines probably haven't adopted it** (my reasoning, not the engine
teams'):

- Batched paging: every sequence has its own window boundary, so formats
  differ per page and per sequence. That means two page pools, conversion
  when pages leave a window, format-aware prefix hashes, and per-page format
  flags in the attention kernels, all in the most tuned code they own.
- Third-party kernels: vLLM and SGLang rely on FlashAttention / FlashInfer,
  tuned for one dtype per tensor. llama.cpp has one quantization type per KV
  tensor, so mixing means two attention ops plus a merge.
- Lower payoff on datacenter hardware: fp8 KV halves the cache with
  negligible loss on native hardware support, next to plenty of HBM. A 4k fp16
  window per sequence × hundreds of sequences also erodes the saving.
- Possibly modest quality gain at 4 bits: the research gains are strongest at
  2–3 bits, and Hadamard-rotated 4-bit already holds up reasonably well.

**Why the kernel overhead should be small here:** split boundaries at the
window edges give each thread block one format (one uniform branch, no
divergence). Because the Hadamard rotation is orthogonal up to a constant, q
can be rotated once per head per step and dotted directly against the
quantized keys; V accumulates in rotated space and is un-rotated once. What
remains per value is an unpack times a scale, while decode reads 3.6× fewer
bytes. One user means one window boundary, so none of the batched-paging
complexity applies.

**Settle it before writing kernels:** an offline KLD replay comparing uniform
4/3/2-bit against the same widths with an fp16 first block + 4k window.
Quantizing and dequantizing stored KV offline is enough, with no new attention
kernel. If the window barely helps at 4 bits but rescues 2–3 bits, the feature
is worth building, and the result says which widths it unlocks.

## Build order

1. Logits for the last position only, plus fixed-size chunked prefill. This is
   the cheapest step, and it removes the biggest late allocation.
2. Up-front KV mapping per request, with the fast `context_length_exceeded`
   path. This can start with plain preallocation before virtual memory exists.
3. Prefix reuse across turns, with GDN state snapshots.
4. The memory-stats binding, the seal test, and the load-time budget
   printout.
5. Sequence-major KV layout plus virtual-memory growth.
6. The chunked prefill attention kernel, plus prefill graph capture.
7. Embedding table in host memory; quantized KV cache.

## Open questions

- The chunk size C trades prefill speed against the fixed activation reserve.
  The right default should come from measurement on the real cards.
- Is `cuMem*` available through the libtorch build in use? The vendored
  `cuda_drv.cpp` suggests the driver API is already being reached.
