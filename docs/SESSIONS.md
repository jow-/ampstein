# Sessions: parked state, the RAM store, slots and time slicing

Status: the host-RAM store, resident device slots, time slicing and the
hot-reload state file are **implemented**; the radix-tree block cache, the
disk tier and lossless/lossy entry compression beyond the age tiers are
**ideas** (marked as such below). Companion to `MEMORY.md` (allocation model)
and `PARALLEL.md` (placement). This document replaces the earlier
`PREFIX_CACHE.md` and `CONCURRENCY.md`.

## 1. The problem

The server is single-user, but an agent harness is not single-request: a
main conversation, a compaction or summary call, title generation, tool
sub-agents, a second chat. Recomputing a 100k-token prompt costs 60-100 s, so
three goals matter:

1. **No re-prefill when switching conversations** (latency, not throughput).
2. **No head-of-line blocking** (a short side request must not wait behind a
   long generation or prefill of the main one).
3. **More aggregate tokens/s** (fill idle pipeline stages) — the weakest goal
   for a single-user rig.

The engine runs one job at a time against one live model state, plus one
snapshot (the stable prefix before the generation prompt). Everything below
is about what happens to *other* conversations' state.

## 2. What is implemented

### 2.1 Live state and snapshot

The live state is the KV cache, recurrent (GDN) state and draft state of the
current conversation. A request that extends it prefills only the new
suffix. The snapshot is the state at the end of the stable prefix (before the
chat template's generation prompt), so the next turn of the same
conversation can resume from there even though the generation prompt and the
reply differ. Every request that is not a plain live continuation logs why:
matched prefix lengths and the tokens around the first divergence.

### 2.2 Host-RAM store (`--cache-ram GiB`)

A request that does not continue the live state moves it to page-locked host
RAM instead of dropping it; a later request that extends a stored state swaps
it back (about a second for 100k tokens against ~100 s of prefill).

- An entry keeps the token ids, the attention KV (chunks of 4096 tokens), the
  GDN states, the draft cache and hidden, and the snapshot, so both reuse
  points of the old state survive.
- LRU by bytes within the budget; states below 2048 tokens are not stored
  (cheaper to recompute), and a hit must save at least 512 tokens.
- **Age tiers** for idle entries of at least 16k tokens: the last 8k tokens
  stay fp16, up to 40k back int8 (groups of 32), older int4; the first chunk
  never drops below int8. `--cache-quant off`, `--cache-quant-min N`,
  `--cache-quant-tail N`, `--cache-quant-q8 N` control it. The far past
  matters least to the next token, so the loss is placed where it hurts least.
- GGUF Qwen3.8-27B on ROCm (single GPU) parks through this path with the fp16 or the Q8 KV cache (Q8 rows are stored verbatim,
  never age-tiered).
- Supported for the Qwen3.5-family models, GLM-5.3-Flash and DeepSeek-V4
  (through the engine's `cache_export` / `cache_import` hooks). GLM-5.3 entries
  hold the MLA latent rows (age-tiered like KV; the Q8 cache form is kept as
  is), the indexer plane and complete pools (always fp16), the KDA states and
  the MTP draft cache; under TP the replicated latent cache is stored once.
  With `--slots` the entries first go to resident device slots (exact, no age
  tiers; one latent-cache replica per layer) and are demoted to host RAM, tiered,
  when the slots run out. Measured on DeepSeek-V4
  with ~3k-token states: swap-out 160-280 ms, swap-in 50-80 ms, ~0.1 GiB per
  entry (dominated by fixed-size parts: window rows and drafter rings).

### 2.3 Resident device slots (`--slots auto|N|0`)

After everything else is allocated, the engine takes full-size state arenas
("slots") until a GPU is out of memory or down to `--slot-margin` MiB
(default 512). A parked conversation in a slot swaps back device-to-device in
milliseconds, with no PCIe traffic. Available where the architecture provides
`new_slot` (DeepSeek-V4, whose MQA / compressed KV makes a slot only 2-4 GB at
256k tokens).

### 2.4 Time slicing (`--slice-tokens N`, `--slice-ms N`)

A decode-phase job that has generated N tokens (default 256) while other
sessions wait is parked at a step boundary through the store (a device slot
when there is one, else host RAM). The next session runs, and the parked one
resumes later exactly where it stopped; sessions alternate round-robin.
Without contention nothing changes. Prefill is not sliced. A slice should be
several times the park + restore cost (~30 ms on a device slot, 100+ ms
through RAM for a few thousand tokens of dense KV).

### 2.5 Hot reload

`save_state` / `restore_state` write the live state to a safetensors file
and read it back, so a server restart (a code change, a model reload)
continues the conversation without prefill. The server saves on SIGINT /
SIGTERM to `/tmp/Ampstein-cache-PORT.safetensors` (+ `.json`, and `.tree` for
the entry form) and restores at the next start; the saved placement (GPUs,
TP, stages, KV form) must match.

Every state file starts with a **header** (a `header` tensor of JSON inside the safetensors): format version, form of the state
(dense / entry), KV cache format (`FP16` / `Q8`), layer count, and what the server passes as `state_header` (a CRC-32 of the model
directory's manifest, or of name + size + mtime for a single file; GPUs, TP, stages, KV bits where they shape the state). A restore
whose engine differs in any of these, or a file without a header, is refused (`incompatible state file: kv was "Q8", is "FP16"`) and the
file is kept for the server it belongs to. Token count, `--ctx` and the MTP window are not compared. The CRC is an accident detector,
not a seal.

- **Qwen3.5 family:** the dense form: token ids, attention KV (used part),
  GDN states, draft cache, last hidden. The prompt snapshot is not saved.
- **GLM-5.3 (and any model with the store's entry layout):** the state the
  host-RAM store would hold, exactly: MLA latent rows (age-tiered like a
  stored entry), indexer planes and pools, KDA states, the MTP draft cache,
  and the snapshot / turn checkpoint, so a first prompt that diverges after the
  restart still rewinds instead of prefilling. The entry's tensors are the
  file, its structure `.tree` (JSON); restoring is the store's own swap-in.
  States below 2048 tokens are not saved. DeepSeek-V4 is not wired into the
  server yet.
- **Qwen3.8-Flash-Next (TP):** the same entry layout: the QSA attention caches of the 12 attention layers (K, V,
  raw indexer keys, pooled blocks; on each layer's owner rank only, exact fp16, ~30 KB per token), the 36 GDN
  states, the PLE state (`aux_get`: conv tail + id context, stored with the entry), the MTP draft cache and the
  snapshot. RAM store, resident device slots (a swap is ~5 ms, 3k tokens) and the hot-reload file all work
  (`tests/engine_store_qwen4.uc`). Measured on the 2.05 bpw quant, 2 GPUs: 38k tokens saved at SIGTERM (1.4 GB
  file), restored in 0.5 s at the next start, and the repeated 38k-token prompt then prefilled 7 tokens
  (snapshot reuse) in 0.3 s end to end.

## 3. What a cache entry is on a hybrid model

- **Attention layers** have per-token KV, so any prefix length can be reused
  (Qwen3.8-27B: 16 of 64 layers, 64 KiB per token at fp16).
- **GDN layers** have no per-token state, only the recurrent and conv state
  at one exact position (48 layers: ~147 MiB per snapshot, independent of
  length).
- **Consequence:** a prefix can only be resumed where a GDN snapshot exists,
  i.e. "attention KV for [0, n) + GDN state at n". The engine keeps the live
  state and the stable-prefix snapshot; the ideas below would add more
  checkpoints.

Sparse-attention / MQA models are different: DeepSeek-V4 caches ~7 KB per
token across the whole model (CSA and HCA layers, window rings), so
~0.7-1.4 GB at 100k tokens and 2-4 GB at 256k — small enough that device
slots are the natural tier.

## 4. Concurrency options

Facts that decide them (DeepSeek-V4, 3 stages x TP-2): decode is one
multi-GPU CUDA graph executed stage after stage, so while one stage runs the
other two are idle; decode reads only 6 of 256 experts per token, so batching
sequences reads the *union* of their experts and saves only the dense part;
speculative decoding already converts spare compute into tokens of one
sequence.

| Option | Effect | Status |
|---|---|---|
| A. Swap sessions through host RAM | goal 1 completely; goal 2 at request granularity (a side request evicts the main one, which swaps back) | **done** (2.2) |
| B. N resident slots + step-level scheduler | a side request neither evicts nor waits behind the generation; inter-token latency grows by one prefill chunk while it runs | **done** for slots (2.3) and time slicing (2.4); a per-step scheduler interleaving prefill chunks with decode steps is not built |
| C. Stage-staggered decode of up to 3 sequences | the only option that raises aggregate tokens/s (estimated up to 2-3x for 3 sequences); needs per-stage graphs on separate streams and a host loop with three rounds in flight | idea |
| D. True batched decode | batch-aware attention/compressor kernels; sub-linear gain, competes with speculative decoding for the same compute | not recommended before C |

Open measurements: per-stage busy fraction in single-stream decode (the
headroom for C), decode step time vs window and the launch-overhead share, and
swap time vs tokens at 100k / 256k.

## 5. Ideas not implemented

### 5.1 Radix tree over token blocks

Fixed 256-token blocks keyed by a hash chain `h_i = hash(h_{i-1}, tokens)`
(as in SGLang and vLLM), KV stored per block and layer and deduplicated
across sessions (a shared system prompt or tool definitions of 10k+ tokens
stored once), storage per layer so a restore survives a changed layer split.
GDN snapshots hang off specific tree nodes.

**Thinned checkpoints:** keep the end of the system prompt + tools, the end
of the latest prompt and the last four message boundaries; keep older
boundaries log-spaced (8k, 16k, 32k... back). At ~10 snapshots per 256k
session that is ~1.4 GiB instead of 18 GiB for one per message.

### 5.2 Write-through KV mirror and immediate snapshot staging

Finished KV blocks never change, so each could be copied to RAM in the
background as it is written; switching sessions then costs only dropping the
GPU copy. Snapshots would be taken as a fast device-side copy into a fixed
staging buffer and drained to pinned RAM on a separate copy stream. With TP,
background copies should be throttled because they compete with the
latency-sensitive decode all-reduces.

### 5.3 Disk tier

Content-addressed files per (block hash, layer group) plus snapshot files,
namespaced by a model identity key (config, quantization, weight headers, KV
dtype, RoPE parameters, engine version — an engine upgrade invalidates
everything, since kernel changes alter numerics). Reads would reuse the
`h2d_file` staging ring; writes a background thread that never touches the
VM. Entries are conversations in another form: store them with tight
permissions.

### 5.4 Compression

| Scheme | Expected effect | Notes |
|---|---|---|
| fp16 + zstd | 5-15% | fp16 mantissas look like noise |
| byte-shuffle + zstd | 20-35% | lossless |
| `q_cache` quantization (8 / 4 bit) | 1.9x / 3.6x | main lever; GPU-side, already vendored; the age tiers of 2.2 use this family |
| GDN state fp32 -> bf16 | 2x on snapshots | risky: the recurrent state accumulates; test first |

The vendored `q_cache` kernels rotate groups of 32 values by a Hadamard
transform, scale by the group maximum (one fp16 per group, +0.5 bit/value) and
quantize to 2-8 bits. K and V may use different widths (keys are typically
more sensitive, e.g. K6/V4). Quantize once and never re-quantize between
tiers. KV-dropping schemes (H2O, SnapKV, windows) are rejected: they break
exact prefix reuse.

Acceptance for any lossy setting: mean KLD after restore below 1e-3 against a
fresh prefill of the same tokens (`ucode/metrics.uc`, `tests/kld.uc`).

### 5.5 Sizes and restore times (Qwen3.8-27B, estimates)

| Tokens | fp16 KV | 8-bit | 4-bit |
|---|---|---|---|
| 256k | 16 GiB | ~8.5 GiB | ~4.5 GiB |
| 512k (pool) | 32 GiB | ~17 GiB | ~9 GiB |

| Source -> GPU | Bandwidth | 256k session, fp16 | 4-bit |
|---|---|---|---|
| RAM, PCIe 4.0 x16 | ~25 GB/s | ~0.75 s | ~0.25 s |
| RAM, PCIe 3.0 x4 | ~3.2 GB/s | ~5.8 s | ~2 s |
| NVMe Gen4 | ~6 GB/s | ~3.1 s | ~1.1 s |
| Recompute (prefill 256k) | | ~5-15 min | |

With a layer split each GPU loads its own layers' share over its own link in
parallel, so the RAM rows shrink by roughly the GPU count.

## 6. Prior art

exllamav3 has a CPU tier in its page table (`generator/pagetable.py`,
`cpu_cache.py`) and recurrent checkpoints in system RAM (`cache/recurrent.py`);
llama.cpp has `--cache-ram`, per-slot save/restore and context checkpoints for
recurrent state; vLLM offers prefix caching and, through LMCache, RAM and disk
tiers. What this design adds on top is the age-tiered quantization, the
thinned checkpoint policy and (as ideas) the write-through mirror and the
identity-keyed disk tier.
