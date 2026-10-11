# Telemetry: histograms and heatmaps

Status: design proposal (not implemented). Companion to
`MEMORY.md`, `PARALLEL.md` and `SESSIONS.md`.

## Goals

Record statistics during normal use, and use them to decide:

1. **Whether, and where, the KV cache can be quantized** below fp16, based on
   the values the model actually produces in real sessions.
2. **Which MoE experts activate**, once MoE support lands, to drive expert
   placement, prefetching, and possibly pruning.

## Ground rules

- **Count on the device, into fixed-size buffers allocated at load.** No
  allocations after load (`MEMORY.md`), and the counters work inside captured
  CUDA graphs: atomics into persistent buffers.
- **The host copies a small snapshot periodically,** every N tokens or at the
  end of each request, asynchronously and off the critical path.
- **Overhead budget: under 1% of decode time.** Every level is switchable on
  its own. The expensive ones are sampled or offline.
- **Statistics, not content.** Recording token sequences (needed for replay)
  is a separate, opt-in switch, stored locally only.

## 1. KV quantization

Value histograms alone are a weak proxy. What matters is how much
quantization changes attention, so three levels, from cheap to expensive:

| Level | What is recorded | Cost | Answers |
| --- | --- | --- | --- |
| **Distributions** | Per layer × KV head × {K, V} × channel: absmax, mean, sum of squares. Per layer × head × {K, V}: a 32-bin magnitude histogram. | Nearly free: folded into `decode_attn`, which already reads each new k/v row. The fp16 exponent field gives natural log₂ bins, so a bin update is a shift plus an atomic add. | Where the outliers are. Keys are stored after RoPE, so outlier channels are expected in the low-frequency dims; the per-channel heatmap shows it. |
| **Shadow quantization error** | For each finished 256-token block, on a side stream: quantize and dequantize with `q_cache` at 8/6/5/4/3 bits, K and V separately. Record the relative RMS error per (layer, head, bits), and the attention-output error for a sampled set of query positions. | Small; off the critical path; sampled (every Nth block) | The actual damage per layer, head and bit width |
| **End-to-end check** | Replay a recorded session's tokens offline with the candidate settings, and compare logits against fp16 with the existing KLD harness (`metrics.uc`) | One extra forward pass per replay, offline | The final yes/no: mean KLD < 1e-3, the `tests/kld.uc` bound |

**Attention mass by position** (for the position-based layout in
`MEMORY.md`): per layer and head, the share of softmax mass landing on the
sink block, on the recent window (at a few candidate sizes: 1k, 2k, 4k, 8k),
and on the middle. `decode_attn` already holds the per-split softmax sums
(`m`, `l`) in its workspace, so for split boundaries aligned to those regions
this is a read of values the kernel computes anyway. Sampled every Nth decode
step.

**Decision output:** the mixed-precision layout from `MEMORY.md`:

- the recent-window size, from the attention mass by position;
- bit widths for the quantized middle, per layer and K/V, and per head where
  sensitivity is concentrated (e.g. "layer 11 keys need 6 bits; everything
  else is fine at 4"), from the shadow error.

The same settings serve the live cache and the stored prefix-cache entries
(`SESSIONS.md`).

**The telemetry proposes, the replay confirms.** Distributions and shadow
errors narrow down the candidates; only the KLD replay decides.

**Model vs session.** Outlier channels are mostly a property of the model.
Content shifts the scales (code vs prose, long contexts). So aggregate across
sessions for the defaults, and keep per-session aggregates to spot workloads
that need more bits.

## 2. MoE expert activation

Counted right after the router's top-k, on the device. Sizes are for an
example of 48 layers × 256 experts:

| Record | Size | Used for |
| --- | --- | --- |
| Selection counts per (layer, expert), split by prefill/decode | 48 × 256 × 4 B ≈ 48 KiB | **Placement.** Hot experts in VRAM on fast cards, cold ones in system RAM (the vendored `exl3_moe_cpu*` path). Usage frequency becomes a weight in the planner (`PARALLEL.md`): expected cost = P(used) × bytes / bandwidth. |
| Co-activation per layer (experts i and j in the same token's top-k) | 48 × 256² × 4 B ≈ 12 MiB | Which experts to keep together on one device |
| Transitions (expert i in layer l → expert j in layer l+1) | ≈ 12 MiB | **Prefetching:** start copying the likely next-layer experts from RAM while the current layer computes |
| Router entropy and weight mass per layer | Tiny | Anomalies; how decided the router is per layer |
| Rarely or never used experts | Derived from the counts | **Pruning candidates.** Dropping dead experts frees memory for a larger model. Riskier than it looks: validate on held-out sessions with the KLD harness before trusting it. |

Heatmaps: counts as layers × experts; co-activation and transitions as one
expert × expert matrix per layer.

## Infrastructure

- **Device:** fixed-size counter buffers per GPU, updated with atomics inside
  the existing kernels (`decode_attn`, the MoE routing kernel) or in a tiny
  separate kernel. Sizes are known at load from the model config, so they
  enter the load-time memory budget.
- **Host:** an asynchronous copy every N tokens or at the end of each
  request; aggregation in ucode; storage as compact binary or JSON lines under
  the user's directory, keyed by the same model identity as the prefix cache
  (an engine upgrade starts a new series).
- **Viewer:** a small HTML page rendering the histograms and heatmaps from the
  stored files. Build it once real data exists.

## Prior art

Surveyed 2026-10-07. The exllamav3 entries come from the local checkout (`~/devel/exllamav3` at `d3739fd`, the same commit as `vendor/exl3`); vLLM and llama.cpp entries from their documentation and search results, not their source. "Not found" means not found, not proven absent.

| Idea | vLLM | llama.cpp | exllamav3 |
| --- | --- | --- | --- |
| Telemetry choosing KV quantization (distributions, shadow error, replay) | Not found | Not found | Not found |
| Expert usage counts driving placement | Expert load balancing exists for large distributed deployments | Manual CPU offload of experts | **Yes:** dynamic expert placement with hit-count histograms and "swap sweeps" between generations (`modules/block_sparse_mlp_cpu.py`, `generator/generator.py`) |

**Study before building the MoE half:** exllamav3's dynamic expert placement
already counts hits on the device and reshuffles experts between VRAM and
system RAM between generations. Co-activation and transition matrices for
prefetching, and pruning candidates, were not found there.

## Build order

1. Counter-buffer infrastructure: allocation at load, the async snapshot,
   storage.
2. KV distributions and attention mass by position, folded into
   `decode_attn`, plus the per-channel heatmap.
3. Opt-in recording of token sequences, plus the offline KLD replay with
   candidate KV settings.
4. Shadow quantization error on a side stream.
5. MoE counters, co-activation and transitions, together with MoE support.
6. The HTML viewer.

## Open questions

- The sampling rate for shadow quantization: every block is cheap at decode
  speeds, but adds up during long prefills.
- Whether the MoE pruning candidates are worth acting on at all, or only
  inform placement.
