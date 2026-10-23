# RPC: pipeline stages across hosts

Status: plan, nothing implemented. First target: GLM-5.3 (`glm5_tp.uc`) on
jai (6 x A5x00) plus j7 (2 x RTX 3090) over 10 GbE.

## Shape

- Pipeline across hosts only. Tensor parallelism stays inside a host (PCIe /
  P2P): a network all-reduce per layer would cost more than the token itself
  (120+ round trips of 100-200 us against a ~17-37 ms token).
- **Coordinator** (jai): HTTP, tokenizer, engine, sampler, prefix cache, call
  guard, the draft/MTP block and the head. The engine talks only to it.
- **Worker** (j7): `ampstein --worker --connect HOST:PORT --model DIR`. It owns
  a contiguous layer range on its local GPUs (a TP group, P2P between the two
  3090s), with the KV / KDA state of those layers only.
- **Workers dial in.** The coordinator listens (`--rpc-listen ADDR:PORT`,
  `--rpc-workers N` to wait for N workers at start-up); a worker retries with
  back-off until it is accepted, and may rejoin at a request boundary after a
  drop. The worker needs no listening port.
- **Stage order: worker first.** j7 is stage 0 (it holds the embedding), jai
  holds the later stages, the head and the sampler. The ring closes with a token
  id (8 bytes) going to j7, and one activation hop per token goes to jai. A
  worker as the last stage would have to return logits or a sampled token each
  step.
- **Weights are pre-distributed**: every host has the same model directory. The
  hello carries a manifest (config hash, safetensors header hashes and sizes,
  ampstein build id, CUDA arch). The coordinator refuses a mismatch with a
  message naming the file. Both boxes are sm_86, so the kernels are the same and
  results should be bit-exact.
- **No TLS, no authentication.** That is the job of an outer VPN / overlay
  (WireGuard etc.). The listener binds to an explicit address, never
  `0.0.0.0` by default, and the docs say so.

## What crosses the wire

The residual stream is `hc_mult` (4) fp32 streams per token: ~40 KB per token
on Qwen4, roughly 100 KB on GLM (`handoff` in `glm5_tp.uc` moves ~200 MB per
2048-token chunk). Per hop that is 30-80 us of serialization at 10 GbE plus
150-300 us of latency (estimate, measure in phase 0). Prefill is capped at
~12k tok/s per hop, far above today's. The replicas inside a stage are
identical: send one copy, fan it out on the receiving host over P2P. fp16
activations would halve this but are not bit-exact; at most an opt-in later.

## Phases (each tested before the next)

0. **Measure, go / no-go.** A ucode ping-pong between pinned buffers (100 KB,
   20 MB) over the real link: p50 and p99, `TCP_NODELAY`, with and without the
   switch. Proceed if p99 is under ~2 ms.
1. **Native bindings** in `torch.so`: `send_tensor(fd, pinned, nbytes)` /
   `recv_tensor(fd, pinned, nbytes)`, non-blocking, over plain POSIX sockets;
   they raise on bad arguments or a closed peer (`tests/bindings_errors.uc`),
   no exceptions cross the module boundary. Device copies use the existing
   `copy_async` / `copy_h2d_async`.
2. **`rpc.uc` transport.** Frames `[type, seq, len]` + payload on one
   persistent connection: JSON for control, raw bytes for tensors. Coordinator
   side non-blocking on `uloop`; worker side a plain blocking loop. Keep-alive
   and a connection-loss path.
3. **Partial load** of `glm5_tp`: `load(dir, {layers: [a, b)})` builds only the
   layers (and embedding / head when owned), their caches and KDA state. It also
   makes the worker's start-up proportional to its share.
4. **Eager path over RPC.** An `rpc_model` proxy exposes the interface the
   engine already calls (`new_cache`, `forward_chunk`, `prefill_pipeline`,
   `graphs`, `mtp_*`, `states`, `cache_export` / `cache_import`, `new_slot`),
   each call becoming one command (`PREFILL`, `STEP`, `TRIM`, `RESET`, `SLOT`,
   `SPEC_RESTORE`). Test against the single-host model: logits bit-identical,
   identical greedy continuation (the oracle rule from AGENTS.md).
5. **Split the step graph.** Today a decode step is one multi-GPU CUDA graph
   over all stages. Each host captures its own segment graph with persistent
   input / output buffers: receive, async H2D, replay, D2H, send. Position-
   dependent buffers are rebuilt on the worker from the frame header (pos, ids,
   mm), never shipped.
6. **Pipelined prefill** across the hop: chunk n on the next stage while chunk
   n+1 runs on the previous one (the existing `prefill_pipeline` wavefront with
   one boundary on the network).
7. **Speculation, store, slots, hot reload.** MTP and `--lookup` drafts run on
   the coordinator; verify rows (up to 12) go through the hop like any step.
   Rollback (`kda` per-position state restore) is a command to each worker.
   Store, slots and hot reload work per worker (each saves its own layers); the
   coordinator checks that positions agree on rejoin.
8. Other model families (`qwen4_tp`, `dsv41_tp`, `gemma4_tp`) after GLM works.

## Planner

Extend the stage planner with per-host capacity weights (VRAM, bandwidth) and
a hop cost so the 3090s and the A5x00s are loaded in proportion. The explicit
`stage_end` boundaries already allow a manual split from the start. Print the
plan at load: layers per host, GiB per GPU, context per stage.

## Failure handling

- A worker drop fails the in-flight requests with an error; the coordinator
  keeps running and takes the worker back at the next request (`RESET`). No
  attempt to resume a half-finished step.
- Manifest mismatch, version skew and a missing worker at start-up are fatal,
  with a clear message.
- Telemetry (`TELEMETRY.md`): per-hop send / wait / receive time and p99 in the
  request log line.

## Out of scope

TLS / authentication / encryption, weight distribution, TP or expert-parallel
across hosts, fp16 activations (opt-in later).

## Open

- The exact hop payload for GLM (`hc_mult` and the replicated streams) is to be
  measured in phase 0, not assumed.
- Whether the MTP / lookup draft chain wants the head on the worker side for any
  layout; revisit after phase 5 timings.
