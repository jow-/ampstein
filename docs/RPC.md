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
1. **Native bindings** (done): in `torch.so`, on the integer of
   `socket.fileno()`, over plain POSIX sockets with `MSG_DONTWAIT |
   MSG_NOSIGNAL`. `fd_write(fd, t, off, n)` / `fd_read(fd, t, off, n)` are one
   non-blocking attempt (bytes moved; 0 for EAGAIN; `fd_read` -1 for an orderly
   close), the building blocks for `uloop` handlers that keep their own
   progress. `fd_send_all(fd, t, timeout_ms?)` / `fd_recv_all(...)` finish the
   whole tensor, polling while the socket is full or empty, and raise on a
   timeout, a reset or a close before the end. `t` is any contiguous CPU tensor
   (pinned for the async device copies); `narrow()` views send slices. Errors
   raise (`tests/bindings_errors.uc`); `tests/fd_tensor.uc` covers the transfers,
   the EAGAIN path, timeouts, EOF and the device round trip. Device copies use
   the existing `copy_async` / `copy_h2d_async`. `rpc_pingpong.uc --native`
   times it (loopback: 100 KB RTT 55 us; the wire figures with the native path
   are still to be measured on jai).
2. **`rpc.uc` transport** (done). Frames are a 32-byte header (magic, seq, json
   length, tensor length), a JSON metadata object (`t` is the type) and the raw
   tensor bytes, on one persistent TCP connection (`TCP_NODELAY`, keep-alive,
   `TCP_USER_TIMEOUT` 15 s). `conn.send(meta, tensor?)` blocks until the frame is
   out (30 s without progress raises); `conn.recv(ms)` waits for a frame, and
   `conn.pump()` takes what has arrived without waiting, for a `uloop` handler.
   The tensor of a received frame is a view of a reusable pinned buffer of the
   connection (valid until the next tensor frame; an `alloc(meta, n)` hook lets
   the caller provide the destination, e.g. the stage's input buffer): allocating
   pinned memory per frame cost ~0.8 ms. `ping`/`pong` is answered inside `recv`;
   `conn.ping()` returns the round trip.
   The coordinator `rpc.listen({bind, port, manifest})` (an explicit bind address
   is required) takes dial-ins: `accept_one(ms)` (from a `uloop` handle on
   `fd()` for rejoins), `wait(n, seconds)` at start-up. The handshake checks the
   protocol version, the manifest id (`rpc.manifest(dir)`: sizes of all files, hash
   of every `.safetensors` header and small `.json`) and an optional `check`
   callback; a refused worker is told why. A worker with a known name replaces its
   old connection. The worker side is `rpc.worker({connect, name, manifest,
   serve})`: dial with back-off, hello (name, manifest, GPUs from
   `torch.device_props`: name, capability, memory), welcome, `serve(conn, welcome)`,
   and again after a drop; a refusal is fatal. Tested in `tests/rpc_transport.uc`
   (framing, byte-wise reassembly, bad and oversize frames, close, the handshake and
   all its refusals, rejoin, manifest) and with `examples/rpc_demo.uc` between two
   processes: on loopback a 64 B echo takes 112 us, 100 KB 138 us, 2 MB 562 us
   (the link figures are still to be measured on jai).
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

## Phase 0 result (2026-10-23): go

`examples/rpc_pingpong.uc` between j7 (`onboard10g`, 192.168.41.1) and jai
(`eno1np0`, 192.168.41.2): a direct 10 GbE cable, no switch, MTU 1500, plain
TCP with `TCP_NODELAY`, the ucode `socket` module with strings (the native
bindings of phase 1 will only be faster). The server echoes after receiving the
whole message, so a round trip is two serial transfers.

| payload | RTT p50 | RTT p99 | max |
|---|---|---|---|
| 64 B | 110 us | 138 us | 223 us |
| 40 KB (Qwen4-sized hop) | 275 us | 349 us | 381 us |
| 100 KB (GLM-sized hop) | 561 us | 840 us | 977 us |
| 2 MB | 4.6 ms | 8.5 ms | 10.8 ms |
| 20 MB (prefill chunk slice) | 51 ms | 92 ms | 93 ms |

Device side on j7 (pinned buffer, D2H or H2D plus synchronize): 100 KB 13-15 us,
2 MB 162 us, 20 MB 1.5 ms (13 GB/s).

Reading: one way is about half an RTT, so a 100 KB hop costs ~0.3 ms (p99
~0.45 ms) plus ~30 us of copies: under 1% of a 37 ms GLM token, ~2% of a 17 ms
Qwen4 token. The p99 is far below the 2 ms bar. Large payloads run at ~0.75 GB/s
per direction through ucode strings, so a 2048-token GLM chunk (~200 MB) is
~0.3 s on this path; fine for prefill, and where the native bindings and jumbo
frames (MTU 9000 on both ends) would help. Not measured: the link under load
from a running model (interrupt and CPU contention); repeat when phase 5 runs.

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
