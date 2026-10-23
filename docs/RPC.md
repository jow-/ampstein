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
   protocol version, the manifest (`rpc.manifest(dir)`: `id` = name, size and safetensors header length of every file as text, computable on every host; `strong` = sha256 of those plus every header and small json, compared only when both hosts have `digest.so`, which jai lacks) and an optional `check`
   callback; a refused worker is told why. A worker with a known name replaces its
   old connection. The worker side is `rpc.worker({connect, name, manifest,
   serve})`: dial with back-off, hello (name, manifest, GPUs from
   `torch.device_props`: name, capability, memory), welcome, `serve(conn, welcome)`,
   and again after a drop; a refusal is fatal. Tested in `tests/rpc_transport.uc`
   (framing, byte-wise reassembly, bad and oversize frames, close, the handshake and
   all its refusals, rejoin, manifest) and with `examples/rpc_demo.uc` between two
   processes: on loopback a 64 B echo takes 112 us, 100 KB 138 us, 2 MB 562 us
   (the link figures are still to be measured on jai).
3. **Partial load** of `glm5_tp` (done, verified on GPUs). `load(dir,
   {devices, stages: NS, own: [first, end), stage_end})`: `stages` is the global
   pipeline, `own` the stages this process builds, `devices` only its local GPUs
   (Gs per owned stage). Layers, caches (`new_cache`), communicators, the embedding
   (stage 0), the head, final norm and MTP block (last stage) exist only on the
   owner; the rest are `null` / `{remote: true}` stubs, so a worker's start-up and
   memory are proportional to its share (the weights are mmap'ed and uploaded by
   key, so nothing else is read). Every host plans the whole pipeline the same
   way; the coordinator passes its `stage_of` as `stage_end` to the workers so a
   different GPU mix cannot make them disagree. `model.own_layers` is the owned
   layer range. The per-layer API (`embed`, `layer`, `handoff`, `head`) is what the
   RPC runs on each side. `tests/glm5_split.uc` loads the whole model, then two
   halves (stage 0 / the rest), hands the residual streams over through the host as the
   RPC will, and compares the logits. It skips (125) without the model or ~18 GiB free per
   GPU. Result on jai (6 x A5000/A5500, 3 stages x 2): the split model differs from the
   whole one by rel-rms 0.042, while two forwards of the whole model differ from each other by
   0.033 (the GLM eager path is not bit-deterministic, so the test requires the split to
   stay within 3x that noise), and the greedy token is the same. A stage 0 of 4 loaded
   alone on j7 (2 x RTX 3090) in 58 s with 14.8 / 13.8 GiB, layers 0-12 only, and the two
   ranks' output replicas are bit-identical, which the one-replica hop relies on.

4. **Eager path over RPC** (done, verified across the two hosts). The worker owns the
   FIRST stages, so the embedding and the first layers run there and one hop carries the
   streams to the coordinator, which owns the rest and the head (a single hop per
   token; the sampled token needs no way back). `rpc_stage.uc`: the worker's command loop
   (`new_cache`, `free`, `forward {cache, pos, n}` + ids -> `h` + one replica of the
   float32 streams, `stats`, `bye`; a failing command answers `error` and the
   connection stays up) and the coordinator's `client` (begin/end halves so that several
   forwards can be in flight). `glm5_tp.forward_chunk` starts from the worker's streams
   when stage 0 is remote (`opts.remote`), copying the replica to every rank of its first
   local stage; `graphs` and `prefill_pipeline` refuse remote stages until phases 5 and 6.
   `ampstein --worker --connect HOST:PORT --model DIR` dials in, receives its assignment
   (`stages`, `own`, `stage_end`, `max_seq`) in the welcome, loads only those stages,
   reports `ready` and serves; the coordinator side so far is `examples/rpc_glm5.uc`
   (listen, plan with `plan_only` for the worker's GPU count, hand over `stage_end`,
   load its stages with the client, prefill and decode greedily with `forward_chunk`).
   `tests/rpc_stage.uc` runs the protocol against a CPU mock. The plan for 4 stages x 2
   ranks is 13.5 / 14.6 / 14.6 / 13.5 GiB per GPU, stages starting at layers 0, 13, 24, 35
   (45 layers).
   **Result (2026-10-23)**, worker on j7 (stage 0, layers 0-12, loaded in 8 s from the page
   cache), coordinator on jai (stages 1-3, 6 GPUs, loaded in 85 s), over the direct 10 GbE
   link: "What is the capital of France?" gives "The capital of France is Paris." Prefill of
   19 tokens 1.85 s, eager decode 83.2 ms/token. The same prompt on jai alone (3 stages,
   `--single`): prefill 1.72 s, decode 88.5 ms/token, and the same first 29 token ids (the run
   then diverges at token 30, within the model's own run-to-run noise). So the network hop
   costs nothing measurable in eager mode: the 4-stage pipeline is not slower than the
   3-stage one on the same prompt. Decode is still eager (every layer launched from ucode);
   the CUDA-graph decode of the single host is phase 5.
5. **Split the step graph** (done, verified). Each process captures the graph of its own stages:
   `glm5_tp.graphs` now handles a partial model. The worker's graph runs from token ids
   (persistent device buffer) through the embedding and its layers and ends with the streams
   of its last stage; the coordinator's graph starts from persistent per-rank input buffers
   (`hin`) that the host fills from the worker's reply before each replay, and ends with the
   head as before. The position broadcast among the ranks and the dense / sparse variant
   choice (`pos + S > index_topk`) depend only on `pos`, so both sides always replay the
   same variant. Protocol: `graphs {cache, sizes}` makes the worker capture (the coordinator
   asks it first: the warm-up clobbers the cache, so `model.reset_cache(cache)` follows and
   resets the remote cache too, `reset {cache}`), `forward {..., graph: true}` replays.
   Decode over the hop is: ids to the worker, one replay there, the streams (160 KB for one
   token) back, H2D into the input buffers, one replay here. Speculative windows (S > 1
   with state rollback) and the draft block are phase 7.
   **Result (2026-10-23)**, j7 stage 0 + jai stages 1-3, over the direct 10 GbE link:
   decode 18.6 ms/token (short prompt) and 17.9 ms/token (3011-token prompt, which takes the
   sparse variant), against 18.2 and 17.4 ms/token for the same run on jai alone (3 stages):
   the hop costs ~0.4-0.5 ms per token. The 32 generated token ids are identical to the
   single-host run for both prompts. Eager decode over RPC was 83 ms/token.
6. **Pipelined prefill** (done, verified): `prefill_pipeline` accepts remote first stages.
   The worker runs a chunk through all of its layers in one `forward` command; the
   coordinator keeps two requests in flight (`forward_begin`), so the worker computes chunk
   n + 1 while the local stages work on chunk n, and each reply is the entry of the first
   local stage of the wavefront (copied to every rank of that stage). The reply of a 512-token
   chunk is 33.5 MB (~30 ms on the link, hidden behind the compute). **Result**: the 3011-token
   prompt prefills in 2.36 s (1276 tok/s) over j7 + jai, against 4.78 s one chunk at a time over
   RPC (`forward_chunk`) and 2.61 s on jai alone with 3 stages (the pipeline has a fourth stage
   here); the generated tokens are identical. `examples/rpc_glm5.uc --pipeline`.
7. **The server, in two cuts.**
   **7a (done, verified): `ampstein --rpc-listen ADDR:PORT` serves a remote-stage GLM.**
   `serve()` listens, waits for the worker (`rpc_coord.attach`: plans with `plan_only` for the
   worker's GPU count, hands over the stage assignment, waits for `ready`), loads its own stages
   with `remote: client`, and runs the unchanged engine on top: pipelined prefill, split decode
   graphs, chat templates, tools, reasoning. `--gpus` / `--stages` are this host's share. Off in
   this mode: MTP, `--lookup`, `--cache-ram` and `--slots` (the engine's host parking and slots are
   switched off by `model.remote`).
   The recurrent (KDA) state of the worker's layers is kept honest by a **mirror**: the coordinator
   holds tensors of the same shapes for each remote KDA layer, and the engine's four state helpers
   (`snap_bufs`, `save_states`, `load_states`, `zero_states`) call `model.remote_states('pull' |
   'push' | 'zero', cache, layer)` around their copies, which moves the layer's bytes over
   `state_get` / `state_put` / `state_zero` (a layer is two ranks of `rec_state` float32 and
   `conv_state` float16, ~2 MB). So prefix reuse via snapshots, the turn checkpoint and resets
   work unchanged; attention KV needs no sync (the worker writes it at the `pos` of each command,
   so a rewind by length is implicit).
   **Result (2026-10-23)**, j7 + jai over the 10 GbE link, the real server: a multi-turn chat reuses
   its prefix (`cached 22 via snapshot`, a 44-token snapshot on the next turn) and answers
   correctly ("PELICAN-77" recalled; "77-NACILEP" spelled backwards after a snapshot restore, also
   after a detour to an unrelated conversation); a needle in 8499 tokens is found, with prefill at
   1623 tok/s and decode at 51.7 tok/s as the server reports them.
   **7b (to do):** speculative windows (MTP, `--lookup`): per-position KDA state rollback and the
   draft block's cache over the hop; host parking and slots of states that live on the worker
   (`cache_export` / `cache_import` per worker); hot reload; a worker that drops and rejoins
   while the server runs (today the in-flight request fails with the connection error and the
   coordinator must be restarted); more than one worker.
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
