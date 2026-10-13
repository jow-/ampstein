# Ampstein — overview

> Details live in the per-topic documents linked at the end; hard-won
> knowledge is in [LESSONS.md](LESSONS.md).

Ampstein is an LLM inference engine written in
[ucode](https://github.com/jow-/ucode). It runs
[EXL3](https://github.com/turboderp-org/exllamav3)-quantized models on
NVIDIA GPUs and serves them through an OpenAI-compatible HTTP API. There is
no Python in the runtime: the native part is one ucode module (`torch.so`)
wrapping libtorch and vendored EXL3 CUDA kernels, and everything else —
model definitions, KV cache, sampling, scheduling, chat templates, the
server — is ucode source you can read and edit.

It is a hobby project and a "works for me" solution: built for one person's
rig and workloads, and only exercised on Ampere. See
[Status and caveats](#7-status-and-caveats).

---

## 1. Motivation

- **To learn how all of it works.** Quantized weight formats, KV caches,
  tensor parallelism, speculative decoding, schedulers: building the stack
  is the most reliable way to understand it.
- **To push the limits of ucode.** Is a libtorch binding plus a custom
  inference engine, driven from a small embedded scripting language,
  feasible at all? It was — and the exercise uncovered a number of ucode
  bugs along the way, which is a result in itself.
- **A clean answer to template injection.** Chat templates are Jinja, and
  the usual way of running them lets user-supplied text turn into special
  tokens (`<|im_start|>` typed by a user, or returned by a tool, becomes a
  real control token). Ampstein transpiles the checkpoint's own template
  and tracks provenance of every piece of text, so only text that came from
  the template itself can ever be tokenized as a special token.
- **A solution tailored to my rigs** (section 2): six 24 GiB Ampere cards
  on PCIe 3.0 x4 links with no NVLink, plus a two-3090 development box.
  That shapes the design: tensor-parallel groups of two, pipeline stages
  across the groups, all-reduce schemes picked for slow links, power caps
  that make decode energy-bound.
- **A codebase I can tinker on.** The point is to be able to try ideas
  quickly and keep what works. Example: the end-of-turn hook
  (`--eos-hook`) came out of watching an agent loop stall and was a
  server-and-engine change plus a ten-line policy script, in one session.
- **Fast, honest startups.** Memory is planned at load time and checked
  before the server accepts requests, so a configuration that doesn't fit
  fails in seconds and not after minutes of loading followed by an OOM
  (looking at you, vLLM). No allocations after load, KV and state slots
  reserved up front.
- **As fast as possible, with flexible parallelism.** Everything is open
  to experiments: layer split, tensor parallel, pipelines of tensor-parallel
  stages (for example 3 stages of 2 GPUs), speculative decoding, CUDA
  graphs, fused kernels.
- **Ownership.** No dependency on a framework's idea of how things should
  work. When something is slow or wrong, the whole path from HTTP request to
  kernel launch is readable.
- **RAM offload of inactive sessions.** An agent harness keeps several
  conversations alive at once (main loop, compaction, sub-agents, titles).
  Their KV and recurrent state moves to pinned host RAM — optionally
  quantized by age — and swaps back when the conversation continues,
  instead of being recomputed.

---

## 2. The test machines

Everything here is developed and measured on two machines, and their
differences drive many decisions.

| | **Two-GPU host** | **Six-GPU host** |
|---|---|---|
| GPUs | 2 × RTX 3090, 24 GiB | 5 × RTX A5000 + 1 × RTX A5500, 24 GiB each |
| Architecture | Ampere (sm_86) | Ampere (sm_86) |
| Link per GPU | PCIe 4.0 x8 | PCIe 3.0 x4 |
| P2P / NVLink | PCIe P2P, no NVLink | PCIe P2P, no NVLink (about 2.8 GB/s per pair, measured) |
| Host RAM | 187 GiB DDR5 | 125 GiB DDR4 (Xeon Gold 5118) |
| Power cap (nvidia-smi) | 220 W per card | 150 W per card |
| Role | development, tests, the small and mid-size models (Qwen3.8-27B, Llama-3.2-1B) | the large checkpoints: GLM-5.3-Flash, DeepSeek-V4-Flash, Qwen3.8-Flash-Next (100-117 GB of weights) |

What that means for the design:

- **the six-GPU host's links are the constraint.** At roughly 3 GB/s per P2P pair, every
  cross-device transfer is expensive: all-reduce schemes are chosen by rank
  count, stage hand-offs run on side streams, and the layer split / TP /
  stages-of-TP-pairs choice is a measured trade-off rather than a default.
  Tensor-parallel groups of two with pipeline stages across the groups
  (`--tp --stages 3`) exists because of this.
- **Power caps matter.** Under sustained tensor-core load the clocks drop
  well below boost; decode is closer to energy-bound than to bandwidth-bound,
  so profiles are taken under the real power regime (L56, L68).
- **The two-GPU host is the quick loop.** Two 3090s with wider links hold the
  mid-size models and the unit/oracle tests; large-model changes are verified
  on the six-GPU host, which is also the one that serves, so GPU tests there can
  only run when the server is stopped (L31, L77).
- **Host RAM sets the parking capacity.** The prompt-cache store's pinned RAM
  budget (`--cache-ram`) is sized against what these machines have left
  after the model and the page cache.

## 3. Architecture

### 3.1 Layers

```
 clients (pi, curl, any OpenAI client)
        │ HTTP / SSE
┌───────▼──────────────────────────────────────────────┐
│ core.uc + httpd.uc                                   │  server: endpoints, sampling params,
│                                                      │  stream parsing, reasoning budget,
│                                                      │  end-of-turn hook
├──────────────────────────────────────────────────────┤
│ engine.uc                                            │  jobs, scheduling, prefill/decode loop,
│                                                      │  speculative rounds, prompt-cache store
├──────────────────────────────────────────────────────┤
│ model families (ucode)                               │  llama · qwen35 · qwen4 · glm5 · dsv4
│ + chat templates (utpl, chat_*.uc)                   │  + *_tp (tensor-parallel variants)
├──────────────────────────────────────────────────────┤
│ torch.so  (C/C++, one extern "C" entry point)        │
│  ├─ libtorch (ATen ops, allocator, CUDA runtime)     │
│  ├─ vendored EXL3 kernels (gemm / gemv / dequant)    │
│  ├─ project CUDA kernels (attention, GDN, MoE, …)    │
│  ├─ safetensors reader, tokenizer, image preprocess  │
│  └─ P2P all-reduce, notifier, CUDA-graph helpers     │
└──────────────────────────────────────────────────────┘
```

The split is deliberate: **ucode owns control flow, the module owns
compute.** The generation loop, sampling policy, scheduling and server stay
editable without recompiling anything.

### 3.2 The native module

`torch.so` is a normal ucode module with a single `extern "C"` entry point;
C++ types, exceptions and templates never cross into the VM. It exposes
tensors as resources (with GC-tied lifetimes for views), ATen operations,
the EXL3 kernels, the project's own kernels, a tokenizer, a safetensors
reader, device and stream control, CUDA graph capture, peer-to-peer
all-reduce, and a completion notifier (an eventfd fed by CUDA host
callbacks) that lets the engine wait for the GPU from the event loop.

The EXL3 kernels are **vendored, not reimplemented**, and kept bit-exact
with upstream (MIT-licensed, license and commit note in `vendor/exl3`).
Only the host-side format knowledge — tensor layout, bitrate derivation,
kernel dispatch — is reimplemented.

torch.so is a hard requirement of the build — there is no enablement flag and
no CUDA-less mode — and no Python exists at runtime; Python appears only in
`tests/oracle/` as a test-time reference against exllamav3.

### 3.3 Models

| Family | File | Notes |
|---|---|---|
| Llama-3 | `llama.uc` | the original bring-up model; small, used for kernel and loop work |
| Qwen3.5 / 3.8 | `qwen35.uc`, `qwen35_tp.uc`, `qwen35_vision.uc` | hybrid GDN + attention, MTP head, vision tower |
| Qwen3.8-Flash-Next | `qwen4.uc`, `qwen4_tp.uc` | MoE, gated-residual hyper-connections, n-gram PLE; `docs/QWEN4.md` |
| GLM-5.3-Flash | `glm5.uc`, `glm5_tp.uc`, `glm5_vision.uc` | MLA, sparse indexer, MoE, MTP; `docs/GLM5.md` |
| DeepSeek-V4-Flash | `dsv4.uc`, `dsv4_tp.uc`, `dsv4_vision.uc` | compressed sparse attention, mHC, MoE, DSpark drafter; `docs/DSV4.md` |

The families are checked against exllamav3's own implementation by oracle
tests (`tests/oracle/`).
Mixed-bitrate checkpoints (per-tensor K derived from the trellis shape) are
handled as they come.

### 3.4 The engine

`engine.uc` runs one job at a time against one live model state, driven by
the event loop:

- **Prefill** is chunked and pipelined across devices; it hands control back
  between rounds so the server keeps serving I/O.
- **Decode** enqueues a step, hands the small results to pinned host buffers
  and waits on the completion notifier. Sampling (temperature, top-k, top-p,
  min-p, presence/frequency penalties) runs on the GPU; the host only walks
  the kept candidates.
- **CUDA graphs** capture decode and verify steps (multi-device,
  tensor-parallel included), with device-side positions so one graph serves
  every position.
- **Speculative decoding** uses the checkpoint's MTP head or the DSpark
  block drafter, with exact one-hot-draft acceptance so sampled output keeps
  the target distribution. Recurrent layers roll back through per-position
  states kept during verify.
- **Forced tokens** (`job.force`, `job.on_eos`) let the server inject text
  at a step boundary or at a sampled stop token: reasoning budgets and the
  end-of-turn hook both use them.
- **Prompt-cache store.** The live state is parked to a resident device
  slot or pinned host RAM when another conversation takes over, and swaps
  back when a request extends it; idle entries are stored in age tiers
  (fp16 recent, int8, int4). Time slicing lets a long generation yield to
  waiting sessions. See `docs/SESSIONS.md`.

### 3.5 Parallelism

Everything is placed by an explicit device plan. The options in use:

- **Layer split** across N devices (pipeline placement, one stream).
- **Tensor / expert parallel** over the devices.
- **Stages of tensor-parallel groups** (`--tp --stages N`): for six GPUs,
  three stages of two. Stages run a wavefront during prefill; decode is one
  graph per stage.
- Peer-to-peer all-reduce with a scheme chosen by rank count, because the
  links are slow (one-shot vs reduce-scatter + all-gather).

See `docs/PARALLEL.md` and the lessons L57, L67–L69, L92, L100.

### 3.6 Memory

Load plans memory and fails early. KV caches, GDN state buffers, MTP and
verify scratch, device slots for parked sessions and the prompt-cache staging
are allocated at load; request handling does not allocate on the device in
the compute path. Per-layer raw KV on the sparse-attention models is a ring,
so 262144-token contexts fit. See `docs/MEMORY.md` and L84, L91, L115d.

### 3.7 Chat templates and the tokenizer

```
chat_template.jinja ──jinja2utpl──▶ utpl source ──loadstring──▶ template function
messages / tools / images ──▶ tainted values ──▶ segment stream ──▶ token ids
```

The checkpoint's own `chat_template.jinja` is transpiled once at load into a
ucode template, so there is no hand-ported template to drift from the real
one (byte-exact against the Jinja oracle on the real Qwen3.8 and GLM-5.3
templates). Request-derived text enters as *tainted* values and is
tokenized without special-token atomization; text that comes from the
template itself is trusted and is. Details: `docs/TEMPLATE.md`.

The tokenizer is a native byte-level BPE over `tokenizer.json` (PCRE2
pre-tokenization, the pre-tokenizer `Sequence` of splits applied in order).

### 3.8 The server

`core.uc` (behind the `ampstein` entry) serves `/v1/chat/completions` (streaming and
not, tools, reasoning), `/v1/completions`, `/v1/models` and `/health`. The
architecture is selected from `config.json`, and any requested model name is
accepted. The output side has a streaming parser per family that separates
reasoning, content and tool calls (Qwen XML-ish calls, GLM, DeepSeek DSML).

---

## 4. Features

**Models and formats**
- EXL3 quantization (QTIP trellis), with mixed bitrates per tensor.
- Dense, hybrid (GDN + attention), MLA, sparse-attention and MoE families
  listed above; vision towers for the Qwen, GLM and DeepSeek checkpoints.

**Speed**
- Vendored EXL3 GEMM/GEMV kernels plus fused project kernels; split-KV
  tensor-core decode attention.
- CUDA-graph decode and verify, device-side positions.
- MTP and DSpark speculative decoding, greedy and sampled.
- Tensor-parallel and pipelined-TP placement with custom all-reduce.
- Async engine: the host never blocks on the GPU in the decode loop.

**Serving**
- OpenAI-compatible chat and completions, SSE streaming, tool calls,
  `reasoning_content`.
- Prompt-cache store: resident device slots and pinned host RAM, tiered
  quantization by age, prefix reuse with divergence logging.
- Time-sliced co-generation so side requests don't starve behind a long one.
- Hot reload: the KV state survives a server restart.
- Reasoning budget: soft nudge after X reasoning tokens, forced close after
  Y (`--reasoning-nudge`, `--reasoning-limit`, per request too).
- **End-of-turn hook** (`--eos-hook`): a script run when the model samples
  its end token, whose return value lets the turn end, redraws the position
  with the stop token penalized, or forces text. Built for agent harnesses
  whose loop stops on a text-only turn.
- **Tool-call hook** (`--call-hook`): a script run when the parser completes
  a tool call, with the conversation's earlier calls and results. A string
  verdict drops the call and answers it with that text as the call's tool
  result, inside the same response (the model replies with a corrected call or
  a change of plan; the client sees only that). A loop breaker for a model
  that resends a failing call: `examples/call_policy_loop.uc`; also usable as
  a call filter / blacklist (raise `--call-hook-max`).
- **Constrained tool calls** (`--call-guard`): the engine refuses, while
  sampling, any token that would take a call out of its tool's schema, and
  redraws the position from the same logits. Once `<function=NAME>` names a tool
  of the request the guard allows only its parameters (each once), keeps the
  bulk ones (`content`, `edits` ...) behind the short required ones (`path`
  first), and refuses to close the call while a required parameter is missing
  (`callguard.uc`, Qwen's XML call format). A refused position continues with the
  determined text (forced, e.g. `path>\n`) or is redrawn. While the
  conversation repeats its last `edit`/`write` (the same call twice, or four in
  a row cycling between two variants, or the same content written twice to
  nearly equal paths, or three writes of one script to numbered files) the
  tool's name is banned for the response. Earlier calls are rendered without the arguments their tool does
  not declare (prompt only, the client's history is untouched): a stray
  parameter the harness ignores otherwise stays in the context as an example.

**Safety and hygiene**
- Taint-tracked chat template rendering (no special-token injection from
  user, tool or image-caption text).
- Memory planned at load; no late device allocations.
- Table-driven wrong-argument tests on the bindings.

---

## 5. Performance without fast interconnects

The emphasis of the project: **useful, fast inference on consumer-class
cards that only talk over PCIe** — no NVLink, no NVSwitch, x4 links on the
big machine. The numbers below give an idea of what that buys. They are
rough: taken at different points during development, with different prompts,
power caps and settings, and not re-measured for this document. Treat them as
orders of magnitude, not as a benchmark suite.

### 5.1 Decode and prefill (single conversation)

| Model (EXL3) | Rig / placement | Plain decode | With speculative decoding | Prefill |
|---|---|---|---|---|
| Llama-3.2-1B, 4.0 bpw | 1 × 3090 | 350-380 tok/s (exllamav3: ~300) | — | — |
| Qwen3.8-27B, 5-6 bpw (hybrid) | two-GPU host, 2 × 3090, layer split | ~25-28 tok/s | 40-65 tok/s (MTP) | 1.0-1.3k tok/s |
| Qwen3.8-27B, 5-6 bpw | two-GPU host, 2 × 3090, TP | ~33 tok/s | 53-83 tok/s (MTP); 98-214 tok/s (DFlash 2, `docs/DFLASH.md`) | ~1.9k tok/s |
| Qwen3.8-27B, 6.0 bpw | six-GPU host, 6 GPUs, layer split | ~25 tok/s | — | — |
| Qwen3.8-27B, 6.0 bpw | six-GPU host, 6 GPUs, TP over 2 / 3 / 4 / 6 ranks | 37 / 46 / 54 / 57 tok/s | — | — |
| Qwen3.8-Flash-Next, 6 bpw (MoE, ~100 GB) | six-GPU host, 3 stages × TP-2 | ~43 tok/s | 52-77 tok/s (MTP) | 2.0-2.1k tok/s at 32k depth |
| GLM-5.3-Flash, 3 bpw (MoE, 117 GB) | six-GPU host, layer split | ~36 tok/s (exllamav3: ~37) | 41-46 tok/s (MTP) | 1.1-1.5k tok/s |
| GLM-5.3-Flash, 3 bpw | six-GPU host, 6 GPUs, TP | 61-64 tok/s (exllamav3: 51-55) | 75-96 tok/s (MTP) | ~0.5k (plain TP), ~1.0k (stages) |
| DeepSeek-V4-Flash, 3 bpw (MoE, 115 GB) | six-GPU host, 3 stages × TP-2 | ~55 tok/s | 50-65+ tok/s (DSpark, 3-4 tokens per round) | 1.4-1.5k tok/s |

Notes:

- Decode is per conversation, greedy or sampled; speculative figures depend
  heavily on the content (code and lists draft well, prose less so).
- DeepSeek-V4 at the full 262144 context: a 239k-token needle prompt
  prefilled at ~1.4k tok/s and decoded at 50-58 tok/s at that depth; Qwen
  27B on the two-GPU host at 210k depth reached ~27 tok/s with MTP.
- The cards are power-capped (220 W / 150 W); sustained tensor-core load
  runs well below boost clocks, so these are not peak-clock numbers.

### 5.2 What makes it work over PCIe

- **Few, small collectives.** TP all-reduces move a hidden state (about
  10 KiB per token). Inside a CUDA graph on the x4 links one takes ~14 us
  with 2 ranks and ~38 us with 6 (reduce-scatter + all-gather; one-shot
  would be 52 us). Cross-device event dependencies in graphs cost 25-90 us
  each, so the all-reduce is a single P2P kernel with direct peer stores.
- **TP pairs, pipeline between them.** Groups of two ranks keep each
  collective small and local; stages hand the residual stream on between
  groups, and prefill runs as a wavefront so all groups stay busy. On the
  27B this is the difference between ~25 and ~57 tok/s on the same six
  cards; on the MoE models, 36 vs ~63 tok/s.
- **Speculative decoding as latency hiding.** A verify of 3-4 rows costs
  little more than one row for dense models and ~2x for sparse MoE
  (the union of experts is read), but it emits 2-4 tokens, which is the
  one lever that keeps scaling when the GPUs are already idle between
  dependent steps.
- **Kernels at the bandwidth floor.** Decode is bound by the EXL3 GEMV
  kernels (~250-500 GB/s effective per card) and the power cap, not by the
  host: graphs capture the multi-device step, device-side positions keep
  one graph valid for every position, and the host thread only enqueues.
- **No stalls on the host.** Prefill launches are minimized per layer
  (stages of two ranks launch for two devices, not six), readbacks that
  force a device sync are removed, and the engine waits on a completion
  notifier instead of blocking.

## 6. Usage

### 6.1 Build

Requirements: a CUDA toolkit, json-c, PCRE2, and a ucode checkout with its
headers and interpreter. libtorch is downloaded by CMake (see the README).

```sh
cmake -B build -DUCODE_INCLUDE_DIR=/path/to/ucode
cmake --build build
cmake --install build --prefix /opt/ampstein
/opt/ampstein/bin/ampstein --version
```

### 6.2 Run the server

```sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # large multi-GPU loads
ampstein --model /data/models/Qwen3.8-27B-exl3-5.00bpw \
    --host 0.0.0.0 --port 8041 --ctx 65536 --mtp 3
```

Qwen3.8-27B on two GPUs with tensor parallelism and the DFlash 2 drafter
(see `docs/DFLASH.md`; needs the target and the `DFlash2` draft checkpoint):

```sh
ampstein --model /data/models/Qwen3.8-27B-exl3-5.00bpw --tp --ctx 262144 \
    --dflash /data/models/Qwen3.8-27B-DFlash2-EXL3-4.00bpw \
    --host 0.0.0.0 --port 8041
```

DeepSeek-V4 on six GPUs, as stages of tensor-parallel pairs, with the
end-of-turn hook for agent use:

```sh
ampstein --model /data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw \
    --ctx 262144 --mtp 3 --tp --stages 3 \
    --eos-hook examples/eos_policy_agentic.uc
```

Common flags (`ampstein --help` lists them all):

| Flag | Meaning |
|---|---|
| `--model DIR` | checkpoint directory (architecture from `config.json`) |
| `--ctx N` | maximum context in tokens |
| `--gpus N`, `--split N` | devices used, layer-split boundary (two GPUs) |
| `--tp`, `--stages N` | tensor parallel, number of pipeline stages |
| `--mtp W` | speculative window (MTP head or DSpark drafter; 0 = off); with `--dflash` the drafts per round (default 7) |
| `--dflash DIR` | DFlash 2 drafter checkpoint (Qwen3.5 / 3.8, needs `--tp`) |
| `--chunk N` | prefill chunk size (default 1024; 4096 for Qwen3.5 / 3.8 and Qwen-Flash-Next with `--tp`) |
| `--cache-ram GiB` | pinned host RAM for parked conversations |
| `--slots auto\|N` | resident device slots for parked conversations |
| `--slice-tokens N` | time slice length for co-generation |
| `--reasoning-nudge X`, `--reasoning-limit Y` | reasoning budget |
| `--eos-hook FILE`, `--eos-hook-max N` | end-of-turn policy script |
| `--call-hook FILE`, `--call-hook-max N` | tool-call policy script (default 3 verdicts per request) |
| `--no-vision`, `--vision-device N` | vision tower control |

### 6.3 Talk to it

```sh
curl -s localhost:8041/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Per-request knobs besides the OpenAI standard ones: `chat_template_kwargs`
(`enable_thinking`, `reasoning_effort`, as the checkpoint's template defines
them), `reasoning_max_tokens`, `reasoning_nudge_tokens`,
`reasoning_budget: {nudge, limit}`.

For agent harnesses such as pi, add the server as an `openai-completions`
provider and give the model a `thinkingLevelMap` and `chatTemplateKwargs`
matching the checkpoint's template (DeepSeek-V4: `enable_thinking` plus
`reasoning_effort` of low / high / max).

### 6.4 End-of-turn policy scripts

A policy is a plain script; the request state arrives as globals and the
return value is the verdict:

```js
// text, reasoning, calls, tools, mode, tokens, overrides
if (length(tools) && !length(calls) && match(text, /:[ \t\r\n]*$/))
	return 4 * (overrides + 1);     // number: redraw with the stop token lowered

return true;                        // true: let the turn end
                                    // "string": force that text and continue
```

See `examples/eos_policy_agentic.uc` for the shipped policy.

### 6.5 Tests and benchmarks

```sh
ctest --test-dir build                     # GPU tests are serialized; they skip without the models
ucode -L $PWD/build -L $PWD/ucode examples/bench.uc <model> 128 64
```

Model tests need the checkpoints and, for the larger ones, several GPUs
that are not busy serving; many skip (exit 125) when they can't run.

---

## 7. Status and caveats

- A hobby project and a "works for me" solution, tuned to one rig and one
  set of workloads. **Only Ampere (sm_86) has been tried**; no idea whether
  it works anywhere else.
- Single-user oriented: latency of one conversation first, sessions
  co-scheduled by parking and time slicing, not batched.
- The EXL3 kernels' numerics are exllamav3's; end-to-end differences against
  the oracle are small but real (fp16 noise, near-tie flips) and the tests
  are written around that (L37, L88, L107d).
- Several designs in `docs/` are proposals or partly implemented; each
  document states its own status.

## 8. Further reading

| Document | Topic |
|---|---|
| `docs/TEMPLATE.md` | chat template transpiler and taint design |
| `docs/MEMORY.md` | memory model: no late allocations, KV growth |
| `docs/PARALLEL.md` | multi-GPU placement over slow links |
| `docs/SESSIONS.md` | session parking, RAM store, slicing |
| `docs/QWEN4.md`, `docs/GLM5.md`, `docs/DSV4.md`, `docs/DFLASH.md` | model families, the DFlash drafter |
| `docs/TELEMETRY.md` | statistics collection (proposal) |
| `LESSONS.md`, `ENVIRONMENT.md`, `../AGENTS.md` | lessons, environment variables, operating guide |
