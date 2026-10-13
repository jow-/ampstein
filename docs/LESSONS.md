# Lessons

Hard-won, non-obvious facts about building this engine. The IDs are stable and
are referenced from comments in the source (`L38` ...); entries are grouped by
the part of the project they belong to, in the order the project was built.
Add new lessons at the end of the matching section.

## The ucode module and the ucode language

- **L1 — Do not guess the module ABI.** The real API is `uc_function_list_t` +
  `uc_function_list_register()`; grep `include/ucode/*.h` and `lib/*.c` in the
  ucode tree, the code is the spec.
- **L4 — C++ modules need `extern "C"` around the ucode headers.** They carry no
  guards, so `uc_module_init` is name-mangled and the VM's `dlsym` fails with
  "provides no 'uc_module_entry' function".
- **L5 — The CMake target must not be called `torch`:** `find_package(Torch)`
  imports an INTERFACE target of that name. The target is `ucode_torch` with
  `OUTPUT_NAME torch`.
- **L7, L12, L13 — Resources.** `uc_fn_this(TYPE)` returns the data *slot*
  (`void **`), `uc_fn_thisval(TYPE)` the value. Plain and extended resources use
  different accessors and free paths — mixing kinds corrupts the heap, so every
  tensor is an extended resource (data area = `at::Tensor *`). Extra value slots
  are GC-marked: pin a dependency (the safetensors file behind a view) with
  `ucv_resource_value_set(res, 0, ucv_get(dep))`.
- **L8, L10 — Ownership and errors.** The VM consumes one reference of a
  handler's return value: `ucv_get()` anything you return that you do not own
  (`this`). `uc_vm_raise_exception()` returns void — raise, then `return NULL`.
- **L9 — Always give factory ops an explicit device** (`.device(at::kCPU)`),
  otherwise the first `device()`-touching call throws.
- **L20, L21 — Numbers and types.** `ucv_to_number()` allocates and returns NULL
  for NaN / non-numeric; read ints and doubles directly after a `ucv_type()`
  check. Call `ucv_resource_type()` only after `ucv_type(v) == UC_RESOURCE`.
- **L54 — Silent NULL returns are a bug class.** A binding that returns NULL
  without raising hands the script a null that corrupts state far away. The
  table-driven wrong-argument test (`tests/bindings_errors.uc`) is the cheap
  guard; write it before the NULL-check cleanup.
- **L14, L23, L24 — Language basics.** `printf("%J")` is JSON (`%j` prints
  literally). `/` on two integers is integer division. Arrays have no named
  properties (`arr["k"] = v` silently does nothing) and `for (k in arr)` yields
  values — use objects for dictionaries. No `in` operator: `index(arr, x) >= 0`.
- **L33 — `math.rand()` returns an integer,** `math.rand(a, b)` a double.
- **L36, L117, L122 — Imports.** A test named like a module self-imports (bare
  names search the importing file's directory first). A module's own static
  imports are resolved at compile time against its directory or `-L`, never
  against `REQUIRE_SEARCH_PATH`; only `require()` / `import()` use the latter.
  The engine layer therefore imports its siblings as `'./x.uc'`, which makes an
  install prefix relocatable with no `-L`.
- **L47, L64 — No function hoisting.** A helper must be declared before its
  callers (also for nested functions); a forward call dies at run time with
  "access to undeclared variable".
- **L55, L101, L102, L103 — Small traps.** `warn()` does not format (`printf`
  does). Strings have no methods — use `substr()`, `trim()`, `split()`, `join()`.
  Error diagnostics go to unbuffered stderr, so in `> out 2>&1` they appear at the
  *top*. A top-level rebinding after a closure captured the variable creates a new
  binding; share mutable state through an object field.
- **L81 — Arrays are references.** `let ids = prefix_ids; push(ids, ...)` mutates
  `prefix_ids`; take lengths before extending, or `slice()`.
- **L85a, L87a — `struct.buffer`.** `pull()` steals the storage, `slice()` copies,
  `length(n)` truncates from the end. `trim()` always allocates; check the edge
  bytes first on zero-copy paths.

## EXL3 kernels and the llama-family model

- **L2, L50, L53 — The tensor format.** `LinearEXL3` in exllamav3 is the spec:
  K = `trellis.shape[-1] / 16`, half-integer K needs the `mul1` codebook, and a
  checkpoint's "bits" is only the average — real files mix K = 4/5/6 per tensor.
  Derive K with a *float* divide (`/ 16.0`); the integer divide truncated 5.5 to 5.
- **L15, L16, L19 — `exl3_gemm`.** It computes `C = A @ dequant(B)` with the
  weight as a (k, n) matrix, no transpose; `suh` / `svh` must not be null (illegal
  memory access); `reconstruct()` yields the Hadamard basis while
  `reconstruct_had_slice()` yields the original basis. A fixture that mixes these
  up produces garbage that looks like kernel corruption.
- **L17, L18 — Autotune cache and oracle discipline.** The vendored kernels share
  exllamav3's on-disk autotune cache; stale entries cause flaky crashes — retry
  with `EXLLAMAV3_TUNE_CACHE` pointing at an empty directory before suspecting
  the code. When kernel output disagrees with a stored reference, recompute the
  reference in the same process first and check its algebra against upstream's
  own tests.
- **L26, L34 — Oracle API.** `Config.from_directory` + `Model.from_config` +
  `Tokenizer(config)`; `Tokenizer.encode(text, add_bos, ..., encode_special_tokens)`
  — the second argument is `add_bos`, atomization is the fourth.
- **L28, L29, L30 — Tokenizer.** The GPT-2 byte map reaches code point 323 (use a
  16-bit table); a decode buffer must be sized from the pieces, not `n * K`;
  special-token *strings* are plain text unless atomization is requested, and
  each plain run must be flushed as it closes. Run new native code under ASan.
- **L32 — RMSNorm variance must be fp32.** Accumulating `x^2` in fp16 goes
  subnormal for tiny embedding rows (BOS) and the logits came out 7.0 off while
  every ordinary token matched. Any reduction feeding an `eps` is fp32.
- **L35 — Render the real chat template** (jinja2) when debugging prompts; the
  llama-3 template injects a default system turn and omitting it produces
  repetition loops that look like an engine bug.
- **L37 — Compare logits by tolerance, not by argmax.** fp16 noise flips near-ties
  on long prompts: use max |diff| and an exact argmax at the last position only.
- **L38, L39 — CUDA graphs and the GC.** Every tensor a captured graph reads must
  stay reachable from long-lived state (put them in one object referenced by a
  returned closure), or the allocator reuses their memory and replays compute
  with garbage. Allocate on the device before capture, warm up first, and use
  value-dependent ops (`index_copy_`, `gather`, `where`) for dynamic behaviour.
- **L40 — Load by sequential `read()` into pinned buffers,** not by copying mmap
  views: the page-fault path ran at 0.4 GB/s against 1.5 GB/s. Evict the model
  file with `posix_fadvise(DONTNEED)` before timing.
- **L41, L42, L43 — Kernel and debugging traps.** `__float2half2_rn(a)` broadcasts
  one value, `__floats2half2_rn(a, b)` packs two. A launcher guard that silently
  returns on unsupported shapes shows up as garbage logits far away — make
  rejections loud. `tolist()` of an N-D tensor nests; flatten before printing.
- **L45 — Concatenating EXL3 weights along n is not exact:** `suh` differs per
  weight and one launch applies one `suh`. Use the sliced multi-gemm.
- **L49 — Link every TU a default-on dispatch path can reach.** The int8 GEMV
  selected itself for the first `m == 1` mul1 gemm and aborted on an undefined
  symbol that no test had exercised. After vendoring a TU, `nm -u` / load the
  module.
- **L31, L77 — Test environment.** GPU tests are environmentally flaky when
  another process holds the cards; run them serially and check `nvidia-smi` (and a
  3-line `arange` + synchronize script) before blaming the code.
- **L44, L46 — Profiling.** Isolated per-op timings are inflated 2-6x by clock
  ramp; budget from ablations in one process. The 1B decode step is ~2.9 ms, of
  which the vendored gemm kernels are ~2.0 ms (a shared floor with exllamav3).

## The Jinja transpiler

- **L98t — Scoping and `set`.** ucode template `let` is block-scoped; emit every
  `{% set %}` as a bare assignment and re-initialise the names in a per-render
  preamble to get Jinja's one-namespace set-leaking semantics. Raw text inside a
  macro goes to the macro's own buffer; whitespace control must be applied at
  text-emit time, not at tag-emit time.
- **L104, L107 — Tagged values.** Bare ucode operators on MIXED tag objects
  compare by identity: every ordering / arithmetic operator emits `raw(a) OP
  raw(b)`, and `tv.idx()` unwraps numeric indices. `tv.eq` is text equality only
  when both sides are strings.
- **L105, L120 — A `loadstring`'d program is a separate module instance and
  cannot import sources.** Per-render state and the `tv` / `utpl` namespaces are
  injected through the call scope (`__utpl_state`).
- **L106 — Macros are function expressions** compiled at their definition site;
  Jinja itself does not hoist macros, and function expressions give scope,
  single emission and recursion for free.
- **L109 — `struct.buffer` is the string builder of record** (O(1) appends,
  zero-copy `pull()`); it is not a script global — import it.

## Qwen3.5 / 3.8 hybrid (Gated DeltaNet)

- **L48 — ATen broadcasting aligns right.** Per-head vectors against
  `(Hv, dk, dv)` states need `(Hv, 1, 1)`; in-place ops also refuse to resize.
- **L56, L68 — Power cap.** Under a 180-220 W cap the SM clock falls from ~1.8 GHz
  to 0.7-1.1 GHz under sustained tensor-core load; check
  `nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active` before
  trusting a component budget. Decode is energy-bound.
- **L57, L58 — One host thread feeds all GPUs.** Enqueueing a whole stage blocks
  the host on that device's full queue while the others idle: interleave the
  enqueue order across devices (the wavefront in `prefill_pipeline`). A `.cuda()`
  of a pageable CPU tensor synchronises the stream — allocate on the device or
  stage through pinned memory.
- **L59 — Large-m gemm.** Beyond ~144 rows, reconstruct the weight once and use
  cuBLAS (≈2x). The two paths differ by fp16 noise, so compare pipelined against
  sequential prefill with identical chunk boundaries.
- **L60 — Causal attention for a chunk with history** needs the FA2 entry
  (`_scaled_dot_product_flash_attention`): the public `is_causal` aligns top-left,
  an explicit mask forces the fp32 math backend.
- **L61 — GeForce Ampere fp16 accumulate** runs at twice the fp32-accumulate rate
  (`CUBLAS_COMPUTE_16F`): +8.5% prefill, perplexity unchanged. Keep an opt-out.
- **L62, L63 — The delta-rule scan.** One block per (head, column slice) with the
  state in registers has a ~1.2 µs/token floor; the chunked WY form only pays
  with tiled tensor-core matmuls (naive loops lost 1.5x), and shared-memory
  blocks > 50 KiB run one per SM, so 96 blocks on 82 SMs take two waves.

## Server and engine

- **L72 — Loops on the per-token path cost ~1 µs per element.** Sampling over
  1000 candidates in ucode took 4.5 ms per call; keep per-candidate work in
  tensor ops and transfer only the kept prefix. A fp64 cumsum is needed for the
  top-p cut.
- **L74 — Per-request host work at 200k tokens** (prefix compare, re-tokenization)
  adds up to a second per agent turn: use `torch.common_prefix`, re-encode only
  the new tail, and never `substr()` through UTF-8.
- **L75 — `uloop.run(timeout)` is one poll round.** Wait with repeated short
  rounds or an indefinite `run()` ended by `uloop.end()`; `uloop.done()` tears the
  loop down; handles are level-triggered, so a callback must drain its fd.
- **L80 — Run a server smoke test after any engine change** when the GPUs are
  free; multi-device bugs hide behind tests that cannot run.
- **L86 — Async D2H and the notifier on a non-primary device.** The default stream
  handle is relative to the current device (guard it), pinned memory must be
  allocated portable, and the engine synchronises after every wake-up. Never
  `pkill -f <pattern>` from a command line that contains the pattern.
- **L118 — The entry is a lean script** (argv, `--help`, bootstrap) that
  `import()`s the engine module; a script cannot have top-level `export`.

## Speculative decoding and the async engine

- **L51, L73 — Recurrent state does not roll back.** The verify forward advances
  every GDN layer's state through all W tokens. Keep per-position states during
  the verify and restore with one batched copy; replay from the *pre-verify*
  length, never the post-rollback one.
- **L65 — int8 GEMV precision** is ~8e-3 rel-rms against ~2.7e-3 for the fp16
  kernel; plain decode and verify therefore do not share numerics, so
  "speculative == plain" holds only for short continuations.
- **L66 — Vocabulary-subset drafting does not pay:** BPE id order is a poor
  frequency proxy and EXL3 columns select in 256-column blocks.
- **L70 — Small-m products:** 4 rows cost ~1.4x one row; an int8 per-slice kernel
  for M = 3, 4 lost to the tensor-core gemm. Upstream's `m <= 2` gate is right.
- **L71 — FA2 varlen sizes its work by the cache capacity.** A 3-row verify at
  180k keys cost 13.6 ms/layer; the split-KV `torch.attn_decode` is 1.2-3.3 ms.
  Benchmark with the context users run.
- **L76 — A prefix count is not a mask sum:** matches can recur after a mismatch.
  Take the first mismatch as `min` over (index where mismatch, W where match).
- **L82a, L83a, L84a — The mma decode kernel.** `mma.sync` accumulates, so give
  each accumulator an explicit lifetime (a fragment declared in a loop is not
  re-zeroed); compute shared-memory offsets from a constexpr row-byte size, not
  another region's element count; kernel attributes are per device — key the
  one-shot setup by `cudaGetDevice()` and test graph capture on the second GPU.

## Tensor parallelism

- **L67 — Cross-device graph edges are slow** (25-90 µs each, 7.5 ms per token
  over 128 all-reduces). Use `torch.p2p_allreduce` (peer stores + a monotonic
  flag); eager work on the side device around a replay needs an explicit barrier.
- **L69 — All-reduce scheme by rank count.** One-shot sends (N-1)x the payload
  and loses from N = 4 on PCIe 3.0 x4; reduce-scatter + all-gather wins.
- **L87 (d, e) — The all-reduce kernel.** Every thread must fence its own remote
  stores before the block barrier (a single-thread fence gave a deterministic
  token flip); a kernel that spins on a peer deadlocks if another kernel is
  lazily module-loaded meanwhile — warm everything up first.

## Vision

- **L78 — Identity versus content of image rows.** Token history compares ids, so
  every image row carries a marker id (`MARK_BASE` + 28 bits of the image's
  SHA-256) and only the embedding gather sees the real pad id. Positions after an
  image differ from KV slots by a constant — graphs take a separate `rpos`.
- **L79 — `int64_tensor.mul(16)` promotes to float** and then fails as an index.
  Scale integer index arrays on the host before `from_array`.

## GLM-5.3-Flash, the prompt-cache store and long contexts

- **L84 — Loading 117 GB over six 24 GiB cards.** The weights cache keeps both the
  raw upload and a cast copy (`ws.release`), expandable segments avoid 2 GiB of
  fragmentation, and placement balances bytes, not layer counts.
- **L85 — Judge a deep model by its continuation.** End-to-end logits rel-rms was
  8% while every layer stayed at 0.2-0.3%; a few outlier tokens carry the error.
  The decisive check is the greedy continuation against the oracle.
- **L87 (a, c) — Decode latency is not throughput:** back-to-back graph steps
  pipeline across devices, so benchmark with each token depending on the
  previous. Concurrent branches in a captured graph are free speed, but never run
  two EXL3 products at once (they share per-device lock buffers).
- **L88 — Near-tie flips are not regressions.** Print the logit margin at the
  first divergence and teacher-force the oracle. cuBLAS runs `(1,K)x(K,N)` fp32 as
  a GEMV but `(2..8,K)` as a slow GEMM — `torch.skinny` replaces it.
- **L90 — Layer-split decode and verify are kernel-bound:** graph-capturing the
  eager MLA layers changed nothing, and a 3-row verify pays for the union of the
  rows' experts.
- **L91 — Long-context allocation.** No arena: libtorch's caching allocator, the
  full KV at load, and a warm-up that runs the worst-case shapes so an unfitting
  context fails at startup. Allocations that scale with the prompt are bugs.
- **L92, L93 — Pipeline of TP stages.** `--tp --stages N` trades 30-40% prefill
  for ~27% decode at long context; the stage all-reduce is only 9-13% of hybrid
  prefill, the rest is half-sized kernels and replicated indexer work.
- **L94 — The DSA indexer dominated prefill, and the host thread was the
  bottleneck.** A fused scoring kernel and 256-row slabs cut launches 58k → 13k
  per chunk (+30%). Count launches before blaming kernels: one ATen op costs
  ~25 µs of ucode + dispatch. Prefill is not deterministic (atomics, router
  near-ties) — compare NLL on real text, and benchmark with ≥ 32k tokens.
- **L96 — Reasoning budget.** A soft nudge and a forced close are injected through
  the engine's `force()` hook at step boundaries; the continuation re-enters the
  ordinary prefill path as a live-reuse prefill of the tail.
- **L97 — The host-RAM store.** One live state plus one snapshot loses every
  other conversation; parking states in pinned RAM (tiered fp16 / int8 / int4 KV
  by distance from the end) restores a 100k-token prefix in ~1 s against ~100 s
  of prefill. Log *why* a request did not reuse the live state.

## Decode and prefill kernel latency (Qwen3.8-27B, 2 x 3090)

- **L123 — A split-KV kernel must size its splits by the live length.**
  `attn_decode` launched `4 * SMs / Hk` splits for every context, so a
  300-token context ran 164 near-empty blocks and its combine kernel walked
  164 workspace rows twice, serially (56 us at one row, 144 us at four, 16
  layers: ~8% of a verify round). Both kernels derive the effective split
  count from the device length (`amp_ad_nse`, at least 128 keys per split);
  the workspace keeps the launch's layout.
- **L124 — Count what a "small" op costs at the verify width.** The fp16
  `in_proj_a/b` matmuls (24 columns) are a 3 us cuBLAS GEMV at one row but a
  28 us cutlass WMMA kernel at four, two per GDN layer: ~9% of the verify
  forward. `gdn_ab_gates` does both products and the gate math in one
  launch. Profile at the speculative window, not at one row.
- **L125 — The decode rmsnorm was latency-bound, not bandwidth-bound.** 2-byte
  loads, 256 threads, two dependent passes: 12 us for 5120 elements, 130 per
  round. 8-byte chunks with a block sized to the row: ~3 us.
- **L126 — The GDN scan is shared-memory-instruction bound.** One thread per
  state column needs its own LDS of every k / q element: ~1500 LDS wavefronts
  per token per head, whatever the block shape. Register-blocking four
  columns per thread with 128-bit loads, rows split over 8 lanes, and
  `cp.async` double-buffered 16-token tiles took it from 1.26 to 0.47 us per
  token (+9% prefill end to end). Splitting only the rows, or only
  prefetching, gained 1.1-1.7x. Run micro-benchmarks for ~1 s so the clocks
  ramp (L44).
- **L127 — Near-tie flips in `qwen35_mtp`.** Changing the scan's summation
  order moved the logit margin of one token from +0.047 to -0.008 (one fp16
  ULP at 15.7) while the kernel still matched the fp32 reference to 1e-8. The
  test now accepts a divergence only when the plain model's own logits tie
  within 0.02 (L37, L88).
- **L128 — Decode on this host is power-capped, not bandwidth-bound.** Both
  cards sit at the 220 W software cap (SM 1.4-1.5 GHz of 2.1, DRAM ~54%
  utilised): the EXL3 trellis decode is ALU work, so the clock is the limit.
  `nvidia-smi -q -d PERFORMANCE` / `clocks_throttle_reasons.sw_power_cap`
  confirms it before any kernel work is blamed.
- **L129 — TP prefill idled the GPUs in the copy-path all-reduce.** Peer
  `copy_` + three fp32 elementwise kernels per all-reduce kept both 3090s at
  ~56% busy (the stalls were cross-device stream events, with neither card
  computing). A second P2P communicator sized to the chunk (direct peer
  stores, as GLM already does) gave +13% at chunk 2048 and, with 4096-row
  chunks, 1.9-2.0k tok/s against 1.4k at the start. A double-buffered
  two-half prefill with the copies on side streams measured no better than
  the plain copy path (as for GLM, L94): fix the mechanism before overlapping.
- **L130 — A bf16 drafter overflows fp16 on its massive channels.** DFlash 2's
  attention output carries ~5e3 in one channel from layer 0 (`h` reached 4.8e4
  after one layer and inf after two); the same weights are fine in bf16. Keep
  the residual stream in fp32 and pre-scale the producing projections' `svh`
  (exact, the output scale is linear). The EXL3 linears themselves matched
  the bf16 checkpoint to 7% rel-rms (4 bpw): check each primitive against the
  unquantized weights before suspecting the architecture.
- **L131 — A 7-row verify costs what a 3-row one does.** With the gemms on
  16-row MMA tiles the verify forward stayed ~26 ms from 3 to 7 rows, so the
  window is nearly free and the drafter quality decides: MTP at 3 drafts
  accepts 2.2-2.9 tokens per round, DFlash 2 at 7 drafts 3.2-6.9. The only
  new cost is attention for > 5 rows (the FA2 path sizes its work by the
  cache capacity, L71; `attn_decode` takes up to 64 rows since L132).
- **L132 — Long-context decode was the attention's combine and its second
  pass.** At 82k keys a DFlash round cost 47 ms against 32 ms at 1k: the
  7-row verify ran two `attn_decode` launches (42 rows did not fit the
  32-row tile), each reading the whole cache, and each combine kernel ran
  one block per kv head walking 164 splits serially (170 us, ~40% of the
  attention). A block per (row, kv head) for the combine, a 48-row tile
  (the e-block aliased onto the fp32 score block to fit 99 KiB), a softmax
  of 8 lanes per row instead of a warp per row, Q fragments held in
  registers and one split per SM (a partial second wave of 1-block-per-SM
  kernels cost up to 40%) took the 7-row attention from 877 to 343 us per
  layer at 82k; decode at 82k went from 102 / 55 to 125 / 69 tok/s
  (code / prose). The 42-row kernel is now bound by the half-rate fp32-
  accumulate mma of GeForce Ampere (~330 us of tensor work at 200k keys
  against a 520 us load floor), not by bandwidth.
- **L133 — A leading-dimension slice is already contiguous.**
  `models.rows` cut row-split EXL3 weights with `narrow(0, ...)
  .contiguous()`, which returns a view: the device that loaded a layer kept
  the whole tensor alive for its own shard (and `.cuda()` copied the other
  rank's half). Every TP model carried a second copy of half its row-split
  weights: 1.2 GB per GPU on Qwen3.8-27B. Copy explicitly when a slice must
  own its memory, and check `mem_info` allocated bytes after a load against
  the checkpoint size. Found while sharding the DFlash drafter, which
  rank 0 had run alone (rank 1 idled ~4.7 ms per round in the next
  all-reduce).

## GLM-5.3 serving (6 x A5000, TP 3 x 2)

- **L134 — A captured graph scores the cache capacity, not the context.**
  The decode graphs have fixed shapes, so the DSA indexer's composite
  (`q_idx @ pools.t()`, scale, clamp, head-weighted sum: four ATen ops)
  scored all 65,537 pools of a 262144-slot cache for every query and layer,
  ~196 us per sparse layer whatever the context. `torch.idx_score` with a
  device-side position (an int64 tensor) skips the tiles no query can see
  and does the products, relu and head sum in one kernel: ~5% of a decode
  round (40.1 -> 38.1 ms). Look for capacity-sized work in every graph-
  captured op.
- **L135 — Short prefills are launch-bound; replay the step graphs.** A
  2-token tail (a re-sent prompt, a small tool result) took ~165 ms of eager
  launches at ~12% GPU utilisation before the first token; replaying the
  decode / verify graphs takes ~45 ms. Each replayed row costs a decode row
  (~9-12 ms: the MoE reads grow with the rows), so the eager path wins from
  ~16 tokens (`AMP_TAIL_GRAPH_MAX`, default 12). Prefill times in the log
  count enqueueing: the replays' device time lands in the first decode step,
  so compare time to first token, not the prefill column.
- **L136 — Stages trade decode for memory and prefill.** On this host 2 x 3
  decodes ~15% faster than 3 x 2 (34 vs 40 ms rounds) but prefills at 870
  instead of 1440 tok/s and does not fit a 262144-token context; TP-6 does
  not fit 131072. A 3 x 2 round is ~3 x 11-12 ms of stages in series plus
  ~5 ms of MTP draft chain and ~2 ms of host work, each GPU busy ~30% of it.
  The MTP window does not matter (2 / 3 / 4 drafts: 61-62.5 tok/s): every
  extra verify row costs 5-8 ms of MoE expert union.

## Qwen3.8-Flash-Next

- **L98 — Bring-up.** A partial-model mode (`layers: N`) checks blocks 0-3 in
  seconds instead of a minute per fix. A shared object links with undefined
  symbols and binds lazily: after vendoring a TU, load the module. `tensor.index`
  is `at::embedding` (2-D table), and the module has no relu / masked_fill /
  permute (`clamp`, `where`, `transpose`).
- **L99 — TP decode is capped by what every rank replicates** (gated-residual
  tables, the router chain, two barriers per layer) and by arrival spread at the
  barriers, not by communication time. Prefill under TP is host-bound.
- **L100 — Stages beat micro-chunks:** fewer launches per layer, not overlap,
  cured the host-bound TP prefill (716 → ~2000 tok/s). Time the second repetition
  of any prefill measurement.

## DeepSeek-V4

- **L104d — Strided views into EXL3 kernels.** `x.narrow(1, g, 1).reshape(...)`
  returns a view with the parent's row stride and the kernels read contiguous
  rows: always `.contiguous()` first.
- **L105d, L109d — Massive activations.** First-token rows reach 1e5, so the fp16
  collapse and EXL3 projections overflow to inf / NaN and the NaN spreads through
  the compressor pools. The reference saturates; so does `hc_collapse`, and tap
  rows are scaled into range before the projection.
- **L106d — An HF pre-tokenizer `Sequence` applies every Split, in order.**
- **L107d — Per-block errors on real text are not a pass criterion** (atomic
  accumulation, router near-ties): judge logits / argmax / NLL and isolated
  single-layer tests fed with the oracle's own inputs.
- **L108d, L112d, L114d — Prefill.** Measure host time per phase before and after
  a change. A count readback per MoE layer stalled the host (2.5 of 3.9 s); once
  it was gone the limiters were launch count, then stage hand-off copies and
  pipeline bubbles. A slice benchmark does not predict the full model.
- **L110d — Speculative decoding on a sparse MoE** verifies the union of the rows'
  experts: a 5-row verify costs ~2x a step, so the useful window is 4-5 drafts.
- **L111d — Engine integration.** A leftover `graphs: false` skipped graph
  construction *and* the drafter's build with a null deref deep in the first
  request; a partial-model engine test exercises the plumbing in 90 s.
- **L113d — Vision.** Image rows route on `gate.bias_vl`, the image proper must be
  one prefill chunk, and the block layout is tied to the compressor group size.
- **L115d — 262144-token context.** Raw key rows live in a ring (attention reads
  128 of them; snapshots copy the window), and the indexer's per-head score
  matrix is never materialised (`idx_score` handles 64 heads in one kernel).
- **L116 — A premature end of turn is a model decision.** Intervene at the stop
  token (`job.on_eos`): redraw with the stop ids lowered, or force text. A stop
  accepted as a speculative draft must truncate the round and become the bonus.
