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
- **L138 — Selection after a scorer is launch-bound too.** With the indexer
  scoring in one kernel, the decode graphs still spent ~40 launches per
  sparse layer between the scores and the gather: `torch.topk` (14 kernels),
  the dead-pool mask, the member expansion, the tail pool, the pad mask.
  `torch.dsa_select` does them in one block per row: a two-level radix select
  of the kp-th key over the complete pools of the row's context only (the
  score array has 65k slots at 262144 tokens, an 87k context has 22k
  complete pools), ties to the lower pool, tiles in index order so the output does not
  depend on thread timing. A first version that read every slot with shared
  atomics took 91 us, no better than the composite (and not visible
  end to end); warp-aggregated histograms (scores cluster in a few bins),
  coalesced tiles and the visible-pools-only scan brought it to 15-47 us at
  3k-87k context: 38.1 -> 37.2 ms per round.

## Agent loops

- **L137 — A model can loop on a call it knows is wrong.** Qwen3.8-27B
  resent a failing `edit` ~190 times (five variants of one stale `oldText`,
  one error), writing "I'm stuck in a loop, stopping" in its reasoning before
  every one: with the identical failed calls in context, the next one is the
  likeliest continuation, whatever the thinking says. Asking in the next user
  turn works slowly. `--call-hook` consults a script when a call completes,
  with the conversation's calls and their results; a string drops the call
  and answers it with that text as the call's tool result (forced through
  `force()`, or the end-token path when the stop comes first), so the model
  reads the objection where it was about to repeat itself and replies with a
  corrected call. A note left inside the model's own turn was not enough: it
  read it as the end of its turn; and the Qwen parser drops text after a call,
  so the stream continues with a fresh parser. Matching on the *result* (same tool, same
  failure, in a row) catches variants that exact-argument matching misses; a
  replay of the 876 calls of that session found one drop before the loop
  began, a byte-identical resend.

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
- **L139 — Expert streaming is link-bound, and a pinned pointer is a miss.** The
  fused MoE kernels read experts through int64 pointer tables, and a table entry
  may point at `cudaHostAlloc` memory: the kernel then runs at link speed with no
  sync (jai: 25 ms for 8 experts at 3 GB/s, gen3 x4). Router-peek prefetch
  (router of layer L on the hidden state of L-1) recalls 67% of the experts with
  8 guesses, history heatmaps lose to LRU, and prefetch only pays above ~12 GB/s.
  `docs/EXPERT_STREAMING.md`.
- **L140 — Match failures by signature, and validate before the history.** The
  openwrt-minibuild session (Qwen3.8-27B, 1421 calls) sent 205 `write` calls
  without `path`, each carrying `content2` ... `content468` (the model
  enumerates parameter names and never reaches `path`). The loop breaker saw
  nothing: the validation error echoes the received arguments, a different
  script every time, so "same failure text" never held (replay: 0 drops, longest
  streak 1). The policy now compares the failure's first paragraph and drops a
  call missing a required parameter at once (`tool_defs` global, from the
  request's tools). Dropping on numbered copies alone would have killed 103
  working calls that carried the junk next to a valid `path`. The call is only
  seen after the parser completes it, so the tokens of a 468-parameter call are
  still spent.
- **L141 — Constrain the sampling, not the conversation.** Notes in the model's
  context did not stop Qwen3.8-27B from resending a `write` without `path`
  (plus `content2` ... `contentN`): with failing calls in its context the next
  one is the likeliest continuation. `--call-guard` follows the generated call
  and, once the tool is named, vetoes tokens outside its schema (unknown or
  repeated parameter, closing with a required parameter missing, `content`
  before `path`); the engine redraws the position from the same logits, like
  the stop-token redraw. Speculative windows are cut at the first refused
  draft, whose position is redrawn as the round's bonus.
- **L142 — A loop of successes, and a model that announces one tool and emits
  another.** After the `path` loop was fixed Qwen3.8-27B repeated one `edit` 69
  times, each answered "Successfully replaced 1 block(s)" (the edit flipped a
  trailing newline in and out), so no failure policy applied; its text said
  "I'll make a bash call" in 27 of 28 turns while the call was the same `edit`
  (the same mismatch existed before `--call-guard`: "via bash", then a
  path-less `write`). The copy of the previous call is the likeliest
  continuation whatever the text says, so the guard bans the repeated tool's
  name at the sampling level (`callguard.repeated`). Two bugs of the first
  guard worth remembering: tokens already emitted must not be passed again as
  the window prefix (the state advanced twice, `w`+`rite` became `wwrite`), and
  a guard must let go once it cannot be satisfied, instead of vetoing
  everything after an invalid name.
- **L143 — Tolerated junk is a seed.** The `write`/`path` loop of the
  openwrt-minibuild session began as one stray `content2: ""` at 15:53:44 that
  the model itself called "a typo ... the tool probably ignores it". The harness
  accepted the call, so it stayed in the context as an example, and every later
  call copied and extended it: 1 stray key, then ~9 per call (16:00), ~18
  (16:30), ~93 (17:00) -- 351 of 1723 calls, ~120k tokens of `contentN` -- until
  `path` was crowded out at 17:23 (0/182 writes failed before, ~85% after, at
  every context size). Each fix after that moved the symptom (path-less `write`,
  identical `edit` successes, numbered `/tmp/statN.sh`) because the cause is
  upstream: the model's own tolerated output is its strongest prior. So: refuse
  junk at the sampling (the guard), do not show it again (`clean_messages`), and
  compaction summaries must not carry "the tool is broken" narratives
  forward (here they did: "write tool BROKEN in this session"). A harness that
  rejects undeclared parameters would have shown the glitch at 15:53.
- **L144 — Identical content to a nearly equal path.** The model wrote a script
  and, in place of running it, wrote it again (`/tmp/pyval.sh` .. `pyval6.sh`,
  the same 842 bytes; its thinking said "let me run pyval.sh" each time): the
  previous call is copied at the function-name position whatever the text says.
  A cap of two consecutive writes stopped it but also blocked ~340 later writes
  and legitimate batches; the signal is the repeat itself: byte-identical
  content (of some size) to another file of the same directory, or short
  contents to a name that differs only by a digit or a letter or two (`_a`,
  `_aa`, `_a1`, `_v2`; the model's suffixes are unbounded: `_again`, `_final`) It fired 0 times on healthy work and 108 on
  the loop. The false premise behind the write-script-then-run workflow
  (`$p_mb_packed` read as a variable, misdiagnosed as a harness IFS/expansion
  quirk and kept in every compaction summary) is the upstream cause.
- **L145 — A rule on the command line misses what the command runs.** The
  long-sleep rejection never fired in the openwrt-minibuild session: the model
  wrote `sleep 330` / `420` into `/tmp/frr6poll.sh` .. `frr9poll.sh` (called
  "poll" scripts) and ran `bash /tmp/frr6poll.sh`, and the command line shows no
  sleep. The policy now also judges `bash -c '...'` and the newest `write` to
  each `*.sh` the command names (the conversation's own history, so it works for
  remote clients). Replay: 14 sleeping bash calls dropped, 5 of them through a
  script.
- **L146 — Block the second copy where the content is known.** The sampling ban
  needs two similar writes before it stops a third, so `/tmp/frr5poll.sh`
  followed 3 s later by `/tmp/frr5poll2.sh` (same 240-s sleep script) always got
  through. At the name position the content is not there yet; the call hook sees
  the completed call, so it drops a `write` whose content a write in the last 6
  calls put into another file of the same directory (>= 24 chars; the same path
  again is harmless and allowed). Replay: 0 drops in the 182 writes before the
  loop, 170 of 548 afterwards.
- **L147 — A compaction request is the one request the server may rewrite.** pi's
  summarizer serializes the *whole* history as text, thinking blocks included
  (tool results are cut to 2000 characters, thinking is not). The live context
  never carries earlier turns' thinking (the chat template drops it), so a
  262k-token session serializes to ~1.4M characters (65% thinking), the
  summarization request overflows the same 262144 window, and compaction fails
  on exactly the sessions that need it (j13 luci session, 2026-10-14/15). Its
  answer replaces the client's history, so unlike rolling compaction of a live
  conversation (no way to sync the prefix back) the server can answer from a
  reduced copy. `--prompt-hook` shows every rendered prompt to a script (with whether it
  fits); `examples/prompt_policy_compaction.uc` drops thinking, cuts tool
  results and leaves out the oldest entries (the real session: 1.44M -> 0.35M
  characters). Separate pi limitation: the threshold check only runs when a run
  ends, so a 230-call run goes from 80% to 100% unchecked.
- **L148 — PCRE2 validates the whole subject on every match.** Encoding was
  quadratic: `pcre2_match` on a long subject re-checks all of it as UTF-8 per
  call unless `PCRE2_NO_UTF_CHECK` is set, and stage 0 of the pre-tokenizer
  matches the whole text once per piece. Measured (Qwen3.8 tokenizer): 10 KB
  7 ms, 100 KB 534 ms, 1.4 MB ~100 s. Checking once per subject and JIT
  compiling the patterns: 100 KB 10 ms, 1.4 MB 126 ms (ids identical on the
  oracle battery and the 1.4 MB session text). The ucode side matters too: a
  per-token `push` loop costs more than the encode (400k ids: 203 ms vs 15 ms
  for `push(ids, ...part)`).
- **L149 — Hot reload rides on the store's entry, not on a second layout.** The
  dense `save_state` (Qwen: per-layer KV rows + GDN states) knows nothing of
  MLA latent rows, indexer planes, pools or KDA states, and a second
  serializer would have to track every change of the store. Saving is one
  `store_put()` with the budget and device slots borrowed (the entry lands in
  host RAM whatever `--cache-ram` / `--slots` say), flattened to tensors +
  a JSON tree; restoring is `store_restore(e, e.n)`. Caveat: the file goes to
  `/tmp`, which is often tmpfs (RAM): a 262k-token GLM state is several GB.
- **L150 — A per-expert bit width is a kernel problem, not a loader problem.** The SAGE packs mix K = 1..6 per
  matrix inside a layer (15 distinct (w1, w3, w2) triples per layer), and the vendored MoE kernels take one K per
  launch. Launching per class costs ~15 x the launches; padding to the widest K costs 4x the memory. What works:
  a `bits == 0` instance whose block switches on its expert's width and runs the unchanged static tile (decode),
  and the existing run-time-switch instance fed per-expert widths (prefill). Bit-identical to the static kernel
  per width (`tests/moe_mixed.uc`). The composite M = 1 EXL3 gemv is the *noisier* path (1.3e-2 against the cuBLAS
  reconstruction, the fused kernel 6e-3), so "fused differs from composite by 1e-2" is not a bug.
- **L151 — When there is no oracle, run the reference.** There is no exllamav3 for V4.1; the model code is DeepSeek's
  `inference/model.py`. Its TileLang kernels are replaced by 40 lines of plain torch (quantizers as identities,
  sinkhorn, sparse attention) and the unmodified `Attention`, `Block.hc_mixes`, `Engram` classes run on the CPU
  with the real weights fetched by HTTP range (a layer's attention is 140 MB). That caught the candidate-filler
  semantics (the reference attends -inf-score fillers when fewer candidates than top-k exist: only matters with a
  toy candidate budget) and fixed the numeric floor of the ratio-1 layers (rounding the index scores to fp16
  moves the *reference's own* output by 1e-2).
- **L152 — sm_86 has no fp8 conversion instruction.** `__nv_cvt_fp8x2_to_halfraw2` is ~10 ALU instructions per
  element there: the GEMV read 217 GB/s of 674. `(b & 0x7f) << 7 | (b & 0x80) << 8` read as a half is the e4m3
  value / 256, exactly (subnormals included); the 2^8 folds into the e8m0 block scale. 2-4x on every shape.
- **L153 — Host-side stalls at the end of each device segment are queue back-pressure, not syncs.** Issuing a
  prefill chunk-major (all 40 layers of chunk 0, then chunk 1) blocks the host on GPU k's full launch queue while
  GPU 0 idles. Issue wavefront-wise (one stage per chunk per round) and the same work runs 2x faster; 5x with the
  prefetch of L154 and the larger chunk.
- **L154 — Hashed table rows are known before they are needed.** The Engram rows depend on the token ids only, so
  a whole prompt can be hashed up front and `posix_fadvise(WILLNEED)`ed: the cold 3.5 s of a 4k prefill became
  0.25 s. The blocking gather then reads warm pages.
- **L155 — ucode details that cost an afternoon.** Functions are not visible before their definition (move
  helpers up; `block_decode` was lost to a careless range replace); `tensor.mul(3)` on int64 yields float32
  (build index arithmetic with `from_array`); `index()` needs a 2-D table; views of an `stfile` mapping outlive the
  file object (keep it referenced); `git stash -u` on a checkout holding a multi-GB `dist/` is not a thing to
  do; `install(DIRECTORY ... PATTERN "*.uc")` silently left the chat template out of the install, and
  `readfile` of a missing file returns null (an empty template).
- **L156 — A mirror that is only clamped keeps the last sequence.** The cross-GPU copies of a pool were brought
  up to date from `m.n` on, and a reset lowered `m.n` to the owner's *new* count: the rows below it (the previous
  conversation's) were never copied again. Every test builds a fresh cache, the engine's warm-up leaves the old
  rows behind, so the first long request after a start read garbage keys through the mirrors (wrong tool names,
  misread digits — greedy and short prompts got away with it, the first sampled long one did not).
  `set_pool_n` now invalidates the mirrors from the first rewritten entry on; `tests/dsv41_attn.uc` runs its
  sequence a second time on the reset caches.
- **L157 — A body reader returning `''` is "done", not "not yet".** `httpd.uc` used `if (!r) return;` to wait for the
  rest of a request body; a request without a body (every GET) completes with the empty string, which is falsy, so
  `GET /v1/models` and `/health` hung forever while the POSTs (non-empty bodies) worked. Compare with `null`.
- **L158 — Shared-memory bank conflicts hide behind "memory-bound".** The FP4 expert GEMV staged the activation row
  in shared memory and let lane `i` read `x[i * 32 + 2j]`: the same bank for all 32 lanes, 32-way serialized, 0.41 ms
  for 94 MB (230 GB/s). One pad float per 32 (stride 33) made it 0.20 ms (480 GB/s). A divergent `__constant__`
  table lookup serializes the same way (use a 16-entry shared table), but that was not the bottleneck here.
- **L159 — Cross-device copies block the host unless asked not to.** `tensor.cuda(dev)` between GPUs waits for the
  source stream; in the wavefront prefill the drafter's tap copies made the host wait for every chunk to leave the
  last stage (454 -> 680 tok/s with `.cuda(dev, true)`). Everything that rides along a pipelined pass must be
  non-blocking.
- **L160 — Speculative verify on an MoE costs ~rows.** A 5-row verify step is 67 ms against 37 ms for one row: the
  fused MoE kernel already decodes an expert once for all rows that picked it, but 5 rows x top-6 hardly overlap, so
  every row pays its experts (~0.12 ms per layer per row). With the drafter at ~10 ms a round of V drafts costs
  ~47 + 8 V ms; the engine picks V per round to maximize (1 + sum of cumulative confidences) / cost.
- **L161 — Device-side positions make multi-row decode batchable.** The single-token graph reads its position from
  `st.meta`; giving every row its own meta row (`new_dec_rows`) lets the rope, the ring write, the compressor
  (pair partner = the previous row, the half-built pair for row 0, rows that only open a pair write a spare slot),
  the indexer (`idx_score` already scores R rows from a device `pos0`) and the attention kernel run once for all
  rows: 83 -> 67 ms for 5 rows over the row-by-row loop (`AMP_DSV41_ROWLOOP=1` keeps that as the reference).

- **L162 — Count kernels before counting bytes, then profile.** A V4.1 decode step was ~4,600 kernels (115 per layer, mostly
  the composite rope / mHC / router / grouped-GEMV ops of ucode); `hc_mix2` (the mHC pass with the previous site's pre,
  vendor patch 5), `v41_rope`, `v41_route` and `fp8_gemv_grouped` took 36.8 -> 31.7 ms (5 rows: 67 -> 60.5). After that
  `nsys stats` (works without root; `ncu` needs counter permission) shows the GPU ~85% busy with three things: routed MoE
  10.5 ms (the coop kernels, 5x off the byte roofline: trellis decode is ALU-bound), FP8 GEMVs 11.3 ms (77% of
  bandwidth), the LM head 1.9 ms (at roofline).
- **L163 — rsync keeps mtimes, make trusts them.** A source file edited while a build was running kept its old mtime on
  the target, the object file looked newer, and the stale object linked in (`undefined symbol`, `not a function` from a
  binding that was in the source). Sync with `--no-times` so changed files get the arrival time.

## L164 — Engram prefetch on the launch path halved prefill speed

`stfile.prefetch` issued one `posix_fadvise(WILLNEED)` per row of the whole prompt (hundreds of thousands of
syscalls) before the first chunk: 7 s of host time, and the kernel's readahead queue then sat in front of the
first chunk's real gathers. Now the calls run on a detached thread and the prompt is prefetched three chunks
ahead (`AMP_DSV41_TP_AHEAD`). Prefill of 62k tokens: 700 -> 1416 tok/s; steady state is 0.89 s per 1536-token round
(1725 tok/s). Always look at the time before the first kernel, not only at the kernel timeline.

## L165 — Double-buffered prefill loses on PCIe x4 (DSV4.1 TP)

The two ranks of a stage reduce over PCIe x4: removing the all-reduce altogether takes a 1536-token round from 0.89 s
to 0.68 s, so it is 24% of the time. Splitting the chunk in two micro-chunks and reducing one on a side stream behind the
other's compute (`AMP_DSV41_TP_OV=1`) made it slower: 1.04 s with the spinning P2P kernel (it slows every concurrent
memory-bound kernel 10-20x: `hc_apply` 0.2 -> 3.9 ms), 0.96 s with copy-engine transfers (GPUs idle 40-50% waiting
for the DMA). Kept as an opt-in experiment; the lever left is fewer bytes over the link, not more overlap.
An early stage hand-off (126 MB of fp32 mHC state per rank, issued when the stage finishes instead of when the next
round starts) was slower too (0.88 vs 0.84 s): the link is the bottleneck, so every byte competes with the all-reduces.

## L166 — Qwen PLE rows: gather exact rows on a worker thread, chunks ahead

`ple_prepare` read the 39 GB n-gram table through a memory mapping on the launching thread: every cold row is a page
fault with kernel readahead (~100 KB around a 122-byte row), and when a chunk reached the PLE layer the host stopped
issuing for all stages. With the page cache evicted by another model, prefill fell from 1900 to 500-600 tok/s after the
cached system prompt. `ple_start` / `ple_finish` now use the pread pool of `stfile.gather_async` (as the DSV4.1 Engram),
started three chunks ahead (`AMP_QWEN4_PLE_AHEAD`) in the TP wavefront.

## L167 — Qwen3.8-Flash-Next TP decode: where the 20 ms went

A decode step is three stages in series, each stage two GPUs: a GPU is busy ~30% of the step by construction, and
inside a stage the all-reduce kernels are mostly *waiting* for the other rank (61 us average against 11 us when
balanced). The waits had three sources: the attention owner rank running the QSA chain alone (~320 us per attention
layer), the shared expert on one rank, and the expert-parallel imbalance. Measured (6 GPUs, ctx 262144, S=1 decode,
4k deep): 21.97 ms -> 17.3 ms from (1) the shared expert on a forked stream beside the router chain inside the
graphs (`glm5.set_par`, -1.3 ms), (2) the decode token selection through `idx_score` + `dsa_select` instead of the
ATen chain, which scanned the whole pool *capacity* (65536 blocks at ctx 262144) for every attention layer and step
(-3.2 ms, identical outputs; `tests/qwen4_qsa_sel.uc`), (3) the MTP draft chain on the device (greedy token feeds the
next embedding, one host read per round: greedy code 62.8 -> 73-77 tok/s). A decode graph sized for the context cap
costs the same at any depth: make fixed-shape graph kernels read the live length from a device value. A second
stream for `in_proj_z` beside the GDN conv/scan chain gained nothing (the projections are bandwidth bound). Verify
rows cost ~3 ms each (S=3 24.3 ms, S=5 ~30 ms), so the window stays 3. Left: the router chain (~35 us per layer,
fusable), the gated-residual mixes (~36 us per site, bandwidth/latency bound), attention head-parallelism, expert
imbalance (EXL3 weights cannot be sharded along K, so experts stay expert-parallel).

## L168 — Q8 KV for the QSA caches, and slots that ate the working set

`--cache-mode Q8` now applies to Qwen3.8-Flash-Next: K and V are int8 codes + a fp16 scale per 32 values
(`kv_q8_put` at write time, also with device-side positions in the decode graphs), and `qsa_attn` dequantizes while
it stages its tiles (bit-identical to the fp16 kernel on the dequantized caches, `tests/qsa_attn.uc`). NLL of real
text through the decode graphs at 14k context, 1500 scored positions: fp16 2.154 / 2.161, Q8 2.158 / 2.159 (the
run-to-run spread of fp16 is as large as the difference). On the 2.05 bpw quant on two 24 GB GPUs at ctx 131072 it
frees ~0.7 GB per GPU (23.6 -> 22.9). The saved/stored entries carry the codes, so they shrink as well.
Resident slots are carved out of the allocator's cached blocks: the warm-up's working set was consumed by the first
slot and the first long prompt then ran out of memory with 800 MiB "free". `alloc_slots` now returns the cache, keeps
the warm-up's cached bytes free on top of the margin, takes the slots and warms up again. A continuation of two
identical engines agrees on only 4 of 12 greedy tokens in about half of the runs (near-ties), so
`engine_store_qwen4` judges a restored state by not diverging at once.

## L169 — Speculative rounds: W rows verify W - 1 drafts; an adaptive window did not pay

A verify of S rows yields min(a + 1, S) tokens (a = accepted drafts): the last row only produces the bonus, so the
W-th draft of a window-W round was wasted work (a draft step is ~1.2 ms). The TP chain now makes W - 1 drafts and a
dummy that never matches (same tokens, same cache state). `gdn_ab_gates` (the a / b projections + gate math in one
launch, already used by Qwen3.5) replaces two cuBLAS calls (15 us each at 2-8 rows) per GDN layer: -0.6 ms per
verify step. Measured step times on the 2.05 bpw quant, 2 GPUs: S = 1 14.6 ms, then 17.5 / 19.0 / 20.7 / 23.1 /
25.5 ms for S = 2..6 (the m = 1 int8 GEMV path is ~2 ms cheaper than anything at m >= 2; per extra row MoE +0.6,
all-reduce waits +0.5, projections +0.5 ms).
The adaptive variant (draft chain with a confidence per token from the sharded head's softmax, calibrated online per
confidence bin, rows chosen by expected tokens per millisecond, `AMP_MTP_ADAPT=1`, verify graphs for every size)
works and its calibration is good (draft confidence >= 0.9 is accepted 98%), but over 12 prompts x 2 temperatures
(4800 tokens each): fixed window 3: 93.6 / 86.1 tok/s (T = 0 / 0.7), adaptive up to 6: 92.6 / 85.4, up to 7:
88.3 / 85.9; fixed 4: 88.3 / 79.7. More accepted drafts per round (2.6 vs 2.2 tokens) cost as much in verify rows
and draft steps as they return, so it is off by default. The lever is the cost of a row, not the window.

## L170 — Gemma 4: sliding rings, graph decode, and what the aux state is

- A sliding layer's ring (window + chunk rows) makes a prefix rewind exact only up to the slack. The snapshot / store state
  therefore carries the last `window` rows of every ring (`gemma4.window_aux`, the model's `aux_get` / `aux_set`) instead of
  trusting the ring after a long generation: the 1200-token test in `engine_store_gemma4` wraps the rings and still rewinds
  to the end-of-prompt snapshot, bit-for-bit against a fresh prefill.
- The chat template drops the empty thought channel of earlier turns, so the cache diverges at `<|turn>model\n` of every new
  turn: reuse is the snapshot at the prefix end, not the live state (cached_tokens was 0 until `aux_get` existed).
- Decode graphs: the position is a device tensor (rope tables, ring slot via `index_copy_`, window slots, causal mask of the
  global layers); the global layers' attention length is the graph's fixed shape, so one graph per power-of-two bucket,
  captured on first use (warm-up writes the rows of the position about to run; the replay rewrites them). 23.6 -> 61-72 tok/s;
  fusing the post-norm residual blocks (`norm32`) only bought 4%: a token is ~1700 small dependent kernels, ~10 ms of GPU time
  on two A5000, so `--tp` (81-84 tok/s) gains little, and tp prefill (1350-1700 tok/s) loses to the pipelined layer split.
- Wavefront prefill needs round-robin layer issue: issuing a whole stage's layers at once blocks the host on the launch queue
  (about 1000 launches per layer pass) and the other GPU runs dry (1750 -> 2580 tok/s at 14k).
- `models.sub_columns` accepts 128-aligned ranges now: Gemma's dense MLP is 2176 wide (2112 padded to 17 x 128), which does not
  split in 256s.
- Vision: patches are flattened (y, x, channel) (`image_prepare` option `hwc`), positions are x / y tables summed, the 2-D rope
  rotates the first half of the head with x and the second with y; soft tokens are not scaled by the embedding factor, and the
  rows of one image attend bidirectionally in the sliding layers only (the global layers stay causal), so prefill chunks may not
  split an image (`plan_bounds`).

## L171 — Host-resident MoE experts: stage prefill, zero-copy decode, and the heavy path is the trap

With part of the experts in pinned host memory, the fused MoE kernels can read them in place through the
pointer tables, which is right for decode (a miss costs its transfer, a hit nothing, bit-identical) and wrong for
prefill: every row tile of every launch re-reads the weights, so a 4096-row chunk moved ~3x the bytes of one pass
(129 ms/layer against 48 resident). The fix is a bulk copy of the layer's cold experts into a device double
buffer, one layer ahead on a side stream, and a second set of tables. Two traps while measuring it: (1) the
"heavy" experts (more rows than the fused tiers take) go through the ordinary linear, and a cold linear that
copies its trellis synchronously per call costs ~20 ms of host time per layer, and one that lacks the
reconstruct path for many rows computes slower than a resident one; both showed up only in a per-layer
timing against the all-resident baseline, not in tokens/s; (2) the all-VRAM model is not bit-reproducible
(the synchronous prefill MoE accumulates with atomics), so a whole-model "bit-exact" test is the wrong
criterion: check the kernel bit for bit and the model against its own run-to-run noise
(`tests/qwen4_host_experts.uc`). Details and numbers: `docs/EXPERT_STREAMING.md`.
