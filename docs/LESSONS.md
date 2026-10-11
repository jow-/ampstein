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
