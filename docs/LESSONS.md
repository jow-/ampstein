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
