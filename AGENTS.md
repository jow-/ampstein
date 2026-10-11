# AGENTS.md — working on Ampstein

Orientation for anyone (human or agent) changing this repository. Read this
file first; the rest is reachable from it.

## What this is

Ampstein runs [EXL3](https://github.com/turboderp-org/exllamav3)-quantized LLMs
from [ucode](https://github.com/jow-/ucode) scripts, with no Python at run time.
The native part is one ucode module (`torch.so`) that wraps libtorch and vendored
EXL3 CUDA kernels; everything else — model loading, architectures, KV cache,
sampling and generation — is ucode in `src/ucode/ampstein/`.

ucode owns control flow, the module owns compute: long-horizon features stay
editable without recompiling the module. The EXL3 kernels are vendored, not
reimplemented (QTIP trellis decoding has bit-exactness requirements).

## Layout

| Path | What |
|---|---|
| `src/torch/` | the `torch.so` module: tensors, safetensors, native bindings |
| `src/ucode/ampstein/` | the ucode layer (`ampstein.<name>` modules) |
| `vendor/exl3/` | exllamav3 kernels (MIT), see `vendor/exl3/VERSION`, `PATCHES.md` |
| `cmake/` | libtorch download / cache, kernel source lists |
| `tests/` | ucode tests run by CTest; `tests/oracle/` generates fixtures (Python, test-only) |
| `examples/` | small runnable scripts and benchmarks |
| `docs/` | design notes and `LESSONS.md` |

## Build, test, run

```sh
cmake -B build -DUCODE_INCLUDE_DIR=/path/to/ucode     # ucode checkout with include/ucode
cmake --build build -j
ctest --test-dir build --output-on-failure             # model tests skip (125) without models
```

- libtorch and the NVIDIA runtime wheels are downloaded once into
  `${XDG_CACHE_HOME:-~/.cache}/ampstein` (`-DAMPSTEIN_TORCH_DIST_DIR=` to use an
  existing site-packages instead). `nvcc` and the GPU architecture are detected.
- Tests find models through `AMP_MODEL_*` environment variables (defaults under
  `/data/models`) and fixtures in `tests/fixtures/` (gitignored; generate them with
  the scripts in `tests/oracle/`, see `tests/README.md`).
- GPU tests run serially (`RUN_SERIAL`); check `nvidia-smi` before blaming code
  for a flaky failure (`docs/LESSONS.md` L31).
- Compile-check a module with `ucode -L src/ucode -cmodule -o /dev/null file.uc`.

## Invariants

1. **No Python at run time.** Python appears only under `tests/oracle/`, as a
   test-time oracle.
2. **`extern "C"` at the module boundary.** No C++ exceptions, types or templates
   cross into the ucode VM; catch everything inside the module functions and
   convert it to a ucode exception.
3. **Vendored kernels stay bit-exact.** Do not "clean up" kernel math. A change in
   `vendor/exl3/` needs a numerics test against the exllamav3 oracle and an entry
   in `PATCHES.md`.
4. **libtorch, CUDA and ucode are hard requirements** — no enablement flag, no
   CUDA-less build.
5. **MIT only** in `src/` and `vendor/`; vendored code keeps its licence file and
   a version note.

## Conventions

- C11 for plain C, C++17 for module glue, CUDA for kernels; ucode's own C style
  (tabs, `type` on its own line, K&R-ish braces). The ucode layer targets the
  ucode dialect (ECMAScript-like, no async).
- Every binding raises on bad arguments (`tests/bindings_errors.uc`); no silent
  null returns (L54).
- A model is correct when it matches the exllamav3 oracle (logits within fp16
  noise, identical greedy continuation) — add the oracle script and test with the
  model.
- Commits: imperative subject, one logical change.

## Where to look

- Hard-won, non-obvious facts: `docs/LESSONS.md` (IDs such as `L38` are referenced
  from source comments).
- Design notes: `docs/` (one file per topic or model family).
- ucode module ABI: the ucode checkout's `include/ucode/*.h` and `lib/*.c` are the
  spec (`lib/math.c` is a compact reference module).
- EXL3 format: exllamav3's `exllamav3/modules/quant/exl3.py` (`LinearEXL3`) and
  `exllamav3_ext/bindings.cpp` (the native API inventory).
