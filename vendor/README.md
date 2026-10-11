# Vendored code

| Directory | What | Licence |
|---|---|---|
| `exl3/` | EXL3 quantized-GEMM and model kernels from [exllamav3](https://github.com/turboderp-org/exllamav3) (`VERSION`, `PATCHES.md`) | MIT (`exl3/LICENSE.exllamav3`) |
| `stb/` | `stb_image.h` (image decoding for the vision input) | public domain / MIT |

Rules: vendored kernel math is never "cleaned up"; a change needs a numerics test
against the exllamav3 oracle and an entry in `exl3/PATCHES.md`. Which translation
units are compiled is decided in `cmake/exl3.cmake`.
