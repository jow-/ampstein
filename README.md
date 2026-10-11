# Ampstein

> **Project Ampstein — Frankenstein's inference on Ampere GPUs**

LLM inference for [ucode](https://github.com/ucode-lang/ucode): run
[EXL3](https://github.com/turboderp-org/exllamav3)-quantized models on NVIDIA
GPUs from ucode scripts, with no Python in the loop.

```
┌────────────────────────────────────────────────────┐
│  ucode scripts                                     │
│  chat · benchmarks · tests                         │
├────────────────────────────────────────────────────┤
│  ucode engine layer (src/ucode/ampstein/*.uc)      │
│  models · KV cache · sampling · generation         │
├────────────────────────────────────────────────────┤
│  torch.so  (ucode module, C++ with extern "C" ABI) │
│  ├─ libtorch (ATen ops, allocator, CUDA runtime)   │
│  ├─ safetensors reader (C, mmap, zero-copy views)  │
│  ├─ vendored EXL3 kernels (gemm / gemv / dequant)  │
│  └─ native BPE tokenizer (HF tokenizer.json)       │
├────────────────────────────────────────────────────┤
│  CUDA driver / cuBLAS / cuDNN (via libtorch)       │
└────────────────────────────────────────────────────┘
```

## Status

- Loads EXL3 checkpoints (`exl3_gemm` / `gemv` from the vendored kernels) and
  runs llama-family models: prefill, KV-cache decode, CUDA-graph decode,
  sampling — numerically matching exllamav3.
- Native BPE tokenizer for HF `tokenizer.json` files (PCRE2 pre-tokenizer).
- Jinja → ucode transpiler for chat templates, with taint tracking so user text
  can never tokenize as a special token.

## Building and installing

```sh
cmake -B build -DUCODE_INCLUDE_DIR=/path/to/ucode   # a ucode checkout (include/ucode)
cmake --build build
```

libtorch, CUDA and ucode are hard requirements: a CUDA toolkit ≥ 12.4, the ucode
headers and the `ucode` interpreter (for the tests). `nvcc` is found via
`$CUDACXX`, `$CUDA_HOME` or `/usr/local/cuda*` (or `-DCMAKE_CUDA_COMPILER=`); the
GPU architecture defaults to `native` (`-DCMAKE_CUDA_ARCHITECTURES=86` on a
build host without a GPU).

**libtorch is downloaded for you.** `cmake/Libtorch.cmake` fetches the pinned
`torch` wheel plus the NVIDIA runtime wheels it links (cudnn, cublas, nccl, …,
~3.5 GB, hashes verified) and caches them *outside* the build tree in
`${XDG_CACHE_HOME:-~/.cache}/ampstein` (`-DAMPSTEIN_CACHE_DIR=` /
`$AMPSTEIN_CACHE_DIR`), so a second build tree never downloads again. To use an
existing distribution (e.g. a venv's `site-packages`):
`-DAMPSTEIN_TORCH_DIST_DIR=<dir containing torch/ and nvidia/>`.

## Testing

```sh
ctest --test-dir build --output-on-failure
```

Tests that need a model or an oracle fixture skip (exit code 125) when it is
absent; see [tests/README.md](tests/README.md) for generating fixtures and
pointing the tests at models (`AMP_MODEL_*`).

## Documentation

- [AGENTS.md](AGENTS.md) — layout, build, invariants, conventions.
- [docs/LESSONS.md](docs/LESSONS.md) — hard-won, non-obvious facts.

## AI assistance

This project was built with liberal use of AI: Claude Sonnet 5.5 (proprietary)
to bootstrap the code base, and locally hosted models — GLM 5.3 Flash and
Qwen 3.8 27B — for iteration, refinement and testing.

## License

MIT. The vendored EXL3 kernels are MIT-licensed, from
[exllamav3](https://github.com/turboderp-org/exllamav3) — see
`vendor/exl3/LICENSE.exllamav3`.
