# Ampstein

> **Project Ampstein — Frankenstein's inference on Ampere GPUs**

LLM inference for [ucode](https://github.com/ucode-lang/ucode): run
[EXL3](https://github.com/turboderp-org/exllamav3)-quantized models on NVIDIA
GPUs from ucode scripts, with no Python in the loop.

```
┌────────────────────────────────────────────────────┐
│  ucode scripts                                     │
│  OpenAI-compatible HTTP server · chat · benchmarks │
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

## Models

| Family | Module | Notes |
|---|---|---|
| Llama-3 style dense | `llama.uc` | EXL3 weights, CUDA-graph decode, library use (no server path) |
| Qwen3.5 / Qwen3.8 hybrid (Gated DeltaNet) | `qwen35.uc` | layer split over several GPUs, CUDA-graph decode |

Engine features:

- Prompt-cache reuse: live state, plus a snapshot at the end of the previous
  message (recurrent layers cannot rewind).
- MTP speculative decoding with the checkpoint's own draft head, an event-driven
  decode loop and exact speculative sampling.

## Building and installing

```sh
cmake -B build -DUCODE_INCLUDE_DIR=/path/to/ucode   # a ucode checkout (include/ucode)
cmake --build build
cmake --install build --prefix /opt/ampstein         # builds first if needed
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

The install prefix is relocatable (loader and `RUNPATH` are `$ORIGIN`-relative):

```
<prefix>/bin/ampstein                       server entry
<prefix>/lib/ampstein/torch.so              native module
<prefix>/lib/ampstein/torch-dist-<ver>/     libtorch distribution (torch/lib, nvidia/*/lib)
<prefix>/share/ampstein/*.uc                ucode engine layer
```

For development, `-DAMPSTEIN_INSTALL_TORCH_DIST=OFF` installs a symlink to the
cache instead of copying the multi-GB distribution.

## Serving

```sh
/opt/ampstein/bin/ampstein --model /path/to/Qwen3.8-27B-exl3-5.00bpw --ctx 65536
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (stream and
non-stream, tools, reasoning content) and `POST /v1/completions`. `ampstein --help`
lists the options.

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
