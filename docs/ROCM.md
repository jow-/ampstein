# The ROCm backend (AMD GPUs)

`torch.so` builds against ROCm/HIP as well as CUDA. One source tree, one module, the same function names for the ucode layer: the
backend is a build-time choice (`AMPSTEIN_GPU_BACKEND=cuda|rocm`), never a run-time one, and the ucode layer does not know which it
runs on. Developed on a Radeon RX 7900 XTX (gfx1100, 24 GB), ROCm 7.1.0.

```sh
cmake -B build-rocm -DAMPSTEIN_GPU_BACKEND=rocm -DUCODE_INCLUDE_DIR=/path/to/ucode     # auto-selected when only hipcc is found
cmake --build build-rocm -j
ucode -L build-rocm/lib -L src/ucode ./ampstein --model DIR --gpus 1
```

Configure downloads the libtorch ROCm wheel once into `~/.cache/ampstein` (`cmake/Libtorch.cmake`, `torch 2.16.0.dev20261006+rocm7.14`,
a nightly: there is no release wheel for ROCm 7.1, the newest is rocm7.0; nightlies roll off the CDN, so the manifest says how to
refresh it). `-DCMAKE_HIP_ARCHITECTURES=gfxNNNN` overrides the default `gfx1100`.

## Scope

What builds and runs on ROCm is what the GGUF-checkpoint path needs: the project-owned kernels (`amp_kernels.cu`, `amp_gguf.cu`,
`amp_dsv4.cu`, compiled as HIP) and every libtorch-only binding. The vendored EXL3 kernels (`vendor/exl3`, raw NVPTX inline asm,
`nvcuda::wmma`) are **CUDA only and not built** on ROCm, and neither is the code that reaches them. Their bindings are not registered
(calling one from ucode raises ucode's "not a function"), so **EXL3 checkpoints do not load on ROCm**. Also CUDA only, for the same
reason (PTX: `mma.sync`, `ldmatrix`, `cp.async`, `__match_any_sync`): the CUDA tensor-core decode attention (its ROCm counterpart is a WMMA kernel,
`attn_decode`'s `mma` flag selects it for head dim 256 and <= 32 rows per kv head, `docs/QWEN38.md`), the sparse-attention indexer (GLM DSA, Qwen QSA), the DeepSeek-V4.1 tensor-core attention, the
chunked gated delta rule (`gdn_chunk`; `gdn_scan` is ported), the MoE prefill tensor-core kernels, the P2P all-reduce (`--tp`),
the mHC / PLE / n-gram kernels, `skinny`, and the MoE experts' EXL3 GEMMs.

## What differs in the sources

| Where | What |
|---|---|
| `src/torch/amp_hip_compat.h` | the one place CUDA names meet HIP: runtime API renames, `__shfl*_sync` (the mask is dropped: every call site passes the full mask), `__dp4a` as `v_dot4_i32_iu8` (`__builtin_amdgcn_sudot4`: RDNA3 has no `sdot4`), `__vsub4`, and small self-contained bf16 / fp8-e4m3 types. `hip_bf16.h` / `hip_fp8.h` are deliberately **not** used: a distro `libamdhip64-dev` in `/usr/include/hip` shadows pieces of them with an older header one `#include` down |
| `src/torch/rocm_cuda_compat/` | stand-in `cuda_runtime_api.h`, `cusparse.h`, ... on the include path of `torch.so` only. libtorch's ROCm wheel ships `c10/cuda` and `ATen/cuda` headers unhipified (they still `#include <cuda_runtime_api.h>`); the names map through `amp_cuda_to_hip.inc`, generated from torch's own `cuda_to_hip_mappings.py` |
| `AMPSTEIN_ROCM` | compile definition guarding the CUDA-only parts above |
| `cp.async` in the gated delta rule scan | a plain synchronous copy, commit / wait no-ops (identical results, the next tile's loads are no longer overlapped) |

Wavefronts are 32 wide on gfx11, which is what the warp-shuffle reductions assume.

## Runtime quirks, handled in `uc_module_init`

- **Graph replay crashes after about 10000 launched kernel nodes** (`hipGraphLaunch` segfaults inside `libhsa-runtime64`; a 64-layer decode
  graph: the third replay): the HIP runtime's graph packet capture leaks. `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` is set unless the environment
  already says otherwise; with it, 60/60 replays of the same graph run. `AMP_NO_GRAPHS=1` runs without CUDA graphs at all.
- **`--mtp` above 4 is lowered to 4**: a verify step is `window + 1` query rows and the scalar decode attention takes at most 5 (CUDA uses the tensor-core kernel beyond that).
- **`expandable_segments:True` (the default on CUDA) fails** with `hipErrorInvalidValue` on the first allocation; it defaults to off.
- **`libamd_smi` double-frees a `std::map` in a static destructor at exit**, so every process ended with SIGABRT (status 134) after
  finishing its work. An `on_exit` handler flushes stdio and leaves with the real status before that destructor runs.

## Performance notes

RDNA3 has a fraction of the integer throughput per byte of the NVIDIA parts the kernels were written on, so the quantized decode GEMVs and
the decode attention are bound by vector-ALU issue, not by memory: count instructions (`-S --offload-device-only`, `rocprofv3
--kernel-trace --stats`) before tuning bytes. The numbers and what changed are in `docs/QWEN38.md` ("Performance work on ROCm").
`__vsub4` and `__shfl_xor_sync` in `amp_hip_compat.h` are the shared building blocks (SWAR byte subtract; DPP / `v_permlanex16`
butterfly, 4-byte types only, all lanes active).

WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, gfx11 wave32): A and B fragments are one row / column of 16 consecutive k per lane (lanes l and l + 16
hold the same data), C / D VGPR i of lane l is row 2 i + l / 16, column l % 16 -- verified against a CPU matmul; `amp_attn_dec_wmma_kernel` is the
reference for using it in this tree.

## Mixed ROCm / CUDA hosts

The RPC layer (`docs/RPC.md`) is vendor neutral: frames are JSON plus raw tensor bytes over TCP, tensors only ever touch the local GPUs of
their host, and P2P is intra-host. A ROCm host and a CUDA host can therefore serve one pipeline. Not done yet: the handshake's manifest
and `torch.device_props` report a CUDA compute capability (`device_props` now also carries `backend: "cuda" | "rocm"`); the manifest needs
that tag in its arch string (`cuda:sm_86`, `rocm:gfx1100`) so the coordinator names a mismatch correctly. The kernels differ numerically at rounding level between vendors (the split-model tests already
tolerate the run-to-run noise of the GLM path), so bit-exactness across vendors is not promised.

## Verified

`tests/gguf_dequant` (every ggml type bit-identical to the `gguf` package, int8 GEMV within 1 %), `gguf_reader`, `gdn_scan`,
`gdn_scan_tiled`, `gdn_fused`, `bindings_errors`, and Qwen3.8-27B end to end (`docs/QWEN38.md`). The CUDA build is unchanged in
structure but was **not rebuilt** on a CUDA host after these changes.
