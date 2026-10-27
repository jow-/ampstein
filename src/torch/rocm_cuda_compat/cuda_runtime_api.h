// cuda_runtime_api.h -- ROCm stand-in.
//
// libtorch's ROCm wheel ships its public c10/cuda and ATen/cuda headers
// unmodified from the CUDA build (hipify only rewrites the translation
// units that go into the wheel's .so, not the installed header tree), so
// they still say `#include <cuda_runtime_api.h>` and expect a real CUDA
// toolkit to provide it. This directory (-I'd ahead of everything else
// when AMPSTEIN_GPU_BACKEND=rocm, see CMakeLists.txt) stands in for that:
// HIP's own runtime header plus a name-for-name #define per CUDA Runtime
// API symbol those headers actually call (amp_cuda_to_hip.inc).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "amp_cuda_to_hip.inc"
