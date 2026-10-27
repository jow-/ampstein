// cuda.h -- ROCm stand-in, see cuda_runtime_api.h in this directory.
// Driver-API header; libtorch's headers that pull this in only use a
// handful of CUresult-family names, already covered by amp_cuda_to_hip.inc.
#pragma once

#include <hip/hip_runtime.h>
#include "amp_cuda_to_hip.inc"
