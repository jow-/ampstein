// amp_hip_compat.h — header/type portability shim for the project-owned
// kernels (amp_kernels.cu, amp_dsv4.cu, amp_gguf.cu) so the same source
// compiles under nvcc (CUDA) and under amdclang++ (HIP/ROCm).
//
// vendor/exl3/ is NOT covered by this: it stays CUDA-only (raw NVPTX inline
// asm, nvcuda::wmma), not built at all on a rocm configuration — see the
// ROCm backend note in AGENTS.md / docs/GGUF.md.
#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// bf16 / fp8-e4m3: NOT hip_bf16.h / hip_fp8.h. Both re-include their own
// sub-headers (amd_detail/amd_hip_*.h) by full search-path lookup rather
// than a self-relative quoted include, so a stale distro-packaged
// /usr/include/hip (if one is installed alongside this ROCm tree, as
// libamdhip64-dev does) transitively shadows pieces of them with an
// older, incompatible header one layer down — chasing every shadowed
// sub-header with forwarding stubs turned into a sinkhole. The two uses
// in this project are narrow (store a dequantized GGUF weight as bf16;
// fake-quantize an activation through fp8 e4m3 and back), so small
// self-contained conversions sidestep the dependency entirely.
//
// __float2bfloat16: round-to-nearest-even truncation of the top 16 bits.
// amp_f32_to_e4m3 / amp_e4m3_to_f32: OCP E4M3FN (1 sign, 4 exponent bits
// [bias 7], 3 mantissa bits; no Inf, one NaN encoding, max finite 448),
// round-to-nearest-even, saturate-to-finite on overflow — matches CUDA's
// default __nv_fp8_e4m3 (SATFINITE, E4M3 interpretation). Verified by an
// exhaustive round-trip (byte -> float -> byte) over all 254 non-NaN
// patterns in /tmp/.../scratchpad/e4m3_test.c during development; not
// cross-checked against CUDA's native instruction or ggml bit-for-bit
// (amp_dsv4.cu is DeepSeek-V4 only, out of scope for the GGUF/ROCm model
// this backend was first built for).
struct __nv_bfloat16 {
	unsigned short x;
};

static __device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float f)
{
	union { float f; unsigned int u; } c;
	c.f = f;
	unsigned int lsb = (c.u >> 16) & 1u;
	unsigned int rounded = c.u + 0x7fffu + lsb;
	__nv_bfloat16 r;
	r.x = (unsigned short)(rounded >> 16);
	return r;
}

static __device__ __forceinline__ unsigned char amp_f32_to_e4m3(float f)
{
	union { float f; unsigned int u; } c;
	c.f = f;

	unsigned int sign = (c.u >> 31) & 1u;
	unsigned int absu = c.u & 0x7fffffffu;

	if (absu == 0)
		return (unsigned char)(sign << 7);
	if (absu >= 0x7f800000u) /* NaN or +-Inf */
		return (unsigned char)((sign << 7) | 0x7f);

	int exp32 = (int)((absu >> 23) & 0xffu) - 127;
	unsigned int mant32 = absu & 0x7fffffu;
	int exp_e = exp32 + 7;
	unsigned int mant_e;

	if (exp_e >= 1) {
		unsigned int round_bit = (mant32 >> 19) & 1u;
		unsigned int sticky = (mant32 & 0x7ffffu) != 0;
		unsigned int m3 = mant32 >> 20;
		unsigned int inc = round_bit && (sticky || (m3 & 1u));
		mant_e = m3 + inc;
		if (mant_e > 7u) { mant_e = 0u; exp_e++; }
	} else {
		int shift = 1 - exp_e;
		if (shift > 24)
			return (unsigned char)(sign << 7);
		unsigned int full = (mant32 | 0x800000u);
		unsigned int shifted = full >> (20 + shift);
		unsigned int rem_mask = (1u << (20 + shift)) - 1u;
		unsigned int round_bit = (full >> (19 + shift)) & 1u;
		unsigned int sticky = (full & (rem_mask >> 1)) != 0;
		unsigned int inc = round_bit && (sticky || (shifted & 1u));
		mant_e = shifted + inc;
		exp_e = 0;
		if (mant_e > 7u) { mant_e = 0u; exp_e = 1; }
	}

	if (exp_e >= 15 && !(exp_e == 15 && mant_e < 7u))
		return (unsigned char)((sign << 7) | 0x7e);

	return (unsigned char)((sign << 7) | ((exp_e & 0xfu) << 3) | (mant_e & 0x7u));
}

static __device__ __forceinline__ float amp_e4m3_to_f32(unsigned char x)
{
	unsigned int sign = (x >> 7) & 1u;
	unsigned int exp_e = (x >> 3) & 0xfu;
	unsigned int mant_e = x & 0x7u;

	if (exp_e == 0xfu && mant_e == 0x7u) {
		union { unsigned int u; float f; } c;
		c.u = (sign << 31) | 0x7fc00000u;
		return c.f;
	}
	if (exp_e == 0) {
		float val = (float)mant_e * 0.001953125f; /* mant/8 * 2^-6 = mant * 2^-9 */
		return sign ? -val : val;
	}

	int exp32 = (int)exp_e - 7 + 127;
	unsigned int mant32 = mant_e << 20;
	union { unsigned int u; float f; } c;
	c.u = (sign << 31) | ((unsigned int)exp32 << 23) | mant32;
	return c.f;
}

struct __nv_fp8_e4m3 {
	unsigned char x;

	__nv_fp8_e4m3() = default;
	__device__ __forceinline__ __nv_fp8_e4m3(float f) : x(amp_f32_to_e4m3(f)) {}
	__device__ __forceinline__ operator float() const { return amp_e4m3_to_f32(x); }
};

// CUDA Runtime API names used by the launcher/host-side code in
// amp_gguf.cu / amp_kernels.cu / amp_dsv4.cu. HIP's hip*() entry points
// mirror the cuda*() ones argument-for-argument on the AMD backend (there
// is no built-in cuda-name compatibility layer there, unlike on HIP's
// NVIDIA backend), so plain renames are enough for the handful used here.
typedef hipStream_t cudaStream_t;
typedef hipError_t cudaError_t;
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// Warp-shuffle "_sync" variants (CUDA, Volta+: an explicit lane mask).
// Every call site in this project passes the full mask (0xffffffffu) and
// the default width, which is exactly what HIP's un-suffixed __shfl_*
// already do across the whole wavefront, so the mask argument is simply
// dropped.
#define __shfl_sync(mask, var, srcLane) __shfl((var), (srcLane))
#define __shfl_up_sync(mask, var, delta) __shfl_up((var), (delta))
#define __shfl_down_sync(mask, var, delta) __shfl_down((var), (delta))

// Full-mask vote / match (the callers pass 0xffffffffu; gfx1100 runs wave32, so the ballot fits 32 bits).
#define __ballot_sync(mask, pred) ((unsigned)__ballot((pred)))
static __device__ __forceinline__ unsigned amp_match_any(unsigned v)
{
	unsigned r = 0;

	for (unsigned l = 0; l < 32; l++)
		r |= (unsigned)(__shfl(v, l) == v) << l;

	return r;
}
#define __match_any_sync(mask, v) amp_match_any((unsigned)(v))

// The butterfly (xor) shuffle of the reductions, without the LDS pipe: HIP's __shfl_xor is ds_bpermute_b32 (an LDS round trip per
// step); lane ^ 1, 2, 4, 8 stay inside a row of 16 lanes (DPP row_xmask, folded into the following add) and lane ^ 16 is the other
// row (v_permlanex16 with the identity selection). All lanes must be active, as the full mask promises. Other widths and
// masks take the stock shuffle.
template <int M>
static __device__ __forceinline__ int amp_xor_dpp(int v)
{
	if (M == 16)
		return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);

	return __builtin_amdgcn_update_dpp(0, v, 0x160 + M, 0xf, 0xf, false);
}

template <typename T>
static __device__ __forceinline__ T amp_shfl_xor(T v, int m)
{
	if constexpr (sizeof(T) == 4) {
		const int x = __builtin_bit_cast(int, v);

		switch (m) {
		case 1:  return __builtin_bit_cast(T, amp_xor_dpp<1>(x));
		case 2:  return __builtin_bit_cast(T, amp_xor_dpp<2>(x));
		case 4:  return __builtin_bit_cast(T, amp_xor_dpp<4>(x));
		case 8:  return __builtin_bit_cast(T, amp_xor_dpp<8>(x));
		case 16: return __builtin_bit_cast(T, amp_xor_dpp<16>(x));
		default: break;
		}
	}

	return __shfl_xor(v, m);
}

#define __shfl_xor_sync(mask, var, laneMask) amp_shfl_xor((var), (laneMask))

// Packed-int8 dot product (CUDA __dp4a: 4 signed bytes of a . 4 signed
// bytes of b, + c, no saturation). RDNA3 (gfx11) has no sdot4 (the
// "dot1-insts" feature is GCN/CDNA only); its native packed-int8 dot is
// v_dot4_i32_iu8, reached as sudot4 with both operands flagged signed
// (verified on a gfx1100 RX 7900 XTX: (1,2,3,-4).(5,6,7,8)+100 = 106).
static __device__ __forceinline__ int __dp4a(int a, int b, int c)
{
	return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// Packed byte-wise subtract (CUDA __vsub4: 4 independent 8-bit lanes, each wrapping mod 256, no cross-lane carry). Used for the
// XOR-mask sign trick ((grid ^ mask) - mask) of the i-quant decode kernels, which are VALU-bound on RDNA3: the classic SWAR form
// (subtract with the top bits forced, then restore them with an xor), 6 ops.
static __device__ __forceinline__ unsigned int __vsub4(unsigned int a, unsigned int b)
{
	return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u);
}

#else

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#endif
