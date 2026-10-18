// amp_kernels.cu — small fused elementwise/reduction kernels owned by
// this project (NOT vendored code; the bit-exactness invariant does not
// apply here, but the numerics must reproduce the composite ATen path
// closely enough for the oracle tests: fp32 variance accumulation for
// RMSNorm, fp32 rotation math for RoPE).
//
// Entry points are extern "C" so amp_torch.cpp can call them without
// C++ ABI concerns; tensor plumbing stays in amp_torch.cpp.
//
// Launcher convention: every uc_*_fp16 launcher returns int — 0 on
// success, -1 after rejecting an unsupported configuration on stderr
// (launch failures must be LOUD, see docs/LESSONS.md L42), otherwise the
// cudaError_t from the launch. The bindings turn nonzero values into
// ucode exceptions.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stddef.h>

#define AMP_RMS_BLOCK 256

// out = rmsnorm(x) * w, one block per row.
//
//   x, out: (rows, dim) fp16 contiguous row-major
//   w:      (dim,) fp16
//
// Numerics follow exllamav3's RMSNorm.forward_torch (see
// docs/LESSONS.md L32): accumulate x^2 in fp32, var = mean + eps, multiply the fp32
// value by rsqrt(var), cast to fp16 BEFORE the weight multiply.
__global__ void amp_rms_norm_kernel(const __half *__restrict__ x,
                                   const __half *__restrict__ w,
                                   __half *__restrict__ out,
                                   int dim, float eps)
{
	const int row = blockIdx.x;
	const __half *xr = x + (size_t)row * dim;
	__half *yr = out + (size_t)row * dim;

	float acc = 0.0f;

	for (int i = threadIdx.x; i < dim; i += blockDim.x) {
		float v = __half2float(xr[i]);

		acc += v * v;
	}

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_down_sync(0xffffffffu, acc, off);

	__shared__ float warp_sum[AMP_RMS_BLOCK / 32];

	if ((threadIdx.x & 31) == 0)
		warp_sum[threadIdx.x >> 5] = acc;

	__syncthreads();

	if (threadIdx.x == 0) {
		float total = 0.0f;

		for (int i = 0; i < AMP_RMS_BLOCK / 32; i++)
			total += warp_sum[i];

		warp_sum[0] = rsqrtf(total / dim + eps);
	}

	__syncthreads();

	const float scale = warp_sum[0];

	for (int i = threadIdx.x; i < dim; i += blockDim.x) {
		float v = __half2float(xr[i]);

		yr[i] = __hmul(__float2half(v * scale), w[i]);
	}
}

// The same RMSNorm with 4-half (8-byte) chunks per thread and a block
// sized to the row, like amp_rms_norm_res_kernel: at decode (1-5 rows) the
// scalar kernel above is latency-bound (12 us for 5120 dims; this one is
// ~3 us). Requires dim % 4 == 0 and 8-byte aligned rows.
__global__ void amp_rms_norm_vec_kernel(const __half *__restrict__ x,
                                       const __half *__restrict__ w,
                                       __half *__restrict__ out,
                                       int dim, float eps)
{
	const int row = blockIdx.x;
	const int cnt = dim >> 2;
	const __half *xr = x + (size_t)row * dim;
	__half *yr = out + (size_t)row * dim;

	float acc = 0.0f;

	for (int c = threadIdx.x; c < cnt; c += blockDim.x) {
		uint2 xv = *(const uint2 *)(xr + c * 4);
		float2 v0 = __half22float2(*reinterpret_cast<__half2 *>(&xv.x));
		float2 v1 = __half22float2(*reinterpret_cast<__half2 *>(&xv.y));

		acc += v0.x * v0.x + v0.y * v0.y;
		acc += v1.x * v1.x + v1.y * v1.y;
	}

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_down_sync(0xffffffffu, acc, off);

	__shared__ float warp_sum[32];

	if ((threadIdx.x & 31) == 0)
		warp_sum[threadIdx.x >> 5] = acc;

	__syncthreads();

	if (threadIdx.x == 0) {
		float total = 0.0f;

		for (int i = 0; i < (int)(blockDim.x >> 5); i++)
			total += warp_sum[i];

		warp_sum[0] = rsqrtf(total / dim + eps);
	}

	__syncthreads();

	const float scale = warp_sum[0];

	for (int c = threadIdx.x; c < cnt; c += blockDim.x) {
		uint2 xv = *(const uint2 *)(xr + c * 4);
		uint2 wv = *(const uint2 *)(w + c * 4);
		float2 v0 = __half22float2(*reinterpret_cast<__half2 *>(&xv.x));
		float2 v1 = __half22float2(*reinterpret_cast<__half2 *>(&xv.y));

		/* normalize in fp32, round to fp16, THEN multiply by the weight
		 * in fp16 — as the scalar kernel */
		__half2 n0 = __hmul2(__floats2half2_rn(v0.x * scale, v0.y * scale),
			*reinterpret_cast<__half2 *>(&wv.x));
		__half2 n1 = __hmul2(__floats2half2_rn(v1.x * scale, v1.y * scale),
			*reinterpret_cast<__half2 *>(&wv.y));

		*(uint2 *)(yr + c * 4) = make_uint2(
			*reinterpret_cast<unsigned *>(&n0),
			*reinterpret_cast<unsigned *>(&n1));
	}
}

extern "C" int
amp_rms_norm_fp16(const void *x, const void *w, void *out,
                 long rows, int dim, double eps, cudaStream_t stream)
{
	if (rows <= 0 || dim <= 0)
		return 0;

	if ((dim & 3) == 0 && !(((uintptr_t)x | (uintptr_t)w | (uintptr_t)out) & 7)) {
		int threads = dim >> 2;

		if (threads > 1024)
			threads = 1024;

		threads = (threads + 31) & ~31;

		amp_rms_norm_vec_kernel<<<(int)rows, threads, 0, stream>>>(
			(const __half *)x, (const __half *)w, (__half *)out,
			dim, (float)eps);

		return (int)cudaGetLastError();
	}

	amp_rms_norm_kernel<<<(int)rows, AMP_RMS_BLOCK, 0, stream>>>(
		(const __half *)x, (const __half *)w, (__half *)out,
		dim, (float)eps);

	return (int)cudaGetLastError();
}

// Fused RoPE (rotate_half convention, matching llama.uc rope_apply):
//
//   out[r, d] = x[r, d] * cos[s, d]
//             + sign * x[r, partner(d)] * sin[s, d]
//
//   partner(d) = d < hd/2 ? d + hd/2 : d - hd/2
//   sign(d)    = d < hd/2 ? -1         : +1
//
//   x, out: (rows, hd) fp16 contiguous; rows = B * heads * seq
//   cos, sin: (seq, hd) fp16 contiguous; s = r % seq
//
// Math runs in fp32; inputs/outputs stay fp16.
__global__ void amp_rope_kernel(const __half *__restrict__ x,
                               const __half *__restrict__ cos,
                               const __half *__restrict__ sin,
                               __half *__restrict__ out,
                               long rows, int hd, int seq)
{
	long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
	long total = rows * (long)hd;

	if (idx >= total)
		return;

	int d = (int)(idx % hd);
	long r = idx / hd;
	int s = (int)(r % seq);
	int half = hd / 2;
	long p = (d < half) ? (d + half) : (d - half);
	float sign = (d < half) ? -1.0f : 1.0f;

	float xf = __half2float(x[idx]);
	float pf = __half2float(x[idx - d + p]);
	float cf = __half2float(cos[s * hd + d]);
	float sf = __half2float(sin[s * hd + d]);

	out[idx] = __float2half(xf * cf + pf * sf * sign);
}

extern "C" int
amp_rope_fp16(const void *x, const void *cos, const void *sin, void *out,
             long rows, int hd, int seq, cudaStream_t stream)
{
	if (rows <= 0 || hd <= 0 || seq <= 0)
		return 0;

	long total = rows * (long)hd;
	int block = 256;
	int grid = (int)((total + block - 1) / block);

	amp_rope_kernel<<<grid, block, 0, stream>>>(
		(const __half *)x, (const __half *)cos, (const __half *)sin,
		(__half *)out, rows, hd, seq);

	return (int)cudaGetLastError();
}
// out = silu(g) * u, elementwise fp16 with fp32 math (matches ATen's
// opmath behavior for half tensors).
__global__ void amp_silu_mul_kernel(const __half *__restrict__ g,
                                   const __half *__restrict__ u,
                                   __half *__restrict__ out,
                                   long n)
{
	long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
	long stride = (long)gridDim.x * blockDim.x;

	for (; idx < n; idx += stride) {
		float gf = __half2float(g[idx]);
		float uf = __half2float(u[idx]);
		float s = gf / (1.0f + __expf(-gf));

		out[idx] = __float2half(s * uf);
	}
}

extern "C" int
amp_silu_mul_fp16(const void *g, const void *u, void *out,
                 long n, cudaStream_t stream)
{
	if (n <= 0)
		return 0;

	int block = 256;
	int grid = (int)((n + block - 1) / block);

	if (grid > 2048)
		grid = 2048;

	amp_silu_mul_kernel<<<grid, block, 0, stream>>>(
		(const __half *)g, (const __half *)u, (__half *)out, n);

	return (int)cudaGetLastError();
}

// Fused residual add + RMSNorm, one block per row:
//
//   y = fp16(fp32(x) + fp32(r))          (updated residual stream)
//   n = fp16(fp32(y) * rsqrt(mean(y^2) + eps)) * w
//
// Normalizing the ROUNDED y reproduces the unfused path exactly
// (fp16 add == fp32 add + round for two fp16 inputs; the old path
// normalized the rounded sum). y may alias x (in-place update).
__global__ void amp_rms_norm_res_kernel(const __half *x,
                                       const __half *__restrict__ r,
                                       const __half *__restrict__ w,
                                       __half *y,
                                       __half *__restrict__ n,
                                       int dim, float eps)
/* NB: x and y carry no __restrict__ — y may alias x (in-place update of
 * the residual stream); r and w never alias either. */
{
	const int row = blockIdx.x;
	const int cnt = dim >> 2;            // 4-half chunks per row
	const __half *xr = x + (size_t)row * dim;
	const __half *rr = r + (size_t)row * dim;
	__half *yr = y + (size_t)row * dim;
	__half *nr = n + (size_t)row * dim;

	float acc = 0.0f;

	/* each thread owns whole 4-dim chunks: one 8-byte load per tensor,
	 * coalesced 512-B warp transactions instead of 2-byte strided ones */
	for (int c = threadIdx.x; c < cnt; c += blockDim.x) {
		uint2 xv = *(const uint2 *)(xr + c * 4);
		uint2 rv = *(const uint2 *)(rr + c * 4);

		float2 x0 = __half22float2(*reinterpret_cast<__half2 *>(&xv.x));
		float2 x1 = __half22float2(*reinterpret_cast<__half2 *>(&xv.y));
		float2 r0 = __half22float2(*reinterpret_cast<__half2 *>(&rv.x));
		float2 r1 = __half22float2(*reinterpret_cast<__half2 *>(&rv.y));

		__half2 h0 = __floats2half2_rn(x0.x + r0.x, x0.y + r0.y);
		__half2 h1 = __floats2half2_rn(x1.x + r1.x, x1.y + r1.y);

		*(uint2 *)(yr + c * 4) = make_uint2(
			*reinterpret_cast<unsigned *>(&h0),
			*reinterpret_cast<unsigned *>(&h1));

		float2 v0 = __half22float2(h0);
		float2 v1 = __half22float2(h1);

		acc += v0.x * v0.x + v0.y * v0.y;
		acc += v1.x * v1.x + v1.y * v1.y;
	}

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_down_sync(0xffffffffu, acc, off);

	__shared__ float warp_sum[32];

	if ((threadIdx.x & 31) == 0)
		warp_sum[threadIdx.x >> 5] = acc;

	__syncthreads();

	if (threadIdx.x == 0) {
		float total = 0.0f;

		for (int i = 0; i < blockDim.x >> 5; i++)
			total += warp_sum[i];

		warp_sum[0] = rsqrtf(total / dim + eps);
	}

	__syncthreads();

	const float scale = warp_sum[0];

	for (int c = threadIdx.x; c < cnt; c += blockDim.x) {
		uint2 yv = *(const uint2 *)(yr + c * 4);
		uint2 wv = *(const uint2 *)(w + c * 4);

		float2 v0 = __half22float2(*reinterpret_cast<__half2 *>(&yv.x));
		float2 v1 = __half22float2(*reinterpret_cast<__half2 *>(&yv.y));

		/* normalize in fp32, round to fp16, THEN multiply by the
		 * weight in fp16 — matches the unfused ATen path */
		__half2 n0 = __hmul2(__floats2half2_rn(v0.x * scale, v0.y * scale),
			*reinterpret_cast<__half2 *>(&wv.x));
		__half2 n1 = __hmul2(__floats2half2_rn(v1.x * scale, v1.y * scale),
			*reinterpret_cast<__half2 *>(&wv.y));

		*(uint2 *)(nr + c * 4) = make_uint2(
			*reinterpret_cast<unsigned *>(&n0),
			*reinterpret_cast<unsigned *>(&n1));
	}
}

extern "C" int
amp_rms_norm_res_fp16(const void *x, const void *r, const void *w,
                     void *y, void *n, long rows, int dim, double eps,
                     cudaStream_t stream)
{
	if (rows <= 0 || dim <= 0)
		return 0;

	/* the kernel loads 4-half (8-byte) chunks and drops the tail:
	 * dim must be a multiple of 4 (checked loudly, not silently) */
	if (dim % 4 != 0) {
		fprintf(stderr, "amp_rms_norm_res: dim %d is not a multiple of 4\n",
			dim);
		return -1;
	}

	int cnt = dim >> 2;
	int threads = cnt;

	if (threads > 1024)
		threads = 1024;
	if (threads < 32)
		threads = 32;
	threads &= ~31;
	if (threads < 32)
		threads = 32;

	amp_rms_norm_res_kernel<<<(int)rows, threads, 0, stream>>>(
		(const __half *)x, (const __half *)r, (const __half *)w,
		(__half *)y, (__half *)n, dim, (float)eps);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Fused decode attention (S=1): KV write + RoPE + flash attention in one.
//
// Replaces, per layer: rope(q) + rope(k) + 2x index_copy_ (KV write) +
// full-length masked SDPA + output copy. Reads only slots [0, pos] (the
// current token inclusive), with pos read from DEVICE memory so the
// kernel is CUDA-graph-safe: the captured grid is static, the work
// follows the position buffer.
//
// Layout contract (all fp16 contiguous unless noted):
//   q        (nh, hd)              unroped query heads, one token
//   k_new    (kvh, hd)             unroped key for the current token
//   v_new    (kvh, hd)
//   k_cache  (kvh, max_seq, hd)    KV cache (written at slot pos)
//   v_cache  (kvh, max_seq, hd)
//   cos/sin  (max_seq, hd)         RoPE tables
//   ws       (kvh, NS, rep, 2+hd)  fp32 split workspace
//   out      (nh, hd)              attention output
//   pos      device int64 scalar   current position
//
// GQA: rep = nh / kvh query heads per kv head. Grid (kvh, NS); each
// block is rep warps (one warp per query head), hd spread across the
// warp's lanes (hd must be a multiple of 32; dpl = hd/32 dims per lane).
//
// Phase 1 (every block): rope k_new[j] with the cos/sin row at pos and
// write it plus v_new[j] into cache slot pos. All NS blocks of a kv
// head write identical bytes (benign), so every block can read slot pos
// afterwards without cross-block sync — but WITHIN a block the write
// and the reads race (with rep=4, hd=64 only warps 0-1 write while
// warps 2-3 read slot pos), so a __syncthreads() separates the phases.
//
// Phase 2: online-softmax attention over the block's slot range
// [s*chunk, min(pos+1, (s+1)*chunk)), chunk = ceil(max_seq / NS).
// Partial (m, l, acc) go to the workspace; a combine kernel merges the
// splits deterministically (fixed order, no atomics).
//
// pos is DEVICE memory and cannot be bounds-checked on the host without
// breaking graph capture, so the kernel guards it: pos outside
// [0, max_seq) skips the cache write and the attention (the combine
// then yields a zeroed output — defined, not out-of-bounds).
// ---------------------------------------------------------------------------

struct UcAttnParams {
	const __half *q;
	const __half *k_new;
	const __half *v_new;
	__half *k_cache;
	__half *v_cache;
	const __half *cos;
	const __half *sin;
	float *ws;
	__half *out;
	const long *pos;
	int nh, kvh, hd, max_seq, ns, rep, dpl, wsrow;
	float scale;
};

__global__ void amp_decode_attn_kernel(const UcAttnParams p)
{
	const int j = blockIdx.x;        // kv head
	const int s = blockIdx.y;        // split
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int h = j * p.rep + warp;  // query head
	const int half = p.hd / 2;

	extern __shared__ __half smem[];
	__half *cos_row = smem;          // hd
	__half *sin_row = smem + p.hd;   // hd
	__half *kn_row = smem + 2 * p.hd;// hd

	const long pos = *p.pos;
	const bool pok = pos >= 0 && pos < p.max_seq;

	/* dynamic split: every block gets ceil((pos+1)/ns) slots, so all
	 * blocks stay busy at every context length (graph-safe: pos is
	 * re-read from device memory on every launch) */
	const int total = pok ? (int)pos + 1 : 0;
	const int chunk = (total + p.ns - 1) / p.ns;
	const int begin = s * chunk;
	const int end = min(total, begin + chunk);

	/* load the cos/sin row for this position and the unroped k row */
	if (pok) {
		for (int d = threadIdx.x; d < p.hd; d += blockDim.x) {
			cos_row[d] = p.cos[pos * p.hd + d];
			sin_row[d] = p.sin[pos * p.hd + d];
		}
	}

	for (int d = threadIdx.x; d < p.hd; d += blockDim.x)
		kn_row[d] = p.k_new[j * p.hd + d];

	__syncthreads();

	/* phase 1: rope k_new and write k/v at slot pos (every block of
	 * this kv head writes the same bytes) */
	if (pok) {
		for (int d = threadIdx.x; d < p.hd; d += blockDim.x) {
			int pd = (d < half) ? (d + half) : (d - half);
			float sign = (d < half) ? -1.0f : 1.0f;

			float k = __half2float(kn_row[d]);
			float kp = __half2float(kn_row[pd]);
			float c = __half2float(cos_row[d]);
			float sn = __half2float(sin_row[d]);

			p.k_cache[((size_t)j * p.max_seq + pos) * p.hd + d] =
				__float2half(k * c + kp * sn * sign);
		}

		for (int d = threadIdx.x; d < p.hd; d += blockDim.x)
			p.v_cache[((size_t)j * p.max_seq + pos) * p.hd + d] =
				p.v_new[j * p.hd + d];
	}

	/* phase-1 writes (slot pos) are read back in phase 2 by OTHER warps
	 * of this block — the barrier below is load-bearing (without it the
	 * last split's reads of slot pos race the first warps' writes). */
	__syncthreads();

	/* phase 2: flash attention over [begin, end) for this warp's head.
	 * q is roped in registers. Each lane owns dpl consecutive dims
	 * (dpl = hd/32, 2 or 4): k/v rows load as 4-byte uint pairs per
	 * lane (fully coalesced 128-B warp transactions), q loads the
	 * same way and exchanges the rotary-partner dims (lane ^ 16)
	 * via shfl. */
	const int dpl = p.dpl;

	float qreg[4];

	{
		uint qq0 = *(const uint *)(p.q + (size_t)h * p.hd + lane * dpl);
		uint qq1 = (dpl == 4) ?
			*(const uint *)(p.q + (size_t)h * p.hd + lane * dpl + 2) : 0;
		uint qp0 = __shfl_sync(0xffffffffu, qq0, lane ^ 16);
		uint qp1 = __shfl_sync(0xffffffffu, qq1, lane ^ 16);

		float2 q0 = __half22float2(*reinterpret_cast<__half2 *>(&qq0));
		float2 q1 = __half22float2(*reinterpret_cast<__half2 *>(&qq1));
		float2 p0 = __half22float2(*reinterpret_cast<__half2 *>(&qp0));
		float2 p1 = __half22float2(*reinterpret_cast<__half2 *>(&qp1));
		float2 c0 = __half22float2(
			*reinterpret_cast<__half2 *>(&cos_row[lane * dpl]));
		float2 c1 = __half22float2(
			*reinterpret_cast<__half2 *>(&cos_row[lane * dpl + 2]));
		float2 s0 = __half22float2(
			*reinterpret_cast<__half2 *>(&sin_row[lane * dpl]));
		float2 s1 = __half22float2(
			*reinterpret_cast<__half2 *>(&sin_row[lane * dpl + 2]));

		qreg[0] = q0.x * c0.x + p0.x * s0.x *
			((lane * dpl + 0 < half) ? -1.0f : 1.0f);
		qreg[1] = q0.y * c0.y + p0.y * s0.y *
			((lane * dpl + 1 < half) ? -1.0f : 1.0f);
		qreg[2] = q1.x * c1.x + p1.x * s1.x *
			((lane * dpl + 2 < half) ? -1.0f : 1.0f);
		qreg[3] = q1.y * c1.y + p1.y * s1.y *
			((lane * dpl + 3 < half) ? -1.0f : 1.0f);

		/* dpl == 2: lanes own 2 dims; the upper registers stay
		 * unused (their smem reads may reach into the sin row) */
		for (int i = dpl; i < 4; i++)
			qreg[i] = 0.0f;
	}

	float m = -1e30f, l = 0.0f;
	float acc[4];

	for (int i = 0; i < 4; i++)
		acc[i] = 0.0f;

	/* Slot loop, UNROLL slots at a time: the k/v loads of a batch are
	 * independent and issue together (amortizing HBM latency), and one
	 * merged online-softmax update covers the batch. Tail slots are
	 * predicated off with -inf scores. */
	const int UN = 8;

	for (int t0 = begin; t0 < end; t0 += UN) {
		float ks[UN][4], vs[UN][4];
		float scores[UN];
		int valid[UN];

		for (int u = 0; u < UN; u++) {
			int t = t0 + u;
			valid[u] = t < end;

			const __half *kt = p.k_cache +
				((size_t)j * p.max_seq + t) * p.hd;
			const __half *vt = p.v_cache +
				((size_t)j * p.max_seq + t) * p.hd;

			uint k0 = *(const uint *)(kt + lane * dpl);
			uint k1 = (dpl == 4) ?
				*(const uint *)(kt + lane * dpl + 2) : 0;
			uint v0 = *(const uint *)(vt + lane * dpl);
			uint v1 = (dpl == 4) ?
				*(const uint *)(vt + lane * dpl + 2) : 0;

			float2 kf0 = valid[u] ?
				__half22float2(*reinterpret_cast<__half2 *>(&k0)) :
				make_float2(0.f, 0.f);
			float2 kf1 = valid[u] ?
				__half22float2(*reinterpret_cast<__half2 *>(&k1)) :
				make_float2(0.f, 0.f);
			float2 vf0 = valid[u] ?
				__half22float2(*reinterpret_cast<__half2 *>(&v0)) :
				make_float2(0.f, 0.f);
			float2 vf1 = valid[u] ?
				__half22float2(*reinterpret_cast<__half2 *>(&v1)) :
				make_float2(0.f, 0.f);

			ks[u][0] = kf0.x; ks[u][1] = kf0.y;
			ks[u][2] = kf1.x; ks[u][3] = kf1.y;
			vs[u][0] = vf0.x; vs[u][1] = vf0.y;
			vs[u][2] = vf1.x; vs[u][3] = vf1.y;
		}

		for (int u = 0; u < UN; u++) {
			float part = qreg[0] * ks[u][0] + qreg[1] * ks[u][1] +
				qreg[2] * ks[u][2] + qreg[3] * ks[u][3];

			for (int off = 16; off > 0; off >>= 1)
				part += __shfl_down_sync(0xffffffffu, part, off);

			part = __shfl_sync(0xffffffffu, part, 0);

			scores[u] = valid[u] ? part * p.scale : -1e30f;
		}

		/* merged online-softmax update over the batch */
		float m_new = m;

		for (int u = 0; u < UN; u++)
			m_new = fmaxf(m_new, scores[u]);

		float corr = __expf(m - m_new);
		float lsum = 0.0f;

		/* rescale the old accumulator FIRST, then add the new
		 * contributions at the new scale */
		for (int i = 0; i < 4; i++)
			acc[i] *= corr;

		for (int u = 0; u < UN; u++) {
			float e = __expf(scores[u] - m_new);

			lsum += e;

			acc[0] += e * vs[u][0];
			acc[1] += e * vs[u][1];
			acc[2] += e * vs[u][2];
			acc[3] += e * vs[u][3];
		}

		l = l * corr + lsum;
		m = m_new;
	}

	/* split workspace row: [m, l, pad, pad, acc(hd)] with the acc
	 * area 16-byte aligned; row stride padded to 4 floats */
	float *w = p.ws + ((size_t)j * p.ns + s) * p.rep * p.wsrow +
		warp * p.wsrow;

	w[0] = (begin < end) ? m : -1e30f;
	w[1] = (begin < end) ? l : 0.0f;

	for (int i = 0; i < dpl; i++)
		w[4 + lane * dpl + i] = (begin < end) ? acc[i] : 0.0f;
}

__global__ void amp_decode_attn_combine_kernel(const UcAttnParams p)
{
	const int j = blockIdx.x;        // kv head
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int h = j * p.rep + warp;
	const int dpl = p.dpl;

	float m = -1e30f;

	/* first pass: the running max over splits */
	for (int s = 0; s < p.ns; s++) {
		const float *w = p.ws +
			((size_t)j * p.ns + s) * p.rep * p.wsrow + warp * p.wsrow;

		m = fmaxf(m, w[0]);
	}

	float l = 0.0f;
	float acc[4];

	for (int i = 0; i < 4; i++)
		acc[i] = 0.0f;

	for (int s = 0; s < p.ns; s++) {
		const float *w = p.ws +
			((size_t)j * p.ns + s) * p.rep * p.wsrow + warp * p.wsrow;

		float e = __expf(w[0] - m);

		l += w[1] * e;

		if (dpl == 4) {
			float4 a = *(const float4 *)(w + 4 + lane * dpl);

			acc[0] += a.x * e;
			acc[1] += a.y * e;
			acc[2] += a.z * e;
			acc[3] += a.w * e;
		} else {
			float2 a = *(const float2 *)(w + 4 + lane * dpl);

			acc[0] += a.x * e;
			acc[1] += a.y * e;
		}
	}

	float inv = (l > 0.0f) ? 1.0f / l : 0.0f;   /* all splits empty
		(pos out of range, or a kv head with no slots): defined zero
		output instead of inf/NaN */

	if (dpl == 4) {
		__half2 o0 = __floats2half2_rn(acc[0] * inv, acc[1] * inv);
		__half2 o1 = __floats2half2_rn(acc[2] * inv, acc[3] * inv);

		*(uint2 *)(p.out + (size_t)h * p.hd + lane * dpl) =
			make_uint2(*(const unsigned *)&o0, *(const unsigned *)&o1);
	} else {
		__half2 o0 = __floats2half2_rn(acc[0] * inv, acc[1] * inv);

		*(uint *)(p.out + (size_t)h * p.hd + lane * dpl) =
			*(const unsigned *)&o0;
	}
}

extern "C" int
amp_decode_attn_fp16(const void *q, const void *k_new, const void *v_new,
                    void *k_cache, void *v_cache, const void *cos,
                    const void *sin, void *ws, void *out,
                    const long *pos, int nh, int kvh, int hd, int max_seq,
                    int ns, float scale, cudaStream_t stream)
{
	int rep = nh / kvh;
	int dpl = hd / 32;
	int wsrow = 4 + ((hd + 3) & ~3);

	/* Unsupported configs must be LOUD (L42): a silent return leaves the
	 * output buffer stale and the failure surfaces far from the cause. */
	if (rep <= 0 || (dpl != 2 && dpl != 4) || hd % 32 != 0 || ns <= 0) {
		fprintf(stderr, "amp_decode_attn: unsupported config "
			"(nh=%d kvh=%d hd=%d ns=%d: dpl %d, rep %d)\n",
			nh, kvh, hd, ns, dpl, rep);
		return -1;
	}

	UcAttnParams p;

	p.q = (const __half *)q;
	p.k_new = (const __half *)k_new;
	p.v_new = (const __half *)v_new;
	p.k_cache = (__half *)k_cache;
	p.v_cache = (__half *)v_cache;
	p.cos = (const __half *)cos;
	p.sin = (const __half *)sin;
	p.ws = (float *)ws;
	p.out = (__half *)out;
	p.pos = pos;
	p.nh = nh;
	p.kvh = kvh;
	p.hd = hd;
	p.max_seq = max_seq;
	p.ns = ns;
	p.rep = rep;
	p.dpl = dpl;
	p.wsrow = wsrow;
	p.scale = scale;

	dim3 grid(kvh, ns);
	int block = rep * 32;
	int smem = 3 * hd * (int)sizeof(__half);

	amp_decode_attn_kernel<<<grid, block, smem, stream>>>(p);

	cudaError_t e = cudaGetLastError();

	if (e != cudaSuccess)
		return (int)e;

	amp_decode_attn_combine_kernel<<<kvh, rep * 32, 0, stream>>>(p);

	return (int)cudaGetLastError();
}

/* ------------------------------------------------------------------ */
/* Fused argmax + decode-state advance.                                */
/*                                                                     */
/* Replaces, in one launch: topk(1) over the vocab (multi-kernel       */
/* reduction), the ids copy_, the pos add_ and the per-step log        */
/* write. Deterministic: ties resolve to the lowest index, and the     */
/* final combine scans the per-block workspace in block order.         */
/* NaN logits compare false against everything and are therefore       */
/* skipped (treated as lowest); if EVERY logit is invalid the combine  */
/* writes id 0 rather than -1 (an out-of-vocab id would send the next  */
/* embedding gather out of bounds).                                    */
/*                                                                     */
/* grid = fixed 64 blocks x 256 threads. Phase 1: each block reduces   */
/* its strided slice to (best value, index) and parks it in the        */
/* workspace. The last block to finish (atomic counter) reduces the    */
/* 64 candidates, then performs the state update:                      */
/*     log[pos]   = ids          (token just processed, at its slot)   */
/*     ids        = argmax(logits)                                     */
/*     pos       += 1                                                  */
/* The counter self-resets so the buffer needs no per-launch setup.    */
/* The log write is guarded by log_cap (the log tensor's numel): a     */
/* pos at or past the capacity skips the write instead of writing      */
/* out of bounds (pos is device memory; the host cannot check it       */
/* without breaking graph capture).                                    */
/* ------------------------------------------------------------------ */

#define AMP_ARGMAX_BLOCKS 64
#define AMP_ARGMAX_THREADS 256

__global__ void amp_argmax_advance_kernel(
	const __half *__restrict__ logits, int n,
	long long *__restrict__ ids, long long *__restrict__ pos,
	long long *__restrict__ log, long long log_cap,
	float *__restrict__ ws, unsigned int *__restrict__ counter)
{
	__shared__ float sval[AMP_ARGMAX_THREADS];
	__shared__ int sidx[AMP_ARGMAX_THREADS];

	float best = -INFINITY;
	int bi = -1;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
			i += gridDim.x * blockDim.x) {
		float v = __half2float(logits[i]);

		if (v > best || (v == best && bi >= 0 && i < bi)) {
			best = v;
			bi = i;
		}
	}

	sval[threadIdx.x] = best;
	sidx[threadIdx.x] = bi;

	__syncthreads();

	/* block-wide reduce (first-index tie-break) */
	for (int off = blockDim.x / 2; off > 0; off >>= 1) {
		if (threadIdx.x < off) {
			float v = sval[threadIdx.x + off];
			int i = sidx[threadIdx.x + off];

			if (v > sval[threadIdx.x] ||
				(v == sval[threadIdx.x] &&
				 i < sidx[threadIdx.x])) {
				sval[threadIdx.x] = v;
				sidx[threadIdx.x] = i;
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		ws[blockIdx.x] = sval[0];
		ws[AMP_ARGMAX_BLOCKS + blockIdx.x] = (float)sidx[0];
		__threadfence();

		unsigned int done =
			atomicAdd(counter, 1u) + 1u;

		if (done == gridDim.x) {
			/* last block: reduce the 64 candidates */
			float b = -INFINITY;
			int bj = -1;

			for (int i = 0; i < AMP_ARGMAX_BLOCKS; i++) {
				float v = ws[i];
				int j = (int)ws[AMP_ARGMAX_BLOCKS + i];

				if (v > b || (v == b && bj >= 0 && j < bj)) {
					b = v;
					bj = j;
				}
			}

			long long p = pos[0];

			if (p >= 0 && p < log_cap)
				log[p] = ids[0];

			ids[0] = (bj >= 0) ? bj : 0;
			pos[0] = p + 1;

			*counter = 0u;
		}
	}
}

extern "C" int
amp_argmax_advance_fp16(const void *logits, int n,
	void *ids, void *pos, void *log, long long log_cap,
	void *ws, cudaStream_t stream)
{
	/* Workspace layout: 2 * AMP_ARGMAX_BLOCKS floats of per-block
	 * candidates followed by one uint32 counter (bit-cast into the
	 * float array's tail).
	 *
	 * ws == NULL falls back to per-device static allocations (the
	 * launcher runs under the binding's device guard, so the current
	 * device selects the slot). A persistent caller-provided buffer
	 * keeps decoders isolated from each other. */
	static float *ws_fb[16] = {};
	static unsigned int *counter_fb[16] = {};
	float *wsp;
	unsigned int *counterp;
	cudaError_t err;
	int dev = 0;

	if (ws) {
		wsp = (float *)ws;
		counterp = (unsigned int *)(wsp + 2 * AMP_ARGMAX_BLOCKS);
	} else {
		cudaGetDevice(&dev);
		if (dev < 0 || dev > 15)
			dev = 0;

		if (!ws_fb[dev]) {
			err = cudaMalloc(&ws_fb[dev],
				2 * AMP_ARGMAX_BLOCKS * sizeof(float));

			if (err == cudaSuccess)
				err = cudaMalloc(&counter_fb[dev],
					sizeof(unsigned int));

			if (err == cudaSuccess)
				err = cudaMemset(counter_fb[dev], 0,
					sizeof(unsigned int));

			if (err != cudaSuccess) {
				if (ws_fb[dev]) {
					cudaFree(ws_fb[dev]);
					ws_fb[dev] = NULL;
				}

				return (int)err;
			}
		}

		wsp = ws_fb[dev];
		counterp = counter_fb[dev];
	}

	amp_argmax_advance_kernel<<<AMP_ARGMAX_BLOCKS, AMP_ARGMAX_THREADS, 0,
		stream>>>((const __half *)logits, n,
		(long long *)ids, (long long *)pos, (long long *)log, log_cap,
		wsp, counterp);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Fused gated delta rule, sequential form (GatedDeltaNet prefill + decode).
//
// One launch processes all S tokens of a chunk for one layer; replaces the
// per-token ATen loop (~12 small kernels per token per layer). Exact math
// of the reference recurrence (ucode/qwen35.uc delta_rule_scan), fp32:
//
//   kv    = S^T k_t                     (S before decay)
//   v_new = beta_t * (v_t - g_t * kv)
//   S     = g_t * S + k_t (x) v_new
//   o_t   = scale * S^T q_t
//
//   q, k   (S, Hk, DK) fp32 contiguous, l2-normalized; v head h reads
//          q/k head h / (Hv / Hk) (GQA grouping, no repeat needed)
//   v      (S, Hv, dv) fp32
//   beta   (S, Hv) fp32
//   gexp   (S, Hv) fp32 (= exp(g), the per-token decay)
//   state  (Hv, DK, dv) fp32, updated in place
//   out    (S, Hv, dv) fp32
//
// Grid (Hv, dv / VT), block VT * RS threads. A state column S[:, j] is
// split over RS adjacent lanes (lane % RS = r owns rows m * RS + r, so
// the shared-memory reads of the RS lanes hit distinct banks): the
// token's two 128-long dependent FMA chains become DK / RS long, the
// partial sums meet in a shuffle. The scan is latency-bound (one token's
// chain is ~1 us at DK = 128 on a single thread per column, and only 48
// blocks of two warps exist per GPU), not throughput-bound.
//
// Tokens are processed in tiles of AMP_GDN_TB: the next tile's q / k /
// v / g / beta are copied to the other half of a double buffer with
// cp.async while the current one computes, so the per-token loop has
// no global loads and no barriers (one pair per tile).
// ---------------------------------------------------------------------------

#define AMP_GDN_TB 16        /* tokens per staged tile */

__device__ __forceinline__ void amp_cp_async16(void *dst, const void *src)
{
	unsigned d = (unsigned)__cvta_generic_to_shared(dst);

	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}

__device__ __forceinline__ void amp_cp_async4(void *dst, const void *src)
{
	unsigned d = (unsigned)__cvta_generic_to_shared(dst);

	asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(src));
}

template <int DK, int VT, int RS>
__global__ __launch_bounds__(VT * RS)
void amp_gdn_scan_kernel(const float *__restrict__ q, const float *__restrict__ k,
                        const float *__restrict__ v, const float *__restrict__ beta,
                        const float *__restrict__ gexp, float *__restrict__ state,
                        float *__restrict__ out, int S, int Hk, int Hv, int dv,
                        float scale, float *__restrict__ states_out, int n_out)
{
	constexpr int NT = VT * RS;
	constexpr int RPT = DK / RS;                 // state rows per thread
	constexpr int TB = AMP_GDN_TB;
	constexpr int BUF = TB * (2 * DK + VT + 2);  // floats per tile buffer

	const int h = blockIdx.x;
	const int r = threadIdx.x % RS;
	const int jl = threadIdx.x / RS;
	const int j0 = blockIdx.y * VT;
	const int j = j0 + jl;
	const int hk = h / (Hv / Hk);

	// tile buffer: q[TB][DK] | k[TB][DK] | v[TB][VT] | g[TB] | beta[TB]
	__shared__ __align__(16) float sbuf[2 * BUF];

	float st[RPT];
	float *sp = state + (size_t)h * DK * dv + j;

#pragma unroll
	for (int m = 0; m < RPT; m++)
		st[m] = sp[(size_t)(m * RS + r) * dv];

	const int ntiles = (S + TB - 1) / TB;

	// stage tile `tile` into buffer `bi`: one async-copy group
	auto stage = [&](int tile, int bi) {
		float *bq = sbuf + bi * BUF;
		float *bk = bq + TB * DK;
		float *bv = bk + TB * DK;
		float *bg = bv + TB * VT;
		float *bb = bg + TB;
		const int t0 = tile * TB;
		const int n = min(TB, S - t0);

		for (int c = threadIdx.x; c < n * (DK / 4); c += NT) {
			int tt = c / (DK / 4), x = c % (DK / 4);
			size_t go = ((size_t)(t0 + tt) * Hk + hk) * DK + x * 4;

			amp_cp_async16(bq + tt * DK + x * 4, q + go);
			amp_cp_async16(bk + tt * DK + x * 4, k + go);
		}

		for (int c = threadIdx.x; c < n * (VT / 4); c += NT) {
			int tt = c / (VT / 4), x = c % (VT / 4);

			amp_cp_async16(bv + tt * VT + x * 4,
				v + ((size_t)(t0 + tt) * Hv + h) * dv + j0 + x * 4);
		}

		for (int tt = threadIdx.x; tt < n; tt += NT) {
			amp_cp_async4(bg + tt, gexp + (size_t)(t0 + tt) * Hv + h);
			amp_cp_async4(bb + tt, beta + (size_t)(t0 + tt) * Hv + h);
		}

		asm volatile("cp.async.commit_group;\n" ::);
	};

	stage(0, 0);

	for (int tile = 0; tile < ntiles; tile++) {
		if (tile + 1 < ntiles) {
			stage(tile + 1, (tile + 1) & 1);
			asm volatile("cp.async.wait_group 1;\n" ::);
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::);

		__syncthreads();

		const float *bq = sbuf + (tile & 1) * BUF;
		const float *bk = bq + TB * DK;
		const float *bv = bk + TB * DK;
		const float *bg = bv + TB * VT;
		const float *bb = bg + TB;
		const int t0 = tile * TB;
		const int n = min(TB, S - t0);

		for (int tt = 0; tt < n; tt++) {
			const int t = t0 + tt;
			const float *skb = bk + tt * DK, *sqb = bq + tt * DK;
			const float g = bg[tt], b = bb[tt], vt = bv[tt * VT + jl];

			float kv0 = 0.0f, kv1 = 0.0f;

#pragma unroll
			for (int m = 0; m < RPT; m += 2) {
				kv0 = fmaf(st[m], skb[m * RS + r], kv0);
				kv1 = fmaf(st[m + 1], skb[(m + 1) * RS + r], kv1);
			}

			float kv = kv0 + kv1;

#pragma unroll
			for (int off = 1; off < RS; off <<= 1)
				kv += __shfl_xor_sync(0xffffffffu, kv, off);

			const float vn = b * (vt - g * kv);

			float o0 = 0.0f, o1 = 0.0f;

#pragma unroll
			for (int m = 0; m < RPT; m += 2) {
				st[m] = fmaf(g, st[m], skb[m * RS + r] * vn);
				st[m + 1] = fmaf(g, st[m + 1], skb[(m + 1) * RS + r] * vn);
				o0 = fmaf(st[m], sqb[m * RS + r], o0);
				o1 = fmaf(st[m + 1], sqb[(m + 1) * RS + r], o1);
			}

			float o = o0 + o1;

#pragma unroll
			for (int off = 1; off < RS; off <<= 1)
				o += __shfl_xor_sync(0xffffffffu, o, off);

			if (r == 0)
				out[((size_t)t * Hv + h) * dv + j] = o * scale;

			// speculative verify: the state after token t (t < n_out),
			// same layout as `state`, so a rejection after token t
			// restores it with a plain copy instead of re-running the
			// accepted tokens
			if (t < n_out) {
				float *so = states_out + ((size_t)t * Hv + h) * DK * dv + j;

#pragma unroll
				for (int m = 0; m < RPT; m++)
					so[(size_t)(m * RS + r) * dv] = st[m];
			}
		}

		// everyone is done with this buffer before it is restaged
		__syncthreads();
	}

#pragma unroll
	for (int m = 0; m < RPT; m++)
		sp[(size_t)(m * RS + r) * dv] = st[m];
}

// Register-blocked variant for DK = 128: a thread owns C = 4 adjacent
// columns x DK / RS contiguous rows of the state, so each k / q value it
// reads from shared memory (one 128-bit load per four rows) serves four
// columns. The scan is bound by shared-memory instruction throughput,
// not by FMAs: the per-column form needs ~1500 LDS wavefronts per token
// per head. Rows of the staged q / k are padded by 4 floats per RS-th so
// the RS row blocks read from distinct bank quads.
#define AMP_GDN_CB_C 4
#define AMP_GDN_CB_DEFAULT 11

template <int DK, int VT, int RS>
__global__ __launch_bounds__((VT / AMP_GDN_CB_C) * RS)
void amp_gdn_scan_cb_kernel(const float *__restrict__ q, const float *__restrict__ k,
                           const float *__restrict__ v, const float *__restrict__ beta,
                           const float *__restrict__ gexp, float *__restrict__ state,
                           float *__restrict__ out, int S, int Hk, int Hv, int dv,
                           float scale, float *__restrict__ states_out, int n_out)
{
	constexpr int C = AMP_GDN_CB_C;
	constexpr int NT = (VT / C) * RS;
	constexpr int RPT = DK / RS;                  // rows per thread
	constexpr int DKP = RS * (RPT + 4);           // padded staged row
	constexpr int TB = AMP_GDN_TB;
	constexpr int BUF = TB * (2 * DKP + VT + 2);

	const int h = blockIdx.x;
	const int r = threadIdx.x % RS;
	const int cg = threadIdx.x / RS;              // column group
	const int j0 = blockIdx.y * VT;
	const int jc = j0 + cg * C;
	const int hk = h / (Hv / Hk);

	__shared__ __align__(16) float sbuf[2 * BUF];

	float st[C][RPT];
	const float *sp = state + (size_t)h * DK * dv + jc;

#pragma unroll
	for (int m = 0; m < RPT; m++) {
		float4 x = *(const float4 *)(sp + (size_t)(r * RPT + m) * dv);

		st[0][m] = x.x;
		st[1][m] = x.y;
		st[2][m] = x.z;
		st[3][m] = x.w;
	}

	const int ntiles = (S + TB - 1) / TB;

	auto stage = [&](int tile, int bi) {
		float *bq = sbuf + bi * BUF;
		float *bk = bq + TB * DKP;
		float *bv = bk + TB * DKP;
		float *bg = bv + TB * VT;
		float *bb = bg + TB;
		const int t0 = tile * TB;
		const int n = min(TB, S - t0);

		for (int c = threadIdx.x; c < n * (DK / 4); c += NT) {
			int tt = c / (DK / 4), x = c % (DK / 4);
			int i = x * 4;
			int dst = tt * DKP + (i / RPT) * (RPT + 4) + (i % RPT);
			size_t go = ((size_t)(t0 + tt) * Hk + hk) * DK + i;

			amp_cp_async16(bq + dst, q + go);
			amp_cp_async16(bk + dst, k + go);
		}

		for (int c = threadIdx.x; c < n * (VT / 4); c += NT) {
			int tt = c / (VT / 4), x = c % (VT / 4);

			amp_cp_async16(bv + tt * VT + x * 4,
				v + ((size_t)(t0 + tt) * Hv + h) * dv + j0 + x * 4);
		}

		for (int tt = threadIdx.x; tt < n; tt += NT) {
			amp_cp_async4(bg + tt, gexp + (size_t)(t0 + tt) * Hv + h);
			amp_cp_async4(bb + tt, beta + (size_t)(t0 + tt) * Hv + h);
		}

		asm volatile("cp.async.commit_group;\n" ::);
	};

	stage(0, 0);

	for (int tile = 0; tile < ntiles; tile++) {
		if (tile + 1 < ntiles) {
			stage(tile + 1, (tile + 1) & 1);
			asm volatile("cp.async.wait_group 1;\n" ::);
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::);

		__syncthreads();

		const float *bq = sbuf + (tile & 1) * BUF;
		const float *bk = bq + TB * DKP;
		const float *bv = bk + TB * DKP;
		const float *bg = bv + TB * VT;
		const float *bb = bg + TB;
		const int t0 = tile * TB;
		const int n = min(TB, S - t0);

		for (int tt = 0; tt < n; tt++) {
			const int t = t0 + tt;
			const float4 *skb = (const float4 *)(bk + tt * DKP + r * (RPT + 4));
			const float4 *sqb = (const float4 *)(bq + tt * DKP + r * (RPT + 4));
			const float g = bg[tt], b = bb[tt];
			const float4 vv = *(const float4 *)(bv + tt * VT + cg * C);

			float kva[C], kvb[C];

#pragma unroll
			for (int c = 0; c < C; c++) {
				kva[c] = 0.0f;
				kvb[c] = 0.0f;
			}

#pragma unroll
			for (int m4 = 0; m4 < RPT / 4; m4++) {
				float4 a = skb[m4];

#pragma unroll
				for (int c = 0; c < C; c++) {
					kva[c] = fmaf(st[c][4 * m4], a.x, kva[c]);
					kvb[c] = fmaf(st[c][4 * m4 + 1], a.y, kvb[c]);
					kva[c] = fmaf(st[c][4 * m4 + 2], a.z, kva[c]);
					kvb[c] = fmaf(st[c][4 * m4 + 3], a.w, kvb[c]);
				}
			}

			float vn[C];
			const float vtv[C] = { vv.x, vv.y, vv.z, vv.w };

#pragma unroll
			for (int c = 0; c < C; c++) {
				float kv = kva[c] + kvb[c];

#pragma unroll
				for (int off = 1; off < RS; off <<= 1)
					kv += __shfl_xor_sync(0xffffffffu, kv, off);

				vn[c] = b * (vtv[c] - g * kv);
			}

			float oa[C], ob[C];

#pragma unroll
			for (int c = 0; c < C; c++) {
				oa[c] = 0.0f;
				ob[c] = 0.0f;
			}

#pragma unroll
			for (int m4 = 0; m4 < RPT / 4; m4++) {
				float4 a = skb[m4];
				float4 e = sqb[m4];

#pragma unroll
				for (int c = 0; c < C; c++) {
					st[c][4 * m4] = fmaf(g, st[c][4 * m4], a.x * vn[c]);
					st[c][4 * m4 + 1] = fmaf(g, st[c][4 * m4 + 1], a.y * vn[c]);
					st[c][4 * m4 + 2] = fmaf(g, st[c][4 * m4 + 2], a.z * vn[c]);
					st[c][4 * m4 + 3] = fmaf(g, st[c][4 * m4 + 3], a.w * vn[c]);
					oa[c] = fmaf(st[c][4 * m4], e.x, oa[c]);
					ob[c] = fmaf(st[c][4 * m4 + 1], e.y, ob[c]);
					oa[c] = fmaf(st[c][4 * m4 + 2], e.z, oa[c]);
					ob[c] = fmaf(st[c][4 * m4 + 3], e.w, ob[c]);
				}
			}

			float o[C];

#pragma unroll
			for (int c = 0; c < C; c++) {
				o[c] = oa[c] + ob[c];

#pragma unroll
				for (int off = 1; off < RS; off <<= 1)
					o[c] += __shfl_xor_sync(0xffffffffu, o[c], off);
			}

			if (r == 0)
				*(float4 *)(out + ((size_t)t * Hv + h) * dv + jc) =
					make_float4(o[0] * scale, o[1] * scale, o[2] * scale,
						o[3] * scale);

			// state after token t for the speculative verify rollback
			if (t < n_out) {
				float *so = states_out + ((size_t)t * Hv + h) * DK * dv + jc;

#pragma unroll
				for (int m = 0; m < RPT; m++)
					*(float4 *)(so + (size_t)(r * RPT + m) * dv) =
						make_float4(st[0][m], st[1][m], st[2][m], st[3][m]);
			}
		}

		__syncthreads();
	}

	float *wp = state + (size_t)h * DK * dv + jc;

#pragma unroll
	for (int m = 0; m < RPT; m++)
		*(float4 *)(wp + (size_t)(r * RPT + m) * dv) =
			make_float4(st[0][m], st[1][m], st[2][m], st[3][m]);
}

/* AMP_GDN_SCAN_CFG = VT*100 + RS, e.g. 3204 (32 columns x 4 lanes);
 * default below */
static int amp_gdn_scan_cfg(void)
{
	static int cfg = -1;

	if (cfg < 0) {
		const char *e = getenv("AMP_GDN_SCAN_CFG");

		cfg = e ? atoi(e) : 0;
	}

	return cfg;
}

extern "C" int
amp_gdn_scan_f32(const void *q, const void *k, const void *v, const void *beta,
                const void *gexp, void *state, void *out, int S, int Hk,
                int Hv, int dk, int dv, float scale, void *states_out,
                int n_out, cudaStream_t stream)
{
	if (S <= 0)
		return 0;

	int cfg = amp_gdn_scan_cfg();

	/* default: the register-blocked kernel (DK = 128; cfg 0 = the
	 * default shape, 10 .. 13 select another for experiments) */
	if ((cfg == 0 || (cfg >= 10 && cfg <= 13)) && dk == 128 && Hk > 0 &&
	    Hv % Hk == 0 && dv % 4 == 0) {
		int c = (cfg == 0) ? AMP_GDN_CB_DEFAULT : cfg;

#define AMP_CB(VT_, RS_) do { \
	if (dv % (VT_) == 0) { \
		dim3 g(Hv, dv / (VT_)); \
		amp_gdn_scan_cb_kernel<128, VT_, RS_><<<g, ((VT_) / AMP_GDN_CB_C) * (RS_), \
			0, stream>>>((const float *)q, (const float *)k, (const float *)v, \
			(const float *)beta, (const float *)gexp, (float *)state, \
			(float *)out, S, Hk, Hv, dv, scale, (float *)states_out, n_out); \
		return (int)cudaGetLastError(); \
	} } while (0)

		switch (c) {
		case 10: AMP_CB(32, 4); break;
		case 11: AMP_CB(32, 8); break;
		case 12: AMP_CB(16, 8); break;
		case 13: AMP_CB(16, 4); break;
		}
#undef AMP_CB
	}

	if (cfg == 0)
		cfg = 3204;

	int VT = cfg / 100, RS = cfg % 100;

	if (Hk <= 0 || Hv % Hk != 0 || dv % VT != 0) {
		fprintf(stderr, "amp_gdn_scan: unsupported heads/dims (Hk %d, Hv %d, dv %d)\n",
			Hk, Hv, dv);
		return -1;
	}

	dim3 grid(Hv, dv / VT);

#define AMP_GS(DK, VT_, RS_) do { \
	amp_gdn_scan_kernel<DK, VT_, RS_><<<grid, (VT_) * (RS_), 0, stream>>>( \
		(const float *)q, (const float *)k, (const float *)v, \
		(const float *)beta, (const float *)gexp, (float *)state, \
		(float *)out, S, Hk, Hv, dv, scale, (float *)states_out, n_out); \
	return (int)cudaGetLastError(); } while (0)

	if (dk == 128) {
		switch (cfg) {
		case 6401: AMP_GS(128, 64, 1);
		case 6404: AMP_GS(128, 64, 4);
		case 3204: AMP_GS(128, 32, 4);
		case 3208: AMP_GS(128, 32, 8);
		case 1604: AMP_GS(128, 16, 4);
		case 1608: AMP_GS(128, 16, 8);
		case 3202: AMP_GS(128, 32, 2);
		case 1616: AMP_GS(128, 16, 16);
		}
	}
	else if (dk == 64) {
		switch (cfg) {
		case 6401: AMP_GS(64, 64, 1);
		case 6404: AMP_GS(64, 64, 4);
		case 3204: AMP_GS(64, 32, 4);
		case 3208: AMP_GS(64, 32, 8);
		case 1604: AMP_GS(64, 16, 4);
		case 1608: AMP_GS(64, 16, 8);
		case 3202: AMP_GS(64, 32, 2);
		case 1616: AMP_GS(64, 16, 16);
		}
	}
	else {
		fprintf(stderr, "amp_gdn_scan: unsupported k head dim %d (64, 128)\n", dk);
		return -1;
	}

#undef AMP_GS

	fprintf(stderr, "amp_gdn_scan: unsupported AMP_GDN_SCAN_CFG %d\n", cfg);
	return -1;
}

// ---------------------------------------------------------------------------
// GatedDeltaNet input preparation, fused: causal conv1d (K taps over the
// carried conv state + the chunk) + SiLU, split into q/k/v, fp32, and
// l2-normalization of the q/k heads — one launch instead of ~20 small
// ATen kernels per GDN layer.
//
//   qkv   (S, D) fp16, D = 2 * Hk * dk + Hv * dv (q | k | v channels)
//   cs    (D, K-1) fp16 conv state (inputs preceding the chunk)
//   w     (D, K) fp16 conv taps
//   q, k  (S, Hk, dk) fp32 out (l2-normalized, eps 1e-6 inside rsqrt)
//   v     (S, Hv, dv) fp32 out
//
// Grid (S, 2*Hk + Hv), block = head dim (dk == dv == blockDim): block
// (t, hb) produces one head vector of token t. Accumulation in fp32
// (the composite path rounded every tap's product and sum to fp16).
//
// The conv state update runs as a second, tiny kernel AFTER all reads.
// ---------------------------------------------------------------------------

__global__ void amp_gdn_prep_kernel(const __half *__restrict__ qkv,
                                   const __half *__restrict__ cs,
                                   const __half *__restrict__ w,
                                   float *__restrict__ q, float *__restrict__ k,
                                   float *__restrict__ v, int S, int D, int K,
                                   int Hk, int Hv, int hd)
{
	const int t = blockIdx.x;
	const int hb = blockIdx.y;              // q heads, then k heads, then v heads
	const int e = threadIdx.x;
	const int d = hb * hd + e;              // channel
	const int km1 = K - 1;

	float acc = 0.0f;

	for (int j = 0; j < K; j++) {
		/* padded input index t + j over [state (K-1) | chunk (S)] */
		int p = t + j;
		float x = (p < km1) ? __half2float(cs[(size_t)d * km1 + p]) :
			__half2float(qkv[(size_t)(p - km1) * D + d]);

		acc = fmaf(__half2float(w[(size_t)d * K + j]), x, acc);
	}

	float y = acc / (1.0f + __expf(-acc));     // silu

	if (hb >= 2 * Hk) {
		int hv = hb - 2 * Hk;

		v[((size_t)t * Hv + hv) * hd + e] = y;
		return;
	}

	/* q/k head: l2-normalize over the head dim */
	__shared__ float red[32];
	float ss = y * y;

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		ss += __shfl_down_sync(0xffffffffu, ss, off);

	if ((e & 31) == 0)
		red[e >> 5] = ss;

	__syncthreads();

	if (e < 32) {
		float s2 = (e < (int)(blockDim.x >> 5)) ? red[e] : 0.0f;

#pragma unroll
		for (int off = 16; off > 0; off >>= 1)
			s2 += __shfl_down_sync(0xffffffffu, s2, off);

		if (e == 0)
			red[0] = rsqrtf(s2 + 1e-6f);
	}

	__syncthreads();

	float yn = y * red[0];

	if (hb < Hk)
		q[((size_t)t * Hk + hb) * hd + e] = yn;
	else
		k[((size_t)t * Hk + (hb - Hk)) * hd + e] = yn;
}

/* Per-token snapshots of the conv state for the speculative verify:
 * snapshot t is the window of K-1 padded inputs ending at token t,
 * (D, K-1) like the state itself. */
#define AMP_GDN_SPEC_MAX 8

struct amp_gdn_spec {
	__half *p[AMP_GDN_SPEC_MAX];
	int n;
};

/* conv state <- last K-1 padded inputs; per channel ascending, so a read
 * of an old state slot (S < K-1) happens before that slot is rewritten.
 * The snapshots read the old state too, so they are written first. */
__global__ void amp_gdn_conv_state_kernel(const __half *__restrict__ qkv,
                                         __half *__restrict__ cs, int S, int D,
                                         int K, amp_gdn_spec spec)
{
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	int km1 = K - 1;

	if (d >= D)
		return;

	for (int t = 0; t < spec.n; t++)
		for (int i = 0; i < km1; i++) {
			int p = t + 1 + i;      // padded index of snapshot t slot i

			spec.p[t][(size_t)d * km1 + i] = (p < km1) ?
				cs[(size_t)d * km1 + p] : qkv[(size_t)(p - km1) * D + d];
		}

	for (int i = 0; i < km1; i++) {
		int p = S + i;      // padded index of the new state slot i

		cs[(size_t)d * km1 + i] = (p < km1) ? cs[(size_t)d * km1 + p] :
			qkv[(size_t)(p - km1) * D + d];
	}
}

extern "C" int
amp_gdn_prep_fp16(const void *qkv, void *conv_state, const void *conv_w,
                 void *q, void *k, void *v, int S, int D, int K, int Hk,
                 int Hv, int hd, void *const *spec, int nspec,
                 cudaStream_t stream)
{
	if (S <= 0)
		return 0;

	if (hd > 1024 || hd % 32 != 0 || D != (2 * Hk + Hv) * hd || K < 1 ||
	    nspec < 0 || nspec > AMP_GDN_SPEC_MAX || nspec > S) {
		fprintf(stderr, "amp_gdn_prep: unsupported dims (D %d, Hk %d, Hv %d, hd %d, K %d, "
			"snapshots %d)\n", D, Hk, Hv, hd, K, nspec);
		return -1;
	}

	dim3 grid(S, 2 * Hk + Hv);

	amp_gdn_prep_kernel<<<grid, hd, 0, stream>>>((const __half *)qkv,
		(const __half *)conv_state, (const __half *)conv_w, (float *)q,
		(float *)k, (float *)v, S, D, K, Hk, Hv, hd);

	if (K > 1) {
		amp_gdn_spec sp;

		sp.n = nspec;

		for (int i = 0; i < AMP_GDN_SPEC_MAX; i++)
			sp.p[i] = (i < nspec) ? (__half *)spec[i] : NULL;

		amp_gdn_conv_state_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
			(const __half *)qkv, (__half *)conv_state, S, D, K, sp);
	}

	return (int)cudaGetLastError();
}

// GDN gates: beta = sigmoid(b), gexp = exp(-exp(A_log) * softplus(a + dt_bias))
//   a, b (S, Hv) fp16; dt_bias (Hv) fp32; nea = -exp(A_log) (Hv) fp32;
//   beta, gexp (S, Hv) fp32 out
__global__ void amp_gdn_gates_kernel(const __half *__restrict__ a,
                                    const __half *__restrict__ b,
                                    const float *__restrict__ dt_bias,
                                    const float *__restrict__ nea,
                                    float *__restrict__ beta,
                                    float *__restrict__ gexp, int n, int Hv)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	int h = i % Hv;
	float bv = __half2float(b[i]);
	float av = __half2float(a[i]) + dt_bias[h];
	float sp = (av > 20.0f) ? av : log1pf(__expf(av));   // torch softplus threshold

	beta[i] = 1.0f / (1.0f + __expf(-bv));
	gexp[i] = __expf(nea[h] * sp);
}

extern "C" int
amp_gdn_gates(const void *a, const void *b, const void *dt_bias,
             const void *nea, void *beta, void *gexp, int S, int Hv,
             cudaStream_t stream)
{
	int n = S * Hv;

	if (n <= 0)
		return 0;

	amp_gdn_gates_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
		(const __half *)a, (const __half *)b, (const float *)dt_bias,
		(const float *)nea, (float *)beta, (float *)gexp, n, Hv);

	return (int)cudaGetLastError();
}

// GDN gates straight from the hidden state: the in_proj_a / in_proj_b
// products (fp16 weights, fp32 accumulation, rounded to fp16 like the
// matmul they replace) and the gate math of amp_gdn_gates_kernel in one
// launch, for S <= AMP_ABG_MAXS rows. One block per v head: it reads that
// head's two weight rows once and serves every row.
//   x (S, K) fp16; wa, wb (Hv, K) fp16; beta, gexp (S, Hv) fp32 out
#define AMP_ABG_MAXS 8
#define AMP_ABG_THREADS 512

__global__ void amp_gdn_ab_gates_kernel(const __half *__restrict__ x,
                                       const __half *__restrict__ wa,
                                       const __half *__restrict__ wb,
                                       const float *__restrict__ dt_bias,
                                       const float *__restrict__ nea,
                                       float *__restrict__ beta,
                                       float *__restrict__ gexp,
                                       int S, int K, int Hv)
{
	const int h = blockIdx.x;
	const __half2 *a2 = (const __half2 *)(wa + (size_t)h * K);
	const __half2 *b2 = (const __half2 *)(wb + (size_t)h * K);
	const __half2 *x2 = (const __half2 *)x;
	const int K2 = K / 2;

	float accA[AMP_ABG_MAXS], accB[AMP_ABG_MAXS];

	#pragma unroll
	for (int s = 0; s < AMP_ABG_MAXS; s++) {
		accA[s] = 0.0f;
		accB[s] = 0.0f;
	}

	for (int k = threadIdx.x; k < K2; k += AMP_ABG_THREADS) {
		float2 wav = __half22float2(a2[k]);
		float2 wbv = __half22float2(b2[k]);

		#pragma unroll
		for (int s = 0; s < AMP_ABG_MAXS; s++) {
			if (s < S) {
				float2 xv = __half22float2(x2[(size_t)s * K2 + k]);

				accA[s] = fmaf(xv.x, wav.x, fmaf(xv.y, wav.y, accA[s]));
				accB[s] = fmaf(xv.x, wbv.x, fmaf(xv.y, wbv.y, accB[s]));
			}
		}
	}

	__shared__ float red[2][AMP_ABG_MAXS][AMP_ABG_THREADS / 32];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

	#pragma unroll
	for (int s = 0; s < AMP_ABG_MAXS; s++) {
		float va = accA[s], vb = accB[s];

		#pragma unroll
		for (int o = 16; o > 0; o >>= 1) {
			va += __shfl_xor_sync(0xffffffffu, va, o);
			vb += __shfl_xor_sync(0xffffffffu, vb, o);
		}

		if (lane == 0) {
			red[0][s][warp] = va;
			red[1][s][warp] = vb;
		}
	}

	__syncthreads();

	if (threadIdx.x < S) {
		const int s = threadIdx.x;
		float va = 0.0f, vb = 0.0f;

		#pragma unroll
		for (int w = 0; w < AMP_ABG_THREADS / 32; w++) {
			va += red[0][s][w];
			vb += red[1][s][w];
		}

		/* the unfused path rounds both products to fp16 */
		float bv = __half2float(__float2half(vb));
		float av = __half2float(__float2half(va)) + dt_bias[h];
		float sp = (av > 20.0f) ? av : log1pf(__expf(av));

		beta[(size_t)s * Hv + h] = 1.0f / (1.0f + __expf(-bv));
		gexp[(size_t)s * Hv + h] = __expf(nea[h] * sp);
	}
}

extern "C" int
amp_gdn_ab_gates(const void *x, const void *wa, const void *wb,
                const void *dt_bias, const void *nea, void *beta, void *gexp,
                int S, int K, int Hv, cudaStream_t stream)
{
	if (S < 1 || S > AMP_ABG_MAXS || (K & 1) || Hv < 1) {
		fprintf(stderr, "amp_gdn_ab_gates: unsupported S %d K %d Hv %d\n", S, K, Hv);
		return -1;
	}

	amp_gdn_ab_gates_kernel<<<Hv, AMP_ABG_THREADS, 0, stream>>>(
		(const __half *)x, (const __half *)wa, (const __half *)wb,
		(const float *)dt_bias, (const float *)nea, (float *)beta,
		(float *)gexp, S, K, Hv);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// DFlash 2 drafter (docs/DFLASH.md)
//
// amp_dflash_conv: the two-tap grouped dynamic causal convolution of one
// draft block, fused: out[r, c] = (base[0][c] + dyn[r, 0, c / G]) * h[r, c]
//                              + (base[1][c] + dyn[r, 1, c / G]) * h[r - 1, c]
// (h[-1] = 0: taps never cross the block start). h (R, H) fp16; dyn: row
// stride `dstride` elements, [tap][group] contiguous from `dyn`; base
// (2, H) fp16 [tap][channel]. fp32 math, one rounding.
//
// amp_dflash_select: the candidate selector's walk. For position p, with
// predecessor q (the anchor for p = 0, else the token chosen at p - 1):
//   score(k) = unary[p, k] + < pred[q] * hp[p], succ[cand[p, k]] >
// and the chosen token is cand[p, argmax_k score(k)]. One block, the
// positions are sequential (each depends on the previous choice).
// ---------------------------------------------------------------------------

__global__ void amp_dflash_conv_kernel(const __half *__restrict__ h,
                                      const __half *__restrict__ dyn,
                                      const __half *__restrict__ base,
                                      __half *__restrict__ out, int H, int G,
                                      long dstride)
{
	const int r = blockIdx.x;
	const int groups = H / G;

	for (int c = threadIdx.x; c < H; c += blockDim.x) {
		const int g = c / G;
		float k0 = __half2float(base[c]) +
			__half2float(dyn[(size_t)r * dstride + g]);
		float k1 = __half2float(base[H + c]) +
			__half2float(dyn[(size_t)r * dstride + groups + g]);
		float v = k0 * __half2float(h[(size_t)r * H + c]);

		if (r > 0)
			v += k1 * __half2float(h[(size_t)(r - 1) * H + c]);

		out[(size_t)r * H + c] = __float2half(v);
	}
}

extern "C" int
amp_dflash_conv(const void *h, const void *dyn, const void *base, void *out,
               int R, int H, int G, long dstride, cudaStream_t stream)
{
	if (R <= 0)
		return 0;

	if (H <= 0 || G <= 0 || H % G != 0) {
		fprintf(stderr, "amp_dflash_conv: unsupported H %d / group %d\n", H, G);
		return -1;
	}

	amp_dflash_conv_kernel<<<R, 256, 0, stream>>>((const __half *)h,
		(const __half *)dyn, (const __half *)base, (__half *)out, H, G, dstride);

	return (int)cudaGetLastError();
}

// DFlash residual-stream kernels (fp32 stream: the drafter's attention /
// MLP outputs carry ~5e3 massive-activation channels that overflow fp16
// after a layer or two; its reference runs in bf16).
//   amp_rms_norm_f32in: out = fp16(x * rsqrt(mean(x^2) + eps)) * w
//   amp_dflash_conv_res: h += scale * conv2(y)   (h fp32 in place)
__global__ void amp_rms_norm_f32in_kernel(const float *__restrict__ x,
                                         const __half *__restrict__ w,
                                         __half *__restrict__ out, int dim,
                                         float eps)
{
	const int row = blockIdx.x;
	const float *xr = x + (size_t)row * dim;
	float acc = 0.0f;

	for (int i = threadIdx.x; i < dim; i += blockDim.x)
		acc += xr[i] * xr[i];

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_down_sync(0xffffffffu, acc, off);

	__shared__ float warp_sum[32];

	if ((threadIdx.x & 31) == 0)
		warp_sum[threadIdx.x >> 5] = acc;

	__syncthreads();

	if (threadIdx.x == 0) {
		float total = 0.0f;

		for (int i = 0; i < (int)(blockDim.x >> 5); i++)
			total += warp_sum[i];

		warp_sum[0] = rsqrtf(total / dim + eps);
	}

	__syncthreads();

	const float sc = warp_sum[0];

	for (int i = threadIdx.x; i < dim; i += blockDim.x)
		out[(size_t)row * dim + i] = __hmul(__float2half(xr[i] * sc), w[i]);
}

extern "C" int
amp_rms_norm_f32in(const void *x, const void *w, void *out, long rows, int dim,
                  double eps, cudaStream_t stream)
{
	if (rows <= 0 || dim <= 0)
		return 0;

	amp_rms_norm_f32in_kernel<<<(int)rows, 256, 0, stream>>>((const float *)x,
		(const __half *)w, (__half *)out, dim, (float)eps);

	return (int)cudaGetLastError();
}

__global__ void amp_dflash_conv_res_kernel(float *__restrict__ h,
                                          const __half *__restrict__ y,
                                          const __half *__restrict__ dyn,
                                          const __half *__restrict__ base,
                                          int H, int G, long dstride,
                                          float scale)
{
	const int r = blockIdx.x;
	const int groups = H / G;

	for (int c = threadIdx.x; c < H; c += blockDim.x) {
		const int g = c / G;
		float k0 = __half2float(base[c]) +
			__half2float(dyn[(size_t)r * dstride + g]);
		float k1 = __half2float(base[H + c]) +
			__half2float(dyn[(size_t)r * dstride + groups + g]);
		float v = k0 * __half2float(y[(size_t)r * H + c]);

		if (r > 0)
			v += k1 * __half2float(y[(size_t)(r - 1) * H + c]);

		h[(size_t)r * H + c] += scale * v;
	}
}

extern "C" int
amp_dflash_conv_res(void *h, const void *y, const void *dyn, const void *base,
                   int R, int H, int G, long dstride, float scale,
                   cudaStream_t stream)
{
	if (R <= 0)
		return 0;

	if (H <= 0 || G <= 0 || H % G != 0) {
		fprintf(stderr, "amp_dflash_conv_res: unsupported H %d / group %d\n", H, G);
		return -1;
	}

	amp_dflash_conv_res_kernel<<<R, 256, 0, stream>>>((float *)h,
		(const __half *)y, (const __half *)dyn, (const __half *)base, H, G,
		dstride, scale);

	return (int)cudaGetLastError();
}

#define AMP_DFS_THREADS 256
#define AMP_DFS_MAXK 32

__global__ void amp_dflash_select_kernel(const float *__restrict__ unary,
                                        const long *__restrict__ cand,
                                        const __half *__restrict__ hp,
                                        const long *__restrict__ anchor,
                                        const __half *__restrict__ pred,
                                        const __half *__restrict__ succ,
                                        long *__restrict__ out, int W, int K,
                                        int R)
{
	__shared__ float prod[1024];
	__shared__ float score[AMP_DFS_MAXK];
	__shared__ long prev;

	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int nwarp = blockDim.x >> 5;

	if (threadIdx.x == 0)
		prev = anchor[0];

	__syncthreads();

	for (int p = 0; p < W; p++) {
		for (int t = threadIdx.x; t < R; t += blockDim.x)
			prod[t] = __half2float(pred[(size_t)prev * R + t]) *
				__half2float(hp[(size_t)p * R + t]);

		__syncthreads();

		for (int k = warp; k < K; k += nwarp) {
			const __half *sr = succ + (size_t)cand[(size_t)p * K + k] * R;
			float d = 0.0f;

			for (int t = lane; t < R; t += 32)
				d += prod[t] * __half2float(sr[t]);

#pragma unroll
			for (int off = 16; off > 0; off >>= 1)
				d += __shfl_down_sync(0xffffffffu, d, off);

			if (lane == 0)
				score[k] = unary[(size_t)p * K + k] + d;
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			int best = 0;

			for (int k = 1; k < K; k++)
				if (score[k] > score[best])
					best = k;

			prev = cand[(size_t)p * K + best];
			out[p] = prev;
		}

		__syncthreads();
	}
}

extern "C" int
amp_dflash_select(const void *unary, const void *cand, const void *hp,
                 const void *anchor, const void *pred, const void *succ,
                 void *out, int W, int K, int R, cudaStream_t stream)
{
	if (W <= 0)
		return 0;

	if (K < 1 || K > AMP_DFS_MAXK || R < 1 || R > 1024) {
		fprintf(stderr, "amp_dflash_select: unsupported K %d / rank %d\n", K, R);
		return -1;
	}

	amp_dflash_select_kernel<<<1, AMP_DFS_THREADS, 0, stream>>>(
		(const float *)unary, (const long *)cand, (const __half *)hp,
		(const long *)anchor, (const __half *)pred, (const __half *)succ,
		(long *)out, W, K, R);

	return (int)cudaGetLastError();
}

// GDN output: gated RMSNorm over the v head dim, then * silu(z):
//   n   = fp16(core * rsqrt(mean(core^2) + eps)) * w     (fp16 multiply)
//   out = n * fp16(silu(z))
//   core (S, Hv, dv) fp32; z (S, Hv * dv) fp16; w (dv) fp16;
//   out (S, Hv * dv) fp16. Block per (token, head), dv threads.
// Rounding points mirror the composite ATen path it replaces.
__global__ void amp_gdn_out_norm_kernel(const float *__restrict__ core,
                                       const __half *__restrict__ z,
                                       const __half *__restrict__ w,
                                       __half *__restrict__ out, int dv,
                                       float eps)
{
	const size_t row = blockIdx.x;          // t * Hv + h
	const int e = threadIdx.x;
	float f = core[row * dv + e];
	float ss = f * f;

	__shared__ float red[32];

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		ss += __shfl_down_sync(0xffffffffu, ss, off);

	if ((e & 31) == 0)
		red[e >> 5] = ss;

	__syncthreads();

	if (e < 32) {
		float s2 = (e < (int)(blockDim.x >> 5)) ? red[e] : 0.0f;

#pragma unroll
		for (int off = 16; off > 0; off >>= 1)
			s2 += __shfl_down_sync(0xffffffffu, s2, off);

		if (e == 0)
			red[0] = rsqrtf(s2 / dv + eps);
	}

	__syncthreads();

	__half n = __hmul(__float2half(f * red[0]), w[e]);
	float zf = __half2float(z[row * dv + e]);
	__half sz = __float2half(zf / (1.0f + __expf(-zf)));

	out[row * dv + e] = __hmul(n, sz);
}

extern "C" int
amp_gdn_out_norm(const void *core, const void *z, const void *w, void *out,
                long rows, int dv, float eps, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	if (dv > 1024 || dv % 32 != 0) {
		fprintf(stderr, "amp_gdn_out_norm: unsupported dv %d\n", dv);
		return -1;
	}

	amp_gdn_out_norm_kernel<<<(unsigned)rows, dv, 0, stream>>>(
		(const float *)core, (const __half *)z, (const __half *)w,
		(__half *)out, dv, eps);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Chunked gated delta rule (prefill): same math as amp_gdn_scan, with the
// sequential token loop replaced by C-token blocks (the WY/UT form used
// by flash-linear-attention). Per head, with lg_t = log(gexp_t), block-
// local cumulative decay G_c = sum_{c'<=c} lg and Gamma_c = exp(G_c),
// the state before the block S0 (dk x dv) and u_t the update vectors:
//
//   S_c   = Gamma_c S0 + sum_{j<=c} (Gamma_c / Gamma_j) k_j u_j^T
//   (I + A) U = diag(beta) V - diag(beta Gamma) K S0,
//       A[c][j] = beta_c (Gamma_c / Gamma_j) (k_c . k_j)   (j < c)
//   T = (I + A)^-1,  Ut = T diag(beta) V,  W = T diag(beta Gamma) K
//   U = Ut - W S0
//   O = scale (diag(Gamma) Q S0 + M U),
//       M[c][j] = (Gamma_c / Gamma_j) (q_c . k_j)          (j <= c)
//   S_end = Gamma_end S0 + Kd^T U,  Kd[c] = (Gamma_end / Gamma_c) k_c
//
// Kernel A (parallel over head x block) computes Ut, W, Qg = diag(Gamma) Q,
// Kd, M and Gamma_end into a workspace; kernel B (per head x column
// slice, sequential over blocks) runs the S0-dependent part. Everything
// is fp32. dk == dv == 128 only (Qwen3.5); other shapes use amp_gdn_scan.
// ---------------------------------------------------------------------------

#define AMP_GDC_C 64          // tokens per block
#define AMP_GDC_D 128         // head dim (dk == dv)
#define AMP_GDC_DP (AMP_GDC_D + 1)
#define AMP_GDC_CP (AMP_GDC_C + 1)
#define AMP_GDC_SL 64         // kernel B: state columns per block

/* workspace floats per (head, block) */
#define AMP_GDC_WS (AMP_GDC_C * AMP_GDC_D * 4 + AMP_GDC_C * AMP_GDC_C + 4)

__global__ __launch_bounds__(256)
void amp_gdc_prep_kernel(const float *__restrict__ q, const float *__restrict__ k,
                        const float *__restrict__ v, const float *__restrict__ beta,
                        const float *__restrict__ gexp, float *__restrict__ ws,
                        int S, int Hk, int Hv, int nb)
{
	constexpr int C = AMP_GDC_C, D = AMP_GDC_D, DP = AMP_GDC_DP, CP = AMP_GDC_CP;

	extern __shared__ float sm[];
	float *sK = sm;                     // C x DP
	float *sQ = sK + C * DP;            // C x DP (Q, later V)
	float *sA = sQ + C * DP;            // C x CP
	float *sT = sA + C * CP;            // C x CP
	float *sG = sT + C * CP;            // C  cumulative log decay
	float *sB = sG + C;                 // C  beta

	const int h = blockIdx.x / nb;
	const int b = blockIdx.x % nb;
	const int hk = h / (Hv / Hk);
	const int t0 = b * C;
	const int n = min(C, S - t0);
	const int tid = threadIdx.x;

	float *w_ut = ws + (size_t)blockIdx.x * AMP_GDC_WS;   // Ut  C x D
	float *w_w = w_ut + C * D;                           // W   C x D
	float *w_qg = w_w + C * D;                           // Qg  C x D
	float *w_kd = w_qg + C * D;                          // Kd  C x D
	float *w_m = w_kd + C * D;                           // M   C x C
	float *w_ge = w_m + C * C;                           // Gamma_end

	for (int e = tid; e < C * D; e += blockDim.x) {
		int c = e / D, i = e % D;
		bool ok = c < n;

		sK[c * DP + i] = ok ? k[((size_t)(t0 + c) * Hk + hk) * D + i] : 0.0f;
		sQ[c * DP + i] = ok ? q[((size_t)(t0 + c) * Hk + hk) * D + i] : 0.0f;
	}

	for (int c = tid; c < C; c += blockDim.x) {
		bool ok = c < n;

		sG[c] = ok ? __logf(gexp[(size_t)(t0 + c) * Hv + h]) : 0.0f;
		sB[c] = ok ? beta[(size_t)(t0 + c) * Hv + h] : 0.0f;
	}

	__syncthreads();

	if (tid == 0) {
		float acc = 0.0f;

		for (int c = 0; c < C; c++) {
			acc += sG[c];
			sG[c] = acc;
		}
	}

	__syncthreads();

	/* A (strictly lower) and M (lower incl. diagonal) */
	for (int e = tid; e < C * C; e += blockDim.x) {
		int c = e / C, j = e % C;
		float kk = 0.0f, qk = 0.0f;

		if (j <= c) {
#pragma unroll 8
			for (int i = 0; i < D; i++) {
				float kj = sK[j * DP + i];

				kk = fmaf(sK[c * DP + i], kj, kk);
				qk = fmaf(sQ[c * DP + i], kj, qk);
			}
		}

		float dec = (j <= c) ? __expf(sG[c] - sG[j]) : 0.0f;

		sA[c * CP + j] = (j < c) ? sB[c] * dec * kk : 0.0f;
		w_m[c * C + j] = dec * qk;
	}

	__syncthreads();

	/* Qg = diag(Gamma) Q, Kd = diag(Gamma_end / Gamma) K (global) */
	const float g_end = sG[C - 1];

	for (int e = tid; e < C * D; e += blockDim.x) {
		int c = e / D, i = e % D;

		w_qg[e] = __expf(sG[c]) * sQ[c * DP + i];
		w_kd[e] = __expf(g_end - sG[c]) * sK[c * DP + i];
	}

	if (tid == 0)
		w_ge[0] = __expf(g_end);

	/* T = (I + A)^-1, unit lower triangular, row by row:
	 * T[c][j] = -sum_{m=j}^{c-1} A[c][m] T[m][j]  (j < c) */
	for (int c = 0; c < C; c++) {
		if (tid < C) {
			int j = tid;
			float acc = 0.0f;

			if (j < c) {
				for (int m = j; m < c; m++)
					acc = fmaf(sA[c * CP + m], sT[m * CP + j], acc);

				acc = -acc;
			}
			else {
				acc = (j == c) ? 1.0f : 0.0f;
			}

			sT[c * CP + j] = acc;
		}

		__syncthreads();
	}

	/* V into the Q buffer (Q is fully consumed above) */
	for (int e = tid; e < C * D; e += blockDim.x) {
		int c = e / D, i = e % D;

		sQ[c * DP + i] = (c < n) ? v[((size_t)(t0 + c) * Hv + h) * D + i] : 0.0f;
	}

	__syncthreads();

	/* fold the per-token factors into T once: Tb = T diag(beta),
	 * Tbg = T diag(beta Gamma) (an expf per inner-loop step was the
	 * dominant cost) */
	for (int e = tid; e < C * C; e += blockDim.x) {
		int c = e / C, j = e % C;
		float tb = sT[c * CP + j] * sB[j];

		sT[c * CP + j] = tb;
		sA[c * CP + j] = tb * __expf(sG[j]);
	}

	__syncthreads();

	/* W = T diag(beta Gamma) K,  Ut = T diag(beta) V */
	for (int e = tid; e < C * D; e += blockDim.x) {
		int c = e / D, i = e % D;
		float aw = 0.0f, au = 0.0f;

		for (int j = 0; j <= c; j++) {
			aw = fmaf(sA[c * CP + j], sK[j * DP + i], aw);
			au = fmaf(sT[c * CP + j], sQ[j * DP + i], au);
		}

		w_w[e] = aw;
		w_ut[e] = au;
	}
}

__global__ __launch_bounds__(256, 2)
void amp_gdc_scan_kernel(const float *__restrict__ ws, float *__restrict__ state,
                        float *__restrict__ out, int S, int Hv, int nb,
                        float scale)
{
	constexpr int C = AMP_GDC_C, D = AMP_GDC_D, SL = AMP_GDC_SL;
	constexpr int NG = 256 / SL;          // thread groups (4)
	constexpr int CPG = C / NG;           // rows c per group (16)
	constexpr int RPG = D / NG;           // state rows i per group (32)

	/* only the state slice and U live in shared memory (48 KiB, two
	 * blocks per SM: all Hv * D/SL blocks run in ONE wave — a second wave
	 * doubles this kernel's time, it is sequential over the blocks). W,
	 * Qg, Kd, M are read straight from global memory: every warp reads
	 * the same element at a time (broadcast), so they hit L1/L2. */
	extern __shared__ float sm[];
	float *sS = sm;                       // D x SL  state slice
	float *sU = sS + D * SL;              // C x SL

	const int nsl = D / SL;
	const int h = blockIdx.x / nsl;
	const int j0 = (blockIdx.x % nsl) * SL;
	const int tid = threadIdx.x;
	const int j = tid % SL;
	const int grp = tid / SL;

	float *st = state + (size_t)h * D * D;

	for (int e = tid; e < D * SL; e += blockDim.x)
		sS[e] = st[(size_t)(e / SL) * D + j0 + e % SL];

	__syncthreads();

	for (int b = 0; b < nb; b++) {
		const float *w = ws + ((size_t)h * nb + b) * AMP_GDC_WS;
		const float *__restrict__ w_ut = w;
		const float *__restrict__ w_w = w + C * D;
		const float *__restrict__ w_qg = w + 2 * C * D;
		const float *__restrict__ w_kd = w + 3 * C * D;
		const float *__restrict__ w_m = w + 4 * C * D;
		const float g_end = __ldg(w + 4 * C * D + C * C);
		const int t0 = b * C;
		const int n = min(C, S - t0);

		/* U = Ut - W S0 */
		{
			float acc[CPG];

#pragma unroll
			for (int r = 0; r < CPG; r++)
				acc[r] = __ldg(w_ut + (size_t)(grp * CPG + r) * D + j0 + j);

			for (int i = 0; i < D; i++) {
				float s0 = sS[i * SL + j];

#pragma unroll
				for (int r = 0; r < CPG; r++)
					acc[r] = fmaf(-__ldg(w_w + (grp * CPG + r) * D + i), s0, acc[r]);
			}

#pragma unroll
			for (int r = 0; r < CPG; r++)
				sU[(grp * CPG + r) * SL + j] = acc[r];
		}

		__syncthreads();

		/* O = scale (Qg S0 + M U) */
		{
			float acc[CPG];

#pragma unroll
			for (int r = 0; r < CPG; r++)
				acc[r] = 0.0f;

			for (int i = 0; i < D; i++) {
				float s0 = sS[i * SL + j];

#pragma unroll
				for (int r = 0; r < CPG; r++)
					acc[r] = fmaf(__ldg(w_qg + (grp * CPG + r) * D + i), s0, acc[r]);
			}

#pragma unroll
			for (int r = 0; r < CPG; r++) {
				int c = grp * CPG + r;

				for (int m = 0; m <= c; m++)
					acc[r] = fmaf(__ldg(w_m + c * C + m), sU[m * SL + j], acc[r]);

				if (c < n)
					out[((size_t)(t0 + c) * Hv + h) * D + j0 + j] = acc[r] * scale;
			}
		}

		/* S = Gamma_end S0 + Kd^T U (each thread updates its own
		 * elements; S0 reads above are done by all threads first) */
		__syncthreads();

		{
			float acc[RPG];

#pragma unroll
			for (int r = 0; r < RPG; r++)
				acc[r] = g_end * sS[(grp * RPG + r) * SL + j];

			for (int c = 0; c < C; c++) {
				float u = sU[c * SL + j];

#pragma unroll
				for (int r = 0; r < RPG; r++)
					acc[r] = fmaf(__ldg(w_kd + c * D + grp * RPG + r), u, acc[r]);
			}

#pragma unroll
			for (int r = 0; r < RPG; r++)
				sS[(grp * RPG + r) * SL + j] = acc[r];
		}

		__syncthreads();
	}

	for (int e = tid; e < D * SL; e += blockDim.x)
		st[(size_t)(e / SL) * D + j0 + e % SL] = sS[e];
}

/* workspace floats needed for S tokens and Hv heads */
extern "C" long
amp_gdn_chunk_ws_floats(int S, int Hv)
{
	long nb = (S + AMP_GDC_C - 1) / AMP_GDC_C;

	return (long)Hv * nb * AMP_GDC_WS;
}

extern "C" int
amp_gdn_chunk_f32(const void *q, const void *k, const void *v, const void *beta,
                 const void *gexp, void *state, void *out, void *ws, int S,
                 int Hk, int Hv, int dk, int dv, float scale,
                 cudaStream_t stream)
{
	constexpr int C = AMP_GDC_C, D = AMP_GDC_D, DP = AMP_GDC_DP, CP = AMP_GDC_CP;

	if (S <= 0)
		return 0;

	if (dk != D || dv != D || Hk <= 0 || Hv % Hk != 0) {
		fprintf(stderr, "amp_gdn_chunk: unsupported dims (dk %d, dv %d, Hk %d, Hv %d)\n",
			dk, dv, Hk, Hv);
		return -1;
	}

	int nb = (S + C - 1) / C;
	size_t smem_a = (size_t)(2 * C * DP + 2 * C * CP + 2 * C) * sizeof(float);
	size_t smem_b = (size_t)(D * AMP_GDC_SL + C * AMP_GDC_SL) * sizeof(float);

	/* the >48 KiB dynamic shared memory opt-in is per device */
	static bool attr_set[64];
	int dev = 0;

	cudaGetDevice(&dev);

	if (dev >= 0 && dev < 64 && !attr_set[dev]) {
		cudaFuncSetAttribute(amp_gdc_prep_kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_a);
		cudaFuncSetAttribute(amp_gdc_scan_kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_b);
		attr_set[dev] = true;
	}

	amp_gdc_prep_kernel<<<Hv * nb, 256, smem_a, stream>>>(
		(const float *)q, (const float *)k, (const float *)v,
		(const float *)beta, (const float *)gexp, (float *)ws, S, Hk, Hv, nb);

	amp_gdc_scan_kernel<<<Hv * (D / AMP_GDC_SL), 256, smem_b, stream>>>(
		(const float *)ws, (float *)state, (float *)out, S, Hv, nb, scale);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Attention-layer fusions (qwen35 full-attention layers).
//
// amp_qk_norm_rope: per-head RMSNorm (fp32 variance, fp16 cast, then the
// fp16 weight multiply — the composite's rounding points) followed by
// partial NEOX RoPE on the first rotary_dim channels (fp32 math on the
// normed fp16 values), one block per (token, head), head_dim threads.
//   x    strided: element (t, h, e) at x[t * st + h * sh + e]
//   w    (hd) fp16 (weight + 1)
//   cos, sin (S, rotary_dim / 2) fp32 contiguous rows
//   out  (S, H, hd) fp16 contiguous
//
// amp_attn_gate: out[t][h][e] = attn(t,h,e) * fp16(sigmoid(g(t,h,e))),
// both inputs strided, out (S, H * hd) contiguous — replaces the
// transpose-copy + sigmoid + multiply after flash attention.
// ---------------------------------------------------------------------------

__global__ void amp_qk_norm_rope_kernel(const __half *__restrict__ x, long st,
                                       long sh, const __half *__restrict__ w,
                                       const float *__restrict__ cos,
                                       const float *__restrict__ sin,
                                       __half *__restrict__ out, int H, int hd,
                                       int rd, float eps)
{
	const int t = blockIdx.x / H;
	const int h = blockIdx.x % H;
	const int e = threadIdx.x;
	const int half = rd / 2;

	__shared__ float red[32];
	__shared__ __half nrm[1024];

	float v = __half2float(x[(size_t)t * st + (size_t)h * sh + e]);
	float ss = v * v;

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		ss += __shfl_down_sync(0xffffffffu, ss, off);

	if ((e & 31) == 0)
		red[e >> 5] = ss;

	__syncthreads();

	if (e < 32) {
		float s2 = (e < (int)(blockDim.x >> 5)) ? red[e] : 0.0f;

#pragma unroll
		for (int off = 16; off > 0; off >>= 1)
			s2 += __shfl_down_sync(0xffffffffu, s2, off);

		if (e == 0)
			red[0] = rsqrtf(s2 / hd + eps);
	}

	__syncthreads();

	__half n = __hmul(__float2half(v * red[0]), w[e]);

	nrm[e] = n;

	__syncthreads();

	__half *o = out + ((size_t)t * H + h) * hd;

	if (e < half) {
		float x1 = __half2float(nrm[e]), x2 = __half2float(nrm[e + half]);
		float c = cos[(size_t)t * half + e], s = sin[(size_t)t * half + e];

		o[e] = __float2half(x1 * c - x2 * s);
		o[e + half] = __float2half(x2 * c + x1 * s);
	}
	else if (e >= rd) {
		o[e] = n;
	}
}

extern "C" int
amp_qk_norm_rope_fp16(const void *x, long st, long sh, const void *w,
                     const void *cos, const void *sin, void *out, int S, int H,
                     int hd, int rd, float eps, cudaStream_t stream)
{
	if (S <= 0)
		return 0;

	if (hd > 1024 || hd % 32 != 0 || rd > hd || rd % 2 != 0) {
		fprintf(stderr, "amp_qk_norm_rope: unsupported dims (hd %d, rotary %d)\n", hd, rd);
		return -1;
	}

	amp_qk_norm_rope_kernel<<<S * H, hd, 0, stream>>>((const __half *)x, st, sh,
		(const __half *)w, (const float *)cos, (const float *)sin,
		(__half *)out, H, hd, rd, eps);

	return (int)cudaGetLastError();
}

__global__ void amp_attn_gate_kernel(const __half *__restrict__ a, long ast,
                                    long ash, const __half *__restrict__ g,
                                    long gst, long gsh, __half *__restrict__ out,
                                    int H, int hd, long n)
{
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	int e = (int)(i % hd);
	long th = i / hd;
	int h = (int)(th % H);
	long t = th / H;

	float gv = __half2float(g[t * gst + h * gsh + e]);
	__half sg = __float2half(1.0f / (1.0f + __expf(-gv)));

	out[i] = __hmul(a[t * ast + h * ash + e], sg);
}

extern "C" int
amp_attn_gate_fp16(const void *a, long ast, long ash, const void *g, long gst,
                  long gsh, void *out, int S, int H, int hd, cudaStream_t stream)
{
	long n = (long)S * H * hd;

	if (n <= 0)
		return 0;

	amp_attn_gate_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
		(const __half *)a, ast, ash, (const __half *)g, gst, gsh,
		(__half *)out, H, hd, n);

	return (int)cudaGetLastError();
}

/*
 * N-way all-reduce over PCIe peer-to-peer (tensor parallelism over N
 * GPUs, N <= AMP_COMM_MAXN). Every rank runs this kernel on its own
 * stream with its partial x (n fp16 values, n % 8 == 0); all ranks must
 * issue the same sequence of calls. Two schemes:
 *
 *   one-shot  (mode 0): rank r stores x into slot [seq & 1][r] of every
 *             peer's buffer A, flags them, waits for all peers' flags,
 *             sums the N slots. Per rank: (N-1) x n sent. Best for N <= 3.
 *   two-phase (mode 1): reduce-scatter + all-gather. The payload is cut
 *             into N chunks (8-value units, as even as possible); rank r
 *             sends chunk c of its partial to rank c (A[slot][r]), rank c
 *             sums chunk c and stores the result into every rank's
 *             gather buffer G[slot] (flag phase 2), everyone copies G to
 *             out. Per rank: 2 (N-1)/N x n sent. Measured on 6 A5000s on
 *             PCIe 3.0 x4: 28 us vs 46 us (one-shot) for 4096 fp16.
 *
 * Sums accumulate in fp32 in rank order and round once, so every rank
 * gets bit-identical values (and for N = 2 the same values as a single
 * correctly rounded fp16 add). Blocks split each chunk further, each
 * with its own flags and counter: ctl = [phase-1 flags MAXB x MAXN |
 * phase-2 flags MAXB x MAXN | counters MAXB]. Two slots suffice because
 * a rank cannot start call seq+2 before every peer has finished seq+1
 * (see amp_p2p_allreduce_kernel). Spins trap after ~2^33 clocks.
 */
#define AMP_COMM_MAXN 8
#define AMP_COMM_MAXB 16

struct amp_comm_peers {
	__half *buf[AMP_COMM_MAXN];   /* per rank: A [2][N][cap] then G [2][cap] */
	int *ctl[AMP_COMM_MAXN];
};

/* [u0, u1) of a [0, units) range cut into parts pieces, piece i */
__device__ __forceinline__ void
amp_comm_part(long units, int parts, int i, long *u0, long *u1)
{
	*u0 = units * i / parts;
	*u1 = units * (i + 1) / parts;
}

__device__ __forceinline__ void
amp_comm_acc(float *acc, uint4 v)
{
	const __half2 *h = (const __half2 *)&v;

	for (int k = 0; k < 4; k++) {
		float2 f = __half22float2(h[k]);

		acc[2 * k] += f.x;
		acc[2 * k + 1] += f.y;
	}
}

__device__ __forceinline__ uint4
amp_comm_pack(const float *acc)
{
	uint4 o;
	__half2 *h = (__half2 *)&o;

	for (int k = 0; k < 4; k++)
		h[k] = __floats2half2_rn(acc[2 * k], acc[2 * k + 1]);

	return o;
}

/* Post this block's flag to every peer and wait for every peer's, in
 * parallel: warp 0, lane p handles peer p (posted atomics, acquire polls
 * of the LOCAL flag array). One block-wide barrier before (all stores of
 * the block are issued) and after; one system fence each way by lane 0.
 * post: offset (in ints) into a peer's ctl at which this rank's flag
 * lives; wait: offset into the local ctl where peer p's flag is. */
__device__ __forceinline__ void
amp_comm_sync(const amp_comm_peers &P, int rank, int N, int post, int wait, int seq)
{
	/* every thread releases ITS OWN remote stores (a fence by one thread
	 * does not flush other warps' stores in hardware), then the block
	 * meets, then warp 0 posts and waits in parallel */
	__threadfence_system();
	__syncthreads();

	if (threadIdx.x < 32) {
		int p = threadIdx.x;
		bool peer = (p < N && p != rank);

		if (peer)
			atomicMax_system(&P.ctl[p][post + rank], seq);

		if (peer) {
			const volatile int *f = (const volatile int *)&P.ctl[rank][wait + p];
			long long t0 = clock64();

			while (*f < seq)
				if (clock64() - t0 > (1LL << 33))
					__trap();
		}

		__syncwarp();

		if (p == 0)
			__threadfence_system();      // acquire: peers' data is visible
	}

	__syncthreads();
}

__global__ void
amp_comm_allreduce_kernel(const __half *__restrict__ x, __half *__restrict__ out,
                         amp_comm_peers P, int rank, int N, long n, long cap, int mode)
{
	__shared__ int s_seq;
	int b = blockIdx.x, nb = gridDim.x;
	int *ctl = P.ctl[rank];
	int *f1 = ctl, *f2 = ctl + AMP_COMM_MAXB * AMP_COMM_MAXN;
	int *ctr = ctl + 2 * AMP_COMM_MAXB * AMP_COMM_MAXN;

	if (threadIdx.x == 0)
		s_seq = *(volatile int *)&ctr[b] + 1;

	__syncthreads();

	int seq = s_seq, sl = seq & 1;
	long units = n / 8;
	const uint4 *xv = (const uint4 *)x;
	uint4 *ov = (uint4 *)out;

	/* A slot of rank p's buffer for sender s, G slot of rank p */
	#define A_OF(p, s) ((uint4 *)(P.buf[p] + ((long)sl * N + (s)) * cap))
	#define G_OF(p) ((uint4 *)(P.buf[p] + (2L * N + sl) * cap))

	if (mode == 0) {
		long u0, u1;

		amp_comm_part(units, nb, b, &u0, &u1);

		for (int p = 0; p < N; p++) {
			if (p == rank)
				continue;

			uint4 *dst = A_OF(p, rank);

			for (long i = u0 + threadIdx.x; i < u1; i += blockDim.x)
				dst[i] = xv[i];
		}

		amp_comm_sync(P, rank, N, b * AMP_COMM_MAXN, b * AMP_COMM_MAXN, seq);

		for (long i = u0 + threadIdx.x; i < u1; i += blockDim.x) {
			float acc[8] = { 0 };

			for (int p = 0; p < N; p++)
				amp_comm_acc(acc, (p == rank) ? xv[i] : __ldcg(&A_OF(rank, p)[i]));

			ov[i] = amp_comm_pack(acc);
		}
	}
	else {
		/* phase 1: my slice of chunk c goes to its owner c */
		for (int c = 0; c < N; c++) {
			if (c == rank)
				continue;

			long c0, c1, u0, u1;

			amp_comm_part(units, N, c, &c0, &c1);
			amp_comm_part(c1 - c0, nb, b, &u0, &u1);

			uint4 *dst = A_OF(c, rank);

			for (long i = c0 + u0 + threadIdx.x; i < c0 + u1; i += blockDim.x)
				dst[i] = xv[i];
		}

		amp_comm_sync(P, rank, N, b * AMP_COMM_MAXN, b * AMP_COMM_MAXN, seq);

		/* reduce my chunk slice, store it into every rank's G */
		long c0, c1, u0, u1;

		amp_comm_part(units, N, rank, &c0, &c1);
		amp_comm_part(c1 - c0, nb, b, &u0, &u1);

		for (long i = c0 + u0 + threadIdx.x; i < c0 + u1; i += blockDim.x) {
			float acc[8] = { 0 };

			for (int p = 0; p < N; p++)
				amp_comm_acc(acc, (p == rank) ? xv[i] : __ldcg(&A_OF(rank, p)[i]));

			uint4 o = amp_comm_pack(acc);

			for (int p = 0; p < N; p++)
				if (p != rank)
					G_OF(p)[i] = o;

			ov[i] = o;
		}

		amp_comm_sync(P, rank, N, (AMP_COMM_MAXB + b) * AMP_COMM_MAXN,
			(AMP_COMM_MAXB + b) * AMP_COMM_MAXN, seq);

		/* gather: the other owners' chunk slices */
		for (int c = 0; c < N; c++) {
			if (c == rank)
				continue;

			amp_comm_part(units, N, c, &c0, &c1);
			amp_comm_part(c1 - c0, nb, b, &u0, &u1);

			for (long i = c0 + u0 + threadIdx.x; i < c0 + u1; i += blockDim.x)
				ov[i] = __ldcg(&G_OF(rank)[i]);
		}
	}

	#undef A_OF
	#undef G_OF

	__syncthreads();

	if (threadIdx.x == 0)
		*(volatile int *)&ctr[b] = seq;
}

extern "C" int
amp_comm_allreduce_fp16(const void *x, void *out, void *const *bufs,
                       int *const *ctls, int rank, int N, long n, long cap,
                       int mode, int blocks, cudaStream_t stream)
{
	if (N < 2 || N > AMP_COMM_MAXN || n <= 0 || n % 8 || n > cap ||
	    blocks < 1 || blocks > AMP_COMM_MAXB)
		return (int)cudaErrorInvalidValue;

	amp_comm_peers P;

	for (int p = 0; p < N; p++) {
		P.buf[p] = (__half *)bufs[p];
		P.ctl[p] = ctls[p];
	}

	amp_comm_allreduce_kernel<<<blocks, 512, 0, stream>>>((const __half *)x,
		(__half *)out, P, rank, N, n, cap, mode);

	return (int)cudaGetLastError();
}

/*
 * Batched contiguous copies on one device: up to AMP_CM_MAX (src, dst,
 * bytes) entries in one launch (the speculative-decoding rollback copies
 * every GDN layer's recurrent + conv state: 96 entries per GPU under
 * 6-way TP, otherwise 96 launches). Bytes must be multiples of 16 and
 * pointers 16-byte aligned (fp32/fp16 tensors from the caching
 * allocator). Grid (blocks per entry, entries).
 */
#define AMP_CM_MAX 128

struct amp_cm_tab {
	const void *src[AMP_CM_MAX];
	void *dst[AMP_CM_MAX];
	long n16[AMP_CM_MAX];
};

__global__ void
amp_copy_many_kernel(amp_cm_tab tab)
{
	const int e = blockIdx.y;
	const uint4 *s = (const uint4 *)tab.src[e];
	uint4 *d = (uint4 *)tab.dst[e];
	long n = tab.n16[e];

	for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += (long)gridDim.x * blockDim.x)
		d[i] = s[i];
}

extern "C" int
amp_copy_many(const void *const *src, void *const *dst, const long *bytes,
             int n, cudaStream_t stream)
{
	for (int base = 0; base < n; base += AMP_CM_MAX) {
		amp_cm_tab tab;
		int m = (n - base < AMP_CM_MAX) ? n - base : AMP_CM_MAX;
		long most = 0;

		for (int i = 0; i < m; i++) {
			if (bytes[base + i] % 16 || ((uintptr_t)src[base + i] | (uintptr_t)dst[base + i]) % 16)
				return (int)cudaErrorInvalidValue;

			tab.src[i] = src[base + i];
			tab.dst[i] = dst[base + i];
			tab.n16[i] = bytes[base + i] / 16;
			most = (tab.n16[i] > most) ? tab.n16[i] : most;
		}

		int bpe = (int)((most + 256 * 4 - 1) / (256 * 4));

		bpe = (bpe < 1) ? 1 : (bpe > 64 ? 64 : bpe);
		amp_copy_many_kernel<<<dim3(bpe, m), 256, 0, stream>>>(tab);

		cudaError_t rc = cudaGetLastError();

		if (rc != cudaSuccess)
			return (int)rc;
	}

	return 0;
}

/*
 * Split-KV decode attention over a KV cache with a DEVICE-side key
 * count (CUDA-graph-safe), for S <= 5 query rows (plain decode, MTP
 * verify windows) with GQA:
 *
 *   q        (S, Hq, E) fp16, contiguous (normed + roped)
 *   k/v      (Hk, max_seq, E) fp16 caches, rows contiguous
 *   len      device int32: keys in use; query row s sits at absolute
 *            position len - S + s and attends keys [0, len - S + s]
 *   out      (S, Hq, E) fp16
 *
 * Grid (Hk, NS): block (j, split) covers keys [split * chunk, ...) with
 * chunk = ceil(len / NS) from the device length — every block has work
 * at every context length (FA2's varlen entry with seqused_k sizes its
 * work by the cache CAPACITY and does not split the keys at all for
 * S > 1: 13.6 ms per layer at 180k keys, S = 3, on a 3090). One warp per
 * (query head of kv head j, row s); lanes own E/32 consecutive dims
 * (16-byte loads at E = 256). The rep*S warps of a block read the same
 * K/V rows, so DRAM sees them about once. Online softmax over UN-key
 * batches; partials (m, l, acc) to ws; a combine kernel merges the
 * splits in fixed order (deterministic).
 */
#define AMP_AD_MAXS 5

/* Effective split count: a split needs at least AMP_AD_MINCHUNK keys to
 * pay for its block and its combine row. The workspace stays laid out for
 * the launch's `ns` splits; blocks beyond the effective count exit and the
 * combine does not read them, so a short context costs a handful of
 * splits instead of all of them. Derived from the device length by both
 * kernels, so they always agree. */
#define AMP_AD_MINCHUNK 128

/* splits per kv head: the combine keeps one weight per split in shared
 * memory */
#define AMP_AD_MAXNS 8192

__device__ __forceinline__ int amp_ad_nse(int len, int ns)
{
	int n = (len + AMP_AD_MINCHUNK - 1) / AMP_AD_MINCHUNK;

	return max(1, min(ns, n));
}


/* SR: the number of query rows S, a template parameter so the per-row
 * register arrays are sized exactly (a 5-row worst case cost the S = 1
 * kernel half its occupancy) */
template <int DPL, int SR>
__global__ void
amp_attn_dec_kernel(const __half *__restrict__ q, const __half *__restrict__ kc,
                   const __half *__restrict__ vc, const int *__restrict__ lenp,
                   float *__restrict__ ws, int S_unused, int Hq, int Hk, int max_seq,
                   int ns, float scale)
{
	constexpr int S = SR;

	constexpr int E = DPL * 32;
	const int j = blockIdx.x, sp = blockIdx.y;
	const int g = threadIdx.x >> 5, lane = threadIdx.x & 31;   // warp = query head in group
	const int rep = Hq / Hk;
	const int h = j * rep + g;
	const int wsrow = 4 + E;

	int len = *lenp;

	len = (len < 0) ? 0 : (len > max_seq ? max_seq : len);

	const int nse = amp_ad_nse(len, ns);

	if (sp >= nse)
		return;

	const int chunk = (len + nse - 1) / nse;
	const int begin = sp * chunk;
	const int end = min(len, begin + chunk);

	/* all S rows of query head h in this warp: one K/V load serves them
	 * all (row s attends keys < len - S + s + 1, causal) */
	float qr[SR][DPL], acc[SR][DPL], m[SR], l[SR];

#pragma unroll
	for (int s = 0; s < SR; s++) {
		m[s] = -1e30f;
		l[s] = 0.0f;

#pragma unroll
		for (int i = 0; i < DPL; i++) {
			acc[s][i] = 0.0f;
			qr[s][i] = 0.0f;
		}

		{
			const __half *qp = q + ((size_t)s * Hq + h) * E + lane * DPL;

#pragma unroll
			for (int i = 0; i < DPL; i += 2) {
				float2 f = __half22float2(*(const __half2 *)(qp + i));

				qr[s][i] = f.x * scale;
				qr[s][i + 1] = f.y * scale;
			}
		}
	}

	const __half *kb = kc + (size_t)j * max_seq * E + lane * DPL;
	const __half *vb = vc + (size_t)j * max_seq * E + lane * DPL;
	constexpr int UN = 4;

	for (int t0 = begin; t0 < end; t0 += UN) {
		__half2 kh[UN][DPL / 2], vh[UN][DPL / 2];

#pragma unroll
		for (int u = 0; u < UN; u++) {
			int t = t0 + u;
			bool ok = t < end;

			if (DPL == 8) {
				*(uint4 *)kh[u] = ok ? *(const uint4 *)(kb + (size_t)t * E) : make_uint4(0, 0, 0, 0);
				*(uint4 *)vh[u] = ok ? *(const uint4 *)(vb + (size_t)t * E) : make_uint4(0, 0, 0, 0);
			}
			else if (DPL == 4) {
				*(uint2 *)kh[u] = ok ? *(const uint2 *)(kb + (size_t)t * E) : make_uint2(0, 0);
				*(uint2 *)vh[u] = ok ? *(const uint2 *)(vb + (size_t)t * E) : make_uint2(0, 0);
			}
			else {
				*(uint *)kh[u] = ok ? *(const uint *)(kb + (size_t)t * E) : 0u;
				*(uint *)vh[u] = ok ? *(const uint *)(vb + (size_t)t * E) : 0u;
			}
		}

#pragma unroll
		for (int s = 0; s < SR; s++) {
			const int lim = len - S + s + 1;
			float sc[UN];

#pragma unroll
			for (int u = 0; u < UN; u++) {
				float part = 0.0f;

#pragma unroll
				for (int i = 0; i < DPL / 2; i++) {
					float2 kf = __half22float2(kh[u][i]);

					part = fmaf(qr[s][2 * i], kf.x, part);
					part = fmaf(qr[s][2 * i + 1], kf.y, part);
				}

#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
					part += __shfl_xor_sync(0xffffffffu, part, off);

				sc[u] = (t0 + u < end && t0 + u < lim) ? part : -1e30f;
			}

			float mn = m[s];

#pragma unroll
			for (int u = 0; u < UN; u++)
				mn = fmaxf(mn, sc[u]);

			float corr = __expf(m[s] - mn), ls = 0.0f;

#pragma unroll
			for (int i = 0; i < DPL; i++)
				acc[s][i] *= corr;

#pragma unroll
			for (int u = 0; u < UN; u++) {
				float e = __expf(sc[u] - mn);

				ls += e;

#pragma unroll
				for (int i = 0; i < DPL / 2; i++) {
					float2 vf = __half22float2(vh[u][i]);

					acc[s][2 * i] = fmaf(e, vf.x, acc[s][2 * i]);
					acc[s][2 * i + 1] = fmaf(e, vf.y, acc[s][2 * i + 1]);
				}
			}

			l[s] = l[s] * corr + ls;
			m[s] = mn;
		}
	}

	/* workspace rows in combine order: row = s * rep + g */
#pragma unroll
	for (int s = 0; s < SR; s++) {
		float *w = ws + (((size_t)j * ns + sp) * rep * S + s * rep + g) * wsrow;
		bool any = begin < min(end, len - S + s + 1);

		if (lane == 0) {
			w[0] = any ? m[s] : -1e30f;
			w[1] = any ? l[s] : 0.0f;
		}

#pragma unroll
		for (int i = 0; i < DPL; i++)
			w[4 + lane * DPL + i] = any ? acc[s][i] : 0.0f;
	}
}

/* Merge the splits: one block per (workspace row = s * rep + g, kv head),
 * E / 2 threads with two dims each. The splits' maxima and weights are
 * reduced across the block (the weights go to shared memory), then every
 * thread sums its dims over the splits in split order (deterministic).
 * One block per kv head with a warp per row walked all splits serially:
 * 170 us at 82k keys and four rows, ~40% of the decode attention (L132). */
template <int E>
__global__ void __launch_bounds__(E / 2)
amp_attn_dec_combine_kernel(const float *__restrict__ ws, __half *__restrict__ out,
                           int S, int Hq, int Hk, int ns,
                           const int *__restrict__ lenp, int max_seq)
{
	constexpr int T = E / 2, NW = T / 32;
	const int row = blockIdx.x, j = blockIdx.y;
	const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const int rep = Hq / Hk;
	const int g = row % rep, s = row / rep;
	const int h = j * rep + g;
	const size_t step = (size_t)rep * S * (4 + E);     /* next split's row */
	const float *w0 = ws + ((size_t)j * ns * rep * S + row) * (4 + E);

	extern __shared__ float amp_adc_e[];               /* ns split weights */
	__shared__ float red[NW];

	int len = *lenp;

	len = (len < 0) ? 0 : (len > max_seq ? max_seq : len);

	const int nse = amp_ad_nse(len, ns);

	float m = -1e30f;

	for (int sp = tid; sp < nse; sp += T)
		m = fmaxf(m, w0[sp * step]);

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

	if (lane == 0)
		red[warp] = m;

	__syncthreads();

	m = red[0];

	#pragma unroll
	for (int w = 1; w < NW; w++)
		m = fmaxf(m, red[w]);

	__syncthreads();

	float l = 0.0f;

	for (int sp = tid; sp < nse; sp += T) {
		float e = __expf(w0[sp * step] - m);

		amp_adc_e[sp] = e;
		l += w0[sp * step + 1] * e;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		l += __shfl_xor_sync(0xffffffffu, l, off);

	if (lane == 0)
		red[warp] = l;

	__syncthreads();

	l = red[0];

	#pragma unroll
	for (int w = 1; w < NW; w++)
		l += red[w];

	const float2 *wp = (const float2 *)(w0 + 4) + tid;
	const size_t step2 = step / 2;
	float a0 = 0.0f, a1 = 0.0f;

	#pragma unroll 8
	for (int sp = 0; sp < nse; sp++) {
		float2 v = wp[sp * step2];
		float e = amp_adc_e[sp];

		a0 = fmaf(v.x, e, a0);
		a1 = fmaf(v.y, e, a1);
	}

	float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
	__half *o = out + ((size_t)s * Hq + h) * E + 2 * tid;

	*(__half2 *)o = __floats2half2_rn(a0 * inv, a1 * inv);
}

/* the combine launch shared by both decode kernels */
static void
amp_attn_dec_combine(const float *ws, __half *out, int S, int Hq, int Hk, int E,
                    int ns, const int *len, int max_seq, cudaStream_t stream)
{
	dim3 grid((Hq / Hk) * S, Hk);
	size_t sm = (size_t)ns * sizeof(float);

	if (E == 256)
		amp_attn_dec_combine_kernel<256><<<grid, 128, sm, stream>>>(ws, out, S, Hq,
			Hk, ns, len, max_seq);
	else if (E == 128)
		amp_attn_dec_combine_kernel<128><<<grid, 64, sm, stream>>>(ws, out, S, Hq,
			Hk, ns, len, max_seq);
	else
		amp_attn_dec_combine_kernel<64><<<grid, 32, sm, stream>>>(ws, out, S, Hq,
			Hk, ns, len, max_seq);
}

extern "C" long
amp_attn_dec_ws_floats(int S, int Hq, int Hk, int E, int ns)
{
	return (long)Hk * ns * (Hq / Hk) * S * (4 + E);
}

extern "C" int
amp_attn_dec_fp16(const void *q, const void *k, const void *v, const int *len,
                 void *ws, void *out, int S, int Hq, int Hk, int E,
                 int max_seq, int ns, float scale, cudaStream_t stream)
{
	if (Hk <= 0 || Hq % Hk || S < 1 || S > AMP_AD_MAXS || Hq / Hk > 32 ||
	    (Hq / Hk) * S > 32 || ns < 1 || ns > AMP_AD_MAXNS ||
	    (E != 64 && E != 128 && E != 256)) {
		fprintf(stderr, "amp_attn_dec: unsupported config (S %d, Hq %d, Hk %d, E %d, ns %d)\n",
			S, Hq, Hk, E, ns);
		return -1;
	}

	dim3 grid(Hk, ns);
	int block = (Hq / Hk) * 32;            /* one warp per query head */

#define AMP_AD_K(D, SR) amp_attn_dec_kernel<D, SR><<<grid, block, 0, stream>>>( \
		(const __half *)q, (const __half *)k, (const __half *)v, len, (float *)ws, \
		S, Hq, Hk, max_seq, ns, scale)
#define AMP_AD_LAUNCH(D) do { \
	switch (S) { \
	case 1: AMP_AD_K(D, 1); break; \
	case 2: AMP_AD_K(D, 2); break; \
	case 3: AMP_AD_K(D, 3); break; \
	case 4: AMP_AD_K(D, 4); break; \
	default: AMP_AD_K(D, 5); break; \
	} } while (0)

	if (E == 256)
		AMP_AD_LAUNCH(8);
	else if (E == 128)
		AMP_AD_LAUNCH(4);
	else
		AMP_AD_LAUNCH(2);

	amp_attn_dec_combine((const float *)ws, (__half *)out, S, Hq, Hk, E, ns, len,
		max_seq, stream);

#undef AMP_AD_LAUNCH
#undef AMP_AD_K

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Kimi delta attention (KDA, GLM-5.3 linear-attention layers): the gated
// delta rule with a PER-KEY-CHANNEL decay. Sequential form, fp32, exact
// math of exllamav3's torch_recurrent_kda:
//
//   S    = diag(exp(g_t)) S                  (decay each key row i)
//   kv   = S^T k_t
//   S    = S + k_t (x) (beta_t (v_t - kv))
//   o_t  = scale * S^T q_t
//
//   q, k   (S, H, DK) fp32, l2-normalized      v (S, H, dv) fp32
//   g      (S, H, DK) fp32 log-decay           beta (S, H) fp32
//   state  (H, DK, dv) fp32, updated in place  out (S, H, dv) fp32
//
// Same grid/register layout as amp_gdn_scan_kernel: thread j owns state
// column j of its head; the per-token q/k/decay rows go through shared
// memory.
// ---------------------------------------------------------------------------

/* column block of the KDA scan (the GDN scan above is configurable) */
#define AMP_GDN_VT 64

template <int DK>
__global__ __launch_bounds__(AMP_GDN_VT)
void amp_kda_scan_kernel(const float *__restrict__ q, const float *__restrict__ k,
                        const float *__restrict__ v, const float *__restrict__ beta,
                        const float *__restrict__ g, float *__restrict__ state,
                        float *__restrict__ out, int S, int H, int dv,
                        float scale, float *__restrict__ states_out, int n_out)
{
	const int h = blockIdx.x;
	const int j = blockIdx.y * AMP_GDN_VT + threadIdx.x;

	__shared__ float sq[DK];
	__shared__ float sk[DK];
	__shared__ float sg[DK];

	float st[DK];
	float *sp = state + (size_t)h * DK * dv + j;

#pragma unroll
	for (int i = 0; i < DK; i++)
		st[i] = sp[(size_t)i * dv];

	for (int t = 0; t < S; t++) {
		const float *qt = q + ((size_t)t * H + h) * DK;
		const float *kt = k + ((size_t)t * H + h) * DK;
		const float *gt = g + ((size_t)t * H + h) * DK;

		for (int i = threadIdx.x; i < DK; i += AMP_GDN_VT) {
			sq[i] = qt[i];
			sk[i] = kt[i];
			sg[i] = expf(gt[i]);
		}

		__syncthreads();

		float kv = 0.0f;

#pragma unroll
		for (int i = 0; i < DK; i++) {
			st[i] *= sg[i];
			kv = fmaf(st[i], sk[i], kv);
		}

		const float vn = beta[(size_t)t * H + h] *
			(v[((size_t)t * H + h) * dv + j] - kv);

		float o = 0.0f;

#pragma unroll
		for (int i = 0; i < DK; i++) {
			st[i] = fmaf(sk[i], vn, st[i]);
			o = fmaf(st[i], sq[i], o);
		}

		out[((size_t)t * H + h) * dv + j] = o * scale;

		if (t < n_out) {
			float *so = states_out + ((size_t)t * H + h) * DK * dv + j;

#pragma unroll
			for (int i = 0; i < DK; i++)
				so[(size_t)i * dv] = st[i];
		}

		__syncthreads();
	}

#pragma unroll
	for (int i = 0; i < DK; i++)
		sp[(size_t)i * dv] = st[i];
}

extern "C" int
amp_kda_scan_f32(const void *q, const void *k, const void *v, const void *beta,
                const void *g, void *state, void *out, int S, int H, int dk,
                int dv, float scale, void *states_out, int n_out,
                cudaStream_t stream)
{
	if (S <= 0)
		return 0;

	if (H <= 0 || dv % AMP_GDN_VT != 0) {
		fprintf(stderr, "amp_kda_scan: unsupported heads/dims (H %d, dv %d)\n", H, dv);
		return -1;
	}

	dim3 grid(H, dv / AMP_GDN_VT);

	if (dk == 128)
		amp_kda_scan_kernel<128><<<grid, AMP_GDN_VT, 0, stream>>>(
			(const float *)q, (const float *)k, (const float *)v,
			(const float *)beta, (const float *)g, (float *)state,
			(float *)out, S, H, dv, scale, (float *)states_out, n_out);
	else if (dk == 64)
		amp_kda_scan_kernel<64><<<grid, AMP_GDN_VT, 0, stream>>>(
			(const float *)q, (const float *)k, (const float *)v,
			(const float *)beta, (const float *)g, (float *)state,
			(float *)out, S, H, dv, scale, (float *)states_out, n_out);
	else {
		fprintf(stderr, "amp_kda_scan: unsupported k head dim %d (64, 128)\n", dk);
		return -1;
	}

	return (int)cudaGetLastError();
}

// Skinny fp32 product for decode / verify rows: out (M, N) = x (M, K) @ w (K, N)
// with M <= 8, all fp32 row-major. cuBLAS runs these as GEMVs for one row but
// as a slow generic GEMM for 2..8 (38 us for 4096 x 128). One warp reads 32
// lanes x VEC consecutive columns of a weight row per step (coalesced), a
// block's 8 warps split a chunk of rows, the chunks of a column tile are
// reduced in a fixed order by the last-arriving block (deterministic).
//   grid (column tiles, chunks); part: (chunks, M, N) workspace; cnt: one
//   int per column tile (zero; reset by the reducer).
#define AMP_SK_WARPS 8

template <int M, int VEC>
__global__ void amp_skinny_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                 float *__restrict__ out, float *__restrict__ part,
                                 int *__restrict__ cnt, int K, int N, int rows_per_chunk)
{
	constexpr int COLS = 32 * VEC;
	__shared__ float red[AMP_SK_WARPS][M][COLS];
	__shared__ int s_last;

	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int col0 = blockIdx.x * COLS + lane * VEC;
	const int r0 = blockIdx.y * rows_per_chunk;
	const int r1 = min(K, r0 + rows_per_chunk);
	const int nchunks = gridDim.y;

	float acc[M][VEC];

#pragma unroll
	for (int m = 0; m < M; m++)
#pragma unroll
		for (int v = 0; v < VEC; v++)
			acc[m][v] = 0.0f;

	for (int r = r0 + warp; r < r1; r += AMP_SK_WARPS) {
		float xv[M];

#pragma unroll
		for (int m = 0; m < M; m++)
			xv[m] = x[(size_t)m * K + r];

		if (col0 < N) {
			const float *wr = w + (size_t)r * N + col0;

			if (VEC == 4) {
				float4 wv = *(const float4 *)wr;

#pragma unroll
				for (int m = 0; m < M; m++) {
					acc[m][0] = fmaf(xv[m], wv.x, acc[m][0]);
					acc[m][1] = fmaf(xv[m], wv.y, acc[m][1]);
					acc[m][2] = fmaf(xv[m], wv.z, acc[m][2]);
					acc[m][3] = fmaf(xv[m], wv.w, acc[m][3]);
				}
			}
			else {
				float wv = *wr;

#pragma unroll
				for (int m = 0; m < M; m++)
					acc[m][0] = fmaf(xv[m], wv, acc[m][0]);
			}
		}
	}

#pragma unroll
	for (int m = 0; m < M; m++)
#pragma unroll
		for (int v = 0; v < VEC; v++)
			red[warp][m][lane * VEC + v] = acc[m][v];

	__syncthreads();

	// sum the warps (fixed order) -> this chunk's partial
	for (int i = threadIdx.x; i < M * COLS; i += blockDim.x) {
		const int m = i / COLS, c = i % COLS;
		const int col = blockIdx.x * COLS + c;

		float sum = 0.0f;

#pragma unroll
		for (int wi = 0; wi < AMP_SK_WARPS; wi++)
			sum += red[wi][m][c];

		if (col < N) {
			if (nchunks == 1)
				out[(size_t)m * N + col] = sum;
			else
				part[((size_t)blockIdx.y * M + m) * N + col] = sum;
		}
	}

	if (nchunks == 1)
		return;

	__threadfence();
	__syncthreads();

	if (threadIdx.x == 0)
		s_last = (atomicAdd(&cnt[blockIdx.x], 1) == nchunks - 1);

	__syncthreads();

	if (!s_last)
		return;

	__threadfence();

	for (int i = threadIdx.x; i < M * COLS; i += blockDim.x) {
		const int m = i / COLS, c = i % COLS;
		const int col = blockIdx.x * COLS + c;

		if (col >= N)
			continue;

		float sum = 0.0f;

		for (int ch = 0; ch < nchunks; ch++)
			sum += __ldcg(&part[((size_t)ch * M + m) * N + col]);

		out[(size_t)m * N + col] = sum;
	}

	if (threadIdx.x == 0)
		cnt[blockIdx.x] = 0;
}

template <int M>
static int amp_skinny_launch_m(const float *x, const float *w, float *out, float *part,
                              int *cnt, int K, int N, int rpc, int ntiles, int nchunks,
                              bool vec4, cudaStream_t stream)
{
	dim3 grid(ntiles, nchunks);

	if (vec4)
		amp_skinny_kernel<M, 4><<<grid, AMP_SK_WARPS * 32, 0, stream>>>(x, w, out, part, cnt, K, N, rpc);
	else
		amp_skinny_kernel<M, 1><<<grid, AMP_SK_WARPS * 32, 0, stream>>>(x, w, out, part, cnt, K, N, rpc);

	return (int)cudaGetLastError();
}

// part must hold >= nchunks_max * M * N floats; cnt >= ntiles ints (zeroed)
extern "C" int
amp_skinny_f32(const float *x, const float *w, float *out, float *part, int *cnt,
              int M, int K, int N, long part_floats, int cnt_len, cudaStream_t stream)
{
	if (M < 1 || M > 8 || K < 1 || N < 1)
		return (int)cudaErrorInvalidValue;

	const bool vec4 = (N % 4 == 0) && (((uintptr_t)w & 15) == 0);
	const int cols = 32 * (vec4 ? 4 : 1);
	const int ntiles = (N + cols - 1) / cols;
	int nchunks = 1;
	int rpc = K;

	// enough blocks to cover the device's bandwidth: aim for >= 64
	if (ntiles < 64 && K >= 128) {
		nchunks = (64 + ntiles - 1) / ntiles;

		if (nchunks > K / 64) nchunks = K / 64;
		if (nchunks < 1) nchunks = 1;

		rpc = ((K + nchunks - 1) / nchunks + 7) / 8 * 8;
		nchunks = (K + rpc - 1) / rpc;
	}

	if (nchunks > 1 && ((long)nchunks * M * N > part_floats || ntiles > cnt_len))
		return (int)cudaErrorInvalidValue;

	switch (M) {
	case 1: return amp_skinny_launch_m<1>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 2: return amp_skinny_launch_m<2>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 3: return amp_skinny_launch_m<3>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 4: return amp_skinny_launch_m<4>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 5: return amp_skinny_launch_m<5>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 6: return amp_skinny_launch_m<6>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	case 7: return amp_skinny_launch_m<7>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	default: return amp_skinny_launch_m<8>(x, w, out, part, cnt, K, N, rpc, ntiles, nchunks, vec4, stream);
	}
}

// KDA gates: g = lb * sigmoid(decay_h * (f + dt_bias)) (log-decay, per key
// channel; decay_h = exp(A_log_h)), beta = sigmoid(braw * beta_scale).
//   f (S, H*DK) fp32, dt_bias (H*DK) fp32, decay (H) fp32, braw (S, H) fp32
//   g (S, H, DK) fp32 out, beta (S, H) fp32 out
__global__ void amp_kda_gates_kernel(const float *__restrict__ f,
                                    const float *__restrict__ dt_bias,
                                    const float *__restrict__ decay,
                                    const float *__restrict__ braw,
                                    float *__restrict__ g, float *__restrict__ beta,
                                    long n, int H, int DK, float lb,
                                    float beta_scale)
{
	long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	int c = (int)(idx % ((long)H * DK));      // channel within the token
	int h = c / DK;
	float x = (f[idx] + dt_bias[c]) * decay[h];

	g[idx] = lb / (1.0f + expf(-x));

	// one beta per (token, head): the first channel's thread writes it
	if (c % DK == 0) {
		long t = idx / ((long)H * DK);
		float b = braw[t * H + h] * beta_scale;

		beta[t * H + h] = 1.0f / (1.0f + expf(-b));
	}
}

extern "C" int
amp_kda_gates(const void *f, const void *dt_bias, const void *decay,
             const void *braw, void *g, void *beta, long S, int H, int DK,
             float lb, float beta_scale, cudaStream_t stream)
{
	long n = S * H * DK;

	if (n <= 0)
		return 0;

	amp_kda_gates_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
		(const float *)f, (const float *)dt_bias, (const float *)decay,
		(const float *)braw, (float *)g, (float *)beta, n, H, DK, lb,
		beta_scale);

	return (int)cudaGetLastError();
}

// KDA output: gated RMSNorm over the v head dim with a SIGMOID gate, fp32:
//   out = w * (core * rsqrt(mean(core^2) + eps)) * sigmoid(z)
//   core (S, H, dv) fp32; z (S, H*dv) fp32; w (dv) fp16/fp32 (as wfmt);
//   out (S, H*dv) fp16. Block per (token, head), dv threads.
template <typename W_T>
__global__ void amp_kda_out_norm_kernel(const float *__restrict__ core,
                                       const float *__restrict__ z,
                                       const W_T *__restrict__ w,
                                       __half *__restrict__ out, int dv,
                                       float eps)
{
	const size_t row = blockIdx.x;
	const int e = threadIdx.x;
	float f = core[row * dv + e];
	float ss = f * f;

	__shared__ float red[32];

#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		ss += __shfl_down_sync(0xffffffffu, ss, off);

	if ((e & 31) == 0)
		red[e >> 5] = ss;

	__syncthreads();

	if (e < 32) {
		float s2 = (e < (int)(blockDim.x >> 5)) ? red[e] : 0.0f;

#pragma unroll
		for (int off = 16; off > 0; off >>= 1)
			s2 += __shfl_down_sync(0xffffffffu, s2, off);

		if (e == 0)
			red[0] = rsqrtf(s2 / dv + eps);
	}

	__syncthreads();

	float wf;

	if constexpr (sizeof(W_T) == 2)
		wf = __half2float(w[e]);
	else
		wf = w[e];

	float zf = z[row * dv + e];

	out[row * dv + e] = __float2half(wf * (f * red[0]) / (1.0f + expf(-zf)));
}

extern "C" int
amp_kda_out_norm(const void *core, const void *z, const void *w, int w_is_half,
                void *out, long rows, int dv, float eps, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	if (dv > 1024 || dv % 32 != 0) {
		fprintf(stderr, "amp_kda_out_norm: unsupported dv %d\n", dv);
		return -1;
	}

	if (w_is_half)
		amp_kda_out_norm_kernel<__half><<<(unsigned)rows, dv, 0, stream>>>(
			(const float *)core, (const float *)z, (const __half *)w,
			(__half *)out, dv, eps);
	else
		amp_kda_out_norm_kernel<float><<<(unsigned)rows, dv, 0, stream>>>(
			(const float *)core, (const float *)z, (const float *)w,
			(__half *)out, dv, eps);

	return (int)cudaGetLastError();
}

// Q8 latent (KV) cache rows: `dim` values per row in groups of 32, one
// symmetric int8 code per value and one fp16 scale (absmax / 127) per
// group — 544 bytes per 512-value row instead of 1024. put quantizes the
// source rows to destination rows idx[row] (or start + row when idx is
// NULL); get dequantizes rows idx[row] / start + row to fp16. One block per
// row, dim / 4 threads (each owns 4 values; 8 threads per group).
__global__ void amp_kvq8_put_kernel(const __half *__restrict__ src,
                                   signed char *__restrict__ q,
                                   __half *__restrict__ s,
                                   const long long *__restrict__ idx,
                                   long start, int dim)
{
	int t = threadIdx.x;
	long dst = idx ? (long)idx[blockIdx.x] : start + blockIdx.x;
	const __half2 *p = (const __half2 *)(src + (long)blockIdx.x * dim) + 2 * t;
	float2 a = __half22float2(p[0]);
	float2 b = __half22float2(p[1]);
	float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));

	for (int o = 1; o < 8; o <<= 1)
		amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

	// quantize against the ROUNDED scale so the dequantized value is the
	// nearest grid point of what is stored
	__half sh = __float2half(amax * (1.0f / 127.0f));
	float sf = __half2float(sh);
	float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
	int c0 = max(-127, min(127, __float2int_rn(a.x * inv)));
	int c1 = max(-127, min(127, __float2int_rn(a.y * inv)));
	int c2 = max(-127, min(127, __float2int_rn(b.x * inv)));
	int c3 = max(-127, min(127, __float2int_rn(b.y * inv)));

	((int *)(q + dst * dim))[t] = (c0 & 0xff) | ((c1 & 0xff) << 8) |
		((c2 & 0xff) << 16) | ((c3 & 0xff) << 24);

	if ((t & 7) == 0)
		s[dst * (dim / 32) + (t >> 3)] = sh;
}

__global__ void amp_kvq8_get_kernel(const signed char *__restrict__ q,
                                   const __half *__restrict__ s,
                                   __half *__restrict__ out,
                                   const long long *__restrict__ idx,
                                   long start, int dim)
{
	int t = threadIdx.x;
	long src = idx ? (long)idx[blockIdx.x] : start + blockIdx.x;
	int w = ((const int *)(q + src * dim))[t];
	float sf = __half2float(s[src * (dim / 32) + (t >> 3)]);
	__half2 *o = (__half2 *)(out + (long)blockIdx.x * dim) + 2 * t;

	o[0] = __floats2half2_rn(sf * (float)(signed char)(w & 0xff),
	                         sf * (float)(signed char)((w >> 8) & 0xff));
	o[1] = __floats2half2_rn(sf * (float)(signed char)((w >> 16) & 0xff),
	                         sf * (float)(signed char)((w >> 24) & 0xff));
}

extern "C" int
amp_kvq8_put(const void *src, void *q, void *s, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	amp_kvq8_put_kernel<<<(unsigned)rows, dim / 4, 0, stream>>>(
		(const __half *)src, (signed char *)q, (__half *)s, idx, start, dim);

	return (int)cudaGetLastError();
}

extern "C" int
amp_kvq8_get(const void *q, const void *s, void *out, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	amp_kvq8_get_kernel<<<(unsigned)rows, dim / 4, 0, stream>>>(
		(const signed char *)q, (const __half *)s, (__half *)out, idx, start, dim);

	return (int)cudaGetLastError();
}

// Q4 cache rows (the host-RAM prompt-cache store's oldest tier): as the Q8
// rows above with a signed 4-bit code per value (absmax / 7 per group of 32),
// two codes per byte: a row of `dim` values is dim / 2 bytes. Each thread
// owns 4 values = 16 bits of code.
__global__ void amp_kvq4_put_kernel(const __half *__restrict__ src,
                                   unsigned short *__restrict__ q,
                                   __half *__restrict__ s,
                                   const long long *__restrict__ idx,
                                   long start, int dim)
{
	int t = threadIdx.x;
	long dst = idx ? (long)idx[blockIdx.x] : start + blockIdx.x;
	const __half2 *p = (const __half2 *)(src + (long)blockIdx.x * dim) + 2 * t;
	float2 a = __half22float2(p[0]);
	float2 b = __half22float2(p[1]);
	float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));

	for (int o = 1; o < 8; o <<= 1)
		amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

	__half sh = __float2half(amax * (1.0f / 7.0f));
	float sf = __half2float(sh);
	float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
	int c0 = max(-7, min(7, __float2int_rn(a.x * inv)));
	int c1 = max(-7, min(7, __float2int_rn(a.y * inv)));
	int c2 = max(-7, min(7, __float2int_rn(b.x * inv)));
	int c3 = max(-7, min(7, __float2int_rn(b.y * inv)));

	q[dst * (dim / 4) + t] = (unsigned short)((c0 & 0xf) | ((c1 & 0xf) << 4) |
		((c2 & 0xf) << 8) | ((c3 & 0xf) << 12));

	if ((t & 7) == 0)
		s[dst * (dim / 32) + (t >> 3)] = sh;
}

__device__ __forceinline__ float amp_nib(unsigned w, int i)
{
	return (float)((int)(w << (28 - 4 * i)) >> 28);
}

__global__ void amp_kvq4_get_kernel(const unsigned short *__restrict__ q,
                                   const __half *__restrict__ s,
                                   __half *__restrict__ out,
                                   const long long *__restrict__ idx,
                                   long start, int dim)
{
	int t = threadIdx.x;
	long src = idx ? (long)idx[blockIdx.x] : start + blockIdx.x;
	unsigned w = q[src * (dim / 4) + t];
	float sf = __half2float(s[src * (dim / 32) + (t >> 3)]);
	__half2 *o = (__half2 *)(out + (long)blockIdx.x * dim) + 2 * t;

	o[0] = __floats2half2_rn(sf * amp_nib(w, 0), sf * amp_nib(w, 1));
	o[1] = __floats2half2_rn(sf * amp_nib(w, 2), sf * amp_nib(w, 3));
}

extern "C" int
amp_kvq4_put(const void *src, void *q, void *s, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	amp_kvq4_put_kernel<<<(unsigned)rows, dim / 4, 0, stream>>>(
		(const __half *)src, (unsigned short *)q, (__half *)s, idx, start, dim);

	return (int)cudaGetLastError();
}

extern "C" int
amp_kvq4_get(const void *q, const void *s, void *out, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream)
{
	if (rows <= 0)
		return 0;

	amp_kvq4_get_kernel<<<(unsigned)rows, dim / 4, 0, stream>>>(
		(const unsigned short *)q, (const __half *)s, (__half *)out, idx, start, dim);

	return (int)cudaGetLastError();
}

// DSA indexer scoring, fused (GLM-5.3 sparse attention):
//
//   out[r][p] = valid(r, p) ? fp16( sum_h w[r][h] * scale * relu(q[r][h] . pool[p]) )
//                           : -65504
//
// with HI = 32 index heads of D = 128 dims and valid = (p * P + P - 1 <=
// pos0 + r) (a pool is selectable once all of its tokens are visible). The
// composite ATen path materialised the (R * HI, n_pools) score matrix and
// made six passes over it (matmul, scale, clamp, weighted bmm: 800 MB per
// 32-query slab at 65k pools); this reads the pooled keys once per 8 queries
// and writes only (R, n_pools).
//
// One CTA = 8 queries (one warp each: its 32 heads are two m16 tiles) x 64
// pools. The pool tile sits in shared memory (row stride 136 halves: the
// B-fragment loads are bank-conflict free); the q fragments of a warp stay in
// registers for the whole K = 128 loop. mma.sync m16n8k16 (fp16 in, fp32
// accumulate); relu, the per-head weight and the sum over heads happen on the
// accumulator fragments (a shuffle reduction over the 8 lanes that share a
// column).
#define IDX_D 128
#define IDX_TILE 64
#define IDX_QPB 8
#define IDX_STRIDE 136

__device__ __forceinline__ void
amp_mma_16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
	             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
	             : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
	             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256)
amp_idx_score_kernel(const __half *__restrict__ q, const __half *__restrict__ pools,
                    const float *__restrict__ w, __half *__restrict__ out,
                    int R, int n_pools, float scale, int pos0, int P, int HI,
                    const long long *__restrict__ posp)
{
	extern __shared__ __half sp[];            // IDX_TILE x IDX_STRIDE

	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
	const int p0 = blockIdx.x * IDX_TILE;
	const int r = blockIdx.y * IDX_QPB + warp;

	// a device-side position (CUDA-graph decode) replaces pos0
	if (posp)
		pos0 = (int)*posp;

	// a tile no query of this block can select yet (its first pool still
	// incomplete for the last query) is all mask: the decode graphs score
	// the whole pool capacity, this keeps the work proportional to the
	// context (L134)
	if (p0 * P + P - 1 > pos0 + min(R - 1, (int)blockIdx.y * IDX_QPB + IDX_QPB - 1)) {
		for (int i = threadIdx.x; i < IDX_QPB * IDX_TILE; i += 256) {
			int rr = blockIdx.y * IDX_QPB + i / IDX_TILE, pa = p0 + i % IDX_TILE;

			if (rr < R && pa < n_pools)
				out[(long)rr * n_pools + pa] = __float2half(-65504.0f);
		}

		return;
	}

	for (int i = threadIdx.x; i < IDX_TILE * (IDX_D / 8); i += 256) {
		int row = i >> 4, c = i & 15;
		int pr = p0 + row;
		uint4 v = make_uint4(0, 0, 0, 0);

		if (pr < n_pools)
			v = *(const uint4 *)(pools + (long)pr * IDX_D + c * 8);

		*(uint4 *)(sp + row * IDX_STRIDE + c * 8) = v;
	}

	__syncthreads();

	if (r >= R)
		return;

	// Heads are processed in halves of 32 (two m tiles of 16): the pool tile in shared memory is
	// shared by all halves and each half adds its relu * weight contribution into S (fp32), so
	// 64 indexer heads cost one pass over the pool and one fp16 rounding of the final sum.
	const int g = lane >> 2, t = lane & 3;
	float S[8][2];

	#pragma unroll
	for (int nt = 0; nt < 8; nt++)
		S[nt][0] = S[nt][1] = 0.0f;

	for (int hh = 0; hh < HI / 32; hh++) {
		const int hb = hh * 32;

		// q fragments: A[mt][ks][0..3]; rows = heads hb + mt * 16 + lane / 4 (+ 8)
		uint32_t A[2][8][4];

		#pragma unroll
		for (int mt = 0; mt < 2; mt++) {
			const __half *r0 = q + ((long)r * HI + hb + mt * 16 + g) * IDX_D + t * 2;
			const __half *r1 = r0 + 8 * IDX_D;

			#pragma unroll
			for (int ks = 0; ks < 8; ks++) {
				A[mt][ks][0] = *(const uint32_t *)(r0 + ks * 16);
				A[mt][ks][1] = *(const uint32_t *)(r1 + ks * 16);
				A[mt][ks][2] = *(const uint32_t *)(r0 + ks * 16 + 8);
				A[mt][ks][3] = *(const uint32_t *)(r1 + ks * 16 + 8);
			}
		}

		float acc[2][8][4];

		#pragma unroll
		for (int mt = 0; mt < 2; mt++)
			#pragma unroll
			for (int nt = 0; nt < 8; nt++)
				acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;

		#pragma unroll
		for (int ks = 0; ks < 8; ks++) {
			uint32_t b[8][2];

			#pragma unroll
			for (int nt = 0; nt < 8; nt++) {
				const __half *bp = sp + (nt * 8 + g) * IDX_STRIDE + ks * 16 + t * 2;

				b[nt][0] = *(const uint32_t *)bp;
				b[nt][1] = *(const uint32_t *)(bp + 8);
			}

			#pragma unroll
			for (int mt = 0; mt < 2; mt++)
				#pragma unroll
				for (int nt = 0; nt < 8; nt++)
					amp_mma_16816(acc[mt][nt], A[mt][ks], b[nt][0], b[nt][1]);
		}

		// head weights of this thread's accumulator rows (g and g + 8 of each m tile)
		float wh[2][2];

		#pragma unroll
		for (int mt = 0; mt < 2; mt++) {
			wh[mt][0] = w[(long)r * HI + hb + mt * 16 + g] * scale;
			wh[mt][1] = w[(long)r * HI + hb + mt * 16 + g + 8] * scale;
		}

		#pragma unroll
		for (int nt = 0; nt < 8; nt++)
			#pragma unroll
			for (int mt = 0; mt < 2; mt++) {
				S[nt][0] += fmaxf(acc[mt][nt][0], 0.0f) * wh[mt][0] + fmaxf(acc[mt][nt][2], 0.0f) * wh[mt][1];
				S[nt][1] += fmaxf(acc[mt][nt][1], 0.0f) * wh[mt][0] + fmaxf(acc[mt][nt][3], 0.0f) * wh[mt][1];
			}
	}

	const int qpos = pos0 + r;

	#pragma unroll
	for (int nt = 0; nt < 8; nt++) {
		float s0 = S[nt][0], s1 = S[nt][1];

		// sum the 8 lanes (g = 0..7) that hold the same columns
		#pragma unroll
		for (int o = 4; o < 32; o <<= 1) {
			s0 += __shfl_xor_sync(0xffffffffu, s0, o);
			s1 += __shfl_xor_sync(0xffffffffu, s1, o);
		}

		if (g == 0) {
			int pa = p0 + nt * 8 + t * 2;

			if (pa < n_pools)
				out[(long)r * n_pools + pa] =
					(pa * P + P - 1 <= qpos) ? __float2half(s0) : __float2half(-65504.0f);

			if (pa + 1 < n_pools)
				out[(long)r * n_pools + pa + 1] =
					((pa + 1) * P + P - 1 <= qpos) ? __float2half(s1) : __float2half(-65504.0f);
		}
	}
}

extern "C" int
amp_idx_score(const void *q, const void *pools, const void *w, void *out,
             int R, int n_pools, float scale, int pos0, int P, int HI,
             const long long *posp, cudaStream_t stream)
{
	if (R <= 0 || n_pools <= 0)
		return 0;

	if (HI < 32 || HI % 32 != 0)
		return -1;

	dim3 grid((unsigned)((n_pools + IDX_TILE - 1) / IDX_TILE),
	          (unsigned)((R + IDX_QPB - 1) / IDX_QPB));
	size_t smem = (size_t)IDX_TILE * IDX_STRIDE * sizeof(__half);

	amp_idx_score_kernel<<<grid, 256, smem, stream>>>(
		(const __half *)q, (const __half *)pools, (const float *)w,
		(__half *)out, R, n_pools, scale, pos0, P, HI, posp);

	return (int)cudaGetLastError();
}

/* ------------------------------------------------------------------ */
/* DSA pool selection for decode (GLM-5.3 sparse MLA layers).          */
/*                                                                     */
/* Per query row: the top `kp` of the complete pools of its context by */
/* indexer score (a deterministic exact selection: score descending,   */
/* ties to the lower pool index; fewer complete pools: all of them),   */
/* expanded to their P member tokens, followed by the P - 1 tokens of  */
/* the query's incomplete tail pool. Replaces torch.topk (14 kernels), */
/* the dead-pool mask, the member expansion, the tail and the pad mask */
/* (about 25 elementwise launches): one block per row.                 */
/*                                                                     */
/*   scores (T, cap) fp16: the indexer scores, -65504 for pools the    */
/*          query cannot select (amp_idx_score)                        */
/*   pos    (1) int64 device: position of row 0 (row r is at pos + r)  */
/*   idx    (T, K) int64: token indices, 0 where padded                */
/*   pad    (T, K) fp16: 0 for a token, -65504 for padding             */
/*   K = kp * P + P - 1                                                */
/*                                                                     */
/* The kp-th largest key is found by a two-level (8 + 8 bit) radix    */
/* select on the order-preserving 16-bit key of the fp16 score; the   */
/* elements above it plus the first ties in index order are then      */
/* written in ascending pool order (per-thread contiguous chunks and a */
/* block scan: the output does not depend on thread timing).          */
/* ------------------------------------------------------------------ */

#define DSS_THREADS 1024

__device__ __forceinline__ unsigned
amp_dss_key(__half h)
{
	unsigned b = (unsigned)__half_as_ushort(h);

	return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

/* One histogram increment per warp and distinct bin: lanes holding the same
 * bin elect a leader that adds the group's size (scores cluster in a few
 * bins, a plain shared atomicAdd would serialise 1024 threads on them).
 * All 32 lanes must execute this (uniform loop trip counts); bin 0xffff
 * is "no element". */
__device__ __forceinline__ void
amp_dss_hist_add(unsigned *hist, unsigned bin, int lane)
{
	unsigned m = __match_any_sync(0xffffffffu, bin);

	if (bin != 0xffffu && lane == (__ffs(m) - 1))
		atomicAdd(&hist[bin], (unsigned)__popc(m));
}

/* The highest bin b of a 256-bin histogram such that the counts of the bins
 * from 255 down to b reach `need` (need <= the total): *bin = b, *above =
 * the count of the bins above it. Threads 0..255 scan the descending order
 * (warp shuffles + 8 warp sums); every thread of the block must call it. */
__device__ __forceinline__ void
amp_dss_find_bin(const unsigned *hist, unsigned need, unsigned *wsum, unsigned *bin, unsigned *above)
{
	const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
	unsigned c = 0, incl = 0;

	if (t < 256) {
		c = hist[255 - t];
		incl = c;

		#pragma unroll
		for (int o = 1; o < 32; o <<= 1) {
			unsigned v = __shfl_up_sync(0xffffffffu, incl, o);

			if (lane >= o)
				incl += v;
		}

		if (lane == 31)
			wsum[warp] = incl;
	}

	__syncthreads();

	if (t < 256) {
		unsigned base = 0;

		for (int w = 0; w < warp; w++)
			base += wsum[w];

		incl += base;

		if (incl >= need && incl - c < need) {
			*bin = (unsigned)(255 - t);
			*above = incl - c;
		}
	}

	__syncthreads();
}

#define DSS_TILES 4        /* tiles of 1024 elements loaded per iteration (latency hiding) */

__global__ void __launch_bounds__(DSS_THREADS)
amp_dsa_select_kernel(const __half *__restrict__ scores, const long long *__restrict__ posp,
                     long long *__restrict__ idx, __half *__restrict__ pad,
                     int cap, int P, int kp)
{
	__shared__ unsigned hist[256];
	__shared__ unsigned wsum[8];
	__shared__ unsigned w_gt[2][32], w_eq[2][32];
	__shared__ unsigned s_hi, s_above, s_lo, s_above2;

	const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
	const int K = kp * P + P - 1;
	const long long qpos = posp[0] + row;
	const __half *sc = scores + (size_t)row * cap;

	/* only the complete pools of the row's context can be selected: the
	 * scan covers n of the cap pool slots (the rest is masked scores) */
	const int n = (int)min((long long)cap, (qpos + 1) / P);
	const int nsel = min(kp, n);
	const int ntile = (n + DSS_THREADS - 1) / DSS_THREADS;

	long long *irow = idx + (size_t)row * K;
	__half *prow = pad + (size_t)row * K;
	const __half NEG = __float2half(-65504.0f), ZERO = __float2half(0.0f);

	unsigned thr = 0, need_eq = 0;

	if (nsel > 0) {
		/* --- level 1: high byte of the key, coalesced tiles */
		if (t < 256)
			hist[t] = 0;

		__syncthreads();

		for (int j = 0; j < ntile; j += DSS_TILES) {
			unsigned k[DSS_TILES];

			#pragma unroll
			for (int u = 0; u < DSS_TILES; u++) {
				int e = (j + u) * DSS_THREADS + t;

				k[u] = (e < n) ? (amp_dss_key(sc[e]) >> 8) : 0xffffu;
			}

			#pragma unroll
			for (int u = 0; u < DSS_TILES; u++)
				amp_dss_hist_add(hist, k[u], lane);
		}

		__syncthreads();
		amp_dss_find_bin(hist, (unsigned)nsel, wsum, &s_hi, &s_above);

		const unsigned hi = s_hi, need1 = (unsigned)nsel - s_above;

		/* --- level 2: low byte within that bin */
		if (t < 256)
			hist[t] = 0;

		__syncthreads();

		for (int j = 0; j < ntile; j += DSS_TILES) {
			unsigned k[DSS_TILES];

			#pragma unroll
			for (int u = 0; u < DSS_TILES; u++) {
				int e = (j + u) * DSS_THREADS + t;
				unsigned key = (e < n) ? amp_dss_key(sc[e]) : 0u;

				k[u] = (e < n && (key >> 8) == hi) ? (key & 0xffu) : 0xffffu;
			}

			#pragma unroll
			for (int u = 0; u < DSS_TILES; u++)
				amp_dss_hist_add(hist, k[u], lane);
		}

		__syncthreads();
		amp_dss_find_bin(hist, need1, wsum, &s_lo, &s_above2);

		thr = (hi << 8) | s_lo;               /* the nsel-th largest key */
		need_eq = need1 - s_above2;           /* ties at the threshold taken */

		/* --- write the selected pools in ascending order, one tile at a
		 * time: the output position is the number of selected elements
		 * before it (warp ballots + the warp totals; the ties taken are
		 * the first need_eq in index order) */
		unsigned eq_total = 0, out_base = 0;
		const unsigned lane_lt = (1u << lane) - 1u;

		for (int j = 0; j < ntile; j++) {
			int e = j * DSS_THREADS + t;
			bool valid = (e < n);
			unsigned k = valid ? amp_dss_key(sc[e]) : 0u;
			bool gt = valid && k > thr, eq = valid && k == thr;
			unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);

			if (lane == 0) {
				w_gt[j & 1][warp] = (unsigned)__popc(bg);
				w_eq[j & 1][warp] = (unsigned)__popc(be);
			}

			__syncthreads();

			/* exclusive prefixes over the 32 warp totals, totals */
			unsigned g = w_gt[j & 1][lane], q = w_eq[j & 1][lane];
			unsigned ig = g, iq = q;

			#pragma unroll
			for (int o = 1; o < 32; o <<= 1) {
				unsigned vg = __shfl_up_sync(0xffffffffu, ig, o), vq = __shfl_up_sync(0xffffffffu, iq, o);

				if (lane >= o) {
					ig += vg;
					iq += vq;
				}
			}

			unsigned gpre = __shfl_sync(0xffffffffu, ig - g, warp) + (unsigned)__popc(bg & lane_lt);
			unsigned epre = __shfl_sync(0xffffffffu, iq - q, warp) + (unsigned)__popc(be & lane_lt);
			unsigned gtot = __shfl_sync(0xffffffffu, ig, 31), etot = __shfl_sync(0xffffffffu, iq, 31);

			unsigned eq_taken_before = min(eq_total, need_eq);

			if (gt || (eq && eq_total + epre < need_eq)) {
				unsigned pos = out_base + gpre + (min(eq_total + epre, need_eq) - eq_taken_before);

				for (int c = 0; c < P; c++) {
					irow[(size_t)pos * P + c] = (long long)e * P + c;
					prow[(size_t)pos * P + c] = ZERO;
				}
			}

			out_base += gtot + (min(eq_total + etot, need_eq) - eq_taken_before);
			eq_total += etot;
		}
	}

	/* --- padding up to kp pools (fewer complete pools than kp) */
	for (int i = nsel * P + t; i < kp * P; i += DSS_THREADS) {
		irow[i] = 0;
		prow[i] = NEG;
	}

	/* --- the incomplete tail pool of the query: tokens start .. qpos */
	if (t < P - 1) {
		long long start = ((qpos + 1) / P) * P;
		long long tf = start + t;
		bool ok = (tf <= qpos);

		irow[(size_t)kp * P + t] = ok ? tf : 0;
		prow[(size_t)kp * P + t] = ok ? ZERO : NEG;
	}
}

extern "C" int
amp_dsa_select(const void *scores, const long long *posp, void *idx, void *pad,
              int T, int cap, int P, int kp, cudaStream_t stream)
{
	if (T < 1 || cap < kp || kp < 1 || P < 1 || P > DSS_THREADS || kp > 4096)
		return -1;

	amp_dsa_select_kernel<<<T, DSS_THREADS, 0, stream>>>((const __half *)scores, posp,
		(long long *)idx, (__half *)pad, cap, P, kp);

	return (int)cudaGetLastError();
}

/* ------------------------------------------------------------------ */
/* Tensor-core (mma) split-KV decode attention.                        */
/*                                                                     */
/* Same contract as amp_attn_dec_fp16 (grid (Hk, ns), device-side len,  */
/* identical workspace + combine), but the score and P*V products run  */
/* on the tensor cores and every K/V byte is read once per block:      */
/*                                                                     */
/*   - Block = (kv head, split), 8 warps. The mma M dimension holds    */
/*     rep*S query rows (<= 32, two m16 tiles); tile row r is query    */
/*     head j*rep + r/S, row-in-head r%S.                              */
/*   - K/V tiles of BK keys are staged in shared memory with cp.async  */
/*     double buffering (the shuffle kernel re-reads them per warp).   */
/*   - Scores: mma.m16n8k16, A = Q rows from a shared tile (ldmatrix), */
/*     B = K keys (plain ldmatrix: the K tile is n-major, exactly the  */
/*     col-major-B layout). Warps split the (m-tile, 8-key n-tile)     */
/*     pairs of a tile — one m-tile and one n-tile per warp. Raw       */
/*     scores land in shared memory; the scale and the causal mask are */
/*     applied per (row, key) in fp32 at the fragment write.           */
/*   - Softmax: each warp owns the rows r % W == w, keeps the running  */
/*     (m, l) and writes e = exp(score - m_new) to a shared fp16       */
/*     e-block plus the per-row correction (for the PV rescale).       */
/*   - PV: mma as well, A = e-block, B = V tile (ldmatrix.trans: the   */
/*     V tile is k-major); warps split the 8-dim n-tiles of E. The     */
/*     accumulator fragment is rescaled per row before each tile's     */
/*     mma accumulates into it.                                        */
/*   - Workspace: m/l by the row-owning warp, acc dims scattered by    */
/*     the fragment-owning warps (disjoint writes, no reduction); the  */
/*     combine kernel is unchanged.                                    */
/*                                                                     */
/* Fragment layouts follow the PTX ISA m16n8k16 f16/f32 table:         */
/*   A (m16k16, row-major): lane l addresses row (l&7) + 8*((l>>3)&1), */
/*     k-piece (l>>4); C (m16n8 f32): lane l holds rows (l>>2) and     */
/*     (l>>2)+8, cols 2*(l&3) and 2*(l&3)+1 in c[0..3] pairs.          */
/* 16-byte smem pieces are XOR-swizzled with the row index so cp.async */
/* stores and ldmatrix loads stay bank-conflict free.                  */
/* ------------------------------------------------------------------ */

#define AMP_ADM_WARPS 8
#define AMP_ADM_BK 32

__device__ __forceinline__ void amp_adm_mma(float *d, const unsigned *a,
                                           const unsigned *b)
{
	asm volatile(
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
		  "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void amp_adm_ldsm_x4(unsigned *r, unsigned addr)
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
		"{%0,%1,%2,%3}, [%4];\n"
		: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void amp_adm_ldsm_x2(unsigned *r, unsigned addr)
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
		"{%0,%1}, [%2];\n"
		: "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void amp_adm_ldsm_x2_t(unsigned *r, unsigned addr)
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
		"{%0,%1}, [%2];\n"
		: "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void amp_adm_cp(unsigned sh_addr,
                                          const void *gl)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
		:: "r"(sh_addr), "l"(gl));
}

__device__ __forceinline__ void amp_adm_cp_commit()
{
	asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void amp_adm_cp_wait()
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/* Swizzled byte offset of 16-byte piece `piece` in row `row` of a tile
 * whose rows are `row_bytes` wide. */
__device__ __forceinline__ int amp_adm_swz(int row, int piece, int row_bytes)
{
	return row * row_bytes + ((piece ^ (row & 7)) << 4);
}

/* Shared-memory address of a tile element: base pointer + swizzled
 * offset (all offsets are BYTE offsets). */
#define AMP_ADM_ADDR(base, row, piece, row_bytes) \
	((unsigned)__cvta_generic_to_shared(base) + \
	 amp_adm_swz(row, piece, row_bytes))

/* E = head dim (64 / 128 / 256); S (query rows per head) is runtime.
 * MT = 16-row m-tiles of the query block (rep * S <= 16 * MT), BK = keys
 * per tile. MT = 2 and 3 with BK = 32 serve up to 32 / 48 rows (a DFlash
 * verify window of 7 at rep 6 is 42: one pass over the keys instead of
 * two row-group launches, L132); MT = 4 takes BK = 16 to stay under the
 * 99 KiB shared-memory opt-in. */
template <int E, int MT, int BK>
__global__ void
amp_attn_dec_mma_kernel(const __half *__restrict__ q,
                       const __half *__restrict__ kc,
                       const __half *__restrict__ vc,
                       const int *__restrict__ lenp, float *__restrict__ ws,
                       int S, int Hq, int Hk, int max_seq, int ns,
                       float scale)
{
	constexpr int W = AMP_ADM_WARPS;
	constexpr int MR = 16 * MT;        /* query block rows */
	constexpr int KS = E / 16;         /* score k-steps per (m, n) */
	constexpr int PNT = E / 8 / W;     /* PV n-tiles per warp */
	constexpr int PKS = BK / 16;       /* PV k-steps */
	constexpr int RB = E * 2;          /* K/V/Q tile row bytes */
	constexpr int EB = 128;            /* e-block row bytes (4 of 8
	                                     * pieces used; the swizzle
	                                     * needs >= 8 pieces/row) */
	constexpr int WSROW = 4 + E;
	/* softmax: 4 keys per lane, LPR lanes per row, RPW rows per warp
	 * step, NIT steps cover the MR rows */
	constexpr int LPR = BK / 4;
	constexpr int RPW = 32 / LPR;
	constexpr int NIT = (MR + W * RPW - 1) / (W * RPW);
	/* at BK = 32 an fp32 score row and an fp16 e row are both 128 bytes:
	 * the e-block overwrites the score block in place (each row is read
	 * into registers by its own lanes before they write it) */
	constexpr bool ALIAS = (BK * 4 == EB);

	/* scores: warp w takes NPW adjacent n-tiles (8 keys each) of one
	 * m-tile; its Q fragments are the same for every key tile and stay
	 * in registers (one block per SM leaves ~255 per thread) */
	constexpr int NT = BK / 8;
	constexpr int NPW = (MT * NT + W - 1) / W;

	static_assert(BK == 16 || BK == 32, "BK");
	static_assert(NT % NPW == 0, "a warp's n-tiles share one m-tile");

	const int j = blockIdx.x, sp = blockIdx.y;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

	const int rep = Hq / Hk;
	const int rows = rep * S;

	int len = *lenp;

	len = (len < 0) ? 0 : (len > max_seq ? max_seq : len);

	const int nse = amp_ad_nse(len, ns);

	if (sp >= nse)
		return;

	const int chunk = (len + nse - 1) / nse;
	const int begin = sp * chunk;
	const int end = min(len, begin + chunk);
	const int ntiles = (end - begin + BK - 1) / BK;

	extern __shared__ __half smem[];

	/* Q tile (MR x E h) | K 2 x (BK x E) | V 2 x (BK x E)
	 * | sblock (MR x BK f) | eblock (MR x 64 h, swizzled; aliases the
	 * sblock at BK = 32) | corr (MR f) */
	__half *qt = smem;
	__half *kt = qt + MR * E;
	__half *vt = kt + 2 * BK * E;
	float *sblock = (float *)(vt + 2 * BK * E);
	__half *eblock = ALIAS ? (__half *)sblock : (__half *)(sblock + MR * BK);
	/* corr sits AFTER the full eblock (MR rows x EB bytes): the
	 * eblock rows are 64 halves wide (8 swizzle pieces), not BK —
	 * (eblock + 32 * BK) landed on row 16 and row-16 e writes
	 * clobbered corr[0..15], zeroing the rows 0-15 pv rescale */
	float *corr = (float *)((char *)eblock + MR * EB);

	/* fill the Q tile (rows >= rows zeroed). 16-byte
	 * pieces, swizzled stores. */
	for (int i = threadIdx.x; i < MR * (E / 8); i += blockDim.x) {
		int row = i / (E / 8), piece = i % (E / 8);
		uint4 v = make_uint4(0u, 0u, 0u, 0u);

		if (row < rows) {
			int s = row % S, h = j * rep + row / S;
			const __half *src = q + ((size_t)s * Hq + h) * E +
				piece * 8;

			v = *(const uint4 *)src;
		}

		*(uint4 *)((char *)qt + amp_adm_swz(row, piece, RB)) = v;
	}

	/* softmax state of row it * W * RPW + warp * RPW + lane / LPR,
	 * replicated over the row's LPR lanes */
	float mst[NIT], lst[NIT];

	#pragma unroll
	for (int it = 0; it < NIT; it++) {
		mst[it] = -1e30f;
		lst[it] = 0.0f;
	}

	/* row s attends keys [0, len - S + s] (bottom-right causal) */
	const int lim_base = len - S;

	if (ntiles <= 0) {
		/* empty split: defined empty workspace rows */
		for (int r = warp; r < rows; r += W) {
			int s = r % S, g = r / S;
			float *w = ws + (((size_t)j * ns + sp) * rep * S +
				s * rep + g) * WSROW;

			if (lane == 0) {
				w[0] = -1e30f;
				w[1] = 0.0f;
			}

			/* the combine multiplies the accumulator by exp(m - M) = 0: it must be
			 * defined (0 * stale workspace NaN/inf = NaN) */
			for (int c = lane; c < E; c += 32)
				w[4 + c] = 0.0f;
		}

		return;
	}

	__syncthreads();   /* Q tile visible to all warps */

	const int s_m = (warp * NPW) / NT, s_n0 = ((warp * NPW) % NT) * 8;
	const bool s_on = s_m < MT && s_m * 16 < rows;
	unsigned qa[KS][4];

	if (s_on) {
		#pragma unroll
		for (int kk = 0; kk < KS; kk++)
			amp_adm_ldsm_x4(qa[kk], AMP_ADM_ADDR(qt,
				s_m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
				kk * 2 + (lane >> 4), RB));
	}

	/* prologue: stage tile 0 (K and V, one commit group). Keys
	 * past the used window are zero-filled in smem — the K rows are
	 * never copied, the V rows are out of the allocation, and a
	 * NaN in an unmasked-by-e product would poison valid rows. */
	{
		const __half *kg = kc + ((size_t)j * max_seq) * E;
		const __half *vg = vc + ((size_t)j * max_seq) * E;
		int nrows = min(BK, end - begin);
		int pieces = RB / 16;

		for (int i = threadIdx.x; i < BK * pieces; i += blockDim.x) {
			int key = i / pieces, piece = i % pieces;

			if (key < nrows) {
				amp_adm_cp(AMP_ADM_ADDR(kt, key, piece, RB),
					kg + (size_t)(begin + key) * E + piece * 8);

				amp_adm_cp(AMP_ADM_ADDR(vt, key, piece, RB),
					vg + (size_t)(begin + key) * E + piece * 8);
			} else {
				uint4 z = make_uint4(0u, 0u, 0u, 0u);

				*(uint4 *)((char *)kt + amp_adm_swz(key, piece, RB)) = z;

				*(uint4 *)((char *)vt + amp_adm_swz(key, piece, RB)) = z;
			}
		}

		amp_adm_cp_commit();
	}

	float pv[PNT][MT][4];

	#pragma unroll
	for (int n = 0; n < PNT; n++)
		#pragma unroll
		for (int m = 0; m < MT; m++)
			#pragma unroll
			for (int i = 0; i < 4; i++)
				pv[n][m][i] = 0.0f;

	for (int t = 0; t < ntiles; t++) {
		int tb = begin + t * BK;
		int stage = t & 1;
		const __half *kt_s = kt + stage * BK * E;
		const __half *vt_s = vt + stage * BK * E;

		/* stage tile t+1 (other buffer), then wait for tile t;
		 * keys past the window zero-filled (see prologue) */
		if (t + 1 < ntiles) {
			int tn = begin + (t + 1) * BK;
			const __half *kg = kc + ((size_t)j * max_seq) * E;
			const __half *vg = vc + ((size_t)j * max_seq) * E;
			__half *kd = kt + (stage ^ 1) * BK * E;
			__half *vd = vt + (stage ^ 1) * BK * E;
			int nrows = min(BK, end - tn);
			int pieces = RB / 16;

			for (int i = threadIdx.x; i < BK * pieces;
			     i += blockDim.x) {
				int key = i / pieces, piece = i % pieces;

				if (key < nrows) {
					amp_adm_cp(AMP_ADM_ADDR(kd, key, piece, RB),
						kg + (size_t)(tn + key) * E + piece * 8);

					amp_adm_cp(AMP_ADM_ADDR(vd, key, piece, RB),
						vg + (size_t)(tn + key) * E + piece * 8);
				} else {
					uint4 z = make_uint4(0u, 0u, 0u, 0u);

					*(uint4 *)((char *)kd +
						amp_adm_swz(key, piece, RB)) = z;

					*(uint4 *)((char *)vd +
						amp_adm_swz(key, piece, RB)) = z;
				}
			}
		}

		amp_adm_cp_commit();
		amp_adm_cp_wait<1>();
		__syncthreads();

		/* --- scores: warp w covers m-tile s_m (16 rows, Q in -----
		 * registers) and the NPW n-tiles (8 keys) from s_n0 ------- */
		if (s_on) {
			const int mrow = s_m * 16;
			float sc[NPW][4];

			/* mma accumulates into sc — it must start at zero for
			 * every tile (the pv accumulator is the opposite: it
			 * persists across tiles and is zeroed once above) */
			#pragma unroll
			for (int nn = 0; nn < NPW; nn++)
				#pragma unroll
				for (int i = 0; i < 4; i++)
					sc[nn][i] = 0.0f;

			#pragma unroll
			for (int kk = 0; kk < KS; kk++) {
				#pragma unroll
				for (int nn = 0; nn < NPW; nn++) {
					unsigned bf[2];
					int key = s_n0 + nn * 8 + (lane & 7);
					int piece2 = kk * 2 + ((lane >> 3) & 1);

					amp_adm_ldsm_x2(bf, AMP_ADM_ADDR(kt_s, key, piece2, RB));
					amp_adm_mma(sc[nn], qa[kk], bf);
				}
			}

			/* scatter with the scale and the causal mask. C layout:
			 * c[0..1] = row (lane>>2), cols 2*(lane&3) + {0,1};
			 * c[2..3] = row + 8, same cols. */
			int r0 = mrow + (lane >> 2);

			#pragma unroll
			for (int nn = 0; nn < NPW; nn++) {
				int c0 = s_n0 + nn * 8 + 2 * (lane & 3);

				#pragma unroll
				for (int half = 0; half < 2; half++) {
					int r = r0 + 8 * half;

					if (r >= rows)
						continue;

					int s = r % S;
					int lim = min(end, lim_base + s + 1);

					#pragma unroll
					for (int i = 0; i < 2; i++) {
						int c = c0 + i;
						bool ok = (tb + c) < lim;

						sblock[r * BK + c] =
							ok ? scale * sc[nn][2 * half + i]
							   : -1e30f;
					}
				}
			}
		}

		__syncthreads();

		/* --- softmax: LPR lanes per row, 4 keys per lane, ------
		 * RPW rows per warp step; the row's maximum and sum are ---
		 * reduced over its lanes ---------------------------------- */
		{
			const int sub = lane / LPR, cl = (lane % LPR) * 4;

			#pragma unroll
			for (int it = 0; it < NIT; it++) {
				int r = (it * W + warp) * RPW + sub;
				bool live = r < rows;
				int s = (live ? r : 0) % S;
				int lim = min(end, lim_base + s + 1);
				float v[4];

				if (live) {
					float4 f = *(const float4 *)&sblock[r * BK + cl];

					v[0] = f.x;
					v[1] = f.y;
					v[2] = f.z;
					v[3] = f.w;
				}

				float mx = -1e30f;

				#pragma unroll
				for (int i = 0; i < 4; i++) {
					if (!live || (tb + cl + i) >= lim)
						v[i] = -1e30f;

					mx = fmaxf(mx, v[i]);
				}

				#pragma unroll
				for (int off = LPR / 2; off > 0; off >>= 1)
					mx = fmaxf(mx, __shfl_xor_sync(
						0xffffffffu, mx, off));

				float m_new = fmaxf(mst[it], mx);
				float cr = __expf(mst[it] - m_new);
				float lsum = 0.0f, e[4];

				#pragma unroll
				for (int i = 0; i < 4; i++) {
					e[i] = __expf(v[i] - m_new);
					lsum += ((tb + cl + i) < lim) ? e[i] : 0.0f;
				}

				/* swizzled write (the PV ldmatrix reads it swizzled);
				 * every lane of the row has read its scores (the
				 * shuffles above), so the aliased write is safe */
				if (live) {
					__half2 eh[2] = { __floats2half2_rn(e[0], e[1]),
						__floats2half2_rn(e[2], e[3]) };

					*(uint2 *)((char *)eblock + amp_adm_swz(r, cl >> 3, EB)
						+ (cl & 7) * 2) = *(const uint2 *)eh;
				}

				#pragma unroll
				for (int off = LPR / 2; off > 0; off >>= 1)
					lsum += __shfl_xor_sync(
						0xffffffffu, lsum, off);

				lst[it] = lst[it] * cr + lsum;
				mst[it] = m_new;

				if (live && (lane % LPR) == 0)
					corr[r] = cr;
			}
		}

		__syncthreads();

		/* --- PV: warp w covers n-tiles w, w+W, ... ------------ */
		/* per k-step: the V fragments of the warp's n-tiles once, the
		 * e-block fragment of each m-tile once */
		{
			float cr0[MT], cr8[MT];

			#pragma unroll
			for (int m = 0; m < MT; m++) {
				int r0 = m * 16 + (lane >> 2);

				cr0[m] = (r0 < rows) ? corr[r0] : 0.0f;
				cr8[m] = (r0 + 8 < rows) ? corr[r0 + 8] : 0.0f;
			}

			#pragma unroll
			for (int n = 0; n < PNT; n++)
				#pragma unroll
				for (int m = 0; m < MT; m++) {
					pv[n][m][0] *= cr0[m];
					pv[n][m][1] *= cr0[m];
					pv[n][m][2] *= cr8[m];
					pv[n][m][3] *= cr8[m];
				}

			#pragma unroll
			for (int kk = 0; kk < PKS; kk++) {
				unsigned bv[PNT][2];

				/* B = V tile: keys kk*16 + {0..15}, dims nt*8
				 * (one 16-B piece) */
				#pragma unroll
				for (int n = 0; n < PNT; n++) {
					int vkey = kk * 16 + (lane & 7) +
						((lane >> 3) & 1) * 8;

					amp_adm_ldsm_x2_t(bv[n],
						AMP_ADM_ADDR(vt_s, vkey, warp + n * W, RB));
				}

				#pragma unroll
				for (int m = 0; m < MT; m++) {
					if (m * 16 >= rows)
						continue;

					unsigned af[4];

					/* A = e-block rows of m-tile m, keys kk*16 */
					int arow = m * 16 +
						(lane & 7) + ((lane >> 3) & 1) * 8;
					int apiece = kk * 2 + (lane >> 4);

					amp_adm_ldsm_x4(af,
						AMP_ADM_ADDR(eblock, arow, apiece, EB));

					#pragma unroll
					for (int n = 0; n < PNT; n++)
						amp_adm_mma(pv[n][m], af, bv[n]);
				}
			}
		}

		__syncthreads();
	}

	/* --- workspace ------------------------------------------------ */
	#pragma unroll
	for (int it = 0; it < NIT; it++) {
		int r = (it * W + warp) * RPW + lane / LPR;

		if (r >= rows || (lane % LPR))
			continue;

		int s = r % S, g = r / S;
		int lim = min(end, lim_base + s + 1);
		bool any = begin < lim;
		float *w = ws + (((size_t)j * ns + sp) * rep * S +
			s * rep + g) * WSROW;

		w[0] = any ? mst[it] : -1e30f;
		w[1] = any ? lst[it] : 0.0f;
	}

	/* acc dims: warp w's PV n-tiles, fragment scatter. c[0..1] =
	 * row (lane>>2), c[2..3] = row + 8; cols 2*(lane&3) + {0,1}. */
	#pragma unroll
	for (int n = 0; n < PNT; n++) {
		int nt = warp + n * W;
		int c0 = nt * 8 + 2 * (lane & 3);

		#pragma unroll
		for (int m = 0; m < MT; m++) {
			int r0 = m * 16 + (lane >> 2);

			#pragma unroll
			for (int half = 0; half < 2; half++) {
				int r = r0 + 8 * half;

				if (r >= rows)
					continue;

				int s = r % S, g = r / S;
				float *w = ws + (((size_t)j * ns + sp) *
					rep * S + s * rep + g) * WSROW;

				w[4 + c0] = pv[n][m][2 * half];
				w[4 + c0 + 1] = pv[n][m][2 * half + 1];
			}
		}
	}
}

/* shared memory of amp_attn_dec_mma_kernel<E, MT, BK> */
static size_t
amp_adm_smem(int E, int MT, int BK)
{
	int MR = 16 * MT;

	return (size_t)(MR * E * 2 +              /* Q tile */
		2 * BK * E * 2 +                  /* K stages */
		2 * BK * E * 2 +                  /* V stages */
		MR * BK * 4 +                     /* sblock */
		(BK * 4 == 128 ? 0 : MR * 128) +  /* eblock (aliases the sblock at BK = 32) */
		MR * 4 +                          /* corr */
		16);                              /* alignment slack */
}

extern "C" size_t
amp_attn_dec_mma_smem(int E)
{
	/* the original 32-row layout with a separate e-block: the QSA kernel's */
	return (size_t)(32 * E * 2 + 4 * AMP_ADM_BK * E * 2 + 32 * AMP_ADM_BK * 4 +
		32 * 128 + 32 * 4 + 16);
}

extern "C" int
amp_attn_dec_mma_fp16(const void *q, const void *k, const void *v,
                     const int *len, void *ws, void *out, int S, int Hq,
                     int Hk, int E, int max_seq, int ns, float scale,
                     cudaStream_t stream)
{
	if (Hk <= 0 || Hq % Hk || S < 1 || Hq / Hk > 32 ||
	    (Hq / Hk) * S > 64 || ns < 1 || ns > AMP_AD_MAXNS ||
	    (E != 64 && E != 128 && E != 256) || (E / 8) % AMP_ADM_WARPS) {
		fprintf(stderr, "amp_attn_dec_mma: unsupported config "
			"(S %d, Hq %d, Hk %d, E %d, ns %d)\n",
			S, Hq, Hk, E, ns);
		return -1;
	}

	dim3 grid(Hk, ns);
	int block = AMP_ADM_WARPS * 32;
	int rows = (Hq / Hk) * S;

#define AMP_ADM_K(EV, MTV, BKV) do { \
	/* the >48 KiB dynamic shared memory opt-in is per device */ \
	static bool attr_done[64] = { false }; \
	size_t smem = amp_adm_smem(EV, MTV, BKV); \
	int adm_dev = 0; \
	cudaGetDevice(&adm_dev); \
	if (adm_dev >= 0 && adm_dev < 64 && !attr_done[adm_dev]) { \
		cudaFuncSetAttribute(amp_attn_dec_mma_kernel<EV, MTV, BKV>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
		attr_done[adm_dev] = true; \
	} \
	amp_attn_dec_mma_kernel<EV, MTV, BKV><<<grid, block, smem, stream>>>( \
		(const __half *)q, (const __half *)k, (const __half *)v, \
		len, (float *)ws, S, Hq, Hk, max_seq, ns, (float)scale); \
} while (0)
#define AMP_ADM_E(EV) do { \
	if (rows <= 32) \
		AMP_ADM_K(EV, 2, AMP_ADM_BK); \
	else if (rows <= 48) \
		AMP_ADM_K(EV, 3, AMP_ADM_BK); \
	else \
		AMP_ADM_K(EV, 4, 16); \
} while (0)

	if (E == 256)
		AMP_ADM_E(256);
	else if (E == 128)
		AMP_ADM_E(128);
	else
		AMP_ADM_E(64);

#undef AMP_ADM_E
#undef AMP_ADM_K

	amp_attn_dec_combine((const float *)ws, (__half *)out, S, Hq, Hk, E, ns, len,
		max_seq, stream);

	return (int)cudaGetLastError();
}

/* ------------------------------------------------------------------ */
/* Gathered GQA attention (Qwen3.8-Flash-Next QSA).                    */
/*                                                                     */
/* Every query row attends to its OWN list of cache rows (the selected */
/* blocks' tokens plus the tail, -1 = padding). Same tile pipeline as  */
/* amp_attn_dec_mma_kernel, with three differences: the K/V tile rows   */
/* are gathered through the token list (cache layout (max_seq, Hk, E), */
/* so a row is Hk*E halves apart), the mask is the list's validity     */
/* instead of a causal window, and the grid is (query, kv head, split) */
/* with rep = Hq/Hk query rows per block (<= 32). Split results go to  */
/* a workspace and amp_qsa_combine_kernel merges them.                  */
/* ------------------------------------------------------------------ */

#define AMP_QSA_BK 32

template <int E>
__global__ void
amp_qsa_attn_kernel(const __half *__restrict__ q,
                   const __half *__restrict__ kc,
                   const __half *__restrict__ vc,
                   const int *__restrict__ idx, float *__restrict__ ws,
                   int Hq, int Hk, int Kt, int ns, float scale)
{
	constexpr int BK = AMP_QSA_BK;
	constexpr int W = AMP_ADM_WARPS;
	constexpr int KS = E / 16;
	constexpr int PNT = E / 8 / W;
	constexpr int PKS = BK / 16;
	constexpr int RB = E * 2;
	constexpr int EB = 128;
	constexpr int WSROW = 4 + E;

	const int r = blockIdx.x, j = blockIdx.y, sp = blockIdx.z;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int rep = Hq / Hk;
	const int rows = rep;

	const int ntiles_all = (Kt + BK - 1) / BK;
	const int tps = (ntiles_all + ns - 1) / ns;
	const int t0 = sp * tps;
	const int t1 = min(ntiles_all, t0 + tps);
	const int ntiles = t1 - t0;
	const int *ix = idx + (size_t)r * Kt;
	const size_t rstride = (size_t)Hk * E;       /* halves between cache rows */

	extern __shared__ __half smem[];

	__half *qt = smem;
	__half *kt = qt + 32 * E;
	__half *vt = kt + 2 * BK * E;
	float *sblock = (float *)(vt + 2 * BK * E);
	__half *eblock = (__half *)(sblock + 32 * BK);
	float *corr = (float *)((char *)eblock + 32 * EB);

	__shared__ unsigned char vflag[2][BK];

	float *wbase = ws + (((size_t)r * Hk + j) * ns + sp) * rep * WSROW;

	if (ntiles <= 0) {
		for (int g = warp; g < rows; g += W) {
			if (lane == 0) {
				wbase[(size_t)g * WSROW] = -1e30f;
				wbase[(size_t)g * WSROW + 1] = 0.0f;
			}

			/* defined accumulator: the combine computes acc * exp(m - M) = acc * 0, and
			 * 0 * stale workspace NaN/inf is NaN (poisoned every later decode step) */
			for (int c = lane; c < E; c += 32)
				wbase[(size_t)g * WSROW + 4 + c] = 0.0f;
		}

		return;
	}

	/* Q tile: rows [0, rep) are this block's heads (j*rep + row) */
	for (int i = threadIdx.x; i < 32 * (E / 8); i += blockDim.x) {
		int row = i / (E / 8), piece = i % (E / 8);
		uint4 v = make_uint4(0u, 0u, 0u, 0u);

		if (row < rows)
			v = *(const uint4 *)(q + ((size_t)r * Hq + j * rep + row) * E + piece * 8);

		*(uint4 *)((char *)qt + amp_adm_swz(row, piece, RB)) = v;
	}

	float mst[4], lst[4];
	int nown = 0;

	for (int rr = warp; rr < rows; rr += W) {
		mst[nown] = -1e30f;
		lst[nown] = 0.0f;
		nown++;
	}

	__syncthreads();

	const int pieces = RB / 16;

	/* stage tile `tile` into buffer `stage`: gathered rows, zero fill for padding */
	auto stage_tile = [&](int tile, int stage) {
		__half *kd = kt + stage * BK * E;
		__half *vd = vt + stage * BK * E;
		int base = tile * BK;

		for (int i = threadIdx.x; i < BK * pieces; i += blockDim.x) {
			int key = i / pieces, piece = i % pieces;
			int tok = (base + key < Kt) ? ix[base + key] : -1;

			if (piece == 0)
				vflag[stage][key] = (tok >= 0) ? 1 : 0;

			if (tok >= 0) {
				size_t off = (size_t)tok * rstride + (size_t)j * E + piece * 8;

				amp_adm_cp(AMP_ADM_ADDR(kd, key, piece, RB), kc + off);
				amp_adm_cp(AMP_ADM_ADDR(vd, key, piece, RB), vc + off);
			} else {
				uint4 z = make_uint4(0u, 0u, 0u, 0u);

				*(uint4 *)((char *)kd + amp_adm_swz(key, piece, RB)) = z;
				*(uint4 *)((char *)vd + amp_adm_swz(key, piece, RB)) = z;
			}
		}
	};

	stage_tile(t0, 0);
	amp_adm_cp_commit();

	float pv[PNT][2][4];

	#pragma unroll
	for (int n = 0; n < PNT; n++)
		#pragma unroll
		for (int m = 0; m < 2; m++)
			#pragma unroll
			for (int i = 0; i < 4; i++)
				pv[n][m][i] = 0.0f;

	for (int t = 0; t < ntiles; t++) {
		int stage = t & 1;
		const __half *kt_s = kt + stage * BK * E;
		const __half *vt_s = vt + stage * BK * E;

		if (t + 1 < ntiles)
			stage_tile(t0 + t + 1, stage ^ 1);

		amp_adm_cp_commit();
		amp_adm_cp_wait<1>();
		__syncthreads();

		/* scores: warp w -> m-tile w/(BK/8), n-tile w%(BK/8) */
		{
			const int mrow = (warp / (BK / 8)) * 16;
			const int n0 = (warp % (BK / 8)) * 8;

			float sc[4];
			unsigned af[4], bf[2];

			#pragma unroll
			for (int i = 0; i < 4; i++)
				sc[i] = 0.0f;

			#pragma unroll
			for (int kk = 0; kk < KS; kk++) {
				int row = mrow + (lane & 7) + ((lane >> 3) & 1) * 8;
				int piece = kk * 2 + (lane >> 4);

				amp_adm_ldsm_x4(af, AMP_ADM_ADDR(qt, row, piece, RB));

				int key = n0 + (lane & 7);
				int piece2 = kk * 2 + ((lane >> 3) & 1);

				amp_adm_ldsm_x2(bf, AMP_ADM_ADDR(kt_s, key, piece2, RB));

				amp_adm_mma(sc, af, bf);
			}

			int r0 = mrow + (lane >> 2);
			int c0 = n0 + 2 * (lane & 3);

			#pragma unroll
			for (int half = 0; half < 2; half++) {
				int rr = r0 + 8 * half;

				if (rr >= rows)
					continue;

				#pragma unroll
				for (int i = 0; i < 2; i++) {
					int c = c0 + i;

					sblock[rr * BK + c] = vflag[stage][c] ? scale * sc[2 * half + i] : -1e30f;
				}
			}
		}

		__syncthreads();

		/* softmax: warp w owns rows rr % W == w */
		{
			int ri = 0;

			for (int rr = warp; rr < rows; rr += W, ri++) {
				float mx = -1e30f;

				for (int c = lane; c < BK; c += 32) {
					float v = vflag[stage][c] ? sblock[rr * BK + c] : -1e30f;

					mx = fmaxf(mx, v);
				}

				#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
					mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));

				float m_new = fmaxf(mst[ri], mx);
				float cr = __expf(mst[ri] - m_new);
				float lsum = 0.0f;

				for (int c = lane; c < BK; c += 32) {
					bool ok = vflag[stage][c] != 0;
					float v = ok ? sblock[rr * BK + c] : -1e30f;
					float e = ok ? __expf(v - m_new) : 0.0f;

					*(__half *)((char *)eblock + amp_adm_swz(rr, c >> 3, EB) + (c & 7) * 2) =
						__float2half(e);

					lsum += e;
				}

				#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
					lsum += __shfl_xor_sync(0xffffffffu, lsum, off);

				lst[ri] = lst[ri] * cr + lsum;
				mst[ri] = m_new;

				if (lane == 0)
					corr[rr] = cr;
			}
		}

		__syncthreads();

		/* PV */
		{
			float cr0[2], cr8[2];

			#pragma unroll
			for (int m = 0; m < 2; m++) {
				int r0 = m * 16 + (lane >> 2);

				cr0[m] = (r0 < rows) ? corr[r0] : 0.0f;
				cr8[m] = (r0 + 8 < rows) ? corr[r0 + 8] : 0.0f;
			}

			#pragma unroll
			for (int n = 0; n < PNT; n++) {
				int nt = warp + n * W;

				#pragma unroll
				for (int m = 0; m < 2; m++) {
					pv[n][m][0] *= cr0[m];
					pv[n][m][1] *= cr0[m];
					pv[n][m][2] *= cr8[m];
					pv[n][m][3] *= cr8[m];

					#pragma unroll
					for (int kk = 0; kk < PKS; kk++) {
						unsigned af[4], bf[2];
						int arow = m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
						int apiece = kk * 2 + (lane >> 4);

						amp_adm_ldsm_x4(af, AMP_ADM_ADDR(eblock, arow, apiece, EB));

						int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;

						amp_adm_ldsm_x2_t(bf, AMP_ADM_ADDR(vt_s, vkey, nt, RB));
						amp_adm_mma(pv[n][m], af, bf);
					}
				}
			}
		}

		__syncthreads();
	}

	/* workspace rows: (m, l, -, -, acc[E]) per head of the block */
	{
		int ri = 0;

		for (int rr = warp; rr < rows; rr += W, ri++)
			if (lane == 0) {
				wbase[(size_t)rr * WSROW] = mst[ri];
				wbase[(size_t)rr * WSROW + 1] = lst[ri];
			}
	}

	#pragma unroll
	for (int n = 0; n < PNT; n++) {
		int nt = warp + n * W;
		int c0 = nt * 8 + 2 * (lane & 3);

		#pragma unroll
		for (int m = 0; m < 2; m++) {
			int r0 = m * 16 + (lane >> 2);

			#pragma unroll
			for (int half = 0; half < 2; half++) {
				int rr = r0 + 8 * half;

				if (rr >= rows)
					continue;

				float *w = wbase + (size_t)rr * WSROW;

				w[4 + c0] = pv[n][m][2 * half];
				w[4 + c0 + 1] = pv[n][m][2 * half + 1];
			}
		}
	}
}

/* Merge the splits: grid (R * Hq), E threads; out (R, Hq, E) fp16. */
template <int E>
__global__ void
amp_qsa_combine_kernel(const float *__restrict__ ws, __half *__restrict__ out,
                      int Hq, int Hk, int ns)
{
	constexpr int WSROW = 4 + E;
	const int rh = blockIdx.x;
	const int r = rh / Hq, h = rh % Hq;
	const int rep = Hq / Hk, j = h / rep, g = h % rep;
	const int d = threadIdx.x;

	float M = -1e30f;

	for (int s = 0; s < ns; s++) {
		const float *w = ws + ((((size_t)r * Hk + j) * ns + s) * rep + g) * WSROW;

		M = fmaxf(M, w[0]);
	}

	float L = 0.0f, acc = 0.0f;

	for (int s = 0; s < ns; s++) {
		const float *w = ws + ((((size_t)r * Hk + j) * ns + s) * rep + g) * WSROW;
		float f = __expf(w[0] - M);

		L += w[1] * f;
		acc += w[4 + d] * f;
	}

	out[((size_t)r * Hq + h) * E + d] = __float2half(L > 0.0f ? acc / L : 0.0f);
}

extern "C" long
amp_qsa_attn_ws_floats(int R, int Hq, int Hk, int E, int ns)
{
	return (long)R * Hk * ns * (Hq / Hk) * (4 + E);
}

extern "C" int
amp_qsa_attn_fp16(const void *q, const void *kc, const void *vc, const int *idx,
                 void *ws, void *out, int R, int Hq, int Hk, int E, int Kt,
                 int ns, float scale, cudaStream_t stream)
{
	if (R < 1 || Hk <= 0 || Hq % Hk || Hq / Hk > 16 || Kt < 1 || ns < 1 ||
	    (E != 64 && E != 128 && E != 256) || (E / 8) % AMP_ADM_WARPS) {
		fprintf(stderr, "amp_qsa_attn: unsupported config (R %d, Hq %d, Hk %d, E %d, Kt %d, ns %d)\n",
			R, Hq, Hk, E, Kt, ns);
		return -1;
	}

	dim3 grid(R, Hk, ns);
	int block = AMP_ADM_WARPS * 32;
	size_t smem = amp_attn_dec_mma_smem(E);

#define AMP_QSA_K(EV) do { \
	static bool attr_done[64] = { false }; \
	int qdev = 0; \
	cudaGetDevice(&qdev); \
	if (qdev >= 0 && qdev < 64 && !attr_done[qdev]) { \
		cudaFuncSetAttribute(amp_qsa_attn_kernel<EV>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, (int)amp_attn_dec_mma_smem(EV)); \
		attr_done[qdev] = true; \
	} \
	amp_qsa_attn_kernel<EV><<<grid, block, smem, stream>>>((const __half *)q, \
		(const __half *)kc, (const __half *)vc, idx, (float *)ws, Hq, Hk, Kt, ns, scale); \
	amp_qsa_combine_kernel<EV><<<(unsigned)R * Hq, EV, 0, stream>>>( \
		(const float *)ws, (__half *)out, Hq, Hk, ns); \
} while (0)

	if (E == 256)
		AMP_QSA_K(256);
	else if (E == 128)
		AMP_QSA_K(128);
	else
		AMP_QSA_K(64);

#undef AMP_QSA_K

	return (int)cudaGetLastError();
}


/* ------------------------------------------------------------------ */
/* Block-scaled FP8 weights (DeepSeek-V4.1 checkpoints)                */
/*
 * w (N, K) float8_e4m3fn, s (N/32, K/32) float8_e8m0fnu: one power-of-two scale per 32x32
 * weight block, value = 2^(e - 127).
 */

/* 2^(e - 127) * 256: the e4m3 -> fp16 conversion below leaves the value divided by 256 (see there) */
static __device__ __forceinline__ float
amp_e8m0(uint8_t e)
{
	return __uint_as_float(((uint32_t)e + 8u) << 23);
}

/* e4m3 byte (s eeee mmm) -> fp16 bits without the exponent bias correction: (b & 0x7f) << 7 puts eeee at the
 * low exponent bits and mmm at the top mantissa bits, so the half holds the e4m3 value / 256 -- exactly,
 * subnormals included (the e4m3 subnormal m * 2^-9 and the half subnormal m * 2^-17 differ by 2^8 too).
 * sm_86 has no fp8 conversion instruction; the library routine costs ~10 instructions per element and made
 * the gemv ALU-bound at a third of the memory bandwidth. */
static __device__ __forceinline__ void
amp_fp8x16_to_f32(const uint4 &v, float *f)
{
	const uint32_t u[4] = { v.x, v.y, v.z, v.w };

#pragma unroll
	for (int i = 0; i < 4; i++) {
		const uint32_t w = u[i];
		const uint32_t lo = ((w & 0x007f007fu) << 7) | ((w & 0x00800080u) << 8);       /* bytes 0 and 2 */
		const uint32_t hi = (((w >> 8) & 0x007f007fu) << 7) | (((w >> 8) & 0x00800080u) << 8);   /* bytes 1 and 3 */
		/* pairs (byte0, byte1) and (byte2, byte3) in element order */
		const uint32_t p01 = (lo & 0xffffu) | (hi << 16);
		const uint32_t p23 = (lo >> 16) | (hi & 0xffff0000u);
		const float2 a = __half22float2(*(const __half2 *)&p01), b = __half22float2(*(const __half2 *)&p23);

		f[4 * i + 0] = a.x;
		f[4 * i + 1] = a.y;
		f[4 * i + 2] = b.x;
		f[4 * i + 3] = b.y;
	}
}

/* One warp per output row; each lane takes 16 consecutive k per step. M <= 8 rows of x (fp16). */
template <int M>
__global__ void amp_fp8_gemv_kernel(const __half *__restrict__ x, const uint8_t *__restrict__ w,
                                    const uint8_t *__restrict__ s, __half *__restrict__ out,
                                    int K, int N)
{
	const int lane = threadIdx.x & 31;
	const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);

	if (n >= N)
		return;

	const uint8_t *wr = w + (size_t)n * K;
	const uint8_t *sr = s + (size_t)(n >> 5) * (K >> 5);
	float acc[M];

#pragma unroll
	for (int m = 0; m < M; m++)
		acc[m] = 0.0f;

	for (int k = lane * 16; k < K; k += 512) {
		uint4 wv = *(const uint4 *)(wr + k);
		float wf[16];

		amp_fp8x16_to_f32(wv, wf);

		const float sc = amp_e8m0(sr[k >> 5]);

#pragma unroll
		for (int m = 0; m < M; m++) {
			const uint4 *xp = (const uint4 *)(x + (size_t)m * K + k);
			uint4 x0 = xp[0], x1 = xp[1];
			const __half2 *h0 = (const __half2 *)&x0, *h1 = (const __half2 *)&x1;
			float d = 0.0f;

#pragma unroll
			for (int i = 0; i < 4; i++) {
				float2 a = __half22float2(h0[i]), b = __half22float2(h1[i]);

				d += wf[2 * i] * a.x + wf[2 * i + 1] * a.y;
				d += wf[8 + 2 * i] * b.x + wf[8 + 2 * i + 1] * b.y;
			}

			acc[m] += d * sc;
		}
	}

#pragma unroll
	for (int m = 0; m < M; m++) {
		float v = acc[m];

#pragma unroll
		for (int o = 16; o > 0; o >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, o);

		if (lane == 0)
			out[(size_t)m * N + n] = __float2half(v);
	}
}

/* Two output rows per warp, the weight loads of the next step issued before the current one is consumed
 * (memory-level parallelism: the first version read ~300 GB/s of the ~750 GB/s of an A5000). */
template <int M>
__global__ void __launch_bounds__(256)
amp_fp8_gemv2_kernel(const __half *__restrict__ x, const uint8_t *__restrict__ w, const uint8_t *__restrict__ s,
                     __half *__restrict__ out, int K, int N, int xs, int os)
{
	const int lane = threadIdx.x & 31;
	const int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2;

	/* blockIdx.z: weight group (the attention's wo_a); a plain call has one */
	x += (size_t)blockIdx.z * K;
	w += (size_t)blockIdx.z * N * K;
	s += (size_t)blockIdx.z * (N >> 5) * (K >> 5);
	out += (size_t)blockIdx.z * N;

	if (n0 >= N)
		return;

	const bool two = (n0 + 1 < N);
	const uint8_t *wr0 = w + (size_t)n0 * K, *wr1 = two ? wr0 + K : wr0;
	const uint8_t *sr0 = s + (size_t)(n0 >> 5) * (K >> 5), *sr1 = s + (size_t)((n0 + 1) >> 5) * (K >> 5);
	float acc0[M], acc1[M];

#pragma unroll
	for (int m = 0; m < M; m++)
		acc0[m] = acc1[m] = 0.0f;

	int k = lane * 16;
	uint4 wa = make_uint4(0, 0, 0, 0), wb = wa;

	if (k < K) {
		wa = *(const uint4 *)(wr0 + k);
		wb = *(const uint4 *)(wr1 + k);
	}

	for (; k < K; k += 512) {
		const int kn = k + 512;
		uint4 na = make_uint4(0, 0, 0, 0), nb = na;

		if (kn < K) {                                  /* prefetch the next step */
			na = *(const uint4 *)(wr0 + kn);
			nb = *(const uint4 *)(wr1 + kn);
		}

		const float sc0 = amp_e8m0(sr0[k >> 5]), sc1 = amp_e8m0(sr1[k >> 5]);
		float f0[16], f1[16];

		amp_fp8x16_to_f32(wa, f0);
		amp_fp8x16_to_f32(wb, f1);

#pragma unroll
		for (int m = 0; m < M; m++) {
			const uint4 *xp = (const uint4 *)(x + (size_t)m * xs + k);
			uint4 x0 = xp[0], x1 = xp[1];
			const __half2 *h0 = (const __half2 *)&x0, *h1 = (const __half2 *)&x1;
			float d0 = 0.0f, d1 = 0.0f;

#pragma unroll
			for (int i = 0; i < 4; i++) {
				float2 a = __half22float2(h0[i]), b = __half22float2(h1[i]);

				d0 += f0[2 * i] * a.x + f0[2 * i + 1] * a.y + f0[8 + 2 * i] * b.x + f0[8 + 2 * i + 1] * b.y;
				d1 += f1[2 * i] * a.x + f1[2 * i + 1] * a.y + f1[8 + 2 * i] * b.x + f1[8 + 2 * i + 1] * b.y;
			}

			acc0[m] += d0 * sc0;
			acc1[m] += d1 * sc1;
		}

		wa = na;
		wb = nb;
	}

#pragma unroll
	for (int m = 0; m < M; m++) {
		float v0 = acc0[m], v1 = acc1[m];

#pragma unroll
		for (int o = 16; o > 0; o >>= 1) {
			v0 += __shfl_xor_sync(0xffffffffu, v0, o);
			v1 += __shfl_xor_sync(0xffffffffu, v1, o);
		}

		if (lane == 0) {
			out[(size_t)m * os + n0] = __float2half(v0);

			if (two)
				out[(size_t)m * os + n0 + 1] = __float2half(v1);
		}
	}
}

/* M = 1: one row per warp, every load of the row issued up front (NS = ceil(K / 512) steps of 16 weights per
 * lane): the rows are short (1.3 - 8 KB), a dependent chain of loads per row left the memory system idle. */
template <int NS>
__global__ void __launch_bounds__(256)
amp_fp8_gemv3_kernel(const __half *__restrict__ x, const uint8_t *__restrict__ w, const uint8_t *__restrict__ s,
                     __half *__restrict__ out, int K, int N)
{
	extern __shared__ uint4 g3_x[];                 /* the activation row, shared by the 8 rows of the block */

	x += (size_t)blockIdx.z * K;                    /* blockIdx.z: weight group */
	w += (size_t)blockIdx.z * N * K;
	s += (size_t)blockIdx.z * (N >> 5) * (K >> 5);
	out += (size_t)blockIdx.z * N;

	const int lane = threadIdx.x & 31;
	const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	const uint8_t *wr = w + (size_t)min(n, N - 1) * K;
	const uint8_t *sr = s + (size_t)(min(n, N - 1) >> 5) * (K >> 5);
	uint4 wv[NS];
	uint8_t sv[NS];

	/* the weight loads first: they do not depend on x */
#pragma unroll
	for (int i = 0; i < NS; i++) {
		const int k = i * 512 + lane * 16;

		if (k < K) {
			wv[i] = *(const uint4 *)(wr + k);
			sv[i] = sr[k >> 5];
		}
	}

	for (int c = threadIdx.x; c < K / 8; c += blockDim.x)
		g3_x[c] = ((const uint4 *)x)[c];

	__syncthreads();

	if (n >= N)
		return;

	float acc = 0.0f;

#pragma unroll
	for (int i = 0; i < NS; i++) {
		const int k = i * 512 + lane * 16;

		if (k >= K)
			continue;

		float f[16];

		amp_fp8x16_to_f32(wv[i], f);

		const uint4 xa = g3_x[k >> 3], xb = g3_x[(k >> 3) + 1];
		const __half2 *h0 = (const __half2 *)&xa, *h1 = (const __half2 *)&xb;
		float d = 0.0f;

#pragma unroll
		for (int j = 0; j < 4; j++) {
			float2 a = __half22float2(h0[j]), b = __half22float2(h1[j]);

			d += f[2 * j] * a.x + f[2 * j + 1] * a.y + f[8 + 2 * j] * b.x + f[8 + 2 * j + 1] * b.y;
		}

		acc += d * amp_e8m0(sv[i]);
	}

#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		acc += __shfl_xor_sync(0xffffffffu, acc, o);

	if (lane == 0)
		out[n] = __float2half(acc);
}

extern "C" int
amp_fp8_gemv(const void *x, const void *w, const void *s, void *out, int M, int K, int N,
             cudaStream_t stream)
{
	if (M < 1 || M > 8 || K < 32 || (K % 32) || N < 1)
		return (int)cudaErrorInvalidValue;

	if (M == 1 && K <= 16 * 512) {
		const int ns = (K + 511) / 512;
		dim3 g1((N + 7) / 8), b1(256);

#define AMP_FP8G3(NSV) case NSV: amp_fp8_gemv3_kernel<NSV><<<g1, b1, (size_t)K * 2, stream>>>((const __half *)x, \
	(const uint8_t *)w, (const uint8_t *)s, (__half *)out, K, N); return (int)cudaGetLastError()

		switch (ns) {
		AMP_FP8G3(1); AMP_FP8G3(2); AMP_FP8G3(3); AMP_FP8G3(4); AMP_FP8G3(5); AMP_FP8G3(6); AMP_FP8G3(7);
		AMP_FP8G3(8); AMP_FP8G3(9); AMP_FP8G3(10); AMP_FP8G3(11); AMP_FP8G3(12); AMP_FP8G3(13); AMP_FP8G3(14);
		AMP_FP8G3(15); AMP_FP8G3(16);
		}

#undef AMP_FP8G3
	}

	dim3 grid((N + 15) / 16), block(256);

#define AMP_FP8G(MM) amp_fp8_gemv2_kernel<MM><<<grid, block, 0, stream>>>((const __half *)x, \
	(const uint8_t *)w, (const uint8_t *)s, (__half *)out, K, N, K, N); break

	switch (M) {
	case 1: AMP_FP8G(1);
	case 2: AMP_FP8G(2);
	case 3: AMP_FP8G(3);
	case 4: AMP_FP8G(4);
	case 5: AMP_FP8G(5);
	case 6: AMP_FP8G(6);
	case 7: AMP_FP8G(7);
	default: AMP_FP8G(8);
	}

#undef AMP_FP8G

	return (int)cudaGetLastError();
}

/* G weight groups in one launch: group g multiplies columns [g K, (g + 1) K) of x (M, G K) by its own (N, K) weight
 * (rows g N .. of w) into columns [g N, (g + 1) N) of out (M, G N). M <= 8, N % 32 == 0. */
extern "C" int
amp_fp8_gemv_grouped(const void *x, const void *w, const void *s, void *out, int M, int K, int N, int G,
                     cudaStream_t stream)
{
	if (M < 1 || M > 8 || K < 32 || (K % 32) || N < 32 || (N % 32) || G < 1 || G > 65535)
		return (int)cudaErrorInvalidValue;

	if (M == 1 && K <= 16 * 512) {
		const int ns = (K + 511) / 512;
		dim3 g1((N + 7) / 8, 1, G), b1(256);

#define AMP_FP8GG3(NSV) case NSV: amp_fp8_gemv3_kernel<NSV><<<g1, b1, (size_t)K * 2, stream>>>((const __half *)x, \
	(const uint8_t *)w, (const uint8_t *)s, (__half *)out, K, N); return (int)cudaGetLastError()

		switch (ns) {
		AMP_FP8GG3(1); AMP_FP8GG3(2); AMP_FP8GG3(3); AMP_FP8GG3(4); AMP_FP8GG3(5); AMP_FP8GG3(6); AMP_FP8GG3(7);
		AMP_FP8GG3(8); AMP_FP8GG3(9); AMP_FP8GG3(10); AMP_FP8GG3(11); AMP_FP8GG3(12); AMP_FP8GG3(13); AMP_FP8GG3(14);
		AMP_FP8GG3(15); AMP_FP8GG3(16);
		}

#undef AMP_FP8GG3
	}

	dim3 grid((N + 15) / 16, 1, G), block(256);

#define AMP_FP8GG(MM) amp_fp8_gemv2_kernel<MM><<<grid, block, 0, stream>>>((const __half *)x, \
	(const uint8_t *)w, (const uint8_t *)s, (__half *)out, K, N, G * K, G * N); break

	switch (M) {
	case 1: AMP_FP8GG(1);
	case 2: AMP_FP8GG(2);
	case 3: AMP_FP8GG(3);
	case 4: AMP_FP8GG(4);
	case 5: AMP_FP8GG(5);
	case 6: AMP_FP8GG(6);
	case 7: AMP_FP8GG(7);
	default: AMP_FP8GG(8);
	}

#undef AMP_FP8GG

	return (int)cudaGetLastError();
}

__global__ void amp_fp8_dequant_kernel(const uint8_t *__restrict__ w, const uint8_t *__restrict__ s,
                                       __half *__restrict__ out, int K, long n16)
{
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n16)
		return;

	long e = i * 16;
	int n = (int)(e / K), k = (int)(e % K);
	uint4 wv = *(const uint4 *)(w + e);
	float wf[16];

	amp_fp8x16_to_f32(wv, wf);

	const float sc = amp_e8m0(s[(size_t)(n >> 5) * (K >> 5) + (k >> 5)]);
	__half r[16];

#pragma unroll
	for (int j = 0; j < 16; j++)
		r[j] = __float2half(wf[j] * sc);

	*(uint4 *)(out + e) = *(uint4 *)r;
	*(uint4 *)(out + e + 8) = *(uint4 *)(r + 8);
}

extern "C" int
amp_fp8_dequant(const void *w, const void *s, void *out, int N, int K, cudaStream_t stream)
{
	if (N < 1 || K < 32 || (K % 32))
		return (int)cudaErrorInvalidValue;

	long n16 = (long)N * K / 16;

	amp_fp8_dequant_kernel<<<(unsigned)((n16 + 255) / 256), 256, 0, stream>>>((const uint8_t *)w,
		(const uint8_t *)s, (__half *)out, K, n16);

	return (int)cudaGetLastError();
}


/* ------------------------------------------------------------------ */
/* FP4 (e2m1, two per byte) block-scaled weights: the DSpark drafter's experts.
 * w (N, K/2) bytes, low nibble = even element; s (N, K/32) e8m0, one scale per 32 values. */

__constant__ float amp_fp4_lut[16] = { 0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                       -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f };

__device__ __forceinline__ float amp_e8m0_exact(uint8_t e)
{
	return __uint_as_float((uint32_t)e << 23);          /* 2^(e-127); e = 0 is 2^-127 (denormal, fine) */
}

__global__ void amp_fp4_dequant_kernel(const uint8_t *__restrict__ w, const uint8_t *__restrict__ s,
                                       __half *__restrict__ out, int K, long nblk)
{
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= nblk)
		return;

	uint4 wv = *(const uint4 *)(w + i * 16);
	const uint8_t *b = (const uint8_t *)&wv;
	const float sc = amp_e8m0_exact(s[i]);
	__half r[32];

#pragma unroll
	for (int j = 0; j < 16; j++) {
		r[2 * j] = __float2half(amp_fp4_lut[b[j] & 15] * sc);
		r[2 * j + 1] = __float2half(amp_fp4_lut[b[j] >> 4] * sc);
	}

	uint4 *o = (uint4 *)(out + i * 32);

	o[0] = *(uint4 *)(r);
	o[1] = *(uint4 *)(r + 8);
	o[2] = *(uint4 *)(r + 16);
	o[3] = *(uint4 *)(r + 24);
}

extern "C" int
amp_fp4_dequant(const void *w, const void *s, void *out, int N, int K, cudaStream_t stream)
{
	if (N < 1 || K < 32 || (K % 32))
		return (int)cudaErrorInvalidValue;

	long nblk = (long)N * (K / 32);

	amp_fp4_dequant_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, stream>>>((const uint8_t *)w,
		(const uint8_t *)s, (__half *)out, K, nblk);

	return (int)cudaGetLastError();
}

/* out[slot][n] = sum_k x[slot / xdiv][k] * W[ids[slot]][n][k] (fp32 accumulate), one warp per output
 * row n; the experts' weight / scale base addresses come from the pointer tables wp / sp. A warp
 * lane reads 16 bytes = 32 weights = exactly one scale block. */
#define AMP_FP4_ROWS 4        /* output rows per warp: the staged x row is reused, the loads overlap */

__global__ void amp_fp4_moe_gemv_kernel(const __half *__restrict__ x, const long long *__restrict__ ids,
                                        const long long *__restrict__ wp, const long long *__restrict__ sp,
                                        float *__restrict__ out, int K, int N, int xdiv)
{
	extern __shared__ float xs[];
	__shared__ float lut[16];

	const int slot = blockIdx.y;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const __half *xr = x + (size_t)(slot / xdiv) * K;

	if (threadIdx.x < 16)
		lut[threadIdx.x] = amp_fp4_lut[threadIdx.x];

	for (int k = threadIdx.x; k < K; k += blockDim.x)
		xs[k + (k >> 5)] = __half2float(xr[k]);              /* one pad float per 32: lanes read distinct banks */

	__syncthreads();

	const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * AMP_FP4_ROWS;

	if (n0 >= N)
		return;

	const long long e = ids[slot];

	if (e < 0) {                                     /* an expert of another rank (expert parallelism): no contribution */
		if (lane < AMP_FP4_ROWS && n0 + lane < N)
			out[(size_t)slot * N + n0 + lane] = 0.f;

		return;
	}

	const uint8_t *wb = (const uint8_t *)wp[e] + (size_t)n0 * (K / 2);
	const uint8_t *sb = (const uint8_t *)sp[e] + (size_t)n0 * (K / 32);
	const int nr = min(AMP_FP4_ROWS, N - n0);
	float acc[AMP_FP4_ROWS];

#pragma unroll
	for (int r = 0; r < AMP_FP4_ROWS; r++)
		acc[r] = 0.f;

	for (int kb = lane; kb < K / 32; kb += 32) {
		uint4 wv[AMP_FP4_ROWS];
		uint8_t sc[AMP_FP4_ROWS];

#pragma unroll
		for (int r = 0; r < AMP_FP4_ROWS; r++) {
			if (r < nr) {
				wv[r] = *(const uint4 *)(wb + (size_t)r * (K / 2) + (size_t)kb * 16);
				sc[r] = sb[(size_t)r * (K / 32) + kb];
			}
		}

		const float *xb = xs + kb * 33;

#pragma unroll
		for (int r = 0; r < AMP_FP4_ROWS; r++) {
			if (r < nr) {
				const uint8_t *b = (const uint8_t *)&wv[r];
				float a = 0.f;

#pragma unroll
				for (int j = 0; j < 16; j++)
					a = fmaf(lut[b[j] & 15], xb[2 * j], fmaf(lut[b[j] >> 4], xb[2 * j + 1], a));

				acc[r] = fmaf(a, amp_e8m0_exact(sc[r]), acc[r]);
			}
		}
	}

#pragma unroll
	for (int r = 0; r < AMP_FP4_ROWS; r++) {
#pragma unroll
		for (int o = 16; o > 0; o >>= 1)
			acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);

		if (lane == 0 && r < nr)
			out[(size_t)slot * N + n0 + r] = acc[r];
	}
}

extern "C" int
amp_fp4_moe_gemv(const void *x, const void *ids, const void *wp, const void *sp, void *out, int slots,
                 int K, int N, int xdiv, cudaStream_t stream)
{
	if (slots < 1 || K < 32 || (K % 32) || N < 1 || xdiv < 1)
		return (int)cudaErrorInvalidValue;

	dim3 grid((N + 8 * AMP_FP4_ROWS - 1) / (8 * AMP_FP4_ROWS), slots), block(256);

	amp_fp4_moe_gemv_kernel<<<grid, block, (size_t)(K + K / 32) * 4, stream>>>((const __half *)x,
		(const long long *)ids, (const long long *)wp, (const long long *)sp, (float *)out, K, N, xdiv);

	return (int)cudaGetLastError();
}


/* ------------------------------------------------------------------ */
/* rotary embedding of DeepSeek-V4.1 rows: the trailing rd channels of every row are rotated pairwise (GPT-J
 * layout) by the angle position * inv_freq; the leading channels are copied. One block per (row, head);
 * the position of row t is read from a device tensor (int64 or fp32, element stride pstride). */
template <typename P_T>
__global__ void amp_v41_rope_kernel(const __half *__restrict__ x, const P_T *__restrict__ pos, long pstride,
                                    const float *__restrict__ inv_freq, __half *__restrict__ out, int H, int D, int rd)
{
	const int row = blockIdx.x;                               /* t * H + h */
	const int t = row / H;
	const float p = (float)pos[(size_t)t * pstride];
	const int lead = (D - rd) / 2;                            /* pairs copied */
	const __half2 *xi = (const __half2 *)(x + (size_t)row * D);
	__half2 *xo = (__half2 *)(out + (size_t)row * D);

	for (int i = threadIdx.x; i < D / 2; i += blockDim.x) {
		if (i < lead) {
			xo[i] = xi[i];
		}
		else {
			float s, c;

			sincosf(p * inv_freq[i - lead], &s, &c);

			float2 v = __half22float2(xi[i]);

			xo[i] = __floats2half2_rn(v.x * c - v.y * s, v.x * s + v.y * c);
		}
	}
}

extern "C" int
amp_v41_rope(const void *x, const void *pos, int pos_f32, long pstride, const float *inv_freq, void *out, int T,
             int H, int D, int rd, cudaStream_t stream)
{
	if (T < 1 || H < 1 || D < rd || (D & 1) || (rd & 1) || rd < 2)
		return (int)cudaErrorInvalidValue;

	if (pos_f32)
		amp_v41_rope_kernel<float><<<(unsigned)(T * H), 128, 0, stream>>>((const __half *)x, (const float *)pos,
			pstride, inv_freq, (__half *)out, H, D, rd);
	else
		amp_v41_rope_kernel<long long><<<(unsigned)(T * H), 128, 0, stream>>>((const __half *)x,
			(const long long *)pos, pstride, inv_freq, (__half *)out, H, D, rd);

	return (int)cudaGetLastError();
}


/* ------------------------------------------------------------------ */
/* DeepSeek-V4 / V4.1 router: affinity sqrt(softplus(logit)); the k experts with the highest affinity + bias (ties:
 * the lower index) in descending order; weights = their affinities normalized to sum 1 (+ 1e-20) times scale.
 * One block (128 threads) per row, k rounds of block argmax. */
#define AMP_ROUTE_THREADS 128
#define AMP_ROUTE_MAXE 512

__global__ void __launch_bounds__(AMP_ROUTE_THREADS)
amp_v41_route_kernel(const float *__restrict__ logits, const float *__restrict__ bias, long long *__restrict__ idx,
                     float *__restrict__ wout, int E, int k, float scale)
{
	__shared__ float sc[AMP_ROUTE_MAXE];
	__shared__ float key[AMP_ROUTE_MAXE];
	__shared__ float wv[32];
	__shared__ float red_v[AMP_ROUTE_THREADS / 32];
	__shared__ int red_i[AMP_ROUTE_THREADS / 32];
	__shared__ int pick;

	const int row = blockIdx.x;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

	for (int e = threadIdx.x; e < E; e += blockDim.x) {
		float x = logits[(size_t)row * E + e];
		float sp = (x > 20.0f) ? x : log1pf(expf(x));
		float a = sqrtf(sp);

		sc[e] = a;
		key[e] = a + bias[e];
	}

	__syncthreads();

	for (int j = 0; j < k; j++) {
		float bv = -INFINITY;
		int bi = 0x7fffffff;

		for (int e = threadIdx.x; e < E; e += blockDim.x) {
			float v = key[e];

			if (v > bv || (v == bv && e < bi)) {
				bv = v;
				bi = e;
			}
		}

#pragma unroll
		for (int o = 16; o > 0; o >>= 1) {
			float ov = __shfl_xor_sync(0xffffffffu, bv, o);
			int oi = __shfl_xor_sync(0xffffffffu, bi, o);

			if (ov > bv || (ov == bv && oi < bi)) {
				bv = ov;
				bi = oi;
			}
		}

		if (lane == 0) {
			red_v[warp] = bv;
			red_i[warp] = bi;
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			float v = red_v[0];
			int i = red_i[0];

			for (int w = 1; w < AMP_ROUTE_THREADS / 32; w++)
				if (red_v[w] > v || (red_v[w] == v && red_i[w] < i)) {
					v = red_v[w];
					i = red_i[w];
				}

			pick = i;
			idx[(size_t)row * k + j] = i;
			wv[j] = sc[i];
			key[i] = -INFINITY;
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		float sum = 0.f;

		for (int j = 0; j < k; j++)
			sum += wv[j];

		const float inv = scale / (sum + 1e-20f);

		for (int j = 0; j < k; j++)
			wout[(size_t)row * k + j] = wv[j] * inv;
	}
}

extern "C" int
amp_v41_route(const float *logits, const float *bias, long long *idx, float *w, int T, int E, int k, float scale,
              cudaStream_t stream)
{
	if (T < 1 || E < 1 || E > AMP_ROUTE_MAXE || k < 1 || k > 32 || k > E)
		return (int)cudaErrorInvalidValue;

	amp_v41_route_kernel<<<(unsigned)T, AMP_ROUTE_THREADS, 0, stream>>>(logits, bias, idx, w, E, k, scale);

	return (int)cudaGetLastError();
}


/* ------------------------------------------------------------------ */
/* DeepSeek-V4.1 sparse MQA attention                                  */
/*
 * One shared key/value head of 512 dims (V = K), H query heads, a per-head sink logit.
 * The keys of query t (absolute position p = pos0 + t):
 *   - its sliding window: positions p-W+1 .. p, read from the window rows kvw (row r holds
 *     position wbase + r)
 *   - Kc compressed rows picked by idx[t][k] into kvc (negative = none)
 * Two kernels: partial results per key split (online softmax state + weighted sum), then a
 * combine that adds the sink, normalizes, and rotates the rope slice back by the position.
 */

#define V41_D 512
#define V41_HG 8                /* heads per block (one warp each) */
#define V41_TILE 16             /* keys per smem tile */

__global__ void __launch_bounds__(256)
amp_v41_partial_kernel(const __half *__restrict__ q, const __half *__restrict__ kvw, long wbase, int Nw,
                       const __half *__restrict__ kvc, int Nc, const int *__restrict__ idx, int Kc,
                       float *__restrict__ ws_m, float *__restrict__ ws_l, float *__restrict__ ws_o,
                       int H, int W, int nsplit, float scale, long pos0, const long long *pos_ptr, int ring)
{
	if (pos_ptr)
		pos0 = *pos_ptr;

	extern __shared__ uint4 v41_sm[];                 /* V41_TILE rows x 64 uint4 */
	__shared__ const __half *row_ptr[V41_TILE];

	const int t = blockIdx.x, hg = blockIdx.y, sp = blockIdx.z;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int hh = hg * V41_HG + warp;
	const long p = pos0 + t;
	const int total = W + Kc;
	int per = (total + nsplit - 1) / nsplit;

	per = (per + V41_TILE - 1) / V41_TILE * V41_TILE;

	const int j0 = sp * per;
	const int j1 = min(total, j0 + per);

	float qf[16];
	{
		const uint4 *qp = (const uint4 *)(q + ((size_t)t * H + hh) * V41_D + lane * 16);
		uint4 a = qp[0], b = qp[1];
		const __half2 *h0 = (const __half2 *)&a, *h1 = (const __half2 *)&b;

#pragma unroll
		for (int i = 0; i < 4; i++) {
			float2 x = __half22float2(h0[i]), y = __half22float2(h1[i]);

			qf[2 * i] = x.x * scale;
			qf[2 * i + 1] = x.y * scale;
			qf[8 + 2 * i] = y.x * scale;
			qf[8 + 2 * i + 1] = y.y * scale;
		}
	}

	float m_run = -INFINITY, l_run = 0.0f, o[16];

#pragma unroll
	for (int i = 0; i < 16; i++)
		o[i] = 0.0f;

	for (int jt = j0; jt < j1; jt += V41_TILE) {
		__syncthreads();

		if (threadIdx.x < V41_TILE) {
			int j = jt + threadIdx.x;
			const __half *rp = NULL;

			if (j < j1) {
				if (j < W) {
					long a = p - W + 1 + j;

					if (ring > 0) {                      /* window rows live in a ring: position a at a % ring */
						if (a >= 0)
							rp = kvw + (size_t)(a % ring) * V41_D;
					}
					else {
						long r = a - wbase;

						if (a >= 0 && r >= 0 && r < Nw)
							rp = kvw + (size_t)r * V41_D;
					}
				}
				else {
					int c = idx[(size_t)t * Kc + (j - W)];

					if (c >= 0 && c < Nc)
						rp = kvc + (size_t)c * V41_D;
				}
			}

			row_ptr[threadIdx.x] = rp;
		}

		__syncthreads();

		for (int c = threadIdx.x; c < V41_TILE * 64; c += 256) {
			const __half *rp = row_ptr[c >> 6];

			if (rp)
				v41_sm[c] = ((const uint4 *)rp)[c & 63];
		}

		__syncthreads();

		float sc[V41_TILE];

#pragma unroll
		for (int i = 0; i < V41_TILE; i++) {
			float d = 0.0f;

			if (row_ptr[i]) {
				const uint4 *kp = v41_sm + i * 64 + lane * 2;
				uint4 a = kp[0], b = kp[1];
				const __half2 *h0 = (const __half2 *)&a, *h1 = (const __half2 *)&b;

#pragma unroll
				for (int k = 0; k < 4; k++) {
					float2 x = __half22float2(h0[k]), y = __half22float2(h1[k]);

					d += qf[2 * k] * x.x + qf[2 * k + 1] * x.y;
					d += qf[8 + 2 * k] * y.x + qf[8 + 2 * k + 1] * y.y;
				}
			}

#pragma unroll
			for (int o2 = 16; o2 > 0; o2 >>= 1)
				d += __shfl_xor_sync(0xffffffffu, d, o2);

			sc[i] = row_ptr[i] ? d : -INFINITY;
		}

		float mt = sc[0];

#pragma unroll
		for (int i = 1; i < V41_TILE; i++)
			mt = fmaxf(mt, sc[i]);

		const float mn = fmaxf(m_run, mt);

		if (mn == -INFINITY)
			continue;

		const float alpha = (m_run == -INFINITY) ? 0.0f : __expf(m_run - mn);

		l_run *= alpha;

#pragma unroll
		for (int k = 0; k < 16; k++)
			o[k] *= alpha;

#pragma unroll
		for (int i = 0; i < V41_TILE; i++) {
			if (!row_ptr[i])
				continue;

			const float pw = __expf(sc[i] - mn);
			const uint4 *kp = v41_sm + i * 64 + lane * 2;
			uint4 a = kp[0], b = kp[1];
			const __half2 *h0 = (const __half2 *)&a, *h1 = (const __half2 *)&b;

			l_run += pw;

#pragma unroll
			for (int k = 0; k < 4; k++) {
				float2 x = __half22float2(h0[k]), y = __half22float2(h1[k]);

				o[2 * k] += pw * x.x;
				o[2 * k + 1] += pw * x.y;
				o[8 + 2 * k] += pw * y.x;
				o[8 + 2 * k + 1] += pw * y.y;
			}
		}

		m_run = mn;
	}

	const size_t wi = ((size_t)t * H + hh) * nsplit + sp;

	if (lane == 0) {
		ws_m[wi] = m_run;
		ws_l[wi] = l_run;
	}

	float4 *op = (float4 *)(ws_o + wi * V41_D + lane * 16);

#pragma unroll
	for (int i = 0; i < 4; i++)
		op[i] = make_float4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}


/* Tensor-core variant (prefill chunks): one block per (token, 16 heads, key split). The 16 keys of a tile are
 * staged in shared memory; QK^T runs as m16n8k16 mma with the head rows as M (k split over four warp pairs,
 * partial sums reduced through shared memory), the online softmax is done by 16 threads, P V as mma with the
 * 512 value dims spread over the 8 warps (64 each, 32 accumulator registers per thread). The state of a split
 * (m, l, O) is written to the same workspace as the CUDA-core kernel, so the combine kernel is shared. */

#define V41T_QS 520              /* halves per smem row: 512 + 8 (bank-conflict free fragment loads) */
#define V41T_PS 24

__device__ __forceinline__ void
v41t_mma(float &c0, float &c1, float &c2, float &c3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
         uint32_t b0, uint32_t b1)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
	             "{%0,%1,%2,%3};\n" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
	             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256)
amp_v41_tc_kernel(const __half *__restrict__ q, const __half *__restrict__ kvw, long wbase, int Nw,
                  const __half *__restrict__ kvc, int Nc, const int *__restrict__ idx, int Kc,
                  float *__restrict__ ws_m, float *__restrict__ ws_l, float *__restrict__ ws_o,
                  int H, int W, int nsplit, float scale, long pos0, const long long *pos_ptr, int ring)
{
	extern __shared__ __align__(16) unsigned char v41t_sm[];
	__half *sQ = (__half *)v41t_sm;                                   /* 16 x QS */
	__half *sK = sQ + 16 * V41T_QS;                                   /* 16 x QS */
	float *sPart = (float *)(sK + 16 * V41T_QS);                      /* 4 x 16 x 16 */
	__half *sP = (__half *)(sPart + 4 * 16 * 16);                     /* 16 x PS */
	float *sAlpha = (float *)(sP + 16 * V41T_PS);                     /* 16 */
	__shared__ const __half *row_ptr[16];

	if (pos_ptr)
		pos0 = *pos_ptr;

	const int t = blockIdx.x, hg = blockIdx.y, sp = blockIdx.z;
	const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const int g = lane >> 2, tid4 = lane & 3;
	const long p = pos0 + t;
	const int total = W + Kc;
	int per = (total + nsplit - 1) / nsplit;

	per = (per + 15) / 16 * 16;

	const int j0 = sp * per, j1 = min(total, j0 + per);

	/* Q of the 16 heads */
	for (int c = tid; c < 16 * 64; c += 256) {
		const int r = c >> 6, col = c & 63;

		*(uint4 *)(sQ + r * V41T_QS + col * 8) =
			((const uint4 *)(q + ((size_t)t * H + hg * 16 + r) * V41_D))[col];
	}

	float o[8][4];

#pragma unroll
	for (int n = 0; n < 8; n++)
		o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;

	float m_run = -INFINITY, l_run = 0.0f;                           /* of row tid, tid < 16 */

	for (int jt = j0; jt < j1; jt += 16) {
		__syncthreads();                                              /* sK / sP free again */

		if (tid < 16) {
			int j = jt + tid;
			const __half *rp = NULL;

			if (j < j1) {
				if (j < W) {
					long a = p - W + 1 + j;

					if (ring > 0) {
						if (a >= 0)
							rp = kvw + (size_t)(a % ring) * V41_D;
					}
					else {
						long r = a - wbase;

						if (a >= 0 && r >= 0 && r < Nw)
							rp = kvw + (size_t)r * V41_D;
					}
				}
				else {
					int c = idx[(size_t)t * Kc + (j - W)];

					if (c >= 0 && c < Nc)
						rp = kvc + (size_t)c * V41_D;
				}
			}

			row_ptr[tid] = rp;
		}

		__syncthreads();

		for (int c = tid; c < 16 * 64; c += 256) {
			const int r = c >> 6, col = c & 63;
			const __half *rp = row_ptr[r];

			*(uint4 *)(sK + r * V41T_QS + col * 8) = rp ? ((const uint4 *)rp)[col] : make_uint4(0, 0, 0, 0);
		}

		__syncthreads();

		/* S = Q K^T, partial over a quarter of the dims per warp pair */
		{
			const int nt = warp & 1, kq = warp >> 1;
			float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

#pragma unroll
			for (int ks = 0; ks < 8; ks++) {
				const int kk = kq * 8 + ks;
				const __half *qa = sQ + g * V41T_QS + kk * 16 + 2 * tid4;
				const __half *kb = sK + (nt * 8 + g) * V41T_QS + kk * 16 + 2 * tid4;
				const uint32_t a0 = *(const uint32_t *)qa, a1 = *(const uint32_t *)(qa + 8 * V41T_QS);
				const uint32_t a2 = *(const uint32_t *)(qa + 8), a3 = *(const uint32_t *)(qa + 8 * V41T_QS + 8);
				const uint32_t b0 = *(const uint32_t *)kb, b1 = *(const uint32_t *)(kb + 8);

				v41t_mma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
			}

			float *dst = sPart + kq * 256;

			dst[g * 16 + nt * 8 + 2 * tid4] = c0;
			dst[g * 16 + nt * 8 + 2 * tid4 + 1] = c1;
			dst[(g + 8) * 16 + nt * 8 + 2 * tid4] = c2;
			dst[(g + 8) * 16 + nt * 8 + 2 * tid4 + 1] = c3;
		}

		__syncthreads();

		/* online softmax of the 16 head rows (one thread each) */
		if (tid < 16) {
			float sc[16], mt = -INFINITY;

#pragma unroll
			for (int k = 0; k < 16; k++) {
				float v = (sPart[tid * 16 + k] + sPart[256 + tid * 16 + k] + sPart[512 + tid * 16 + k] +
				           sPart[768 + tid * 16 + k]) * scale;

				sc[k] = row_ptr[k] ? v : -INFINITY;
				mt = fmaxf(mt, sc[k]);
			}

			const float mn = fmaxf(m_run, mt);
			float alpha = 1.0f, ls = 0.0f;

			if (mn != -INFINITY) {
				alpha = (m_run == -INFINITY) ? 0.0f : __expf(m_run - mn);

#pragma unroll
				for (int k = 0; k < 16; k++) {
					const float pw = (sc[k] == -INFINITY) ? 0.0f : __expf(sc[k] - mn);

					ls += pw;
					sP[tid * V41T_PS + k] = __float2half_rn(pw);
				}

				l_run = l_run * alpha + ls;
				m_run = mn;
			}
			else {
#pragma unroll
				for (int k = 0; k < 16; k++)
					sP[tid * V41T_PS + k] = __float2half_rn(0.0f);
			}

			sAlpha[tid] = alpha;
		}

		__syncthreads();

		/* O = alpha O + P V over this warp's 64 dims */
		{
			const float al0 = sAlpha[g], al1 = sAlpha[g + 8];
			const __half *pa = sP + g * V41T_PS + 2 * tid4;
			const uint32_t a0 = *(const uint32_t *)pa, a1 = *(const uint32_t *)(pa + 8 * V41T_PS);
			const uint32_t a2 = *(const uint32_t *)(pa + 8), a3 = *(const uint32_t *)(pa + 8 * V41T_PS + 8);
			const int mi = lane >> 3, rr = lane & 7;

#pragma unroll
			for (int n = 0; n < 8; n += 2) {
				const int d0 = warp * 64 + n * 8;
				const __half *src = sK + ((mi & 1) * 8 + rr) * V41T_QS + d0 + (mi >> 1) * 8;
				uint32_t b0, b1, b2, b3;

				asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
				             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
				             : "r"((uint32_t)__cvta_generic_to_shared(src)));

				o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1;
				o[n + 1][0] *= al0; o[n + 1][1] *= al0; o[n + 1][2] *= al1; o[n + 1][3] *= al1;
				v41t_mma(o[n][0], o[n][1], o[n][2], o[n][3], a0, a1, a2, a3, b0, b1);
				v41t_mma(o[n + 1][0], o[n + 1][1], o[n + 1][2], o[n + 1][3], a0, a1, a2, a3, b2, b3);
			}
		}
	}

	/* the state of this split: head rows hg * 16 + row */
	if (tid < 16) {
		const size_t wi = ((size_t)t * H + hg * 16 + tid) * nsplit + sp;

		ws_m[wi] = m_run;
		ws_l[wi] = l_run;
	}

#pragma unroll
	for (int n = 0; n < 8; n++) {
		const int d = warp * 64 + n * 8 + 2 * tid4;
		const size_t w0 = (((size_t)t * H + hg * 16 + g) * nsplit + sp) * V41_D + d;
		const size_t w1 = (((size_t)t * H + hg * 16 + g + 8) * nsplit + sp) * V41_D + d;

		ws_o[w0] = o[n][0];
		ws_o[w0 + 1] = o[n][1];
		ws_o[w1] = o[n][2];
		ws_o[w1 + 1] = o[n][3];
	}
}

#define V41T_SMEM ((size_t)(2 * 16 * V41T_QS * 2 + 4 * 16 * 16 * 4 + 16 * V41T_PS * 2 + 16 * 4))

__global__ void __launch_bounds__(64)
amp_v41_combine_kernel(const float *__restrict__ ws_m, const float *__restrict__ ws_l,
                       const float *__restrict__ ws_o, const float *__restrict__ sinks,
                       const float *__restrict__ inv_freq, __half *__restrict__ out,
                       int H, int nsplit, long pos0, int rd, const long long *pos_ptr)
{
	if (pos_ptr)
		pos0 = *pos_ptr;

	const int t = blockIdx.x, hh = blockIdx.y, d0 = threadIdx.x * 8;
	const size_t base = ((size_t)t * H + hh) * nsplit;
	const float sink = sinks[hh];
	float M = sink;

	for (int s = 0; s < nsplit; s++)
		M = fmaxf(M, ws_m[base + s]);

	float L = __expf(sink - M), o[8];

#pragma unroll
	for (int i = 0; i < 8; i++)
		o[i] = 0.0f;

	for (int s = 0; s < nsplit; s++) {
		float ms = ws_m[base + s];

		if (ms == -INFINITY)
			continue;

		const float w = __expf(ms - M);

		L += w * ws_l[base + s];

		const float4 *op = (const float4 *)(ws_o + (base + s) * V41_D + d0);
		float4 a = op[0], b = op[1];

		o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
		o[4] += w * b.x; o[5] += w * b.y; o[6] += w * b.z; o[7] += w * b.w;
	}

	const float inv = 1.0f / L;

#pragma unroll
	for (int i = 0; i < 8; i++)
		o[i] *= inv;

	if (d0 >= V41_D - rd) {                       /* rotate the rope slice back by the position */
		const float pos = (float)(pos0 + t);

#pragma unroll
		for (int i = 0; i < 4; i++) {
			const float ang = pos * inv_freq[(d0 - (V41_D - rd)) / 2 + i];
			float sn, cs;

			sincosf(ang, &sn, &cs);

			const float a = o[2 * i], b = o[2 * i + 1];

			o[2 * i] = a * cs + b * sn;
			o[2 * i + 1] = b * cs - a * sn;
		}
	}

	__half *op = out + ((size_t)t * H + hh) * V41_D + d0;
	__half2 r[4];

#pragma unroll
	for (int i = 0; i < 4; i++)
		r[i] = __floats2half2_rn(o[2 * i], o[2 * i + 1]);

	*(uint4 *)op = *(uint4 *)r;
}

extern "C" long
amp_v41_attn_ws_floats(int T, int H, int nsplit)
{
	return (long)T * H * nsplit * (V41_D + 2);
}

extern "C" int
amp_v41_attn(const void *q, const void *kvw, long wbase, int Nw, const void *kvc, int Nc, const void *idx,
             int Kc, const float *sinks, const float *inv_freq, void *out, float *ws, int T, int H, int W,
             int nsplit, float scale, long pos0, const long long *pos_ptr, int ring, cudaStream_t stream)
{
	if (T < 1 || H % V41_HG || nsplit < 1 || W < 1 || (Kc > 0 && !(kvc && idx)))
		return (int)cudaErrorInvalidValue;

	float *ws_m = ws, *ws_l = ws + (size_t)T * H * nsplit, *ws_o = ws_l + (size_t)T * H * nsplit;
	dim3 grid(T, H / V41_HG, nsplit);
	size_t smem = (size_t)V41_TILE * 64 * sizeof(uint4);

	static const bool tc_off = getenv("AMP_V41_NO_TC") != NULL;

	if (!tc_off && T >= 8 && H % 16 == 0) {
		static bool attr_set[64] = { false };
		int qdev = 0;

		cudaGetDevice(&qdev);

		if (qdev >= 0 && qdev < 64 && !attr_set[qdev]) {
			cudaFuncSetAttribute(amp_v41_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)V41T_SMEM);
			attr_set[qdev] = true;
		}

		amp_v41_tc_kernel<<<dim3(T, H / 16, nsplit), 256, V41T_SMEM, stream>>>((const __half *)q,
			(const __half *)kvw, wbase, Nw, (const __half *)kvc, Nc, (const int *)idx, Kc, ws_m, ws_l, ws_o, H, W,
			nsplit, scale, pos0, pos_ptr, ring);
	}
	else
		amp_v41_partial_kernel<<<grid, 256, smem, stream>>>((const __half *)q, (const __half *)kvw, wbase, Nw,
			(const __half *)kvc, Nc, (const int *)idx, Kc, ws_m, ws_l, ws_o, H, W, nsplit, scale, pos0, pos_ptr, ring);
	amp_v41_combine_kernel<<<dim3(T, H), 64, 0, stream>>>(ws_m, ws_l, ws_o, sinks, inv_freq,
		(__half *)out, H, nsplit, pos0, 64, pos_ptr);

	return (int)cudaGetLastError();
}
