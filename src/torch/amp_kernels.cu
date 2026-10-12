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

template <int DPL>
__global__ void
amp_attn_dec_combine_kernel(const float *__restrict__ ws, __half *__restrict__ out,
                           int S, int Hq, int Hk, int ns,
                           const int *__restrict__ lenp, int max_seq)
{
	constexpr int E = DPL * 32;
	const int j = blockIdx.x;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int rep = Hq / Hk;
	const int g = warp % rep, s = warp / rep;
	const int h = j * rep + g;
	const int wsrow = 4 + E;

	int len = *lenp;

	len = (len < 0) ? 0 : (len > max_seq ? max_seq : len);

	const int nse = amp_ad_nse(len, ns);

	float m = -1e30f;

	#pragma unroll 8
	for (int sp = 0; sp < nse; sp++)
		m = fmaxf(m, ws[(((size_t)j * ns + sp) * rep * S + warp) * wsrow]);

	float l = 0.0f, acc[DPL];

	for (int i = 0; i < DPL; i++)
		acc[i] = 0.0f;

	#pragma unroll 4
	for (int sp = 0; sp < nse; sp++) {
		const float *w = ws + (((size_t)j * ns + sp) * rep * S + warp) * wsrow;
		float e = __expf(w[0] - m);

		l += w[1] * e;

		for (int i = 0; i < DPL; i++)
			acc[i] = fmaf(w[4 + lane * DPL + i], e, acc[i]);
	}

	float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
	__half *o = out + ((size_t)s * Hq + h) * E + lane * DPL;

	for (int i = 0; i < DPL; i += 2)
		*(__half2 *)(o + i) = __floats2half2_rn(acc[i] * inv, acc[i + 1] * inv);
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
	    (Hq / Hk) * S > 32 || ns < 1 || (E != 64 && E != 128 && E != 256)) {
		fprintf(stderr, "amp_attn_dec: unsupported config (S %d, Hq %d, Hk %d, E %d, ns %d)\n",
			S, Hq, Hk, E, ns);
		return -1;
	}

	dim3 grid(Hk, ns);
	int block = (Hq / Hk) * 32;            /* one warp per query head */
	int cblock = (Hq / Hk) * S * 32;       /* combine: one warp per (row, head) */

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
	} \
	amp_attn_dec_combine_kernel<D><<<Hk, cblock, 0, stream>>>((const float *)ws, \
		(__half *)out, S, Hq, Hk, ns, len, max_seq); } while (0)

	if (E == 256)
		AMP_AD_LAUNCH(8);
	else if (E == 128)
		AMP_AD_LAUNCH(4);
	else
		AMP_AD_LAUNCH(2);

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
                    int R, int n_pools, float scale, int pos0, int P, int HI)
{
	extern __shared__ __half sp[];            // IDX_TILE x IDX_STRIDE

	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
	const int p0 = blockIdx.x * IDX_TILE;
	const int r = blockIdx.y * IDX_QPB + warp;

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
             int R, int n_pools, float scale, int pos0, int P, int HI, cudaStream_t stream)
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
		(__half *)out, R, n_pools, scale, pos0, P, HI);

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

/* E = head dim (64 / 128 / 256); S (query rows per head) is runtime. */
template <int E>
__global__ void
amp_attn_dec_mma_kernel(const __half *__restrict__ q,
                       const __half *__restrict__ kc,
                       const __half *__restrict__ vc,
                       const int *__restrict__ lenp, float *__restrict__ ws,
                       int S, int Hq, int Hk, int max_seq, int ns,
                       float scale)
{
	constexpr int BK = AMP_ADM_BK;
	constexpr int W = AMP_ADM_WARPS;
	constexpr int KS = E / 16;         /* score k-steps per (m, n) */
	constexpr int PNT = E / 8 / W;     /* PV n-tiles per warp */
	constexpr int PKS = BK / 16;       /* PV k-steps */
	constexpr int RB = E * 2;          /* K/V/Q tile row bytes */
	constexpr int EB = 128;            /* e-block row bytes (4 of 8
	                                     * pieces used; the swizzle
	                                     * needs >= 8 pieces/row) */
	constexpr int WSROW = 4 + E;

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

	/* Q tile (32 x E h) | K 2 x (BK x E) | V 2 x (BK x E)
	 * | sblock (32 x BK f) | eblock (32 x 64 h, swizzled) | corr (32 f) */
	__half *qt = smem;
	__half *kt = qt + 32 * E;
	__half *vt = kt + 2 * BK * E;
	float *sblock = (float *)(vt + 2 * BK * E);
	__half *eblock = ( __half *)(sblock + 32 * BK);
	/* corr sits AFTER the full eblock (32 rows x EB bytes): the
	 * eblock rows are 64 halves wide (8 swizzle pieces), not BK —
	 * (eblock + 32 * BK) landed on row 16 and row-16 e writes
	 * clobbered corr[0..15], zeroing the rows 0-15 pv rescale */
	float *corr = (float *)((char *)eblock + 32 * EB);

	/* fill the Q tile (rows >= rows zeroed). 16-byte
	 * pieces, swizzled stores. */
	for (int i = threadIdx.x; i < 32 * (E / 8); i += blockDim.x) {
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

	/* per-warp softmax state: rows r % W == warp (<= 4 rows) */
	float mst[4], lst[4];
	int nown = 0;

	for (int r = warp; r < rows; r += W) {
		mst[nown] = -1e30f;
		lst[nown] = 0.0f;
		nown++;
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

	float pv[PNT][2][4];

	#pragma unroll
	for (int n = 0; n < PNT; n++)
		#pragma unroll
		for (int m = 0; m < 2; m++)
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

		/* --- scores: warp w covers m-tile w/(BK/8) (16 rows) ----
		 * and n-tile w%(BK/8) (8 keys); 2 x 4 = W warps cover ---
		 * the tile ------------------------------------------------ */
		{
			const int mrow = (warp / (BK / 8)) * 16;
			const int n0 = (warp % (BK / 8)) * 8;

			float sc[4];
			unsigned af[4], bf[2];

			/* mma accumulates into sc — it must start at zero for
			 * every tile (the pv accumulator is the opposite: it
			 * persists across tiles and is zeroed once above) */
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

			/* scatter with the scale and the causal mask. C layout:
			 * c[0..1] = row (lane>>2), cols 2*(lane&3) + {0,1};
			 * c[2..3] = row + 8, same cols. */
			int r0 = mrow + (lane >> 2);
			int c0 = n0 + 2 * (lane & 3);

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
						ok ? scale * sc[2 * half + i]
						   : -1e30f;
				}
			}
		}

		__syncthreads();

		/* --- softmax: warp w owns rows r % W == w ------------- */
		{
			int ri = 0;

			for (int r = warp; r < rows; r += W, ri++) {
				int s = r % S;
				int lim = min(end, lim_base + s + 1);
				float mx = -1e30f;

				for (int c = lane; c < BK; c += 32) {
					bool ok = (tb + c) < lim;
					float v = ok ? sblock[r * BK + c]
						: -1e30f;

					mx = fmaxf(mx, v);
				}

				#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
					mx = fmaxf(mx, __shfl_xor_sync(
						0xffffffffu, mx, off));

				float m_new = fmaxf(mst[ri], mx);
				float cr = __expf(mst[ri] - m_new);
				float lsum = 0.0f;

				for (int c = lane; c < BK; c += 32) {
					bool ok = (tb + c) < lim;
					float v = ok ? sblock[r * BK + c]
						: -1e30f;
					float e = __expf(v - m_new);
					__half eh = __float2half(e);

					/* swizzled write (the PV ldmatrix reads it swizzled) */
					*( __half *)((char *)eblock + amp_adm_swz(r, c >> 3, EB)
						+ (c & 7) * 2) = eh;

					lsum += ok ? e : 0.0f;
				}

				#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
					lsum += __shfl_xor_sync(
						0xffffffffu, lsum, off);

				lst[ri] = lst[ri] * cr + lsum;
				mst[ri] = m_new;

				if (lane == 0)
					corr[r] = cr;
			}
		}

		__syncthreads();

		/* --- PV: warp w covers n-tiles w, w+W, ... ------------ */
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

						/* A = e-block rows of m-tile m, keys kk*16 */
						int arow = m * 16 +
							(lane & 7) + ((lane >> 3) & 1) * 8;
						int apiece = kk * 2 + (lane >> 4);

						amp_adm_ldsm_x4(af,
							AMP_ADM_ADDR(eblock, arow, apiece, EB));

						/* B = V tile: keys kk*16 + {0..15},
						 * dims nt*8 (one 16-B piece) */
						int vkey = kk * 16 + (lane & 7) +
							((lane >> 3) & 1) * 8;

						amp_adm_ldsm_x2_t(bf,
							AMP_ADM_ADDR(vt_s, vkey, nt, RB));

						amp_adm_mma(pv[n][m], af, bf);
					}
				}
			}
		}

		__syncthreads();
	}

	/* --- workspace ------------------------------------------------ */
	{
		int ri = 0;

		for (int r = warp; r < rows; r += W, ri++) {
			int s = r % S, g = r / S;
			int lim = min(end, lim_base + s + 1);
			bool any = begin < lim;
			float *w = ws + (((size_t)j * ns + sp) * rep * S +
				s * rep + g) * WSROW;

			if (lane == 0) {
				w[0] = any ? mst[ri] : -1e30f;
				w[1] = any ? lst[ri] : 0.0f;
			}
		}
	}

	/* acc dims: warp w's PV n-tiles, fragment scatter. c[0..1] =
	 * row (lane>>2), c[2..3] = row + 8; cols 2*(lane&3) + {0,1}. */
	#pragma unroll
	for (int n = 0; n < PNT; n++) {
		int nt = warp + n * W;
		int c0 = nt * 8 + 2 * (lane & 3);

		#pragma unroll
		for (int m = 0; m < 2; m++) {
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

extern "C" size_t
amp_attn_dec_mma_smem(int E)
{
	return (size_t)(32 * E * 2 +              /* Q tile */
		2 * AMP_ADM_BK * E * 2 +           /* K stages */
		2 * AMP_ADM_BK * E * 2 +           /* V stages */
		32 * AMP_ADM_BK * 4 +              /* sblock */
		32 * 128 +                        /* eblock */
		32 * 4 +                          /* corr */
		16);                              /* alignment slack */
}

extern "C" int
amp_attn_dec_mma_fp16(const void *q, const void *k, const void *v,
                     const int *len, void *ws, void *out, int S, int Hq,
                     int Hk, int E, int max_seq, int ns, float scale,
                     cudaStream_t stream)
{
	if (Hk <= 0 || Hq % Hk || S < 1 || S > 5 || Hq / Hk > 32 ||
	    (Hq / Hk) * S > 32 || ns < 1 ||
	    (E != 64 && E != 128 && E != 256) || (E / 8) % AMP_ADM_WARPS) {
		fprintf(stderr, "amp_attn_dec_mma: unsupported config "
			"(S %d, Hq %d, Hk %d, E %d, ns %d)\n",
			S, Hq, Hk, E, ns);
		return -1;
	}

	dim3 grid(Hk, ns);
	int block = AMP_ADM_WARPS * 32;
	int cblock = (Hq / Hk) * S * 32;
	size_t smem = amp_attn_dec_mma_smem(E);

#define AMP_ADM_K(EV) do { \
	/* the >48 KiB dynamic shared memory opt-in is per device */ \
	static bool attr_done[64] = { false }; \
	int adm_dev = 0; \
	cudaGetDevice(&adm_dev); \
	if (adm_dev >= 0 && adm_dev < 64 && !attr_done[adm_dev]) { \
		cudaFuncSetAttribute(amp_attn_dec_mma_kernel<EV>, \
			cudaFuncAttributeMaxDynamicSharedMemorySize, \
			(int)amp_attn_dec_mma_smem(EV)); \
		attr_done[adm_dev] = true; \
	} \
	amp_attn_dec_mma_kernel<EV><<<grid, block, smem, stream>>>( \
		(const __half *)q, (const __half *)k, (const __half *)v, \
		len, (float *)ws, S, Hq, Hk, max_seq, ns, (float)scale); \
} while (0)
#define AMP_ADM_C(D) \
	amp_attn_dec_combine_kernel<D><<<Hk, cblock, 0, stream>>>( \
		(const float *)ws, (__half *)out, S, Hq, Hk, ns, len, max_seq)

	if (E == 256) {
		AMP_ADM_K(256);
		AMP_ADM_C(8);
	} else if (E == 128) {
		AMP_ADM_K(128);
		AMP_ADM_C(4);
	} else {
		AMP_ADM_K(64);
		AMP_ADM_C(2);
	}

#undef AMP_ADM_K
#undef AMP_ADM_C

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
