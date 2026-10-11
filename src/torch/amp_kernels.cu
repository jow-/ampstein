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

extern "C" int
amp_rms_norm_fp16(const void *x, const void *w, void *out,
                 long rows, int dim, double eps, cudaStream_t stream)
{
	if (rows <= 0 || dim <= 0)
		return 0;

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
