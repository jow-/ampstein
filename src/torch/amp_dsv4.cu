// src/amp_dsv4.cu — DeepSeek-V4 decode-step fusions (project-owned, not vendored).
//
// Every kernel reads the position from a DEVICE scalar (pos_f float32, pos_i
// int64), so a captured CUDA graph replays correctly at any position.
//
//   amp_dsv4_qkv     per-head RMSNorm + GPT-J rope of the queries, weighted
//                   RMSNorm + rope of the shared K=V row, written straight
//                   into the window cache row `pos`
//   amp_dsv4_comp    one token into a compressor: buffer the projected rows,
//                   and when the window completes pool it (softmax over the
//                   window, optional Ca/Cb overlap), RMSNorm, rope at the
//                   window start, append to the pool, carry the overlap
//   amp_dsv4_attn    attention of one token (all heads) over the sliding
//                   window and the compressed pool (dense with a causal
//                   bound, or gathered by an index list), attention sink,
//                   V = K, output rope slice rotated back by the position
//
// Rounding points follow the composite path in ucode/dsv4.uc (rms_norm in
// fp32 with an fp16 cast before the weight multiply, rope in fp32).

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define DS_NEG (-1.0e30f)

static __device__ __forceinline__ float
block_sum(float v, float *red)
{
	const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
	const int nw = (blockDim.x + 31) >> 5;

#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		v += __shfl_xor_sync(0xffffffffu, v, o);

	__syncthreads();

	if (lane == 0)
		red[w] = v;

	__syncthreads();

	float t = (lane < nw) ? red[lane] : 0.0f;

#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		t += __shfl_xor_sync(0xffffffffu, t, o);

	return t;
}

static __device__ __forceinline__ float
block_max(float v, float *red)
{
	const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
	const int nw = (blockDim.x + 31) >> 5;

#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));

	__syncthreads();

	if (lane == 0)
		red[w] = v;

	__syncthreads();

	float t = (lane < nw) ? red[lane] : DS_NEG;

#pragma unroll
	for (int o = 16; o > 0; o >>= 1)
		t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));

	return t;
}

// ---------------------------------------------------------------------------
// qkv prep: grid H + 1, block D / 2 (each thread owns the channel pair
// (2t, 2t + 1), which is also the GPT-J rope pair)
// ---------------------------------------------------------------------------

__global__ void amp_dsv4_qkv_kernel(const __half *__restrict__ q_raw,
                                   const __half *__restrict__ kv_raw,
                                   const __half *__restrict__ kv_w,
                                   const float *__restrict__ inv_freq,
                                   const float *__restrict__ pos_f,
                                   const int64_t *__restrict__ pos_i,
                                   __half *__restrict__ kvcache,
                                   __half *__restrict__ q_out, int H, int D,
                                   int rd, float eps, float qscale, int fp8, int kvcap)
{
	__shared__ float red[32];

	// grid: S * (H + 1) blocks; row r = blockIdx.x / (H + 1) is the token at position pos + r
	const int r_row = blockIdx.x / (H + 1), b = blockIdx.x % (H + 1), t = threadIdx.x;
	const bool is_kv = (b == H);
	const __half2 *src = (const __half2 *)(is_kv ? kv_raw + (size_t)r_row * D
	                                              : q_raw + ((size_t)r_row * H + b) * D);
	float2 v = __half22float2(src[t]);
	float ss = block_sum(v.x * v.x + v.y * v.y, red);
	float r = rsqrtf(ss / D + eps);
	float x0, x1;

	if (is_kv) {
		float2 w = __half22float2(((const __half2 *)kv_w)[t]);

		// cast to fp16 before the weight multiply (the composite's rounding)
		x0 = __half2float(__hmul(__float2half(v.x * r), __float2half(w.x)));
		x1 = __half2float(__hmul(__float2half(v.y * r), __float2half(w.y)));
	}
	else {
		x0 = v.x * r;
		x1 = v.y * r;
	}

	const int c0 = 2 * t;

	if (c0 >= D - rd) {
		float sn, cs;

		sincosf((pos_f[0] + (float)r_row) * inv_freq[(c0 - (D - rd)) >> 1], &sn, &cs);

		float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;

		x0 = y0;
		x1 = y1;
	}

	if (is_kv) {
		// DSpark drafter rows: e4m3 fake quantization of the nope part in groups of 64
		// channels (one warp), power-of-two scales — the distribution the drafter was
		// trained on (the reference's act_quant)
		if (fp8 && c0 < D - rd) {
			float am = fmaxf(fabsf(x0), fabsf(x1));

#pragma unroll
			for (int o = 16; o > 0; o >>= 1)
				am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));

			float sc = exp2f(ceilf(log2f(fmaxf(am, 1e-12f) / 448.0f)));

			x0 = (float)__nv_fp8_e4m3(x0 / sc) * sc;
			x1 = (float)__nv_fp8_e4m3(x1 / sc) * sc;
		}

		((__half2 *)(kvcache + (size_t)((pos_i[0] + r_row) % kvcap) * D))[t] = __floats2half2_rn(x0, x1);
	}
	else {
		// (a prefill folds the attention scale into q: one rounding instead of two)
		((__half2 *)(q_out + ((size_t)r_row * H + b) * D))[t] = __floats2half2_rn(x0 * qscale, x1 * qscale);
	}
}

extern "C" int
amp_dsv4_qkv_fp16(const void *q_raw, const void *kv_raw, const void *kv_w,
                 const void *inv_freq, const void *pos_f, const void *pos_i,
                 void *kvcache, void *q_out, int S, int H, int D, int rd, float eps,
                 float qscale, int fp8, int kvcap, cudaStream_t stream)
{
	if (D > 2048 || D % 64 != 0 || rd > D || rd % 2 != 0) {
		fprintf(stderr, "amp_dsv4_qkv: unsupported dims (D %d, rope %d)\n", D, rd);
		return -1;
	}

	amp_dsv4_qkv_kernel<<<S * (H + 1), D / 2, 0, stream>>>((const __half *)q_raw,
		(const __half *)kv_raw, (const __half *)kv_w, (const float *)inv_freq,
		(const float *)pos_f, (const int64_t *)pos_i, (__half *)kvcache,
		(__half *)q_out, H, D, rd, eps, qscale, fp8, kvcap);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// compressor step: one block, hd threads (thread c owns channel c)
// ---------------------------------------------------------------------------

__global__ void amp_dsv4_comp_kernel(const __half *__restrict__ kvr,
                                    const __half *__restrict__ gtr,
                                    const float *__restrict__ ape,
                                    float *__restrict__ buf_kv,
                                    float *__restrict__ buf_g,
                                    float *__restrict__ ovl_kv,
                                    float *__restrict__ ovl_g,
                                    __half *__restrict__ pool,
                                    const __half *__restrict__ norm_w,
                                    const float *__restrict__ inv_freq,
                                    const int64_t *__restrict__ pos_i, int S,
                                    int m, int hd, int W, int rd, float eps)
{
	__shared__ float red[32];
	__shared__ __half nrm[1024];

	const int c = threadIdx.x;
	const bool ovl = (ovl_kv != nullptr);
	const int off = ovl ? hd : 0;           // second-half channels

	// The S tokens in order. The pool row of a completed window is pos / m
	// (no counter: a rolled-back step needs no repair), the overlap half of
	// window w sits in ring slot w % 3 — three slots so that two windows of
	// one multi-token step cannot overwrite what a rollback must restore.
	for (int r = 0; r < S; r++) {
		const int64_t pos = pos_i[0] + r;
		const int slot = (int)(pos % m);

		for (int j = c; j < W; j += blockDim.x) {
			buf_kv[(size_t)slot * W + j] = __half2float(kvr[(size_t)r * W + j]);
			buf_g[(size_t)slot * W + j] = __half2float(gtr[(size_t)r * W + j]);
		}

		if (slot != m - 1)
			continue;

		const int64_t w = pos / m;
		const float *pk = ovl ? ovl_kv + (size_t)((w + 2) % 3) * m * hd : nullptr;   // window w - 1
		const float *pg = ovl ? ovl_g + (size_t)((w + 2) % 3) * m * hd : nullptr;

		__syncthreads();

		float mx = DS_NEG;

		if (ovl)
			for (int j = 0; j < m; j++)
				mx = fmaxf(mx, pg[(size_t)j * hd + c]);

		for (int j = 0; j < m; j++)
			mx = fmaxf(mx, buf_g[(size_t)j * W + off + c] + ape[(size_t)j * W + off + c]);

		float sum = 0.0f, acc = 0.0f;

		if (ovl)
			for (int j = 0; j < m; j++) {
				float e = __expf(pg[(size_t)j * hd + c] - mx);

				sum += e;
				acc += e * pk[(size_t)j * hd + c];
			}

		for (int j = 0; j < m; j++) {
			float e = __expf(buf_g[(size_t)j * W + off + c] + ape[(size_t)j * W + off + c] - mx);

			sum += e;
			acc += e * buf_kv[(size_t)j * W + off + c];
		}

		float xh = __half2float(__float2half(acc / sum));
		float ss = block_sum(xh * xh, red);
		float rr = rsqrtf(ss / hd + eps);
		__half nv = __hmul(__float2half(xh * rr), norm_w[c]);

		nrm[c] = nv;
		__syncthreads();

		float y = __half2float(nv);

		if (c >= hd - rd) {
			float sn, cs;
			int i = (c - (hd - rd)) >> 1;

			sincosf((float)(pos - (m - 1)) * inv_freq[i], &sn, &cs);

			if ((c & 1) == 0)
				y = __half2float(nv) * cs - __half2float(nrm[c + 1]) * sn;
			else
				y = __half2float(nrm[c - 1]) * sn + __half2float(nv) * cs;
		}

		pool[(size_t)w * hd + c] = __float2half(y);

		if (ovl) {
			float *wk = ovl_kv + (size_t)(w % 3) * m * hd;
			float *wg = ovl_g + (size_t)(w % 3) * m * hd;

			for (int j = 0; j < m; j++) {
				wk[(size_t)j * hd + c] = buf_kv[(size_t)j * W + c];
				wg[(size_t)j * hd + c] = buf_g[(size_t)j * W + c] + ape[(size_t)j * W + c];
			}
		}

		__syncthreads();
	}
}

extern "C" int
amp_dsv4_comp_fp16(const void *kvr, const void *gtr, const void *ape,
                  void *buf_kv, void *buf_g, void *ovl_kv, void *ovl_g,
                  void *pool, const void *norm_w, const void *inv_freq,
                  const void *pos_i, int S, int m, int hd, int W, int rd,
                  float eps, cudaStream_t stream)
{
	if (hd > 1024 || hd % 32 != 0 || rd > hd || rd % 2 != 0 ||
	    (ovl_kv ? W != 2 * hd : W != hd)) {
		fprintf(stderr, "amp_dsv4_comp: unsupported dims (hd %d, W %d, rope %d)\n", hd, W, rd);
		return -1;
	}

	amp_dsv4_comp_kernel<<<1, hd, 0, stream>>>((const __half *)kvr,
		(const __half *)gtr, (const float *)ape, (float *)buf_kv, (float *)buf_g,
		(float *)ovl_kv, (float *)ovl_g, (__half *)pool, (const __half *)norm_w,
		(const float *)inv_freq, (const int64_t *)pos_i, S, m, hd, W, rd, eps);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// decode attention: grid H, block 256 (8 warps). Keys: the W-row sliding
// window (cache rows pos - W + 1 .. pos) then the pool entries: dense mode
// (idx == null) takes entries j with (j + 1) * m <= pos + 1, gathered mode the
// rows listed in idx (-1 = invalid).
// ---------------------------------------------------------------------------

#define DS_MAX_KEYS 11000

__global__ void amp_dsv4_attn_kernel(const __half *__restrict__ q,
                                    const __half *__restrict__ kvc,
                                    const int64_t *__restrict__ pos_i,
                                    const float *__restrict__ pos_f,
                                    const __half *__restrict__ pool,
                                    const int64_t *__restrict__ idx,
                                    const float *__restrict__ sinks,
                                    const float *__restrict__ inv_freq,
                                    __half *__restrict__ out, int H, int D, int rd,
                                    int W, int m, int np_max, float scale, int nc_block, int kvcap)
{
	extern __shared__ float sc[];
	__shared__ float qs[1024];
	__shared__ float red[32];

	// grid: S * H blocks; block (r, h) is the head h of the token at position pos + r
	const int r_row = blockIdx.x / H, h = blockIdx.x % H, tid = threadIdx.x;
	const int lane = tid & 31, warp = tid >> 5;
	const int64_t pos = pos_i[0] + r_row;

	// window keys: causal mode = the W rows ending at pos; block mode (nc_block = B
	// queries at p0 .. p0 + B - 1, all seeing the same keys) = the last min(W, p0)
	// rows before p0 plus the B block rows
	int nw;
	int64_t wbase;

	if (nc_block) {
		int hist = (int)min((int64_t)W, pos_i[0]);

		nw = hist + nc_block;
		wbase = pos_i[0] - hist;
	}
	else {
		nw = W;
		wbase = pos - (W - 1);
	}

	for (int d = tid; d < D; d += blockDim.x)
		qs[d] = __half2float(q[((size_t)r_row * H + h) * D + d]);

	__syncthreads();

	int np;

	if (pool == nullptr)
		np = 0;
	else if (idx != nullptr)
		np = np_max;
	else {
		np = (int)((pos + 1) / m);

		if (np > np_max)
			np = np_max;
	}

	const int nk = nw + np;

	// scores: a warp per key, lanes split the channels
	for (int t = warp; t < nk; t += 8) {
		const __half *kp = nullptr;

		if (t < nw) {
			int64_t r = wbase + t;

			if (r >= 0)
				kp = kvc + (size_t)(r % kvcap) * D;
		}
		else if (idx != nullptr) {
			int64_t id = idx[(size_t)r_row * np_max + (t - nw)];

			if (id >= 0)
				kp = pool + (size_t)id * D;
		}
		else {
			kp = pool + (size_t)(t - nw) * D;
		}

		float s = 0.0f;

		if (kp) {
			for (int d = lane * 8; d < D; d += 256) {
				uint4 u = *(const uint4 *)(kp + d);
				const __half2 *p2 = (const __half2 *)&u;

#pragma unroll
				for (int e = 0; e < 4; e++) {
					float2 f = __half22float2(p2[e]);

					s += f.x * qs[d + 2 * e] + f.y * qs[d + 2 * e + 1];
				}
			}
		}

#pragma unroll
		for (int o = 16; o > 0; o >>= 1)
			s += __shfl_xor_sync(0xffffffffu, s, o);

		if (lane == 0)
			sc[t] = kp ? s * scale : DS_NEG;
	}

	__syncthreads();

	float mx = sinks[h];

	for (int t = tid; t < nk; t += blockDim.x)
		mx = fmaxf(mx, sc[t]);

	mx = block_max(mx, red);

	float sum = 0.0f;

	for (int t = tid; t < nk; t += blockDim.x) {
		float e = __expf(sc[t] - mx);

		sc[t] = e;
		sum += e;
	}

	sum = block_sum(sum, red);
	sum += __expf(sinks[h] - mx);

	const float inv = 1.0f / sum;

	__syncthreads();

	// P @ K: thread owns the channel pair (2 tid, 2 tid + 1)
	float a0 = 0.0f, a1 = 0.0f;
	const int c0 = 2 * tid;

	if (c0 < D) {
		for (int t = 0; t < nk; t++) {
			float p = sc[t];

			if (p == 0.0f)
				continue;

			const __half *kp;

			if (t < nw)
				kp = kvc + (size_t)((wbase + t) % kvcap) * D;
			else if (idx != nullptr)
				kp = pool + (size_t)idx[(size_t)r_row * np_max + (t - nw)] * D;
			else
				kp = pool + (size_t)(t - nw) * D;

			float2 f = __half22float2(*(const __half2 *)(kp + c0));

			a0 += p * f.x;
			a1 += p * f.y;
		}

		a0 *= inv;
		a1 *= inv;

		// rotate the rope slice back by the query position
		if (c0 >= D - rd) {
			float sn, cs;

			sincosf((pos_f[0] + (float)r_row) * inv_freq[(c0 - (D - rd)) >> 1], &sn, &cs);

			float y0 = a0 * cs + a1 * sn, y1 = -a0 * sn + a1 * cs;

			a0 = y0;
			a1 = y1;
		}

		((__half2 *)(out + ((size_t)r_row * H + h) * D))[tid] = __floats2half2_rn(a0, a1);
	}
}

extern "C" int
amp_dsv4_attn_fp16(const void *q, const void *kvc, const void *pos_i,
                  const void *pos_f, const void *pool, const void *idx,
                  const void *sinks, const void *inv_freq, void *out, int S, int H,
                  int D, int rd, int W, int m, int np_max, float scale, int nc_block, int kvcap,
                  cudaStream_t stream)
{
	if (D > 1024 || D % 64 != 0 || rd > D || rd % 2 != 0 || W + nc_block + np_max > DS_MAX_KEYS) {
		fprintf(stderr, "amp_dsv4_attn: unsupported dims (D %d, keys %d)\n", D, W + nc_block + np_max);
		return -1;
	}

	size_t smem = (size_t)(W + nc_block + np_max) * sizeof(float);

	amp_dsv4_attn_kernel<<<S * H, 256, smem, stream>>>((const __half *)q,
		(const __half *)kvc, (const int64_t *)pos_i, (const float *)pos_f,
		(const __half *)pool, (const int64_t *)idx, (const float *)sinks,
		(const float *)inv_freq, (__half *)out, H, D, rd, W, m, np_max, scale, nc_block, kvcap);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// output de-rotation (prefill): o (R, H, D) fp16 -> rope slice rotated back by
// the query position pos_f + r. Grid R * H, block D / 2.
// ---------------------------------------------------------------------------

__global__ void amp_dsv4_derot_kernel(const __half *__restrict__ o,
                                     const float *__restrict__ inv_freq,
                                     const float *__restrict__ pos_f,
                                     __half *__restrict__ out, int H, int D, int rd)
{
	const int blk = blockIdx.x, t = threadIdx.x;
	const int r = blk / H;
	float2 v = __half22float2(((const __half2 *)(o + (size_t)blk * D))[t]);
	const int c0 = 2 * t;

	if (c0 >= D - rd) {
		float sn, cs;

		sincosf((pos_f[0] + (float)r) * inv_freq[(c0 - (D - rd)) >> 1], &sn, &cs);

		float y0 = v.x * cs + v.y * sn, y1 = -v.x * sn + v.y * cs;

		v.x = y0;
		v.y = y1;
	}

	((__half2 *)(out + (size_t)blk * D))[t] = __floats2half2_rn(v.x, v.y);
}

extern "C" int
amp_dsv4_derot_fp16(const void *o, const void *inv_freq, const void *pos_f, void *out,
                   int R, int H, int D, int rd, cudaStream_t stream)
{
	if (D > 2048 || D % 64 != 0 || rd > D || rd % 2 != 0)
		return -1;

	amp_dsv4_derot_kernel<<<R * H, D / 2, 0, stream>>>((const __half *)o,
		(const float *)inv_freq, (const float *)pos_f, (__half *)out, H, D, rd);

	return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// additive attention mask (prefill sub-chunk): (R, nk + ne) fp16, 0 where the key is
// visible, -60000 where not. Columns [0, nk): window rows base + c visible to the query
// qpos0 + r iff qpos - (W - 1) <= key <= qpos; columns [nk, nk + ne): pool entry j visible
// iff (j + 1) * m <= qpos + 1.
// ---------------------------------------------------------------------------

__global__ void amp_dsv4_mask_kernel(__half *__restrict__ out, int R, int nk, int ne,
                                    long qpos0, long base, int W, int m, long ncstart)
{
	const int ntot = nk + ne;
	long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= (long)R * ntot)
		return;

	const int r = (int)(i / ntot), c = (int)(i % ntot);
	const long qpos = qpos0 + r;
	bool ok;

	if (c < nk) {
		long key = base + c;

		ok = (key <= qpos) && (key >= qpos - (W - 1));

		// image chunk (ncstart >= 0): every key of the chunk is visible to every chunk query
		if (ncstart >= 0 && key >= ncstart && qpos >= ncstart)
			ok = true;
	}
	else {
		ok = ((long)(c - nk + 1) * m <= qpos + 1);
	}

	out[i] = ok ? __float2half(0.0f) : __float2half(-60000.0f);
}

extern "C" int
amp_dsv4_mask_fp16(void *out, int R, int nk, int ne, long qpos0, long base, int W, int m,
                  long ncstart, cudaStream_t stream)
{
	long n = (long)R * (nk + ne);

	amp_dsv4_mask_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((__half *)out, R, nk, ne,
		qpos0, base, W, m, ncstart);

	return (int)cudaGetLastError();
}
