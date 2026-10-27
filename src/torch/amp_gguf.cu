// amp_gguf.cu — dequantization of ggml block types (GGUF weights) on the GPU.
//
// Written from the reference dequantize_row_* functions of ggml (ggml/src/ggml-quants.c, MIT, see vendor/ggml/VERSION): the
// values are bit-identical to them (tests/gguf_dequant.uc against tests/oracle/make_gguf_dequant.py). A quantized weight
// is a uint8 tensor of rows of blocks (the layout of the file); a thread turns one group of 8 consecutive weights of a
// block into 8 outputs (fp32 math, then the output dtype), so a block's reads are shared by neighbouring threads and
// the stores are 16 bytes wide in fp16 / bf16.
//
// Block layouts (bytes), all little endian, a block is 256 weights (QK_K) unless noted:
//   Q8_0     (32 w)   34: d f16, qs[32] i8
//   IQ4_NL   (32 w)   18: d f16, qs[16] (two 4-bit indices into kvalues_iq4nl)
//   Q5_K             176: d f16, dmin f16, scales[12] (6-bit scale + min per 32 w, packed), qh[32], qs[128]
//   IQ2_XXS           66: d f16, qs[32] u16 (per 32 w: 4 grid indices of 8 bits, then 4 x 7 sign bits + 4 scale bits)
//   IQ2_XS            74: d f16, qs[32] u16 (9-bit grid index + 7-bit signs), scales[8] (two 4-bit scales per 32 w)
//   IQ1_M             56: qs[32], qh[16], scales[8] (four u16: 3-bit scales, the top nibbles carry the f16 super scale)
//   IQ1_S             50: d f16, qs[32], qh[8] u16 (per 32 w: a 3-bit scale, a sign of the +-1/8 offset and 4 x 3 high bits of the grid indices)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f

#define GGML_TABLE_BEGIN(type, name, size) static __device__ const type name[size] = {
#define GGML_TABLE_END() };
#include "../../vendor/ggml/ggml_tables.inc"

enum {
	AMP_Q_Q8_0, AMP_Q_IQ4_NL, AMP_Q_Q5_K, AMP_Q_IQ2_XXS, AMP_Q_IQ2_XS, AMP_Q_IQ1_M, AMP_Q_IQ1_S,
	AMP_Q_Q2_0, AMP_Q_Q4_K, AMP_Q_Q3_K, AMP_Q_Q6_K, AMP_Q_IQ2_S, AMP_Q_IQ3_XXS, AMP_Q_IQ3_S, AMP_Q_IQ4_XS,
	AMP_Q_Q4_0,
};

struct amp_qinfo {
	int block;       /* weights per block */
	int bytes;       /* bytes per block */
};

static __host__ __device__ constexpr amp_qinfo amp_qtype_info(int t)
{
	switch (t) {
	case AMP_Q_Q4_0:    return { 32, 18 };
	case AMP_Q_Q8_0:    return { 32, 34 };
	case AMP_Q_IQ4_NL:  return { 32, 18 };
	case AMP_Q_Q5_K:    return { 256, 176 };
	case AMP_Q_IQ2_XXS: return { 256, 66 };
	case AMP_Q_IQ2_XS:  return { 256, 74 };
	case AMP_Q_IQ1_M:   return { 256, 56 };
	case AMP_Q_IQ1_S:   return { 256, 50 };
	case AMP_Q_Q2_0:    return { 64, 18 };
	case AMP_Q_Q4_K:    return { 256, 144 };
	case AMP_Q_Q3_K:    return { 256, 110 };
	case AMP_Q_Q6_K:    return { 256, 210 };
	case AMP_Q_IQ2_S:   return { 256, 82 };
	case AMP_Q_IQ3_XXS: return { 256, 98 };
	case AMP_Q_IQ3_S:   return { 256, 110 };
	case AMP_Q_IQ4_XS:  return { 256, 136 };
	default:            return { 256, 56 };
	}
}

static __host__ __device__ constexpr int amp_qgpb(int t)
{
	return t == AMP_Q_Q4_0 || t == AMP_Q_Q8_0 || t == AMP_Q_IQ4_NL ? 4 : t == AMP_Q_Q2_0 ? 8 : 32;
}

static __host__ __device__ constexpr int amp_qbytes(int t)
{
	return amp_qtype_info(t).bytes;
}

static __device__ inline float amp_h2f(const uint8_t *p)
{
	return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

static __device__ inline unsigned amp_u16(const uint8_t *p)
{
	return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static __device__ inline void amp_scale_min_k4(int j, const uint8_t *q, int *d, int *m)
{
	if (j < 4) {
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	} else {
		*d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

/* group g (8 weights) of a block -> v[8] */
/* gr / sg: the lookup tables (iq2xxs_grid or iq2xs_grid, ksigns_iq2xs): the global ones, or copies in shared memory */
template <int T> __device__ void amp_dq_group(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg);

template <> __device__ void amp_dq_group<AMP_Q_Q4_0>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2 + (g & 1) * 8;
	const int hi = g >> 1;

#pragma unroll
	for (int j = 0; j < 8; j++) {
		const int q = hi ? qs[j] >> 4 : qs[j] & 0xf;

		v[j] = (float)(q - 8) * d;
	}
}

template <> __device__ void amp_dq_group<AMP_Q_Q8_0>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = (float)(int8_t)b[2 + g * 8 + j] * d;
}

template <> __device__ void amp_dq_group<AMP_Q_IQ4_NL>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2 + (g & 1) * 8;
	const int hi = g >> 1;

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = d * (float)kvalues_iq4nl[hi ? qs[j] >> 4 : qs[j] & 0xf];
}

template <> __device__ void amp_dq_group<AMP_Q_Q5_K>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b), dmin = amp_h2f(b + 2);
	const uint8_t *scales = b + 4, *qh = b + 16, *qs = b + 48;
	const int jj = g >> 3, half = (g >> 2) & 1, l0 = (g & 3) * 8;     /* 64-weight chunk, its half, offset in the half */
	int sc, m;

	amp_scale_min_k4(2 * jj + half, scales, &sc, &m);

	const float d1 = d * sc, m1 = dmin * m;

#pragma unroll
	for (int l = 0; l < 8; l++) {
		const int q = half ? qs[32 * jj + l0 + l] >> 4 : qs[32 * jj + l0 + l] & 0xF;

		v[l] = d1 * (float)(q + ((qh[l0 + l] >> (2 * jj + half)) & 1 ? 16 : 0)) - m1;
	}
}

template <> __device__ void amp_dq_group<AMP_Q_IQ2_XXS>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const int ib32 = g >> 2, l = g & 3;
	const uint8_t *q = b + 2 + 8 * ib32;
	const unsigned aux1 = (unsigned)q[4] | ((unsigned)q[5] << 8) | ((unsigned)q[6] << 16) | ((unsigned)q[7] << 24);
	const float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
	const uint64_t grid = ((const uint64_t *)gr)[q[l]];
	const uint8_t signs = sg[(aux1 >> (7 * l)) & 127];

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = db * (float)((grid >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template <> __device__ void amp_dq_group<AMP_Q_IQ2_XS>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const int ib32 = g >> 2, l = g & 3;
	const unsigned q = amp_u16(b + 2 + 2 * (4 * ib32 + l));
	const uint8_t sc = b[2 + 64 + ib32];
	const float db = d * (0.5f + (float)(l < 2 ? sc & 0xf : sc >> 4)) * 0.25f;
	const uint64_t grid = ((const uint64_t *)gr)[q & 511];
	const uint8_t signs = sg[q >> 9];

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = db * (float)((grid >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template <> __device__ void amp_dq_group<AMP_Q_IQ1_M>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const uint8_t *qs = b, *qh = b + 32, *sb = b + 48;
	const unsigned s0 = amp_u16(sb), s1 = amp_u16(sb + 2), s2 = amp_u16(sb + 4), s3 = amp_u16(sb + 6);
	const unsigned su = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
	const float d = __half2float(__ushort_as_half((unsigned short)su));
	const int ib = g >> 2, l = g & 3;
	const unsigned sc = amp_u16(sb + 2 * (ib >> 1));
	const float dl = d * (float)(2 * ((sc >> (6 * (ib & 1) + (l >> 1) * 3)) & 0x7) + 1);
	const unsigned h = qh[2 * ib + (l >> 1)];
	const unsigned idx = qs[4 * ib + l] | (((l & 1) ? h << 4 : h << 8) & 0x700);
	const float delta = h & ((l & 1) ? 0x80 : 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
	const uint64_t grid = iq1s_grid[idx];

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = dl * ((float)(int8_t)((grid >> (8 * j)) & 0xff) + delta);
}

template <> __device__ void amp_dq_group<AMP_Q_IQ1_S>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const int ib = g >> 2, l = g & 3;
	const unsigned qh = amp_u16(b + 34 + 2 * ib);
	const unsigned idx = b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
	const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
	const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
	const uint64_t grid = iq1s_grid[idx];

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = dl * ((float)(int8_t)((grid >> (8 * j)) & 0xff) + delta);
}


template <> __device__ void amp_dq_group<AMP_Q_Q2_0>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2 + 2 * g;

#pragma unroll
	for (int j = 0; j < 8; j++) {
		const int q = (qs[j / 4] >> ((j % 4) * 2)) & 3;

		v[j] = ((float)q - 1.f) * d;
	}
}

template <> __device__ void amp_dq_group<AMP_Q_Q4_K>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b), dmin = amp_h2f(b + 2);
	const int c = g >> 3, half = (g >> 2) & 1, l0 = (g & 3) * 8;
	int sc, m;

	amp_scale_min_k4(2 * c + half, b + 4, &sc, &m);

	const float d1 = d * (float)sc, m1 = dmin * (float)m;
	const uint8_t *qs = b + 16 + 32 * c + l0;

#pragma unroll
	for (int l = 0; l < 8; l++) {
		const int q = half ? qs[l] >> 4 : qs[l] & 0xF;

		v[l] = d1 * (float)q - m1;
	}
}

template <> __device__ void amp_dq_group<AMP_Q_Q3_K>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b + 108);
	const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
	uint32_t aux[4];

#pragma unroll
	for (int i = 0; i < 4; i++)
		aux[i] = (uint32_t)b[96 + 4 * i] | ((uint32_t)b[97 + 4 * i] << 8) |
		         ((uint32_t)b[98 + 4 * i] << 16) | ((uint32_t)b[99 + 4 * i] << 24);

	const uint32_t tmp = aux[2];

	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

	const int8_t *scales = (const int8_t *)aux;
	const int n = g >> 4, local = g & 15, j = local >> 2, sub = local & 3;
	const int si = n * 8 + j * 2 + (sub >= 2 ? 1 : 0);
	const float dl = d * ((float)scales[si] - 32.f);
	const int shift = 2 * j, half2 = (sub >= 2);
	const uint8_t mbit = (uint8_t)(1 << (j + 4 * n));
	const int hm_off = (sub & 1) * 8 + (half2 ? 16 : 0);
	const int q_off = 32 * n + (sub & 1) * 8 + (half2 ? 16 : 0);

#pragma unroll
	for (int l = 0; l < 8; l++) {
		const int q = (b[32 + q_off + l] >> shift) & 3;
		const int h = (b[hm_off + l] & mbit) ? 0 : 4;

		v[l] = dl * ((float)q - (float)h);
	}
}

template <> __device__ void amp_dq_group<AMP_Q_Q6_K>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b + 208);
	const int n = g >> 4, local = g & 15;
	const int qidx = local >> 2, l0 = (local & 3) * 8;
	const uint8_t *ql = b + n * 64, *qh = b + 128 + n * 32;
	const int8_t *sc = (const int8_t *)(b + 192 + n * 8);
	const float dl = d * (float)sc[l0 / 16 + 2 * qidx];

#pragma unroll
	for (int j = 0; j < 8; j++) {
		const int l = l0 + j;
		int q;

		if (qidx == 0)
			q = (int)(ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
		else if (qidx == 1)
			q = (int)(ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
		else if (qidx == 2)
			q = (int)(ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
		else
			q = (int)(ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);

		v[j] = dl * (float)(q - 32);
	}
}

template <> __device__ void amp_dq_group<AMP_Q_IQ2_S>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2, *qh = qs + 64, *signs = qs + 32, *scales = qs + 64 + 8;
	const int ib32 = g >> 2, l = g & 3;
	const float db = d * (0.5f + (float)(l < 2 ? scales[ib32] & 0xf : scales[ib32] >> 4)) * 0.25f;
	const uint64_t grid = ((const uint64_t *)gr)[qs[ib32 * 4 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
	const uint8_t sgn = signs[ib32 * 4 + l];

#pragma unroll
	for (int j = 0; j < 8; j++)
		v[j] = db * (float)((grid >> (8 * j)) & 0xff) * (sgn & kmask_iq2xs[j] ? -1.f : 1.f);
}

template <> __device__ void amp_dq_group<AMP_Q_IQ3_XXS>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2, *ss = qs + 64;
	const int ib32 = g >> 2, l = g & 3;
	const uint32_t aux = (uint32_t)ss[4 * ib32] | ((uint32_t)ss[4 * ib32 + 1] << 8) |
	                     ((uint32_t)ss[4 * ib32 + 2] << 16) | ((uint32_t)ss[4 * ib32 + 3] << 24);
	const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
	const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
	const uint32_t g1 = ((const uint32_t *)gr)[qs[ib32 * 8 + 2 * l]];
	const uint32_t g2 = ((const uint32_t *)gr)[qs[ib32 * 8 + 2 * l + 1]];

#pragma unroll
	for (int j = 0; j < 4; j++) {
		v[j] = db * (float)((g1 >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
		v[j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
	}
}

template <> __device__ void amp_dq_group<AMP_Q_IQ3_S>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint8_t *qs = b + 2, *qh = qs + 64, *signs = qh + 8, *scales = signs + 32;
	const int ib32 = g >> 2, p = ib32 >> 1, half = ib32 & 1, l = g & 3;
	const float db = d * (1.f + 2.f * (float)(half ? scales[p] >> 4 : scales[p] & 0xf));
	const uint8_t qhb = qh[2 * p + half];
	const int qoff = p * 16 + half * 8 + 2 * l;
	const uint32_t g1 = ((const uint32_t *)gr)[qs[qoff] | ((qhb << (8 - 2 * l)) & 256)];
	const uint32_t g2 = ((const uint32_t *)gr)[qs[qoff + 1] | ((qhb << (7 - 2 * l)) & 256)];
	const uint8_t sgn = signs[p * 8 + half * 4 + l];

#pragma unroll
	for (int j = 0; j < 4; j++) {
		v[j] = db * (float)((g1 >> (8 * j)) & 0xff) * (sgn & kmask_iq2xs[j] ? -1.f : 1.f);
		v[j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * (sgn & kmask_iq2xs[j + 4] ? -1.f : 1.f);
	}
}

template <> __device__ void amp_dq_group<AMP_Q_IQ4_XS>(const uint8_t *b, int g, float *v, const void *gr, const uint8_t *sg)
{
	const float d = amp_h2f(b);
	const uint16_t sh = (uint16_t)(b[2] | (b[3] << 8));
	const uint8_t *sl = b + 4, *qs = b + 8;
	const int ib = g >> 2, sub = g & 3;
	const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
	const float dl = d * (float)(ls - 32);
	const int off = ib * 16 + (sub & 1) * 8;

#pragma unroll
	for (int j = 0; j < 8; j++) {
		const int q = qs[off + j];
		const int idx = (sub < 2) ? (q & 0xf) : (q >> 4);

		v[j] = dl * (float)kvalues_iq4nl[idx];
	}
}

template <typename O> __device__ inline void amp_store8(O *out, const float *v);

template <> __device__ inline void amp_store8<float>(float *out, const float *v)
{
	((float4 *)out)[0] = make_float4(v[0], v[1], v[2], v[3]);
	((float4 *)out)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

template <> __device__ inline void amp_store8<__half>(__half *out, const float *v)
{
	__half h[8];

#pragma unroll
	for (int j = 0; j < 8; j++)
		h[j] = __float2half(v[j]);

	*(uint4 *)out = *(uint4 *)h;
}

template <> __device__ inline void amp_store8<__nv_bfloat16>(__nv_bfloat16 *out, const float *v)
{
	__nv_bfloat16 h[8];

#pragma unroll
	for (int j = 0; j < 8; j++)
		h[j] = __float2bfloat16(v[j]);

	*(uint4 *)out = *(uint4 *)h;
}

template <int T, typename O>
__global__ void amp_gguf_dequant_kernel(const uint8_t *__restrict__ x, O *__restrict__ y, long ngroups)
{
	const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= ngroups)
		return;

	constexpr int gpb = amp_qgpb(T);
	constexpr int bytes = amp_qbytes(T);
	const void *gr = T == AMP_Q_IQ2_XS ? (const void *)iq2xs_grid :
	                 T == AMP_Q_IQ2_XXS ? (const void *)iq2xxs_grid :
	                 T == AMP_Q_IQ2_S ? (const void *)iq2s_grid :
	                 T == AMP_Q_IQ3_XXS ? (const void *)iq3xxs_grid :
	                 T == AMP_Q_IQ3_S ? (const void *)iq3s_grid :
	                 (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? (const void *)iq1s_grid : NULL;
	float v[8];

	amp_dq_group<T>(x + (i / gpb) * bytes, (int)(i % gpb), v, gr, ksigns_iq2xs);
	amp_store8<O>(y + i * 8, v);
}

template <int T, typename O>
static int amp_gguf_dequant_launch(const void *x, void *y, long nblocks, cudaStream_t stream)
{
	const long ngroups = nblocks * (amp_qtype_info(T).block / 8);

	amp_gguf_dequant_kernel<T, O><<<(unsigned)((ngroups + 255) / 256), 256, 0, stream>>>((const uint8_t *)x, (O *)y,
		ngroups);

	return (int)cudaGetLastError();
}

/* qtype: AMP_Q_*; odt: 0 fp32, 1 fp16, 2 bf16. x holds nblocks blocks, y nblocks * block weights. */
extern "C" int
amp_gguf_dequant(int qtype, int odt, const void *x, void *y, long nblocks, cudaStream_t stream)
{
	if (nblocks < 0 || ((uintptr_t)y & 15))
		return (int)cudaErrorInvalidValue;

#define AMP_DQ_CASE(T) \
	case T: \
		return odt == 0 ? amp_gguf_dequant_launch<T, float>(x, y, nblocks, stream) : \
		       odt == 1 ? amp_gguf_dequant_launch<T, __half>(x, y, nblocks, stream) : \
		                  amp_gguf_dequant_launch<T, __nv_bfloat16>(x, y, nblocks, stream)

	switch (qtype) {
	AMP_DQ_CASE(AMP_Q_Q4_0);
	AMP_DQ_CASE(AMP_Q_Q8_0);
	AMP_DQ_CASE(AMP_Q_IQ4_NL);
	AMP_DQ_CASE(AMP_Q_Q5_K);
	AMP_DQ_CASE(AMP_Q_IQ2_XXS);
	AMP_DQ_CASE(AMP_Q_IQ2_XS);
	AMP_DQ_CASE(AMP_Q_IQ1_M);
	AMP_DQ_CASE(AMP_Q_IQ1_S);
	AMP_DQ_CASE(AMP_Q_Q2_0);
	AMP_DQ_CASE(AMP_Q_Q4_K);
	AMP_DQ_CASE(AMP_Q_Q3_K);
	AMP_DQ_CASE(AMP_Q_Q6_K);
	AMP_DQ_CASE(AMP_Q_IQ2_S);
	AMP_DQ_CASE(AMP_Q_IQ3_XXS);
	AMP_DQ_CASE(AMP_Q_IQ3_S);
	AMP_DQ_CASE(AMP_Q_IQ4_XS);
	}
#undef AMP_DQ_CASE

	return (int)cudaErrorInvalidValue;
}

/* ---- matrix-vector products on the quantized rows --------------------------------------------------------------------
 *
 * y[m][n] = sum_k x[m][k] * W[n][k] for up to 8 rows m (decode, speculative verify), fp32 accumulation; and the routed-
 * experts form: out[slot][n] = sum_k x[slot / xdiv][k] * W[ids[slot]][n][k] over a stack of experts (E, N, row bytes).
 * One warp per output row; a lane takes groups of 8 weights (dequantized by the functions above, so the values are the
 * ones the dequantization kernel gives) and multiplies them with the 8 x values of every row. The x rows are read
 * through the caches (they are a few KB). */

template <int T, bool MOE>
__global__ void amp_gguf_gemv_kernel(const __half *__restrict__ x, const uint8_t *__restrict__ w,
                                     const long long *__restrict__ ids, void *__restrict__ out, int M, int N, int K,
                                     int xdiv, const long long *__restrict__ ptrs)
{
	constexpr int gpb = amp_qgpb(T);
	constexpr int bytes = amp_qbytes(T);
	constexpr int blk = gpb * 8;

	/* the i-quants' lookup tables in shared memory: a lane's index is random, the global copies miss */
	constexpr int NGRID = T == AMP_Q_IQ2_XXS ? 256 : T == AMP_Q_IQ2_XS ? 512 : 0;
	__shared__ uint64_t s_grid[NGRID ? NGRID : 1];
	__shared__ uint8_t s_signs[NGRID ? 128 : 1];

	if (NGRID) {
		for (int i = threadIdx.x; i < NGRID; i += blockDim.x)
			s_grid[i] = (T == AMP_Q_IQ2_XS) ? iq2xs_grid[i] : iq2xxs_grid[i];

		for (int i = threadIdx.x; i < 128; i += blockDim.x)
			s_signs[i] = ksigns_iq2xs[i];

		__syncthreads();
	}

	const void *gr = NGRID ? (const void *)s_grid :
	                 T == AMP_Q_IQ2_S ? (const void *)iq2s_grid :
	                 T == AMP_Q_IQ3_XXS ? (const void *)iq3xxs_grid :
	                 T == AMP_Q_IQ3_S ? (const void *)iq3s_grid :
	                 (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? (const void *)iq1s_grid : NULL;
	const uint8_t *sg = NGRID ? s_signs : ksigns_iq2xs;

	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int n = blockIdx.x * (blockDim.x >> 5) + warp;
	const int slot = blockIdx.y;

	if (n >= N)
		return;

	const size_t rb = (size_t)(K / blk) * bytes;
	const uint8_t *row;
	const __half *xr = x;
	int rows = M;

	if (MOE) {
		const long long e = ids[slot];

		if (e < 0) {                       /* an expert of another rank: no contribution */
			if (lane == 0)
				((float *)out)[(size_t)slot * N + n] = 0.f;

			return;
		}

		if (ptrs)
			row = (const uint8_t *)ptrs[e] + (size_t)n * rb;
		else
			row = w + ((size_t)e * N + n) * rb;

		xr = x + (size_t)(slot / xdiv) * K;
		rows = 1;
	} else {
		row = w + (size_t)n * rb;
	}

	float acc[8];

#pragma unroll
	for (int m = 0; m < 8; m++)
		acc[m] = 0.f;

	const int ngroups = K / 8;

	for (int g = lane; g < ngroups; g += 32) {
		float v[8];

		amp_dq_group<T>(row + (size_t)(g / gpb) * bytes, g % gpb, v, gr, sg);

#pragma unroll
		for (int m = 0; m < 8; m++) {
			if (m < rows) {
				const uint4 xv = *(const uint4 *)(xr + (size_t)m * K + (size_t)g * 8);
				const __half2 *h = (const __half2 *)&xv;
				float s = 0.f;

#pragma unroll
				for (int j = 0; j < 4; j++) {
					const float2 f = __half22float2(h[j]);

					s += v[2 * j] * f.x + v[2 * j + 1] * f.y;
				}

				acc[m] += s;
			}
		}
	}

#pragma unroll
	for (int m = 0; m < 8; m++) {
		float s = acc[m];

#pragma unroll
		for (int o = 16; o > 0; o >>= 1)
			s += __shfl_xor_sync(0xffffffffu, s, o);

		if (lane == 0 && m < rows) {
			if (MOE)
				((float *)out)[(size_t)slot * N + n] = s;
			else
				((__half *)out)[(size_t)m * N + n] = __float2half(s);
		}
	}
}

template <int T>
static int amp_gguf_gemv_launch(bool moe, const void *x, const void *w, const void *ids, void *out, int M, int N, int K,
                                int xdiv, int slots, const void *ptrs, cudaStream_t stream)
{
	const dim3 blk(256);

	if (moe) {
		const dim3 grid((N + 7) / 8, slots);

		amp_gguf_gemv_kernel<T, true><<<grid, blk, 0, stream>>>((const __half *)x, (const uint8_t *)w,
			(const long long *)ids, out, M, N, K, xdiv, (const long long *)ptrs);
	} else {
		const dim3 grid((N + 7) / 8, 1);

		amp_gguf_gemv_kernel<T, false><<<grid, blk, 0, stream>>>((const __half *)x, (const uint8_t *)w, NULL, out, M,
			N, K, xdiv, NULL);
	}

	return (int)cudaGetLastError();
}

/* Dense (ids == NULL): x (M, K) fp16, w (N, row bytes), y (M, N) fp16, M <= 8. Experts: x (R, K) fp16, ids (slots) int64,
 * w (E, N, row bytes), out (slots, N) fp32, slot s using x row s / xdiv. K is a multiple of the block size. */
extern "C" int
amp_gguf_gemv(int qtype, const void *x, const void *w, const void *ids, void *out, int M, int N, int K, int xdiv,
              int slots, cudaStream_t stream)
{
	const amp_qinfo qi = amp_qtype_info(qtype);
	const bool moe = ids != NULL;

	if (N < 1 || K < qi.block || (K % qi.block) || (!moe && (M < 1 || M > 8)) || (moe && (slots < 1 || xdiv < 1)) ||
	    ((uintptr_t)x & 15))
		return (int)cudaErrorInvalidValue;

#define AMP_GV_CASE(T) \
	case T: return amp_gguf_gemv_launch<T>(moe, x, w, ids, out, M, N, K, xdiv, slots, NULL, stream)

	switch (qtype) {
	AMP_GV_CASE(AMP_Q_Q4_0);
	AMP_GV_CASE(AMP_Q_Q8_0);
	AMP_GV_CASE(AMP_Q_IQ4_NL);
	AMP_GV_CASE(AMP_Q_Q5_K);
	AMP_GV_CASE(AMP_Q_IQ2_XXS);
	AMP_GV_CASE(AMP_Q_IQ2_XS);
	AMP_GV_CASE(AMP_Q_IQ1_M);
	AMP_GV_CASE(AMP_Q_IQ1_S);
	AMP_GV_CASE(AMP_Q_Q2_0);
	AMP_GV_CASE(AMP_Q_Q4_K);
	AMP_GV_CASE(AMP_Q_Q3_K);
	AMP_GV_CASE(AMP_Q_Q6_K);
	AMP_GV_CASE(AMP_Q_IQ2_S);
	AMP_GV_CASE(AMP_Q_IQ3_XXS);
	AMP_GV_CASE(AMP_Q_IQ3_S);
	AMP_GV_CASE(AMP_Q_IQ4_XS);
	}
#undef AMP_GV_CASE

	return (int)cudaErrorInvalidValue;
}

/* Pointer-table variant: ptrs (E) int64 device pointers to each expert's row start (resident VRAM or mapped host
 * memory). The same fp16 gemv kernel reads the quantized row from ptrs[e] + n * row bytes, so a mixed resident/cold
 * expert set needs no split or staging. */
extern "C" int
amp_gguf_gemv_pt(int qtype, const void *x, const void *ptrs, const void *ids, void *out, int M, int N, int K, int xdiv,
                 int slots, cudaStream_t stream)
{
	const amp_qinfo qi = amp_qtype_info(qtype);

	if (!ptrs || !ids || N < 1 || K < qi.block || (K % qi.block) || slots < 1 || xdiv < 1 ||
	    ((uintptr_t)x & 15) || ((uintptr_t)ptrs & 15))
		return (int)cudaErrorInvalidValue;

#define AMP_GV_PT_CASE(T) 	case T: return amp_gguf_gemv_launch<T>(true, x, NULL, ids, out, M, N, K, xdiv, slots, ptrs, stream)

	switch (qtype) {
	AMP_GV_PT_CASE(AMP_Q_Q4_0);
	AMP_GV_PT_CASE(AMP_Q_Q8_0);
	AMP_GV_PT_CASE(AMP_Q_IQ4_NL);
	AMP_GV_PT_CASE(AMP_Q_Q5_K);
	AMP_GV_PT_CASE(AMP_Q_IQ2_XXS);
	AMP_GV_PT_CASE(AMP_Q_IQ2_XS);
	AMP_GV_PT_CASE(AMP_Q_IQ1_M);
	AMP_GV_PT_CASE(AMP_Q_IQ1_S);
	AMP_GV_PT_CASE(AMP_Q_Q2_0);
	AMP_GV_PT_CASE(AMP_Q_Q4_K);
	AMP_GV_PT_CASE(AMP_Q_Q3_K);
	AMP_GV_PT_CASE(AMP_Q_Q6_K);
	AMP_GV_PT_CASE(AMP_Q_IQ2_S);
	AMP_GV_PT_CASE(AMP_Q_IQ3_XXS);
	AMP_GV_PT_CASE(AMP_Q_IQ3_S);
	AMP_GV_PT_CASE(AMP_Q_IQ4_XS);
	}
#undef AMP_GV_PT_CASE

	return (int)cudaErrorInvalidValue;
}

/* ---- integer dot products (dp4a) for the i-quants ---------------------------------------------------------------------
 *
 * The i-quants' weights are small integers (a grid byte, a sign, an integer sub-scale) times one f16 scale, so a row dot
 * product is cheap in int8 arithmetic: the activation row is quantized once per call to int8 with an fp32 scale per 32
 * values (ggml's q8_1: scale = amax / 127, q = round(x / scale)), the weights are decoded to signed bytes
 * ((grid ^ mask) - mask, four at a time) and multiplied with __dp4a. About 2 instructions per weight instead of 12;
 * the activation rounding (~0.3 % of a dot product) is what llama.cpp's kernels have as well. */

__global__ void amp_gguf_quant_x_kernel(const __half *__restrict__ x, int8_t *__restrict__ xq, float *__restrict__ xs,
                                        long nchunks)
{
	const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (c >= nchunks)
		return;

	const uint4 *p = (const uint4 *)(x + c * 32);
	uint4 v[4] = { p[0], p[1], p[2], p[3] };
	const __half *h = (const __half *)v;
	float f[32], amax = 0.f;

#pragma unroll
	for (int i = 0; i < 32; i++) {
		f[i] = __half2float(h[i]);
		amax = fmaxf(amax, fabsf(f[i]));
	}

	const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
	int8_t q[32];

#pragma unroll
	for (int i = 0; i < 32; i++)
		q[i] = (int8_t)__float2int_rn(f[i] * id);

	uint4 *o = (uint4 *)(xq + c * 32);

	o[0] = *(uint4 *)q;
	o[1] = *(uint4 *)(q + 16);
	xs[c] = d;
}

/* One item = 32 weights (an ib32 of a 256-weight block): w[8] the signed bytes of the four groups of 8 (two ints each),
 * mult[4] the integer multiplier of every group, returns the f16 scale folded with the constant of the format. */
template <int T> __device__ float amp_dec32_i8(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr, const uint64_t *sg);

template <> __device__ float amp_dec32_i8<AMP_Q_IQ2_XXS>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                         const uint64_t *sg)
{
	const uint8_t *q = b + 2 + 8 * j;
	const unsigned aux1 = amp_u16(q + 4) | (amp_u16(q + 6) << 16);
	const unsigned g4 = amp_u16(q) | (amp_u16(q + 2) << 16);
	const int ls = 2 * (int)(aux1 >> 28) + 1;

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const uint64_t grid = gr[(g4 >> (8 * l)) & 0xff], m = sg[(aux1 >> (7 * l)) & 127];
		const unsigned g0 = (unsigned)grid, g1 = (unsigned)(grid >> 32), m0 = (unsigned)m, m1 = (unsigned)(m >> 32);

		w[2 * l] = (int)__vsub4(g0 ^ m0, m0);
		w[2 * l + 1] = (int)__vsub4(g1 ^ m1, m1);
		mult[l] = ls;
	}

	return amp_h2f(b) * 0.125f;
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ2_XS>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                        const uint64_t *sg)
{
	const uint8_t *q = b + 2 + 8 * j;
	const unsigned sc = b[2 + 64 + j];

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const unsigned v = amp_u16(q + 2 * l);
		const uint64_t grid = gr[v & 511], m = sg[v >> 9];
		const unsigned g0 = (unsigned)grid, g1 = (unsigned)(grid >> 32), m0 = (unsigned)m, m1 = (unsigned)(m >> 32);

		w[2 * l] = (int)__vsub4(g0 ^ m0, m0);
		w[2 * l + 1] = (int)__vsub4(g1 ^ m1, m1);
		mult[l] = 2 * (int)(l < 2 ? sc & 0xf : sc >> 4) + 1;
	}

	return amp_h2f(b) * 0.125f;
}

/* IQ1_M: weights are dl * (grid + delta) with grid in {-1, 0, 1} (signed bytes, no sign mask) and delta = +-0.125 per group
 * of 8: the dot product is dl * (dp4a(grid, x) + delta * sum(x)); mult[l] = the integer part of dl, dlt[l] = delta. */
template <> __device__ float amp_dec32_i8<AMP_Q_IQ1_M>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                       const uint64_t *sg)
{
	/* (blocks are 56 bytes: always 8-byte aligned, so three wide loads replace a dozen byte loads) */
	const unsigned qs4 = *(const unsigned *)(b + 4 * j);                       /* qs[4j .. 4j + 3] */
	const unsigned qh2 = *(const unsigned short *)(b + 32 + 2 * j);            /* qh[2j], qh[2j + 1] */
	const unsigned long long sb = *(const unsigned long long *)(b + 48);       /* the four u16 scale words */
	const unsigned su = (((unsigned)sb >> 12) & 0xf) | (((unsigned)(sb >> 16) >> 8) & 0x00f0) | (((unsigned)(sb >> 32) >> 4) & 0x0f00) |
	                    ((unsigned)(sb >> 48) & 0xf000);
	const unsigned sc = (unsigned)(sb >> (16 * (j >> 1))) & 0xffff;

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const unsigned h = (qh2 >> (8 * (l >> 1))) & 0xff;
		const unsigned idx = ((qs4 >> (8 * l)) & 0xff) | (((l & 1) ? h << 4 : h << 8) & 0x700);
		const uint64_t grid = ((const uint64_t *)gr)[idx];

		w[2 * l] = (int)(unsigned)grid;
		w[2 * l + 1] = (int)(unsigned)(grid >> 32);
		mult[l] = 2 * (int)((sc >> (6 * (j & 1) + (l >> 1) * 3)) & 0x7) + 1;
		dlt[l] = (h & ((l & 1) ? 0x80 : 0x08)) ? -IQ1S_DELTA : IQ1S_DELTA;
	}

	return __half2float(__ushort_as_half((unsigned short)su));
}

/* Q8_0 (an item is a whole 34-byte block) and Q5_K (an item is 32 weights: a quantized value 0..31 per byte, scale and min
 * per item): the dot product of an item is xs * (a * sum(w * x) - b * sum(x)) with a = d * scale, b = dmin * min.
 * Returns a, *dlt = b. */
template <> __device__ float amp_dec32_i8<AMP_Q_Q8_0>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                      const uint64_t *sg)
{
#pragma unroll
	for (int k = 0; k < 8; k++)
		w[k] = (int)(amp_u16(b + 2 + 4 * k) | (amp_u16(b + 4 + 4 * k) << 16));

	dlt[0] = 0.f;

	return amp_h2f(b);
}

template <> __device__ float amp_dec32_i8<AMP_Q_Q5_K>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                      const uint64_t *sg)
{
	const int jj = j >> 1, half = j & 1;
	const uint4 q0 = ((const uint4 *)(b + 48 + 32 * jj))[0], q1 = ((const uint4 *)(b + 48 + 32 * jj))[1];
	const uint4 h0 = ((const uint4 *)(b + 16))[0], h1 = ((const uint4 *)(b + 16))[1];
	const unsigned qs[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
	const unsigned qh[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
	int sc, m;

	amp_scale_min_k4(j, b + 4, &sc, &m);

#pragma unroll
	for (int k = 0; k < 8; k++)
		w[k] = (int)((((half ? qs[k] >> 4 : qs[k]) & 0x0f0f0f0fu)) | (((qh[k] >> j) & 0x01010101u) << 4));

	dlt[0] = amp_h2f(b + 2) * (float)m;

	return amp_h2f(b) * (float)sc;
}

/* Q6_K: an item is 32 weights, a half h = j >> 2 and quarter r = j & 3 of the block: the low nibbles of ql[64 h + 32 (r & 1) ..] (high nibbles
 * for r >= 2) plus two bits of qh[32 h ..] at 2 r, minus 32; two int8 scales of 16 weights each (groups 0, 1 and 2, 3). */
template <> __device__ float amp_dec32_i8<AMP_Q_Q6_K>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                      const uint64_t *sg)
{
	const int h = j >> 2, r = j & 3;
	const uint8_t *ql = b + 64 * h + 32 * (r & 1), *qh = b + 128 + 32 * h;

#pragma unroll
	for (int k = 0; k < 8; k++) {
		const unsigned lo = amp_u16(ql + 4 * k) | (amp_u16(ql + 4 * k + 2) << 16);
		const unsigned hi = amp_u16(qh + 4 * k) | (amp_u16(qh + 4 * k + 2) << 16);
		const unsigned q = ((r < 2 ? lo : lo >> 4) & 0x0f0f0f0fu) | (((hi >> (2 * r)) & 0x03030303u) << 4);

		w[k] = (int)__vsub4(q, 0x20202020u);
	}

	mult[0] = mult[1] = (int)(int8_t)b[192 + 2 * j];
	mult[2] = mult[3] = (int)(int8_t)b[192 + 2 * j + 1];

	return amp_h2f(b + 208);
}

/* kvalues_iq4nl[nibble] for the eight nibbles of q4, as two ints: .x the low nibbles of its four bytes, .y the high nibbles (byte_perm tables of 4 entries). */
static __device__ inline int2 amp_iq4_lut(int q4)
{
	const unsigned W0 = 0xBFAD9881u, W1 = 0xF6EADDCFu, W2 = 0x26190D01u, W3 = 0x71594535u;        /* kvalues_iq4nl, little-endian */
	const unsigned sel = 0x32103210u | (((unsigned)q4 & 0x88888888u) >> 1);
	unsigned tmp[2];

#pragma unroll
	for (int i = 0; i < 2; i++) {
		const unsigned lo = __byte_perm(W0, W1, (unsigned)q4 >> (16 * i)), hi = __byte_perm(W2, W3, (unsigned)q4 >> (16 * i));

		tmp[i] = __byte_perm(lo, hi, sel >> (16 * i));
	}

	return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

/* Q4_K: Q5_K without the fifth bit (the nibbles are at b + 16). IQ4_XS: an item is a 32-weight sub-block with one 6-bit scale; the weights are
 * the table values of the nibbles of qs[16 j ..] (low nibbles: weights 0 .. 15, high: 16 .. 31). IQ4_NL: the same for a whole 18-byte block. */
template <> __device__ float amp_dec32_i8<AMP_Q_Q4_K>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                      const uint64_t *sg)
{
	const int jj = j >> 1, half = j & 1;
	const uint4 q0 = ((const uint4 *)(b + 16 + 32 * jj))[0], q1 = ((const uint4 *)(b + 16 + 32 * jj))[1];
	const unsigned qs[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
	int sc, m;

	amp_scale_min_k4(j, b + 4, &sc, &m);

#pragma unroll
	for (int k = 0; k < 8; k++)
		w[k] = (int)((half ? qs[k] >> 4 : qs[k]) & 0x0f0f0f0fu);

	dlt[0] = amp_h2f(b + 2) * (float)m;

	return amp_h2f(b) * (float)sc;
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ4_XS>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                        const uint64_t *sg)
{
	const unsigned sh = amp_u16(b + 2);
	const int ls = (int)(((b[4 + (j >> 1)] >> (4 * (j & 1))) & 0xf) | (((sh >> (2 * j)) & 3) << 4)) - 32;
	const uint2 a = ((const uint2 *)(b + 8 + 16 * j))[0], c = ((const uint2 *)(b + 8 + 16 * j))[1];
	const int q4[4] = { (int)a.x, (int)a.y, (int)c.x, (int)c.y };

#pragma unroll
	for (int k = 0; k < 4; k++) {
		const int2 v = amp_iq4_lut(q4[k]);

		w[k] = v.x;
		w[4 + k] = v.y;
		mult[k] = ls;
	}

	return amp_h2f(b);
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ4_NL>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                        const uint64_t *sg)
{
#pragma unroll
	for (int k = 0; k < 4; k++) {
		const int2 v = amp_iq4_lut((int)(amp_u16(b + 2 + 4 * k) | (amp_u16(b + 4 + 4 * k) << 16)));

		w[k] = v.x;
		w[4 + k] = v.y;
		mult[k] = 1;
	}

	return amp_h2f(b);
}

/* The sign bits of four weights (a nibble) as the byte masks of the (g ^ m) - m negation. */
static __device__ inline unsigned amp_expand4(unsigned n)
{
	return ((n * 0x00204081u) & 0x01010101u) * 0xffu;
}

/* Q2_0: 64 weights in 18 bytes, an item is half of it (8 bytes of four 2-bit values each, minus 1). IQ3_XXS / IQ3_S: the grid entries are
 * 4 weights (uint32, two per group of 8, from the shared copy of the table), IQ2_S: 8 weights (uint64). The signs of IQ3_XXS are 7-bit indices
 * (the byte masks of s_signs), those of IQ3_S / IQ2_S raw bits. */
template <> __device__ float amp_dec32_i8<AMP_Q_Q2_0>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                      const uint64_t *sg)
{
#pragma unroll
	for (int k = 0; k < 4; k++) {
		const unsigned v = amp_u16(b + 2 + 8 * j + 2 * k);
		const unsigned b0 = v & 0xff, b1 = v >> 8;

		w[2 * k] = (int)__vsub4((b0 & 0x03u) | ((b0 & 0x0cu) << 6) | ((b0 & 0x30u) << 12) | ((b0 & 0xc0u) << 18), 0x01010101u);
		w[2 * k + 1] = (int)__vsub4((b1 & 0x03u) | ((b1 & 0x0cu) << 6) | ((b1 & 0x30u) << 12) | ((b1 & 0xc0u) << 18), 0x01010101u);
		mult[k] = 1;
	}

	return amp_h2f(b);
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ3_XXS>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                         const uint64_t *sg)
{
	const unsigned *g32 = (const unsigned *)gr;
	const uint8_t *q = b + 2 + 8 * j;
	const unsigned aux = amp_u16(b + 66 + 4 * j) | (amp_u16(b + 68 + 4 * j) << 16);
	const int ls = 2 * (int)(aux >> 28) + 1;

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const uint64_t m = sg[(aux >> (7 * l)) & 127];
		const unsigned m0 = (unsigned)m, m1 = (unsigned)(m >> 32);

		w[2 * l] = (int)__vsub4(g32[q[2 * l]] ^ m0, m0);
		w[2 * l + 1] = (int)__vsub4(g32[q[2 * l + 1]] ^ m1, m1);
		mult[l] = ls;
	}

	return amp_h2f(b) * 0.25f;
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ3_S>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                       const uint64_t *sg)
{
	const unsigned *g32 = (const unsigned *)gr;
	const int p = j >> 1, half = j & 1;
	const int ls = 2 * (int)((b[106 + p] >> (4 * half)) & 0xf) + 1;
	const unsigned qhb = b[66 + 2 * p + half];
	const uint8_t *qs = b + 2 + p * 16 + half * 8;

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const unsigned sgn = b[74 + p * 8 + half * 4 + l];
		const unsigned m0 = amp_expand4(sgn & 0xf), m1 = amp_expand4(sgn >> 4);

		w[2 * l] = (int)__vsub4(g32[qs[2 * l] | ((qhb << (8 - 2 * l)) & 256)] ^ m0, m0);
		w[2 * l + 1] = (int)__vsub4(g32[qs[2 * l + 1] | ((qhb << (7 - 2 * l)) & 256)] ^ m1, m1);
		mult[l] = ls;
	}

	return amp_h2f(b);
}

template <> __device__ float amp_dec32_i8<AMP_Q_IQ2_S>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                       const uint64_t *sg)
{
	const unsigned sc = b[74 + j], qhb = b[66 + j];

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const uint64_t grid = gr[b[2 + 4 * j + l] | ((qhb << (8 - 2 * l)) & 0x300)];
		const unsigned sgn = b[34 + 4 * j + l];
		const unsigned m0 = amp_expand4(sgn & 0xf), m1 = amp_expand4(sgn >> 4);

		w[2 * l] = (int)__vsub4((unsigned)grid ^ m0, m0);
		w[2 * l + 1] = (int)__vsub4((unsigned)(grid >> 32) ^ m1, m1);
		mult[l] = 2 * (int)(l < 2 ? sc & 0xf : sc >> 4) + 1;
	}

	return amp_h2f(b) * 0.125f;
}

/* IQ1_S: one scale (3 bits) and one offset sign per 32 weights; the groups of 8 take 3 high bits of their grid index from qh. */
template <> __device__ float amp_dec32_i8<AMP_Q_IQ1_S>(const uint8_t *b, int j, int *w, int *mult, float *dlt, const uint64_t *gr,
                                                       const uint64_t *sg)
{
	const unsigned qh = amp_u16(b + 34 + 2 * j);
	const int m = 2 * (int)((qh >> 12) & 7) + 1;
	const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

#pragma unroll
	for (int l = 0; l < 4; l++) {
		const uint64_t grid = ((const uint64_t *)gr)[b[2 + 4 * j + l] | (((qh >> (3 * l)) & 7) << 8)];

		w[2 * l] = (int)(unsigned)grid;
		w[2 * l + 1] = (int)(unsigned)(grid >> 32);
		mult[l] = m;
		dlt[l] = delta;
	}

	return amp_h2f(b);
}

/* The picks of a verify step grouped by expert: ids (S) int64 -> grp (S, 9) int32, row g = {expert, up to 8 slots that picked it (-1 pads)}, the groups first
 * in order of their first slot, the unused rows {-1, -1 ...}. A pick < 0 is a group of its own (so the kernels still write its zero row). */
__global__ void amp_moe_group_kernel(const long long *__restrict__ ids, int *__restrict__ grp, int S)
{
	__shared__ int lead[64];
	const int i = threadIdx.x;
	const long long e = (i < S) ? ids[i] : -1;
	bool leader = i < S;

	for (int j = 0; j < i && leader; j++)
		if (e >= 0 && ids[j] == e)
			leader = false;

	lead[i] = leader ? 1 : 0;
	__syncthreads();

	int g = 0;

	for (int j = 0; j < i; j++)
		g += lead[j];

	/* every group row starts empty */
	for (int k = i; k < S * 9; k += blockDim.x)
		grp[k] = -1;

	__syncthreads();

	if (leader) {
		int *row = grp + (size_t)g * 9;
		int n = 0;

		row[0] = (e >= 0) ? (int)e : -1;

		for (int j = i; j < S && n < 8; j++)
			if (j == i || (e >= 0 && ids[j] == e))
				row[1 + n++] = j;
	}
}

extern "C" int
amp_moe_group(const void *ids, void *grp, int S, cudaStream_t stream)
{
	if (S < 1 || S > 64)
		return (int)cudaErrorInvalidValue;

	amp_moe_group_kernel<<<1, 64, 0, stream>>>((const long long *)ids, (int *)grp, S);

	return (int)cudaGetLastError();
}

/* rows per warp a block works through (the types with big lookup tables fill them once per block: see the kernel) */
__host__ __device__ constexpr int amp_gemv_iter(int T)
{
	return (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S || T == AMP_Q_IQ2_S) ? 4 : (T == AMP_Q_IQ2_XS || T == AMP_Q_IQ3_S) ? 2 : 1;
}

/* QX (one activation row per block: the experts, or a dense product of one row): `xq` is the fp16 activation row(s) (M, G * K) instead of the
 * quantized ones; the block quantizes the row (and column slice) it reads into shared memory first, with the arithmetic of amp_gguf_quant_x_kernel
 * (bit-identical), which saves the launch of that kernel and its gap.
 * GRP (experts, MR >= 2 rows, RPW 1): `ids` is the (S, 9) int32 table of amp_moe_group instead: row y = {expert, slot x 8} -- the picks of a verify step grouped by
 * expert, so an expert's rows are read once for all the slots that chose it (each slot's activation row is slot / xdiv, its output row the slot).
 * MR: the number of activation rows the instance handles (1 .. 5, or 8; the experts always 1): a one-token step carries no loops, predicates or
 * registers for the other rows. RPW: output rows per warp (1 or 2): with 2 the activation loads are shared and the weight loads of both rows are in
 * flight together (the expert kernels are latency-bound: a warp does only a few items per row). */
template <int T, bool MOE, bool GLU, int MR, int RPW, bool GRP = false, bool QX = false>
__global__ void amp_gguf_gemv_i8_kernel(const int8_t *__restrict__ xq, const float *__restrict__ xs,
                                        const uint8_t *__restrict__ w, const uint8_t *__restrict__ w_up,
                                        const long long *__restrict__ ids, const float *__restrict__ scale,
                                        void *__restrict__ out, int M, int N, int K, int xdiv, int G, float lim, int e0, int ne,
                                        const long long *__restrict__ ptrs, const long long *__restrict__ ptrs_up)
{
	constexpr int bytes = T == AMP_Q_Q8_0 ? 34 : T == AMP_Q_Q5_K ? 176 : T == AMP_Q_IQ2_XXS ? 66 : T == AMP_Q_IQ2_XS ? 74 : T == AMP_Q_IQ1_S ? 50 : T == AMP_Q_Q6_K ? 210 : T == AMP_Q_Q4_K ? 144 : T == AMP_Q_IQ4_XS ? 136 : T == AMP_Q_IQ4_NL ? 18 : T == AMP_Q_Q2_0 ? 18 : T == AMP_Q_IQ3_XXS ? 98 : T == AMP_Q_IQ3_S ? 110 : T == AMP_Q_IQ2_S ? 82 : 56;
	constexpr int IPB = (T == AMP_Q_Q8_0 || T == AMP_Q_IQ4_NL) ? 1 : T == AMP_Q_Q2_0 ? 2 : 8;                    /* items of 32 weights per block */
	constexpr bool AFFINE = T == AMP_Q_Q8_0 || T == AMP_Q_Q5_K || T == AMP_Q_Q4_K;
	constexpr int ITER = amp_gemv_iter(T);
	constexpr bool NOLUT = AFFINE || T == AMP_Q_Q6_K || T == AMP_Q_IQ4_XS || T == AMP_Q_IQ4_NL || T == AMP_Q_Q2_0;   /* (no lookup tables) */
	constexpr int NGRID = NOLUT ? 0 : T == AMP_Q_IQ2_XXS ? 256 : T == AMP_Q_IQ2_XS ? 512 : T == AMP_Q_IQ3_XXS ? 128 : T == AMP_Q_IQ3_S ? 256 : T == AMP_Q_IQ2_S ? 1024 : 2048;
	__shared__ uint64_t s_grid[NGRID ? NGRID : 1];
	__shared__ uint64_t s_signs[NOLUT ? 1 : 128];

	for (int i = threadIdx.x; i < NGRID; i += blockDim.x)
		s_grid[i] = (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? iq1s_grid[i] : (T == AMP_Q_IQ2_XS) ? iq2xs_grid[i] : (T == AMP_Q_IQ2_S) ? iq2s_grid[i] :
			(T == AMP_Q_IQ3_XXS) ? ((uint64_t)iq3xxs_grid[2 * i] | ((uint64_t)iq3xxs_grid[2 * i + 1] << 32)) :
			(T == AMP_Q_IQ3_S) ? ((uint64_t)iq3s_grid[2 * i] | ((uint64_t)iq3s_grid[2 * i + 1] << 32)) : iq2xxs_grid[i];

	for (int i = threadIdx.x; !NOLUT && i < 128; i += blockDim.x) {
		uint64_t m = 0;

		for (int b = 0; b < 8; b++)
			if (ksigns_iq2xs[i] & kmask_iq2xs[b])
				m |= (uint64_t)0xff << (8 * b);

		s_signs[i] = m;
	}

	const int8_t *xqe = xq;               /* where the main loop reads the quantized activation (and its scales) */
	const float *xse = xs;
	size_t xbase = 0;                       /* QX: the element offset of this block's row slice in the fp16 activation */

	if (QX) {
		extern __shared__ __align__(16) unsigned char qx_smem[];
		int8_t *sq = (int8_t *)qx_smem;
		float *ss = (float *)(qx_smem + K);
		bool live = true;

		if (MOE) {
			const long long e = ids[blockIdx.y] - e0;

			live = !(e < 0 || e >= ne);
			xbase = (size_t)(blockIdx.y / xdiv) * K;
		} else {
			xbase = (size_t)blockIdx.y * K;                                  /* (dense: row 0, the columns of group blockIdx.y) */
		}

		if (live) {
			const __half *xh = (const __half *)xq + xbase;

			for (int c = threadIdx.x; c < K / 32; c += blockDim.x) {
				const uint4 *p = (const uint4 *)(xh + c * 32);
				uint4 v[4] = { p[0], p[1], p[2], p[3] };
				const __half *h = (const __half *)v;
				float f[32], amax = 0.f;

#pragma unroll
				for (int j = 0; j < 32; j++) {
					f[j] = __half2float(h[j]);
					amax = fmaxf(amax, fabsf(f[j]));
				}

				const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
				int8_t q[32];

#pragma unroll
				for (int j = 0; j < 32; j++)
					q[j] = (int8_t)__float2int_rn(f[j] * id);

				uint4 *o = (uint4 *)(sq + c * 32);

				o[0] = *(uint4 *)q;
				o[1] = *(uint4 *)(q + 16);
				ss[c] = d;
			}
		}

		xqe = sq;
		xse = ss;
	}

	__syncthreads();

	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	/* a block of the types with big lookup tables (IQ1_*: 16 KiB, IQ2_XS: 4 KiB, filled from global memory above) works through ITER rows per warp,
	 * so the fill is paid once per ITER * 8 * RPW output rows */
	for (int it = 0; it < ITER; it++) {
		const int n0 = ((blockIdx.x * ITER + it) * (blockDim.x >> 5) + warp) * RPW;     /* this warp's output rows n0 .. n0 + RPW - 1 */
		const int slot = blockIdx.y;
		const int grp = MOE ? 0 : blockIdx.y;        /* G groups: rows [grp * N, (grp + 1) * N) see x columns [grp * K, (grp + 1) * K) */

		if (n0 >= N)
			break;

		const size_t rb = (size_t)(K / (32 * IPB)) * bytes;
		const uint8_t *row[RPW], *row_up[RPW];
		size_t xrow = 0;
		size_t xr[GRP ? MR : 1];
		int os[GRP ? MR : 1];
		int rows = M;

		if (MOE) {
			long long e;

			if (GRP) {
				const int *g = (const int *)ids + (size_t)slot * 9;

				e = (long long)g[0] - e0;
				rows = 0;

	#pragma unroll
				for (int m = 0; m < MR; m++) {
					const int sl = g[1 + m];

					if (sl >= 0) {
						xr[m] = (size_t)(sl / xdiv);
						os[m] = sl;
						rows = m + 1;
					}
				}

				if (rows == 0)
					continue;
			} else {
				e = ids[slot] - e0;               /* (e0: the first expert of a shard; a pick outside [e0, e0 + ne) is none) */
			}

			if (e < 0 || e >= ne) {
				if (lane == 0) {
					for (int m = 0; m < (GRP ? rows : 1); m++) {
						const size_t o = GRP ? (size_t)os[m] : (size_t)slot;

	#pragma unroll
						for (int r = 0; r < RPW; r++) {
							if (GLU)
								((__half *)out)[o * N + n0 + r] = __float2half(0.f);
							else
								((float *)out)[o * N + n0 + r] = 0.f;
						}
					}
				}

				continue;
			}

	#pragma unroll
			for (int r = 0; r < RPW; r++) {
				row[r] = ptrs ? (const uint8_t *)ptrs[e] + (size_t)(n0 + r) * rb
					: w + ((size_t)e * N + n0 + r) * rb;
				row_up[r] = GLU ? (ptrs_up ? (const uint8_t *)ptrs_up[e] + (size_t)(n0 + r) * rb
					: w_up + ((size_t)e * N + n0 + r) * rb) : NULL;
			}

			if (!GRP) {
				xrow = (size_t)(slot / xdiv);
				rows = 1;
			}
		} else {
	#pragma unroll
			for (int r = 0; r < RPW; r++) {
				row[r] = w + ((size_t)grp * N + n0 + r) * rb;
				row_up[r] = GLU ? (ptrs_up ? (const uint8_t *)ptrs_up[grp] + (size_t)(n0 + r) * rb
					: w_up + ((size_t)grp * N + n0 + r) * rb) : NULL;
			}
		}

		const int nchunks = K / 32;
		float acc[RPW][MR], acc2[RPW][GLU ? MR : 1];

	#pragma unroll
		for (int r = 0; r < RPW; r++) {
	#pragma unroll
			for (int m = 0; m < MR; m++) {
				acc[r][m] = 0.f;

				if (GLU)
					acc2[r][m] = 0.f;
			}
		}

		/* the dot product of one item (32 weights) of a row with the item's activations (xv, xsc: of every activation row m) */
		auto item_dot = [&](const uint8_t *rp, int i, float *accv, const int (*xv)[8], const float *xsc) {
			int wv[8], mult[4];
			float dlt[4];
			const float ds = amp_dec32_i8<T>(rp + (size_t)(i / IPB) * bytes, i % IPB, wv, mult, dlt, s_grid, s_signs);

	#pragma unroll
			for (int m = 0; m < MR; m++) {
				if (MR == 1 || m < rows) {
					if (AFFINE) {
						int ig = 0, sx = 0;

	#pragma unroll
						for (int k = 0; k < 8; k++)
							ig = __dp4a(wv[k], xv[m][k], ig);

						if (T != AMP_Q_Q8_0) {
	#pragma unroll
							for (int k = 0; k < 8; k++)
								sx = __dp4a(0x01010101, xv[m][k], sx);
						}

						accv[m] += xsc[m] * (ds * (float)ig - dlt[0] * (float)sx);
					} else {
						int s = 0;
						float sf = 0.f;

	#pragma unroll
						for (int l = 0; l < 4; l++) {
							const int ig = __dp4a(wv[2 * l], xv[m][2 * l], 0) + __dp4a(wv[2 * l + 1], xv[m][2 * l + 1], 0);

							if (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) {
								const int sx = __dp4a(0x01010101, xv[m][2 * l], 0) + __dp4a(0x01010101, xv[m][2 * l + 1], 0);

								sf += (float)mult[l] * ((float)ig + dlt[l] * (float)sx);
							} else {
								s += mult[l] * ig;
							}
						}

						accv[m] += (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S ? sf : (float)s) * ds * xsc[m];
					}
				}
			}
		};

		for (int i = lane; i < nchunks; i += 32) {
			int xv[MR][8];
			float xsc[MR];

	#pragma unroll
			for (int m = 0; m < MR; m++) {
				if (MR == 1 || m < rows) {
					const size_t r = GRP ? xr[m] : MOE ? xrow : (size_t)m;
					const uint4 *xp = QX ? (const uint4 *)(xqe + (size_t)i * 32) : (const uint4 *)(xq + r * K * G + (size_t)grp * K + (size_t)i * 32);
					const uint4 x0 = xp[0], x1 = xp[1];

					xv[m][0] = (int)x0.x; xv[m][1] = (int)x0.y; xv[m][2] = (int)x0.z; xv[m][3] = (int)x0.w;
					xv[m][4] = (int)x1.x; xv[m][5] = (int)x1.y; xv[m][6] = (int)x1.z; xv[m][7] = (int)x1.w;
					xsc[m] = QX ? xse[i] : xs[r * nchunks * G + grp * nchunks + i];
				}
			}

	#pragma unroll
			for (int r = 0; r < RPW; r++) {
				item_dot(row[r], i, acc[r], xv, xsc);

				if (GLU)
					item_dot(row_up[r], i, acc2[r], xv, xsc);
			}
		}

	#pragma unroll
		for (int r = 0; r < RPW; r++) {
	#pragma unroll
			for (int m = 0; m < MR; m++) {
				float s = acc[r][m], u = GLU ? acc2[r][m] : 0.f;

	#pragma unroll
				for (int o = 16; o > 0; o >>= 1) {
					s += __shfl_xor_sync(0xffffffffu, s, o);

					if (GLU)
						u += __shfl_xor_sync(0xffffffffu, u, o);
				}

				if (lane == 0 && (MR == 1 || m < rows)) {
					const int n = n0 + r;

					if (GLU) {
						/* clamped SwiGLU: silu(min(gate, lim)) * clamp(up, -lim, lim) (lim == 0: no clamp) */
						if (lim > 0.f) {
							s = fminf(s, lim);
							u = fminf(fmaxf(u, -lim), lim);
						}

						const float h = s / (1.f + __expf(-s)) * u;

						((__half *)out)[MOE ? (size_t)(GRP ? os[m] : slot) * N + n : (size_t)m * N * G + (size_t)grp * N + n] = __float2half(h);
					} else if (MOE) {
						((float *)out)[(size_t)(GRP ? os[m] : slot) * N + n] = scale ? s * scale[GRP ? os[m] : slot] : s;
					} else {
						((__half *)out)[(size_t)m * N * G + (size_t)grp * N + n] = __float2half(s);
					}
				}
			}
		}

	}
}

/* Quantize x (R, K) fp16 for the integer kernels: xq (R, K) int8, xs (R, K / 32) fp32. */
extern "C" int
amp_gguf_quant_x(const void *x, void *xq, void *xs, int R, int K, cudaStream_t stream)
{
	if (R < 1 || K < 32 || (K % 32) || ((uintptr_t)x & 15))
		return (int)cudaErrorInvalidValue;

	const long nchunks = (long)R * (K / 32);

	amp_gguf_quant_x_kernel<<<(unsigned)((nchunks + 127) / 128), 128, 0, stream>>>((const __half *)x, (int8_t *)xq,
		(float *)xs, nchunks);

	return (int)cudaGetLastError();
}

/* G > 1 (dense only): G groups of N rows, group g seeing the columns [g * K, (g + 1) * K) of the (M, G * K) activation rows,
 * y (M, G * N). w_up != NULL: the fused gate / up product, w the gate rows and w_up the up rows (same type and shape), the
 * output the clamped SwiGLU of the two in fp16 ((M, N) dense, (slots, N) experts; lim: swiglu_limit, 0 = none). scale != NULL
 * (experts): out[slot] is multiplied by scale[slot]. The integer-dot variant of amp_gguf_gemv for the types that have one
 * (all but Q4_0 and Q3_K): xq / xs from amp_gguf_quant_x.
 * Returns cudaErrorNotSupported for the others (the caller then uses the fp16 kernel). */
extern "C" int
amp_gguf_gemv_i8(int qtype, const void *xq, const void *xs, const void *w, const void *w_up, const void *ids, const void *scale,
                 void *out, int M, int N, int K, int xdiv, int slots, int G, float lim, int e0, int ne, int mr, int qx,
                 const void *ptrs, const void *ptrs_up, cudaStream_t stream)
{
	const bool moe = ids != NULL;

	if (qtype != AMP_Q_IQ2_XXS && qtype != AMP_Q_IQ2_XS && qtype != AMP_Q_IQ1_M && qtype != AMP_Q_IQ1_S && qtype != AMP_Q_Q5_K && qtype != AMP_Q_Q6_K && qtype != AMP_Q_Q4_K && qtype != AMP_Q_IQ4_XS && qtype != AMP_Q_IQ4_NL && qtype != AMP_Q_Q2_0 && qtype != AMP_Q_IQ3_XXS && qtype != AMP_Q_IQ3_S && qtype != AMP_Q_IQ2_S && qtype != AMP_Q_Q8_0)
		return (int)cudaErrorNotSupported;

	if (N < 1 || K < amp_qtype_info(qtype).block || (K % amp_qtype_info(qtype).block) || (!moe && (M < 1 || M > 8)) ||
	    (moe && (slots < 1 || xdiv < 1)) || (w_up && G != 1) || (ptrs_up && G != 1) || (!w && !ptrs))
		return (int)cudaErrorInvalidValue;

	const dim3 blk(256);
	static const bool rpw1 = getenv("AMP_GGUF_RPW1") != NULL;
	const bool rpw2 = moe && !rpw1 && (N % 16) == 0;                 /* two output rows per warp for the experts (AMP_GGUF_RPW1=1: one) */
	const bool rpw2d = !moe && !rpw1 && (N % 16) == 0;               /* ... and for a single-row dense product */
	const bool glu = w_up || ptrs_up;                                 /* fused gate/up: raw stack or pointer table */

#define AMP_I8_K(T, MOE_, GLU_, MR_, RPW_) \
	amp_gguf_gemv_i8_kernel<T, MOE_, GLU_, MR_, RPW_><<<dim3((N + 8 * RPW_ * amp_gemv_iter(T) - 1) / (8 * RPW_ * amp_gemv_iter(T)), moe ? slots : G), blk, 0, stream>>>( \
		(const int8_t *)xq, (const float *)xs, (const uint8_t *)w, (const uint8_t *)w_up, (const long long *)ids, (const float *)scale, out, M, N, K, \
		xdiv, G, lim, e0, ne, (const long long *)ptrs, (const long long *)ptrs_up)
#define AMP_I8_KQ(T, MOE_, GLU_, RPW_) \
	amp_gguf_gemv_i8_kernel<T, MOE_, GLU_, 1, RPW_, false, true><<<dim3((N + 8 * RPW_ * amp_gemv_iter(T) - 1) / (8 * RPW_ * amp_gemv_iter(T)), moe ? slots : G), \
		blk, (size_t)K + (size_t)(K / 32) * 4, stream>>>((const int8_t *)xq, (const float *)xs, (const uint8_t *)w, (const uint8_t *)w_up, \
		(const long long *)ids, (const float *)scale, out, M, N, K, xdiv, G, lim, e0, ne, (const long long *)ptrs, (const long long *)ptrs_up)
#define AMP_I8_KG(T, GLU_, MR_) \
	amp_gguf_gemv_i8_kernel<T, true, GLU_, MR_, 1, true><<<dim3((N + 8 * amp_gemv_iter(T) - 1) / (8 * amp_gemv_iter(T)), slots), blk, 0, stream>>>( \
		(const int8_t *)xq, (const float *)xs, (const uint8_t *)w, (const uint8_t *)w_up, (const long long *)ids, (const float *)scale, out, M, N, K, \
		xdiv, G, lim, e0, ne, (const long long *)ptrs, (const long long *)ptrs_up)
#define AMP_I8_GRP(T, GLU_) \
	switch (mr) { \
	case 2: AMP_I8_KG(T, GLU_, 2); break; \
	case 3: AMP_I8_KG(T, GLU_, 3); break; \
	case 4: AMP_I8_KG(T, GLU_, 4); break; \
	case 5: AMP_I8_KG(T, GLU_, 5); break; \
	default: AMP_I8_KG(T, GLU_, 8); break; \
	}
#define AMP_I8_ROWS(T, GLU_) \
	switch (M) { \
	case 1: if (rpw2d) AMP_I8_K(T, false, GLU_, 1, 2); else AMP_I8_K(T, false, GLU_, 1, 1); break; \
	case 2: AMP_I8_K(T, false, GLU_, 2, 1); break; \
	case 3: AMP_I8_K(T, false, GLU_, 3, 1); break; \
	case 4: AMP_I8_K(T, false, GLU_, 4, 1); break; \
	case 5: AMP_I8_K(T, false, GLU_, 5, 1); break; \
	default: AMP_I8_K(T, false, GLU_, 8, 1); break; \
	}
#define AMP_I8_LAUNCH(T) \
	do { \
		if (qx && moe && glu) { \
			if (rpw2) AMP_I8_KQ(T, true, true, 2); else AMP_I8_KQ(T, true, true, 1); \
		} else if (qx && moe) { \
			if (rpw2) AMP_I8_KQ(T, true, false, 2); else AMP_I8_KQ(T, true, false, 1); \
		} else if (qx && glu) { \
			if (rpw2d) AMP_I8_KQ(T, false, true, 2); else AMP_I8_KQ(T, false, true, 1); \
		} else if (qx) { \
			if (rpw2d) AMP_I8_KQ(T, false, false, 2); else AMP_I8_KQ(T, false, false, 1); \
		} else if (moe && mr > 1 && glu) { \
			AMP_I8_GRP(T, true) \
		} else if (moe && mr > 1) { \
			AMP_I8_GRP(T, false) \
		} else if (moe && glu) { \
			if (rpw2) AMP_I8_K(T, true, true, 1, 2); else AMP_I8_K(T, true, true, 1, 1); \
		} else if (moe) { \
			if (rpw2) AMP_I8_K(T, true, false, 1, 2); else AMP_I8_K(T, true, false, 1, 1); \
		} else if (glu) { \
			AMP_I8_ROWS(T, true) \
		} else { \
			AMP_I8_ROWS(T, false) \
		} \
	} while (0)

	if (qtype == AMP_Q_IQ2_XXS)
		AMP_I8_LAUNCH(AMP_Q_IQ2_XXS);
	else if (qtype == AMP_Q_IQ2_XS)
		AMP_I8_LAUNCH(AMP_Q_IQ2_XS);
	else if (qtype == AMP_Q_IQ1_M)
		AMP_I8_LAUNCH(AMP_Q_IQ1_M);
	else if (qtype == AMP_Q_IQ1_S)
		AMP_I8_LAUNCH(AMP_Q_IQ1_S);
	else if (qtype == AMP_Q_Q5_K)
		AMP_I8_LAUNCH(AMP_Q_Q5_K);
	else if (qtype == AMP_Q_Q6_K)
		AMP_I8_LAUNCH(AMP_Q_Q6_K);
	else if (qtype == AMP_Q_Q4_K)
		AMP_I8_LAUNCH(AMP_Q_Q4_K);
	else if (qtype == AMP_Q_IQ4_XS)
		AMP_I8_LAUNCH(AMP_Q_IQ4_XS);
	else if (qtype == AMP_Q_IQ4_NL)
		AMP_I8_LAUNCH(AMP_Q_IQ4_NL);
	else if (qtype == AMP_Q_Q2_0)
		AMP_I8_LAUNCH(AMP_Q_Q2_0);
	else if (qtype == AMP_Q_IQ3_XXS)
		AMP_I8_LAUNCH(AMP_Q_IQ3_XXS);
	else if (qtype == AMP_Q_IQ3_S)
		AMP_I8_LAUNCH(AMP_Q_IQ3_S);
	else if (qtype == AMP_Q_IQ2_S)
		AMP_I8_LAUNCH(AMP_Q_IQ2_S);
	else
		AMP_I8_LAUNCH(AMP_Q_Q8_0);
#undef AMP_I8_K
#undef AMP_I8_LAUNCH

	return (int)cudaGetLastError();
}

/* ---- grouped GEMM for the routed experts of a prefill chunk (tensor cores) ----------------------------------------------
 *
 * The tokens of a chunk are sorted by expert (torch.moe_prep: tok[], wts[] per sorted slot, start[e], count[e]). One thread block
 * computes a tile of BM tokens x BN outputs of one expert: the activation rows are gathered into shared memory, the quantized weight
 * rows are dequantized straight into shared memory (the group functions above: fp16, as the dequantization kernel gives), and
 * WMMA (fp16 in, fp32 accumulate) multiplies them. Stage A (K = hidden): gate and up of the tile -> the clamped SwiGLU -> h[slot]
 * (fp16); stage B (K = the expert's inner size): down of h -> out[tok] += routing weight * result (atomic fp32 adds). The weights are
 * read from their quantized form once per token tile instead of being dequantized to a global fp16 copy first. */

#include <mma.h>

#define PF_BM 32
#define PF_BN 64
#define PF_BK 64
#define PF_LD (PF_BK + 8)             /* smem row pitch (halves): 16-byte multiples, staggered banks */

template <int T, bool STAGE_A>
__global__ void __launch_bounds__(128)
amp_gguf_moe_pf_kernel(const __half *__restrict__ xin, const long long *__restrict__ tok, const __half *__restrict__ wts,
                       const long long *__restrict__ count, const long long *__restrict__ start,
                       const uint8_t *__restrict__ w0, const uint8_t *__restrict__ w1, __half *__restrict__ hbuf,
                       float *__restrict__ out, int N, int K, float lim, bool det)
{
	using namespace nvcuda;

	constexpr int bytes = T == AMP_Q_Q8_0 ? 34 : T == AMP_Q_Q5_K ? 176 : T == AMP_Q_IQ2_XXS ? 66 : T == AMP_Q_IQ2_XS ? 74 : T == AMP_Q_IQ1_S ? 50 : 56;
	constexpr int blk = T == AMP_Q_Q8_0 ? 32 : 256;
	constexpr bool IQ = T == AMP_Q_IQ2_XXS || T == AMP_Q_IQ2_XS || T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S;
	constexpr int NGRID = !IQ ? 0 : T == AMP_Q_IQ2_XXS ? 256 : T == AMP_Q_IQ2_XS ? 512 : 2048;

	__shared__ __align__(16) __half s_x[PF_BM * PF_LD];
	__shared__ __align__(16) __half s_w0[PF_BN * PF_LD];
	__shared__ __align__(16) __half s_w1[STAGE_A ? PF_BN * PF_LD : 8];
	__shared__ uint64_t s_grid[NGRID ? NGRID : 1];
	__shared__ uint8_t s_signs[IQ ? 128 : 1];

	const int e = blockIdx.y;
	const int cnt = (int)count[e];
	const int m0 = blockIdx.z * PF_BM;

	if (m0 >= cnt)
		return;

	const int n0 = blockIdx.x * PF_BN;
	const long long st = start[e];
	const int tid = threadIdx.x, warp = tid >> 5;
	const int wm = warp >> 1, wn = warp & 1;              /* 2 x 2 warps: each 16 rows x 32 columns */
	const size_t rb = (size_t)(K / blk) * bytes;
	const uint8_t *e0 = w0 + (size_t)e * N * rb, *e1 = STAGE_A ? w1 + (size_t)e * N * rb : NULL;

	if (IQ) {
		for (int i = tid; i < NGRID; i += 128)
			s_grid[i] = (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? iq1s_grid[i] : (T == AMP_Q_IQ2_XS) ? iq2xs_grid[i] : iq2xxs_grid[i];

		for (int i = tid; i < 128; i += 128)
			s_signs[i] = ksigns_iq2xs[i];

		__syncthreads();
	}

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0[2], c1[2];

#pragma unroll
	for (int j = 0; j < 2; j++) {
		wmma::fill_fragment(c0[j], 0.f);
		wmma::fill_fragment(c1[j], 0.f);
	}

	for (int k0 = 0; k0 < K; k0 += PF_BK) {
		/* activation tile: 32 rows x 64 halves, 16 bytes per thread-load */
		for (int i = tid; i < PF_BM * 8; i += 128) {
			const int r = i >> 3, c8 = i & 7;
			uint4 v = make_uint4(0, 0, 0, 0);

			if (m0 + r < cnt) {
				const size_t row = STAGE_A ? (size_t)tok[st + m0 + r] : (size_t)(st + m0 + r);

				v = *(const uint4 *)(xin + row * K + k0 + c8 * 8);
			}

			*(uint4 *)(s_x + r * PF_LD + c8 * 8) = v;
		}

		/* weight tiles: 64 rows x 8 groups of 8 weights, dequantized to fp16 */
		for (int i = tid; i < PF_BN * 8; i += 128) {
			const int r = i >> 3, g8 = i & 7;
			const size_t off = (size_t)(k0 / blk) * bytes;
			const int grp = (k0 % blk) / 8 + g8;
			float v[8];
			__half h[8];

			amp_dq_group<T>(e0 + (size_t)(n0 + r) * rb + off, grp, v, s_grid, (const uint8_t *)s_signs);
#pragma unroll
			for (int j = 0; j < 8; j++)
				h[j] = __float2half(v[j]);
			*(uint4 *)(s_w0 + r * PF_LD + g8 * 8) = *(uint4 *)h;

			if (STAGE_A) {
				amp_dq_group<T>(e1 + (size_t)(n0 + r) * rb + off, grp, v, s_grid, (const uint8_t *)s_signs);
#pragma unroll
				for (int j = 0; j < 8; j++)
					h[j] = __float2half(v[j]);
				*(uint4 *)(s_w1 + r * PF_LD + g8 * 8) = *(uint4 *)h;
			}
		}

		__syncthreads();

#pragma unroll
		for (int kk = 0; kk < PF_BK; kk += 16) {
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;

			wmma::load_matrix_sync(a, s_x + wm * 16 * PF_LD + kk, PF_LD);

#pragma unroll
			for (int j = 0; j < 2; j++) {
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;

				wmma::load_matrix_sync(b, s_w0 + (wn * 32 + j * 16) * PF_LD + kk, PF_LD);
				wmma::mma_sync(c0[j], a, b, c0[j]);

				if (STAGE_A) {
					wmma::load_matrix_sync(b, s_w1 + (wn * 32 + j * 16) * PF_LD + kk, PF_LD);
					wmma::mma_sync(c1[j], a, b, c1[j]);
				}
			}
		}

		__syncthreads();
	}

	/* epilogue through shared memory (the staging tiles are reused: 32 x 64 floats each) */
	float *sc0 = (float *)s_w0;                            /* 64 * 72 halves = 9216 bytes >= 32 * 64 * 4 = 8192 */
	float *sc1 = (float *)s_w1;

#pragma unroll
	for (int j = 0; j < 2; j++) {
		wmma::store_matrix_sync(sc0 + wm * 16 * PF_BN + wn * 32 + j * 16, c0[j], PF_BN, wmma::mem_row_major);

		if (STAGE_A)
			wmma::store_matrix_sync(sc1 + wm * 16 * PF_BN + wn * 32 + j * 16, c1[j], PF_BN, wmma::mem_row_major);
	}

	__syncthreads();

	for (int i = tid; i < PF_BM * PF_BN; i += 128) {
		const int r = i / PF_BN, c = i % PF_BN;

		if (m0 + r >= cnt)
			continue;

		if (STAGE_A) {
			float g = sc0[i], u = sc1[i];

			if (lim > 0.f) {
				g = fminf(g, lim);
				u = fminf(fmaxf(u, -lim), lim);
			}

			hbuf[(size_t)(st + m0 + r) * N + n0 + c] = __float2half(g / (1.f + __expf(-g)) * u);
		} else {
			/* det: every assignment owns the row of its sorted position (a fixed-order gather sums them: bit-reproducible); else an atomic add into the token row */
			if (det)
				out[(size_t)(st + m0 + r) * N + n0 + c] = __half2float(wts[st + m0 + r]) * sc0[i];
			else
				atomicAdd(out + (size_t)tok[st + m0 + r] * N + n0 + c, __half2float(wts[st + m0 + r]) * sc0[i]);
		}
	}
}

/* y (T, H) fp16 tokens; tok / wts / count / start from torch.moe_prep; w1 / w3 (E, I, row bytes) gate / up of type t13, w2 (E, H, row
 * bytes) down of type t2; out (T, H) fp32 (zeroed by the caller) receives sum_slots routing weight * down(swiglu(gate, up));
 * hbuf (A, I) fp16 scratch; max_rows: an upper bound of the tokens one expert gets. */
extern "C" int
amp_gguf_moe_prefill(int t13, int t2, const void *y, const void *tok, const void *wts, const void *count, const void *start, const void *w1,
                     const void *w3, const void *w2, void *hbuf, void *out, int E, int H, int I, int max_rows, float lim, int det,
                     cudaStream_t stream)
{
	if (E < 1 || H < 256 || I < 256 || (H % PF_BN) || (I % PF_BN) || (H % 256) || (I % 256) || max_rows < 1 || ((uintptr_t)y & 15) ||
	    ((uintptr_t)hbuf & 15))
		return (int)cudaErrorInvalidValue;

	const dim3 blk(128);
	const unsigned mt = (unsigned)((max_rows + PF_BM - 1) / PF_BM);

#define AMP_PF_CASE(T, STAGE) \
	case T: \
		amp_gguf_moe_pf_kernel<T, STAGE><<<STAGE ? dim3(I / PF_BN, E, mt) : dim3(H / PF_BN, E, mt), blk, 0, stream>>>( \
			(const __half *)(STAGE ? y : hbuf), (const long long *)tok, (const __half *)wts, (const long long *)count, \
			(const long long *)start, (const uint8_t *)(STAGE ? w1 : w2), (const uint8_t *)w3, (__half *)hbuf, (float *)out, \
			STAGE ? I : H, STAGE ? H : I, lim, det != 0); \
		break

	switch (t13) {
	AMP_PF_CASE(AMP_Q_IQ2_XXS, true);
	AMP_PF_CASE(AMP_Q_IQ2_XS, true);
	AMP_PF_CASE(AMP_Q_IQ1_M, true);
	AMP_PF_CASE(AMP_Q_IQ1_S, true);
	default: return (int)cudaErrorNotSupported;
	}

	switch (t2) {
	AMP_PF_CASE(AMP_Q_IQ2_XXS, false);
	AMP_PF_CASE(AMP_Q_IQ2_XS, false);
	AMP_PF_CASE(AMP_Q_IQ1_M, false);
	AMP_PF_CASE(AMP_Q_IQ1_S, false);
	default: return (int)cudaErrorNotSupported;
	}
#undef AMP_PF_CASE

	return (int)cudaGetLastError();
}

/* ---- the same on int8 tensor cores (mma.sync m16n8k16 s8) -------------------------------------------------------------------
 *
 * The weight rows are the M operand, the tokens the N operand. A thread decodes straight from the quantized bytes the 4 weights of
 * each of its two rows that the fragment layout gives it (a grid half + its sign mask, a few instructions) - no dequantization to
 * fp16 and no staging of weights in shared memory; the activation tile (int8 rows + an fp32 scale per 32 values, quantized once per
 * layer) is gathered into shared memory. Each mma covers 16 weights of the k axis, which is inside one scale group of the format
 * (IQ2_XS and IQ1_M change their sub-scale every 16), so the int32 result is converted and multiplied with the row's scale and the
 * token's scale right away. IQ1_M's +-1/8 offsets are folded into the weights as 8 * grid +- 1 (the scale carries the 1/8). */

#define MM_BN 64                       /* weight rows per block: 4 warps x 16 */
#define MM_BM 32                       /* tokens per block: 4 n8 tiles */
#define MM_XLD 272                     /* smem pitch of an activation row (256 bytes + 16, 16-byte aligned, staggered banks) */

__device__ inline void amp_mma_s8_16816(int c[4], unsigned a0, unsigned a1, unsigned b0)
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
	             : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
	             : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

/* the A word (4 weights as signed bytes) and the scale of one row for the k16 half `h` of item j of a block: tig selects the 4 weights */
template <int T> __device__ void amp_mma_dec(const uint8_t *b, int j, int h, int tig, const uint64_t *gr, const uint64_t *sg, unsigned *a, float *sc);

template <> __device__ void amp_mma_dec<AMP_Q_IQ2_XXS>(const uint8_t *b, int j, int h, int tig, const uint64_t *gr, const uint64_t *sg,
                                                      unsigned *a, float *sc)
{
	const uint8_t *q = b + 2 + 8 * j;
	const unsigned aux1 = amp_u16(q + 4) | (amp_u16(q + 6) << 16);
	const int l = 2 * h + (tig >> 1);
	const uint64_t grid = gr[q[l]], m = sg[(aux1 >> (7 * l)) & 127];
	const unsigned g = (tig & 1) ? (unsigned)(grid >> 32) : (unsigned)grid, mk = (tig & 1) ? (unsigned)(m >> 32) : (unsigned)m;

	*a = __vsub4(g ^ mk, mk);
	*sc = amp_h2f(b) * 0.125f * (float)(2 * (int)(aux1 >> 28) + 1);
}

template <> __device__ void amp_mma_dec<AMP_Q_IQ2_XS>(const uint8_t *b, int j, int h, int tig, const uint64_t *gr, const uint64_t *sg,
                                                     unsigned *a, float *sc)
{
	const uint8_t *q = b + 2 + 8 * j;
	const int l = 2 * h + (tig >> 1);
	const unsigned v = amp_u16(q + 2 * l);
	const uint64_t grid = gr[v & 511], m = sg[v >> 9];
	const unsigned g = (tig & 1) ? (unsigned)(grid >> 32) : (unsigned)grid, mk = (tig & 1) ? (unsigned)(m >> 32) : (unsigned)m;
	const unsigned s8 = b[2 + 64 + j];

	*a = __vsub4(g ^ mk, mk);
	*sc = amp_h2f(b) * 0.125f * (float)(2 * (int)(h ? s8 >> 4 : s8 & 0xf) + 1);
}

template <> __device__ void amp_mma_dec<AMP_Q_IQ1_M>(const uint8_t *b, int j, int h, int tig, const uint64_t *gr, const uint64_t *sg,
                                                    unsigned *a, float *sc)
{
	const int l = 2 * h + (tig >> 1);
	const unsigned hh = b[32 + 2 * j + (l >> 1)];
	const unsigned idx = b[4 * j + l] | (((l & 1) ? hh << 4 : hh << 8) & 0x700);
	const unsigned long long sb = *(const unsigned long long *)(b + 48);
	const unsigned su = (((unsigned)sb >> 12) & 0xf) | (((unsigned)(sb >> 16) >> 8) & 0x00f0) | (((unsigned)(sb >> 32) >> 4) & 0x0f00) |
	                    ((unsigned)(sb >> 48) & 0xf000);
	const unsigned scw = (unsigned)(sb >> (16 * (j >> 1))) & 0xffff;
	const uint64_t grid = ((const uint64_t *)gr)[idx];
	const unsigned g = (tig & 1) ? (unsigned)(grid >> 32) : (unsigned)grid;
	const unsigned g8 = (g << 3) & 0xF8F8F8F8u;                                   /* 8 * grid, bytewise (grid in -1..1) */
	const bool neg = hh & ((l & 1) ? 0x80 : 0x08);

	*a = neg ? __vsub4(g8, 0x01010101u) : __vadd4(g8, 0x01010101u);               /* 8 * (grid +- 1/8) */
	*sc = __half2float(__ushort_as_half((unsigned short)su)) * (float)(2 * (int)((scw >> (6 * (j & 1) + h * 3)) & 0x7) + 1) * 0.125f;
}

template <> __device__ void amp_mma_dec<AMP_Q_IQ1_S>(const uint8_t *b, int j, int h, int tig, const uint64_t *gr, const uint64_t *sg,
                                                    unsigned *a, float *sc)
{
	const unsigned qh = amp_u16(b + 34 + 2 * j);
	const int l = 2 * h + (tig >> 1);
	const uint64_t grid = ((const uint64_t *)gr)[b[2 + 4 * j + l] | (((qh >> (3 * l)) & 7) << 8)];
	const unsigned g = (tig & 1) ? (unsigned)(grid >> 32) : (unsigned)grid;
	const unsigned g8 = (g << 3) & 0xF8F8F8F8u;                                   /* 8 * grid, bytewise (grid in -1..1) */

	*a = (qh & 0x8000) ? __vsub4(g8, 0x01010101u) : __vadd4(g8, 0x01010101u);     /* 8 * (grid +- 1/8) */
	*sc = amp_h2f(b) * (float)(2 * (int)((qh >> 12) & 7) + 1) * 0.125f;
}

/* Stage A (gate and up of K = hidden -> h fp16 (slot, N)) or stage B (down of K = inner size, h quantized -> out[tok] += weight * result). */
template <int T, bool STAGE_A>
__global__ void __launch_bounds__(128)
amp_gguf_moe_mma_kernel(const int8_t *__restrict__ xq, const float *__restrict__ xs, const long long *__restrict__ tok,
                        const __half *__restrict__ wts, const long long *__restrict__ count, const long long *__restrict__ start,
                        const uint8_t *__restrict__ w0, const uint8_t *__restrict__ w1, __half *__restrict__ hbuf,
                        float *__restrict__ out, int N, int K, float lim, bool det)
{
	constexpr int bytes = T == AMP_Q_IQ2_XXS ? 66 : T == AMP_Q_IQ2_XS ? 74 : T == AMP_Q_IQ1_S ? 50 : 56;
	constexpr int NGRID = T == AMP_Q_IQ2_XXS ? 256 : T == AMP_Q_IQ2_XS ? 512 : 2048;

	/* the lookup tables are dead once the k loop is over: the epilogue tiles (gate, up / result: 2 x 32 x 65 floats) reuse their space */
	constexpr int TAB_BYTES = NGRID * 8 + 128 * 8;
	constexpr int CTILE_BYTES = 2 * MM_BM * (MM_BN + 1) * 4;
	__shared__ __align__(16) int8_t s_x[MM_BM * MM_XLD];
	__shared__ float s_xs[MM_BM * 8];
	__shared__ __align__(16) unsigned char s_union[TAB_BYTES > CTILE_BYTES ? TAB_BYTES : CTILE_BYTES];
	uint64_t *s_grid = (uint64_t *)s_union;
	uint64_t *s_signs = s_grid + NGRID;
	float (*s_c)[MM_BM * (MM_BN + 1)] = (float (*)[MM_BM * (MM_BN + 1)])s_union;

	const int e = blockIdx.y;
	const int cnt = (int)count[e];
	const int m0 = blockIdx.z * MM_BM;

	if (m0 >= cnt)
		return;

	const int n0 = blockIdx.x * MM_BN;
	const long long st = start[e];
	const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;

	for (int i = tid; i < NGRID; i += 128)
		s_grid[i] = (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? iq1s_grid[i] : (T == AMP_Q_IQ2_XS) ? iq2xs_grid[i] : iq2xxs_grid[i];

	for (int i = tid; i < 128; i += 128) {
		uint64_t m = 0;

		for (int b = 0; b < 8; b++)
			if (ksigns_iq2xs[i] & kmask_iq2xs[b])
				m |= (uint64_t)0xff << (8 * b);

		s_signs[i] = m;
	}

	const size_t rb = (size_t)(K / 256) * bytes;
	const int r0 = n0 + warp * 16 + gid;                           /* this thread's weight rows: r0 and r0 + 8 */
	const uint8_t *ra0 = w0 + ((size_t)e * N + r0) * rb, *ra1 = w0 + ((size_t)e * N + r0 + 8) * rb;
	const uint8_t *rb0 = STAGE_A ? w1 + ((size_t)e * N + r0) * rb : NULL, *rb1 = STAGE_A ? w1 + ((size_t)e * N + r0 + 8) * rb : NULL;
	float acc0[4][4], acc1[4][4];                                  /* [n8 tile][c0..c3]: gate (or the only matrix) / up */

#pragma unroll
	for (int t = 0; t < 4; t++)
#pragma unroll
		for (int i = 0; i < 4; i++)
			acc0[t][i] = acc1[t][i] = 0.f;

	for (int kc = 0; kc < K; kc += 256) {
		__syncthreads();

		/* activation tile: 32 tokens x 256 int8 and their 8 scales */
		for (int i = tid; i < MM_BM * 16; i += 128) {
			const int r = i >> 4, c16 = i & 15;
			uint4 v = make_uint4(0, 0, 0, 0);

			if (m0 + r < cnt) {
				const size_t row = STAGE_A ? (size_t)tok[st + m0 + r] : (size_t)(st + m0 + r);

				v = *(const uint4 *)(xq + row * K + kc + c16 * 16);
			}

			*(uint4 *)(s_x + r * MM_XLD + c16 * 16) = v;
		}

		for (int i = tid; i < MM_BM * 8; i += 128) {
			const int r = i >> 3, c = i & 7;

			s_xs[i] = (m0 + r < cnt) ? xs[(STAGE_A ? (size_t)tok[st + m0 + r] : (size_t)(st + m0 + r)) * (K / 32) + kc / 32 + c] : 0.f;
		}

		__syncthreads();

		const size_t boff = (size_t)(kc / 256) * bytes;

#pragma unroll 2
		for (int j = 0; j < 8; j++) {
#pragma unroll
			for (int h = 0; h < 2; h++) {
				unsigned a0, a1, g0, g1;
				float sa0, sa1, sg0, sg1;

				amp_mma_dec<T>(ra0 + boff, j, h, tig, s_grid, s_signs, &a0, &sa0);
				amp_mma_dec<T>(ra1 + boff, j, h, tig, s_grid, s_signs, &a1, &sa1);

				if (STAGE_A) {
					amp_mma_dec<T>(rb0 + boff, j, h, tig, s_grid, s_signs, &g0, &sg0);
					amp_mma_dec<T>(rb1 + boff, j, h, tig, s_grid, s_signs, &g1, &sg1);
				}

#pragma unroll
				for (int t = 0; t < 4; t++) {
					const unsigned b = *(const unsigned *)(s_x + (t * 8 + gid) * MM_XLD + j * 32 + h * 16 + tig * 4);
					const float x0 = s_xs[(t * 8 + 2 * tig) * 8 + j], x1 = s_xs[(t * 8 + 2 * tig + 1) * 8 + j];
					int c[4];

					amp_mma_s8_16816(c, a0, a1, b);
					acc0[t][0] += (float)c[0] * sa0 * x0;
					acc0[t][1] += (float)c[1] * sa0 * x1;
					acc0[t][2] += (float)c[2] * sa1 * x0;
					acc0[t][3] += (float)c[3] * sa1 * x1;

					if (STAGE_A) {
						amp_mma_s8_16816(c, g0, g1, b);
						acc1[t][0] += (float)c[0] * sg0 * x0;
						acc1[t][1] += (float)c[1] * sg0 * x1;
						acc1[t][2] += (float)c[2] * sg1 * x0;
						acc1[t][3] += (float)c[3] * sg1 * x1;
					}
				}
			}
		}
	}

	/* epilogue: the accumulators (weight row, token) -> smem [token][row] -> coalesced global writes */
	__syncthreads();

#pragma unroll
	for (int t = 0; t < 4; t++) {
#pragma unroll
		for (int i = 0; i < 4; i++) {
			const int row = warp * 16 + gid + (i >> 1) * 8, col = t * 8 + 2 * tig + (i & 1);

			s_c[0][col * (MM_BN + 1) + row] = acc0[t][i];

			if (STAGE_A)
				s_c[1][col * (MM_BN + 1) + row] = acc1[t][i];
		}
	}

	__syncthreads();

	for (int i = tid; i < MM_BM * MM_BN; i += 128) {
		const int col = i / MM_BN, row = i % MM_BN;

		if (m0 + col >= cnt)
			continue;

		float g = s_c[0][col * (MM_BN + 1) + row];

		if (STAGE_A) {
			float u = s_c[1][col * (MM_BN + 1) + row];

			if (lim > 0.f) {
				g = fminf(g, lim);
				u = fminf(fmaxf(u, -lim), lim);
			}

			hbuf[(size_t)(st + m0 + col) * N + n0 + row] = __float2half(g / (1.f + __expf(-g)) * u);
		} else {
			if (det)
				out[(size_t)(st + m0 + col) * N + n0 + row] = __half2float(wts[st + m0 + col]) * g;
			else
				atomicAdd(out + (size_t)tok[st + m0 + col] * N + n0 + row, __half2float(wts[st + m0 + col]) * g);
		}
	}
}

/* amp_gguf_moe_prefill on the int8 tensor-core kernel: yq / ys the quantized token rows (amp_gguf_quant_x), hq / hs scratch for the
 * quantized inner activations (A, I) / (A, I / 32); hbuf (A, I) fp16 scratch. */
extern "C" int
amp_gguf_moe_prefill_mma(int t13, int t2, const void *yq, const void *ys, const void *tok, const void *wts, const void *count,
                         const void *start, const void *w1, const void *w3, const void *w2, void *hbuf, void *hq, void *hs, void *out,
                         int E, int H, int I, int A, int max_rows, float lim, int det, cudaStream_t stream)
{
	if (E < 1 || (H % 256) || (I % 256) || (H % MM_BN) || (I % MM_BN) || max_rows < 1 || A < 1)
		return (int)cudaErrorInvalidValue;

	const dim3 blk(128);
	const unsigned mt = (unsigned)((max_rows + MM_BM - 1) / MM_BM);

#define AMP_MM_CASE(T, STAGE, GRID, XQ, XS, WA, WB, NN, KK) \
	case T: \
		amp_gguf_moe_mma_kernel<T, STAGE><<<GRID, blk, 0, stream>>>((const int8_t *)XQ, (const float *)XS, (const long long *)tok, \
			(const __half *)wts, (const long long *)count, (const long long *)start, (const uint8_t *)WA, (const uint8_t *)WB, \
			(__half *)hbuf, (float *)out, NN, KK, lim, det != 0); \
		break

	const dim3 ga(I / MM_BN, E, mt), gb(H / MM_BN, E, mt);

	switch (t13) {
	AMP_MM_CASE(AMP_Q_IQ2_XXS, true, ga, yq, ys, w1, w3, I, H);
	AMP_MM_CASE(AMP_Q_IQ2_XS, true, ga, yq, ys, w1, w3, I, H);
	AMP_MM_CASE(AMP_Q_IQ1_M, true, ga, yq, ys, w1, w3, I, H);
	AMP_MM_CASE(AMP_Q_IQ1_S, true, ga, yq, ys, w1, w3, I, H);
	default: return (int)cudaErrorNotSupported;
	}

	int rc = amp_gguf_quant_x(hbuf, hq, hs, A, I, stream);

	if (rc)
		return rc;

	switch (t2) {
	AMP_MM_CASE(AMP_Q_IQ2_XXS, false, gb, hq, hs, w2, w2, H, I);
	AMP_MM_CASE(AMP_Q_IQ2_XS, false, gb, hq, hs, w2, w2, H, I);
	AMP_MM_CASE(AMP_Q_IQ1_M, false, gb, hq, hs, w2, w2, H, I);
	AMP_MM_CASE(AMP_Q_IQ1_S, false, gb, hq, hs, w2, w2, H, I);
	default: return (int)cudaErrorNotSupported;
	}
#undef AMP_MM_CASE

	return (int)cudaGetLastError();
}

/* ---- embedding rows: gather + dequantize from a quantized table, by device-side ids (usable inside CUDA graphs) ----------- */

template <int T>
__global__ void amp_gguf_embed_kernel(const uint8_t *__restrict__ w, const long long *__restrict__ ids, __half *__restrict__ out, long nrows,
                                      long vocab, int K)
{
	constexpr int bytes = amp_qbytes(T);
	constexpr int gpb = amp_qgpb(T);       /* groups of 8 per block */
	const int gper = K / 8;
	const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= nrows * gper)
		return;

	const long r = i / gper;
	const int g = (int)(i % gper);
	long long id = ids[r];
	float v[8];
	__half h[8];

	if (id < 0 || id >= vocab)
		id = 0;

	const size_t rb = (size_t)(K / (gpb * 8)) * bytes;

	const void *gr = T == AMP_Q_IQ2_XS ? (const void *)iq2xs_grid :
	                 T == AMP_Q_IQ2_XXS ? (const void *)iq2xxs_grid :
	                 T == AMP_Q_IQ2_S ? (const void *)iq2s_grid :
	                 T == AMP_Q_IQ3_XXS ? (const void *)iq3xxs_grid :
	                 T == AMP_Q_IQ3_S ? (const void *)iq3s_grid :
	                 (T == AMP_Q_IQ1_M || T == AMP_Q_IQ1_S) ? (const void *)iq1s_grid : NULL;
	amp_dq_group<T>(w + (size_t)id * rb + (size_t)(g / gpb) * bytes, g % gpb, v, gr, ksigns_iq2xs);
#pragma unroll
	for (int j = 0; j < 8; j++)
		h[j] = __float2half(v[j]);
	*(uint4 *)(out + r * K + g * 8) = *(uint4 *)h;
}

extern "C" int
amp_gguf_embed(int qtype, const void *w, const void *ids, void *out, long nrows, long vocab, int K, cudaStream_t stream)
{
	const amp_qinfo qi = amp_qtype_info(qtype);

	if (nrows < 0 || vocab < 1 || K < qi.block || (K % qi.block) || ((uintptr_t)out & 15))
		return (int)cudaErrorInvalidValue;

	if (nrows == 0)
		return 0;

	const long n = nrows * (K / 8);
	const unsigned grid = (unsigned)((n + 255) / 256);

#define AMP_EMB_CASE(T) \
	case T: amp_gguf_embed_kernel<T><<<grid, 256, 0, stream>>>((const uint8_t *)w, (const long long *)ids, (__half *)out, nrows, vocab, K); break

	switch (qtype) {
	AMP_EMB_CASE(AMP_Q_Q4_0);
	AMP_EMB_CASE(AMP_Q_Q8_0);
	AMP_EMB_CASE(AMP_Q_IQ4_NL);
	AMP_EMB_CASE(AMP_Q_Q5_K);
	AMP_EMB_CASE(AMP_Q_IQ2_XXS);
	AMP_EMB_CASE(AMP_Q_IQ2_XS);
	AMP_EMB_CASE(AMP_Q_IQ1_M);
	AMP_EMB_CASE(AMP_Q_IQ1_S);
	AMP_EMB_CASE(AMP_Q_Q2_0);
	AMP_EMB_CASE(AMP_Q_Q4_K);
	AMP_EMB_CASE(AMP_Q_Q3_K);
	AMP_EMB_CASE(AMP_Q_Q6_K);
	AMP_EMB_CASE(AMP_Q_IQ2_S);
	AMP_EMB_CASE(AMP_Q_IQ3_XXS);
	AMP_EMB_CASE(AMP_Q_IQ3_S);
	AMP_EMB_CASE(AMP_Q_IQ4_XS);
	default: return (int)cudaErrorInvalidValue;
	}
#undef AMP_EMB_CASE

	return (int)cudaGetLastError();
}
