/* image.c — see image.h. The resampler reproduces Pillow's
 * Resample.c (BICUBIC, a = -0.5, support scaled by the reduction
 * factor, 22-bit fixed-point coefficients, horizontal pass first with
 * an 8-bit intermediate) so the output matches exllamav3's
 * PIL-based preprocessing exactly. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_ONLY_TGA
#define STBI_ONLY_PSD
#define STBI_ONLY_PNM
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MAX_DIMENSIONS (1 << 14)
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "image.h"

#define PRECISION_BITS (32 - 8 - 2)
#define MAX_SOURCE_PIXELS (96L * 1024 * 1024)

static void
set_err(char *err, size_t errlen, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

#include <stdarg.h>
static void
set_err(char *err, size_t errlen, const char *fmt, ...)
{
	va_list ap;

	if (!err || !errlen)
		return;

	va_start(ap, fmt);
	vsnprintf(err, errlen, fmt, ap);
	va_end(ap);
}

int
amp_smart_resize(int height, int width, int factor, long min_pixels,
	long max_pixels, int *out_h, int *out_w, char *err, size_t errlen)
{
	if (height < factor || width < factor) {
		set_err(err, errlen, "image too small (%dx%d, minimum %d per side)",
			width, height, factor);
		return -1;
	}

	double mx = height > width ? height : width;
	double mn = height > width ? width : height;

	if (mx / mn > 200.0) {
		set_err(err, errlen, "aspect ratio %.1f exceeds 200", mx / mn);
		return -1;
	}

	/* Python's round() is half-to-even; nearbyint() under the default
	 * rounding mode is the same. */
	long h_bar = (long)nearbyint((double)height / factor) * factor;
	long w_bar = (long)nearbyint((double)width / factor) * factor;

	if (h_bar * w_bar > max_pixels) {
		double beta = sqrt(((double)height * width) / (double)max_pixels);

		h_bar = (long)floor(height / beta / factor) * factor;
		w_bar = (long)floor(width / beta / factor) * factor;
	}
	else if (h_bar * w_bar < min_pixels) {
		double beta = sqrt((double)min_pixels / ((double)height * width));

		h_bar = (long)ceil(height * beta / factor) * factor;
		w_bar = (long)ceil(width * beta / factor) * factor;
	}

	if (h_bar < factor || w_bar < factor) {
		set_err(err, errlen, "image does not fit the pixel budget");
		return -1;
	}

	*out_h = (int)h_bar;
	*out_w = (int)w_bar;

	return 0;
}

/* ---- Pillow-compatible resampling -------------------------------- */

static double
bicubic_filter(double x)
{
	const double a = -0.5;

	if (x < 0.0)
		x = -x;

	if (x < 1.0)
		return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;

	if (x < 2.0)
		return (((x - 5) * x + 8) * x - 4) * a;

	return 0.0;
}

/* Per output sample: first input index, tap count, and fixed-point
 * taps (stride ksize). Returns the tap stride or -1. */
static int
precompute_coeffs(int in_size, int out_size, int **bounds_out, int **kk_out)
{
	double scale = (double)in_size / out_size;
	double filterscale = scale < 1.0 ? 1.0 : scale;
	double support = 2.0 * filterscale;
	int ksize = (int)ceil(support) * 2 + 1;
	int *bounds = malloc(sizeof(int) * 2 * out_size);
	int *kk = malloc(sizeof(int) * (size_t)out_size * ksize);
	double *k = malloc(sizeof(double) * ksize);

	if (!bounds || !kk || !k) {
		free(bounds);
		free(kk);
		free(k);
		return -1;
	}

	for (int xx = 0; xx < out_size; xx++) {
		double center = (xx + 0.5) * scale;
		double ww = 0.0;
		double ss = 1.0 / filterscale;
		int xmin = (int)(center - support + 0.5);
		int xmax = (int)(center + support + 0.5);

		if (xmin < 0)
			xmin = 0;

		if (xmax > in_size)
			xmax = in_size;

		xmax -= xmin;

		for (int x = 0; x < xmax; x++) {
			double w = bicubic_filter((x + xmin - center + 0.5) * ss);

			k[x] = w;
			ww += w;
		}

		for (int x = 0; x < xmax; x++)
			if (ww != 0.0)
				k[x] /= ww;

		for (int x = 0; x < ksize; x++) {
			double v = x < xmax ? k[x] : 0.0;

			kk[(size_t)xx * ksize + x] = (int)(v < 0 ?
				-0.5 + v * (1 << PRECISION_BITS) :
				0.5 + v * (1 << PRECISION_BITS));
		}

		bounds[xx * 2 + 0] = xmin;
		bounds[xx * 2 + 1] = xmax;
	}

	free(k);
	*bounds_out = bounds;
	*kk_out = kk;

	return ksize;
}

static inline unsigned char
clip8(int v)
{
	v >>= PRECISION_BITS;

	return v < 0 ? 0 : v > 255 ? 255 : (unsigned char)v;
}

/* RGB (3 interleaved bytes/pixel) resize, horizontal pass first. */
static unsigned char *
resample_rgb(const unsigned char *src, int sw, int sh, int dw, int dh)
{
	unsigned char *cur = NULL, *tmp = NULL, *out = NULL;
	const unsigned char *in = src;
	int iw = sw;

	if (dw != sw) {
		int *bounds, *kk;
		int ks = precompute_coeffs(sw, dw, &bounds, &kk);

		if (ks < 0)
			return NULL;

		tmp = malloc((size_t)dw * sh * 3);

		if (!tmp) {
			free(bounds);
			free(kk);
			return NULL;
		}

		for (int y = 0; y < sh; y++) {
			for (int xx = 0; xx < dw; xx++) {
				int xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
				const int *k = &kk[(size_t)xx * ks];

				for (int c = 0; c < 3; c++) {
					int ss = 1 << (PRECISION_BITS - 1);

					for (int x = 0; x < xmax; x++)
						ss += in[((size_t)y * iw + xmin + x) * 3 + c] * k[x];

					tmp[((size_t)y * dw + xx) * 3 + c] = clip8(ss);
				}
			}
		}

		free(bounds);
		free(kk);
		cur = tmp;
		in = tmp;
		iw = dw;
	}

	if (dh != sh) {
		int *bounds, *kk;
		int ks = precompute_coeffs(sh, dh, &bounds, &kk);

		if (ks < 0) {
			free(tmp);
			return NULL;
		}

		out = malloc((size_t)dw * dh * 3);

		if (!out) {
			free(bounds);
			free(kk);
			free(tmp);
			return NULL;
		}

		for (int yy = 0; yy < dh; yy++) {
			int ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
			const int *k = &kk[(size_t)yy * ks];

			for (int x = 0; x < dw * 3; x++) {
				int ss = 1 << (PRECISION_BITS - 1);

				for (int y = 0; y < ymax; y++)
					ss += in[(size_t)(ymin + y) * iw * 3 + x] * k[y];

				out[(size_t)yy * dw * 3 + x] = clip8(ss);
			}
		}

		free(bounds);
		free(kk);
		free(tmp);

		return out;
	}

	if (cur)
		return cur;

	out = malloc((size_t)dw * dh * 3);

	if (out)
		memcpy(out, src, (size_t)dw * dh * 3);

	return out;
}

/* Pillow's MULDIV255 / BLEND for compositing on white. */
static inline int
muldiv255(int a, int b)
{
	int tmp = a * b + 128;

	return ((tmp >> 8) + tmp) >> 8;
}

/* exllamav3 glm5_vision_canvas (mm_processing/glm5_next.py): aligned canvas
 * (height, width) within the spatiotemporal token budget. */
static void
glm5_canvas(int num_frames, int height, int width, int temporal, int factor,
            long min_tokens, long max_tokens, int *out_h, int *out_w)
{
	long ppt = (long)temporal * factor * factor;
	long min_pixels = min_tokens * ppt, max_pixels = max_tokens * ppt;
	#define ALIGN(v, f) ((((v) + (f) - 1) / (f)) * (f))
	int rf = (int)floor((double)num_frames / temporal + 0.5);   /* python round(): half to even below */
	long frames = rf * temporal;

	/* Python's round() is banker's rounding; num_frames / temporal is an
	 * exact .0 or .5 here — resolve .5 to even */
	{
		double q = (double)num_frames / temporal;
		double fl = floor(q);

		if (q - fl == 0.5)
			rf = ((long)fl % 2 == 0) ? (int)fl : (int)fl + 1;
		else
			rf = (int)floor(q + 0.5);

		frames = (long)rf * temporal;
	}

	if (frames < temporal)
		frames = temporal;

	long ah = ALIGN(height, factor), aw = ALIGN(width, factor);
	long budget = frames * ah * aw;

	if (budget < min_pixels) {
		double scale = sqrt((double)min_pixels / ((double)num_frames * height * width));
		long sh = (long)ceil(height * scale), sw = (long)ceil(width * scale);

		ah = ALIGN(sh < 1 ? 1 : sh, factor);
		aw = ALIGN(sw < 1 ? 1 : sw, factor);
		budget = frames * ah * aw;
	}

	if (budget > max_pixels) {
		long low = 1, high = height;
		long best_h = factor, best_w = factor;

		while (low <= high) {
			long ch = (low + high) / 2;
			long cw = (long)floor((double)width * ch / height);

			if (cw < 1) cw = 1;

			long cand_h = ALIGN(ch, factor), cand_w = ALIGN(cw, factor);

			if (frames * cand_h * cand_w <= max_pixels) {
				best_h = cand_h;
				best_w = cand_w;
				low = ch + 1;
			}
			else
				high = ch - 1;
		}

		ah = best_h;
		aw = best_w;
	}

	#undef ALIGN
	*out_h = (int)ah;
	*out_w = (int)aw;
}

int
amp_image_info(const unsigned char *data, size_t len, int *w, int *h)
{
	int comp;

	if (len > 0x7fffffffu || !stbi_info_from_memory(data, (int)len, w, h, &comp))
		return -1;

	return 0;
}

int
amp_image_prepare(const unsigned char *data, size_t len,
	const struct amp_image_params *p, struct amp_image_result *res,
	char *err, size_t errlen)
{
	int w, h, comp;
	unsigned char *rgba = NULL, *rgb = NULL, *rs = NULL;
	float *patches = NULL;

	if (len > 0x7fffffffu) {
		set_err(err, errlen, "image data too large");
		return -1;
	}

	if (!stbi_info_from_memory(data, (int)len, &w, &h, &comp)) {
		set_err(err, errlen, "unsupported or corrupt image (%s)",
			stbi_failure_reason());
		return -1;
	}

	if ((long)w * h > MAX_SOURCE_PIXELS) {
		set_err(err, errlen, "image too large (%dx%d)", w, h);
		return -1;
	}

	rgba = stbi_load_from_memory(data, (int)len, &w, &h, &comp, 4);

	if (!rgba) {
		set_err(err, errlen, "image decode failed (%s)",
			stbi_failure_reason());
		return -1;
	}

	int nh, nw;
	int ch = 0, cw = 0;          /* canvas mode: content size inside the canvas */
	int ox = 0, oy = 0, fill = 0;
	int canvas = p->min_tokens > 0 || p->target_w > 0;

	if (p->target_w > 0) {
		/* PIL ImageOps.contain / pad (round() is half-even) */
		nh = p->target_h;
		nw = p->target_w;
		cw = nw;
		ch = nh;
		fill = 127;

		if (p->pad && (double)w / h != (double)nw / nh) {
			if ((double)w / h > (double)nw / nh)
				ch = (int)nearbyint((double)h / w * nw);
			else
				cw = (int)nearbyint((double)w / h * nh);

			ch = ch < 1 ? 1 : ch;
			cw = cw < 1 ? 1 : cw;

			if (cw != nw)
				ox = (int)nearbyint((double)(nw - cw) * 0.5);
			else
				oy = (int)nearbyint((double)(nh - ch) * 0.5);
		}
	}
	else if (p->min_tokens > 0) {
		int factor = p->patch_size * p->merge_size;
		int tps = p->temporal_patch_size;

		glm5_canvas(tps, h, w, tps, factor, p->min_tokens, p->max_tokens, &nh, &nw);

		double scale = fmin((double)nh / h, (double)nw / w);

		if ((long)tps * h * w >= (long)tps * factor * factor * p->min_tokens)
			scale = fmin(1.0, scale);

		ch = (int)floor(h * scale);
		cw = (int)floor(w * scale);
		ch = ch < 1 ? 1 : ch > nh ? nh : ch;
		cw = cw < 1 ? 1 : cw > nw ? nw : cw;
	}
	else if (amp_smart_resize(h, w, p->patch_size * p->merge_size, p->min_pixels,
	                         p->max_pixels, &nh, &nw, err, errlen)) {
		stbi_image_free(rgba);
		return -1;
	}

	rgb = malloc((size_t)w * h * 3);

	if (!rgb) {
		stbi_image_free(rgba);
		set_err(err, errlen, "out of memory");
		return -1;
	}

	/* transparency over white (Pillow: paste with the alpha as mask) */
	for (size_t i = 0; i < (size_t)w * h; i++) {
		int a = rgba[i * 4 + 3];

		for (int c = 0; c < 3; c++) {
			int s = rgba[i * 4 + c];

			rgb[i * 3 + c] = a == 255 ? s : a == 0 ? 255 :
				muldiv255(255, 255 - a) + muldiv255(s, a);
		}
	}

	stbi_image_free(rgba);

	if (canvas) {
		/* resize the content to (cw, ch) (if it differs), paste it on a
		 * nh x nw canvas (black, or gray in DeepSeek mode) at (ox, oy) */
		unsigned char *content = (cw != w || ch != h) ?
			resample_rgb(rgb, w, h, cw, ch) : NULL;
		const unsigned char *src = content ? content : rgb;

		rs = calloc((size_t)nw * nh * 3, 1);

		if (rs && fill)
			memset(rs, fill, (size_t)nw * nh * 3);

		if (rs && (content || (cw == w && ch == h)))
			for (int y = 0; y < ch; y++)
				memcpy(rs + ((size_t)(y + oy) * nw + ox) * 3, src + (size_t)y * cw * 3, (size_t)cw * 3);
		else if (rs) {
			free(rs);
			rs = NULL;
		}

		free(content);
	}
	else
		rs = resample_rgb(rgb, w, h, nw, nh);

	free(rgb);

	if (!rs) {
		set_err(err, errlen, "out of memory");
		return -1;
	}

	/* normalize + flatten into (grid_h * grid_w, 3 * tp * ps * ps) rows
	 * in merge-block order; both temporal slots carry the same frame */
	int ps = p->patch_size, m = p->merge_size, tp = p->temporal_patch_size;
	int gh = nh / ps, gw = nw / ps;
	size_t row = (size_t)3 * tp * ps * ps;
	float rescale = (float)(1.0 / 255.0);

	patches = malloc((size_t)gh * gw * row * sizeof(float));

	if (!patches) {
		free(rs);
		set_err(err, errlen, "out of memory");
		return -1;
	}

	for (int a = 0; a < gh / m; a++)
	for (int b = 0; b < gw / m; b++)
	for (int i = 0; i < m; i++)
	for (int j = 0; j < m; j++) {
		size_t r = (((size_t)a * (gw / m) + b) * m + i) * m + j;
		float *o = patches + r * row;

		if (p->hwc) {
			for (int py = 0; py < ps; py++)
			for (int px = 0; px < ps; px++)
			for (int c = 0; c < 3; c++) {
				int y = (a * m + i) * ps + py;
				int x = (b * m + j) * ps + px;
				float v = (float)rs[((size_t)y * nw + x) * 3 + c] * rescale;

				*o++ = (v - p->mean[c]) / p->std[c];
			}

			continue;
		}

		for (int c = 0; c < 3; c++)
		for (int t = 0; t < tp; t++)
		for (int py = 0; py < ps; py++)
		for (int px = 0; px < ps; px++) {
			int y = (a * m + i) * ps + py;
			int x = (b * m + j) * ps + px;
			float v = (float)rs[((size_t)y * nw + x) * 3 + c] * rescale;

			*o++ = (v - p->mean[c]) / p->std[c];
		}
	}

	free(rs);

	res->patches = patches;
	res->grid_h = gh;
	res->grid_w = gw;
	res->width = w;
	res->height = h;
	res->new_width = nw;
	res->new_height = nh;

	return 0;
}
