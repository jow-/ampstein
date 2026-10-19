/* image.h — native image preprocessing for the vision tower.
 *
 * Decodes PNG/JPEG/BMP/GIF/... (stb_image), composites transparency on
 * white, applies the Qwen2/3-VL "smart resize" with a PIL-compatible
 * antialiased bicubic filter, normalizes and flattens the pixels into
 * the patch order the vision tower's patch embedding consumes.
 * Pure C, no ucode/torch types.
 */
#ifndef AMP_IMAGE_H
#define AMP_IMAGE_H

#include <stddef.h>

struct amp_image_params {
	int patch_size;          /* 16 */
	int merge_size;          /* spatial merge, 2 */
	int temporal_patch_size; /* 2 */
	long min_pixels;
	long max_pixels;
	float mean[3];
	float std[3];
	/* GLM-5 canvas mode (min_tokens > 0): the aligned canvas within a token
	 * budget (tokens of tp * (ps * merge)^2 pixels), the content scaled
	 * to fit keeping the aspect (never upscaled once the raw pixels meet
	 * the minimum), zero-padded top-left — exllamav3's Glm5NextVisionModel
	 * .preprocess; min_pixels / max_pixels are ignored. */
	long min_tokens;
	long max_tokens;
	/* DeepSeek-V4 mode (target_w > 0): the caller computed the final size
	 * (target_w x target_h); the image is either stretched to it (pad == 0)
	 * or fitted keeping the aspect and centred on a gray (127) canvas
	 * (pad == 1: PIL ImageOps.pad, bicubic). Use merge_size = 1. */
	int target_w, target_h;
	int pad;
	/* Gemma 4: every patch is flattened (y, x, channel) instead of (channel, t, y, x) */
	int hwc;
};

struct amp_image_result {
	float *patches;          /* malloc'd: n_patches * (3 * tp * ps * ps) */
	int grid_h, grid_w;      /* patch grid (before merging) */
	int width, height;       /* decoded source size */
	int new_width, new_height;
};

/* Returns 0 on success (caller frees res->patches), -1 on failure with
 * a message in err (size errlen). */
int amp_image_prepare(const unsigned char *data, size_t len,
	const struct amp_image_params *p, struct amp_image_result *res,
	char *err, size_t errlen);

/* Decoded size of an encoded image without decoding the pixels; 0 or -1. */
int amp_image_info(const unsigned char *data, size_t len, int *w, int *h);

/* Qwen2-VL smart resize: dims are (height, width); returns 0 or -1. */
int amp_smart_resize(int height, int width, int factor, long min_pixels,
	long max_pixels, int *out_h, int *out_w, char *err, size_t errlen);

#endif
