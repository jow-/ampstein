// gguf.c — GGUF reader (pure C, json-c): presents a .gguf file as a safetensors-style st_file_t.
// See st.h for the interface.
//
// The file is UNTRUSTED input like a safetensors header (model files come from the network): every count, length and
// offset is checked against the file before use. The tensors become entries {dtype, shape, data_offsets} in the
// synthesized header, so st_lookup validates them again (type known, last dimension a multiple of the block size, byte
// range inside the file and equal to shape x type) and everything built on st_file_t (zero-copy views, h2d_file,
// read_into, gather) works on GGUF weights unchanged.
//
// Layout (v2 / v3, little endian): "GGUF", u32 version, u64 tensor count, u64 kv count, the kv pairs (string key, u32
// value type, value), the tensor infos (string name, u32 n_dims, u64 ne[n_dims], u32 ggml type, u64 offset), padding
// to general.alignment (default 32), then the tensor data; offsets are relative to the start of the data.
// ggml lists the dimensions innermost first (ne[0] is the row length): the shape here is the reverse.

#include "st.h"

#include <json-c/json.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAX_DIMS   8
#define GGUF_MAX_ITEMS  ((uint64_t)1 << 22)     /* tensors / kv pairs / array elements: far above any real file */
#define GGUF_MAX_DEPTH  4

static const char *gguf_types[] = {
	[0] = "F32", [1] = "F16", [2] = "Q4_0", [3] = "Q4_1", [6] = "Q5_0", [7] = "Q5_1", [8] = "Q8_0",
	[10] = "Q2_K", [11] = "Q3_K", [12] = "Q4_K", [13] = "Q5_K", [14] = "Q6_K",
	[16] = "IQ2_XXS", [17] = "IQ2_XS", [18] = "IQ3_XXS", [19] = "IQ1_S", [20] = "IQ4_NL", [21] = "IQ3_S", [22] = "IQ2_S",
	[23] = "IQ4_XS", [24] = "I8", [25] = "I16", [26] = "I32", [27] = "I64", [28] = "F64", [29] = "IQ1_M", [30] = "BF16", [42] = "Q2_0",
};

typedef struct {
	const uint8_t *p, *end;
	int            bad;
} rd_t;

static int
rd_take(rd_t *r, void *dst, uint64_t n)
{
	if (r->bad || n > (uint64_t)(r->end - r->p)) {
		r->bad = 1;
		return -1;
	}

	if (dst)
		memcpy(dst, r->p, n);

	r->p += n;

	return 0;
}

static uint32_t rd_u32(rd_t *r) { uint32_t v = 0; rd_take(r, &v, 4); return v; }
static uint64_t rd_u64(rd_t *r) { uint64_t v = 0; rd_take(r, &v, 8); return v; }

/* a length-prefixed string; NULL on error (malloc'ed otherwise) */
static char *
rd_str(rd_t *r)
{
	uint64_t n = rd_u64(r);

	if (r->bad || n > (uint64_t)(r->end - r->p)) {
		r->bad = 1;
		return NULL;
	}

	char *s = malloc(n + 1);

	if (!s) {
		r->bad = 1;
		return NULL;
	}

	memcpy(s, r->p, n);
	s[n] = 0;
	r->p += n;

	return s;
}

/* one value of kv type t as a json object (NULL on error) */
static json_object *
rd_value(rd_t *r, uint32_t t, int depth)
{
	union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16; uint32_t u32; int32_t i32; uint64_t u64; int64_t i64;
	        float f32; double f64; } v;

	switch (t) {
	case 0:  if (rd_take(r, &v, 1)) return NULL; return json_object_new_int64(v.u8);
	case 1:  if (rd_take(r, &v, 1)) return NULL; return json_object_new_int64(v.i8);
	case 2:  if (rd_take(r, &v, 2)) return NULL; return json_object_new_int64(v.u16);
	case 3:  if (rd_take(r, &v, 2)) return NULL; return json_object_new_int64(v.i16);
	case 4:  if (rd_take(r, &v, 4)) return NULL; return json_object_new_int64(v.u32);
	case 5:  if (rd_take(r, &v, 4)) return NULL; return json_object_new_int64(v.i32);
	case 6:  if (rd_take(r, &v, 4)) return NULL; return json_object_new_double(v.f32);
	case 7:  if (rd_take(r, &v, 1)) return NULL; return json_object_new_boolean(v.u8 != 0);
	case 10: if (rd_take(r, &v, 8)) return NULL; return json_object_new_int64((int64_t)v.u64);
	case 11: if (rd_take(r, &v, 8)) return NULL; return json_object_new_int64(v.i64);
	case 12: if (rd_take(r, &v, 8)) return NULL; return json_object_new_double(v.f64);
	case 8: {
		char *s = rd_str(r);

		if (!s)
			return NULL;

		json_object *o = json_object_new_string(s);

		free(s);

		return o;
	}
	case 9: {
		uint32_t et = rd_u32(r);
		uint64_t n = rd_u64(r);

		if (r->bad || depth >= GGUF_MAX_DEPTH || n > GGUF_MAX_ITEMS || (et == 9 && !n))
			return NULL;

		json_object *a = json_object_new_array_ext((int)n);

		for (uint64_t i = 0; a && i < n; i++) {
			json_object *e = rd_value(r, et, depth + 1);

			if (!e) {
				json_object_put(a);
				return NULL;
			}

			json_object_array_add(a, e);
		}

		return a;
	}
	}

	return NULL;
}

int
gguf_open(const char *path, st_file_t *out)
{
	struct stat sb;
	int fd = open(path, O_RDONLY);

	if (fd < 0)
		return -1;

	if (fstat(fd, &sb) < 0 || sb.st_size < 24) {
		close(fd);
		errno = EINVAL;
		return -1;
	}

	void *base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

	if (base == MAP_FAILED) {
		close(fd);
		return -1;
	}

	rd_t r = { base, (const uint8_t *)base + sb.st_size, 0 };
	json_object *root = json_object_new_object(), *meta = json_object_new_object();
	uint64_t alignment = 32, ntens = 0, nkv = 0;
	uint32_t version = 0;
	char magic[4];

	if (!root || !meta)
		goto bad;

	rd_take(&r, magic, 4);
	version = rd_u32(&r);
	ntens = rd_u64(&r);
	nkv = rd_u64(&r);

	if (r.bad || memcmp(magic, "GGUF", 4) || (version != 2 && version != 3) || ntens > GGUF_MAX_ITEMS ||
	    nkv > GGUF_MAX_ITEMS)
		goto bad;

	for (uint64_t i = 0; i < nkv; i++) {
		char *key = rd_str(&r);
		uint32_t t = rd_u32(&r);
		json_object *val = key ? rd_value(&r, t, 0) : NULL;

		if (!key || !val) {
			free(key);
			goto bad;
		}

		if (!strcmp(key, "general.alignment") && json_object_is_type(val, json_type_int))
			alignment = (uint64_t)json_object_get_int64(val);

		json_object_object_add(meta, key, val);
		free(key);
	}

	if (!alignment || (alignment & (alignment - 1)) || alignment > ((uint64_t)1 << 20))
		goto bad;

	/* the tensor infos; their offsets are relative to the data start, known only after the last info */
	struct { char *name; const char *dtype; int64_t dims[GGUF_MAX_DIMS]; uint32_t nd; uint64_t off; } *ti =
		calloc(ntens ? ntens : 1, sizeof(*ti));

	if (!ti)
		goto bad;

	uint64_t i;

	for (i = 0; i < ntens; i++) {
		ti[i].name = rd_str(&r);
		ti[i].nd = rd_u32(&r);

		if (!ti[i].name || ti[i].nd > GGUF_MAX_DIMS)
			goto bad_ti;

		uint64_t ne[GGUF_MAX_DIMS];

		for (uint32_t d = 0; d < ti[i].nd; d++) {
			ne[d] = rd_u64(&r);

			if (ne[d] > ((uint64_t)1 << 62))
				goto bad_ti;
		}

		for (uint32_t d = 0; d < ti[i].nd; d++)
			ti[i].dims[d] = (int64_t)ne[ti[i].nd - 1 - d];

		uint32_t gt = rd_u32(&r);

		ti[i].off = rd_u64(&r);

		if (r.bad || gt >= sizeof(gguf_types) / sizeof(gguf_types[0]) || !gguf_types[gt])
			goto bad_ti;

		ti[i].dtype = gguf_types[gt];
	}

	uint64_t hdr_end = (uint64_t)(r.p - (const uint8_t *)base);
	uint64_t data_off = (hdr_end + alignment - 1) & ~(alignment - 1);

	if (r.bad || data_off > (uint64_t)sb.st_size)
		goto bad_ti;

	for (i = 0; i < ntens; i++) {
		size_t blck, bytes;
		uint64_t product = 1;

		if (st_dtype_info(ti[i].dtype, &blck, &bytes))
			goto bad_ti;

		for (uint32_t d = 0; d < ti[i].nd; d++) {
			if (ti[i].dims[d] && product > ((uint64_t)1 << 62) / (uint64_t)ti[i].dims[d])
				goto bad_ti;

			product *= (uint64_t)ti[i].dims[d];
		}

		if (blck > 1 && (!ti[i].nd || ti[i].dims[ti[i].nd - 1] % (int64_t)blck))
			goto bad_ti;

		uint64_t nbytes = product / blck * bytes;

		if (ti[i].off > (uint64_t)sb.st_size - data_off || nbytes > (uint64_t)sb.st_size - data_off - ti[i].off)
			goto bad_ti;

		json_object *ent = json_object_new_object(), *shape = json_object_new_array();
		json_object *offs = json_object_new_array();

		for (uint32_t d = 0; d < ti[i].nd; d++)
			json_object_array_add(shape, json_object_new_int64(ti[i].dims[d]));

		json_object_array_add(offs, json_object_new_int64((int64_t)ti[i].off));
		json_object_array_add(offs, json_object_new_int64((int64_t)(ti[i].off + nbytes)));
		json_object_object_add(ent, "dtype", json_object_new_string(ti[i].dtype));
		json_object_object_add(ent, "shape", shape);
		json_object_object_add(ent, "data_offsets", offs);
		json_object_object_add(root, ti[i].name, ent);
	}

	for (i = 0; i < ntens; i++)
		free(ti[i].name);

	free(ti);
	json_object_object_add(root, "__metadata__", meta);

	out->fd = fd;
	out->base = base;
	out->size = (size_t)sb.st_size;
	out->data_off = (size_t)data_off;
	out->json = root;

	return 0;

bad_ti:
	for (uint64_t j = 0; j < ntens; j++)
		free(ti[j].name);

	free(ti);
bad:
	json_object_put(root);      /* (the metadata object is not attached to it yet) */
	json_object_put(meta);
	munmap(base, (size_t)sb.st_size);
	close(fd);
	errno = EINVAL;

	return -1;
}
