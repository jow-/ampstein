// st.c — minimal safetensors file reader (pure C, json-c for the header).
// See st.h for the interface.
//
// The header is UNTRUSTED input (model dirs come from the network): every
// offset, size and shape is validated against the file before it is
// handed out, so a malformed header fails st_open/st_lookup instead of
// becoming an out-of-bounds read or (through h2d_file) a device write.

#include "st.h"

#include <json-c/json.h>

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Known dtype strings and their element sizes (the safetensors spec
 * table). Anything else is rejected. */
static const struct {
	const char *name;
	size_t size;
} st_dtypes[] = {
	{ "F64",  8 }, { "F32", 4 }, { "F16", 2 }, { "BF16", 2 },
	{ "I64",  8 }, { "I32", 4 }, { "I16", 2 }, { "I8",   1 },
	{ "U8",   1 }, { "U16", 2 }, { "U32", 4 }, { "U64",  8 },
	{ "BOOL", 1 }, { "F8_E4M3", 1 }, { "F8_E5M2", 1 }, { "F8_E8M0", 1 },
};

static size_t
st_elem_size(const char *dt)
{
	for (size_t i = 0; i < sizeof(st_dtypes) / sizeof(st_dtypes[0]); i++)
		if (!strcmp(dt, st_dtypes[i].name))
			return st_dtypes[i].size;

	return 0;
}

int
st_open(const char *path, st_file_t *out)
{
	struct stat st;
	uint64_t header_len;
	int fd;

	fd = open(path, O_RDONLY);

	if (fd < 0)
		return -1;

	if (fstat(fd, &st) < 0 || st.st_size < 8) {
		close(fd);
		return -1;
	}

	void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

	if (base == MAP_FAILED) {
		close(fd);
		return -1;
	}

	memcpy(&header_len, base, sizeof(header_len));

	/* overflow-safe bounds: the header must fit between the 8-byte
	 * length prefix and the end of the file, and json_tokener takes
	 * an int length */
	if (header_len < 2 || header_len > (uint64_t)st.st_size - 8 ||
	    header_len > 0x7fffffff) {
		munmap(base, (size_t)st.st_size);
		close(fd);
		return -1;
	}

	char *hdr = (char *)base + 8;
	json_tokener *tok = json_tokener_new();
	json_object *js = json_tokener_parse_ex(tok, hdr, (int)header_len);
	enum json_tokener_error err = json_tokener_get_error(tok);
	json_tokener_free(tok);

	if (!js || err != json_tokener_success ||
	    !json_object_is_type(js, json_type_object)) {
		munmap(base, (size_t)st.st_size);
		close(fd);
		return -1;
	}

	out->fd = fd;
	out->base = base;
	out->size = (size_t)st.st_size;
	out->data_off = 8 + header_len;
	out->json = js;

	return 0;
}

void
st_close(st_file_t *f)
{
	if (f->json)
		json_object_put((json_object *)f->json);

	if (f->base)
		munmap(f->base, f->size);

	if (f->fd >= 0)
		close(f->fd);

	f->json = NULL;
	f->base = NULL;
	f->fd = -1;
}

int
st_lookup(st_file_t *f, const char *name, const char **dtype,
          int64_t **shape, size_t *ndim, size_t *offset, size_t *nbytes)
{
	json_object *root = (json_object *)f->json;
	json_object *ent;

	if (!json_object_object_get_ex(root, name, &ent))
		return ST_NOT_FOUND;

	json_object *jd, *js, *jo;

	if (!json_object_is_type(ent, json_type_object) ||
	    !json_object_object_get_ex(ent, "dtype", &jd) ||
	    !json_object_object_get_ex(ent, "shape", &js) ||
	    !json_object_object_get_ex(ent, "data_offsets", &jo) ||
	    !json_object_is_type(jd, json_type_string) ||
	    !json_object_is_type(js, json_type_array) ||
	    !json_object_is_type(jo, json_type_array) ||
	    json_object_array_length(jo) != 2)
		return ST_MALFORMED;

	const char *dt = json_object_get_string(jd);
	size_t esize = st_elem_size(dt);

	if (!esize)
		return ST_MALFORMED;

	size_t nd = (size_t)json_object_array_length(js);
	int64_t *dims = malloc((nd ? nd : 1) * sizeof(int64_t));

	if (!dims)
		return ST_MALFORMED;

	/* product of the shape; nd == 0 is a scalar (one element) */
	uint64_t product = 1;

	for (size_t i = 0; i < nd; i++) {
		json_object *d = json_object_array_get_idx(js, i);

		if (!d || !json_object_is_type(d, json_type_int)) {
			free(dims);
			return ST_MALFORMED;
		}

		int64_t v = json_object_get_int64(d);

		if (v < 0) {
			free(dims);
			return ST_MALFORMED;
		}

		dims[i] = v;
		product *= (uint64_t)v;

		if (product > (uint64_t)1 << 62) {
			free(dims);
			return ST_MALFORMED;
		}
	}

	json_object *o0 = json_object_array_get_idx(jo, 0);
	json_object *o1 = json_object_array_get_idx(jo, 1);

	if (!o0 || !o1 || !json_object_is_type(o0, json_type_int) ||
	    !json_object_is_type(o1, json_type_int)) {
		free(dims);
		return ST_MALFORMED;
	}

	int64_t begin = json_object_get_int64(o0);
	int64_t end = json_object_get_int64(o1);
	uint64_t data_size = f->size - f->data_off;

	/* end >= begin, both inside the data region, and the byte count
	 * must match shape x dtype (an oversized nbytes would otherwise
	 * read past the mmap — or write past a device buffer sized from
	 * the shape, through h2d_file) */
	if (begin < 0 || end < begin || (uint64_t)end > data_size ||
	    product * esize != (uint64_t)(end - begin)) {
		free(dims);
		return ST_MALFORMED;
	}

	*dtype = dt;
	*ndim = nd;
	*shape = dims;
	*offset = (size_t)begin;
	*nbytes = (size_t)(end - begin);

	return 0;
}

size_t
st_count(st_file_t *f)
{
	json_object *root = (json_object *)f->json;

	if (!root)
		return 0;

	size_t n = (size_t)json_object_object_length(root);
	json_object *meta;

	/* __metadata__ is a file-level key, not a tensor */
	if (json_object_object_get_ex(root, "__metadata__", &meta) && n > 0)
		n--;

	return n;
}

const char *
st_name(st_file_t *f, size_t i)
{
	json_object *root = (json_object *)f->json;
	json_object_object_foreach(root, key, val)
	{
		(void)val;

		if (!strcmp(key, "__metadata__"))
			continue;

		if (i == 0)
			return key;

		i--;
	}

	return NULL;
}