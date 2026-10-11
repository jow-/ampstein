// st.h — minimal safetensors file reader (pure C, json-c for the header).
//
// Format: 8-byte little-endian header length N, then N bytes of JSON
// describing the tensors (name -> {dtype, shape, data_offsets}), then the
// raw buffer. Offsets are relative to the end of the header.
// See https://github.com/huggingface/safetensors/blob/main/docs/format.md

#ifndef AMP_ST_H
#define AMP_ST_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
	int         fd;
	void       *base;      /* mmap base (whole file) */
	size_t      size;      /* file size */
	size_t      data_off;  /* start of data region (8 + header_len) */
	void       *json;      /* json_object * of the header dict */
} st_file_t;

/* Open `path`, mmap it, parse the header. Returns 0 and fills *out on
 * success, -1 on error (errno set for I/O errors). */
int st_open(const char *path, st_file_t *out);

/* Release mappings and the parsed header. */
void st_close(st_file_t *out);

/* Look up tensor `name`. Returns 0 and fills *dtype (string literal),
 * *shape (array of ndim int64s, caller frees), *offset and *nbytes
 * (relative to the data region) on success; ST_NOT_FOUND (-1) if the
 * name is absent; ST_MALFORMED (-2) when the entry exists but fails
 * validation (unknown dtype, negative shape, offsets outside the file
 * or inconsistent with shape x dtype). */
#define ST_NOT_FOUND -1
#define ST_MALFORMED  -2
int st_lookup(st_file_t *f, const char *name, const char **dtype,
              int64_t **shape, size_t *ndim, size_t *offset, size_t *nbytes);

/* Number of tensors in the file (excluding __metadata__). */
size_t st_count(st_file_t *f);

/* Name of the i-th tensor; NULL if out of range. Caller must not free. */
const char *st_name(st_file_t *f, size_t i);

#endif /* AMP_ST_H */