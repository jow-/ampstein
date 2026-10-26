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

/* Ask the kernel to drop the file's clean page cache and mapped pages.
 * Safe for files no longer read after load; do not use for a table that is
 * still streamed row by row. */
void st_drop_cache(st_file_t *f);

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

/* Element size / block layout of a dtype string: *blck elements are stored in *bytes bytes (blck is 1 for the plain
 * safetensors types, 32 / 256 for the ggml block types a GGUF file carries: Q8_0, Q5_K, IQ2_XXS, ...). Returns -1 for an
 * unknown dtype. A block type's last dimension is a multiple of blck, and its row is last_dim / blck * bytes bytes. */
int st_dtype_info(const char *dt, size_t *blck, size_t *bytes);

/* Open a GGUF file (v2 / v3), mmap it and present it as a safetensors-style header: every tensor becomes an entry with
 * the ggml type's name as its dtype, the shape in row-major order (ggml's ne[] reversed) and its byte range in the data
 * region; the key/value metadata sits under "__metadata__" (see st_meta_json). Returns 0 or -1 (errno for I/O errors,
 * EINVAL for a malformed file). */
int gguf_open(const char *path, st_file_t *out);

/* The "__metadata__" object of the header as a JSON string (malloc'ed, caller frees), or NULL when there is none. */
char *st_meta_json(st_file_t *f);

/* Number of tensors in the file (excluding __metadata__). */
size_t st_count(st_file_t *f);

/* Name of the i-th tensor; NULL if out of range. Caller must not free. */
const char *st_name(st_file_t *f, size_t i);

#endif /* AMP_ST_H */