// tokenizer.h — HF tokenizer.json byte-level BPE engine (pure C).
//
// The engine is ucode-agnostic: it works on plain C buffers and returns
// malloc'd results owned by the caller. The ucode glue lives in
// amp_tokenizer.cpp.

#ifndef AMP_TOKENIZER_H
#define AMP_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct tok tok_t;

/* Load tokenizer.json (+ added tokens) from a model directory.
 * Returns NULL on error (parse failure, missing files, OOM). */
tok_t *tok_load(const char *dir);

/* Free the tokenizer and all lookup structures. */
void tok_free(tok_t *t);

/* Encode UTF-8 text to token ids. When `special` is true, special-token
 * strings (e.g. <|begin_of_text|>) are matched atomically via the trie;
 * otherwise they are encoded as plain text — matching the HF reference
 * default (encode_special_tokens = False). Fills *out (malloc'd, caller
 * frees) and *nout. Returns false on error. */
bool tok_encode(const tok_t *t, const char *text, size_t len, bool special,
                int64_t **out, size_t *nout);

/* Decode token ids to UTF-8 text. Fills *out (malloc'd, caller frees,
 * NUL-padded) and *outlen. Returns false on error. */
bool tok_decode(const tok_t *t, const int64_t *ids, size_t n,
                char **out, size_t *outlen);

#endif /* AMP_TOKENIZER_H */