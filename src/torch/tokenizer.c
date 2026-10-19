// tokenizer.c — HF tokenizer.json byte-level BPE engine (pure C).
//
// Implements the reference `tokenizers` behaviour for the llama-3 family:
//
//   1. pretokenize: GPT-2 split regex (via PCRE2, UTF-8 + UCP modes)
//   2. byte-level: map UTF-8 bytes through the GPT-2 byte<->unicode table
//   3. BPE: greedy lowest-rank pair merges per piece
//   4. vocab lookup: piece -> id
//
// The JSON is parsed with json-c (already a dependency via st.c); the
// parsed tree is walked once to build compact lookup structures, then
// freed. PCRE2 is linked directly (no dlopen).
//
// See tokenizer.h for the interface.

#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <json-c/json.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- GPT-2 byte <-> unicode map ---------------------------------------- */

static void
byte_map_init(unsigned short map[256], unsigned char inv[324])
{
	size_t n = 0;

	memset(inv, 0, 324);

	for (int b = 0; b < 256; b++) {
		if (b >= 33 && b <= 126) { map[b] = (unsigned char)b; continue; }
		if (b >= 161 && b <= 172) { map[b] = (unsigned char)b; continue; }
		if (b >= 174 && b <= 255) { map[b] = (unsigned char)b; continue; }
		n++;
	}

	/* second pass: assign the remaining bytes to 256+n in order */
	n = 0;

	for (int b = 0; b < 256; b++) {
		if (b >= 33 && b <= 126) continue;
		if (b >= 161 && b <= 172) continue;
		if (b >= 174 && b <= 255) continue;
		map[b] = (unsigned short)(256 + n);
		n++;
	}

	for (int b = 0; b < 256; b++)
		inv[map[b]] = (unsigned char)b;
}

/* --- open-addressing hash table ---------------------------------------- */
/* key = "A B" concatenation with alen bytes of A up front; for the vocab
 * table blen == 0 and the key is just the piece. */

typedef struct {
	char    *key;      /* owned, NUL-padded */
	size_t   alen;     /* length of A (== whole key for vocab) */
	size_t   blen;     /* length of B (0 for vocab) */
	int64_t  value;    /* vocab id or merge rank */
	uint64_t hash;
} ht_entry_t;

typedef struct {
	ht_entry_t *slots;     /* key == NULL: empty slot */
	size_t      cap;       /* power of two */
	size_t      len;
} ht_t;

static bool
ht_grow(ht_t *t);

static uint64_t
ht_hash(const char *a, size_t alen, const char *b, size_t blen)
{
	uint64_t h = 0xcbf29ce484222325ULL;

	for (size_t i = 0; i < alen; i++) {
		h ^= (unsigned char)a[i];
		h *= 0x100000001b3ULL;
	}

	h ^= 0x9e3779b97f4a7c15ULL;   /* separator */
	h *= 0x100000001b3ULL;

	for (size_t i = 0; i < blen; i++) {
		h ^= (unsigned char)b[i];
		h *= 0x100000001b3ULL;
	}

	return h;
}

static bool
ht_init(ht_t *t, size_t n)
{
	size_t cap = 16;

	while (cap < n * 2)
		cap <<= 1;

	t->slots = calloc(cap, sizeof(*t->slots));

	if (!t->slots)
		return false;

	t->cap = cap;
	t->len = 0;

	return true;
}

static void
ht_free(ht_t *t)
{
	if (!t->slots)
		return;

	for (size_t i = 0; i < t->cap; i++)
		free(t->slots[i].key);

	free(t->slots);
	t->slots = NULL;
	t->cap = 0;
	t->len = 0;
}

/* Find the slot for (a, b). Returns true and fills *out if present;
 * returns false and fills *out with the empty slot index otherwise. */
static bool
ht_find(const ht_t *t, const char *a, size_t alen, const char *b, size_t blen,
        uint64_t h, size_t *out)
{
	size_t len = alen + blen;
	size_t i = h & (t->cap - 1);

	while (t->slots[i].key) {
		if (t->slots[i].hash == h && t->slots[i].alen == alen &&
		    t->slots[i].blen == blen && t->slots[i].key[len] == '\0' &&
		    !memcmp(t->slots[i].key, a, alen) &&
		    !memcmp(t->slots[i].key + alen, b, blen)) {
			*out = i;
			return true;
		}

		i = (i + 1) & (t->cap - 1);
	}

	*out = i;

	return false;
}

static bool
ht_put(ht_t *t, const char *a, size_t alen, const char *b, size_t blen,
       int64_t value)
{
	if (t->len * 2 >= t->cap && !ht_grow(t))
		return false;

	size_t len = alen + blen;
	uint64_t h = ht_hash(a, alen, b, blen);
	size_t i;

	if (ht_find(t, a, alen, b, blen, h, &i)) {
		t->slots[i].value = value;   /* replace */
		return true;
	}

	t->slots[i].key = malloc(len + 1);

	if (!t->slots[i].key)
		return false;

	memcpy(t->slots[i].key, a, alen);
	memcpy(t->slots[i].key + alen, b, blen);
	t->slots[i].key[len] = '\0';
	t->slots[i].alen = alen;
	t->slots[i].blen = blen;
	t->slots[i].value = value;
	t->slots[i].hash = h;
	t->len++;

	return true;
}

static bool
ht_get(const ht_t *t, const char *a, size_t alen, const char *b, size_t blen,
       int64_t *out)
{
	uint64_t h = ht_hash(a, alen, b, blen);
	size_t i;

	if (!ht_find(t, a, alen, b, blen, h, &i))
		return false;

	*out = t->slots[i].value;

	return true;
}

static bool
ht_grow(ht_t *t)
{
	ht_t n = { 0 };

	if (!ht_init(&n, t->len * 2 + 1))
		return false;

	for (size_t i = 0; i < t->cap; i++) {
		if (t->slots[i].key) {
			if (!ht_put(&n, t->slots[i].key, t->slots[i].alen,
			            t->slots[i].key + t->slots[i].alen,
			            t->slots[i].blen, t->slots[i].value)) {
				ht_free(&n);
				return false;
			}
		}
	}

	free(t->slots);
	*t = n;

	return true;
}

/* --- special-token trie --------------------------------------------------- */

/* Per-position longest-match trie over the special pieces.
 *
 * The reference implementation runs an Aho-Corasick automaton over the
 * input; we get the same longest-match semantics with a plain trie walk
 * restarted at every input position. That is O(len * max_piece_len)
 * worst case — special pieces are short (<= ~30 bytes), so effectively
 * linear — and needs no failure links. Children are kept as small
 * byte-sorted arrays: with the few hundred special pieces a model
 * carries, each lookup is one or two binary-search probes, and shared
 * prefixes (llama-3's <| family) cost one probe for the whole run.
 */

typedef struct st_node {
	struct st_node *kids;      /* byte-sorted child array */
	uint16_t        nkids;
	uint16_t        cap;
	unsigned char   byte;
	int64_t         id;        /* >= 0: terminal, token id */
} st_node_t;

static st_node_t *
st_find(const st_node_t *n, unsigned char b)
{
	size_t lo = 0, hi = n->nkids;

	while (lo < hi) {
		size_t mid = (lo + hi) / 2;

		if (n->kids[mid].byte < b)
			lo = mid + 1;
		else if (n->kids[mid].byte > b)
			hi = mid;
		else
			return &n->kids[mid];
	}

	return NULL;
}

static st_node_t *
st_insert(st_node_t *n, unsigned char b)
{
	size_t lo = 0, hi = n->nkids;

	while (lo < hi) {
		size_t mid = (lo + hi) / 2;

		if (n->kids[mid].byte < b)
			lo = mid + 1;
		else if (n->kids[mid].byte > b)
			hi = mid;
		else
			return &n->kids[mid];
	}

	if (n->nkids == n->cap) {
		size_t ncap = n->cap ? n->cap * 2 : 4;
		st_node_t *nk = realloc(n->kids, ncap * sizeof(*nk));

		if (!nk)
			return NULL;

		n->kids = nk;
		n->cap = (uint16_t)ncap;
	}

	memmove(&n->kids[lo + 1], &n->kids[lo],
		(n->nkids - lo) * sizeof(*n->kids));
	memset(&n->kids[lo], 0, sizeof(*n->kids));
	n->kids[lo].byte = b;
	n->kids[lo].id = -1;
	n->nkids++;

	return &n->kids[lo];
}

static void
st_free(st_node_t *n)
{
	for (size_t i = 0; i < n->nkids; i++)
		st_free(&n->kids[i]);

	free(n->kids);
	n->kids = NULL;
	n->nkids = 0;
	n->cap = 0;
}

/* Insert one special piece into the trie. Empty pieces are skipped
 * (they would match at every position). */
static bool
st_add(st_node_t *root, const char *piece, size_t len, int64_t id)
{
	if (!len)
		return true;

	st_node_t *n = root;

	for (size_t i = 0; i < len; i++) {
		n = st_insert(n, (unsigned char)piece[i]);

		if (!n)
			return false;
	}

	n->id = id;

	return true;
}

/* --- tokenizer state ----------------------------------------------------- */

struct st_node;               /* special-token trie node (defined below) */

#define TOK_MAX_STAGES 8

struct tok {
	ht_t          vocab;    /* piece -> id */
	ht_t          merges;   /* "A B" -> rank */
	ht_t          special;  /* special piece -> id (encode atomization) */
	ht_t          added;    /* ALL added tokens -> id (decode, rev) */
	char        **rev;      /* id -> piece (borrowed vocab keys) */
	int64_t       rev_max;  /* highest id in rev */
	struct st_node st_root; /* special-token trie (empty root) */
	struct st_node ad_root; /* non-special added tokens (always atomic) */
	unsigned short byte_map[256];  /* byte -> codepoint (up to 323) */
	unsigned char  byte_inv[324];  /* codepoint -> byte */
	pcre2_code   *re[TOK_MAX_STAGES];  /* pre-tokenizer split stages, applied in
	                                    * order (HF Sequence of Split/Isolated) */
	int           nre;
	uint32_t      re_size;
	bool          ignore_merges;   /* model.ignore_merges: emit a whole
	                                * pre-token straight from the vocab when
	                                * it is a single entry, skipping merges */
	bool          spm;      /* SentencePiece-style BPE (Gemma): spaces become U+2581, merges run over the whole
	                         * text of a run (no pre-tokenizer), unknown characters fall back to <0xXX> byte tokens */
	int64_t       byte_tok[256];   /* spm: the <0xXX> token ids (-1: none) */
};

/* --- json-c loading ------------------------------------------------------ */

static bool
ht_from_object(ht_t *t, struct json_object *obj, int64_t offset)
{
	size_t n = json_object_object_length(obj);

	if (!ht_init(t, n))
		return false;

	struct json_object_iterator it = json_object_iter_begin(obj);
	struct json_object_iterator end = json_object_iter_end(obj);

	while (!json_object_iter_equal(&it, &end)) {
		const char *key = json_object_iter_peek_name(&it);
		struct json_object *val = json_object_iter_peek_value(&it);

		if (!ht_put(t, key, strlen(key), "", 0,
		            offset + json_object_get_int64(val)))
			goto err;

		json_object_iter_next(&it);
	}

	return true;

err:
	ht_free(t);

	return false;
}

static bool
merges_from_array(struct tok *t, struct json_object *arr)
{
	size_t n = json_object_array_length(arr);

	if (!ht_init(&t->merges, n))
		return false;

	for (size_t i = 0; i < n; i++) {
		struct json_object *m = json_object_array_get_idx(arr, i);
		const char *a, *b;
		size_t alen, blen;

		/* two encodings exist in the wild: the classic "a b" string
		 * and (newer tokenizers releases) a ["a", "b"] pair */
		if (json_object_is_type(m, json_type_array)) {
			if (json_object_array_length(m) != 2)
				goto err;

			a = json_object_get_string(
				json_object_array_get_idx(m, 0));
			b = json_object_get_string(
				json_object_array_get_idx(m, 1));

			if (!a || !b)
				goto err;

			alen = strlen(a);
			blen = strlen(b);
		} else {
			const char *s = json_object_get_string(m);
			const char *sp;

			if (!s)
				goto err;

			sp = strchr(s, ' ');

			if (!sp)
				goto err;

			a = s;
			alen = (size_t)(sp - s);
			b = sp + 1;
			blen = strlen(b);
		}

		if (!ht_put(&t->merges, a, alen, b, blen, (int64_t)i))
			goto err;
	}

	return true;

err:
	ht_free(&t->merges);

	return false;
}

static bool
added_tokens_from_array(struct tok *t, struct json_object *arr)
{
	size_t n = json_object_array_length(arr);

	if (!n)
		return true;

	if (!ht_init(&t->added, n))
		return false;

	if (!ht_init(&t->special, n)) {
		ht_free(&t->added);
		return false;
	}

	for (size_t i = 0; i < n; i++) {
		struct json_object *at = json_object_array_get_idx(arr, i);
		struct json_object *id, *special, *content;
		const char *piece;

		if (!json_object_object_get_ex(at, "id", &id) ||
		    !json_object_object_get_ex(at, "content", &content))
			continue;

		piece = json_object_get_string(content);

		/* every added token is a real id: it must decode verbatim and
		 * appear in the reverse table. Only special=true ones join the
		 * encode-side special table (atomization pre-scan). */
		if (!ht_put(&t->added, piece, strlen(piece), "", 0,
		            json_object_get_int64(id)))
			goto err;

		if (json_object_object_get_ex(at, "special", &special) &&
		    !json_object_get_boolean(special))
			continue;

		if (!ht_put(&t->special, piece, strlen(piece), "", 0,
		            json_object_get_int64(id)))
			goto err;
	}

	return true;

err:
	ht_free(&t->added);
	ht_free(&t->special);

	return false;
}

static pcre2_code *
compile_re(const char *pattern, size_t plen)
{
	int ecode;
	PCRE2_SIZE eoff;
	pcre2_code *re;
	uint32_t size;

	re = pcre2_compile((PCRE2_SPTR)pattern, plen,
		PCRE2_UTF | PCRE2_UCP, &ecode, &eoff, NULL);

	if (!re)
		return NULL;

	pcre2_pattern_info(re, PCRE2_INFO_SIZE, &size);

	/* JIT is an optimisation only: a failed compile (unsupported platform
	 * or pattern) leaves the interpreter in place */
	(void)pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);

	return re;
}

/* Built-in fallback: the GPT-2 split pattern (the ByteLevel
 * pre-tokenizer's own regex). */
static pcre2_code *
compile_split_re(void)
{
	static const char pattern[] =
		"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
		"\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
		"\\s+(?!\\S)|\\s+";

	return compile_re(pattern, sizeof(pattern) - 1);
}

/* Extract the split regex from one pre-tokenizer stage object. Returns:
 *   1  a usable regex was compiled into *out
 *   0  the stage contributes no splitting (ByteLevel with use_regex
 *      false) — the caller keeps looking
 *  -1  the stage is not something this engine can reproduce
 */
static int
resolve_stage(struct json_object *item, pcre2_code **out)
{
	struct json_object *pat, *re_str, *use;

	if (!json_object_object_get_ex(item, "type", &pat) ||
	    !json_object_is_type(pat, json_type_string))
		return -1;

	const char *ty = json_object_get_string(pat);

	/* ByteLevel: its own regex IS the GPT-2 pattern (use_regex
	 * defaults to true); with use_regex false it does not split */
	if (!strcmp(ty, "ByteLevel")) {
		if (json_object_object_get_ex(item, "use_regex", &use) &&
		    !json_object_get_boolean(use))
			return 0;

		if (!(*out = compile_split_re()))
			return -1;

		return 1;
	}

	if (strcmp(ty, "Split"))
		return -1;

	/* only the Isolated behavior (matches and the gaps between them
	 * become pieces) is implemented */
	if (json_object_object_get_ex(item, "behavior", &use) &&
	    strcmp(json_object_get_string(use), "Isolated"))
		return -1;

	if (json_object_object_get_ex(item, "invert", &use) &&
	    json_object_get_boolean(use))
		return -1;

	if (!json_object_object_get_ex(item, "pattern", &pat) ||
	    !json_object_object_get_ex(pat, "Regex", &re_str) ||
	    !json_object_is_type(re_str, json_type_string))
		return -1;

	if (!(*out = compile_re(json_object_get_string(re_str),
	    strlen(json_object_get_string(re_str)))))
		return -1;

	return 1;
}

/* Resolve the pre-tokenizer split regex from tokenizer.json. Returns:
 *   1  and sets *out — a regex was resolved
 *   0  no pre_tokenizer in the file — caller falls back to built-in
 *  -1  unsupported pre_tokenizer — the caller must FAIL the load
 *      (silently substituting the GPT-2 pattern would mis-encode
 *      every Metaspace-style SentencePiece model)
 */
static int
resolve_split_re(struct json_object *root, pcre2_code **out, int *nout)
{
	struct json_object *pt, *seq;

	*nout = 0;

	if (!json_object_object_get_ex(root, "pre_tokenizer", &pt))
		return 0;

	if (!json_object_is_type(pt, json_type_object))
		return -1;

	if (json_object_object_get_ex(pt, "pretokenizers", &seq) &&
	    json_object_is_type(seq, json_type_array)) {
		size_t n = json_object_array_length(seq);

		/* every splitting stage, in order: HF applies a Sequence stage
		 * by stage, each one splitting the pieces of the previous */
		for (size_t i = 0; i < n; i++) {
			struct json_object *item =
				json_object_array_get_idx(seq, i);
			int rc;

			if (!item)
				continue;

			if (*nout >= TOK_MAX_STAGES)
				goto fail;

			rc = resolve_stage(item, &out[*nout]);

			if (rc < 0)
				goto fail;

			if (rc == 1)
				(*nout)++;
		}

		/* a Sequence of only non-splitting stages: this engine
		 * always splits, so that is unsupported too */
		if (*nout == 0)
			return -1;

		return 1;

fail:
		for (int i = 0; i < *nout; i++)
			pcre2_code_free(out[i]);

		*nout = 0;

		return -1;
	}

	int rc = resolve_stage(pt, &out[0]);

	if (rc == 1)
		*nout = 1;

	return rc;
}

tok_t *
tok_load(const char *dir)
{
	char path[4096];
	struct json_object *root = NULL, *model, *v;
	tok_t *t = NULL;
	int fdlen = snprintf(path, sizeof(path), "%s/tokenizer.json", dir);

	if (fdlen < 0 || (size_t)fdlen >= sizeof(path))
		return NULL;

	root = json_object_from_file(path);

	if (!root)
		return NULL;

	t = calloc(1, sizeof(*t));

	if (!t)
		goto err;

	if (!json_object_object_get_ex(root, "model", &model))
		goto err;

	/* this engine implements byte-level BPE only; anything else
	 * (Unigram, WordPiece) must fail loudly, not mis-encode */
	if (!json_object_object_get_ex(model, "type", &v) ||
	    !json_object_is_type(v, json_type_string) ||
	    strcmp(json_object_get_string(v), "BPE"))
		goto err;

	if (!json_object_object_get_ex(model, "vocab", &v) ||
	    !ht_from_object(&t->vocab, v, 0))
		goto err;

	if (json_object_object_get_ex(model, "merges", &v) &&
	    !merges_from_array(t, v))
		goto err;

	if (json_object_object_get_ex(model, "ignore_merges", &v))
		t->ignore_merges = json_object_get_boolean(v);

	/* byte-fallback BPE without a byte-level pre-tokenizer: the SentencePiece family (Gemma) */
	if (json_object_object_get_ex(model, "byte_fallback", &v) && json_object_get_boolean(v)) {
		struct json_object *nz, *nt, *pat, *cont;

		if (!json_object_object_get_ex(root, "normalizer", &nz) || !json_object_is_type(nz, json_type_object) ||
		    !json_object_object_get_ex(nz, "type", &nt) || strcmp(json_object_get_string(nt), "Replace") ||
		    !json_object_object_get_ex(nz, "pattern", &pat) || !json_object_object_get_ex(pat, "String", &pat) ||
		    strcmp(json_object_get_string(pat), " ") || !json_object_object_get_ex(nz, "content", &cont) ||
		    strcmp(json_object_get_string(cont), "\xe2\x96\x81"))
			goto err;

		t->spm = true;
	}

	if (json_object_object_get_ex(root, "added_tokens", &v) &&
	    !added_tokens_from_array(t, v))
		goto err;

	byte_map_init(t->byte_map, t->byte_inv);

	if (t->spm)
		for (int b = 0; b < 256; b++) {
			char nm[8];
			int64_t id;

			snprintf(nm, sizeof(nm), "<0x%02X>", b);
			t->byte_tok[b] = ht_get(&t->vocab, nm, 6, "", 0, &id) ? id : -1;
		}

	/* reverse table: id -> piece, borrowed from the vocab keys */
	int64_t max_id = -1;

	for (size_t i = 0; i < t->vocab.cap; i++) {
		if (t->vocab.slots[i].key && t->vocab.slots[i].value > max_id)
			max_id = t->vocab.slots[i].value;
	}

	for (size_t i = 0; i < t->special.cap; i++) {
		if (t->special.slots[i].key && t->special.slots[i].value > max_id)
			max_id = t->special.slots[i].value;
	}

	for (size_t i = 0; i < t->added.cap; i++) {
		if (t->added.slots[i].key && t->added.slots[i].value > max_id)
			max_id = t->added.slots[i].value;
	}

	if (max_id >= 0) {
		t->rev = calloc((size_t)max_id + 1, sizeof(*t->rev));

		if (!t->rev)
			goto err;

		t->rev_max = max_id;

		for (size_t i = 0; i < t->vocab.cap; i++) {
			if (t->vocab.slots[i].key)
				t->rev[t->vocab.slots[i].value] = t->vocab.slots[i].key;
		}

		for (size_t i = 0; i < t->special.cap; i++) {
			if (t->special.slots[i].key)
				t->rev[t->special.slots[i].value] = t->special.slots[i].key;
		}

		for (size_t i = 0; i < t->added.cap; i++) {
			if (t->added.slots[i].key)
				t->rev[t->added.slots[i].value] = t->added.slots[i].key;
		}
	}

	/* special-token trie for the encode pre-scan */
	for (size_t i = 0; i < t->special.cap; i++) {
		if (t->special.slots[i].key &&
		    !st_add(&t->st_root, t->special.slots[i].key,
		            t->special.slots[i].alen, t->special.slots[i].value))
			goto err;
	}

	/* non-special added tokens (e.g. Qwen's <think>, <tool_call>): HF
	 * tokenizers extract every added token atomically, special or not;
	 * the special flag only governs the special-token switch */
	for (size_t i = 0; i < t->added.cap; i++) {
		int64_t sid;

		if (!t->added.slots[i].key ||
		    ht_get(&t->special, t->added.slots[i].key,
		           t->added.slots[i].alen, "", 0, &sid))
			continue;

		if (!st_add(&t->ad_root, t->added.slots[i].key,
		            t->added.slots[i].alen, t->added.slots[i].value))
			goto err;
	}

	/* pre-tokenizer regex from the config. No pre_tokenizer at all
	 * falls back to the built-in GPT-2 pattern; an UNSUPPORTED one
	 * (Metaspace, Whitespace, ...) fails the load — silently
	 * substituting GPT-2 would mis-encode every piece. */
	if (t->spm) {
		json_object_put(root);

		return t;
	}

	switch (resolve_split_re(root, t->re, &t->nre)) {
	case 1:
		break;

	case 0:
		t->re[0] = compile_split_re();
		t->nre = t->re[0] ? 1 : 0;
		break;

	default:
		goto err;
	}

	if (t->nre < 1)
		goto err;

	json_object_put(root);

	return t;

err:
	if (root)
		json_object_put(root);

	tok_free(t);

	return NULL;
}

void
tok_free(tok_t *t)
{
	if (!t)
		return;

	ht_free(&t->vocab);
	ht_free(&t->merges);
	ht_free(&t->special);
	ht_free(&t->added);
	st_free(&t->st_root);
	st_free(&t->ad_root);
	free(t->rev);

	for (int i = 0; i < t->nre; i++)
		pcre2_code_free(t->re[i]);

	free(t);
}

/* --- pretokenize --------------------------------------------------------- */

/* Split `text` (UTF-8, len bytes) into pieces at the regex boundaries.
 * Fills *pieces (malloc'd array of {offset, len}) and *npieces. */
typedef struct {
	size_t off;
	size_t len;
} span_t;

static bool
span_push(span_t **spans, size_t *nspans, size_t *cap, size_t off, size_t len)
{
	if (*nspans == *cap) {
		size_t ncap = *cap ? *cap * 2 : 16;
		span_t *ns = realloc(*spans, ncap * sizeof(*ns));

		if (!ns)
			return false;

		*spans = ns;
		*cap = ncap;
	}

	(*spans)[*nspans].off = off;
	(*spans)[*nspans].len = len;
	(*nspans)++;

	return true;
}

/* Split one piece text[off, off + len) at the matches of `re` (HF Split /
 * Isolated: the matches and the gaps between them all become pieces),
 * appending to *spans. The regex sees the piece as its own subject. */
static bool
split_piece(const pcre2_code *re, pcre2_match_data *md, pcre2_match_context *mc,
            const char *text, size_t off, size_t len, span_t **spans,
            size_t *nspans, size_t *cap)
{
	const char *sub = text + off;
	size_t start = 0;
	uint32_t flags = 0;

	while (start < len) {
		/* PCRE2 validates the WHOLE subject on every call that lacks
		 * NO_UTF_CHECK: one check per subject, not one per match (that is
		 * quadratic: ~0.5 s for 100 KB of text) */
		int rc = pcre2_match(re, (PCRE2_SPTR)sub, len, start, flags, md, mc);

		flags = PCRE2_NO_UTF_CHECK;

		if (rc < 0) {
			if (rc != PCRE2_ERROR_NOMATCH)
				return false;

			/* trailing gap */
			return span_push(spans, nspans, cap, off + start, len - start);
		}

		PCRE2_SIZE *ov = pcre2_get_ovector_pointer(md);
		size_t s = ov[0], e = ov[1];

		if (s > start && !span_push(spans, nspans, cap, off + start, s - start))
			return false;

		if (e == s) {
			/* empty match: consume one byte to make progress */
			e = s + 1;
		}

		if (!span_push(spans, nspans, cap, off + s, e - s))
			return false;

		start = e;
	}

	return true;
}

static bool
pretokenize(const tok_t *t, const char *text, size_t len,
            span_t **out, size_t *nout)
{
	pcre2_match_data *md = pcre2_match_data_create(2, NULL);
	pcre2_match_context *mc = pcre2_match_context_create(NULL);
	pcre2_jit_stack *js = pcre2_jit_stack_create(32 * 1024, 16 * 1024 * 1024, NULL);
	span_t *cur = NULL, *next = NULL;
	size_t ncur = 0, ccur = 0;

	if (!md || !mc)
		goto err;

	/* a stack for the JIT'd matcher that grows with the subject (the
	 * default 32 KB one fails on long runs) */
	if (js)
		pcre2_jit_stack_assign(mc, NULL, js);

	/* stage 0 splits the whole text, each later stage splits the pieces
	 * the previous one produced */
	if (!split_piece(t->re[0], md, mc, text, 0, len, &cur, &ncur, &ccur))
		goto err;

	for (int st = 1; st < t->nre; st++) {
		size_t nnext = 0, cnext = 0;

		for (size_t i = 0; i < ncur; i++)
			if (!split_piece(t->re[st], md, mc, text, cur[i].off, cur[i].len,
			                 &next, &nnext, &cnext)) {
				goto err;
			}

		free(cur);
		cur = next;
		ncur = nnext;
		ccur = cnext;
		next = NULL;
	}

	pcre2_match_data_free(md);
	pcre2_match_context_free(mc);
	pcre2_jit_stack_free(js);
	*out = cur;
	*nout = ncur;

	return true;

err:
	pcre2_match_data_free(md);
	pcre2_match_context_free(mc);
	pcre2_jit_stack_free(js);
	free(cur);
	free(next);

	return false;
}

/* --- BPE ------------------------------------------------------------------ */

/* Encode one piece: byte-map it to the GPT-2 unicode space (UTF-8
 * re-encoded), run the greedy merge loop, look up ids.
 *
 * buf holds the UTF-8 bytes of the mapped symbols; off[i]..off[i+1] is
 * symbol i's byte range. A merge joins symbols i and i+1 by deleting
 * the boundary off[i+1]. */
static bool
bpe_piece(const tok_t *t, const char *piece, size_t len, int64_t **ids,
          size_t *nids)
{
	/* worst case: every byte maps to a 2-byte UTF-8 sequence */
	unsigned char *buf = malloc(2 * len);
	size_t *off = malloc((len + 1) * sizeof(*off));
	size_t nsym = 0, nbuf = 0;
	int64_t *out = NULL;

	if (!buf || !off)
		goto err;

	for (size_t i = 0; i < len; i++) {
		unsigned cp = t->byte_map[(unsigned char)piece[i]];

		/* mapped codepoints are <= 323: 1-byte ASCII or 2-byte UTF-8 */
		if (cp < 0x80) {
			buf[nbuf++] = (unsigned char)cp;
		} else {
			buf[nbuf++] = (unsigned char)(0xc0 | (cp >> 6));
			buf[nbuf++] = (unsigned char)(0x80 | (cp & 0x3f));
		}

		off[++nsym] = nbuf;
	}

	off[0] = 0;

	/* model.ignore_merges (llama-3 sets it): a pre-token that is a
	 * single vocab entry is emitted as-is, bypassing merges. The
	 * lookup is on the byte-MAPPED string (HF checks the mapped word,
	 * not the raw text). This matters for vocab entries the merge
	 * path cannot reach (e.g. 'ektedir' -> one token vs three). */
	if (t->ignore_merges) {
		int64_t id;

		if (ht_get(&t->vocab, (const char *)buf, nbuf, "", 0, &id)) {
			out = malloc(sizeof(*out));

			if (!out)
				goto err;

			out[0] = id;
			*ids = out;
			*nids = 1;

			free(buf);
			free(off);

			return true;
		}
	}

	/* greedy merge loop: always apply the lowest-ranked applicable merge */
	while (nsym > 1) {
		size_t best = 0;
		int64_t best_rank = -1;

		for (size_t i = 0; i + 1 < nsym; i++) {
			int64_t rank;

			if (!ht_get(&t->merges,
			            (const char *)buf + off[i],
			            off[i + 1] - off[i],
			            (const char *)buf + off[i + 1],
			            off[i + 2] - off[i + 1], &rank))
				continue;

			if (best_rank < 0 || rank < best_rank) {
				best_rank = rank;
				best = i;
			}
		}

		if (best_rank < 0)
			break;

		/* join symbols best and best+1 */
		memmove(&off[best + 1], &off[best + 2],
			(nsym - best - 1) * sizeof(*off));
		nsym--;
	}

	out = malloc(nsym * sizeof(*out));

	if (!out)
		goto err;

	for (size_t i = 0; i < nsym; i++) {
		int64_t id;

		if (!ht_get(&t->vocab, (const char *)buf + off[i],
		            off[i + 1] - off[i], "", 0, &id))
			goto err;   /* piece not in vocab: no unk fallback yet */

		out[i] = id;
	}

	free(buf);
	free(off);
	*ids = out;
	*nids = nsym;

	return true;

err:
	free(buf);
	free(off);
	free(out);

	return false;
}

/* Longest match at text[0] in one added-token trie. Returns the length
 * and id, or false when no piece starts here. */
static bool
match_trie(const st_node_t *n, const char *text, size_t len,
           size_t *mlen, int64_t *mid)
{
	bool found = false;
	size_t i = 0;

	while (i < len) {
		n = st_find(n, (unsigned char)text[i]);

		if (!n)
			break;

		i++;

		if (n->id >= 0) {
			*mlen = i;
			*mid = n->id;
			found = true;
		}
	}

	return found;
}

/* Longest added-token match at text[0]: non-special added tokens always,
 * special tokens only when `special` is set. */
static bool
match_special(const tok_t *t, const char *text, size_t len, bool special,
              size_t *mlen, int64_t *mid)
{
	size_t l1 = 0, l2 = 0;
	int64_t i1 = 0, i2 = 0;
	bool f1 = special && match_trie(&t->st_root, text, len, &l1, &i1);
	bool f2 = match_trie(&t->ad_root, text, len, &l2, &i2);

	if (f1 && (!f2 || l1 >= l2)) {
		*mlen = l1;
		*mid = i1;
		return true;
	}

	if (f2) {
		*mlen = l2;
		*mid = i2;
		return true;
	}

	return false;
}

/* --- public API ------------------------------------------------------------ */

/* --- SentencePiece-style BPE (Gemma) ------------------------------------- */

typedef struct {
	size_t  off, len;      /* slice of the normalized text (fixed symbols: unused) */
	int     prev, next;
	int64_t fixed;         /* >= 0: a byte-fallback token that never merges */
	bool    alive;
} spm_sym_t;

typedef struct {
	int64_t rank;
	int     left;
	size_t  total;         /* len(left) + len(right) when pushed: a stale entry no longer matches */
} spm_cand_t;

static bool
spm_cand_less(const spm_cand_t *a, const spm_cand_t *b)
{
	return a->rank < b->rank || (a->rank == b->rank && a->left < b->left);
}

static void
spm_heap_push(spm_cand_t *h, size_t *n, spm_cand_t c)
{
	size_t i = (*n)++;

	h[i] = c;

	while (i > 0) {
		size_t p = (i - 1) / 2;

		if (!spm_cand_less(&h[i], &h[p]))
			break;

		spm_cand_t tmp = h[i];

		h[i] = h[p];
		h[p] = tmp;
		i = p;
	}
}

static spm_cand_t
spm_heap_pop(spm_cand_t *h, size_t *n)
{
	spm_cand_t top = h[0];

	h[0] = h[--(*n)];

	size_t i = 0;

	for (;;) {
		size_t l = 2 * i + 1, r = l + 1, m = i;

		if (l < *n && spm_cand_less(&h[l], &h[m]))
			m = l;

		if (r < *n && spm_cand_less(&h[r], &h[m]))
			m = r;

		if (m == i)
			break;

		spm_cand_t tmp = h[i];

		h[i] = h[m];
		h[m] = tmp;
		i = m;
	}

	return top;
}

/* spaces -> U+2581, one symbol per UTF-8 character (a character missing from the vocab becomes its byte tokens), then
 * the lowest-ranked merge first (ties: leftmost), over the whole run */
static bool
spm_encode_run(const tok_t *t, const char *text, size_t len, int64_t *ids, size_t *nids)
{
	char *buf = malloc(3 * len + 1);
	size_t nb = 0, nsym = 0, cap = len * 4 + 1;
	spm_sym_t *sym = malloc(cap * sizeof(*sym));
	spm_cand_t *heap = NULL;
	size_t nheap = 0, hcap = 0;
	bool ok = false;

	if (!buf || !sym)
		goto out;

	for (size_t i = 0; i < len; i++) {
		if (text[i] == ' ') {
			buf[nb++] = (char)0xe2;
			buf[nb++] = (char)0x96;
			buf[nb++] = (char)0x81;
		}
		else
			buf[nb++] = text[i];
	}

	for (size_t i = 0; i < nb; ) {
		size_t cl = 1;
		unsigned char c = (unsigned char)buf[i];

		if (c >= 0xf0)
			cl = 4;
		else if (c >= 0xe0)
			cl = 3;
		else if (c >= 0xc0)
			cl = 2;

		if (i + cl > nb)
			cl = nb - i;

		int64_t id;

		if (ht_get(&t->vocab, buf + i, cl, "", 0, &id)) {
			sym[nsym].off = i;
			sym[nsym].len = cl;
			sym[nsym].fixed = -1;
			sym[nsym].alive = true;
			nsym++;
		}
		else {
			for (size_t k = 0; k < cl; k++) {
				int64_t bt = t->byte_tok[(unsigned char)buf[i + k]];

				if (bt < 0)
					goto out;

				sym[nsym].off = i + k;
				sym[nsym].len = 1;
				sym[nsym].fixed = bt;
				sym[nsym].alive = true;
				nsym++;
			}
		}

		i += cl;
	}

	for (size_t i = 0; i < nsym; i++) {
		sym[i].prev = (int)i - 1;
		sym[i].next = (i + 1 < nsym) ? (int)i + 1 : -1;
	}

	hcap = nsym + 1;
	heap = malloc(hcap * sizeof(*heap));

	if (!heap)
		goto out;

	/* heap entries: at most one per adjacent pair at any time plus the stale ones: grow on demand */
#define SPM_TRY(L, R) do { \
		int l_ = (L), r_ = (R); \
		int64_t rk_; \
		if (l_ >= 0 && r_ >= 0 && sym[l_].fixed < 0 && sym[r_].fixed < 0 && \
		    ht_get(&t->merges, buf + sym[l_].off, sym[l_].len, buf + sym[r_].off, sym[r_].len, &rk_)) { \
			if (nheap == hcap) { \
				hcap *= 2; \
				spm_cand_t *nh_ = realloc(heap, hcap * sizeof(*heap)); \
				if (!nh_) goto out; \
				heap = nh_; \
			} \
			spm_heap_push(heap, &nheap, (spm_cand_t){rk_, l_, sym[l_].len + sym[r_].len}); \
		} \
	} while (0)

	for (size_t i = 0; i + 1 < nsym; i++)
		SPM_TRY((int)i, (int)i + 1);

	while (nheap > 0) {
		spm_cand_t c = spm_heap_pop(heap, &nheap);
		int l = c.left, r = sym[l].next;

		if (!sym[l].alive || r < 0 || !sym[r].alive || sym[l].len + sym[r].len != c.total)
			continue;

		sym[l].len += sym[r].len;
		sym[r].alive = false;
		sym[l].next = sym[r].next;

		if (sym[r].next >= 0)
			sym[sym[r].next].prev = l;

		SPM_TRY(sym[l].prev, l);
		SPM_TRY(l, sym[l].next);
	}

#undef SPM_TRY

	for (int i = 0; i >= 0 && nsym; i = sym[i].next) {
		if (sym[i].fixed >= 0) {
			ids[(*nids)++] = sym[i].fixed;
			continue;
		}

		int64_t id;

		if (!ht_get(&t->vocab, buf + sym[i].off, sym[i].len, "", 0, &id))
			goto out;

		ids[(*nids)++] = id;
	}

	ok = true;

out:
	free(buf);
	free(sym);
	free(heap);

	return ok;
}

/* Pretokenize + BPE one run of plain text, appending to ids/nids. */
static bool
encode_run(const tok_t *t, const char *text, size_t len, int64_t *ids,
           size_t *nids)
{
	span_t *spans = NULL;
	size_t nspans = 0;
	bool ok = false;

	if (t->spm)
		return spm_encode_run(t, text, len, ids, nids);

	if (!pretokenize(t, text, len, &spans, &nspans))
		return false;

	for (size_t i = 0; i < nspans; i++) {
		const char *piece = text + spans[i].off;
		size_t plen = spans[i].len;
		int64_t *pids = NULL;
		size_t npids = 0;

		if (!bpe_piece(t, piece, plen, &pids, &npids))
			goto err;

		memcpy(ids + *nids, pids, npids * sizeof(*pids));
		*nids += npids;
		free(pids);
	}

	ok = true;

err:
	free(spans);

	return ok;
}

bool
tok_encode(const tok_t *t, const char *text, size_t len, bool special,
           int64_t **out, size_t *nout)
{
	int64_t *ids = NULL;
	size_t nids = 0;

	/* >= 1 id per byte: BPE never yields more ids than bytes, and
	 * special tokens are longer than their single id */
	ids = malloc((len ? len : 1) * sizeof(*ids));

	if (!ids)
		return false;

	/* Walk the input, matching special tokens (longest match wins) when
	 * `special` is set; plain-text runs between them are flushed through
	 * pretokenize + BPE as they close, preserving text order. With
	 * `special` false the whole input is one run and special strings go
	 * through the BPE as plain text — the HF reference default. */
	size_t run = 0;

	for (size_t i = 0; i < len; ) {
		size_t mlen = 0;
		int64_t mid = 0;

		if (match_special(t, text + i, len - i, special, &mlen, &mid)) {
			if (i > run &&
			    !encode_run(t, text + run, i - run, ids, &nids))
				goto err;

			ids[nids++] = mid;
			i += mlen;
			run = i;
			continue;
		}

		i++;
	}

	if (len > run && !encode_run(t, text + run, len - run, ids, &nids))
		goto err;

	*out = ids;
	*nout = nids;

	return true;

err:
	free(ids);

	return false;
}

bool
tok_decode(const tok_t *t, const int64_t *ids, size_t n, char **out,
           size_t *outlen)
{
	/* a single BPE token can decode to many bytes (e.g. "aaaaaaaa"),
	 * so size the buffer exactly: each token contributes at most
	 * strlen(piece) bytes (one byte per mapped codepoint, and every
	 * codepoint costs at least one UTF-8 byte in the piece) */
	size_t cap = 1;

	for (size_t i = 0; i < n; i++) {
		int64_t id = ids[i];

		if (id < 0 || id > t->rev_max || !t->rev[id])
			return false;

		cap += strlen(t->rev[id]);
	}

	unsigned char *buf = malloc(cap);
	size_t len = 0;

	if (!buf)
		return false;

	if (t->spm) {
		/* SentencePiece decoder: U+2581 -> space, <0xXX> -> the byte, special pieces verbatim */
		for (size_t i = 0; i < n; i++) {
			const char *piece = t->rev[ids[i]];
			size_t plen = strlen(piece);
			int64_t sid;
			unsigned bv;
			char tail;

			if (ht_get(&t->added, piece, plen, "", 0, &sid) && sid == ids[i]) {
				memcpy(buf + len, piece, plen);
				len += plen;
			}
			else if (plen == 6 && sscanf(piece, "<0x%2X%c", &bv, &tail) == 2 && tail == '>') {
				buf[len++] = (unsigned char)bv;
			}
			else {
				for (size_t k = 0; k < plen; ) {
					if (k + 2 < plen && (unsigned char)piece[k] == 0xe2 && (unsigned char)piece[k + 1] == 0x96 &&
					    (unsigned char)piece[k + 2] == 0x81) {
						buf[len++] = ' ';
						k += 3;
					}
					else
						buf[len++] = (unsigned char)piece[k++];
				}
			}
		}

		*out = (char *)buf;
		*outlen = len;

		return true;
	}

	for (size_t i = 0; i < n; i++) {
		int64_t id = ids[i];
		const char *piece;

		if (id < 0 || id > t->rev_max || !(piece = t->rev[id]))
			goto err;   /* unknown id: no replacement yet */

		/* special pieces decode verbatim; BPE pieces carry the byte
		 * mapping and must be unmapped through the inverse table */
		int64_t sid;

		if (ht_get(&t->added, piece, strlen(piece), "", 0, &sid) &&
		    sid == id) {
			size_t plen = strlen(piece);

			memcpy(buf + len, piece, plen);
			len += plen;
		} else {
			/* decode UTF-8 codepoints (all <= 323 here) and unmap */
			for (const unsigned char *p = (const unsigned char *)piece;
			     *p; p++) {
				unsigned cp;

				if ((*p & 0x80) == 0) {
					cp = *p;
				} else if ((*p & 0xe0) == 0xc0) {
					cp = ((*p & 0x1f) << 6) | (p[1] & 0x3f);
					p++;
				} else
					goto err;

				if (cp > 323)
					goto err;

				buf[len++] = t->byte_inv[cp];
			}
		}
	}

	*out = (char *)buf;
	*outlen = len;

	return true;

err:
	free(buf);

	return false;
}