// amp_tokenizer.cpp — ucode glue for the native tokenizer engine.
//
//   import * as torch from 'torch';
//   let t = torch.tokenizer('/path/to/model-dir');
//   let ids = t.encode('Hello');              // int array
//   let text = t.decode(ids);                 // string
//
// The engine (tokenizer.c) is pure C; this file only marshals values.
// encode() returns a preallocated ucode array: ucv_array_new_length(n)
// reserves capacity >= n+1 with count == 0, setting the LAST slot first
// grows count to n in one step without reallocating; the remaining slots
// are then plain overwrites (ucv_array_set on index < count only puts the
// old value, and untouched slots are NULL, where ucv_put(NULL) is a
// no-op). No intermediate reallocations.

#include <cstring>

extern "C" {
#include <ucode/module.h>
#include <ucode/vm.h>
#include "tokenizer.h"
}

#define TOKENIZER_TYPE "torch.tokenizer"

/* Coerce an argument to bool (booleans and numbers). */
static bool
arg_bool(uc_vm_t *vm, uc_value_t *v, bool *out)
{
	uc_type_t t = ucv_type(v);

	if (t == UC_BOOLEAN) {
		*out = ucv_boolean_get(v);
		return true;
	}

	if (t == UC_INTEGER || t == UC_DOUBLE) {
		*out = ucv_type(v) == UC_INTEGER ?
			(double)ucv_int64_get(v) : ucv_double_get(v);
		return true;
	}

	uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a boolean");

	return false;
}

static bool
res_is_type(uc_value_t *res, const char *want)
{
	if (ucv_type(res) != UC_RESOURCE)
		return false;

	uc_resource_type_t *rt = ucv_resource_type(res);

	return rt && !strcmp(rt->name, want);
}

static void
tokenizer_free(void *data)
{
	tok_t *t = static_cast<tok_t *>(data);

	if (t) {
		tok_free(t);
	}
}

/* Fetch the tok_t for `this`, or raise and return NULL. */
static tok_t *
tokenizer_this(uc_vm_t *vm)
{
	uc_value_t *res = _uc_fn_this_res(vm);
	tok_t *t = NULL;

	if (res && res_is_type(res, TOKENIZER_TYPE))
		t = static_cast<tok_t *>(ucv_resource_data(res, TOKENIZER_TYPE));

	if (!t)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"called on non-tokenizer or freed value");

	return t;
}

/*
 * tokenizer(path: string): torch.tokenizer
 *
 * Loads tokenizer.json from a model directory.
 */
static uc_value_t *
tokenizer_load(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *path = uc_fn_arg(0);

	if (nargs != 1 || ucv_type(path) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a path string");
		return NULL;
	}

	tok_t *t = tok_load(ucv_string_get(path));

	if (!t) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"cannot load tokenizer.json");
		return NULL;
	}

	return uc_resource_new(ucv_resource_type_lookup(vm, TOKENIZER_TYPE), t);
}

/*
 * encode(text: string, special?: boolean): number[]
 *
 * Encodes UTF-8 text to token ids (byte-level BPE with the GPT-2 split
 * regex). With `special` (default false), special-token strings such as
 * <|begin_of_text|> are matched atomically; otherwise they are encoded
 * as plain text, matching the HF reference default.
 */
static uc_value_t *
tokenizer_encode(uc_vm_t *vm, size_t nargs)
{
	tok_t *t = tokenizer_this(vm);
	uc_value_t *text = uc_fn_arg(0);
	bool special = false;
	int64_t *ids = NULL;
	size_t nids = 0;

	if (!t)
		return NULL;

	if (nargs < 1 || nargs > 2 || ucv_type(text) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a string");
		return NULL;
	}

	if (nargs == 2 && !arg_bool(vm, uc_fn_arg(1), &special))
		return NULL;

	const char *s = ucv_string_get(text);
	size_t len = ucv_string_length(text);

	if (!tok_encode(t, s, len, special, &ids, &nids)) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "encode failed");
		return NULL;
	}

	/* preallocate exactly nids slots; see the file comment for why the
	 * last slot is written first */
	uc_value_t *arr = ucv_array_new_length(vm, nids);

	if (nids > 0) {
		ucv_array_set(arr, nids - 1, ucv_int64_new(ids[nids - 1]));

		for (size_t i = 0; i + 1 < nids; i++)
			ucv_array_set(arr, i, ucv_int64_new(ids[i]));
	}

	free(ids);

	return arr;
}

/*
 * decode(ids: number[]): string
 *
 * Decodes token ids back to UTF-8 text.
 */
static uc_value_t *
tokenizer_decode(uc_vm_t *vm, size_t nargs)
{
	tok_t *t = tokenizer_this(vm);
	uc_value_t *arg = uc_fn_arg(0);
	char *text = NULL;
	size_t tlen = 0;

	if (!t)
		return NULL;

	if (nargs != 1 || ucv_type(arg) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected an array");
		return NULL;
	}

	size_t n = ucv_array_length(arg);
	int64_t *ids = static_cast<int64_t *>(malloc(n ? n * sizeof(*ids) : 1));

	if (!ids)
		return NULL;

	for (size_t i = 0; i < n; i++) {
		uc_value_t *v = ucv_array_get(arg, i);

		if (ucv_type(v) != UC_INTEGER) {
			free(ids);
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"expected an array of integers");
			return NULL;
		}

		ids[i] = ucv_int64_get(v);
	}

	if (!tok_decode(t, ids, n, &text, &tlen)) {
		free(ids);
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "decode failed");
		return NULL;
	}

	free(ids);

	uc_value_t *rv = ucv_string_new_length(text, tlen);

	free(text);

	return rv;
}

/*
 * free(): release the tokenizer early (the GC finalizer does the same).
 */
static uc_value_t *
tokenizer_free_method(uc_vm_t *vm, size_t nargs)
{
	void **slot = ucv_resource_dataptr(_uc_fn_this_res(vm), TOKENIZER_TYPE);

	if (slot && *slot) {
		tokenizer_free(static_cast<tok_t *>(*slot));
		*slot = NULL;
	}

	return NULL;
}

static const uc_function_list_t tokenizer_methods[] = {
	{ "encode",	tokenizer_encode },
	{ "decode",	tokenizer_decode },
	{ "free",	tokenizer_free_method },
};

static const uc_function_list_t tokenizer_fns[] = {
	{ "tokenizer",	tokenizer_load },
};

void
amp_tokenizer_init(uc_vm_t *vm, uc_value_t *scope)
{
	uc_type_declare(vm, TOKENIZER_TYPE, tokenizer_methods, tokenizer_free);
	uc_function_list_register(scope, tokenizer_fns);
}