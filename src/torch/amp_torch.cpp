// amp_torch.cpp — ucode module glue for libtorch.
//
// One module, five resource types and ~80 bindings:
//
//   - torch.tensor   at::Tensor wrapper (constructors, views, elementwise
//                    ops, reductions, in-place ops, device moves)
//   - torch.stfile   safetensors mmap reader (src/st.c) + zero-copy views
//   - torch.graph    CUDA graph capture / replay
//   - torch.comm     N-way P2P all-reduce communicator
//   - torch.notifier eventfd completion notifier (uloop handle source)
//   - torch.tokenizer  HF tokenizer.json BPE engine (src/tokenizer.c)
//   plus the fused project kernels (src/amp_kernels.cu), the vendored
//   EXL3 gemm/reconstruct bindings and the batched pinned-async H2D
//   loaders (h2d / h2d_file).
//
//   import * as torch from 'torch';
//   let t = torch.from_array([1.0, 2.0, 3.0]);
//   t.item();  t.tolist();  t.shape();  t.dtype();
//   let g = t.cuda();  g.cpu();
//
// Tensors are ucode resources ("torch.tensor") wrapping an at::Tensor*.
// The GC finalizer deletes the tensor; an explicit .free() does the same
// early and nulls the slot so double-free is impossible.

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <new>
#include <unistd.h>
#include <sys/eventfd.h>
#include <vector>
#include <map>
#include <optional>

// ucode headers are plain C and carry no extern "C" guards; without this
// wrapper the module entry points compile to C++-mangled symbols and the
// VM's dlsym("uc_module_entry") fails. (Lesson L4.)
extern "C" {
#include <ucode/module.h>
#include <ucode/vm.h>
}

#define TENSOR_TYPE "torch.tensor"
#define STFILE_TYPE "torch.stfile"
#define GRAPH_TYPE "torch.graph"
#define COMM_TYPE "torch.comm"
#define NOTIFIER_TYPE "torch.notifier"

/* ucv_resource_check() is internal to the VM; this is the public equivalent. */
static bool
res_is_type(uc_value_t *res, const char *want)
{
	/* ucv_resource_type() reads the value's struct unconditionally, so
	 * the caller must guarantee the value is a resource first. */
	if (ucv_type(res) != UC_RESOURCE)
		return false;

	uc_resource_type_t *rt = ucv_resource_type(res);

	return rt && !strcmp(rt->name, want);
}

// st.c is compiled as C; its symbols have C linkage from C++'s view.
extern "C" {
#include "st.h"
#include "tokenizer.h"
}

extern void amp_tokenizer_init(uc_vm_t *vm, uc_value_t *scope);

static uc_value_t *
amp_torch_version(uc_vm_t *vm, size_t nargs)
{
	char buf[128];

	snprintf(buf, sizeof(buf), "libtorch %s / Ampstein %s",
	         TORCH_VERSION, AMP_VERSION);

	return ucv_string_new(buf);
}

/* Type freefn. The VM calls this with:
 *   - plain resources: the stored data pointer (= the at::Tensor* itself)
 *   - extended resources: the data-area address (slot 0 = at::Tensor*)
 * All Ampstein tensors are extended resources, so `data` is always the
 * slot address here; the slot may already be NULLed by an explicit .free(). */
static void
tensor_free(void *data)
{
	if (!data)
		return;

	void *p = *(void **)data;

	if (p) {
		delete static_cast<at::Tensor *>(p);
		*(void **)data = NULL;
	}
}

/* Fetch the at::Tensor* for `this`, or raise and return NULL.
 *
 * NB: the resource kinds differ (Lesson L12):
 *   - plain resource (from_array/zeros): ucv_resource_data() returns the
 *     stored void* VALUE, i.e. the at::Tensor* itself.
 *   - extended resource (st_tensor views): ucv_resource_data() returns the
 *     data-area ADDRESS; the at::Tensor* is stored IN the first slot and
 *     must be dereferenced.
 * uc_fn_this()/uc_fn_thisval() hide this only for the plain case — using
 * them on extended resources reads the slot address as the tensor pointer,
 * which manifests as garbage tensors (dim=0, dtype=uint8) or "tensor does
 * not have a device" aborts. */
static at::Tensor *
tensor_this(uc_vm_t *vm)
{
	uc_value_t *res = _uc_fn_this_res(vm);
	at::Tensor *t = NULL;

	if (res && res_is_type(res, TENSOR_TYPE) && ucv_resource_is_extended(res)) {
		void *data = ucv_resource_data(res, TENSOR_TYPE);

		t = data ? *reinterpret_cast<at::Tensor **>(data) : NULL;
	}

	if (!t)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"called on non-tensor or freed value");

	return t;
}

/* Fetch the at::Tensor* from an arbitrary argument value (resource), or
 * NULL if the argument is null/not a tensor. `raise` controls whether a
 * type mismatch raises (true) or yields NULL (false, for optional args). */
static at::Tensor *
tensor_arg_impl(uc_vm_t *vm, uc_value_t *v, bool raise)
{
	at::Tensor *t = NULL;

	if (!v)
		return NULL;

	if (res_is_type(v, TENSOR_TYPE) && ucv_resource_is_extended(v)) {
		void *data = ucv_resource_data(v, TENSOR_TYPE);

		t = data ? *reinterpret_cast<at::Tensor **>(data) : NULL;
	}

	if (!t && raise)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected a tensor argument");

	return t;
}

static at::Tensor *
tensor_arg(uc_vm_t *vm, uc_value_t *v)
{
	return tensor_arg_impl(vm, v, true);
}

/* Multi-GPU: raw kernel launches and cudaMemcpyAsync calls run on the
 * CURRENT device's stream — a tensor living on another device would be
 * dereferenced across devices (illegal memory access). ATen ops set
 * their own device guard from the tensor; raw launches do not. Bindings
 * that bypass ATen set this guard from their primary input tensor
 * before touching the current stream. */
struct TensorDeviceGuard {
	c10::cuda::OptionalCUDAGuard guard;

	explicit TensorDeviceGuard(const at::Tensor &t)
	{
		if (t.is_cuda())
			guard.set_index(t.get_device());
	}
};

/* Raise a runtime exception from a C++ failure without letting the
 * exception cross the module boundary (invariant 2). */
static void
raise_from_cpp(uc_vm_t *vm, const char *what)
{
	uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "%s", what);
}

/* The raise-and-bail idiom used throughout the bindings: raise a runtime
 * exception and return NULL from the enclosing function. Assumes `vm`
 * is in scope (every handler and helper receives it). */
#define RAISE(...) \
	do { raise_from_cpp(vm, __VA_ARGS__); return NULL; } while (0)

/* Check a uc_*_fp16 launcher result: 0 is success, anything else is a
 * CUDA error or a rejected configuration (the launcher already printed
 * details to stderr). Raises and returns false on failure. */
static bool
launch_ok(uc_vm_t *vm, int rc, const char *what)
{
	if (rc == 0)
		return true;

	if (rc < 0)
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"%s: unsupported configuration (see stderr)", what);
	else
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"%s: CUDA launch failed: %s", what,
			cudaGetErrorString((cudaError_t)rc));

	return false;
}

/* Wrap an at::Tensor in a (uniformly extended) torch.tensor resource.
 * The data area holds the at::Tensor*; optional pinned values (e.g. an
 * stfile keeping an mmap alive) can be set by the caller afterwards via
 * ucv_resource_value_set(). */
static uc_value_t *
tensor_wrap(uc_vm_t *vm, at::Tensor &&t, size_t pin_count)
{
	void *data = NULL;
	uc_value_t *res = ucv_resource_create_ex(vm, TENSOR_TYPE,
		&data, pin_count, sizeof(void *));

	if (!res)
		return NULL;

	*reinterpret_cast<at::Tensor **>(data) = new at::Tensor(std::move(t));

	return res;
}

/* Coerce an argument to double, raising on non-numeric input.
 * Returns false (after raising) on bad input.
 *
 * NB: ucv_to_number() allocates a new reference the caller must put, and
 * returns NULL for NaN doubles / non-numeric strings — do not use it in
 * hot paths. Integer and double values are read directly via their
 * accessors, which do not allocate. (Lesson L20.) */
static bool
arg_number(uc_vm_t *vm, uc_value_t *v, double *out)
{
	uc_type_t t = ucv_type(v);

	if (t == UC_INTEGER) {
		*out = (double)ucv_int64_get(v);
		return true;
	}

	if (t == UC_DOUBLE) {
		*out = ucv_double_get(v);
		return true;
	}

	uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a number");

	return false;
}

/* Coerce an argument to int64 (numbers only). */
static bool
arg_int(uc_vm_t *vm, uc_value_t *v, int64_t *out)
{
	double d;

	if (!arg_number(vm, v, &d))
		return false;

	*out = (int64_t)d;

	return true;
}

/* Coerce an argument to a dtype (dtype name string). */
static bool dtype_from_name(const char *name, at::ScalarType *out);

static bool
arg_dtype(uc_vm_t *vm, uc_value_t *v, at::ScalarType *out)
{
	if (ucv_type(v) != UC_STRING ||
	    !dtype_from_name(ucv_string_get(v), out)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected a dtype name");
		return false;
	}

	return true;
}

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

/* Parse dimension arguments: each argument is either a number or an
 * array of numbers. Appends to dims; returns false after raising. */
static bool
parse_dims(uc_vm_t *vm, size_t nargs, uc_value_t **args,
	std::vector<int64_t> *dims)
{
	for (size_t i = 0; i < nargs; i++) {
		uc_value_t *arg = args[i];

		if (ucv_type(arg) == UC_ARRAY) {
			size_t len = ucv_array_length(arg);

			for (size_t j = 0; j < len; j++) {
				double d;

				if (!arg_number(vm, ucv_array_get(arg, j), &d))
					return false;

				dims->push_back((int64_t)d);
			}
		} else {
			double d;

			if (!arg_number(vm, arg, &d))
				return false;

			dims->push_back((int64_t)d);
		}
	}

	return true;
}

/* Extract the `other` operand of a binary op: a tensor or a number.
 * Sets exactly one of *other_t / *other_s. Returns false after raising
 * on bad input. */
static bool
binary_other(uc_vm_t *vm, uc_value_t *v, at::Tensor **other_t, double *other_s)
{
	at::Tensor *t = tensor_arg_impl(vm, v, false);

	if (t) {
		*other_t = t;
		return true;
	}

	uc_type_t ty = v ? ucv_type(v) : UC_NULL;

	if (ty == UC_INTEGER || ty == UC_DOUBLE)
		return arg_number(vm, v, other_s);

	uc_vm_raise_exception(vm, EXCEPTION_TYPE,
		"expected a tensor or a number");

	return false;
}

/* Shared body of binary elementwise ops (add/mul/...). `other` is a
 * tensor or a number; computes out-of-place via ft/fs. */
template <typename FT, typename FS>
static uc_value_t *
tensor_binary_out(uc_vm_t *vm, size_t nargs, const char *name,
	FT ft, FS fs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	if (nargs != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s expects one argument", name);
		return NULL;
	}

	at::Tensor *ot = NULL;
	double os = 0.0;

	if (!binary_other(vm, uc_fn_arg(0), &ot, &os))
		return NULL;

	try {
		at::Tensor r = ot ? ft(*t, *ot) : fs(*t, os);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* Shared body of in-place binary elementwise ops (add_/mul_/...). */
template <typename FT, typename FS>
static uc_value_t *
tensor_binary_in(uc_vm_t *vm, size_t nargs, const char *name,
	FT fti, FS fsi)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	if (nargs != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s expects one argument", name);
		return NULL;
	}

	at::Tensor *ot = NULL;
	double os = 0.0;

	if (!binary_other(vm, uc_fn_arg(0), &ot, &os))
		return NULL;

	try {
		if (ot)
			fti(*t, *ot);
		else {
			/* integer tensors need an integer Scalar — a double
			 * scalar promotes the result to Float and the in-place
			 * op then fails type checking (e.g. pos.add_(1) on
			 * an int64 position buffer) */
			c10::Scalar s = at::isIntegralType(t->scalar_type(), false)
				? c10::Scalar((int64_t)os) : c10::Scalar(os);

			fsi(*t, s);
		}
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/* Shared body of unary elementwise ops (sqrt/silu/...). */
template <typename F>
static uc_value_t *
tensor_unary(uc_vm_t *vm, size_t nargs, const char *name, F f)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	if (nargs != 0) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s expects no arguments", name);
		return NULL;
	}

	try {
		at::Tensor r = f(*t);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static const char *
dtype_name(at::ScalarType t)
{
	switch (t) {
	case at::kFloat:    return "float32";
	case at::kDouble:   return "float64";
	case at::kHalf:     return "float16";
	case at::kBFloat16: return "bfloat16";
	case at::kInt:      return "int32";
	case at::kLong:     return "int64";
	case at::kShort:    return "int16";
	case at::kChar:     return "int8";
	case at::kByte:     return "uint8";
	case at::kUInt16:   return "uint16";
	case at::kUInt32:   return "uint32";
	case at::kUInt64:   return "uint64";
	case at::kBool:     return "bool";
	default:            return "unknown";
	}
}

static const char *
dtype_name(at::Tensor &t)
{
	return dtype_name(t.scalar_type());
}

/* Parse a dtype name; returns true when known. */
static bool
dtype_from_name(const char *name, at::ScalarType *out)
{
	if (!strcmp(name, "float32")) { *out = at::kFloat;    return true; }
	if (!strcmp(name, "float16")) { *out = at::kHalf;     return true; }
	if (!strcmp(name, "bfloat16")) { *out = at::kBFloat16; return true; }
	if (!strcmp(name, "float64")) { *out = at::kDouble;   return true; }
	if (!strcmp(name, "int64"))   { *out = at::kLong;     return true; }
	if (!strcmp(name, "int32"))   { *out = at::kInt;      return true; }
	if (!strcmp(name, "int16"))   { *out = at::kShort;    return true; }
	if (!strcmp(name, "int8"))    { *out = at::kChar;     return true; }
	if (!strcmp(name, "uint8"))   { *out = at::kByte;     return true; }
	if (!strcmp(name, "uint16"))  { *out = at::kUInt16;   return true; }
	if (!strcmp(name, "uint32"))  { *out = at::kUInt32;   return true; }
	if (!strcmp(name, "uint64"))  { *out = at::kUInt64;   return true; }
	if (!strcmp(name, "bool"))    { *out = at::kBool;     return true; }

	return false;
}

/* --- constructors ------------------------------------------------------- */

/* Parse a device name: "cpu", "cuda" or "cuda:N" (N = device index,
 * defaulting to the current device when omitted). Returns false after
 * raising on an unknown name. */
static bool
parse_device_name(uc_vm_t *vm, uc_value_t *v, at::Device *dev)
{
	const char *dn = ucv_string_get(v);

	if (!dn) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected a device name string");
		return false;
	}

	if (!strcmp(dn, "cuda")) {
		*dev = at::Device(at::kCUDA, /* noproto */ 0);
		return true;
	}

	if (!strncmp(dn, "cuda:", 5) && dn[5] >= '0' && dn[5] <= '9') {
		char *end = NULL;
		long idx = strtol(dn + 5, &end, 10);

		if (end && !*end && idx >= 0 && idx <= 255) {
			*dev = at::Device(at::kCUDA, (int)idx);
			return true;
		}
	}

	if (!strcmp(dn, "cpu")) {
		*dev = at::Device(at::kCPU);
		return true;
	}

	uc_vm_raise_exception(vm, EXCEPTION_TYPE,
		"unknown device '%s'", dn);
	return false;
}

/* Optional trailing device argument for the upload paths: absent or
 * null means cuda:0; an integer is a device index, a string goes
 * through parse_device_name ("cuda", "cuda:N"). */
static bool
parse_target_device(uc_vm_t *vm, uc_value_t *v, at::Device *dev)
{
	*dev = at::Device(at::kCUDA, 0);

	if (!v || ucv_type(v) == UC_NULL)
		return true;

	if (ucv_type(v) == UC_INTEGER) {
		int64_t idx = ucv_int64_get(v);

		if (idx < 0 || idx > 255) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"device index out of range");
			return false;
		}

		*dev = at::Device(at::kCUDA, (int)idx);
		return true;
	}

	return parse_device_name(vm, v, dev);
}

/* Parse one dims argument: an array of numbers, or a single number.
 * Appends to dims; returns false after raising. */
static bool
parse_dims_arg(uc_vm_t *vm, uc_value_t *d, std::vector<int64_t> *dims)
{
	if (ucv_type(d) == UC_ARRAY) {
		size_t len = ucv_array_length(d);

		for (size_t j = 0; j < len; j++) {
			double v;

			if (!arg_number(vm, ucv_array_get(d, j), &v) ||
			    std::isnan(v) || v < 0) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"invalid dimension");
				return false;
			}

			dims->push_back((int64_t)v);
		}
	}
	else {
		double v;

		if (!arg_number(vm, d, &v) || std::isnan(v) || v < 0) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"invalid dimension");
			return false;
		}

		dims->push_back((int64_t)v);
	}

	return true;
}

/* Parse the common constructor signature: dims, dtype?, device? —
 *
 *   zeros(dims, dtype?, device?)
 *   ones (dims, dtype?, device?)
 *
 * dims (arg 0) is an array of numbers (a bare number is accepted as a
 * 1-element shape); dtype (arg 1) and device (arg 2) are name strings
 * ("float16", "cuda", ...), each optional and skippable with null.
 * Returns false after raising on bad input. */
static bool
parse_factory_args(uc_vm_t *vm, size_t nargs, std::vector<int64_t> *dims,
	at::ScalarType *dt, at::Device *dev)
{
	if (nargs < 1 || nargs > 3) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dims, dtype?, device?)");
		return false;
	}

	/* dims: array of numbers, or a single number */
	if (!parse_dims_arg(vm, uc_fn_arg(0), dims))
		return false;

	/* dtype: name string or null */
	if (nargs >= 2 && ucv_type(uc_fn_arg(1)) != UC_NULL) {
		uc_value_t *v = uc_fn_arg(1);

		if (ucv_type(v) != UC_STRING ||
		    !dtype_from_name(ucv_string_get(v), dt)) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"unknown dtype '%s'", ucv_string_get(v));
			return false;
		}
	}

	/* device: name string or null */
	if (nargs >= 3 && ucv_type(uc_fn_arg(2)) != UC_NULL &&
	    !parse_device_name(vm, uc_fn_arg(2), dev))
		return false;

	return true;
}

/*
 * zeros(dims, dtype?, device?): torch.tensor
 *
 * Zero-initialized tensor, float32 on CPU by default. dims is an array
 * of numbers; dtype and device are name strings ("float16", "cuda",
 * ...), each optional and skippable with null. Direct device
 * allocation matters for graph capture: an H2D copy inside a capture
 * is illegal.
 */
static uc_value_t *
tensor_zeros(uc_vm_t *vm, size_t nargs)
{
	std::vector<int64_t> dims;
	at::ScalarType dt = at::kFloat;
	at::Device dev = at::Device(at::kCPU);

	if (!parse_factory_args(vm, nargs, &dims, &dt, &dev))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor t = at::zeros(dims,
			at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * empty(dims, dtype?, device?): torch.tensor
 *
 * As zeros(), uninitialized — for outputs a kernel fully overwrites.
 */
static uc_value_t *
tensor_empty(uc_vm_t *vm, size_t nargs)
{
	std::vector<int64_t> dims;
	at::ScalarType dt = at::kFloat;
	at::Device dev(at::kCPU);

	if (!parse_factory_args(vm, nargs, &dims, &dt, &dev))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor t = at::empty(dims, at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * ones(dims, dtype?, device?): torch.tensor
 *
 * As zeros(), filled with ones.
 */
static uc_value_t *
tensor_ones(uc_vm_t *vm, size_t nargs)
{
	std::vector<int64_t> dims;
	at::ScalarType dt = at::kFloat;
	at::Device dev = at::Device(at::kCPU);

	if (!parse_factory_args(vm, nargs, &dims, &dt, &dev))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor t = at::ones(dims,
			at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * rand(dims, dtype?, device?): torch.tensor
 *
 * As zeros(), filled with uniform [0, 1) values from the device's
 * default generator. For test/benchmark fixtures that are too big to
 * build on the host (a 184M-element from_array is minutes; this is
 * milliseconds — and no pageable H2D, L58).
 */
static uc_value_t *
tensor_rand(uc_vm_t *vm, size_t nargs)
{
	std::vector<int64_t> dims;
	at::ScalarType dt = at::kFloat;
	at::Device dev = at::Device(at::kCPU);

	if (!parse_factory_args(vm, nargs, &dims, &dt, &dev))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor t = at::rand(dims,
			at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * full(dims, value, dtype?, device?): torch.tensor
 *
 * As zeros(), filled with `value`.
 */
static uc_value_t *
tensor_full(uc_vm_t *vm, size_t nargs)
{
	std::vector<int64_t> dims;
	at::ScalarType dt = at::kFloat;
	at::Device dev = at::Device(at::kCPU);
	double value;

	/* full(dims, value, dtype?, device?) */
	if (nargs < 2 || nargs > 4 ||
	    !arg_number(vm, uc_fn_arg(1), &value)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dims, value, dtype?, device?)");
		return NULL;
	}

	/* dims: array of numbers, or a single number */
	if (!parse_dims_arg(vm, uc_fn_arg(0), &dims))
		return NULL;

	/* dtype: name string or null */
	if (nargs >= 3 && ucv_type(uc_fn_arg(2)) != UC_NULL) {
		uc_value_t *v = uc_fn_arg(2);

		if (ucv_type(v) != UC_STRING ||
		    !dtype_from_name(ucv_string_get(v), &dt)) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"unknown dtype '%s'", ucv_string_get(v));
			return NULL;
		}
	}

	/* device: name string or null */
	if (nargs >= 4 && ucv_type(uc_fn_arg(3)) != UC_NULL &&
	    !parse_device_name(vm, uc_fn_arg(3), &dev))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor t = at::full(dims, value,
			at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * arange(n: number, dtype?: string): torch.tensor
 *
 * 1-D tensor [0, 1, ..., n-1], float32 by default. The RoPE-cache
 * building block.
 */
static uc_value_t *
tensor_arange(uc_vm_t *vm, size_t nargs)
{
	int64_t n;
	at::ScalarType dt = at::kFloat;
	at::Device dev(at::kCPU);

	if (nargs < 1 || nargs > 3 || !arg_int(vm, uc_fn_arg(0), &n)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (n, dtype?, device?)");
		return NULL;
	}

	if (nargs >= 2 && ucv_type(uc_fn_arg(1)) != UC_NULL) {
		uc_value_t *dn = uc_fn_arg(1);

		if (ucv_type(dn) != UC_STRING ||
		    !dtype_from_name(ucv_string_get(dn), &dt)) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"unknown dtype");
			return NULL;
		}
	}

	/* device: direct device allocation — no blocking H2D copy (a
	 * pageable-source .cuda() synchronizes the stream, which stalls
	 * multi-GPU pipelining) */
	if (nargs == 3 && ucv_type(uc_fn_arg(2)) != UC_NULL &&
	    !parse_device_name(vm, uc_fn_arg(2), &dev))
		return NULL;

	try {
		at::Tensor t = at::arange(n,
			at::TensorOptions().dtype(dt).device(dev));

		return tensor_wrap(vm, std::move(t), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * from_array(data: number[]|number[][], dtype?, device?): torch.tensor
 *
 * Creates a tensor (float32 by default) from a (nested) ucode array. 1-D
 * and 2-D supported; deeper nesting raises. With a CUDA device the values
 * are staged in pinned memory and uploaded asynchronously on the current
 * stream — no host wait, unlike from_array(...).cuda().
 */
static bool
flatten_array(uc_value_t *arr, std::vector<double> &out, size_t *width,
	size_t depth)
{
	size_t len = ucv_array_length(arr);

	/* The width is the length of the innermost arrays (leaf rows).
	 * Capture it only below the top level: at depth 0 the array length
	 * is the row COUNT for 2-D input (or the width for 1-D input, where
	 * the elements are numbers and no recursion happens). */
	if (depth > 0) {
		if (*width == 0)
			*width = len;
		else if (*width != len)
			return false;
	}

	for (size_t i = 0; i < len; i++) {
		uc_value_t *v = ucv_array_get(arr, i);

		switch (ucv_type(v)) {
		case UC_INTEGER:
		case UC_DOUBLE:
			out.push_back(ucv_type(v) == UC_INTEGER ?
				(double)ucv_int64_get(v) : ucv_double_get(v));
			break;

		case UC_ARRAY:
			/* 2-D only: an array below the top level means 3-D+ input.
			 * The dims derivation below cannot express it, and
			 * continuing would overflow the destination (the width
			 * check alone does not catch [[[1,2],[3,4]],...]). */
			if (depth >= 1)
				return false;

			if (!flatten_array(v, out, width, depth + 1))
				return false;
			break;

		default:
			return false;
		}
	}

	return true;
}

static uc_value_t *
tensor_from_array(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *arg = uc_fn_arg(0);
	at::ScalarType dt = at::kFloat;

	at::Device dev(at::kCPU);

	if (nargs < 1 || nargs > 3 || ucv_type(arg) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (array, dtype?, device?)");
		return NULL;
	}

	if (nargs >= 2 && ucv_type(uc_fn_arg(1)) != UC_NULL &&
	    !arg_dtype(vm, uc_fn_arg(1), &dt))
		return NULL;

	if (nargs == 3 && ucv_type(uc_fn_arg(2)) != UC_NULL &&
	    !parse_device_name(vm, uc_fn_arg(2), &dev))
		return NULL;

	std::vector<double> data;
	size_t width = 0;

	if (!flatten_array(arg, data, &width, 0)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected a rectangular 1-D or 2-D array of numbers");
		return NULL;
	}

	std::vector<int64_t> dims;

	if (ucv_type(ucv_array_get(arg, 0)) == UC_ARRAY) {
		dims = { (int64_t)ucv_array_length(arg), (int64_t)width };

		/* belt and braces: the flattened element count must match the
		 * derived shape exactly before anything is copied */
		if ((int64_t)data.size() != dims[0] * dims[1]) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"expected a rectangular 1-D or 2-D array of numbers");
			return NULL;
		}
	}
	else
		dims = { (int64_t)data.size() };

	at::Tensor t;

	try {
		/* a CUDA target stages through pinned memory, so the upload
		 * below can be asynchronous (no host wait for the stream) */
		t = at::zeros(dims,
			at::TensorOptions().dtype(dt).device(at::kCPU)
				.pinned_memory(dev.is_cuda()));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	switch (dt) {
	case at::kFloat:
		std::copy(data.begin(), data.end(), t.data_ptr<float>());
		break;

	case at::kDouble:
		std::copy(data.begin(), data.end(), t.data_ptr<double>());
		break;

	case at::kLong:
		std::copy(data.begin(), data.end(), t.data_ptr<int64_t>());
		break;

	case at::kInt:
	case at::kShort:
	case at::kChar:
	case at::kByte:
		/* integer dtypes narrower than int64: values arrive as
		 * doubles/int64s from ucode; range-check then narrow */
		for (size_t i = 0; i < data.size(); i++) {
			double v = data[i];

			if (v != (int64_t)v) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"from_array: non-integer value for an "
					"integer dtype");
				return NULL;
			}
		}

		switch (dt) {
		case at::kInt:
			for (size_t i = 0; i < data.size(); i++)
				t.data_ptr<int32_t>()[i] = (int32_t)data[i];
			break;
		case at::kShort:
			for (size_t i = 0; i < data.size(); i++)
				t.data_ptr<int16_t>()[i] = (int16_t)data[i];
			break;
		case at::kChar:
			for (size_t i = 0; i < data.size(); i++)
				t.data_ptr<int8_t>()[i] = (int8_t)data[i];
			break;
		default:
			for (size_t i = 0; i < data.size(); i++)
				t.data_ptr<uint8_t>()[i] = (uint8_t)data[i];
			break;
		}
		break;

	default:
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"unsupported dtype for from_array");
		return NULL;
	}

	if (dev.is_cuda()) {
		try {
			t = t.to(dev, /* non_blocking */ true);
		}
		catch (const std::exception &e) {
			RAISE(e.what());
		}
	}

	return tensor_wrap(vm, std::move(t), 0);
}

/* --- methods ------------------------------------------------------------ */

/*
 * item(): number
 *
 * Returns the single element of a scalar (numel == 1) tensor as a number.
 * Integer tensors yield exact integers; everything else a double.
 */
static uc_value_t *
tensor_item(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	if (t->numel() != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"item() requires a tensor with exactly one element");
		return NULL;
	}

	at::Tensor cpu;

	try {
		cpu = t->cpu();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	switch (cpu.scalar_type()) {
	case at::kLong:
		return ucv_int64_new(cpu.item<int64_t>());

	case at::kInt:
		return ucv_int64_new(cpu.item<int32_t>());

	case at::kShort:
		return ucv_int64_new(cpu.item<int16_t>());

	case at::kChar:
		return ucv_int64_new(cpu.item<int8_t>());

	case at::kByte:
		return ucv_int64_new(cpu.item<uint8_t>());

	default:
		return ucv_double_new(cpu.item<double>());
	}
}

/*
 * tolist(): array|number
 *
 * Returns a (nested) ucode array copy of a CPU tensor, any dimensionality.
 * Float dtypes become numbers (doubles); integer dtypes exact integers;
 * bool 1.0/0.0. Scalars (numel == 1, dim == 0) return the bare number.
 */
static uc_value_t *
tolist_recurse(uc_vm_t *vm, const int64_t *dims, size_t ndim, size_t d,
	int64_t *off, const double *dp, const int64_t *ip, bool is_int)
{
	int64_t n = dims[d];
	uc_value_t *arr = ucv_array_new_length(vm, (size_t)n);

	for (int64_t i = 0; i < n; i++) {
		uc_value_t *v;

		if (d + 1 == ndim) {
			v = is_int ? ucv_int64_new(ip[*off]) :
			    ucv_double_new(dp[*off]);
			(*off)++;
		} else
			v = tolist_recurse(vm, dims, ndim, d + 1, off, dp, ip,
				is_int);

		ucv_array_push(arr, v);
	}

	return arr;
}

static uc_value_t *
tensor_tolist(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	at::Tensor flat;

	try {
		flat = t->cpu().contiguous();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	at::ScalarType st = flat.scalar_type();
	bool is_int;
	at::Tensor conv;
	const double *dp = NULL;
	const int64_t *ip = NULL;

	switch (st) {
	case at::kLong:
	case at::kInt:
	case at::kShort:
	case at::kChar:
		conv = flat.to(at::kLong);
		ip = conv.data_ptr<int64_t>();
		is_int = true;
		break;

	default:
		conv = flat.to(at::kDouble);
		dp = conv.data_ptr<double>();
		is_int = false;
		break;
	}

	size_t ndim = (size_t)conv.dim();

	if (ndim == 0)
		return is_int ? ucv_int64_new(ip[0]) : ucv_double_new(dp[0]);

	std::vector<int64_t> dims(conv.sizes().begin(), conv.sizes().end());
	int64_t off = 0;

	return tolist_recurse(vm, dims.data(), ndim, 0, &off, dp, ip, is_int);
}

/*
 * shape(): number[]
 */
static uc_value_t *
tensor_shape(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	uc_value_t *arr = ucv_array_new_length(vm, t->dim());

	for (int64_t i = 0; i < t->dim(); i++)
		ucv_array_push(arr, ucv_int64_new(t->size(i)));

	return arr;
}

/*
 * dtype(): string
 */
static uc_value_t *
tensor_dtype(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	return ucv_string_new(dtype_name(*t));
}

/*
 * fill_(value: number): this
 *
 * Fills the tensor in place with a scalar. Returns the same resource so
 * calls can chain.
 */
static uc_value_t *
tensor_fill(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	double v;

	if (!t)
		return NULL;

	if (nargs != 1 || !arg_number(vm, uc_fn_arg(0), &v)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a number");
		return NULL;
	}

	try {
		t->fill_((float)v);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	/* Return `this` — but the VM takes ownership of our return value, so
	 * hand it a fresh reference rather than the borrowed ctx pointer. */
	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * cuda(): torch.tensor          — move to cuda:0 (or the current device)
 * cuda(n): torch.tensor         — move to cuda:n
 * cpu(): torch.tensor
 *
 * Returns a new tensor on the target device (the source is untouched).
 * Cross-device moves (cuda:0 -> cuda:1) go through ATen, which stages
 * through host memory or P2P as available.
 */
static uc_value_t *
tensor_to_device(uc_vm_t *vm, at::Device dev, bool non_blocking)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	try {
		at::Tensor out;

		/* non_blocking H2D: stage through pinned memory (from the
		 * caching host allocator, which keeps the buffer alive until
		 * the async copy has consumed it) and copy without a host
		 * sync. A plain pageable copy synchronizes the stream. */
		if (non_blocking && t->is_cpu() && dev.is_cuda()) {
			at::Tensor src = t->is_pinned() ? *t : t->pin_memory();

			out = src.to(dev, /* non_blocking */ true);
		}
		else {
			out = t->to(dev, non_blocking);
		}

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * cuda(idx?, non_blocking?): torch.tensor
 *
 * Copy to CUDA device `idx` (default 0). non_blocking (default false):
 * for CPU sources, stage through pinned memory and copy asynchronously
 * on the current stream — the host does not wait (needed to keep
 * several GPUs busy at once). Later ops on the stream see the data.
 */
static uc_value_t *
tensor_cuda(uc_vm_t *vm, size_t nargs)
{
	int64_t idx = 0;
	bool nb = false;

	if (nargs >= 1 && ucv_type(uc_fn_arg(0)) != UC_NULL &&
	    !arg_int(vm, uc_fn_arg(0), &idx))
		return NULL;

	if (nargs >= 2 && !arg_bool(vm, uc_fn_arg(1), &nb))
		return NULL;

	return tensor_to_device(vm, at::Device(at::kCUDA, (int)idx), nb);
}

static uc_value_t *
tensor_cpu(uc_vm_t *vm, size_t nargs)
{
	return tensor_to_device(vm, at::kCPU, false);
}

/*
 * data_ptr(): integer
 *
 * The tensor's first element device address — the building block for
 * the int64 pointer tables mgemm consumes. Stable for the tensor's
 * lifetime (weights and persistent decode buffers never reallocate).
 */
static uc_value_t *
tensor_data_ptr(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	return ucv_int64_new((int64_t)(uintptr_t)t->data_ptr());
}

/*
 * free()
 *
 * Explicitly releases the underlying tensor. The resource stays valid but
 * becomes inert; the GC finalizer will not double-free.
 */
static uc_value_t *
tensor_free_method(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *res = _uc_fn_this_res(vm);

	if (!res || !res_is_type(res, TENSOR_TYPE))
		return NULL;

	void *data = ucv_resource_data(res, TENSOR_TYPE);

	if (data && *(void **)data) {
		delete static_cast<at::Tensor *>(*(void **)data);
		*(void **)data = NULL;
	}

	return NULL;
}

/* --- Phase 1: tensor API surface ----------------------------------------- */

/*
 * to(dtype: string): torch.tensor
 * to(device: string): torch.tensor   ("cpu", "cuda", "cuda:N")
 *
 * Casts to the named dtype (keeping the device) or moves to the named
 * device (keeping the dtype).
 */
static uc_value_t *
tensor_to(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	uc_value_t *dn;

	if (!t)
		return NULL;

	if (nargs != 1 || ucv_type(dn = uc_fn_arg(0)) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected a dtype or device name");
		return NULL;
	}

	at::ScalarType dt;

	if (dtype_from_name(ucv_string_get(dn), &dt)) {
		try {
			at::Tensor r = t->to(dt);

			return tensor_wrap(vm, std::move(r), 0);
		}
		catch (const std::exception &e) {
			RAISE(e.what());
		}
	}

	at::Device dev(at::kCPU);

	if (!parse_device_name(vm, dn, &dev))
		return NULL;

	try {
		at::Tensor r = t->to(dev);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * view(dims...) / reshape(dims...): torch.tensor
 *
 * Reinterprets (view) or recomputes (reshape) the shape with the same
 * number of elements. Dimensions as an array or separate arguments.
 */
static uc_value_t *
tensor_view_impl(uc_vm_t *vm, size_t nargs, bool strict)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	std::vector<uc_value_t *> args;

	for (size_t i = 0; i < nargs; i++)
		args.push_back(uc_fn_arg(i));

	std::vector<int64_t> dims;

	if (!parse_dims(vm, args.size(), args.data(), &dims))
		return NULL;

	if (dims.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"at least one dimension required");
		return NULL;
	}

	try {
		at::Tensor r = strict ? t->view(dims) : t->reshape(dims);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
tensor_view(uc_vm_t *vm, size_t nargs)
{
	return tensor_view_impl(vm, nargs, true);
}

static uc_value_t *
tensor_reshape(uc_vm_t *vm, size_t nargs)
{
	return tensor_view_impl(vm, nargs, false);
}

/*
 * transpose(dim0: number, dim1: number): torch.tensor
 * t(): torch.tensor
 *
 * Returns a transposed view. t() is 2-D transpose.
 */
static uc_value_t *
tensor_transpose(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t d0, d1;

	if (!t)
		return NULL;

	if (nargs != 2 || !arg_int(vm, uc_fn_arg(0), &d0) ||
	    !arg_int(vm, uc_fn_arg(1), &d1)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dim0, dim1)");
		return NULL;
	}

	try {
		at::Tensor r = t->transpose(d0, d1);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
tensor_t(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "t",
		[](const at::Tensor &x) { return x.t(); });
}

/*
 * contiguous(): torch.tensor
 *
 * Returns a contiguous copy (or the same values re-laid-out); views made
 * by transpose/narrow must be made contiguous before passing them to
 * kernels that require contiguous storage.
 */
static uc_value_t *
tensor_contiguous(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "contiguous",
		[](const at::Tensor &x) { return x.contiguous(); });
}

/*
 * is_contiguous(): boolean
 */
static uc_value_t *
tensor_is_contiguous(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	return ucv_boolean_new(t->is_contiguous());
}

/*
 * numel(): number
 */
static uc_value_t *
tensor_numel(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	return ucv_int64_new(t->numel());
}

/*
 * narrow(dim: number, start: number, length: number): torch.tensor
 *
 * Returns a view narrowing `dim` to [start, start + length). This is the
 * building block for KV-cache slice writes and RoPE halves.
 */
static uc_value_t *
tensor_narrow(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim, start, length;

	if (!t)
		return NULL;

	if (nargs != 3 || !arg_int(vm, uc_fn_arg(0), &dim) ||
	    !arg_int(vm, uc_fn_arg(1), &start) ||
	    !arg_int(vm, uc_fn_arg(2), &length)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dim, start, length)");
		return NULL;
	}

	try {
		at::Tensor r = t->narrow(dim, start, length);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * unsqueeze(dim: number): torch.tensor
 * squeeze(dim?: number): torch.tensor
 *
 * Size-1 dimension insertion/removal.
 */
static uc_value_t *
tensor_unsqueeze(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim;

	if (!t)
		return NULL;

	if (nargs != 1 || !arg_int(vm, uc_fn_arg(0), &dim)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dim)");
		return NULL;
	}

	try {
		at::Tensor r = t->unsqueeze(dim);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
tensor_squeeze(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim = 0;

	if (!t)
		return NULL;

	if (nargs > 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at most one argument");
		return NULL;
	}

	if (nargs == 1 && !arg_int(vm, uc_fn_arg(0), &dim))
		return NULL;

	try {
		at::Tensor r = nargs ? t->squeeze(dim) : t->squeeze();

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * copy_(src: torch.tensor): this
 *
 * In-place copy of `src` into this tensor (broadcasting src to this
 * tensor's shape). The KV-cache write primitive.
 */
/*
 * index(ids: tensor int64): torch.tensor
 *
 * Embedding-style gather along dim 0: returns a tensor shaped like ids
 * with the last dimension replaced by this tensor's trailing shape.
 * ids must be an int64 tensor of any shape.
 */
static uc_value_t *
tensor_index(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	at::Tensor *ids;

	if (!t)
		return NULL;

	if (nargs != 1 || !(ids = tensor_arg(vm, uc_fn_arg(0))))
		return NULL;

	try {
		if (ids->scalar_type() != at::kLong) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"index ids must be an int64 tensor");
			return NULL;
		}

		at::Tensor r = at::embedding(*t, *ids, -1, false, false);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * gather(dim, index): torch.tensor
 *
 * at::gather along `dim`: index must be an int64 tensor with the same
 * rank as this tensor; the result has index's shape with every other
 * dimension equal to this tensor's. The per-row target-logprob
 * primitive (log_softmax(logits).gather(-1, ids[:, None])).
 */
static uc_value_t *
tensor_gather(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	at::Tensor *idx;
	int64_t dim;

	if (!t)
		return NULL;

	if (nargs != 2 || !arg_int(vm, uc_fn_arg(0), &dim) ||
	    !(idx = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	try {
		if (idx->scalar_type() != at::kLong) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"gather index must be an int64 tensor");
			return NULL;
		}

		at::Tensor r = at::gather(*t, dim, *idx);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * expand(...sizes): torch.tensor
 *
 * Broadcast-expands this tensor to the given shape (size -1 keeps the
 * original dimension). Dimensions of size 1 may only be expanded.
 */
static uc_value_t *
tensor_expand(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	std::vector<uc_value_t *> args;
	std::vector<int64_t> dims;

	if (!t)
		return NULL;

	for (size_t i = 0; i < nargs; i++)
		args.push_back(uc_fn_arg(i));

	if (!parse_dims(vm, args.size(), args.data(), &dims))
		return NULL;

	try {
		at::Tensor r = t->expand(
			at::IntArrayRef(dims.data(), dims.size()));

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
tensor_copy_(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);

	if (!t)
		return NULL;

	if (nargs != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected one argument");
		return NULL;
	}

	at::Tensor *src = tensor_arg(vm, uc_fn_arg(0));

	if (!src)
		return NULL;

	try {
		t->copy_(*src);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * index_copy_(dim, index, src): this
 *
 * In-place scatter of `src` into this tensor along `dim` at the
 * positions given by the int64 `index` tensor. The captured-graph KV
 * write: the slot comes from a persistent position tensor whose VALUE
 * changes between replays (narrow+copy_ would bake the offset in).
 */
static uc_value_t *
tensor_index_copy_(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm), *idx, *src;
	int64_t dim;

	if (!t)
		return NULL;

	if (nargs != 3 || !arg_int(vm, uc_fn_arg(0), &dim) ||
	    !(idx = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(src = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	try {
		if (idx->scalar_type() != at::kLong) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"index must be an int64 tensor");
			return NULL;
		}

		t->index_copy_(dim, *idx, *src);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/* --- elementwise arithmetic ---------------------------------------------- */

/*
 * add(other) / mul(other) / div(other) / sub(other): torch.tensor
 * add_(other) / mul_(other): this
 *
 * `other` is a tensor (broadcast) or a number. The trailing-underscore
 * variants write in place and return the same tensor.
 */
static uc_value_t *
tensor_add(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_out(vm, nargs, "add",
		[](const at::Tensor &a, const at::Tensor &b) { return at::add(a, b); },
		[](const at::Tensor &a, double s) { return at::add(a, s); });
}

static uc_value_t *
tensor_add_(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_in(vm, nargs, "add_",
		[](at::Tensor &a, const at::Tensor &b) { a.add_(b); },
		[](at::Tensor &a, const c10::Scalar &s) { a.add_(s); });
}

static uc_value_t *
tensor_mul(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_out(vm, nargs, "mul",
		[](const at::Tensor &a, const at::Tensor &b) { return at::mul(a, b); },
		[](const at::Tensor &a, double s) { return at::mul(a, s); });
}

static uc_value_t *
tensor_mul_(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_in(vm, nargs, "mul_",
		[](at::Tensor &a, const at::Tensor &b) { a.mul_(b); },
		[](at::Tensor &a, const c10::Scalar &s) { a.mul_(s); });
}

static uc_value_t *
tensor_div(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_out(vm, nargs, "div",
		[](const at::Tensor &a, const at::Tensor &b) { return at::div(a, b); },
		[](const at::Tensor &a, double s) { return at::div(a, s); });
}

static uc_value_t *
tensor_sub(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_out(vm, nargs, "sub",
		[](const at::Tensor &a, const at::Tensor &b) { return at::sub(a, b); },
		[](const at::Tensor &a, double s) { return at::sub(a, s); });
}

/*
 * pow(exp: number): torch.tensor
 *
 * Raises every element to a scalar power.
 */
static uc_value_t *
tensor_pow(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	double e;

	if (!t)
		return NULL;

	if (nargs != 1 || !arg_number(vm, uc_fn_arg(0), &e)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected an exponent");
		return NULL;
	}

	try {
		at::Tensor r = at::pow(*t, e);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e2) {
		RAISE(e2.what());
	}
}

/*
 * Unary math: sqrt, rsqrt, neg, exp, log, cos, sin, silu.
 * All out-of-place, shape-preserving.
 */
static uc_value_t *
tensor_sqrt(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "sqrt",
		[](const at::Tensor &x) { return at::sqrt(x); });
}

static uc_value_t *
tensor_abs(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "abs",
		[](const at::Tensor &x) { return at::abs(x); });
}

static uc_value_t *
tensor_rsqrt(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "rsqrt",
		[](const at::Tensor &x) { return at::rsqrt(x); });
}

static uc_value_t *
tensor_neg(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "neg",
		[](const at::Tensor &x) { return at::neg(x); });
}

static uc_value_t *
tensor_exp(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "exp",
		[](const at::Tensor &x) { return at::exp(x); });
}

static uc_value_t *
tensor_log(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "log",
		[](const at::Tensor &x) { return at::log(x); });
}

static uc_value_t *
tensor_cos(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "cos",
		[](const at::Tensor &x) { return at::cos(x); });
}

static uc_value_t *
tensor_sin(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "sin",
		[](const at::Tensor &x) { return at::sin(x); });
}

static uc_value_t *
tensor_silu(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "silu",
		[](const at::Tensor &x) { return at::silu(x); });
}

static uc_value_t *
tensor_sigmoid(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "sigmoid",
		[](const at::Tensor &x) { return at::sigmoid(x); });
}

/* softplus with the default threshold 20 (x > 20 passes through
 * identity) — matches torch.nn.functional.softplus defaults, which is
 * what the GDN decay chain specifies. */
static uc_value_t *
tensor_softplus(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "softplus",
		[](const at::Tensor &x) { return at::softplus(x); });
}

/*
 * softmax(dim?: number): torch.tensor
 *
 * Softmax over `dim` (default -1).
 */
static uc_value_t *
tensor_softmax(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim = -1;

	if (!t)
		return NULL;

	if (nargs > 1 || (nargs == 1 && !arg_int(vm, uc_fn_arg(0), &dim))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at most one dimension argument");
		return NULL;
	}

	try {
		at::Tensor r = at::softmax(*t, dim);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * mean(dim?: number, keepdim?: boolean): torch.tensor
 * sum(dim?: number, keepdim?: boolean): torch.tensor
 *
 * Reduction over all dimensions, or over one (named) dimension.
 */
static uc_value_t *
tensor_reduction(uc_vm_t *vm, size_t nargs, bool is_mean)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim = -1;
	bool keepdim = false, has_dim = false;

	if (!t)
		return NULL;

	if (nargs > 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at most (dim, keepdim)");
		return NULL;
	}

	if (nargs >= 1) {
		if (!arg_int(vm, uc_fn_arg(0), &dim))
			return NULL;

		has_dim = true;
	}

	if (nargs >= 2 && !arg_bool(vm, uc_fn_arg(1), &keepdim))
		return NULL;

	try {
		at::Tensor r;

		if (is_mean)
			r = has_dim ? t->mean(dim, keepdim) : t->mean();
		else
			r = has_dim ? t->sum(dim, keepdim) : t->sum();

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
tensor_mean(uc_vm_t *vm, size_t nargs)
{
	return tensor_reduction(vm, nargs, true);
}

static uc_value_t *
tensor_sum(uc_vm_t *vm, size_t nargs)
{
	return tensor_reduction(vm, nargs, false);
}

static uc_value_t *
tensor_max(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim = -1;
	bool keepdim = false, has_dim = false;

	if (!t)
		return NULL;

	if (nargs > 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at most (dim, keepdim)");
		return NULL;
	}

	if (nargs >= 1) {
		if (!arg_int(vm, uc_fn_arg(0), &dim))
			return NULL;

		has_dim = true;
	}

	if (nargs >= 2 && !arg_bool(vm, uc_fn_arg(1), &keepdim))
		return NULL;

	try {
		at::Tensor r = has_dim ? t->amax(dim, keepdim) : t->max();

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * topk(k: number, dim?: number): { values: torch.tensor, indices: torch.tensor }
 *
 * Largest k elements along `dim` (default -1), values sorted descending.
 */
static uc_value_t *
tensor_topk(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t k, dim = -1;

	if (!t)
		return NULL;

	if (nargs < 1 || nargs > 2 || !arg_int(vm, uc_fn_arg(0), &k) ||
	    (nargs == 2 && !arg_int(vm, uc_fn_arg(1), &dim))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (k, dim?)");
		return NULL;
	}

	try {
		auto [values, indices] = at::topk(*t, k, dim);

		uc_value_t *obj = ucv_object_new(vm);

		ucv_object_add(obj, "values",
			tensor_wrap(vm, std::move(values), 0));
		ucv_object_add(obj, "indices",
			tensor_wrap(vm, std::move(indices), 0));

		return obj;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * argmax(dim?: number): torch.tensor
 *
 * at::argmax along `dim` (default -1). Draft/verify path: the winning
 * token id per position without materializing a top-k heap.
 */
static uc_value_t *
tensor_argmax(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	int64_t dim = -1;
	bool has_dim = false;

	if (!t)
		return NULL;

	if (nargs > 1 || (nargs == 1 && !arg_int(vm, uc_fn_arg(0), &dim))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dim?)");
		return NULL;
	}

	has_dim = nargs >= 1;

	try {
		at::Tensor r = has_dim ? t->argmax(dim) : t->argmax();

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * pin(): torch.tensor
 *
 * Returns a page-locked CPU copy of a CPU tensor, suitable for async
 * H2D copies. The basis of the weight loader's staging path.
 */
static uc_value_t *
tensor_pin(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "pin",
		[](const at::Tensor &x) {
			/* page-locked in EVERY device's context (cudaHostAllocPortable):
			 * with ATen's allocator the memory counts as pinned only for
			 * the allocating device, and an async D2H issued from another
			 * device degrades to a staged copy that completes after the
			 * stream (copy_async + notifier then deliver stale data) */
			at::Tensor c = x.cpu().contiguous();
			size_t n = c.nbytes();
			void *p = NULL;

			if (n == 0 || cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess)
				return x.pin_memory();

			memcpy(p, c.data_ptr(), n);

			return at::from_blob(p, c.sizes(), [](void *q) { cudaFreeHost(q); },
				c.options());
		});
}

/* --- Phase 1: structural ops --------------------------------------------- */

/*
 * cat(tensors: torch.tensor[], dim?: number): torch.tensor
 *
 * Concatenates tensors along `dim` (default 0).
 */
static uc_value_t *
torch_cat(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *arr = uc_fn_arg(0);
	int64_t dim = 0;

	if (nargs < 1 || nargs > 2 || ucv_type(arr) != UC_ARRAY ||
	    (nargs == 2 && !arg_int(vm, uc_fn_arg(1), &dim))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (tensors, dim?)");
		return NULL;
	}

	std::vector<at::Tensor> ts;

	for (size_t i = 0; i < ucv_array_length(arr); i++) {
		at::Tensor *t = tensor_arg(vm, ucv_array_get(arr, i));

		if (!t)
			return NULL;

		ts.push_back(*t);
	}

	if (ts.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at least one tensor");
		return NULL;
	}

	try {
		at::Tensor r = at::cat(ts, dim);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * stack(tensors: torch.tensor[], dim?: number): torch.tensor
 *
 * Stacks tensors along a new `dim` (default 0).
 */
static uc_value_t *
torch_stack(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *arr = uc_fn_arg(0);
	int64_t dim = 0;

	if (nargs < 1 || nargs > 2 || ucv_type(arr) != UC_ARRAY ||
	    (nargs == 2 && !arg_int(vm, uc_fn_arg(1), &dim))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (tensors, dim?)");
		return NULL;
	}

	std::vector<at::Tensor> ts;

	for (size_t i = 0; i < ucv_array_length(arr); i++) {
		at::Tensor *t = tensor_arg(vm, ucv_array_get(arr, i));

		if (!t)
			return NULL;

		ts.push_back(*t);
	}

	if (ts.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected at least one tensor");
		return NULL;
	}

	try {
		at::Tensor r = at::stack(ts, dim);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * matmul(a, b): torch.tensor
 *
 * Matrix product; handles 2-D @ 2-D and batched (3-D+) operands.
 */
static uc_value_t *
torch_matmul(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *a, *b;

	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (a, b)");
		return NULL;
	}

	if (!(a = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(b = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	try {
		at::Tensor r = at::matmul(*a, *b);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * sdpa(q, k, v, is_causal?: boolean, attn_mask?: tensor): torch.tensor
 *
 * scaled_dot_product_attention: q/k/v are (..., L, E) with the last two
 * dimensions query length and head size; dispatches to libtorch's fused
 * kernels where available. attn_mask is an optional additive mask
 * broadcastable to (..., L, S) — same dtype as q, or bool. The Phase 1
 * attention path.
 */
static uc_value_t *
torch_sdpa(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *q, *k, *v, *mask = NULL;
	bool is_causal = false, enable_gqa = false;

	if (nargs < 3 || nargs > 6) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, is_causal?, attn_mask?, enable_gqa?)");
		return NULL;
	}

	if (!(q = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(k = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(v = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	if (nargs >= 4 && !arg_bool(vm, uc_fn_arg(3), &is_causal))
		return NULL;

	if (nargs >= 5 && ucv_type(uc_fn_arg(4)) != UC_NULL &&
	    !(mask = tensor_arg(vm, uc_fn_arg(4))))
		return NULL;

	if (nargs >= 6 && !arg_bool(vm, uc_fn_arg(5), &enable_gqa))
		return NULL;

	try {
		at::Tensor r = at::scaled_dot_product_attention(
			*q, *k, *v,
			mask ? c10::optional<at::Tensor>(*mask) : c10::nullopt,
			0.0, is_causal, c10::nullopt, enable_gqa);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* --- CUDA graphs --------------------------------------------------------- */

/* A captured GPU computation. The resource owns an at::cuda::CUDAGraph
 * (graph exec + private mempool). Capture spans multiple ucode calls:
 * capture_begin() switches the thread to a pool stream, the VM then
 * executes arbitrary ucode (whose ATen ops are recorded), capture_end()
 * stops recording and restores the previous stream. Only one capture
 * may be active per thread (CUDA-wide constraint), guarded by
 * g_active below — runtime guard state, not a singleton graph.
 *
 * Shapes and tensor ADDRESSES are baked in at capture time; VALUES are
 * read at replay. Persistent input buffers (token ids, positions,
 * masks) are updated between replays with plain ops; outputs are the
 * tensors allocated during capture — the script must keep them
 * referenced (the graph pool reuses their blocks on replay). */

struct graph_state {
	at::cuda::CUDAGraph *graph;
	/* CUDAStream has no default constructor, so the streams are
	 * optionals: empty until capture_begin() fills them in. */
	c10::optional<c10::cuda::CUDAStream> pool_stream;
	c10::optional<c10::cuda::CUDAStream> prev_stream;
	int dev_index;		/* device the graph was captured on */
	bool capturing;

	/* multi-device capture: per extra device a side stream forked from
	 * the capture stream (joined back in capture_end), the stream it
	 * replaced, and an allocator pool (the graph's mempool id) that
	 * keeps the device's capture-time allocations private to it */
	struct side {
		int dev;
		c10::optional<c10::cuda::CUDAStream> stream, prev;
	};
	std::vector<side> sides;
	at::cuda::MempoolId_t pool_id;
	bool side_pools;

	graph_state() : graph(NULL), dev_index(-1), capturing(false),
		side_pools(false) {}
};

/* fork (join = false: capture stream -> side) or join (side -> capture
 * stream) through an event: inside a capture this becomes a graph edge */
static void
graph_link(cudaStream_t from, int from_dev, cudaStream_t to)
{
	cudaEvent_t ev;
	c10::cuda::CUDAGuard g((c10::DeviceIndex)from_dev);

	C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
	C10_CUDA_CHECK(cudaEventRecord(ev, from));
	C10_CUDA_CHECK(cudaStreamWaitEvent(to, ev, 0));
	C10_CUDA_CHECK(cudaEventDestroy(ev));
}

/* end the side streams of a capture: join, restore, close the pools */
static void
graph_end_sides(graph_state *g)
{
	for (auto &sd : g->sides) {
		if (!sd.stream)
			continue;

		try {
			graph_link(sd.stream->stream(), sd.dev, g->pool_stream->stream());
		}
		catch (...) {
		}

		if (sd.prev)
			c10::cuda::setCurrentCUDAStream(*sd.prev);

		c10::cuda::CUDACachingAllocator::endAllocateToPool(
			(c10::DeviceIndex)sd.dev, g->pool_id);
	}
}

/* The one capture currently active on this thread, if any. */
static graph_state *g_active = NULL;

static void
graph_free(void *data)
{
	graph_state *g = static_cast<graph_state *>(data);

	if (!g)
		return;

	/* freed mid-capture (exception, explicit free(), GC): end the
	 * capture and restore the stream, or every later ATen op would
	 * keep recording into a dead pool stream */
	if (g->capturing) {
		graph_end_sides(g);

		try {
			c10::cuda::CUDAGuard dg((c10::DeviceIndex)g->dev_index);

			g->graph->capture_end();
		}
		catch (...) {
		}

		g->capturing = false;

		if (g->prev_stream)
			c10::cuda::setCurrentCUDAStream(*g->prev_stream);
	}

	if (g == g_active)
		g_active = NULL;

	delete g->graph;

	if (g->side_pools)
		for (auto &sd : g->sides)
			c10::cuda::CUDACachingAllocator::releasePool(
				(c10::DeviceIndex)sd.dev, g->pool_id);

	delete g;
}

static graph_state *
graph_this(uc_vm_t *vm)
{
	auto *g = static_cast<graph_state *>(uc_fn_thisval(GRAPH_TYPE));

	if (!g)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"called on non-graph value");

	return g;
}

/*
 * graph(): torch.graph
 *
 * Creates an empty (uncaptured) graph resource.
 */
static uc_value_t *
torch_graph_new(uc_vm_t *vm, size_t nargs)
{
	/* C++ members — proper construction, no malloc+memset */
	graph_state *g = new (std::nothrow) graph_state();

	if (!g)
		return NULL;

	try {
		g->graph = new at::cuda::CUDAGraph();
	}
	catch (const std::exception &e) {
		delete g;
		RAISE(e.what());
	}

	return uc_resource_new(ucv_resource_type_lookup(vm, GRAPH_TYPE), g);
}

/*
 * capture_begin(device?, extra_devices?): this
 *
 * Starts recording on a pool stream. Every ATen op the VM runs until
 * capture_end() is recorded into the graph. device (optional, null =
 * default) selects the capture device: an index (default: current
 * device) or a "cuda:N" string. Captured ops must run on tensors of
 * THAT device — or of one of extra_devices (array of indices): each
 * gets a side stream forked from the capture stream (and joined back by
 * capture_end) plus a private allocator pool, so ops on several GPUs,
 * cross-device copies included (their event barriers become graph
 * edges), record into ONE multi-device graph that one replay() runs.
 * ATen ops and guarded raw launches follow the tensor's device. A
 * multi-device graph cannot be recaptured.
 */
static uc_value_t *
graph_capture_begin(uc_vm_t *vm, size_t nargs)
{
	graph_state *g = graph_this(vm);
	at::Device dev(at::kCUDA, 0);

	if (!g)
		return NULL;

	std::vector<int> extra;
	bool shared = false;
	at::cuda::MempoolId_t shared_id = { 0, 0 };

	if (nargs > 3 || (nargs >= 1 && ucv_type(uc_fn_arg(0)) != UC_NULL &&
	    !parse_target_device(vm, uc_fn_arg(0), &dev))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (device?, extra_devices?, pool?)");
		return NULL;
	}

	/* pool: the [a, b] pair another graph's pool() returned — this
	 * capture then allocates from the same private pool (graphs that are
	 * replayed one at a time can share it; the outputs of every capture
	 * must stay referenced, the temporaries are reused) */
	if (nargs == 3 && ucv_type(uc_fn_arg(2)) != UC_NULL) {
		uc_value_t *pa = uc_fn_arg(2);

		if (ucv_type(pa) != UC_ARRAY || ucv_array_length(pa) != 2 ||
		    ucv_type(ucv_array_get(pa, 0)) != UC_INTEGER ||
		    ucv_type(ucv_array_get(pa, 1)) != UC_INTEGER) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"pool must be the [a, b] array returned by pool()");
			return NULL;
		}

		shared_id = { (unsigned long long)ucv_int64_get(ucv_array_get(pa, 0)),
		              (unsigned long long)ucv_int64_get(ucv_array_get(pa, 1)) };
		shared = true;
	}

	if (nargs >= 2 && ucv_type(uc_fn_arg(1)) != UC_NULL) {
		uc_value_t *arr = uc_fn_arg(1);

		if (ucv_type(arr) != UC_ARRAY) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"extra_devices must be an array of device indices");
			return NULL;
		}

		for (size_t i = 0; i < ucv_array_length(arr); i++) {
			uc_value_t *v = ucv_array_get(arr, i);
			int64_t d = (ucv_type(v) == UC_INTEGER) ? ucv_int64_get(v) : -1;

			if (d < 0 || d >= at::cuda::device_count() || d == dev.index()) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"bad extra device index");
				return NULL;
			}

			extra.push_back((int)d);
		}
	}

	if (g->graph && !g->sides.empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"multi-device graph cannot be recaptured");
		return NULL;
	}

	if (g->capturing || g_active) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"a graph capture is already in progress");
		return NULL;
	}

	try {
		c10::cuda::CUDAGuard dev_guard(dev);

		g->dev_index = dev.index();
		g->prev_stream = c10::cuda::getCurrentCUDAStream(dev.index());
		g->pool_stream = c10::cuda::getStreamFromPool(
			/* high_priority */ false, dev.index());
		c10::cuda::setCurrentCUDAStream(*g->pool_stream);

		if (!extra.empty() || shared) {
			g->pool_id = shared ? shared_id : at::cuda::graph_pool_handle();
			g->graph->capture_begin(g->pool_id);
		}
		else {
			g->graph->capture_begin();
		}

		g->capturing = true;
		g_active = g;

		if (!extra.empty()) {
			cudaStreamCaptureStatus cs;
			unsigned long long cid;

			C10_CUDA_CHECK(cudaStreamGetCaptureInfo(g->pool_stream->stream(),
				&cs, &cid));

			for (int d : extra) {
				graph_state::side sd;
				c10::cuda::CUDAGuard sg((c10::DeviceIndex)d);

				sd.dev = d;
				c10::cuda::CUDACachingAllocator::createOrIncrefPool(
					(c10::DeviceIndex)d, g->pool_id);
				g->side_pools = true;
				c10::cuda::CUDACachingAllocator::beginAllocateToPool(
					(c10::DeviceIndex)d, g->pool_id,
					[cid](cudaStream_t st) {
						cudaStreamCaptureStatus s2;
						unsigned long long id2;

						return cudaStreamGetCaptureInfo(st, &s2, &id2) == cudaSuccess &&
						       s2 == cudaStreamCaptureStatusActive && id2 == cid;
					});
				sd.prev = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)d);
				sd.stream = c10::cuda::getStreamFromPool(false, (c10::DeviceIndex)d);
				graph_link(g->pool_stream->stream(), g->dev_index, sd.stream->stream());
				c10::cuda::setCurrentCUDAStream(*sd.stream);
				g->sides.push_back(sd);
			}
		}
	}
	catch (const std::exception &e) {
		if (g->prev_stream)
			c10::cuda::setCurrentCUDAStream(*g->prev_stream);

		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * capture_end(): this
 *
 * Stops recording and instantiates the graph.
 */
static uc_value_t *
graph_capture_end(uc_vm_t *vm, size_t nargs)
{
	graph_state *g = graph_this(vm);

	if (!g)
		return NULL;

	if (!g->capturing) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"not capturing");
		return NULL;
	}

	/* CUDAGraph::capture_end checks the CURRENT device's current stream
	 * against the capture stream: end on the capture's device (the
	 * guard in capture_begin is scoped to that call, so the current
	 * device is back to the default here — capturing on cuda:1 failed
	 * with "Capture must end on the same stream it began on") */
	c10::cuda::CUDAGuard dev_guard((c10::DeviceIndex)g->dev_index);

	graph_end_sides(g);

	try {
		g->graph->capture_end();
	}
	catch (const std::exception &e) {
		if (g->prev_stream)
			c10::cuda::setCurrentCUDAStream(*g->prev_stream);

		g->capturing = false;
		g_active = NULL;
		RAISE(e.what());
	}

	g->capturing = false;
	g_active = NULL;

	if (g->prev_stream)
		c10::cuda::setCurrentCUDAStream(*g->prev_stream);

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * replay(): this
 *
 * Replays the captured work with the current buffer contents.
 */
static uc_value_t *
graph_replay(uc_vm_t *vm, size_t nargs)
{
	graph_state *g = graph_this(vm);
	uc_value_t *uv;
	int64_t count = 1;

	if (!g)
		return NULL;

	if (nargs >= 1) {
		uv = uc_fn_arg(0);

		if (ucv_type(uv) != UC_INTEGER || ucv_int64_get(uv) < 1) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"count must be a positive integer");
			return NULL;
		}

		count = ucv_int64_get(uv);
	}

	try {
		c10::cuda::OptionalCUDAGuard dev_guard(
			g->dev_index >= 0 ? c10::optional<c10::DeviceIndex>(g->dev_index)
					  : c10::nullopt);

		for (int64_t i = 0; i < count; i++)
			g->graph->replay();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * free(): this
 *
 * Explicitly releases the graph (same as letting the GC collect it).
 */
static uc_value_t *
graph_free_method(uc_vm_t *vm, size_t nargs)
{
	void **slot = ucv_resource_dataptr(_uc_fn_this_res(vm), GRAPH_TYPE);

	if (slot && *slot) {
		graph_free(static_cast<graph_state *>(*slot));
		*slot = NULL;
	}

	return NULL;
}

static const uc_function_list_t graph_methods[] = {
	{ "capture_begin",	graph_capture_begin },
	{ "capture_end",	graph_capture_end },
	{ "replay",		graph_replay },
	{ "free",		graph_free_method },
};

/*
 * where(cond, a, b): torch.tensor
 *
 * Elementwise select: cond (bool tensor), a and b tensors of
 * broadcast-compatible shapes. All-shape-static, value-dependent —
 * the graph-safe way to build additive masks from a persistent
 * position tensor.
 */
static uc_value_t *
torch_where(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *cond, *a, *b;

	if (nargs != 3) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (cond, a, b)");
		return NULL;
	}

	if (!(cond = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(a = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(b = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	try {
		if (cond->scalar_type() != at::kBool) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"where condition must be a bool tensor");
			return NULL;
		}

		at::Tensor r = at::where(*cond, *a, *b);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* --- safetensors -------------------------------------------------------- */

static void
stfile_free(void *data)
{
	st_file_t *f = static_cast<st_file_t *>(data);

	if (f) {
		st_close(f);
		free(f);
	}
}

static st_file_t *
stfile_this(uc_vm_t *vm)
{
	auto *f = static_cast<st_file_t *>(uc_fn_thisval(STFILE_TYPE));

	if (!f)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"called on non-stfile value");

	return f;
}

/*
 * st(path: string): torch.stfile
 *
 * Memory-maps a .safetensors file and parses its header.
 */
static uc_value_t *
torch_st_open(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *path = uc_fn_arg(0);

	if (nargs != 1 || ucv_type(path) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a path string");
		return NULL;
	}

	st_file_t *f = static_cast<st_file_t *>(malloc(sizeof(*f)));

	if (!f)
		return NULL;

	memset(f, 0, sizeof(*f));

	if (st_open(ucv_string_get(path), f) < 0) {
		free(f);
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "cannot open file");
		return NULL;
	}

	return uc_resource_new(ucv_resource_type_lookup(vm, STFILE_TYPE), f);
}

/*
 * st_list(f: torch.stfile): string[]
 */
static uc_value_t *
torch_st_list(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);

	if (!f)
		return NULL;

	size_t n = st_count(f);
	uc_value_t *arr = ucv_array_new_length(vm, n);

	for (size_t i = 0; i < n; i++) {
		const char *name = st_name(f, i);

		if (name)
			ucv_array_push(arr, ucv_string_new(name));
	}

	return arr;
}

/*
 * st_info(f: torch.stfile, name: string): object
 *
 * Returns { dtype: string, shape: number[], nbytes: number }.
 */
static uc_value_t *
torch_st_info(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);
	uc_value_t *name = uc_fn_arg(0);
	const char *dtype;
	int64_t *shape;
	size_t ndim, offset, nbytes;

	if (!f)
		return NULL;

	if (nargs != 1 || ucv_type(name) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a tensor name");
		return NULL;
	}

	if (st_lookup(f, ucv_string_get(name), &dtype, &shape, &ndim,
	              &offset, &nbytes) == ST_NOT_FOUND) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "no such tensor");
		return NULL;
	}

	uc_value_t *info = ucv_object_new(vm);
	uc_value_t *shapearr = ucv_array_new_length(vm, ndim);

	for (size_t i = 0; i < ndim; i++)
		ucv_array_push(shapearr, ucv_int64_new(shape[i]));

	ucv_object_add(info, "dtype", ucv_string_new(dtype));
	ucv_object_add(info, "shape", shapearr);
	ucv_object_add(info, "nbytes", ucv_int64_new((int64_t)nbytes));

	free(shape);

	return info;
}

/* Map a safetensors dtype string to an at::ScalarType. Returns true when
 * known, false for dtypes we cannot represent yet. */
static bool
st_dtype_to_at(const char *dt, at::ScalarType *out)
{
	if (!strcmp(dt, "F64"))  { *out = at::kDouble;   return true; }
	if (!strcmp(dt, "F32"))  { *out = at::kFloat;    return true; }
	if (!strcmp(dt, "F16"))  { *out = at::kHalf;     return true; }
	if (!strcmp(dt, "BF16")) { *out = at::kBFloat16; return true; }
	if (!strcmp(dt, "I64"))  { *out = at::kLong;     return true; }
	if (!strcmp(dt, "I32"))  { *out = at::kInt;      return true; }
	if (!strcmp(dt, "I16"))  { *out = at::kShort;    return true; }
	if (!strcmp(dt, "I8"))   { *out = at::kChar;     return true; }
	if (!strcmp(dt, "U8"))   { *out = at::kByte;     return true; }
	if (!strcmp(dt, "U16"))  { *out = at::kUInt16;  return true; }
	if (!strcmp(dt, "U32"))  { *out = at::kUInt32;  return true; }
	if (!strcmp(dt, "U64"))  { *out = at::kUInt64;  return true; }
	if (!strcmp(dt, "BOOL")) { *out = at::kBool;     return true; }

	return false;
}

/* Map an at::ScalarType to its safetensors dtype string (the write
 * counterpart of st_dtype_to_at). */
static const char *
at_to_st_dtype(at::ScalarType t)
{
	switch (t) {
	case at::kDouble:	return "F64";
	case at::kFloat:	return "F32";
	case at::kHalf:		return "F16";
	case at::kBFloat16:	return "BF16";
	case at::kLong:		return "I64";
	case at::kInt:		return "I32";
	case at::kShort:	return "I16";
	case at::kChar:		return "I8";
	case at::kByte:		return "U8";
	case at::kUInt16:	return "U16";
	case at::kUInt32:	return "U32";
	case at::kUInt64:	return "U64";
	case at::kBool:		return "BOOL";
	default:		return NULL;
	}
}

/*
 * st_save(path, entries): number
 *
 * Write a safetensors file: entries is an array of [name, tensor] pairs
 * (name unique, matching /^[A-Za-z0-9_.-]+$/ so the JSON header needs no
 * escaping). Tensors are made contiguous and copied to the host (a
 * shutdown-path sync is fine), then written as
 * <8-byte LE header length><header JSON><tensor bytes> — readable back
 * with torch.st() / h2d_file and by any safetensors consumer. Returns
 * the number of tensors written.
 */
static uc_value_t *
torch_st_save(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *pathv = uc_fn_arg(0), *entsv = uc_fn_arg(1);

	if (nargs != 2 || ucv_type(pathv) != UC_STRING ||
	    ucv_type(entsv) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (path, [[name, tensor], ...])");
		return NULL;
	}

	const char *path = ucv_string_get(pathv);
	size_t n = ucv_array_length(entsv);

	/* pass 1: validate the entries, make contiguous host copies and
	 * measure; offsets are assigned in entry order */
	std::vector<at::Tensor> host;
	std::vector<std::string> names;
	std::vector<const char *> dts;
	std::vector<size_t> bytes;
	size_t total = 0;

	host.reserve(n);
	names.reserve(n);
	dts.reserve(n);
	bytes.reserve(n);

	for (size_t i = 0; i < n; i++) {
		char why[256];
		uc_value_t *ent = ucv_array_get(entsv, i);

		if (!ent || ucv_type(ent) != UC_ARRAY ||
		    ucv_array_length(ent) != 2) {
			snprintf(why, sizeof(why), "st_save: entry %zu must be "
				"[name, tensor]", i);
			RAISE(why);
		}

		uc_value_t *nv = ucv_array_get(ent, 0);

		if (ucv_type(nv) != UC_STRING) {
			snprintf(why, sizeof(why), "st_save: entry %zu name must "
				"be a string", i);
			RAISE(why);
		}

		const char *nm = ucv_string_get(nv);

		for (const char *c = nm; *c; c++)
			if (!((*c >= 'A' && *c <= 'Z') || (*c >= 'a' && *c <= 'z') ||
			      (*c >= '0' && *c <= '9') || *c == '_' || *c == '.' ||
			      *c == '-')) {
				snprintf(why, sizeof(why), "st_save: entry %zu name "
					"has characters outside [A-Za-z0-9_.-]", i);
				RAISE(why);
			}

		for (size_t k = 0; k < names.size(); k++)
			if (names[k] == nm) {
				snprintf(why, sizeof(why), "st_save: duplicate "
					"name '%s'", nm);
				RAISE(why);
			}

		at::Tensor *t = tensor_arg(vm, ucv_array_get(ent, 1));

		if (!t)
			return NULL;

		const char *dt = at_to_st_dtype(t->scalar_type());

		if (!dt) {
			snprintf(why, sizeof(why), "st_save: '%s' has an "
				"unsupported dtype", nm);
			RAISE(why);
		}

		try {
			TensorDeviceGuard dg(*t);

			host.push_back(t->contiguous().cpu());
		}
		catch (const std::exception &e) {
			RAISE(e.what());
		}

		names.push_back(nm);
		dts.push_back(dt);
		bytes.push_back(host.back().numel() * host.back().element_size());
		total += bytes.back();
	}

	/* build the header: offsets in entry order, relative to the data
	 * region start */
	std::string hdr = "{";
	size_t off = 0;

	for (size_t i = 0; i < n; i++) {
		char ent[512];
		int m = snprintf(ent, sizeof(ent),
			"%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
			i ? "," : "", names[i].c_str(), dts[i]);

		hdr.append(ent, (size_t)m);

		const auto &sz = host[i].sizes();

		for (size_t d = 0; d < sz.size(); d++) {
			m = snprintf(ent, sizeof(ent), "%s%lld",
				d ? "," : "", (long long)sz[d]);

			hdr.append(ent, (size_t)m);
		}

		m = snprintf(ent, sizeof(ent),
			"],\"data_offsets\":[%zu,%zu]}", off, off + bytes[i]);

		hdr.append(ent, (size_t)m);
		off += bytes[i];
	}

	hdr += "}";

	if (hdr.size() > 0x7fffffff)
		RAISE("st_save: header too large");

	uint64_t hlen = (uint64_t)hdr.size();
	char why[256];
	FILE *fp = fopen(path, "wb");

	if (!fp) {
		snprintf(why, sizeof(why), "st_save: cannot open '%s' for "
			"writing", path);
		RAISE(why);
	}

	bool ok = true;

	ok = ok && fwrite(&hlen, 1, sizeof(hlen), fp) == sizeof(hlen);
	ok = ok && fwrite(hdr.data(), 1, hdr.size(), fp) == hdr.size();

	for (size_t i = 0; ok && i < n; i++) {
		size_t nb = bytes[i];

		ok = (nb == 0) || fwrite(host[i].data_ptr(), 1, nb, fp) == nb;
	}

	if (fclose(fp) != 0)
		ok = false;

	if (!ok) {
		unlink(path);
		snprintf(why, sizeof(why), "st_save: write to '%s' failed", path);
		RAISE(why);
	}

	uc_value_t *rc = ucv_int64_new((int64_t)n);

	return rc;
}

/*
 * st_tensor(f: torch.stfile, name: string): torch.tensor
 *
 * Zero-copy CPU view of the tensor's storage inside the mmap. The returned
 * tensor keeps the stfile resource alive (pinned via the resource's GC-
 * marked value slot), so the view stays valid as long as the tensor does.
 */
static uc_value_t *
torch_st_tensor(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);
	uc_value_t *name = uc_fn_arg(0);
	const char *dtype;
	int64_t *shape;
	size_t ndim, offset, nbytes;

	if (!f)
		return NULL;

	if (nargs != 1 || ucv_type(name) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected a tensor name");
		return NULL;
	}

	int rc = st_lookup(f, ucv_string_get(name), &dtype, &shape, &ndim,
		&offset, &nbytes);

	if (rc == ST_NOT_FOUND) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "no such tensor");
		return NULL;
	}

	if (rc < 0) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"malformed safetensors entry for '%s'",
			ucv_string_get(name));
		return NULL;
	}

	at::ScalarType st;

	if (!st_dtype_to_at(dtype, &st)) {
		free(shape);
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"unsupported dtype %s", dtype);
		return NULL;
	}

	std::vector<int64_t> dims(shape, shape + ndim);
	free(shape);

	void *ptr = (char *)f->base + f->data_off + offset;

	at::Tensor t;

	try {
		t = at::from_blob(ptr, dims,
			at::TensorOptions().dtype(st).device(at::kCPU));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	uc_value_t *res = tensor_wrap(vm, std::move(t), 1);

	if (res)
		ucv_resource_value_set(res, 0, ucv_get(_uc_fn_this_res(vm)));

	return res;
}

// Upstream signatures (global namespace, from quant/exl3_gemm.cuh and
// quant/reconstruct.cuh). Compiled into the exl3_kernels object lib.
int exl3_gemm
(
    const at::Tensor &A,
    const at::Tensor &B,
    at::Tensor &C,
    const c10::optional<at::Tensor> &suh,
    const c10::optional<at::Tensor> &A_had,
    const c10::optional<at::Tensor> &svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
);

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K_,
    bool mcg,
    bool mul1
);

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);

/*
 * synchronize()
 *
 * Blocks until all queued CUDA work completes on every CUDA device
 * (multi-GPU: work queued on different devices is independent). The
 * benchmarking primitive: without it, wall-clock around GPU ops
 * measures kernel-launch overhead, not compute.
 */
static uc_value_t *
torch_synchronize(uc_vm_t *vm, size_t nargs)
{
	try {
		int ndev = 0;

		if (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 1) {
			for (int d = 0; d < ndev; d++) {
				c10::cuda::CUDAGuard guard(d);

				c10::cuda::device_synchronize();
			}
		}
		else
			c10::cuda::device_synchronize();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* --- fused kernels (src/amp_kernels.cu) ----------------------------------- */

extern "C" int
amp_rms_norm_fp16(const void *x, const void *w, void *out,
                 long rows, int dim, double eps, cudaStream_t stream);

extern "C" int
amp_rope_fp16(const void *x, const void *cos, const void *sin, void *out,
             long rows, int hd, int seq, cudaStream_t stream);

extern "C" int
amp_silu_mul_fp16(const void *g, const void *u, void *out,
                 long n, cudaStream_t stream);

extern "C" int
amp_rms_norm_res_fp16(const void *x, const void *r, const void *w,
                     void *y, void *n, long rows, int dim, double eps,
                     cudaStream_t stream);

extern "C" int
amp_decode_attn_fp16(const void *q, const void *k_new, const void *v_new,
                    void *k_cache, void *v_cache, const void *cos,
                    const void *sin, void *ws, void *out,
                    const long *pos, int nh, int kvh, int hd, int max_seq,
                    int ns, float scale, cudaStream_t stream);

extern "C" int
amp_argmax_advance_fp16(const void *logits, int n,
                       void *ids, void *pos, void *log, long long log_cap,
                       void *ws, cudaStream_t stream);

extern "C" int
amp_gdn_scan_f32(const void *q, const void *k, const void *v, const void *beta,
                const void *gexp, void *state, void *out, int S, int Hk,
                int Hv, int dk, int dv, float scale, void *states_out,
                int n_out, cudaStream_t stream);

extern "C" long
amp_attn_dec_ws_floats(int S, int Hq, int Hk, int E, int ns);

extern "C" int
amp_attn_dec_fp16(const void *q, const void *k, const void *v, const int *len,
                 void *ws, void *out, int S, int Hq, int Hk, int E,
                 int max_seq, int ns, float scale, cudaStream_t stream);

extern "C" int
amp_attn_dec_mma_fp16(const void *q, const void *k, const void *v,
                     const int *len, void *ws, void *out, int S, int Hq,
                     int Hk, int E, int max_seq, int ns, float scale,
                     cudaStream_t stream);

extern "C" int
amp_qk_norm_rope_fp16(const void *x, long st, long sh, const void *w,
                     const void *cos, const void *sin, void *out, int S, int H,
                     int hd, int rd, float eps, cudaStream_t stream);

extern "C" int
amp_attn_gate_fp16(const void *a, long ast, long ash, const void *g, long gst,
                  long gsh, void *out, int S, int H, int hd, cudaStream_t stream);

extern "C" long
amp_gdn_chunk_ws_floats(int S, int Hv);

extern "C" int
amp_gdn_chunk_f32(const void *q, const void *k, const void *v, const void *beta,
                 const void *gexp, void *state, void *out, void *ws, int S,
                 int Hk, int Hv, int dk, int dv, float scale,
                 cudaStream_t stream);

extern "C" int
amp_gdn_prep_fp16(const void *qkv, void *conv_state, const void *conv_w,
                 void *q, void *k, void *v, int S, int D, int K, int Hk,
                 int Hv, int hd, cudaStream_t stream);

extern "C" int
amp_gdn_gates(const void *a, const void *b, const void *dt_bias,
             const void *nea, void *beta, void *gexp, int S, int Hv,
             cudaStream_t stream);

extern "C" int
amp_gdn_out_norm(const void *core, const void *z, const void *w, void *out,
                long rows, int dv, float eps, cudaStream_t stream);

/* Validate a tensor argument for the fused kernels: fp16 CUDA, and
 * contiguous unless allow_noncontig. Returns NULL after raising on bad
 * input. */
static at::Tensor *
fused_arg(uc_vm_t *vm, uc_value_t *v, const char *what,
          bool allow_noncontig)
{
	at::Tensor *t = tensor_arg(vm, v);

	if (!t)
		return NULL;

	if (t->scalar_type() != at::kHalf ||
	    t->device().type() != at::kCUDA ||
	    (!allow_noncontig && !t->is_contiguous())) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: must be a contiguous fp16 CUDA tensor", what);
		return NULL;
	}

	return t;
}

/*
 * rms_norm(x, weight, eps): torch.tensor
 *
 * Fused RMSNorm over the last dim: x * rsqrt(mean(x^2) + eps) * weight.
 * fp32 variance accumulation, fp16 I/O — the numerics exllamav3's
 * RMSNorm.forward_torch specifies (docs/LESSONS.md L32). One kernel instead
 * of the composite path's ~8.
 */
static uc_value_t *
torch_rms_norm(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *eps_v = uc_fn_arg(2);
	double eps;

	if (nargs != 3 || !arg_number(vm, eps_v, &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, weight, eps)");
		return NULL;
	}

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "rms_norm: x", false);

	if (!x)
		return NULL;

	at::Tensor *w = fused_arg(vm, uc_fn_arg(1), "rms_norm: weight", false);

	if (!w)
		return NULL;

	long dim = (long)w->numel();

	if (x->dim() < 1 || x->size(-1) != dim) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rms_norm: weight (%ld) does not match x's last dim (%lld)",
			dim, (long long)x->size(-1));
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	long rows = (long)(x->numel() / dim);

	try {
		at::Tensor out = at::empty_like(*x);

		int rc = amp_rms_norm_fp16(x->data_ptr(), w->data_ptr(),
			out.data_ptr(), rows, (int)dim, eps,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "rms_norm"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * rope(x, cos, sin): torch.tensor
 *
 * Fused RoPE (rotate_half convention, see ucode/llama.uc rope_apply).
 * x is (..., seq, head_dim); cos/sin are (seq, head_dim). One kernel
 * instead of the composite path's ~7; non-contiguous x (the transposed
 * prefill views) is made contiguous first.
 */
static uc_value_t *
torch_rope(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 3) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, cos, sin)");
		return NULL;
	}

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "rope: x", true);

	if (!x)
		return NULL;

	at::Tensor *cos_t = fused_arg(vm, uc_fn_arg(1), "rope: cos", false);

	if (!cos_t)
		return NULL;

	at::Tensor *sin_t = fused_arg(vm, uc_fn_arg(2), "rope: sin", false);

	if (!sin_t)
		return NULL;

	long hd = (long)x->size(-1);
	long seq = (long)x->size(-2);

	if (x->dim() < 2 || cos_t->dim() != 2 || sin_t->dim() != 2 ||
	    cos_t->size(0) != seq || cos_t->size(1) != hd ||
	    sin_t->size(0) != seq || sin_t->size(1) != hd) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rope: cos/sin must be (seq, head_dim) matching x");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	long rows = (long)(x->numel() / hd);

	try {
		at::Tensor xc = x->is_contiguous() ? *x : x->contiguous();
		at::Tensor out = at::empty_like(xc);

		int rc = amp_rope_fp16(xc.data_ptr(), cos_t->data_ptr(),
			sin_t->data_ptr(), out.data_ptr(), rows,
			(int)hd, (int)seq,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "rope"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * decode_attn(q, k_new, v_new, k_cache, v_cache, cos, sin, ws, out,
 *             pos, scale): torch.tensor
 *
 * Fused single-token decode attention (src/amp_kernels.cu): ropes q and
 * k, writes k/v at the device-computed slot pos, attends over slots
 * [0, pos] with online softmax and GQA, and writes the (nh, hd)
 * output. pos is a device int64 tensor — the kernel reads it at every
 * launch, so the op is CUDA-graph-safe with a static grid.
 *
 * Replaces rope(q) + rope(k) + 2x index_copy_ + full-length masked
 * SDPA + output copy with two kernels (split attention + combine).
 * ws is the fp32 split workspace (kvh, NS, rep, 2 + hd); NS (the
 * split count, from ws.shape()[1]) sets the parallelism of the slot
 * dimension — higher NS = more blocks, shorter per-block slot chains.
 * rep = nh / kvh; hd must be a multiple of 32.
 */
static uc_value_t *
torch_decode_attn(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *scale_v = uc_fn_arg(10);
	double scale;

	if (nargs != 11 || !arg_number(vm, scale_v, &scale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k_new, v_new, k_cache, v_cache, cos, sin, "
			"ws, out, pos, scale)");
		return NULL;
	}

	at::Tensor *q = fused_arg(vm, uc_fn_arg(0), "decode_attn: q", false);
	at::Tensor *kn = fused_arg(vm, uc_fn_arg(1), "decode_attn: k_new", false);
	at::Tensor *vn = fused_arg(vm, uc_fn_arg(2), "decode_attn: v_new", false);
	at::Tensor *kc = fused_arg(vm, uc_fn_arg(3), "decode_attn: k_cache", false);
	at::Tensor *vc = fused_arg(vm, uc_fn_arg(4), "decode_attn: v_cache", false);
	at::Tensor *cos_t = fused_arg(vm, uc_fn_arg(5), "decode_attn: cos", false);
	at::Tensor *sin_t = fused_arg(vm, uc_fn_arg(6), "decode_attn: sin", false);

	if (!cos_t || !sin_t)
		return NULL;

	/* fp32 workspace */
	at::Tensor *ws = tensor_arg(vm, uc_fn_arg(7));

	if (!ws || ws->scalar_type() != at::kFloat ||
	    ws->device().type() != at::kCUDA || !ws->is_contiguous()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"decode_attn: ws must be a contiguous fp32 CUDA tensor");
		return NULL;
	}

	at::Tensor *out = fused_arg(vm, uc_fn_arg(8), "decode_attn: out", false);

	if (!out)
		return NULL;

	at::Tensor *pos = tensor_arg(vm, uc_fn_arg(9));

	if (!pos || pos->scalar_type() != at::kLong ||
	    pos->device().type() != at::kCUDA || pos->numel() != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"decode_attn: pos must be a 1-element int64 CUDA tensor");
		return NULL;
	}

	long nh = q->size(0);
	long hd = q->size(1);
	long kvh = kn->size(0);
	long max_seq = kc->size(1);

	if (q->dim() != 2 || kn->dim() != 2 || vn->dim() != 2 ||
	    kc->dim() != 3 || vc->dim() != 3 || cos_t->dim() != 2 ||
	    sin_t->dim() != 2 || out->numel() != q->numel() ||
	    kc->size(0) != kvh || kc->size(2) != hd ||
	    vc->sizes() != kc->sizes() || kn->size(1) != hd ||
	    vn->sizes() != kn->sizes() || cos_t->size(1) != hd ||
	    sin_t->sizes() != cos_t->sizes() || cos_t->size(0) < max_seq ||
	    nh % kvh != 0 || hd % 32 != 0) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"decode_attn: incompatible shapes");
		return NULL;
	}

	int ns;
	long rep = nh / kvh;

	/* workspace row: [m, l, pad, pad, acc(hd)] with the acc area
	 * 16-byte aligned and the row stride padded to 4 floats */
	long wsrow = 4 + ((hd + 3) & ~3L);

	if (ws->dim() != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"decode_attn: ws must be (kvh, NS, rep, wsrow)");
		return NULL;
	}

	ns = (int)ws->size(1);

	if (ws->numel() < kvh * ns * rep * wsrow) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"decode_attn: workspace too small");
		return NULL;
	}

	TensorDeviceGuard dg(*q);

	try {
		amp_decode_attn_fp16(q->data_ptr(), kn->data_ptr(),
			vn->data_ptr(), kc->data_ptr(), vc->data_ptr(),
			cos_t->data_ptr(), sin_t->data_ptr(), ws->data_ptr(),
			out->data_ptr(),
			(const long *)pos->data_ptr(),
			(int)nh, (int)kvh, (int)hd, (int)max_seq, ns,
			(float)scale,
			c10::cuda::getCurrentCUDAStream().stream());
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * argmax_advance(logits, ids, pos, log, ws?): null
 *
 * One kernel replacing the decode tail: argmax over the vocab
 * (deterministic, first-index tie-break; NaN logits count as lowest),
 * log[pos] = ids, ids = argmax, pos += 1. All tensors CUDA; logits
 * fp16, the state tensors int64. The log write is bounded by log's
 * numel on the device (pos is read from device memory, so the host
 * cannot bound it without breaking graph capture).
 *
 * ws (optional) is the persistent fp32 workspace: 2 * 64 candidate
 * floats + 1 uint32 counter (129 floats). Pass a per-decoder buffer to
 * keep concurrent decoders isolated; without it a per-device static
 * buffer is used (allocated outside capture on first call).
 */
static uc_value_t *
torch_argmax_advance(uc_vm_t *vm, size_t nargs)
{
	if (nargs < 4 || nargs > 5) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (logits, ids, pos, log, ws?)");
		return NULL;
	}

	at::Tensor *logits = fused_arg(vm, uc_fn_arg(0),
		"argmax_advance: logits", false);

	if (!logits)
		return NULL;

	at::Tensor *ids = tensor_arg(vm, uc_fn_arg(1));
	at::Tensor *pos = tensor_arg(vm, uc_fn_arg(2));
	at::Tensor *log = tensor_arg(vm, uc_fn_arg(3));

	if (!ids || !pos || !log || ids->scalar_type() != at::kLong ||
	    pos->scalar_type() != at::kLong ||
	    log->scalar_type() != at::kLong ||
	    ids->device().type() != at::kCUDA ||
	    pos->device().type() != at::kCUDA ||
	    log->device().type() != at::kCUDA ||
	    ids->numel() != 1 || pos->numel() != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"argmax_advance: ids/pos/log must be int64 CUDA "
			"tensors (ids and pos 1-element)");
		return NULL;
	}

	at::Tensor *ws = NULL;

	if (nargs == 5 && ucv_type(uc_fn_arg(4)) != UC_NULL) {
		ws = tensor_arg(vm, uc_fn_arg(4));

		if (!ws || ws->scalar_type() != at::kFloat ||
		    ws->device().type() != at::kCUDA ||
		    !ws->is_contiguous() ||
		    ws->numel() < 2 * 64 + 1) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"argmax_advance: ws must be a contiguous fp32 CUDA "
				"tensor of at least 129 floats");
			return NULL;
		}
	}

	TensorDeviceGuard dg(*logits);

	try {
		int rc = amp_argmax_advance_fp16(logits->data_ptr(),
			(int)logits->numel(),
			ids->data_ptr(), pos->data_ptr(), log->data_ptr(),
			(long long)log->numel(),
			ws ? ws->data_ptr() : NULL,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "argmax_advance"))
			return NULL;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	/* pure state update: nothing meaningful to return */
	return NULL;
}

/*
 * silu_mul(g, u): torch.tensor
 *
 * Fused SwiGLU activation: out = silu(g) * u, elementwise fp16 with
 * fp32 math. One kernel instead of silu + mul.
 */
static uc_value_t *
torch_silu_mul(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (gate, up)");
		return NULL;
	}

	at::Tensor *g = fused_arg(vm, uc_fn_arg(0), "silu_mul: gate", false);

	if (!g)
		return NULL;

	at::Tensor *u = fused_arg(vm, uc_fn_arg(1), "silu_mul: up", false);

	if (!u)
		return NULL;

	if (g->numel() != u->numel()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"silu_mul: gate/up size mismatch");
		return NULL;
	}

	TensorDeviceGuard dg(*g);

	try {
		at::Tensor out = at::empty_like(*g);

		int rc = amp_silu_mul_fp16(g->data_ptr(), u->data_ptr(),
			out.data_ptr(), (long)g->numel(),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "silu_mul"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * kv_q8_put(src, q, s, pos)
 * kv_q8_get(q, s, idx) / kv_q8_get(q, s, start, rows): torch.tensor
 *
 * Q8 latent-cache rows (amp_kvq8_* in src/amp_kernels.cu): q (cap, dim) int8
 * codes, s (cap, dim/32) fp16 group scales. put quantizes src (rows, dim)
 * fp16 into rows pos.. (an integer start) or pos[i] (an int64 tensor of
 * `rows` destination rows — graph-safe, the device-side position); get
 * returns the dequantized (rows, dim) fp16 rows idx[i] / start..start+rows.
 */
/* RAISE with printf-style arguments (the handler's `vm` in scope) */
#define RAISEF(...) \
	do { uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, __VA_ARGS__); return NULL; } while (0)

/*
 * gdn_scan(q, k, v, beta, gexp, state, scale): torch.tensor
 *
 * Fused gated delta rule over a chunk of S tokens (src/amp_kernels.cu),
 * one launch for the whole chunk. All fp32 CUDA contiguous on one device:
 *
 *   q, k   (S, Hk, dk)   l2-normalized; v head h reads q/k head
 *                        h / (Hv / Hk) (no head repeat needed)
 *   v      (S, Hv, dv)
 *   beta   (S, Hv)
 *   gexp   (S, Hv)       per-token decay exp(g)
 *   state  (Hv, dk, dv)  updated in place
 *
 * Returns out (S, Hv, dv) fp32. dk must be 64 or 128, dv a multiple of
 * 64. Same math as ucode/qwen35.uc delta_rule_scan (the reference).
 */
static at::Tensor *
f32_arg(uc_vm_t *vm, uc_value_t *v, const char *what)
{
	at::Tensor *t = tensor_arg(vm, v);

	if (!t)
		return NULL;

	if (t->scalar_type() != at::kFloat || !t->is_cuda() ||
	    !t->is_contiguous()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: must be a contiguous float32 CUDA tensor", what);
		return NULL;
	}

	return t;
}

static uc_value_t *
gdn_scan_impl(uc_vm_t *vm, size_t nargs, bool chunked)
{
	double scale;

	if ((nargs != 7 && nargs != 8) || !arg_number(vm, uc_fn_arg(6), &scale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, beta, gexp, state, scale, states_out?)");
		return NULL;
	}

	at::Tensor *q = f32_arg(vm, uc_fn_arg(0), "gdn_scan: q");
	at::Tensor *k = q ? f32_arg(vm, uc_fn_arg(1), "gdn_scan: k") : NULL;
	at::Tensor *v = k ? f32_arg(vm, uc_fn_arg(2), "gdn_scan: v") : NULL;
	at::Tensor *beta = v ? f32_arg(vm, uc_fn_arg(3), "gdn_scan: beta") : NULL;
	at::Tensor *gexp = beta ? f32_arg(vm, uc_fn_arg(4), "gdn_scan: gexp") : NULL;
	at::Tensor *st = gexp ? f32_arg(vm, uc_fn_arg(5), "gdn_scan: state") : NULL;

	if (!st)
		return NULL;

	/* states_out (n, Hv, dk, dv) fp32: the state after each of the
	 * first n tokens (speculative verify), sequential kernel only */
	at::Tensor *so = NULL;
	uc_value_t *sov = uc_fn_arg(7);

	if (nargs == 8 && ucv_type(sov) != UC_NULL) {
		if (chunked) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"gdn_chunk: states_out is not supported");
			return NULL;
		}

		if (!(so = f32_arg(vm, sov, "gdn_scan: states_out")))
			return NULL;

		if (so->dim() != 4 || so->size(0) > q->size(0) ||
		    so->size(1) != st->size(0) || so->size(2) != st->size(1) ||
		    so->size(3) != st->size(2) || so->get_device() != st->get_device()) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"gdn_scan: states_out must be (n <= S, Hv, dk, dv) on the state's device");
			return NULL;
		}
	}

	if (q->dim() != 3 || k->sizes() != q->sizes() || v->dim() != 3 ||
	    v->size(0) != q->size(0) || st->dim() != 3 ||
	    st->size(0) != v->size(1) || st->size(1) != q->size(2) ||
	    st->size(2) != v->size(2) || beta->numel() != v->size(0) * v->size(1) ||
	    gexp->numel() != beta->numel()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_scan: incompatible shapes");
		return NULL;
	}

	int dev = q->get_device();

	if (k->get_device() != dev || v->get_device() != dev ||
	    beta->get_device() != dev || gexp->get_device() != dev ||
	    st->get_device() != dev) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_scan: all tensors must be on the same device");
		return NULL;
	}

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty({ v->size(0), v->size(1), v->size(2) },
			v->options());

		int rc;

		if (chunked) {
			at::Tensor ws = at::empty({ (int64_t)amp_gdn_chunk_ws_floats(
				(int)q->size(0), (int)v->size(1)) }, v->options());

			rc = amp_gdn_chunk_f32(q->data_ptr(), k->data_ptr(),
				v->data_ptr(), beta->data_ptr(), gexp->data_ptr(),
				st->data_ptr(), out.data_ptr(), ws.data_ptr(),
				(int)q->size(0), (int)q->size(1), (int)v->size(1),
				(int)q->size(2), (int)v->size(2), (float)scale,
				c10::cuda::getCurrentCUDAStream().stream());
		}
		else {
			rc = amp_gdn_scan_f32(q->data_ptr(), k->data_ptr(),
				v->data_ptr(), beta->data_ptr(), gexp->data_ptr(),
				st->data_ptr(), out.data_ptr(), (int)q->size(0),
				(int)q->size(1), (int)v->size(1), (int)q->size(2),
				(int)v->size(2), (float)scale,
				so ? so->data_ptr() : NULL, so ? (int)so->size(0) : 0,
				c10::cuda::getCurrentCUDAStream().stream());
		}

		if (!launch_ok(vm, rc, chunked ? "gdn_chunk" : "gdn_scan"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
torch_gdn_scan(uc_vm_t *vm, size_t nargs)
{
	return gdn_scan_impl(vm, nargs, false);
}

/*
 * gdn_chunk(q, k, v, beta, gexp, state, scale): torch.tensor
 *
 * Same contract as gdn_scan, computed in 64-token blocks (WY/UT form:
 * block-parallel factorization + a per-block state recurrence; see
 * src/amp_kernels.cu). Prefill path; dk == dv == 128 only.
 */
static uc_value_t *
torch_gdn_chunk(uc_vm_t *vm, size_t nargs)
{
	return gdn_scan_impl(vm, nargs, true);
}

/*
 * gdn_prep(qkv, conv_state, conv_w, Hk, Hv, head_dim): [q, k, v]
 *
 * Fused GatedDeltaNet input preparation (src/amp_kernels.cu): causal
 * conv1d over [conv_state | chunk] + SiLU, split into q/k/v heads, fp32,
 * l2-normalized q/k; updates conv_state (the last K-1 inputs) in place.
 *
 *   qkv        (1, S, D) fp16, D = (2 * Hk + Hv) * head_dim
 *   conv_state (1, D, K-1) fp16
 *   conv_w     (D, K) fp16
 *
 * Returns q, k (S, Hk, head_dim) and v (S, Hv, head_dim), fp32.
 */
static uc_value_t *
torch_gdn_prep(uc_vm_t *vm, size_t nargs)
{
	int64_t Hk, Hv, hd;

	if (nargs != 6 || !arg_int(vm, uc_fn_arg(3), &Hk) ||
	    !arg_int(vm, uc_fn_arg(4), &Hv) || !arg_int(vm, uc_fn_arg(5), &hd)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (qkv, conv_state, conv_w, Hk, Hv, head_dim)");
		return NULL;
	}

	at::Tensor *qkv = fused_arg(vm, uc_fn_arg(0), "gdn_prep: qkv", false);
	at::Tensor *cs = qkv ? fused_arg(vm, uc_fn_arg(1), "gdn_prep: conv_state", false) : NULL;
	at::Tensor *cw = cs ? fused_arg(vm, uc_fn_arg(2), "gdn_prep: conv_w", false) : NULL;

	if (!cw)
		return NULL;

	int64_t S = qkv->size(-2), D = qkv->size(-1);

	if (qkv->numel() != S * D || cw->dim() != 2 || cw->size(0) != D ||
	    cs->numel() != D * (cw->size(1) - 1) || D != (2 * Hk + Hv) * hd) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_prep: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*qkv);

	try {
		auto f32 = qkv->options().dtype(at::kFloat);
		at::Tensor q = at::empty({ S, Hk, hd }, f32);
		at::Tensor k = at::empty({ S, Hk, hd }, f32);
		at::Tensor v = at::empty({ S, Hv, hd }, f32);

		int rc = amp_gdn_prep_fp16(qkv->data_ptr(), cs->data_ptr(),
			cw->data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
			(int)S, (int)D, (int)cw->size(1), (int)Hk, (int)Hv, (int)hd,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "gdn_prep"))
			return NULL;

		uc_value_t *arr = ucv_array_new(vm);

		ucv_array_push(arr, tensor_wrap(vm, std::move(q), 0));
		ucv_array_push(arr, tensor_wrap(vm, std::move(k), 0));
		ucv_array_push(arr, tensor_wrap(vm, std::move(v), 0));

		return arr;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * gdn_gates(a, b, dt_bias, neg_exp_a_log): [beta, gexp]
 *
 * beta = sigmoid(b), gexp = exp(neg_exp_a_log * softplus(a + dt_bias)).
 * a, b (…, Hv) fp16 contiguous; dt_bias, neg_exp_a_log (Hv) fp32.
 * Returns beta, gexp as (S, Hv) fp32.
 */
static uc_value_t *
torch_gdn_gates(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (a, b, dt_bias, neg_exp_a_log)");
		return NULL;
	}

	at::Tensor *a = fused_arg(vm, uc_fn_arg(0), "gdn_gates: a", false);
	at::Tensor *b = a ? fused_arg(vm, uc_fn_arg(1), "gdn_gates: b", false) : NULL;
	at::Tensor *dtb = b ? f32_arg(vm, uc_fn_arg(2), "gdn_gates: dt_bias") : NULL;
	at::Tensor *nea = dtb ? f32_arg(vm, uc_fn_arg(3), "gdn_gates: neg_exp_a_log") : NULL;

	if (!nea)
		return NULL;

	int64_t Hv = dtb->numel();

	if (b->numel() != a->numel() || nea->numel() != Hv || Hv == 0 ||
	    a->numel() % Hv != 0) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_gates: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*a);

	try {
		int64_t S = a->numel() / Hv;
		auto f32 = a->options().dtype(at::kFloat);
		at::Tensor beta = at::empty({ S, Hv }, f32);
		at::Tensor gexp = at::empty({ S, Hv }, f32);

		int rc = amp_gdn_gates(a->data_ptr(), b->data_ptr(), dtb->data_ptr(),
			nea->data_ptr(), beta.data_ptr(), gexp.data_ptr(), (int)S,
			(int)Hv, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "gdn_gates"))
			return NULL;

		uc_value_t *arr = ucv_array_new(vm);

		ucv_array_push(arr, tensor_wrap(vm, std::move(beta), 0));
		ucv_array_push(arr, tensor_wrap(vm, std::move(gexp), 0));

		return arr;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * gdn_out_norm(core, z, weight, eps): torch.tensor
 *
 * GatedDeltaNet output: per (token, head) RMSNorm over the v head dim,
 * weight multiply, then * silu(z). core (S, Hv, dv) fp32 contiguous;
 * z (…, Hv * dv) fp16; weight (dv) fp16. Returns (1, S, Hv * dv) fp16.
 */
static uc_value_t *
torch_gdn_out_norm(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs != 4 || !arg_number(vm, uc_fn_arg(3), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (core, z, weight, eps)");
		return NULL;
	}

	at::Tensor *core = f32_arg(vm, uc_fn_arg(0), "gdn_out_norm: core");
	at::Tensor *z = core ? fused_arg(vm, uc_fn_arg(1), "gdn_out_norm: z", false) : NULL;
	at::Tensor *w = z ? fused_arg(vm, uc_fn_arg(2), "gdn_out_norm: weight", false) : NULL;

	if (!w)
		return NULL;

	if (core->dim() != 3 || z->numel() != core->numel() ||
	    w->numel() != core->size(2)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_out_norm: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*core);

	try {
		int64_t S = core->size(0), Hv = core->size(1), dv = core->size(2);
		at::Tensor out = at::empty({ 1, S, Hv * dv }, z->options());

		int rc = amp_gdn_out_norm(core->data_ptr(), z->data_ptr(),
			w->data_ptr(), out.data_ptr(), (long)(S * Hv), (int)dv,
			(float)eps, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "gdn_out_norm"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * qk_norm_rope(x, weight, cos, sin, rotary_dim, eps): torch.tensor
 *
 * Fused per-head RMSNorm + partial NEOX RoPE (src/amp_kernels.cu). x is
 * (1, S, H, hd) fp16, may be strided over tokens and heads (e.g. the q
 * half of the interleaved q/gate projection) but contiguous over hd;
 * weight (hd) fp16; cos/sin (S, rotary_dim/2) fp32 contiguous. Returns
 * (1, S, H, hd) fp16 contiguous.
 */
static uc_value_t *
torch_qk_norm_rope(uc_vm_t *vm, size_t nargs)
{
	int64_t rd;
	double eps;

	if (nargs != 6 || !arg_int(vm, uc_fn_arg(4), &rd) ||
	    !arg_number(vm, uc_fn_arg(5), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, weight, cos, sin, rotary_dim, eps)");
		return NULL;
	}

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "qk_norm_rope: x", true);
	at::Tensor *w = x ? fused_arg(vm, uc_fn_arg(1), "qk_norm_rope: weight", false) : NULL;
	at::Tensor *cs = w ? f32_arg(vm, uc_fn_arg(2), "qk_norm_rope: cos") : NULL;
	at::Tensor *sn = cs ? f32_arg(vm, uc_fn_arg(3), "qk_norm_rope: sin") : NULL;

	if (!sn)
		return NULL;

	if (x->dim() != 4 || x->size(0) != 1 || x->stride(3) != 1 ||
	    w->numel() != x->size(3) || cs->dim() != 2 ||
	    cs->size(0) != x->size(1) || cs->size(1) * 2 != rd ||
	    sn->sizes() != cs->sizes()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"qk_norm_rope: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		int64_t S = x->size(1), H = x->size(2), hd = x->size(3);
		at::Tensor out = at::empty({ 1, S, H, hd }, x->options());

		int rc = amp_qk_norm_rope_fp16(x->data_ptr(), (long)x->stride(1),
			(long)x->stride(2), w->data_ptr(), cs->data_ptr(), sn->data_ptr(),
			out.data_ptr(), (int)S, (int)H, (int)hd, (int)rd, (float)eps,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "qk_norm_rope"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * attn_gate(attn, g): torch.tensor
 *
 * out = attn * fp16(sigmoid(g)), attn (1, H, S, hd) and g (1, S, H, hd)
 * fp16, both may be strided (contiguous over hd). Returns (1, S, H * hd)
 * fp16 contiguous — the o_proj input.
 */
static uc_value_t *
torch_attn_gate(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (attn, g)");
		return NULL;
	}

	at::Tensor *a = fused_arg(vm, uc_fn_arg(0), "attn_gate: attn", true);
	at::Tensor *g = a ? fused_arg(vm, uc_fn_arg(1), "attn_gate: g", true) : NULL;

	if (!g)
		return NULL;

	if (a->dim() != 4 || g->dim() != 4 || a->size(1) != g->size(2) ||
	    a->size(2) != g->size(1) || a->size(3) != g->size(3) ||
	    a->stride(3) != 1 || g->stride(3) != 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"attn_gate: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*a);

	try {
		int64_t H = a->size(1), S = a->size(2), hd = a->size(3);
		at::Tensor out = at::empty({ 1, S, H * hd }, a->options());

		int rc = amp_attn_gate_fp16(a->data_ptr(), (long)a->stride(2),
			(long)a->stride(1), g->data_ptr(), (long)g->stride(1),
			(long)g->stride(2), out.data_ptr(), (int)S, (int)H, (int)hd,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "attn_gate"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * attn_decode(q, k_cache, v_cache, len, scale, splits?, mma?):
 * torch.tensor
 *
 * Split-KV decode attention with a DEVICE-side key count (src/
 * amp_kernels.cu amp_attn_dec_kernel / amp_attn_dec_mma_kernel),
 * CUDA-graph-safe: q (S, Hq, E) fp16 contiguous (S <= 5,
 * (Hq/Hk) * S <= 32), k/v caches (Hk, max_seq, E) fp16 contiguous,
 * len (1) int32 on the device = keys in use (row s attends keys
 * [0, len - S + s]). splits: key splits per kv head (default:
 * 2 blocks per SM over the kv heads). mma = 1 selects the
 * tensor-core (mma.sync) variant, which is ~2.2x faster at long
 * key counts (E = 64/128/256); default 0 is the shuffle kernel.
 * Returns (S, Hq, E).
 */
static uc_value_t *
torch_attn_decode(uc_vm_t *vm, size_t nargs)
{
	double scale;
	int64_t splits = 0, use_mma = 0;
	uc_value_t *scv = uc_fn_arg(4), *spv = uc_fn_arg(5);
	uc_value_t *mmav = uc_fn_arg(6);

	if ((nargs != 5 && nargs != 6 && nargs != 7) ||
	    !arg_number(vm, scv, &scale) ||
	    (nargs >= 6 && ucv_type(spv) != UC_NULL && !arg_int(vm, spv, &splits)) ||
	    (nargs == 7 && ucv_type(mmav) != UC_NULL && !arg_int(vm, mmav, &use_mma))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k_cache, v_cache, len, scale, splits?, mma?)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *k = q ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *v = k ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *len = v ? tensor_arg(vm, uc_fn_arg(3)) : NULL;

	if (!len)
		return NULL;

	int dev = q->get_device();

	if (q->dim() != 3 || k->dim() != 3 || v->sizes() != k->sizes() ||
	    q->scalar_type() != at::kHalf || k->scalar_type() != at::kHalf ||
	    v->scalar_type() != at::kHalf || len->scalar_type() != at::kInt ||
	    !q->is_contiguous() || !k->is_contiguous() || !v->is_contiguous() ||
	    q->size(2) != k->size(2) || k->get_device() != dev ||
	    v->get_device() != dev || len->get_device() != dev) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"attn_decode: q (S, Hq, E), k/v (Hk, max_seq, E) contiguous fp16, "
			"len int32, one device");
		return NULL;
	}

	int S = (int)q->size(0), Hq = (int)q->size(1), Hk = (int)k->size(0);
	int E = (int)q->size(2), max_seq = (int)k->size(1);

	TensorDeviceGuard dg(*q);

	try {
		if (splits <= 0) {
			int sms = 0;

			C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
			/* ~4 blocks per SM over the kv heads: enough unique K/V
			 * bytes in flight (the block's warps share rows); measured
			 * best on a 3090 */
			splits = std::max<int64_t>(1, (4 * (int64_t)sms + Hk - 1) / Hk);
		}

		at::Tensor ws = at::empty({ (int64_t)amp_attn_dec_ws_floats(S, Hq, Hk, E, (int)splits) },
			q->options().dtype(at::kFloat));
		at::Tensor out = at::empty_like(*q);
		int rc;

		if (use_mma)
			rc = amp_attn_dec_mma_fp16(q->data_ptr(), k->data_ptr(),
				v->data_ptr(), (const int *)len->data_ptr(),
				ws.data_ptr(), out.data_ptr(), S, Hq, Hk, E, max_seq,
				(int)splits, (float)scale,
				c10::cuda::getCurrentCUDAStream().stream());
		else
			rc = amp_attn_dec_fp16(q->data_ptr(), k->data_ptr(), v->data_ptr(),
				(const int *)len->data_ptr(), ws.data_ptr(), out.data_ptr(), S, Hq,
				Hk, E, max_seq, (int)splits, (float)scale,
				c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "attn_decode"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * device_count(): int — visible CUDA devices (0 without CUDA).
 */
static uc_value_t *
torch_device_count(uc_vm_t *vm, size_t nargs)
{
	return ucv_int64_new(at::cuda::device_count());
}

/*
 * N-way P2P communicator (src/amp_kernels.cu amp_comm_allreduce_kernel).
 *
 * p2p_comm(devices, cap, mode?): torch.comm
 *   devices: array of CUDA indices (rank = position, 2..8 of them);
 *   cap: largest all-reduce payload in fp16 elements; mode: "auto"
 *   (default: one-shot for N <= 3, two-phase above), "oneshot",
 *   "twophase". Allocates per rank (2N + 2) * cap fp16 of buffers and
 *   the flags (zeroed), and enables peer access between all pairs.
 *
 * comm.allreduce(x, rank): torch.tensor — rank's half of the all-reduce
 *   of the ranks' contiguous fp16 partials (x on devices[rank]); every
 *   rank must issue the same sequence of calls. Returns the sum, bit-
 *   identical on all ranks. Graph-capturable (no host sync, no events).
 */
#define AMP_COMM_MAXN 8
#define AMP_COMM_MAXB 16

/* --- stream fork / join (intra-graph concurrency) ------------------------ */

/* Side-stream branches for independent op chains. Slot k of a device is one
 * side stream:
 *   stream_side(dev, k)  the side stream waits for everything queued so far
 *                        on the current stream; subsequent ops (ATen sets
 *                        the stream per device) run on the side stream
 *   stream_main(dev, k)  switch back to the origin stream and record the
 *                        side stream's completion
 *   stream_wait(dev, k)  the origin stream waits for that completion
 * All three are legal during CUDA graph capture (fork/join through events
 * becomes graph dependencies); eagerly they overlap kernels the same way.
 * Tensors used by a branch must stay referenced until stream_wait (the
 * caching allocator knows nothing about the side stream). */
#define AMP_FORK_MAXD 16
#define AMP_FORK_MAXK 4

/*
 * flash_attn_kvcache(q, k, v, cu_q, cu_k, seqused_k, max_k, causal):
 * torch.tensor
 *
 * FlashAttention-2 variable-length entry (_flash_attention_forward) for
 * one sequence with a DEVICE-side key length: q (S, Hq, E) fp16, k/v
 * (max_k, Hk, E) fp16 views of the KV cache (any row/head strides,
 * contiguous over E); cu_q = [0, S], cu_k = [0, max_k] (int32 device),
 * seqused_k (1,) int32 device = keys in use. Attends to keys
 * [0, seqused_k) with the causal mask aligned bottom-right. Shapes are
 * fixed and the length is read on the device, so the op can live inside
 * a captured CUDA graph. Returns (S, Hq, E).
 */
static uc_value_t *
torch_flash_attn_kvcache(uc_vm_t *vm, size_t nargs)
{
	int64_t max_k;
	bool causal = true;

	if (nargs != 8 || !arg_int(vm, uc_fn_arg(6), &max_k) ||
	    !arg_bool(vm, uc_fn_arg(7), &causal)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, cu_q, cu_k, seqused_k, max_k, causal)");
		return NULL;
	}

	at::Tensor *t[6];

	for (int i = 0; i < 6; i++)
		if (!(t[i] = tensor_arg(vm, uc_fn_arg(i))))
			return NULL;

	TensorDeviceGuard dg(*t[0]);

	try {
		auto r = at::_flash_attention_forward(*t[0], *t[1], *t[2], *t[3],
			*t[4], t[0]->size(0), max_k, 0.0, causal, false,
			c10::nullopt, c10::nullopt, c10::nullopt, *t[5], c10::nullopt);

		return tensor_wrap(vm, std::move(std::get<0>(r)), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * flash_attn(q, k, v, causal, scale?): torch.tensor
 *
 * FlashAttention-2 via ATen's _scaled_dot_product_flash_attention. q
 * (B, Hq, L, E), k/v (B, Hk, S, E) fp16 CUDA; returns (B, Hq, L, E).
 * Unlike the public sdpa(is_causal) — top-left aligned, so unusable for
 * a chunk on top of history — the flash kernel aligns the causal mask
 * BOTTOM-RIGHT when L < S: query i sees keys 0 .. S - L + i, exactly a
 * prefill chunk at position S - L. No mask tensor, no materialized
 * score matrix. Raises if the flash kernel rejects the configuration.
 */
static uc_value_t *
torch_flash_attn(uc_vm_t *vm, size_t nargs)
{
	bool causal = false;
	double scale = 0;
	bool has_scale = false;

	if (nargs < 4 || nargs > 5) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, causal, scale?)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *k = q ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *v = k ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!v || !arg_bool(vm, uc_fn_arg(3), &causal))
		return NULL;

	if (nargs == 5 && ucv_type(uc_fn_arg(4)) != UC_NULL) {
		if (!arg_number(vm, uc_fn_arg(4), &scale))
			return NULL;

		has_scale = true;
	}

	TensorDeviceGuard dg(*q);

	try {
		auto r = at::_scaled_dot_product_flash_attention(*q, *k, *v,
			0.0, causal, false,
			has_scale ? c10::optional<double>(scale) : c10::nullopt);

		return tensor_wrap(vm, std::move(std::get<0>(r)), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * rms_norm_res(x, r, weight, eps): [residual, normed]
 *
 * Fused residual add + RMSNorm: y = x + r (fp16, elementwise-exact),
 * n = rmsnorm(y) * weight. Returns BOTH — the updated residual stream
 * and the normed input for the next block. y may alias x (in-place
 * update of the residual stream). One kernel instead of add + rms_norm.
 */
static uc_value_t *
torch_rms_norm_res(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *eps_v = uc_fn_arg(3);
	double eps;

	if (nargs != 4 || !arg_number(vm, eps_v, &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, r, weight, eps)");
		return NULL;
	}

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "rms_norm_res: x", false);

	if (!x)
		return NULL;

	at::Tensor *r = fused_arg(vm, uc_fn_arg(1), "rms_norm_res: r", false);

	if (!r)
		return NULL;

	at::Tensor *w = fused_arg(vm, uc_fn_arg(2), "rms_norm_res: weight", false);

	if (!w)
		return NULL;

	long dim = (long)w->numel();

	if (x->dim() < 1 || r->dim() < 1 || x->size(-1) != dim ||
	    r->size(-1) != dim || r->numel() != x->numel()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rms_norm_res: weight/last-dim mismatch");
		return NULL;
	}

	if (dim % 4 != 0) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rms_norm_res: last dim (%ld) must be a multiple of 4",
			(long)dim);
		return NULL;
	}

	/* the kernel loads 8-byte chunks: base pointers must be 8-byte
	 * aligned (fresh allocations always are; views into larger
	 * buffers are not necessarily) */
	if (((uintptr_t)x->data_ptr() | (uintptr_t)r->data_ptr() |
	     (uintptr_t)w->data_ptr()) & 7) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rms_norm_res: tensors must be 8-byte aligned");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	long rows = (long)(x->numel() / dim);

	try {
		at::Tensor y = at::empty_like(*x);
		at::Tensor n = at::empty_like(*x);

		int rc = amp_rms_norm_res_fp16(x->data_ptr(), r->data_ptr(),
			w->data_ptr(), y.data_ptr(), n.data_ptr(), rows,
			(int)dim, eps,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "rms_norm_res"))
			return NULL;

		uc_value_t *yv = tensor_wrap(vm, std::move(y), 0);
		uc_value_t *nv = tensor_wrap(vm, std::move(n), 0);
		uc_value_t *arr = ucv_array_new(vm);

		ucv_array_push(arr, yv);
		ucv_array_push(arr, nv);

		return arr;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * mgemm(x, B_ptrs, C, suh_ptrs, A_had, svh_ptrs, K, mcg, mul1, opts?):
 * torch.tensor
 *
 * Batched EXL3 gemm: several weights against one shared input in ONE
 * cooperative launch (vendored exl3_mgemm, see exl3_gemm.cu). Weight
 * identities are device pointer tables (int64 tensors):
 *
 *   x        (1, m, k) fp16 CUDA — the shared input
 *   B_ptrs   (num_matrices,) int64 CUDA — trellis data pointers
 *   C        (num_matrices, m, n) fp16 CUDA — stacked output
 *   suh_ptrs (num_matrices,) int64 CUDA
 *   A_had    (num_matrices, m, k) fp16 CUDA — input-Hadamard scratch
 *   svh_ptrs (num_matrices,) int64 CUDA
 *   K        bitrate (number)
 *
 * opts (optional object) selects SLICED mode for unequal output widths
 * (e.g. Q 2048 / K 512 / V 512): every matrix is cut into equal-width
 * column slices scheduled as independent groups; outputs go through
 * c_ptrs instead of C (C is then only a (num_slices, m, width)
 * descriptor):
 *
 *   size_n_list  (num_slices,) int32 — slice width
 *   c_ptrs       (num_slices,) int64 — per-slice output pointers
 *   n_stride_list (num_slices,) int32 — full row width of the slice's
 *                                     output matrix
 *   had_src_list (num_slices,) int32 — source index per slice
 *   num_had_src  number — one A_had slab + suh entry per source
 */
static uc_value_t *
torch_mgemm(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *k_v = uc_fn_arg(6), *mcg_v = uc_fn_arg(7),
		*mul1_v = uc_fn_arg(8), *opts_v = uc_fn_arg(9);
	double Kd;
	bool mcg, mul1;

	if (nargs < 9 || nargs > 10 || !arg_number(vm, k_v, &Kd) ||
	    !arg_bool(vm, mcg_v, &mcg) || !arg_bool(vm, mul1_v, &mul1)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, B_ptrs, C, suh_ptrs, A_had, svh_ptrs, "
			"K, mcg, mul1, opts?)");
		return NULL;
	}

	at::Tensor *A = fused_arg(vm, uc_fn_arg(0), "mgemm: x", false);

	if (!A)
		return NULL;

	/* pointer tables: int64 1-D CUDA */
	at::Tensor *B = tensor_arg(vm, uc_fn_arg(1));

	if (!B || B->scalar_type() != at::kLong ||
	    B->device().type() != at::kCUDA) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"mgemm: B_ptrs must be an int64 CUDA tensor");
		return NULL;
	}

	at::Tensor *C = fused_arg(vm, uc_fn_arg(2), "mgemm: C", false);

	if (!C)
		return NULL;

	at::Tensor *suh = tensor_arg(vm, uc_fn_arg(3));

	if (!suh || suh->scalar_type() != at::kLong ||
	    suh->device().type() != at::kCUDA) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"mgemm: suh_ptrs must be an int64 CUDA tensor");
		return NULL;
	}

	at::Tensor *A_had = fused_arg(vm, uc_fn_arg(4), "mgemm: A_had", false);

	if (!A_had)
		return NULL;

	at::Tensor *svh = tensor_arg(vm, uc_fn_arg(5));

	if (!svh || svh->scalar_type() != at::kLong ||
	    svh->device().type() != at::kCUDA) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"mgemm: svh_ptrs must be an int64 CUDA tensor");
		return NULL;
	}

	c10::optional<at::Tensor> size_n_list, c_ptrs, n_stride_list,
		had_src_list;
	int num_had_src = 0;

	if (nargs == 10) {
		if (ucv_type(opts_v) != UC_OBJECT) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"mgemm: opts must be an object");
			return NULL;
		}

		/* NB: tensor_arg() returns NULL after raising — test it BEFORE
		 * dereferencing (the old `*tensor_arg(...)` pattern segfaulted
		 * the VM on a wrong argument instead of raising). */
		struct { const char *key; c10::optional<at::Tensor> *dst;
			at::ScalarType want; } tabs[] = {
			{ "size_n_list", &size_n_list, at::kInt },
			{ "c_ptrs", &c_ptrs, at::kLong },
			{ "n_stride_list", &n_stride_list, at::kInt },
			{ "had_src_list", &had_src_list, at::kInt },
		};

		for (size_t ti = 0; ti < sizeof(tabs) / sizeof(tabs[0]); ti++) {
			uc_value_t *v = ucv_object_get(opts_v, tabs[ti].key, NULL);
			at::Tensor *t = v ? tensor_arg(vm, v) : NULL;

			if (!t)
				return NULL;

			if (t->scalar_type() != tabs[ti].want ||
			    t->device().type() != at::kCUDA ||
			    !t->is_contiguous()) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"mgemm: opts.%s must be a contiguous %s CUDA tensor",
					tabs[ti].key,
					tabs[ti].want == at::kInt ? "int32" : "int64");
				return NULL;
			}

			*tabs[ti].dst = *t;
		}

		uc_value_t *v = ucv_object_get(opts_v, "num_had_src", NULL);

		if (!v || ucv_type(v) != UC_INTEGER) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"mgemm: num_had_src missing");
			return NULL;
		}

		num_had_src = (int)ucv_int64_get(v);
	}

	TensorDeviceGuard dg(*A);

	try {
		exl3_mgemm(*A, *B, *C, *suh, *A_had, *svh,
			c10::nullopt, c10::nullopt, (float)Kd,
			-1, (uint32_t)(mcg ? 1 : 0), (uint32_t)(mul1 ? 1 : 0),
			-1, -1, 0,
			1, size_n_list, c_ptrs, n_stride_list,
			had_src_list, num_had_src);

		/* in-place convention: return the destination (the stacked C
		 * carrier in sliced mode) */
		return ucv_get(uc_fn_arg(2));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * ptr_array(values: array): torch.tensor
 *
 * Builds a 1-D int64 CUDA tensor from an array of ucode integers —
 * the device pointer tables mgemm consumes. Values must be exact
 * integers (device pointers fit ucode's integer range).
 */
static uc_value_t *
torch_ptr_array(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *arr = uc_fn_arg(0);
	size_t n, i;

	int64_t didx = -1;

	if ((nargs != 1 && nargs != 2) || ucv_type(arr) != UC_ARRAY ||
	    (nargs == 2 && ucv_type(uc_fn_arg(1)) != UC_NULL && !arg_int(vm, uc_fn_arg(1), &didx))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (array of integers, device?)");
		return NULL;
	}

	n = ucv_array_length(arr);

	std::vector<int64_t> vals;

	vals.reserve(n);

	for (i = 0; i < n; i++) {
		uc_value_t *v = ucv_array_get(arr, i);

		if (ucv_type(v) == UC_INTEGER)
			vals.push_back(ucv_int64_get(v));
		else if (ucv_type(v) == UC_DOUBLE) {
			double d = ucv_double_get(v);

			/* these become device pointers: a fractional value is a
			 * bug in the caller, not something to truncate */
			if (d != (int64_t)d) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"ptr_array: entry %zu is not an integer",
					(unsigned long)i);
				return NULL;
			}

			vals.push_back((int64_t)d);
		}
		else {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"ptr_array: entry %zu is not an integer",
				(unsigned long)i);
			return NULL;
		}
	}

	try {
		at::Tensor t = at::from_blob(vals.data(),
			{(int64_t)n},
			at::TensorOptions().dtype(at::kLong).device(at::kCPU));

		at::Tensor dev = (didx >= 0) ? t.to(at::Device(at::kCUDA, (c10::DeviceIndex)didx))
		                             : t.to(at::kCUDA);

		return tensor_wrap(vm, std::move(dev), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* Pinned staging ring shared by h2d and h2d_file (RAII: the destructor
 * drains the stream BEFORE the pinned stage dies, so an exception
 * mid-loop can no longer free a buffer an in-flight cudaMemcpyAsync is
 * still reading, and the events are always destroyed). */
struct h2d_ring {
	static constexpr size_t chunk_bytes = 32 * 1024 * 1024;
	static constexpr size_t num_slots = 4;

	/* declaration order matters: slot_done is destroyed BEFORE stage,
	 * and the destructor syncs before either runs */
	std::vector<cudaEvent_t> slot_done;
	at::Tensor stage;
	uint8_t *base;
	cudaStream_t stream;
	size_t chunk_idx;

	h2d_ring() : base(NULL), stream(NULL), chunk_idx(0) {}

	~h2d_ring()
	{
		if (stream)
			cudaStreamSynchronize(stream);   /* best effort */

		for (size_t s = 0; s < slot_done.size(); s++)
			if (slot_done[s])
				cudaEventDestroy(slot_done[s]);
	}

	bool init(at::Device dev, std::string *err)
	{
		c10::cuda::CUDAGuard dev_guard(dev);

		try {
			/* pinned staging ring (L9: explicit device) */
			stage = at::empty(
				{ (int64_t)(num_slots * chunk_bytes) },
				at::TensorOptions().dtype(at::kByte)
					.device(at::kCPU).pinned_memory(true));
		}
		catch (const std::exception &e) {
			*err = e.what();
			return false;
		}

		base = stage.data_ptr<uint8_t>();
		stream = c10::cuda::getCurrentCUDAStream(dev.index()).stream();
		slot_done.resize(num_slots);

		for (size_t s = 0; s < num_slots; s++) {
			cudaError_t e = cudaEventCreateWithFlags(&slot_done[s],
				cudaEventDisableTiming);

			if (e != cudaSuccess) {
				*err = cudaGetErrorString(e);
				return false;
			}
		}

		return true;
	}

	/* Wait until the next slot is reusable; returns its host address. */
	uint8_t *slot_acquire(std::string *err)
	{
		size_t slot = chunk_idx % num_slots;

		/* slot reuse guard: the copy reading this slot completed
		 * before we overwrite it */
		if (chunk_idx >= num_slots) {
			cudaError_t e = cudaEventSynchronize(slot_done[slot]);

			if (e != cudaSuccess) {
				*err = cudaGetErrorString(e);
				return NULL;
			}
		}

		return base + slot * chunk_bytes;
	}

	/* Enqueue the copy of one chunk from the acquired slot. */
	bool slot_submit(void *dst, size_t dst_off, size_t len,
		std::string *err)
	{
		size_t slot = chunk_idx % num_slots;
		cudaError_t e = cudaMemcpyAsync(
			(uint8_t *)dst + dst_off, base + slot * chunk_bytes,
			len, cudaMemcpyHostToDevice, stream);

		if (e != cudaSuccess) {
			*err = cudaGetErrorString(e);
			return false;
		}

		e = cudaEventRecord(slot_done[slot], stream);

		if (e != cudaSuccess) {
			*err = cudaGetErrorString(e);
			return false;
		}

		chunk_idx++;

		return true;
	}

	/* One chunk: host memcpy into the slot, then async H2D. */
	bool push_mem(void *dst, size_t dst_off, const void *src, size_t len,
		std::string *err)
	{
		uint8_t *host = slot_acquire(err);

		if (!host)
			return false;

		memcpy(host, src, len);

		return slot_submit(dst, dst_off, len, err);
	}

	/* One chunk: blocking sequential pread into the slot (in-flight
	 * copies drain meanwhile), then async H2D. */
	bool push_read(void *dst, size_t dst_off, int fd, off_t file_off,
		size_t len, std::string *err)
	{
		uint8_t *host = slot_acquire(err);

		if (!host)
			return false;

		size_t done = 0;

		while (done < len) {
			ssize_t got = pread(fd, host + done, len - done,
				(off_t)(file_off + done));

			if (got <= 0) {
				*err = "short read from safetensors file";
				return false;
			}

			done += (size_t)got;
		}

		return slot_submit(dst, dst_off, len, err);
	}

	bool finish(std::string *err)
	{
		cudaError_t e = cudaStreamSynchronize(stream);

		if (e != cudaSuccess) {
			*err = cudaGetErrorString(e);
			return false;
		}

		return true;
	}
};

/*
 * h2d(tensors: array, device?): array
 *
 * Batched host-to-device upload. Copies an array of contiguous CPU
 * tensors to CUDA and returns the device copies (same shapes, same
 * dtypes), in order. device (optional) is the target: an index
 * (default 0) or a "cuda:N" string.
 *
 * Unlike per-tensor .cuda(), the transfers run asynchronously through
 * a pinned staging ring: host-side page-in of the next chunk (the
 * disk read, when the source is an mmap'ed weight file) overlaps the
 * in-flight device copy of the previous one. The stream is
 * synchronized before returning, so the results are immediately
 * usable and the staging buffer is provably idle at teardown.
 */
static uc_value_t *
torch_h2d(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *arr = uc_fn_arg(0);
	size_t n, i;
	at::Device dev(at::kCUDA, 0);

	if (nargs < 1 || nargs > 2 || ucv_type(arr) != UC_ARRAY ||
	    !parse_target_device(vm, uc_fn_arg(1), &dev)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (array of CPU tensors, device?)");
		return NULL;
	}

	n = ucv_array_length(arr);

	std::vector<at::Tensor *> srcs;

	srcs.reserve(n);

	for (i = 0; i < n; i++) {
		at::Tensor *t = tensor_arg(vm, ucv_array_get(arr, i));

		if (!t)
			return NULL;

		if (t->device().type() != at::kCPU || !t->is_contiguous()) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"tensor %zu is not a contiguous CPU tensor",
				(unsigned long)i);
			return NULL;
		}

		srcs.push_back(t);
	}

	uc_value_t *out = ucv_array_new_length(vm, (size_t)n);

	if (!out)
		return NULL;

	try {
		c10::cuda::CUDAGuard dev_guard(dev);
		h2d_ring ring;
		std::string err;

		if (!ring.init(dev, &err))
			throw std::runtime_error(err);

		for (i = 0; i < n; i++) {
			const at::Tensor &src = *srcs[i];
			size_t bytes = (size_t)src.nbytes();

			at::Tensor dst = at::empty(src.sizes(),
				src.options().device(dev));

			for (size_t off = 0; off < bytes; off += ring.chunk_bytes) {
				size_t len = std::min(ring.chunk_bytes,
					bytes - off);

				if (!ring.push_mem(dst.data_ptr(), off,
						(uint8_t *)src.data_ptr() + off,
						len, &err))
					throw std::runtime_error(err);
			}

			ucv_array_set(out, i,
				tensor_wrap(vm, std::move(dst), 0));
		}

		if (!ring.finish(&err))
			throw std::runtime_error(err);
	}
	catch (const std::exception &e) {
		ucv_put(out);
		RAISE(e.what());
	}

	return out;
}

/*
 * h2d_file(items: array, device?): array
 *
 * Batched weight loading straight from safetensors files. items is an
 * array of [stfile, name] pairs; returns one CUDA tensor per pair
 * (same shape/dtype as stored), in order. device (optional) is the
 * target: an index (default 0) or a "cuda:N" string.
 *
 * This is the cold-load fast path: tensor bytes are read with
 * sequential pread() into a pinned staging ring — avoiding the mmap
 * page-fault path, which tops out around 0.4 GB/s on 4 KB faults —
 * while previously enqueued device copies drain concurrently. The
 * stream is synchronized before returning.
 */
static uc_value_t *
torch_h2d_file(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *items = uc_fn_arg(0);
	size_t n, i;
	at::Device dev(at::kCUDA, 0);

	if (nargs < 1 || nargs > 2 || ucv_type(items) != UC_ARRAY ||
	    !parse_target_device(vm, uc_fn_arg(1), &dev)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (array of [stfile, name] pairs, device?)");
		return NULL;
	}

	n = ucv_array_length(items);

	struct src_ent {
		st_file_t *f;
		size_t file_off;
		size_t nbytes;
		at::ScalarType dt;
		std::vector<int64_t> dims;
	};

	std::vector<src_ent> ents;

	ents.reserve(n);

	for (i = 0; i < n; i++) {
		uc_value_t *pair = ucv_array_get(items, i);

		if (ucv_type(pair) != UC_ARRAY ||
		    ucv_array_length(pair) != 2) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"item %zu: expected [stfile, name]",
				(unsigned long)i);
			return NULL;
		}

		uc_value_t *fv = ucv_array_get(pair, 0);
		uc_value_t *nv = ucv_array_get(pair, 1);
		st_file_t *f = NULL;

		if (ucv_type(fv) == UC_RESOURCE && res_is_type(fv, STFILE_TYPE))
			f = static_cast<st_file_t *>(
				ucv_resource_data(fv, STFILE_TYPE));

		if (!f) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"item %zu: not a torch.stfile value",
				(unsigned long)i);
			return NULL;
		}

		if (ucv_type(nv) != UC_STRING) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"item %zu: expected a tensor name",
				(unsigned long)i);
			return NULL;
		}

		const char *dtype;
		int64_t *shape;
		size_t ndim, offset, nbytes;
		int rc;

		rc = st_lookup(f, ucv_string_get(nv), &dtype, &shape, &ndim,
			&offset, &nbytes);

		if (rc == ST_NOT_FOUND) {
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
				"item %zu: no such tensor '%s'",
				(unsigned long)i, ucv_string_get(nv));
			return NULL;
		}

		if (rc < 0) {
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
				"item %zu: malformed safetensors entry for '%s'",
				(unsigned long)i, ucv_string_get(nv));
			return NULL;
		}

		src_ent e;

		e.f = f;

		if (!st_dtype_to_at(dtype, &e.dt)) {
			free(shape);
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
				"item %zu: unsupported dtype %s",
				(unsigned long)i, dtype);
			return NULL;
		}

		e.dims.assign(shape, shape + ndim);
		free(shape);
		e.file_off = f->data_off + offset;
		e.nbytes = nbytes;

		ents.push_back(std::move(e));
	}

	uc_value_t *out = ucv_array_new_length(vm, (size_t)n);

	if (!out)
		return NULL;

	try {
		c10::cuda::CUDAGuard dev_guard(dev);
		h2d_ring ring;
		std::string err;

		if (!ring.init(dev, &err))
			throw std::runtime_error(err);

		for (i = 0; i < n; i++) {
			src_ent &ent = ents[i];

			at::Tensor dst = at::empty(ent.dims,
				at::TensorOptions().dtype(ent.dt)
					.device(dev));

			for (size_t off = 0; off < ent.nbytes;
					off += ring.chunk_bytes) {
				size_t len = std::min(ring.chunk_bytes,
					ent.nbytes - off);

				if (!ring.push_read(dst.data_ptr(), off,
						ent.f->fd,
						(off_t)(off + ent.file_off),
						len, &err))
					throw std::runtime_error(err);
			}

			ucv_array_set(out, i,
				tensor_wrap(vm, std::move(dst), 0));
		}

		if (!ring.finish(&err))
			throw std::runtime_error(err);
	}
	catch (const std::exception &e) {
		ucv_put(out);
		RAISE(e.what());
	}

	return out;
}

/*
 * exl3_reconstruct(unpacked, packed, K, mul1?, mcg?): torch.tensor
 *
 * Dequantizes an EXL3 trellis tensor into `unpacked` (fp16, in-place).
 * K is the bitrate (trellis.shape[-1] / 16). mul1 (optional bool,
 * default false) selects the mul1 codebook variant — required by
 * checkpoints quantized with "codebook": "mul1" (the .mul1 marker
 * tensor is present in the weight map). mcg (optional bool, default
 * false) selects the mcg codebook (cb=1; the .mcg marker tensor).
 */
static uc_value_t *
torch_exl3_reconstruct(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *unpacked_v = uc_fn_arg(0), *packed_v = uc_fn_arg(1);
	uc_value_t *k_v = uc_fn_arg(2);
	bool mul1 = false, mcg = false;

	if (nargs < 3 || nargs > 5 || ucv_type(unpacked_v) != UC_RESOURCE ||
	    ucv_type(packed_v) != UC_RESOURCE ||
	    !(ucv_type(k_v) == UC_INTEGER || ucv_type(k_v) == UC_DOUBLE)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (unpacked, packed, K, mul1?, mcg?)");
		return NULL;
	}

	if (nargs >= 4 && !arg_bool(vm, uc_fn_arg(3), &mul1))
		return NULL;

	if (nargs == 5 && !arg_bool(vm, uc_fn_arg(4), &mcg))
		return NULL;

	at::Tensor *unpacked = tensor_arg(vm, unpacked_v);
	at::Tensor *packed = tensor_arg(vm, packed_v);

	if (!unpacked || !packed)
		return NULL;

	float K;

	if (ucv_type(k_v) == UC_INTEGER)
		K = (float)ucv_int64_get(k_v);
	else
		K = (float)ucv_double_get(k_v);

	try {
		TensorDeviceGuard dg(*packed);

		::reconstruct(*unpacked, *packed,
			(float)K, mcg, mul1);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(unpacked_v);
}

void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const float scale
);

void reconstruct_had_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    at::Tensor suh,
    at::Tensor svh,
    float K_,
    bool mcg,
    bool mul1,
    int64_t n_offset
);

/* y = a @ b for the reconstruct path: cublasGemmEx with
 * CUBLAS_COMPUTE_16F — fp16 accumulation, the 2x tensor-core rate on
 * GeForce Ampere (fp32-accumulate MMA runs at half rate there). Measured
 * on Qwen3.8-27B: +8.5% prefill, perplexity on 4k tokens identical to 4
 * decimals, GEMM rel-rms vs exl3_gemm 1.8e-3 (fp32 accumulation: 0.9e-3).
 * AMP_GEMM_F16ACC=0 selects ATen/cuBLAS fp32 accumulation (e.g. if a
 * model's activations ever overflow fp16 partial sums). NB: exllamav3's
 * own hgemm_f16acc kernel (32-term fp16 partials flushed to fp32) was
 * SLOWER than cuBLAS on the user's power-capped 3090s (841 vs 962 tok/s)
 * and is not built. */
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

static void
recon_mm(at::Tensor &y, const at::Tensor &a, const at::Tensor &b)
{
	static int f16acc = -1;

	if (f16acc < 0) {
		const char *e = getenv("AMP_GEMM_F16ACC");

		f16acc = (e && *e == '0') ? 0 : 1;
	}

	if (!f16acc) {
		at::mm_out(y, a, b);
		return;
	}

	/* row-major y[m,n] = a[m,k] @ b[k,n]  ==  column-major
	 * y^T[n,m] = b^T[n,k] @ a^T[k,m] */
	int m = (int)a.size(0), k = (int)a.size(1), n = (int)b.size(1);
	cublasHandle_t h = at::cuda::getCurrentCUDABlasHandle();
	__half one = __float2half(1.0f), zero = __float2half(0.0f);

	cublasSetStream(h, at::cuda::getCurrentCUDAStream().stream());

	cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
		&one, b.data_ptr(), CUDA_R_16F, n, a.data_ptr(), CUDA_R_16F, k,
		&zero, y.data_ptr(), CUDA_R_16F, n, CUBLAS_COMPUTE_16F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);

	if (st != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error("recon_mm: cublasGemmEx failed");
}

/*
 * exl3_gemm_recon(x, trellis, suh, svh, K, mul1?, mcg?): torch.tensor
 *
 * EXL3 matmul for MANY rows (prefill): dequantize the weight to fp16
 * once, then one cuBLAS fp16 gemm — instead of exl3_gemm's on-the-fly
 * trellis decode, which redoes the decode for every row tile and loses
 * to reconstruct+hgemm beyond ~144 rows (exllamav3's
 * AUTO_RECONSTRUCT_THRESHOLD; same two paths as its
 * LinearEXL3.reconstruct_hgemm):
 *
 *   rows <  1024: xh = had(x * suh); W = reconstruct (rotated basis);
 *                 y = xh @ W; y = had(y) * svh
 *   rows >= 1024: W = reconstruct_had_slice (original basis, both
 *                 Hadamards + scales folded in); y = x @ W
 *
 * x: (rows, k) fp16 contiguous; returns y (rows, n) fp16. The fp16 W
 * scratch (k * n * 2 bytes) is allocated per call from the caching
 * allocator. n must be <= 32768 (the reconstruct slice limit).
 */
static uc_value_t *
torch_exl3_gemm_recon(uc_vm_t *vm, size_t nargs)
{
	bool mul1 = false, mcg = false;
	double Kd;

	if (nargs < 5 || nargs > 7 || !arg_number(vm, uc_fn_arg(4), &Kd)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, trellis, suh, svh, K, mul1?, mcg?)");
		return NULL;
	}

	if (nargs >= 6 && !arg_bool(vm, uc_fn_arg(5), &mul1))
		return NULL;

	if (nargs == 7 && !arg_bool(vm, uc_fn_arg(6), &mcg))
		return NULL;

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "exl3_gemm_recon: x", false);

	if (!x)
		return NULL;

	at::Tensor *trellis = tensor_arg(vm, uc_fn_arg(1));
	at::Tensor *suh = trellis ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *svh = suh ? tensor_arg(vm, uc_fn_arg(3)) : NULL;

	if (!svh)
		return NULL;

	if (trellis->dim() != 3) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"exl3_gemm_recon: trellis must be 3-D");
		return NULL;
	}

	int64_t k = trellis->size(0) * 16;
	int64_t n = trellis->size(1) * 16;

	if (x->dim() != 2 || x->size(1) != k || suh->numel() != k ||
	    svh->numel() != n || n > 32768) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"exl3_gemm_recon: incompatible shapes (x %lld x %lld, weight %lld x %lld)",
			(long long)x->size(0), (long long)(x->dim() > 1 ? x->size(1) : 0),
			(long long)k, (long long)n);
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		int64_t rows = x->size(0);
		at::Tensor y = at::empty({ rows, n }, x->options());
		at::Tensor w = at::empty({ k, n }, x->options());

		static int64_t fused_rows = -1;

		if (fused_rows < 0) {
			const char *e = getenv("AMP_RECON_FUSED_ROWS");

			fused_rows = e ? atoll(e) : 2048;
		}

		if (rows >= fused_rows) {
			::reconstruct_had_slice(w, *trellis, *suh, *svh,
				(float)Kd, mcg, mul1, 0);
			recon_mm(y, *x, w);
		}
		else {
			at::Tensor xh = at::empty_like(*x);

			::had_r_128(*x, xh, *suh, c10::nullopt, 1.0f);
			::reconstruct(w, *trellis, (float)Kd, mcg, mul1);
			recon_mm(y, xh, w);
			::had_r_128(y, y, c10::nullopt, *svh, 1.0f);
		}

		return tensor_wrap(vm, std::move(y), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * exl3_gemm(A, B, C, suh, svh, mul1?, mcg?)
 *
 * EXL3 quantized matmul C = A @ dequant(B) (dequantized weight as (k, n)).
 * A: (m, k) fp16, B: trellis (k/16, n/16, 16K) uint16, C: (m, n) fp16,
 * suh: (k) fp16, svh: (n) fp16. Both scale vectors are required: real EXL3
 * tensors always carry them (LinearEXL3 asserts their presence), and the
 * upstream kernel crashes with an illegal memory access when suh/svh are
 * null (verified against the stock wheel — not a vendoring artifact).
 *
 * Semantics: C = A @ dequant(B), the dequantized weight consumed as a
 * (k, n) matrix — no transpose (matches upstream test_sm75_gemm.py).
 *
 * mul1 (optional bool, default false) selects the mul1 codebook
 * variant (checkpoints quantized with "codebook": "mul1"); mcg
 * (optional bool, default false) the mcg variant ("codebook": "mcg",
 * the .mcg marker tensor). Mutually exclusive upstream.
 */
static uc_value_t *
torch_exl3_gemm(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *A_v = uc_fn_arg(0), *B_v = uc_fn_arg(1), *C_v = uc_fn_arg(2);
	uc_value_t *suh_v = uc_fn_arg(3), *svh_v = uc_fn_arg(4);
	bool mul1 = false, mcg = false;

	if (nargs < 5 || nargs > 7 || ucv_type(A_v) != UC_RESOURCE ||
	    ucv_type(B_v) != UC_RESOURCE || ucv_type(C_v) != UC_RESOURCE) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (A, B, C, suh, svh, mul1?, mcg?)");
		return NULL;
	}

	if (nargs >= 6 && !arg_bool(vm, uc_fn_arg(5), &mul1))
		return NULL;

	if (nargs == 7 && !arg_bool(vm, uc_fn_arg(6), &mcg))
		return NULL;

	at::Tensor *A = tensor_arg(vm, A_v);
	at::Tensor *B = tensor_arg(vm, B_v);
	at::Tensor *C = tensor_arg(vm, C_v);
	at::Tensor *suh = tensor_arg(vm, suh_v);
	at::Tensor *svh = tensor_arg(vm, svh_v);

	if (!A || !B || !C || !suh || !svh)
		return NULL;

	try {
		TensorDeviceGuard dg(*A);

		c10::optional<at::Tensor> A_had = at::empty_like(*A);

		exl3_gemm(*A, *B, *C,
			*suh, A_had, *svh,
			0, mcg, mul1, 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(C_v);
}

/* --- module glue -------------------------------------------------------- */

static uc_value_t *
stfile_free_method(uc_vm_t *vm, size_t nargs)
{
	void **slot = ucv_resource_dataptr(_uc_fn_this_res(vm), STFILE_TYPE);

	if (slot && *slot) {
		stfile_free(static_cast<st_file_t *>(*slot));
		*slot = NULL;
	}

	return NULL;
}

static const uc_function_list_t stfile_methods[] = {
	{ "list",	torch_st_list },
	{ "info",	torch_st_info },
	{ "tensor",	torch_st_tensor },
	{ "free",	stfile_free_method },
};

static const uc_function_list_t tensor_methods[] = {
	{ "item",	tensor_item },
	{ "tolist",	tensor_tolist },
	{ "shape",	tensor_shape },
	{ "dtype",	tensor_dtype },
	{ "numel",	tensor_numel },
	{ "is_contiguous",	tensor_is_contiguous },
	{ "fill_",	tensor_fill },
	{ "cuda",	tensor_cuda },
	{ "cpu",	tensor_cpu },
	{ "to",		tensor_to },
	{ "view",	tensor_view },
	{ "reshape",	tensor_reshape },
	{ "transpose",	tensor_transpose },
	{ "t",		tensor_t },
	{ "contiguous",	tensor_contiguous },
	{ "narrow",	tensor_narrow },
	{ "unsqueeze",	tensor_unsqueeze },
	{ "squeeze",	tensor_squeeze },
	{ "expand",	tensor_expand },
	{ "index",	tensor_index },
	{ "gather",	tensor_gather },
	{ "copy_",	tensor_copy_ },
	{ "index_copy_",	tensor_index_copy_ },
	{ "add",	tensor_add },
	{ "add_",	tensor_add_ },
	{ "mul",	tensor_mul },
	{ "mul_",	tensor_mul_ },
	{ "div",	tensor_div },
	{ "sub",	tensor_sub },
	{ "pow",	tensor_pow },
	{ "sqrt",	tensor_sqrt },
	{ "abs",	tensor_abs },
	{ "rsqrt",	tensor_rsqrt },
	{ "neg",	tensor_neg },
	{ "exp",	tensor_exp },
	{ "log",	tensor_log },
	{ "cos",	tensor_cos },
	{ "sin",	tensor_sin },
	{ "silu",	tensor_silu },
	{ "sigmoid",	tensor_sigmoid },
	{ "softplus",	tensor_softplus },
	{ "softmax",	tensor_softmax },
	{ "mean",	tensor_mean },
	{ "sum",	tensor_sum },
	{ "max",	tensor_max },
	{ "argmax",	tensor_argmax },
	{ "topk",	tensor_topk },
	{ "pin",	tensor_pin },
	{ "data_ptr",	tensor_data_ptr },
	{ "free",	tensor_free_method },
};

static const uc_function_list_t torch_fns[] = {
	{ "version",	amp_torch_version },
	{ "zeros",	tensor_zeros },
	{ "empty",	tensor_empty },
	{ "ones",	tensor_ones },
	{ "rand",	tensor_rand },
	{ "full",	tensor_full },
	{ "arange",	tensor_arange },
	{ "from_array",	tensor_from_array },
	{ "st",		torch_st_open },
	{ "st_save",	torch_st_save },
	{ "synchronize",	torch_synchronize },
	{ "rms_norm",	torch_rms_norm },
	{ "rope",	torch_rope },
	{ "silu_mul",	torch_silu_mul },
	{ "gdn_scan",	torch_gdn_scan },
	{ "flash_attn_kvcache",	torch_flash_attn_kvcache },
	{ "device_count",	torch_device_count },
	{ "attn_decode",	torch_attn_decode },
	{ "qk_norm_rope",	torch_qk_norm_rope },
	{ "attn_gate",	torch_attn_gate },
	{ "gdn_chunk",	torch_gdn_chunk },
	{ "gdn_prep",	torch_gdn_prep },
	{ "gdn_gates",	torch_gdn_gates },
	{ "gdn_out_norm",	torch_gdn_out_norm },
	{ "flash_attn",	torch_flash_attn },
	{ "rms_norm_res",	torch_rms_norm_res },
	{ "decode_attn",	torch_decode_attn },
	{ "argmax_advance",	torch_argmax_advance },
	{ "mgemm",	torch_mgemm },
	{ "ptr_array",	torch_ptr_array },
	{ "h2d",		torch_h2d },
	{ "h2d_file",	torch_h2d_file },
	{ "graph",		torch_graph_new },
	{ "cat",	torch_cat },
	{ "stack",	torch_stack },
	{ "matmul",	torch_matmul },
	{ "sdpa",	torch_sdpa },
	{ "where",	torch_where },
	{ "exl3_reconstruct",	torch_exl3_reconstruct },
	{ "exl3_gemm_recon",	torch_exl3_gemm_recon },
	{ "exl3_gemm",	torch_exl3_gemm },
};

void
uc_module_init(uc_vm_t *vm, uc_value_t *scope)
{
	/* int8-activation GEMV for K = 6 tensors too: exllamav3 caps it at
	 * K <= 5 on Ampere, but on the 3090s it beats the fp16 tensor-core
	 * kernel at m == 1 for the 27B's K = 6 down/out projections (decode
	 * 37.5 -> 35.0 ms/token). Precision: ~8e-3 rel-rms vs the fp16
	 * dequantized product, as for every K <= 5 tensor by default (the
	 * fp16 kernel: ~2.7e-3). EXL3_INT8_GEMV_MAX_K=5 restores upstream's
	 * cap; an explicit environment value always wins (docs/LESSONS.md L65). */
	setenv("EXL3_INT8_GEMV_MAX_K", "6", 0);

	/* The caching allocator's expandable segments (exllamav3 / TabbyAPI run with
	 * them): fragmentation-free growth, so the memory the startup warm-up
	 * reserved is what later requests reuse. Read when the allocator is
	 * first used, i.e. after this runs; an explicit environment value wins
	 * (PYTORCH_CUDA_ALLOC_CONF is also the older spelling's name for
	 * PYTORCH_ALLOC_CONF — set either to override). */
	if (!getenv("PYTORCH_ALLOC_CONF"))
		setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 0);

	uc_type_declare(vm, TENSOR_TYPE, tensor_methods, tensor_free);
	uc_type_declare(vm, STFILE_TYPE, stfile_methods, stfile_free);
	uc_type_declare(vm, GRAPH_TYPE, graph_methods, graph_free);
	uc_function_list_register(scope, torch_fns);
	amp_tokenizer_init(vm, scope);
}