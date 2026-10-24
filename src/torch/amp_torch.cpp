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
#include <climits>
#include <cstring>
#include <cstdio>
#include <memory>
#include <new>
#include <fcntl.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <cuda_profiler_api.h>
#include <sys/socket.h>
#include <poll.h>
#include <errno.h>
#include <time.h>
#include <vector>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <list>
#include <unordered_map>
#include <memory>
#include <functional>
#include <optional>

// ucode headers are plain C and carry no extern "C" guards; without this
// wrapper the module entry points compile to C++-mangled symbols and the
// VM's dlsym("uc_module_entry") fails. (Lesson L4.)
extern "C" {
#include <ucode/module.h>
#include <ucode/vm.h>
#include "image.h"
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

static at::Tensor *
tensor_arg_opt(uc_vm_t *vm, uc_value_t *v)
{
	if (v && ucv_type(v) == UC_NULL)
		return NULL;

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
	case at::kFloat8_e4m3fn:  return "float8_e4m3fn";
	case at::kFloat8_e5m2:    return "float8_e5m2";
	case at::kFloat8_e8m0fnu: return "float8_e8m0fnu";
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
	if (!strcmp(name, "float8_e4m3fn"))  { *out = at::kFloat8_e4m3fn;  return true; }
	if (!strcmp(name, "float8_e5m2"))    { *out = at::kFloat8_e5m2;    return true; }
	if (!strcmp(name, "float8_e8m0fnu")) { *out = at::kFloat8_e8m0fnu; return true; }

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

/*
 * view_dtype(dtype): torch.tensor
 *
 * The same bytes read as another dtype (a bit reinterpretation, no copy): a
 * contiguous tensor whose last dimension times the element size is a multiple
 * of the new element size. The receiver of an RPC frame gets raw bytes and
 * views them as the float32 streams or the int64 ids they were (docs/RPC.md).
 */
static uc_value_t *
tensor_view_dtype(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	at::ScalarType dt;

	if (!t)
		return NULL;

	if (nargs != 1 || !arg_dtype(vm, uc_fn_arg(0), &dt)) {
		if (nargs != 1)
			uc_vm_raise_exception(vm, EXCEPTION_TYPE, "view_dtype: expected (dtype)");

		return NULL;
	}

	try {
		return tensor_wrap(vm, t->view(dt), 0);
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

static uc_value_t *
tensor_index_add_(uc_vm_t *vm, size_t nargs)
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

		t->index_add_(dim, *idx, *src);
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

/*
 * le(other): torch.tensor
 *
 * Elementwise <= (tensor broadcast or number). Returns a bool tensor;
 * the comparison primitive for data-dependent masks (arange <= pos).
 */
static uc_value_t *
tensor_le(uc_vm_t *vm, size_t nargs)
{
	return tensor_binary_out(vm, nargs, "le",
		[](const at::Tensor &a, const at::Tensor &b) { return at::le(a, b); },
		[](const at::Tensor &a, double s) { return at::le(a, s); });
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
 * clamp(min, max): torch.tensor — either bound may be null (open).
 */
static uc_value_t *
tensor_clamp(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t = tensor_this(vm);
	double lo = 0, hi = 0;
	bool has_lo = false, has_hi = false;

	if (!t)
		return NULL;

	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (min, max)");
		return NULL;
	}

	if (ucv_type(uc_fn_arg(0)) != UC_NULL) {
		if (!arg_number(vm, uc_fn_arg(0), &lo))
			return NULL;

		has_lo = true;
	}

	if (ucv_type(uc_fn_arg(1)) != UC_NULL) {
		if (!arg_number(vm, uc_fn_arg(1), &hi))
			return NULL;

		has_hi = true;
	}

	if (!has_lo && !has_hi)
		RAISE("clamp: at least one bound is required");

	try {
		at::Tensor r = at::clamp(*t,
			has_lo ? c10::optional<at::Scalar>(lo) : c10::nullopt,
			has_hi ? c10::optional<at::Scalar>(hi) : c10::nullopt);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e2) {
		RAISE(e2.what());
	}
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
 * cumsum(dim?: number): torch.tensor
 *
 * Inclusive prefix sum over `dim` (default -1), same dtype.
 */
static uc_value_t *
tensor_cumsum(uc_vm_t *vm, size_t nargs)
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
		at::Tensor r = at::cumsum(*t, dim);

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

/*
 * pinned_empty(dims, dtype?): torch.tensor
 *
 * Uninitialized page-locked CPU tensor (cudaHostAllocPortable, like pin()):
 * the target of copy_async and the source of async H2D copies. Backs the
 * engine's host-RAM prompt-cache store.
 */
static uc_value_t *
torch_pinned_empty(uc_vm_t *vm, size_t nargs)
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
		at::Tensor t = at::empty(dims, at::TensorOptions().dtype(dt).device(at::kCPU));
		size_t n = t.nbytes();
		void *p = NULL;

		if (n == 0)
			return tensor_wrap(vm, std::move(t), 0);

		if (cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess)
			RAISE("pinned_empty: out of page-locked memory");

		return tensor_wrap(vm, at::from_blob(p, t.sizes(),
			[](void *q) { cudaFreeHost(q); }, t.options()), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
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
 * sdpa(q, k, v, is_causal?: boolean, attn_mask?: tensor, enable_gqa?: boolean, scale?: number): torch.tensor
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
	double scale = 0;
	bool has_scale = false;

	if (nargs < 3 || nargs > 7) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, is_causal?, attn_mask?, enable_gqa?, scale?)");
		return NULL;
	}

	if (nargs >= 7 && ucv_type(uc_fn_arg(6)) != UC_NULL) {
		if (!arg_number(vm, uc_fn_arg(6), &scale))
			return NULL;

		has_scale = true;
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
			0.0, is_causal, has_scale ? c10::optional<double>(scale) : c10::nullopt, enable_gqa);

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
 * pool(): [a, b]
 *
 * The id of the graph's private allocator pool, to pass as the third
 * argument of another graph's capture_begin() (shared pool). [0, 0] for a
 * graph that was never captured.
 */
static uc_value_t *
graph_pool(uc_vm_t *vm, size_t nargs)
{
	graph_state *g = graph_this(vm);

	if (!g)
		return NULL;

	at::cuda::MempoolId_t id = g->graph->pool();
	uc_value_t *a = ucv_array_new_length(vm, 2);

	ucv_array_push(a, ucv_int64_new((int64_t)id.first));
	ucv_array_push(a, ucv_int64_new((int64_t)id.second));

	return a;
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
	{ "pool",		graph_pool },
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
	if (!strcmp(dt, "F8_E4M3")) { *out = at::kFloat8_e4m3fn;  return true; }
	if (!strcmp(dt, "F8_E5M2")) { *out = at::kFloat8_e5m2;    return true; }
	if (!strcmp(dt, "F8_E8M0")) { *out = at::kFloat8_e8m0fnu; return true; }

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
	case at::kFloat8_e4m3fn:	return "F8_E4M3";
	case at::kFloat8_e5m2:		return "F8_E5M2";
	case at::kFloat8_e8m0fnu:	return "F8_E8M0";
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

/* --- EXL3 kernels (vendored, see vendor/exl3) --------------------------- */

// quant/exl3_moe_coop.cu: fused decode-shaped MoE (two launches per layer)
void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w,
    const c10::optional<at::Tensor>& bits_g,
    const c10::optional<at::Tensor>& bits_u,
    const c10::optional<at::Tensor>& bits_d
);

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

// mHC hyper-connection kernels (vendor/exl3/exllamav3_ext/hc_mix.cu)
int hc_mix_num_chunks(int R, int row_len);

void hc_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed
);

void hc_mix2
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    const at::Tensor& pre_in,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor pre_out,
    at::Tensor collapsed
);

void hc_apply
(
    at::Tensor x,
    const at::Tensor& y,
    const at::Tensor& post,
    const c10::optional<at::Tensor>& comb
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

/* --- CPU experts (vendor/exl3/exllamav3_ext/cpu/moe_mul1.cpp) --------------------------------------------------- */

#include "exl3/exllamav3_ext/cpu/moe_mul1.h"

/* A list of tensors from an array argument. */
static bool
tensor_list_arg(uc_vm_t *vm, uc_value_t *v, std::vector<at::Tensor> &out, const char *what)
{
	if (ucv_type(v) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "%s: expected an array of tensors", what);
		return false;
	}

	for (size_t i = 0; i < ucv_array_length(v); i++) {
		at::Tensor *t = tensor_arg(vm, ucv_array_get(v, i));

		if (!t)
			return false;

		out.push_back(*t);
	}

	return true;
}

/*
 * moe_cpu_make_layer(gate_trellis, gate_suh, gate_svh, up_trellis, up_suh, up_svh, down_trellis, down_suh, down_svh,
 *                    act_limit): int
 *
 * Registers the experts of one layer for the CPU kernels: nine arrays (one tensor per expert) of contiguous CPU
 * tensors, the trellis as stored in the checkpoint (int16 [k/16, n/16, 16K]), suh / svh fp16. mul1 codebook only,
 * SiLU gated experts with the clamp act_limit (DeepSeek-V4: swiglu_limit). The tensors are referenced, not copied.
 * Returns the handle for moe_cpu_forward / moe_cpu_enqueue.
 */
static uc_value_t *
torch_moe_cpu_make_layer(uc_vm_t *vm, size_t nargs)
{
	std::vector<at::Tensor> a[9];
	double lim;

	if (nargs != 10) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (nine arrays of tensors, act_limit)");
		return NULL;
	}

	for (int i = 0; i < 9; i++)
		if (!tensor_list_arg(vm, uc_fn_arg(i), a[i], "moe_cpu_make_layer"))
			return NULL;

	if (!arg_number(vm, uc_fn_arg(9), &lim))
		return NULL;

	if (a[3].empty()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "moe_cpu_make_layer: no experts");
		return NULL;
	}

	try {
		std::vector<at::Tensor> none;
		int64_t h = exl3_moe_cpu_make_layer(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
			none, none, none, 0, lim, 0);

		return ucv_int64_new(h);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

static uc_value_t *
torch_moe_cpu_free_layer(uc_vm_t *vm, size_t nargs)
{
	int64_t h;

	if (nargs != 1 || !arg_int(vm, uc_fn_arg(0), &h)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (handle)");
		return NULL;
	}

	exl3_moe_cpu_free_layer(h);

	return NULL;
}

/* The tensors of one call, checked: x [m, H] fp16, sel [m, K] int32, w [m, K] fp16, out [m, H] fp32. */
struct moe_cpu_args {
	at::Tensor *x, *sel, *w, *out;
	int rows, topk;
};

static bool
moe_cpu_args_check(uc_vm_t *vm, uc_value_t **v, bool want_cuda, moe_cpu_args *a, const char *what)
{
	a->x = tensor_arg(vm, v[0]);
	a->sel = tensor_arg(vm, v[1]);
	a->w = tensor_arg(vm, v[2]);
	a->out = tensor_arg(vm, v[3]);

	if (!a->x || !a->sel || !a->w || !a->out)
		return false;

	if (a->x->is_cuda() != want_cuda || a->sel->is_cuda() != want_cuda || a->w->is_cuda() != want_cuda ||
	    a->out->is_cuda() != want_cuda || !a->x->is_contiguous() || !a->sel->is_contiguous() ||
	    !a->w->is_contiguous() || !a->out->is_contiguous() || a->x->dim() != 2 || a->sel->dim() != 2 ||
	    a->x->scalar_type() != at::kHalf || a->w->scalar_type() != at::kHalf ||
	    a->sel->scalar_type() != at::kInt || a->out->scalar_type() != at::kFloat ||
	    a->sel->size(0) != a->x->size(0) || a->out->size(0) != a->x->size(0) || a->w->numel() != a->sel->numel() ||
	    a->out->size(1) != a->x->size(1) || (want_cuda && a->x->get_device() != a->out->get_device())) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: x [m, H] fp16, sel [m, K] int32, w [m, K] fp16, out [m, H] fp32, contiguous, all %s", what,
			want_cuda ? "on one device" : "on the CPU");
		return false;
	}

	a->rows = (int)a->x->size(0);
	a->topk = (int)a->sel->size(1);

	return true;
}

/* The optional thread count argument. */
static bool
moe_cpu_threads_arg(uc_vm_t *vm, size_t nargs, size_t at, int64_t *threads)
{
	*threads = 8;

	if (nargs > at && (!arg_int(vm, uc_fn_arg(at), threads) || *threads < 1 || *threads > 64)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "threads: expected an integer 1..64");
		return false;
	}

	return true;
}

/*
 * moe_cpu_forward(handle, x, sel, w, out, threads?): null
 *
 * Runs the layer's experts for the picks `sel` (ids into the layer's expert list; ids outside it are skipped) now, on
 * `threads` threads (default 8); CPU tensors. out is overwritten with the weighted sum of the experts' outputs.
 */
static uc_value_t *
torch_moe_cpu_forward(uc_vm_t *vm, size_t nargs)
{
	int64_t h, threads;
	moe_cpu_args a;
	uc_value_t *v[4];

	if (nargs < 5 || nargs > 6 || !arg_int(vm, uc_fn_arg(0), &h)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (handle, x, sel, w, out, threads?)");
		return NULL;
	}

	for (int i = 0; i < 4; i++)
		v[i] = uc_fn_arg(1 + i);

	if (!moe_cpu_args_check(vm, v, false, &a, "moe_cpu_forward") || !moe_cpu_threads_arg(vm, nargs, 5, &threads))
		return NULL;

	try {
		exl3_moe_cpu_forward_raw(h, (const at::Half *)a.x->data_ptr(), (const int32_t *)a.sel->data_ptr(),
			(const at::Half *)a.w->data_ptr(), (float *)a.out->data_ptr(), a.rows, a.topk, (int)threads);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* One queued CPU expert call: kept for the life of the process when a graph captured it (the graph replays it), freed
 * by its host function when it was queued eagerly (oneshot). */
struct moe_cpu_job {
	int64_t handle;
	const at::Half *x;
	const int32_t *sel;
	const at::Half *w;
	float *out;
	int rows, topk, threads;
	bool oneshot;
};

static std::mutex moe_cpu_jobs_mutex;
static std::vector<std::unique_ptr<moe_cpu_job>> moe_cpu_jobs;

/* AMP_EXPERT_CPU_PROF=N: every N host calls, their mean wall time, picks, overlap (calls already running when one
 * starts: a serialised pool shows up as wall time spent waiting) and the number of distinct calling threads */
static struct {
	long every, calls, picks, empty, overlap;
	double us;
	std::atomic<int> running;
	std::mutex lock;
	std::vector<pthread_t> tids;
} moe_cpu_prof = { -1 };

/* With the statistics on: would a cache of decoded experts help? The distinct experts of every decode-sized call (at
 * most 16 rows; prefill calls touch most of a layer) go through simulated LRU caches of 256 / 1024 / 4096 experts: the
 * hit rates say how often the CPU decodes an expert it decoded recently. */
struct moe_cpu_lru {
	size_t cap;
	long hits;
	std::list<uint64_t> order;
	std::unordered_map<uint64_t, std::list<uint64_t>::iterator> at;

	bool
	touch(uint64_t k)
	{
		auto it = at.find(k);

		if (it != at.end()) {
			order.splice(order.begin(), order, it->second);
			hits++;

			return true;
		}

		order.push_front(k);
		at[k] = order.begin();

		if (order.size() > cap) {
			at.erase(order.back());
			order.pop_back();
		}

		return false;
	}
};

static moe_cpu_lru moe_cpu_lrus[3] = { { 256 }, { 1024 }, { 4096 } };
static long moe_cpu_lru_n = 0;

/* the thread count that overrides the enqueued one (0: none), set by moe_cpu_threads (live tuning) */
static std::atomic<int> moe_cpu_threads_over{0};


static int
moe_cpu_threads(int dflt)
{
	int n = moe_cpu_threads_over.load(std::memory_order_relaxed);

	return n ? n : dflt;
}

static void
moe_cpu_run(moe_cpu_job *j)
{
	if (moe_cpu_prof.every < 0) {
		const char *e = getenv("AMP_EXPERT_CPU_PROF");

		moe_cpu_prof.every = e ? atol(e) : 0;
		exl3_moe_cpu_set_prof(getenv("EXL3_MOE_CPU_PROF") != NULL);
	}

	int threads = moe_cpu_threads(j->threads);

	if (!moe_cpu_prof.every) {
		try {
			exl3_moe_cpu_forward_raw(j->handle, j->x, j->sel, j->w, j->out, j->rows, j->topk, threads);
		}
		catch (const std::exception &e) {
			fprintf(stderr, "ampstein: CPU experts: %s\n", e.what());
		}

		return;
	}

	int busy = moe_cpu_prof.running.fetch_add(1);
	int picks = 0;
	struct timespec t0, t1;

	for (int i = 0; i < j->rows * j->topk; i++)
		picks += (j->sel[i] >= 0);

	clock_gettime(CLOCK_MONOTONIC, &t0);

	try {
		exl3_moe_cpu_forward_raw(j->handle, j->x, j->sel, j->w, j->out, j->rows, j->topk, threads);
	}
	catch (const std::exception &e) {
		fprintf(stderr, "ampstein: CPU experts: %s\n", e.what());
	}

	clock_gettime(CLOCK_MONOTONIC, &t1);
	moe_cpu_prof.running.fetch_sub(1);

	std::lock_guard<std::mutex> lock(moe_cpu_prof.lock);
	pthread_t self = pthread_self();
	bool known = false;

	for (pthread_t t : moe_cpu_prof.tids)
		known = known || pthread_equal(t, self);

	if (!known)
		moe_cpu_prof.tids.push_back(self);

	if (j->rows <= 16) {
		uint64_t seen[16 * 16];
		int ns = 0;

		for (int i = 0; i < j->rows * j->topk && ns < 16 * 16; i++) {
			if (j->sel[i] < 0)
				continue;

			uint64_t k = ((uint64_t)j->handle << 20) | (uint32_t)j->sel[i];
			bool dup = false;

			for (int q = 0; q < ns && !dup; q++)
				dup = (seen[q] == k);

			if (dup)
				continue;

			seen[ns++] = k;
			moe_cpu_lru_n++;

			for (auto &c : moe_cpu_lrus)
				c.touch(k);
		}
	}

	moe_cpu_prof.calls++;
	moe_cpu_prof.picks += picks;
	moe_cpu_prof.empty += (picks == 0);
	moe_cpu_prof.overlap += (busy > 0);
	moe_cpu_prof.us += (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

	if (moe_cpu_prof.calls % moe_cpu_prof.every == 0) {
		fprintf(stderr, "ampstein: CPU experts: %ld calls, %.1f us/call, %.2f picks/call, %.0f%% empty, "
			"%.0f%% overlapped, %zu threads; decoded-expert LRU hits (since start, %ld experts) 256: %.0f%%, "
			"1024: %.0f%%, 4096: %.0f%%\n", moe_cpu_prof.calls, moe_cpu_prof.us / moe_cpu_prof.calls,
			(double)moe_cpu_prof.picks / moe_cpu_prof.calls, 100.0 * moe_cpu_prof.empty / moe_cpu_prof.calls,
			100.0 * moe_cpu_prof.overlap / moe_cpu_prof.calls, moe_cpu_prof.tids.size(), moe_cpu_lru_n,
			moe_cpu_lru_n ? 100.0 * moe_cpu_lrus[0].hits / moe_cpu_lru_n : 0.0,
			moe_cpu_lru_n ? 100.0 * moe_cpu_lrus[1].hits / moe_cpu_lru_n : 0.0,
			moe_cpu_lru_n ? 100.0 * moe_cpu_lrus[2].hits / moe_cpu_lru_n : 0.0);
		moe_cpu_prof.calls = moe_cpu_prof.picks = moe_cpu_prof.empty = moe_cpu_prof.overlap = 0;
		moe_cpu_prof.us = 0;
	}
}

static void CUDART_CB
moe_cpu_host_fn(void *p)
{
	moe_cpu_job *j = static_cast<moe_cpu_job *>(p);

	moe_cpu_run(j);

	if (j->oneshot)
		delete j;
}

/*
 * moe_cpu_threads(n): null
 *
 * From now on every CPU expert call (queued or captured) runs on n threads instead of the count it was enqueued with;
 * 0 restores the enqueued counts.
 */
static uc_value_t *
torch_moe_cpu_threads(uc_vm_t *vm, size_t nargs)
{
	int64_t n;

	if (nargs != 1 || !arg_int(vm, uc_fn_arg(0), &n) || n < 0 || n > 64) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (threads 0..64)");
		return NULL;
	}

	moe_cpu_threads_over.store((int)n);

	return NULL;
}

/*
 * moe_cpu_prof(every, phases?): null
 *
 * CPU expert call statistics on stderr every `every` calls (0: off; the AMP_EXPERT_CPU_PROF default), and with
 * `phases` the exllamav3 kernels' per-phase times (stdout, every 512 calls; EXL3_MOE_CPU_PROF).
 */
static uc_value_t *
torch_moe_cpu_prof(uc_vm_t *vm, size_t nargs)
{
	int64_t n;
	uc_value_t *ph = uc_fn_arg(1);

	if (nargs < 1 || nargs > 2 || !arg_int(vm, uc_fn_arg(0), &n) || n < 0 ||
	    (nargs == 2 && ucv_type(ph) != UC_BOOLEAN)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (every >= 0, phases?)");
		return NULL;
	}

	std::lock_guard<std::mutex> lock(moe_cpu_prof.lock);

	moe_cpu_prof.every = n;
	moe_cpu_prof.calls = moe_cpu_prof.picks = moe_cpu_prof.empty = moe_cpu_prof.overlap = 0;
	moe_cpu_prof.us = 0;

	if (nargs == 2)
		exl3_moe_cpu_set_prof(ucv_boolean_get(ph));

	return NULL;
}

/*
 * moe_cpu_enqueue(handle, x, sel, w, out, x_h, sel_h, w_h, out_h, threads?): null
 *
 * The same call, queued on the current stream of the device tensors' device (x, sel, w, out: device tensors; *_h:
 * pinned CPU tensors of the same shapes and dtypes): x / sel / w are copied down, a host function runs the experts,
 * out is copied back up. Legal during graph capture (a host node and four copies); the pinned buffers and the layer
 * must stay alive as long as the stream work (or the graph) does. The host function holds the stream until the CPU is
 * done.
 */
static uc_value_t *
torch_moe_cpu_enqueue(uc_vm_t *vm, size_t nargs)
{
	int64_t h, threads;
	moe_cpu_args d, c;
	uc_value_t *v[4];

	if (nargs < 9 || nargs > 10 || !arg_int(vm, uc_fn_arg(0), &h)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (handle, x, sel, w, out, x_h, sel_h, w_h, out_h, threads?)");
		return NULL;
	}

	for (int i = 0; i < 4; i++)
		v[i] = uc_fn_arg(1 + i);

	if (!moe_cpu_args_check(vm, v, true, &d, "moe_cpu_enqueue"))
		return NULL;

	for (int i = 0; i < 4; i++)
		v[i] = uc_fn_arg(5 + i);

	if (!moe_cpu_args_check(vm, v, false, &c, "moe_cpu_enqueue") || !moe_cpu_threads_arg(vm, nargs, 9, &threads))
		return NULL;

	if (!c.x->is_pinned() || !c.sel->is_pinned() || !c.w->is_pinned() || !c.out->is_pinned() ||
	    c.x->sizes() != d.x->sizes() || c.sel->sizes() != d.sel->sizes() || c.out->sizes() != d.out->sizes()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"moe_cpu_enqueue: the host buffers must be pinned and shaped like the device tensors");
		return NULL;
	}

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)d.x->get_device());
		cudaStream_t st = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)d.x->get_device()).stream();
		cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
		auto job = std::make_unique<moe_cpu_job>();

		if (cudaStreamIsCapturing(st, &cs) != cudaSuccess)
			RAISE("moe_cpu_enqueue: cannot query the stream's capture status");

		job->handle = h;
		job->x = (const at::Half *)c.x->data_ptr();
		job->sel = (const int32_t *)c.sel->data_ptr();
		job->w = (const at::Half *)c.w->data_ptr();
		job->out = (float *)c.out->data_ptr();
		job->rows = d.rows;
		job->topk = d.topk;
		job->threads = (int)threads;
		job->oneshot = (cs == cudaStreamCaptureStatusNone);

		cudaError_t e = cudaMemcpyAsync(c.x->data_ptr(), d.x->data_ptr(), d.x->nbytes(), cudaMemcpyDeviceToHost, st);

		if (e == cudaSuccess)
			e = cudaMemcpyAsync(c.sel->data_ptr(), d.sel->data_ptr(), d.sel->nbytes(), cudaMemcpyDeviceToHost, st);

		if (e == cudaSuccess)
			e = cudaMemcpyAsync(c.w->data_ptr(), d.w->data_ptr(), d.w->nbytes(), cudaMemcpyDeviceToHost, st);

		bool oneshot = job->oneshot;

		if (e == cudaSuccess)
			e = cudaLaunchHostFunc(st, moe_cpu_host_fn, job.get());

		// (an eager job belongs to its host function once queued, which may already have run and freed it)
		if (e == cudaSuccess && oneshot)
			job.release();

		if (e == cudaSuccess)
			e = cudaMemcpyAsync(d.out->data_ptr(), c.out->data_ptr(), d.out->nbytes(), cudaMemcpyHostToDevice, st);

		if (e != cudaSuccess)
			RAISE(cudaGetErrorString(e));

		if (oneshot)
			return NULL;

		std::lock_guard<std::mutex> lock(moe_cpu_jobs_mutex);

		moe_cpu_jobs.push_back(std::move(job));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/*
 * profile(on: bool): null
 *
 * cudaProfilerStart() / cudaProfilerStop(): the capture range of `nsys profile --capture-range=cudaProfilerApi`
 * (a server is profiled over one prefill or a few decode steps instead of its whole life). A no-op without a profiler.
 */
static uc_value_t *
torch_profile(uc_vm_t *vm, size_t nargs)
{
	bool on = true;

	if (nargs != 1 || !arg_bool(vm, uc_fn_arg(0), &on)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (on: bool)");
		return NULL;
	}

	if (on)
		cudaProfilerStart();
	else
		cudaProfilerStop();

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

extern "C" int
amp_kda_scan_f32(const void *q, const void *k, const void *v, const void *beta,
                const void *g, void *state, void *out, int S, int H, int dk,
                int dv, float scale, void *states_out, int n_out,
                cudaStream_t stream);

extern "C" int
amp_kda_gates(const void *f, const void *dt_bias, const void *decay,
             const void *braw, void *g, void *beta, long S, int H, int DK,
             float lb, float beta_scale, cudaStream_t stream);

extern "C" int
amp_kda_out_norm(const void *core, const void *z, const void *w, int w_is_half,
                void *out, long rows, int dv, float eps, cudaStream_t stream);

extern "C" int
amp_copy_many(const void *const *src, void *const *dst, const long *bytes,
             int n, cudaStream_t stream);

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
amp_dsv4_qkv_fp16(const void *q_raw, const void *kv_raw, const void *kv_w,
                 const void *inv_freq, const void *pos_f, const void *pos_i,
                 void *kvcache, void *q_out, int S, int H, int D, int rd, float eps,
                 float qscale, int fp8, int kvcap, cudaStream_t stream);

extern "C" int
amp_dsv4_derot_fp16(const void *o, const void *inv_freq, const void *pos_f, void *out,
                   int R, int H, int D, int rd, cudaStream_t stream);

extern "C" int
amp_dsv4_mask_fp16(void *out, int R, int nk, int ne, long qpos0, long base, int W, int m, long ncstart,
                  cudaStream_t stream);

extern "C" int
amp_dsv4_comp_fp16(const void *kvr, const void *gtr, const void *ape,
                  void *buf_kv, void *buf_g, void *ovl_kv, void *ovl_g,
                  void *pool, const void *norm_w, const void *inv_freq,
                  const void *pos_i, int S, int m, int hd, int W, int rd,
                  float eps, cudaStream_t stream);

extern "C" int
amp_dsv4_attn_fp16(const void *q, const void *kvc, const void *pos_i,
                  const void *pos_f, const void *pool, const void *idx,
                  const void *sinks, const void *inv_freq, void *out, int S, int H,
                  int D, int rd, int W, int m, int np_max, float scale, int nc_block, int kvcap,
                  cudaStream_t stream);

extern "C" int
amp_attn_gate_fp16(const void *a, long ast, long ash, const void *g, long gst,
                  long gsh, void *out, int S, int H, int hd, cudaStream_t stream);

extern "C" long
amp_gdn_chunk_ws_floats(int S, int Hv);

extern "C" int
amp_comm_allreduce_fp16(const void *x, void *out, void *const *bufs,
                       int *const *ctls, int rank, int N, long n, long cap,
                       int mode, int blocks, cudaStream_t stream);

extern "C" int
amp_gdn_chunk_f32(const void *q, const void *k, const void *v, const void *beta,
                 const void *gexp, void *state, void *out, void *ws, int S,
                 int Hk, int Hv, int dk, int dv, float scale,
                 cudaStream_t stream);

extern "C" int
amp_gdn_prep_fp16(const void *qkv, void *conv_state, const void *conv_w,
                 void *q, void *k, void *v, int S, int D, int K, int Hk,
                 int Hv, int hd, void *const *spec, int nspec,
                 cudaStream_t stream);

extern "C" int
amp_gdn_gates(const void *a, const void *b, const void *dt_bias,
             const void *nea, void *beta, void *gexp, int S, int Hv,
             cudaStream_t stream);

extern "C" int
amp_gdn_ab_gates(const void *x, const void *wa, const void *wb,
                const void *dt_bias, const void *nea, void *beta, void *gexp,
                int S, int K, int Hv, cudaStream_t stream);

extern "C" int
amp_dflash_conv(const void *h, const void *dyn, const void *base, void *out,
               int R, int H, int G, long dstride, cudaStream_t stream);

extern "C" int
amp_rms_norm_f32in(const void *x, const void *w, void *out, long rows, int dim,
                  double eps, cudaStream_t stream);

extern "C" int
amp_norm32(const void *x, int x_half, const void *w, const void *res, const void *scale, void *out,
          long rows, int dim, double eps, cudaStream_t stream);

extern "C" int
amp_dflash_conv_res(void *h, const void *y, const void *dyn, const void *base,
                   int R, int H, int G, long dstride, float scale,
                   cudaStream_t stream);

extern "C" int
amp_dflash_select(const void *unary, const void *cand, const void *hp,
                 const void *anchor, const void *pred, const void *succ,
                 void *out, int W, int K, int R, cudaStream_t stream);

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
 * idx_score(q, pools, w, scale, pos0, P): torch.tensor
 *
 * Fused DSA indexer scoring (amp_idx_score in src/amp_kernels.cu):
 *   out[r][p] = pool p selectable by query r (p * P + P - 1 <= pos0 + r) ?
 *       sum_h w[r][h] * scale * relu(q[r][h] . pools[p]) : -65504   (fp16)
 * pos0: an integer, or a one-element int64 tensor on q's device (read by
 * the kernel: CUDA-graph decode with device-side positions). Pool tiles
 * no query can select yet are written as mask without being scored.
 * q (R, H, 128) fp16 with H a multiple of 32 (GLM 32, DeepSeek 64: the heads run in halves of
 * 32 inside one kernel, summed in fp32), pools (n_pools, 128) fp16, w (R, H) fp32, all
 * contiguous on one CUDA device; returns (R, n_pools) fp16.
 */
extern "C" int
amp_idx_score_q8(const void *q, const void *pools, const void *pq, const void *ps, const void *w, void *out,
                 int R, int n_pools, float scale, int pos0, int P, int HI,
                 const long long *posp, cudaStream_t stream);
extern "C" int
amp_idx_score(const void *q, const void *pools, const void *w, void *out,
             int R, int n_pools, float scale, int pos0, int P, int HI,
             const long long *posp, cudaStream_t stream);

static uc_value_t *
torch_idx_score(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 6)
		RAISE("idx_score: expected (q, pools, w, scale, pos0, P)");

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *pk = NULL, *pqs = NULL;
	uc_value_t *pa = q ? uc_fn_arg(1) : NULL;

	/* pools: a (n, 128) fp16 tensor, or [codes (n, 128) int8, scales (n, 4) fp16] (a Q8 pool) */
	if (pa && ucv_type(pa) == UC_ARRAY) {
		if (ucv_array_length(pa) != 2 || !(pk = tensor_arg(vm, ucv_array_get(pa, 0))) ||
		    !(pqs = tensor_arg(vm, ucv_array_get(pa, 1))))
			return NULL;
	}
	else if (pa) {
		pk = tensor_arg(vm, pa);
	}

	at::Tensor *w = pk ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!q || !pk || !w)
		return NULL;

	if (pqs && (pk->scalar_type() != at::kChar || pqs->scalar_type() != at::kHalf || pqs->dim() != 2 ||
	            pqs->size(0) != pk->size(0) || pqs->size(1) != 4 || !pqs->is_contiguous() || pqs->device() != pk->device()))
		RAISE("idx_score: a Q8 pool is [codes (n, 128) int8, scales (n, 4) fp16]");

	uc_value_t *sv = uc_fn_arg(3), *pv = uc_fn_arg(4), *kv = uc_fn_arg(5);
	at::Tensor *pt = NULL;

	if (pv && ucv_type(pv) == UC_RESOURCE) {
		pt = tensor_arg(vm, pv);

		if (!pt)
			return NULL;

		if (pt->scalar_type() != at::kLong || pt->numel() != 1 ||
		    pt->device() != q->device())
			RAISE("idx_score: a pos0 tensor must be one int64 on q's device");
	}

	if (!sv || !pv || !kv || (!pt && ucv_type(pv) != UC_INTEGER) || ucv_type(kv) != UC_INTEGER ||
	    (ucv_type(sv) != UC_DOUBLE && ucv_type(sv) != UC_INTEGER))
		RAISE("idx_score: scale must be a number, pos0 an integer or int64 tensor, P an integer");

	if (q->scalar_type() != at::kHalf || pk->scalar_type() != (pqs ? at::kChar : at::kHalf) ||
	    w->scalar_type() != at::kFloat || q->device().type() != at::kCUDA ||
	    pk->device() != q->device() || w->device() != q->device() ||
	    !q->is_contiguous() || !pk->is_contiguous() || !w->is_contiguous() ||
	    q->dim() != 3 || q->size(1) < 32 || q->size(1) % 32 || q->size(2) != 128 ||
	    pk->dim() != 2 || pk->size(1) != 128 ||
	    w->dim() != 2 || w->size(0) != q->size(0) || w->size(1) != q->size(1))
		RAISE("idx_score: q (R,H,128) fp16 with H a multiple of 32, pools (n,128) fp16, w (R,H) fp32, "
		      "contiguous on one CUDA device");

	long R = q->size(0), n = pk->size(0);
	double scale = (ucv_type(sv) == UC_DOUBLE) ? ucv_double_get(sv) : (double)ucv_int64_get(sv);
	int64_t pos0 = pt ? 0 : ucv_int64_get(pv), P = ucv_int64_get(kv);

	if (P < 1 || pos0 < 0 || n > 0x7fffff00L / (P + 1))
		RAISE("idx_score: bad pos0 / P");

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty({ R, n },
			at::TensorOptions().dtype(at::kHalf).device(q->device()));

		int rc = amp_idx_score_q8(q->data_ptr(), pqs ? NULL : pk->data_ptr(), pqs ? pk->data_ptr() : NULL,
			pqs ? pqs->data_ptr() : NULL, w->data_ptr(), out.data_ptr(),
			(int)R, (int)n, (float)scale, (int)pos0, (int)P, (int)q->size(1),
			pt ? (const long long *)pt->data_ptr() : NULL,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "idx_score"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dsa_select(scores, pos, P, kp): [idx, pad]
 *
 * Decode-time pool selection of the DSA sparse MLA layers (amp_dsa_select
 * in src/torch/amp_kernels.cu): per row of scores (T, cap) fp16 the top kp
 * pools (score descending, ties to the lower index; pools scored -65504
 * are padding), their P member tokens in ascending pool order, then the
 * P - 1 tokens of the row's incomplete tail pool (row r is at position
 * pos + r, pos a one-element int64 device tensor). Returns idx (T, K)
 * int64 (0 where padded) and pad (T, K) fp16 (0 / -65504), K = kp * P + P - 1.
 */
extern "C" int
amp_dsa_select(const void *scores, const long long *posp, void *idx, void *pad,
              int T, int cap, int P, int kp, cudaStream_t stream);

static uc_value_t *
torch_dsa_select(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 4)
		RAISE("dsa_select: expected (scores, pos, P, kp)");

	at::Tensor *sc = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *ps = sc ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	uc_value_t *pv = uc_fn_arg(2), *kv = uc_fn_arg(3);

	if (!sc || !ps)
		return NULL;

	if (!pv || !kv || ucv_type(pv) != UC_INTEGER || ucv_type(kv) != UC_INTEGER)
		RAISE("dsa_select: P and kp must be integers");

	int64_t P = ucv_int64_get(pv), kp = ucv_int64_get(kv);

	if (sc->scalar_type() != at::kHalf || sc->device().type() != at::kCUDA || sc->dim() != 2 ||
	    !sc->is_contiguous() || ps->scalar_type() != at::kLong || ps->numel() != 1 ||
	    ps->device() != sc->device() || P < 1 || P > 1024 || kp < 1 || kp > 4096 ||
	    sc->size(1) < kp || sc->size(0) < 1 || sc->size(0) > 1024)
		RAISE("dsa_select: scores (T, cap >= kp) fp16 contiguous on CUDA, pos one int64 on the same "
		      "device, 1 <= P <= 1024, 1 <= kp <= 4096");

	TensorDeviceGuard dg(*sc);

	try {
		int64_t T = sc->size(0), K = kp * P + P - 1;
		at::Tensor idx = at::empty({ T, K }, at::TensorOptions().dtype(at::kLong).device(sc->device()));
		at::Tensor pad = at::empty({ T, K }, at::TensorOptions().dtype(at::kHalf).device(sc->device()));

		int rc = amp_dsa_select(sc->data_ptr(), (const long long *)ps->data_ptr(), idx.data_ptr(),
			pad.data_ptr(), (int)T, (int)sc->size(1), (int)P, (int)kp,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dsa_select"))
			return NULL;

		uc_value_t *out = ucv_array_new(vm);

		ucv_array_push(out, tensor_wrap(vm, std::move(idx), 0));
		ucv_array_push(out, tensor_wrap(vm, std::move(pad), 0));

		return out;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * mem_info(dev): [allocated, reserved, free, total, retries, mallocs, frees] of CUDA device
 * `dev` — what the caching allocator hands to tensors, what it holds from
 * the driver, and the driver's view (cudaMemGetInfo, i.e. including the
 * CUDA context and non-PyTorch allocations), then the allocator's counts of
 * retries (an allocation that failed, freed the cache and tried again),
 * cudaMalloc and cudaFree calls since the start.
 */
static uc_value_t *
torch_mem_info(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *dv = uc_fn_arg(0);

	if (nargs != 1 || !dv || ucv_type(dv) != UC_INTEGER)
		RAISE("mem_info: expected (device index)");

	int dev = (int)ucv_int64_get(dv);

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)dev);
		int64_t alloc = 0, reserved = 0, retries = 0, mallocs = 0, frees = 0;

		// (the allocator knows a device only after its first tensor: until then there is nothing
		// allocated, and asking raises "Invalid device argument")
		try {
			auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(dev);

			alloc = (int64_t)st.allocated_bytes[0].current;
			reserved = (int64_t)st.reserved_bytes[0].current;
			retries = (int64_t)st.num_alloc_retries;
			mallocs = (int64_t)st.num_device_alloc;
			frees = (int64_t)st.num_device_free;
		}
		catch (const std::exception &) {
		}

		size_t fr = 0, tot = 0;

		cudaMemGetInfo(&fr, &tot);

		uc_value_t *a = ucv_array_new_length(vm, 7);

		ucv_array_push(a, ucv_int64_new(alloc));
		ucv_array_push(a, ucv_int64_new(reserved));
		ucv_array_push(a, ucv_int64_new((int64_t)fr));
		ucv_array_push(a, ucv_int64_new((int64_t)tot));
		ucv_array_push(a, ucv_int64_new(retries));
		ucv_array_push(a, ucv_int64_new(mallocs));
		ucv_array_push(a, ucv_int64_new(frees));

		return a;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * device_props(dev): { name, major, minor, total, sm_count } of CUDA device
 * `dev` (cudaGetDeviceProperties) — the identity an RPC worker announces in
 * its hello (docs/RPC.md). Does not create a context on the device.
 */
static uc_value_t *
torch_device_props(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *dv = uc_fn_arg(0);
	int n = 0;

	if (nargs != 1 || !dv || ucv_type(dv) != UC_INTEGER)
		RAISE("device_props: expected (device index)");

	int dev = (int)ucv_int64_get(dv);

	if (cudaGetDeviceCount(&n) != cudaSuccess || dev < 0 || dev >= n)
		RAISE("device_props: no such CUDA device");

	cudaDeviceProp pr;
	cudaError_t e = cudaGetDeviceProperties(&pr, dev);

	if (e != cudaSuccess)
		RAISE(cudaGetErrorString(e));

	uc_value_t *o = ucv_object_new(vm);

	ucv_object_add(o, "name", ucv_string_new(pr.name));
	ucv_object_add(o, "major", ucv_int64_new(pr.major));
	ucv_object_add(o, "minor", ucv_int64_new(pr.minor));
	ucv_object_add(o, "total", ucv_int64_new((int64_t)pr.totalGlobalMem));
	ucv_object_add(o, "sm_count", ucv_int64_new(pr.multiProcessorCount));

	return o;
}

/* torch.empty_cache(): give the caching allocator's free blocks (on every device) back to the driver */
static uc_value_t *
torch_empty_cache(uc_vm_t *vm, size_t nargs)
{
	try {
		c10::cuda::CUDACachingAllocator::emptyCache();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
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

extern "C" int
amp_kvq8_put(const void *src, void *q, void *s, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream);

extern "C" int
amp_kvq8_get(const void *q, const void *s, void *out, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream);

static bool
kvq8_check(uc_vm_t *vm, at::Tensor *q, at::Tensor *s, long *dim, const char *what)
{
	if (!q || !s)
		return false;

	if (q->scalar_type() != at::kChar || q->device().type() != at::kCUDA ||
	    !q->is_contiguous() || q->dim() != 2 ||
	    s->scalar_type() != at::kHalf || s->device() != q->device() ||
	    !s->is_contiguous() || s->dim() != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: q must be a contiguous (cap, dim) int8 CUDA tensor and s "
			"a contiguous (cap, dim/32) fp16 tensor on the same device", what);
		return false;
	}

	*dim = q->size(1);

	if (*dim % 32 || *dim > 4096 || s->size(1) != *dim / 32 || s->size(0) != q->size(0)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: dim must be a multiple of 32 (<= 4096) with matching scales", what);
		return false;
	}

	return true;
}

static uc_value_t *
torch_kv_q8_put(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (src, q, s, pos)");
		return NULL;
	}

	at::Tensor *src = fused_arg(vm, uc_fn_arg(0), "kv_q8_put: src", false);
	at::Tensor *q = src ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *s = q ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	long dim = 0;

	if (!kvq8_check(vm, q, s, &dim, "kv_q8_put"))
		return NULL;

	if (src->device() != q->device() || src->dim() != 2 || src->size(1) != dim)
		RAISEF("kv_q8_put: src must be (rows, %ld) fp16 on the cache's device", dim);

	long rows = src->size(0), start = 0;
	const long long *idx = NULL;
	uc_value_t *pv = uc_fn_arg(3);

	if (pv && ucv_type(pv) == UC_RESOURCE) {
		at::Tensor *pt = tensor_arg(vm, pv);

		if (!pt)
			return NULL;

		if (pt->scalar_type() != at::kLong || pt->device() != q->device() ||
		    !pt->is_contiguous() || pt->numel() != rows)
			RAISEF("kv_q8_put: pos must be an int64 tensor of %ld rows on the cache's device", rows);

		idx = (const long long *)pt->data_ptr();
	}
	else {
		if (!pv || ucv_type(pv) != UC_INTEGER)
			RAISE("kv_q8_put: pos must be an integer or an int64 tensor");

		start = (long)ucv_int64_get(pv);

		if (start < 0 || start + rows > q->size(0))
			RAISEF("kv_q8_put: rows %ld..%ld out of range (capacity %ld)",
				start, start + rows, (long)q->size(0));
	}

	TensorDeviceGuard dg(*q);

	try {
		int rc = amp_kvq8_put(src->data_ptr(), q->data_ptr(), s->data_ptr(), idx,
			start, rows, (int)dim, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kv_q8_put"))
			return NULL;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

static uc_value_t *
torch_kv_q8_get(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 3 && nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, s, idx) or (q, s, start, rows)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *s = q ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	long dim = 0, rows = 0, start = 0;
	const long long *idx = NULL;

	if (!kvq8_check(vm, q, s, &dim, "kv_q8_get"))
		return NULL;

	uc_value_t *av = uc_fn_arg(2);

	if (nargs == 3) {
		at::Tensor *it = tensor_arg(vm, av);

		if (!it)
			return NULL;

		if (it->scalar_type() != at::kLong || it->device() != q->device() ||
		    !it->is_contiguous())
			RAISE("kv_q8_get: idx must be a contiguous int64 tensor on the cache's device");

		idx = (const long long *)it->data_ptr();
		rows = it->numel();
	}
	else {
		uc_value_t *bv = uc_fn_arg(3);

		if (!av || !bv || ucv_type(av) != UC_INTEGER || ucv_type(bv) != UC_INTEGER)
			RAISE("kv_q8_get: start and rows must be integers");

		start = (long)ucv_int64_get(av);
		rows = (long)ucv_int64_get(bv);

		if (start < 0 || rows < 0 || start + rows > q->size(0))
			RAISEF("kv_q8_get: rows %ld..%ld out of range (capacity %ld)",
				start, start + rows, (long)q->size(0));
	}

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty({ rows, dim },
			at::TensorOptions().dtype(at::kHalf).device(q->device()));

		int rc = amp_kvq8_get(q->data_ptr(), s->data_ptr(), out.data_ptr(), idx,
			start, rows, (int)dim, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kv_q8_get"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* Q4 variants of the above (amp_kvq4_*): q is (cap, dim/2) bytes, two signed
 * 4-bit codes each, scales absmax / 7. */
extern "C" int
amp_kvq4_put(const void *src, void *q, void *s, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream);

extern "C" int
amp_kvq4_get(const void *q, const void *s, void *out, const long long *idx,
            long start, long rows, int dim, cudaStream_t stream);

static bool
kvq4_check(uc_vm_t *vm, at::Tensor *q, at::Tensor *s, long *dim, const char *what)
{
	if (!q || !s)
		return false;

	if (q->scalar_type() != at::kChar || q->device().type() != at::kCUDA ||
	    !q->is_contiguous() || q->dim() != 2 ||
	    s->scalar_type() != at::kHalf || s->device() != q->device() ||
	    !s->is_contiguous() || s->dim() != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: q must be a contiguous (cap, dim/2) int8 CUDA tensor and s "
			"a contiguous (cap, dim/32) fp16 tensor on the same device", what);
		return false;
	}

	*dim = q->size(1) * 2;

	if (*dim % 32 || *dim > 4096 || s->size(1) != *dim / 32 || s->size(0) != q->size(0)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: dim must be a multiple of 32 (<= 4096) with matching scales", what);
		return false;
	}

	return true;
}

static uc_value_t *
torch_kv_q4_put(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (src, q, s, pos)");
		return NULL;
	}

	at::Tensor *src = fused_arg(vm, uc_fn_arg(0), "kv_q4_put: src", false);
	at::Tensor *q = src ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *s = q ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	long dim = 0;

	if (!kvq4_check(vm, q, s, &dim, "kv_q4_put"))
		return NULL;

	if (src->device() != q->device() || src->dim() != 2 || src->size(1) != dim)
		RAISEF("kv_q4_put: src must be (rows, %ld) fp16 on the cache's device", dim);

	long rows = src->size(0), start = 0;
	const long long *idx = NULL;
	uc_value_t *pv = uc_fn_arg(3);

	if (pv && ucv_type(pv) == UC_RESOURCE) {
		at::Tensor *pt = tensor_arg(vm, pv);

		if (!pt)
			return NULL;

		if (pt->scalar_type() != at::kLong || pt->device() != q->device() ||
		    !pt->is_contiguous() || pt->numel() != rows)
			RAISEF("kv_q4_put: pos must be an int64 tensor of %ld rows on the cache's device", rows);

		idx = (const long long *)pt->data_ptr();
	}
	else {
		if (!pv || ucv_type(pv) != UC_INTEGER)
			RAISE("kv_q4_put: pos must be an integer or an int64 tensor");

		start = (long)ucv_int64_get(pv);

		if (start < 0 || start + rows > q->size(0))
			RAISEF("kv_q4_put: rows %ld..%ld out of range (capacity %ld)",
				start, start + rows, (long)q->size(0));
	}

	TensorDeviceGuard dg(*q);

	try {
		int rc = amp_kvq4_put(src->data_ptr(), q->data_ptr(), s->data_ptr(), idx,
			start, rows, (int)dim, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kv_q4_put"))
			return NULL;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

static uc_value_t *
torch_kv_q4_get(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 3 && nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, s, idx) or (q, s, start, rows)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *s = q ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	long dim = 0, rows = 0, start = 0;
	const long long *idx = NULL;

	if (!kvq4_check(vm, q, s, &dim, "kv_q4_get"))
		return NULL;

	uc_value_t *av = uc_fn_arg(2);

	if (nargs == 3) {
		at::Tensor *it = tensor_arg(vm, av);

		if (!it)
			return NULL;

		if (it->scalar_type() != at::kLong || it->device() != q->device() ||
		    !it->is_contiguous())
			RAISE("kv_q4_get: idx must be a contiguous int64 tensor on the cache's device");

		idx = (const long long *)it->data_ptr();
		rows = it->numel();
	}
	else {
		uc_value_t *bv = uc_fn_arg(3);

		if (!av || !bv || ucv_type(av) != UC_INTEGER || ucv_type(bv) != UC_INTEGER)
			RAISE("kv_q4_get: start and rows must be integers");

		start = (long)ucv_int64_get(av);
		rows = (long)ucv_int64_get(bv);

		if (start < 0 || rows < 0 || start + rows > q->size(0))
			RAISEF("kv_q4_get: rows %ld..%ld out of range (capacity %ld)",
				start, start + rows, (long)q->size(0));
	}

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty({ rows, dim },
			at::TensorOptions().dtype(at::kHalf).device(q->device()));

		int rc = amp_kvq4_get(q->data_ptr(), s->data_ptr(), out.data_ptr(), idx,
			start, rows, (int)dim, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kv_q4_get"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

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

/* int64 CUDA tensor argument */
static at::Tensor *
i64_arg(uc_vm_t *vm, uc_value_t *v, const char *what)
{
	at::Tensor *t = tensor_arg(vm, v);

	if (!t)
		return NULL;

	if (t->scalar_type() != at::kLong || !t->is_cuda() || !t->is_contiguous()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: must be a contiguous int64 CUDA tensor", what);
		return NULL;
	}

	return t;
}

/*
 * dsv4_qkv(q_raw, kv_raw, kv_norm_w, inv_freq, pos_f, pos_i, kvcache,
 *          rope_dim, eps): q_out
 *
 * DeepSeek-V4 decode: q_raw (H, D) fp16 -> per-head RMSNorm + GPT-J rope at
 * the DEVICE position; kv_raw (D) -> weighted RMSNorm + rope, written to
 * kvcache row pos_i. pos_f float32 (1), pos_i int64 (1), inv_freq float32.
 */
static uc_value_t *
torch_dsv4_qkv(uc_vm_t *vm, size_t nargs)
{
	int64_t rd, fp8 = 0;
	double eps, qscale = 1.0;

	if (nargs < 9 || nargs > 11 || !arg_int(vm, uc_fn_arg(7), &rd) ||
	    !arg_number(vm, uc_fn_arg(8), &eps) ||
	    (nargs >= 10 && !arg_number(vm, uc_fn_arg(9), &qscale)) ||
	    (nargs == 11 && !arg_int(vm, uc_fn_arg(10), &fp8))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q_raw, kv_raw, kv_norm_w, inv_freq, pos_f, pos_i, kvcache, rope_dim, eps, qscale?)");
		return NULL;
	}

	at::Tensor *q = fused_arg(vm, uc_fn_arg(0), "dsv4_qkv: q_raw", false);
	at::Tensor *kv = q ? fused_arg(vm, uc_fn_arg(1), "dsv4_qkv: kv_raw", false) : NULL;
	at::Tensor *w = kv ? fused_arg(vm, uc_fn_arg(2), "dsv4_qkv: kv_norm_w", false) : NULL;
	at::Tensor *fr = w ? f32_arg(vm, uc_fn_arg(3), "dsv4_qkv: inv_freq") : NULL;
	at::Tensor *pf = fr ? f32_arg(vm, uc_fn_arg(4), "dsv4_qkv: pos_f") : NULL;
	at::Tensor *pi = pf ? i64_arg(vm, uc_fn_arg(5), "dsv4_qkv: pos_i") : NULL;
	at::Tensor *cache = pi ? fused_arg(vm, uc_fn_arg(6), "dsv4_qkv: kvcache", false) : NULL;

	if (!cache)
		return NULL;

	/* q_raw (H, D) or (S, H, D); kv_raw (D) or (S, D) */
	int64_t qS = (q->dim() == 3) ? q->size(0) : 1;
	int64_t qH = (q->dim() == 3) ? q->size(1) : q->size(0);
	int64_t qD = q->size(-1);

	if ((q->dim() != 2 && q->dim() != 3) || kv->numel() != qS * qD || w->numel() != qD ||
	    fr->numel() * 2 != rd || cache->dim() != 2 || cache->size(1) != qD) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "dsv4_qkv: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty_like(*q);
		int rc = amp_dsv4_qkv_fp16(q->data_ptr(), kv->data_ptr(), w->data_ptr(),
			fr->data_ptr(), pf->data_ptr(), pi->data_ptr(), cache->data_ptr(),
			out.data_ptr(), (int)qS, (int)qH, (int)qD, (int)rd, (float)eps, (float)qscale, (int)fp8, (int)cache->size(0),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dsv4_qkv"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dsv4_comp(kv_row, gate_row, ape, buf_kv, buf_g, ovl_kv|null, ovl_g|null,
 *           n, pool, norm_w, inv_freq, pos_f, pos_i, m, rope_dim, eps)
 *
 * One token into a V4 compressor (in place): buffers the projected rows
 * (fp16 (W)) in buf_* (m, W) fp32 at slot pos % m and, when the window is
 * complete, pools / norms / ropes it and appends it to pool row n (int64
 * (1), incremented). ovl_* (m, hd) fp32 select the overlapping scheme.
 */
static uc_value_t *
torch_dsv4_comp(uc_vm_t *vm, size_t nargs)
{
	int64_t m, rd;
	double eps;

	if (nargs != 14 || !arg_int(vm, uc_fn_arg(11), &m) || !arg_int(vm, uc_fn_arg(12), &rd) ||
	    !arg_number(vm, uc_fn_arg(13), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (kv_rows, gate_rows, ape, buf_kv, buf_g, ovl_kv, ovl_g, pool, norm_w, "
			"inv_freq, pos_i, m, rope_dim, eps)");
		return NULL;
	}

	at::Tensor *kv = fused_arg(vm, uc_fn_arg(0), "dsv4_comp: kv_rows", false);
	at::Tensor *gt = kv ? fused_arg(vm, uc_fn_arg(1), "dsv4_comp: gate_rows", false) : NULL;
	at::Tensor *ape = gt ? f32_arg(vm, uc_fn_arg(2), "dsv4_comp: ape") : NULL;
	at::Tensor *bk = ape ? f32_arg(vm, uc_fn_arg(3), "dsv4_comp: buf_kv") : NULL;
	at::Tensor *bg = bk ? f32_arg(vm, uc_fn_arg(4), "dsv4_comp: buf_g") : NULL;
	at::Tensor *ok = NULL, *og = NULL;

	if (bg && ucv_type(uc_fn_arg(5)) != UC_NULL) {
		ok = f32_arg(vm, uc_fn_arg(5), "dsv4_comp: ovl_kv");
		og = ok ? f32_arg(vm, uc_fn_arg(6), "dsv4_comp: ovl_g") : NULL;

		if (!og)
			return NULL;
	}

	at::Tensor *pool = bg ? fused_arg(vm, uc_fn_arg(7), "dsv4_comp: pool", false) : NULL;
	at::Tensor *nw = pool ? fused_arg(vm, uc_fn_arg(8), "dsv4_comp: norm_w", false) : NULL;
	at::Tensor *fr = nw ? f32_arg(vm, uc_fn_arg(9), "dsv4_comp: inv_freq") : NULL;
	at::Tensor *pi = fr ? i64_arg(vm, uc_fn_arg(10), "dsv4_comp: pos_i") : NULL;

	if (!pi)
		return NULL;

	int hd = (int)pool->size(1);
	int64_t W = (ok ? 2 : 1) * (int64_t)hd;
	int64_t S = kv->numel() / W;

	if (S < 1 || kv->numel() != S * W || gt->numel() != S * W || ape->numel() != (int64_t)m * W ||
	    bk->numel() != (int64_t)m * W || bg->numel() != (int64_t)m * W || nw->numel() != hd ||
	    fr->numel() * 2 != rd ||
	    (ok && (ok->numel() != 3 * (int64_t)m * hd || og->numel() != 3 * (int64_t)m * hd))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "dsv4_comp: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*kv);

	int rc = amp_dsv4_comp_fp16(kv->data_ptr(), gt->data_ptr(), ape->data_ptr(),
		bk->data_ptr(), bg->data_ptr(), ok ? ok->data_ptr() : NULL,
		og ? og->data_ptr() : NULL, pool->data_ptr(), nw->data_ptr(),
		fr->data_ptr(), pi->data_ptr(), (int)S, (int)m, hd, (int)W, (int)rd,
		(float)eps, c10::cuda::getCurrentCUDAStream().stream());

	if (!launch_ok(vm, rc, "dsv4_comp"))
		return NULL;

	return ucv_boolean_new(true);
}

/*
 * dsv4_attn(q, kvcache, pos_i, pos_f, pool|null, idx|null, sinks, inv_freq,
 *           window, m, rope_dim, np_max, scale): out (H, D) fp16
 *
 * One-token V4 attention for all heads: sliding window of `window` rows of
 * kvcache ending at pos_i, plus the pool (dense: entries j with
 * (j + 1) * m <= pos + 1, at most np_max; gathered: idx int64 (K) rows,
 * -1 invalid; np_max = K), attention sinks (H) fp32, V = K, output rope
 * slice rotated back by the position.
 */
static uc_value_t *
torch_dsv4_attn(uc_vm_t *vm, size_t nargs)
{
	int64_t W, m, rd, npm, ncb = 0;
	double scale;

	if ((nargs != 13 && nargs != 14) || !arg_int(vm, uc_fn_arg(8), &W) || !arg_int(vm, uc_fn_arg(9), &m) ||
	    !arg_int(vm, uc_fn_arg(10), &rd) || !arg_int(vm, uc_fn_arg(11), &npm) ||
	    !arg_number(vm, uc_fn_arg(12), &scale) ||
	    (nargs == 14 && !arg_int(vm, uc_fn_arg(13), &ncb))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, kvcache, pos_i, pos_f, pool, idx, sinks, inv_freq, window, m, "
			"rope_dim, np_max, scale, nc_block?)");
		return NULL;
	}

	at::Tensor *q = fused_arg(vm, uc_fn_arg(0), "dsv4_attn: q", false);
	at::Tensor *kc = q ? fused_arg(vm, uc_fn_arg(1), "dsv4_attn: kvcache", false) : NULL;
	at::Tensor *pi = kc ? i64_arg(vm, uc_fn_arg(2), "dsv4_attn: pos_i") : NULL;
	at::Tensor *pf = pi ? f32_arg(vm, uc_fn_arg(3), "dsv4_attn: pos_f") : NULL;
	at::Tensor *pool = NULL, *idx = NULL;

	if (pf && ucv_type(uc_fn_arg(4)) != UC_NULL) {
		pool = fused_arg(vm, uc_fn_arg(4), "dsv4_attn: pool", false);

		if (!pool)
			return NULL;
	}

	if (pf && ucv_type(uc_fn_arg(5)) != UC_NULL) {
		idx = i64_arg(vm, uc_fn_arg(5), "dsv4_attn: idx");

		if (!idx)
			return NULL;
	}

	at::Tensor *sk = pf ? f32_arg(vm, uc_fn_arg(6), "dsv4_attn: sinks") : NULL;
	at::Tensor *fr = sk ? f32_arg(vm, uc_fn_arg(7), "dsv4_attn: inv_freq") : NULL;

	if (!fr)
		return NULL;

	int64_t aS = (q->dim() == 3) ? q->size(0) : 1;
	int64_t aH = (q->dim() == 3) ? q->size(1) : q->size(0);
	int64_t aD = q->size(-1);

	if ((q->dim() != 2 && q->dim() != 3) || sk->numel() != aH || kc->size(1) != aD ||
	    fr->numel() * 2 != rd || (idx && !pool) || (idx && idx->numel() != aS * npm)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "dsv4_attn: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty_like(*q);
		int rc = amp_dsv4_attn_fp16(q->data_ptr(), kc->data_ptr(), pi->data_ptr(),
			pf->data_ptr(), pool ? pool->data_ptr() : NULL, idx ? idx->data_ptr() : NULL,
			sk->data_ptr(), fr->data_ptr(), out.data_ptr(), (int)aS, (int)aH,
			(int)aD, (int)rd, (int)W, (int)m, (int)npm, (float)scale, (int)ncb, (int)kc->size(0),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dsv4_attn"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dsv4_derot(o, inv_freq, pos_f): out
 *
 * o (R, H, D) fp16: rotate the trailing rope slice back by the position pos_f[0] + r
 * (GPT-J pairs); returns a new fp16 tensor of the same shape.
 */
static uc_value_t *
torch_dsv4_derot(uc_vm_t *vm, size_t nargs)
{
	int64_t rd;

	if (nargs != 4 || !arg_int(vm, uc_fn_arg(3), &rd)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (o, inv_freq, pos_f, rope_dim)");
		return NULL;
	}

	at::Tensor *o = fused_arg(vm, uc_fn_arg(0), "dsv4_derot: o", false);
	at::Tensor *fr = o ? f32_arg(vm, uc_fn_arg(1), "dsv4_derot: inv_freq") : NULL;
	at::Tensor *pf = fr ? f32_arg(vm, uc_fn_arg(2), "dsv4_derot: pos_f") : NULL;

	if (!pf)
		return NULL;

	if (o->dim() != 3 || fr->numel() * 2 != rd) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "dsv4_derot: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*o);

	try {
		at::Tensor out = at::empty_like(*o);
		int rc = amp_dsv4_derot_fp16(o->data_ptr(), fr->data_ptr(), pf->data_ptr(), out.data_ptr(),
			(int)o->size(0), (int)o->size(1), (int)o->size(2), (int)rd,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dsv4_derot"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dsv4_mask(device, R, nk, ne, qpos0, base, window, m): (R, nk + ne) fp16
 *
 * Additive prefill mask: window columns c < nk (key row base + c) and pool columns
 * (entry j visible iff (j + 1) * m <= qpos + 1); query r is at qpos0 + r.
 */
static uc_value_t *
torch_dsv4_mask(uc_vm_t *vm, size_t nargs)
{
	int64_t dev, R, nk, ne, qpos0, base, W, m, ncstart = -1;

	if (nargs == 9 && !arg_int(vm, uc_fn_arg(8), &ncstart))
		return NULL;

	if ((nargs != 8 && nargs != 9) || !arg_int(vm, uc_fn_arg(0), &dev) || !arg_int(vm, uc_fn_arg(1), &R) ||
	    !arg_int(vm, uc_fn_arg(2), &nk) || !arg_int(vm, uc_fn_arg(3), &ne) ||
	    !arg_int(vm, uc_fn_arg(4), &qpos0) || !arg_int(vm, uc_fn_arg(5), &base) ||
	    !arg_int(vm, uc_fn_arg(6), &W) || !arg_int(vm, uc_fn_arg(7), &m)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (device, R, nk, ne, qpos0, base, window, m)");
		return NULL;
	}

	c10::cuda::CUDAGuard g((c10::DeviceIndex)dev);

	try {
		at::Tensor out = at::empty({ R, nk + ne }, at::TensorOptions().dtype(at::kHalf)
			.device(at::Device(at::kCUDA, (c10::DeviceIndex)dev)));
		int rc = amp_dsv4_mask_fp16(out.data_ptr(), (int)R, (int)nk, (int)ne, (long)qpos0, (long)base,
			(int)W, (int)(m > 0 ? m : 1), (long)ncstart, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dsv4_mask"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
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

	uc_value_t *specv = uc_fn_arg(6);

	if ((nargs != 6 && nargs != 7) || !arg_int(vm, uc_fn_arg(3), &Hk) ||
	    !arg_int(vm, uc_fn_arg(4), &Hv) || !arg_int(vm, uc_fn_arg(5), &hd) ||
	    (nargs == 7 && ucv_type(specv) != UC_NULL && ucv_type(specv) != UC_ARRAY)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (qkv, conv_state, conv_w, Hk, Hv, head_dim, snapshots?)");
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

	/* snapshots: per-token copies of the conv state window (same
	 * shape and layout as conv_state), written in the state kernel */
	void *snap[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	int nsnap = 0;

	if (nargs == 7 && ucv_type(specv) == UC_ARRAY) {
		nsnap = (int)ucv_array_length(specv);

		if (nsnap > 8 || nsnap > S) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"gdn_prep: at most 8 snapshots, and not more than the rows");
			return NULL;
		}

		for (int i = 0; i < nsnap; i++) {
			at::Tensor *t = fused_arg(vm, ucv_array_get(specv, i),
				"gdn_prep: snapshot", false);

			if (!t)
				return NULL;

			if (t->numel() != cs->numel() || t->get_device() != cs->get_device()) {
				uc_vm_raise_exception(vm, EXCEPTION_TYPE,
					"gdn_prep: snapshot must match conv_state");
				return NULL;
			}

			snap[i] = t->data_ptr();
		}
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
			snap, nsnap, c10::cuda::getCurrentCUDAStream().stream());

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
 * gdn_ab_gates(x, wa, wb, dt_bias, neg_exp_a_log): [beta, gexp]
 *
 * gdn_gates fed straight from the hidden state: a = x @ wa^T and
 * b = x @ wb^T (fp16 weights, fp32 accumulation, rounded to fp16) plus
 * the gate math, one launch, for S <= 8 rows. x (…, K) fp16 contiguous,
 * wa / wb (Hv, K) fp16 contiguous, dt_bias / neg_exp_a_log (Hv) fp32.
 * Returns beta, gexp as (S, Hv) fp32.
 */
static uc_value_t *
torch_gdn_ab_gates(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 5) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, wa, wb, dt_bias, neg_exp_a_log)");
		return NULL;
	}

	at::Tensor *x = fused_arg(vm, uc_fn_arg(0), "gdn_ab_gates: x", false);
	at::Tensor *wa = x ? fused_arg(vm, uc_fn_arg(1), "gdn_ab_gates: wa", false) : NULL;
	at::Tensor *wb = wa ? fused_arg(vm, uc_fn_arg(2), "gdn_ab_gates: wb", false) : NULL;
	at::Tensor *dtb = wb ? f32_arg(vm, uc_fn_arg(3), "gdn_ab_gates: dt_bias") : NULL;
	at::Tensor *nea = dtb ? f32_arg(vm, uc_fn_arg(4), "gdn_ab_gates: neg_exp_a_log") : NULL;

	if (!nea)
		return NULL;

	if (wa->dim() != 2 || wb->sizes() != wa->sizes() || x->dim() < 1) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_ab_gates: wa and wb must be (Hv, K)");
		return NULL;
	}

	int64_t Hv = wa->size(0), K = wa->size(1);

	if (Hv == 0 || K == 0 || (K & 1) || x->size(-1) != K || x->numel() % K != 0 ||
	    dtb->numel() != Hv || nea->numel() != Hv || x->numel() / K > 8) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"gdn_ab_gates: incompatible shapes (K even, at most 8 rows)");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		int64_t S = x->numel() / K;
		auto f32 = x->options().dtype(at::kFloat);
		at::Tensor beta = at::empty({ S, Hv }, f32);
		at::Tensor gexp = at::empty({ S, Hv }, f32);

		int rc = amp_gdn_ab_gates(x->data_ptr(), wa->data_ptr(), wb->data_ptr(),
			dtb->data_ptr(), nea->data_ptr(), beta.data_ptr(), gexp.data_ptr(),
			(int)S, (int)K, (int)Hv, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "gdn_ab_gates"))
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
 * dflash_conv(h, dyn, base, which, group): torch.tensor
 *
 * DFlash 2 two-tap grouped dynamic causal convolution over one draft
 * block (rows are positions; row 0 reads zero padding for tap 1):
 *   out[r] = (base[which][0] + dyn[r][which][0][g]) * h[r]
 *          + (base[which][1] + dyn[r][which][1][g]) * h[r - 1]
 * h (R, H) fp16 contiguous; dyn (R, 2 * 2 * H / group) fp16 contiguous
 * (the kernel projection's output, [which][tap][group]); base (2, 2, H)
 * fp16 contiguous. Returns (R, H).
 */
static uc_value_t *
torch_dflash_conv(uc_vm_t *vm, size_t nargs)
{
	int64_t which, group;

	if (nargs != 5 || !arg_int(vm, uc_fn_arg(3), &which) ||
	    !arg_int(vm, uc_fn_arg(4), &group)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (h, dyn, base, which, group)");
		return NULL;
	}

	at::Tensor *h = fused_arg(vm, uc_fn_arg(0), "dflash_conv: h", false);
	at::Tensor *dyn = h ? fused_arg(vm, uc_fn_arg(1), "dflash_conv: dyn", false) : NULL;
	at::Tensor *base = dyn ? fused_arg(vm, uc_fn_arg(2), "dflash_conv: base", false) : NULL;

	if (!base)
		return NULL;

	if (h->dim() != 2 || which < 0 || which > 1 || group < 1 ||
	    base->dim() != 3 || base->size(0) != 2 || base->size(1) != 2 ||
	    base->size(2) != h->size(1) || h->size(1) % group != 0 ||
	    dyn->dim() != 2 || dyn->size(0) != h->size(0) ||
	    dyn->size(1) != 4 * (h->size(1) / group)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"dflash_conv: incompatible shapes");
		return NULL;
	}

	TensorDeviceGuard dg(*h);

	try {
		int64_t R = h->size(0), H = h->size(1);
		int64_t groups = H / group;
		at::Tensor out = at::empty_like(*h);

		/* dyn is [which][tap][group] per row: this call's two taps */
		const __half *dp = (const __half *)dyn->data_ptr() + which * 2 * groups;
		const __half *bp = (const __half *)base->data_ptr() + which * 2 * H;

		int rc = amp_dflash_conv(h->data_ptr(), dp, bp, out.data_ptr(), (int)R,
			(int)H, (int)group, (long)dyn->size(1),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dflash_conv"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * rms_norm_f32(x, weight, eps): torch.tensor
 *
 * RMSNorm of an fp32 (R, H) tensor to fp16: fp16(x * rsqrt(mean(x^2) +
 * eps)) * weight, the rounding points of rms_norm. weight (H) fp16.
 */
static uc_value_t *
torch_rms_norm_f32(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs != 3 || !arg_number(vm, uc_fn_arg(2), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (x, weight, eps)");
		return NULL;
	}

	at::Tensor *x = f32_arg(vm, uc_fn_arg(0), "rms_norm_f32: x");
	at::Tensor *w = x ? fused_arg(vm, uc_fn_arg(1), "rms_norm_f32: weight", false) : NULL;

	if (!w)
		return NULL;

	if (x->dim() != 2 || !x->is_contiguous() || w->numel() != x->size(1) ||
	    x->device() != w->device()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"rms_norm_f32: x must be contiguous (R, H) fp32 and weight (H) on its device");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		at::Tensor out = at::empty(x->sizes(), x->options().dtype(at::kHalf));
		int rc = amp_rms_norm_f32in(x->data_ptr(), w->data_ptr(), out.data_ptr(),
			(long)x->size(0), (int)x->size(1), eps,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "rms_norm_f32"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * norm32(x, weight, eps [, res [, scale]]): (res + rms(x) * weight) * scale, fp32
 *
 * x (R, H) fp16 or fp32 contiguous, weight fp32 (H), res fp32 (R, H) or null, scale fp32 (1) or null.
 */
static uc_value_t *
torch_norm32(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs < 3 || nargs > 5 || !arg_number(vm, uc_fn_arg(2), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (x, weight, eps [, res [, scale]])");
		return NULL;
	}

	at::Tensor *x = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *w = x ? f32_arg(vm, uc_fn_arg(1), "norm32: weight") : NULL;
	at::Tensor *res = NULL, *scale = NULL;

	if (!w)
		return NULL;

	if (nargs >= 4 && ucv_type(uc_fn_arg(3)) != UC_NULL && !(res = f32_arg(vm, uc_fn_arg(3), "norm32: res")))
		return NULL;

	if (nargs >= 5 && ucv_type(uc_fn_arg(4)) != UC_NULL && !(scale = f32_arg(vm, uc_fn_arg(4), "norm32: scale")))
		return NULL;

	if ((x->scalar_type() != at::kFloat && x->scalar_type() != at::kHalf) || !x->is_cuda() ||
	    !x->is_contiguous() || x->dim() != 2 || w->numel() != x->size(1) || x->device() != w->device() ||
	    (res && (res->sizes() != x->sizes() || res->device() != x->device())) ||
	    (scale && (scale->numel() != 1 || scale->device() != x->device()))) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"norm32: x must be contiguous (R, H) fp16 / fp32, weight (H) fp32, res like x, scale one element, one device");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		at::Tensor out = at::empty(x->sizes(), x->options().dtype(at::kFloat));
		int rc = amp_norm32(x->data_ptr(), x->scalar_type() == at::kHalf, w->data_ptr(),
			res ? res->data_ptr() : NULL, scale ? scale->data_ptr() : NULL, out.data_ptr(),
			(long)x->size(0), (int)x->size(1), eps, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "norm32"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dflash_conv_res(h, y, dyn, base, which, group, scale): h
 *
 * h (R, H) fp32 += scale * conv2(y) in place, conv2 as dflash_conv.
 * Returns h.
 */
static uc_value_t *
torch_dflash_conv_res(uc_vm_t *vm, size_t nargs)
{
	int64_t which, group;
	double scale;

	if (nargs != 7 || !arg_int(vm, uc_fn_arg(4), &which) ||
	    !arg_int(vm, uc_fn_arg(5), &group) || !arg_number(vm, uc_fn_arg(6), &scale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (h, y, dyn, base, which, group, scale)");
		return NULL;
	}

	at::Tensor *h = f32_arg(vm, uc_fn_arg(0), "dflash_conv_res: h");
	at::Tensor *y = h ? fused_arg(vm, uc_fn_arg(1), "dflash_conv_res: y", false) : NULL;
	at::Tensor *dyn = y ? fused_arg(vm, uc_fn_arg(2), "dflash_conv_res: dyn", false) : NULL;
	at::Tensor *base = dyn ? fused_arg(vm, uc_fn_arg(3), "dflash_conv_res: base", false) : NULL;

	if (!base)
		return NULL;

	if (h->dim() != 2 || !h->is_contiguous() || y->sizes() != h->sizes() ||
	    which < 0 || which > 1 || group < 1 || base->dim() != 3 ||
	    base->size(0) != 2 || base->size(1) != 2 || base->size(2) != h->size(1) ||
	    h->size(1) % group != 0 || dyn->dim() != 2 || dyn->size(0) != h->size(0) ||
	    dyn->size(1) != 4 * (h->size(1) / group) || y->device() != h->device() ||
	    dyn->device() != h->device() || base->device() != h->device()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"dflash_conv_res: incompatible shapes or devices");
		return NULL;
	}

	TensorDeviceGuard dg(*h);

	try {
		int64_t R = h->size(0), H = h->size(1);
		int64_t groups = H / group;
		const __half *dp = (const __half *)dyn->data_ptr() + which * 2 * groups;
		const __half *bp = (const __half *)base->data_ptr() + which * 2 * H;

		int rc = amp_dflash_conv_res(h->data_ptr(), y->data_ptr(), dp, bp,
			(int)R, (int)H, (int)group, (long)dyn->size(1), (float)scale,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dflash_conv_res"))
			return NULL;

		return ucv_get(uc_fn_arg(0));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * dflash_select(unary, cand, hp, anchor, pred_codebook, succ_codebook):
 * torch.tensor
 *
 * The DFlash 2 candidate selector's greedy walk (see amp_kernels.cu):
 * unary (W, K) fp32, cand (W, K) int64, hp (W, R) fp16 (the hidden
 * projection of the draft hidden), anchor (1) int64, pred / succ
 * codebooks (V, R) fp16. Returns the chosen token ids (W) int64.
 */
static uc_value_t *
torch_dflash_select(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 6) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (unary, cand, hp, anchor, pred_codebook, succ_codebook)");
		return NULL;
	}

	at::Tensor *un = f32_arg(vm, uc_fn_arg(0), "dflash_select: unary");
	at::Tensor *cd = un ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *hp = cd ? fused_arg(vm, uc_fn_arg(2), "dflash_select: hp", false) : NULL;
	at::Tensor *an = hp ? tensor_arg(vm, uc_fn_arg(3)) : NULL;
	at::Tensor *pr = an ? fused_arg(vm, uc_fn_arg(4), "dflash_select: pred", false) : NULL;
	at::Tensor *sc = pr ? fused_arg(vm, uc_fn_arg(5), "dflash_select: succ", false) : NULL;

	if (!sc)
		return NULL;

	if (un->dim() != 2 || cd->scalar_type() != at::kLong || !cd->is_contiguous() ||
	    cd->sizes() != un->sizes() || hp->dim() != 2 || hp->size(0) != un->size(0) ||
	    an->scalar_type() != at::kLong || an->numel() != 1 ||
	    pr->dim() != 2 || pr->sizes() != sc->sizes() || pr->size(1) != hp->size(1) ||
	    un->size(1) > 32 || hp->size(1) > 1024 || un->size(0) < 1 ||
	    cd->device() != un->device() || hp->device() != un->device() ||
	    an->device() != un->device() || pr->device() != un->device() ||
	    sc->device() != un->device()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"dflash_select: incompatible shapes, dtypes or devices");
		return NULL;
	}

	TensorDeviceGuard dg(*un);

	try {
		int64_t W = un->size(0), K = un->size(1), R = hp->size(1);
		at::Tensor out = at::empty({ W }, cd->options());

		int rc = amp_dflash_select(un->data_ptr(), cd->data_ptr(), hp->data_ptr(),
			an->data_ptr(), pr->data_ptr(), sc->data_ptr(), out.data_ptr(),
			(int)W, (int)K, (int)R, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "dflash_select"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
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
 * CUDA-graph-safe: q (S, Hq, E) fp16 contiguous ((Hq/Hk) * S <= 32
 * and S <= 5, or <= 64 rows of any S for the mma variant), k/v caches
 * (Hk, max_seq, E) fp16 contiguous,
 * len (1) int32 on the device = keys in use (row s attends keys
 * [0, len - S + s]). splits: key splits per kv head (default: 4
 * blocks per SM over the kv heads, one per SM for mma). mma = 1 selects the
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
			/* shuffle kernel: ~4 blocks per SM over the kv heads,
			 * enough unique K/V bytes in flight (the block's warps
			 * share rows). The mma kernel runs one block per SM (its
			 * shared memory): exactly one wave, a partial second wave
			 * costs up to 40% (measured on a 3090, L132) */
			splits = use_mma ? std::max<int64_t>(1, (int64_t)sms / Hk) :
				std::max<int64_t>(1, (4 * (int64_t)sms + Hk - 1) / Hk);
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
 * copy_many(dsts, srcs): null
 *
 * dsts[i].copy_(srcs[i]) for every i, batched: one kernel launch per
 * device (pointer table) instead of one per pair. Pairs must be
 * contiguous, of equal byte size (a multiple of 16), on the same CUDA
 * device; dtypes are not converted. Runs on each device's current
 * stream, like copy_.
 */
static uc_value_t *
torch_copy_many(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *da = uc_fn_arg(0), *sa = uc_fn_arg(1);

	if (ucv_type(da) != UC_ARRAY || ucv_type(sa) != UC_ARRAY ||
	    ucv_array_length(da) != ucv_array_length(sa)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (dsts, srcs): tensor arrays of equal length");
		return NULL;
	}

	std::map<int, std::vector<size_t>> by_dev;
	std::vector<at::Tensor *> d, s;

	for (size_t i = 0; i < ucv_array_length(da); i++) {
		at::Tensor *dt = tensor_arg(vm, ucv_array_get(da, i));
		at::Tensor *st = dt ? tensor_arg(vm, ucv_array_get(sa, i)) : NULL;

		if (!st)
			return NULL;

		size_t bytes = dt->numel() * dt->element_size();

		if (!dt->is_cuda() || !st->is_cuda() || dt->get_device() != st->get_device() ||
		    !dt->is_contiguous() || !st->is_contiguous() ||
		    bytes != st->numel() * st->element_size() || bytes % 16) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"copy_many: pair %zu: contiguous CUDA tensors on one device, "
				"equal byte size (multiple of 16)", i);
			return NULL;
		}

		d.push_back(dt);
		s.push_back(st);
		by_dev[dt->get_device()].push_back(i);
	}

	try {
		for (auto &kv : by_dev) {
			c10::cuda::CUDAGuard g((c10::DeviceIndex)kv.first);
			std::vector<const void *> src;
			std::vector<void *> dst;
			std::vector<long> bytes;

			for (size_t i : kv.second) {
				src.push_back(s[i]->data_ptr());
				dst.push_back(d[i]->data_ptr());
				bytes.push_back((long)(d[i]->numel() * d[i]->element_size()));
			}

			int rc = amp_copy_many(src.data(), dst.data(), bytes.data(),
				(int)src.size(), c10::cuda::getCurrentCUDAStream().stream());

			if (!launch_ok(vm, rc, "copy_many"))
				return NULL;
		}
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/*
 * copy_async(dst, src): null
 *
 * Enqueues a device-to-host copy of src into dst WITHOUT synchronizing:
 * dst must be a contiguous PINNED CPU tensor, src a contiguous CUDA
 * tensor of the same byte size (no dtype conversion). The transfer
 * runs on the source device's current stream, so a notifier armed on
 * that device afterwards fires once the results are visible in dst —
 * the host never waits for the copy.
 */
static uc_value_t *
torch_copy_async(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *dst, *src;

	if (nargs != 2 || !(dst = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(src = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	if (!src->is_cuda() || dst->device().type() != at::kCPU ||
	    !dst->is_pinned() || !dst->is_contiguous() || !src->is_contiguous() ||
	    dst->numel() * dst->element_size() !=
	    src->numel() * src->element_size()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"copy_async: contiguous CUDA src, pinned CPU dst, "
			"equal byte size expected");
		return NULL;
	}

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)src->get_device());
		cudaError_t e = cudaMemcpyAsync(dst->data_ptr(), src->data_ptr(),
			src->numel() * src->element_size(), cudaMemcpyDeviceToHost,
			c10::cuda::getCurrentCUDAStream(
				(c10::DeviceIndex)src->get_device()).stream());

		if (e != cudaSuccess)
			RAISE(cudaGetErrorString(e));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/*
 * copy_h2d_async(dst, src): null
 *
 * Enqueues a host-to-device copy of src into dst WITHOUT synchronizing:
 * src must be a contiguous PINNED CPU tensor, dst a contiguous CUDA tensor
 * of the same byte size (no dtype conversion). The transfer runs on the
 * destination device's current stream (a stream_side branch for copies that
 * overlap compute); the pinned source must stay alive and unchanged until
 * the stream has passed the copy. at::Tensor::copy_ would synchronize the
 * host thread.
 */
static uc_value_t *
torch_copy_h2d_async(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *dst, *src;

	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"copy_h2d_async: expected (dst, src)");
		return NULL;
	}

	if (!(dst = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(src = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	if (!dst->is_cuda() || src->device().type() != at::kCPU ||
	    !src->is_pinned() || !dst->is_contiguous() || !src->is_contiguous() ||
	    dst->numel() * dst->element_size() !=
	    src->numel() * src->element_size()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"copy_h2d_async: contiguous pinned CPU src, contiguous CUDA dst, "
			"equal byte size expected");
		return NULL;
	}

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)dst->get_device());
		cudaError_t e = cudaMemcpyAsync(dst->data_ptr(), src->data_ptr(),
			dst->numel() * dst->element_size(), cudaMemcpyHostToDevice,
			c10::cuda::getCurrentCUDAStream(
				(c10::DeviceIndex)dst->get_device()).stream());

		if (e != cudaSuccess)
			RAISE(cudaGetErrorString(e));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/*
 * Raw bytes of a CPU tensor over a socket (docs/RPC.md, phase 1): the
 * activations of a pipeline hop go from a pinned host buffer to the wire
 * and back without passing through a ucode string.
 *
 *   fd_write(fd, t, off, n): int   one non-blocking send of n bytes at byte
 *                                   offset off; returns the bytes written,
 *                                   0 when the socket is full (EAGAIN)
 *   fd_read(fd, t, off, n): int    one non-blocking receive into t; returns
 *                                   the bytes read, 0 when nothing is there
 *                                   (EAGAIN), -1 on an orderly shutdown
 *   fd_send_all(fd, t, timeout_ms?): null   the whole tensor, polling while
 *                                   the socket is full
 *   fd_recv_all(fd, t, timeout_ms?): null   fills the whole tensor; raises
 *                                   when the peer closes first
 *
 * fd is the integer of socket.fileno(); t a contiguous CPU tensor (a view
 * from narrow() sends a slice). timeout_ms (default: none) bounds the whole
 * call; a timeout, a reset connection or a bad fd raise. Nothing is
 * retained: the bytes are read from / written to t's storage in place, so t
 * must stay alive and unchanged until the call returns.
 */
static bool
fd_buf_args(uc_vm_t *vm, const char *who, uc_value_t *fdv, uc_value_t *tv,
            int *fd, char **base, size_t *size)
{
	at::Tensor *t;

	if (ucv_type(fdv) != UC_INTEGER || ucv_int64_get(fdv) < 0 ||
	    ucv_int64_get(fdv) > INT_MAX) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: expected a file descriptor (integer >= 0)", who);
		return false;
	}

	if (!(t = tensor_arg(vm, tv)))
		return false;

	if (t->device().type() != at::kCPU || !t->is_contiguous()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"%s: a contiguous CPU tensor is expected", who);
		return false;
	}

	*fd = (int)ucv_int64_get(fdv);
	*base = (char *)t->data_ptr();
	*size = (size_t)t->numel() * t->element_size();

	return true;
}

static bool
fd_range_args(uc_vm_t *vm, const char *who, uc_value_t *offv, uc_value_t *nv,
              size_t size, size_t *off, size_t *n)
{
	if (ucv_type(offv) != UC_INTEGER || ucv_type(nv) != UC_INTEGER ||
	    ucv_int64_get(offv) < 0 || ucv_int64_get(nv) < 0 ||
	    (uint64_t)ucv_int64_get(offv) > size ||
	    (uint64_t)ucv_int64_get(nv) > size - (uint64_t)ucv_int64_get(offv)) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"%s: offset / length outside the tensor (%zu bytes)", who, size);
		return false;
	}

	*off = (size_t)ucv_int64_get(offv);
	*n = (size_t)ucv_int64_get(nv);

	return true;
}

/* One send / recv attempt: bytes moved, 0 for EAGAIN, -1 for an orderly
 * EOF (recv only); raises (returns -2) on every other error. */
static ssize_t
fd_once(uc_vm_t *vm, const char *who, int fd, char *p, size_t n, bool wr)
{
	for (;;) {
		ssize_t r = wr ? send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT)
		               : recv(fd, p, n, MSG_DONTWAIT);

		if (r > 0 || (r == 0 && n == 0))
			return r;

		if (r == 0)
			return -1;

		if (errno == EINTR)
			continue;

		if (errno == EAGAIN || errno == EWOULDBLOCK)
			return 0;

		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "%s: %s", who, strerror(errno));

		return -2;
	}
}

static uc_value_t *
fd_step(uc_vm_t *vm, size_t nargs, const char *who, bool wr)
{
	int fd;
	char *base;
	size_t size, off, n;

	if (nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "%s: expected (fd, tensor, offset, length)", who);
		return NULL;
	}

	if (!fd_buf_args(vm, who, uc_fn_arg(0), uc_fn_arg(1), &fd, &base, &size) ||
	    !fd_range_args(vm, who, uc_fn_arg(2), uc_fn_arg(3), size, &off, &n))
		return NULL;

	ssize_t r = fd_once(vm, who, fd, base + off, n, wr);

	if (r == -2)
		return NULL;

	return ucv_int64_new(r);
}

static uc_value_t *
torch_fd_write(uc_vm_t *vm, size_t nargs)
{
	return fd_step(vm, nargs, "fd_write", true);
}

static uc_value_t *
torch_fd_read(uc_vm_t *vm, size_t nargs)
{
	return fd_step(vm, nargs, "fd_read", false);
}

static int64_t
mono_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);

	return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static uc_value_t *
fd_all(uc_vm_t *vm, size_t nargs, const char *who, bool wr)
{
	int fd;
	char *base;
	size_t size, done = 0;
	int64_t deadline = -1;

	if (nargs < 2 || nargs > 3) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "%s: expected (fd, tensor, timeout_ms?)", who);
		return NULL;
	}

	if (!fd_buf_args(vm, who, uc_fn_arg(0), uc_fn_arg(1), &fd, &base, &size))
		return NULL;

	if (nargs == 3 && ucv_type(uc_fn_arg(2)) != UC_NULL) {
		if (ucv_type(uc_fn_arg(2)) != UC_INTEGER || ucv_int64_get(uc_fn_arg(2)) < 0) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"%s: timeout_ms must be an integer >= 0", who);
			return NULL;
		}

		deadline = mono_ms() + ucv_int64_get(uc_fn_arg(2));
	}

	while (done < size) {
		ssize_t r = fd_once(vm, who, fd, base + done, size - done, wr);

		if (r == -2)
			return NULL;

		if (r < 0) {
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
				"%s: connection closed after %zu of %zu bytes", who, done, size);
			return NULL;
		}

		if (r > 0) {
			done += (size_t)r;
			continue;
		}

		int wait_ms = -1;

		if (deadline >= 0) {
			int64_t left = deadline - mono_ms();

			if (left <= 0) {
				uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
					"%s: timeout after %zu of %zu bytes", who, done, size);
				return NULL;
			}

			wait_ms = left > INT_MAX ? INT_MAX : (int)left;
		}

		struct pollfd pfd = { .fd = fd, .events = (short)(wr ? POLLOUT : POLLIN), .revents = 0 };
		int pr = poll(&pfd, 1, wait_ms);

		if (pr < 0 && errno != EINTR) {
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "%s: poll: %s", who, strerror(errno));
			return NULL;
		}
	}

	return NULL;
}

static uc_value_t *
torch_fd_send_all(uc_vm_t *vm, size_t nargs)
{
	return fd_all(vm, nargs, "fd_send_all", true);
}

static uc_value_t *
torch_fd_recv_all(uc_vm_t *vm, size_t nargs)
{
	return fd_all(vm, nargs, "fd_recv_all", false);
}

/*
 * common_prefix(a, b): int
 *
 * Length of the common prefix of two arrays of integers (token ids),
 * compared natively — the per-request prompt-reuse check walks arrays of
 * 100k+ ids, ~1 us per element in a ucode loop.
 */
static uc_value_t *
torch_common_prefix(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *a = uc_fn_arg(0), *b = uc_fn_arg(1);

	if (ucv_type(a) != UC_ARRAY || ucv_type(b) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (array, array)");
		return NULL;
	}

	size_t n = std::min(ucv_array_length(a), ucv_array_length(b)), i = 0;

	for (; i < n; i++) {
		uc_value_t *x = ucv_array_get(a, i), *y = ucv_array_get(b, i);

		if (ucv_type(x) != UC_INTEGER || ucv_type(y) != UC_INTEGER ||
		    ucv_int64_get(x) != ucv_int64_get(y))
			break;
	}

	return ucv_int64_new((int64_t)i);
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

struct comm_state {
	int N, mode;
	long cap;
	std::vector<int> dev;
	std::vector<at::Tensor> buf, ctl;
	void *bufp[AMP_COMM_MAXN];
	int *ctlp[AMP_COMM_MAXN];
};

static void
comm_free(void *data)
{
	delete static_cast<comm_state *>(data);
}

static uc_value_t *
torch_p2p_comm(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *devs = uc_fn_arg(0);
	int64_t cap;
	int mode = -1;

	if (ucv_type(devs) != UC_ARRAY || !arg_int(vm, uc_fn_arg(1), &cap) || cap <= 0 || cap % 8) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (devices array, cap (multiple of 8), mode?)");
		return NULL;
	}

	uc_value_t *mv = uc_fn_arg(2);

	if (nargs > 2 && ucv_type(mv) == UC_STRING) {
		const char *m = ucv_string_get(mv);

		mode = !strcmp(m, "oneshot") ? 0 : !strcmp(m, "twophase") ? 1 :
			!strcmp(m, "auto") ? -1 : -2;

		if (mode == -2) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"mode must be auto, oneshot or twophase");
			return NULL;
		}
	}

	size_t N = ucv_array_length(devs);

	if (N < 2 || N > AMP_COMM_MAXN) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "2..8 devices supported");
		return NULL;
	}

	auto *c = new comm_state();

	c->N = (int)N;
	c->cap = cap;
	c->mode = (mode >= 0) ? mode : (N <= 3 ? 0 : 1);

	try {
		for (size_t i = 0; i < N; i++) {
			uc_value_t *v = ucv_array_get(devs, i);

			if (ucv_type(v) != UC_INTEGER || ucv_int64_get(v) < 0 ||
			    ucv_int64_get(v) >= at::cuda::device_count())
				throw std::runtime_error("bad device index");

			c->dev.push_back((int)ucv_int64_get(v));
		}

		for (int a : c->dev)
			for (int b : c->dev) {
				if (a == b)
					continue;

				c10::cuda::CUDAGuard g((c10::DeviceIndex)a);
				cudaError_t rc = cudaDeviceEnablePeerAccess(b, 0);

				if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled)
					throw std::runtime_error(cudaGetErrorString(rc));

				(void)cudaGetLastError();
			}

		/* The buffers below are written by the peers. With the allocator's
		 * expandable segments, memory is only mapped to the peers whose
		 * access ATen has already enabled — which it does lazily, on the
		 * first cross-device copy of a pair. Make that copy now. */
		for (int a : c->dev)
			for (int b : c->dev)
				if (a != b)
					(void)at::zeros({ 8 }, at::TensorOptions().device(at::kCUDA, a))
						.to(at::Device(at::kCUDA, b));

		at::cuda::device_synchronize();

		for (size_t i = 0; i < N; i++) {
			auto o = at::TensorOptions().device(at::kCUDA, c->dev[i]);

			c->buf.push_back(at::zeros({ (int64_t)(2 * N + 2) * cap }, o.dtype(at::kHalf)));
			c->ctl.push_back(at::zeros({ (2 * AMP_COMM_MAXN + 1) * AMP_COMM_MAXB }, o.dtype(at::kInt)));
			c->bufp[i] = c->buf[i].data_ptr();
			c->ctlp[i] = (int *)c->ctl[i].data_ptr();
		}

		/* the zero fills must have landed on EVERY rank's device before
		 * the first all-reduce writes flags into a peer's buffers (a plain
		 * device_synchronize only covers the current device) */
		for (int d : c->dev) {
			c10::cuda::CUDAGuard g((c10::DeviceIndex)d);

			at::cuda::device_synchronize();
		}
	}
	catch (const std::exception &e) {
		delete c;
		RAISE(e.what());
	}

	return uc_resource_new(ucv_resource_type_lookup(vm, COMM_TYPE), c);
}

static uc_value_t *
comm_allreduce(uc_vm_t *vm, size_t nargs)
{
	auto *c = static_cast<comm_state *>(uc_fn_thisval(COMM_TYPE));
	int64_t rank;

	if (!c) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "called on non-comm value");
		return NULL;
	}

	at::Tensor *x = tensor_arg(vm, uc_fn_arg(0));

	if (!x || !arg_int(vm, uc_fn_arg(1), &rank) || rank < 0 || rank >= c->N) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (x, rank)");
		return NULL;
	}

	if (!x->is_cuda() || x->get_device() != c->dev[rank] ||
	    x->scalar_type() != at::kHalf || !x->is_contiguous() ||
	    x->numel() % 8 || x->numel() > c->cap) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"comm.allreduce: x must be contiguous fp16 on the rank's device, "
			"n %% 8 == 0, n <= cap");
		return NULL;
	}

	TensorDeviceGuard dg(*x);

	try {
		long n = (long)x->numel();
		/* blocks: one per ~16k values per chunk, as all ranks compute it */
		static const long blk_div = getenv("AMP_COMM_BLOCK_DIV") ?
			atol(getenv("AMP_COMM_BLOCK_DIV")) : 16384;
		int blocks = (int)std::min<long>(AMP_COMM_MAXB,
			std::max<long>(1, n / c->N / blk_div));
		at::Tensor out = at::empty_like(*x);
		int rc = amp_comm_allreduce_fp16(x->data_ptr(), out.data_ptr(),
			c->bufp, c->ctlp, (int)rank, c->N, n, c->cap, c->mode, blocks,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "comm.allreduce"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

static uc_value_t *
comm_info(uc_vm_t *vm, size_t nargs)
{
	auto *c = static_cast<comm_state *>(uc_fn_thisval(COMM_TYPE));
	uc_value_t *o;

	if (!c)
		return NULL;

	o = ucv_object_new(vm);
	ucv_object_add(o, "ranks", ucv_int64_new(c->N));
	ucv_object_add(o, "cap", ucv_int64_new(c->cap));
	ucv_object_add(o, "mode", ucv_string_new(c->mode ? "twophase" : "oneshot"));

	return o;
}

static const uc_function_list_t comm_methods[] = {
	{ "allreduce",	comm_allreduce },
	{ "info",	comm_info },
};

int exl3_moe_max_concurrency(int device);
void exl3_moe_gather
(
    at::Tensor output_state,
    const at::Tensor& output_scratch,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& expert_start,
    const at::Tensor& slot_base,
    const at::Tensor& slot_kind,
    const at::Tensor& weight_sorted
);

void exl3_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const float K_gate,
    const float K_up,
    const float K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active,
    const c10::optional<at::Tensor>& output_scratch,
    const c10::optional<at::Tensor>& fused_base,
    const int count_lo,
    const int count_hi,
    const int m_tile,
    const c10::optional<at::Tensor>& bits_g,
    const c10::optional<at::Tensor>& bits_u,
    const c10::optional<at::Tensor>& bits_d
);

/* torch.moe_coop(x, sel, rw, cfg, sh_out?)
 *
 * Fused decode-shaped MoE: two launches run every (token, expert) slot of
 * x (bsz <= 8 rows) through gate/up, the activation and down, weighted by
 * rw and summed per token into cfg.out (fp32). Routing stays on the device
 * (sel int64 (bsz, topk), rw fp16) — no host round trip, graph-capturable.
 * cfg is an object holding the static parts: tensors g_trellis g_suh g_svh
 * u_* d_* (int64 device pointer tables), had_g had_u gu_g gu_u act_out
 * d_out ctr out (scratch, see exl3_moe_coop_prepare) and numbers Hi Kg Ku
 * Kd act act_limit, booleans mcg mul1. sh_out: optional fp32 (bsz, H)
 * shared-expert output added into the sum. */
static uc_value_t *
torch_moe_coop(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *x, *sel, *rw;
	uc_value_t *cfg = (nargs >= 4) ? uc_fn_arg(3) : NULL;

	if (nargs < 4 || nargs > 5 || !(x = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(sel = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(rw = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	if (ucv_type(cfg) != UC_OBJECT)
		RAISE("moe_coop: cfg must be an object");

	std::map<std::string, at::Tensor *> t;
	static const char *names[] = { "g_trellis", "g_suh", "g_svh", "u_trellis",
		"u_suh", "u_svh", "d_trellis", "d_suh", "d_svh", "had_g", "had_u",
		"gu_g", "gu_u", "act_out", "d_out", "ctr", "out" };

	for (const char *n : names) {
		at::Tensor *tt = tensor_arg_opt(vm, ucv_object_get(cfg, n, NULL));

		if (!tt)
			RAISE((std::string("moe_coop: cfg.") + n + " must be a tensor").c_str());

		t[n] = tt;
	}

	auto num = [&](const char *k, double dflt) {
		uc_value_t *v = ucv_object_get(cfg, k, NULL);

		return (v && ucv_type(v) == UC_DOUBLE) ? ucv_double_get(v) :
		       (v && ucv_type(v) == UC_INTEGER) ? (double)ucv_int64_get(v) : dflt;
	};
	auto flag = [&](const char *k) {
		uc_value_t *v = ucv_object_get(cfg, k, NULL);

		return v && ucv_type(v) == UC_BOOLEAN && ucv_boolean_get(v);
	};

	c10::optional<at::Tensor> sh;

	if (nargs == 5 && ucv_type(uc_fn_arg(4)) != UC_NULL) {
		at::Tensor *s = tensor_arg(vm, uc_fn_arg(4));

		if (!s)
			return NULL;

		sh = *s;
	}

	/* mixed bit widths (DeepSeek-V4.1 packs): uint8 tables g_bits / u_bits / d_bits, one entry per expert */
	c10::optional<at::Tensor> bg, bu, bd;
	{
		at::Tensor *tg = tensor_arg_opt(vm, ucv_object_get(cfg, "g_bits", NULL));
		at::Tensor *tu = tensor_arg_opt(vm, ucv_object_get(cfg, "u_bits", NULL));
		at::Tensor *td = tensor_arg_opt(vm, ucv_object_get(cfg, "d_bits", NULL));

		if (tg) bg = *tg;
		if (tu) bu = *tu;
		if (td) bd = *td;
	}

	try {
		exl3_moe_coop(*x, *sel, *rw, (int)num("min_expert", -1), (int)num("max_expert", -1),
			(int)num("Hi", 0),
			*t["g_trellis"], *t["g_suh"], *t["g_svh"],
			*t["u_trellis"], *t["u_suh"], *t["u_svh"],
			*t["d_trellis"], *t["d_suh"], *t["d_svh"],
			c10::nullopt, c10::nullopt, c10::nullopt,
			(float)num("Kg", 0), (float)num("Ku", 0), (float)num("Kd", 0),
			flag("mcg"), flag("mul1"), (int)num("act", 0),
			(float)num("act_limit", 0), true,
			*t["had_g"], *t["had_u"], *t["gu_g"], *t["gu_u"],
			*t["act_out"], *t["d_out"], *t["ctr"], *t["out"],
			sh, c10::nullopt, bg, bu, bd);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

extern "C" int amp_fp8_gemv(const void *x, const void *w, const void *s, void *out, int M, int K,
                            int N, cudaStream_t stream);
extern "C" int amp_fp8_gemv_grouped(const void *x, const void *w, const void *s, void *out, int M, int K, int N,
                                    int G, cudaStream_t stream);

extern "C" int amp_fp8_dequant(const void *w, const void *s, void *out, int N, int K,
                               cudaStream_t stream);

static bool
fp8_args_ok(at::Tensor *w, at::Tensor *s)
{
	return w->dim() == 2 && s->dim() == 2 && w->scalar_type() == at::kFloat8_e4m3fn &&
	       s->scalar_type() == at::kFloat8_e8m0fnu && w->is_contiguous() && s->is_contiguous() &&
	       w->size(0) % 32 == 0 && w->size(1) % 32 == 0 &&
	       s->size(0) == w->size(0) / 32 && s->size(1) == w->size(1) / 32 &&
	       w->get_device() == s->get_device();
}

/* torch.fp8_gemv(x, w, s): x (M<=8, K) fp16, w (N, K) float8_e4m3fn, s (N/32, K/32)
 * float8_e8m0fnu -> (M, N) fp16 (x @ w^T with the block scales applied). */
static uc_value_t *
torch_fp8_gemv(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *x, *w, *s;

	if (nargs != 3 || !(x = tensor_arg(vm, uc_fn_arg(0))) || !(w = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(s = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	if (!fp8_args_ok(w, s) || x->dim() != 2 || x->scalar_type() != at::kHalf || !x->is_contiguous() ||
	    x->size(1) != w->size(1) || x->size(0) < 1 || x->size(0) > 8 || x->get_device() != w->get_device() ||
	    w->size(1) % 32 != 0)
		RAISE("fp8_gemv: x (M<=8, K) fp16, w (N, K) e4m3, s (N/32, K/32) e8m0, one device");

	TensorDeviceGuard dg(*x);

	try {
		at::Tensor out = at::empty({ x->size(0), w->size(0) }, x->options());
		int rc = amp_fp8_gemv(x->data_ptr(), w->data_ptr(), s->data_ptr(), out.data_ptr(),
			(int)x->size(0), (int)w->size(1), (int)w->size(0),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "fp8_gemv"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* torch.fp8_gemv_grouped(x, w, s, G): x (M<=8, G * K) fp16; w (G * N, K) e4m3, s (G * N / 32, K / 32) e8m0 -> (M, G * N)
 * fp16, group g using columns [g K, (g + 1) K) of x and rows [g N, (g + 1) N) of w. */
static uc_value_t *
torch_fp8_gemv_grouped(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *x, *w, *s;

	if (nargs != 4 || !(x = tensor_arg(vm, uc_fn_arg(0))) || !(w = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(s = tensor_arg(vm, uc_fn_arg(2))) || ucv_type(uc_fn_arg(3)) != UC_INTEGER)
		RAISE("fp8_gemv_grouped: (x, w, s, G)");

	long G = ucv_int64_get(uc_fn_arg(3));

	if (!fp8_args_ok(w, s) || G < 1 || w->size(0) % G || (w->size(0) / G) % 32 || x->dim() != 2 ||
	    x->scalar_type() != at::kHalf || !x->is_contiguous() || x->size(1) != G * w->size(1) ||
	    x->size(0) < 1 || x->size(0) > 8 || x->get_device() != w->get_device())
		RAISE("fp8_gemv_grouped: x (M<=8, G*K) fp16, w (G*N, K) e4m3, s (G*N/32, K/32) e8m0, N % 32 == 0");

	TensorDeviceGuard dg(*x);

	try {
		long N = w->size(0) / G;
		at::Tensor out = at::empty({ x->size(0), G * N }, x->options());
		int rc = amp_fp8_gemv_grouped(x->data_ptr(), w->data_ptr(), s->data_ptr(), out.data_ptr(),
			(int)x->size(0), (int)w->size(1), (int)N, (int)G, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "fp8_gemv_grouped"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* torch.fp8_dequant(w, s, out?): the fp16 (N, K) matrix of a block-scaled fp8 weight */
static uc_value_t *
torch_fp8_dequant(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *w, *s, *o = NULL;

	if (nargs < 2 || nargs > 3 || !(w = tensor_arg(vm, uc_fn_arg(0))) || !(s = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	if (nargs == 3 && !(o = tensor_arg_opt(vm, uc_fn_arg(2))) && ucv_type(uc_fn_arg(2)) != UC_NULL)
		return NULL;

	if (!fp8_args_ok(w, s) || (w->size(1) % 32) != 0)
		RAISE("fp8_dequant: w (N, K) e4m3, s (N/32, K/32) e8m0, contiguous, one device");

	if (o && (o->scalar_type() != at::kHalf || !o->is_contiguous() || o->numel() < w->numel() ||
	          o->get_device() != w->get_device()))
		RAISE("fp8_dequant: out must be a contiguous fp16 tensor of at least N*K elements");

	TensorDeviceGuard dg(*w);

	try {
		at::Tensor out = o ? o->view({ -1 }).narrow(0, 0, w->numel()).view({ w->size(0), w->size(1) }) :
			at::empty({ w->size(0), w->size(1) }, w->options().dtype(at::kHalf));
		int rc = amp_fp8_dequant(w->data_ptr(), s->data_ptr(), out.data_ptr(), (int)w->size(0),
			(int)w->size(1), c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "fp8_dequant"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" int amp_v41_rope(const void *x, const void *pos, int pos_f32, long pstride, const float *inv_freq,
                            void *out, int T, int H, int D, int rd, cudaStream_t stream);

/* torch.v41_rope(x, pos, inv_freq): x (T, H, D) fp16 contiguous; pos int64 or fp32 tensor of at least T elements
 * (row t reads element t * stride(0)); inv_freq (rd / 2) fp32 -> (T, H, D) fp16 with the trailing rd = 2 * len(inv_freq)
 * channels of every row rotated pairwise by pos * inv_freq. */
static uc_value_t *
torch_v41_rope(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *x, *pos, *inv;

	if (nargs != 3 || !(x = tensor_arg(vm, uc_fn_arg(0))) || !(pos = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(inv = tensor_arg(vm, uc_fn_arg(2))))
		return NULL;

	bool pf = pos->scalar_type() == at::kFloat;

	if (x->dim() != 3 || x->scalar_type() != at::kHalf || !x->is_contiguous() || !x->is_cuda() ||
	    (!pf && pos->scalar_type() != at::kLong) || pos->numel() < 1 || pos->device() != x->device() ||
	    inv->scalar_type() != at::kFloat || !inv->is_contiguous() || inv->device() != x->device() ||
	    inv->numel() * 2 > x->size(2) || x->size(2) % 2)
		RAISE("v41_rope: x (T, H, D) fp16, pos int64 / fp32 (one per row), inv_freq fp32, one CUDA device");

	long ps = pos->numel() > 1 ? pos->stride(0) : 0;

	if (pos->numel() < x->size(0) && pos->numel() != 1)
		RAISE("v41_rope: pos has fewer elements than x has rows");

	if (pos->numel() == 1 && x->size(0) != 1)
		RAISE("v41_rope: pos has fewer elements than x has rows");

	TensorDeviceGuard dg(*x);

	try {
		at::Tensor out = at::empty_like(*x);
		int rc = amp_v41_rope(x->data_ptr(), pos->data_ptr(), pf, ps, (const float *)inv->data_ptr(), out.data_ptr(),
			(int)x->size(0), (int)x->size(1), (int)x->size(2), (int)(inv->numel() * 2),
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "v41_rope"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" int amp_v41_route(const float *logits, const float *bias, long long *idx, float *w, int T, int E, int k,
                             float scale, cudaStream_t stream);

/* torch.v41_route(logits, bias, k, scale): logits (T, E) fp32, bias (E) or (1, E) fp32 -> [idx (T, k) int64,
 * w (T, k) fp32]: the sqrt(softplus) router of DeepSeek-V4 / V4.1 (top-k on affinity + bias, weights = the plain
 * affinities normalized to sum 1, times scale). */
static uc_value_t *
torch_v41_route(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *lg, *bias;

	/* bias null: the plain softmax router (weights = softmax of the k largest logits, times scale) */
	bool nobias = (nargs == 4 && uc_fn_arg(1) == NULL);

	bias = NULL;

	if (nargs != 4 || !(lg = tensor_arg(vm, uc_fn_arg(0))) || (!nobias && !(bias = tensor_arg(vm, uc_fn_arg(1)))) ||
	    ucv_type(uc_fn_arg(2)) != UC_INTEGER ||
	    (ucv_type(uc_fn_arg(3)) != UC_DOUBLE && ucv_type(uc_fn_arg(3)) != UC_INTEGER))
		RAISE("v41_route: (logits, bias | null, k, scale)");

	long k = ucv_int64_get(uc_fn_arg(2));
	double scale = (ucv_type(uc_fn_arg(3)) == UC_DOUBLE) ? ucv_double_get(uc_fn_arg(3)) : (double)ucv_int64_get(uc_fn_arg(3));

	if (lg->dim() != 2 || lg->scalar_type() != at::kFloat || !lg->is_contiguous() || !lg->is_cuda() ||
	    (bias && (bias->scalar_type() != at::kFloat || !bias->is_contiguous() || bias->numel() != lg->size(1) ||
	    bias->device() != lg->device())) || lg->size(1) > 512 || k < 1 || k > lg->size(1) || k > 32)
		RAISE("v41_route: logits (T, E<=512) fp32 contiguous CUDA, bias (E) fp32, 1 <= k <= 32");

	TensorDeviceGuard dg(*lg);

	try {
		at::Tensor idx = at::empty({ lg->size(0), k }, lg->options().dtype(at::kLong));
		at::Tensor w = at::empty({ lg->size(0), k }, lg->options());
		int rc = amp_v41_route((const float *)lg->data_ptr(), bias ? (const float *)bias->data_ptr() : NULL,
			(long long *)idx.data_ptr(), (float *)w.data_ptr(), (int)lg->size(0), (int)lg->size(1), (int)k,
			(float)scale, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "v41_route"))
			return NULL;

		uc_value_t *r = ucv_array_new(vm);

		ucv_array_push(r, tensor_wrap(vm, std::move(idx), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(w), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" int amp_moe_prep(const long long *idx, const float *w, int A, int K, int E, long long e0, long long *fe,
                            long long *order, long long *tok, void *wts, long long *count, long long *start,
                            long long *inv, long long *kind, cudaStream_t stream);

/* torch.moe_prep(idx, w, E, e0): idx (T, K) int64 global expert ids, w (T, K) fp32 -> [fe (A) int64, order (A) int64,
 * tok (A) int64, wts (A) fp16, count (E + 1) int64, start (E + 1) int64, inv (A) int64, kind (E) int64]: the token-sorted
 * dispatch of the experts [e0, e0 + E) (see amp_moe_prep_kernel). */
static uc_value_t *
torch_moe_prep(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *idx, *w;

	if (nargs != 4 || !(idx = tensor_arg(vm, uc_fn_arg(0))) || !(w = tensor_arg(vm, uc_fn_arg(1))) ||
	    ucv_type(uc_fn_arg(2)) != UC_INTEGER || ucv_type(uc_fn_arg(3)) != UC_INTEGER)
		RAISE("moe_prep: (idx, w, E, e0)");

	long E = ucv_int64_get(uc_fn_arg(2)), e0 = ucv_int64_get(uc_fn_arg(3));

	if (idx->dim() != 2 || idx->scalar_type() != at::kLong || !idx->is_contiguous() || !idx->is_cuda() ||
	    w->scalar_type() != at::kFloat || !w->is_contiguous() || w->sizes() != idx->sizes() ||
	    w->device() != idx->device() || E < 1 || E > 4096)
		RAISE("moe_prep: idx (T, K) int64 and w (T, K) fp32, contiguous on one CUDA device, 1 <= E <= 4096");

	TensorDeviceGuard dg(*idx);

	try {
		long A = idx->numel(), K = idx->size(1);
		auto lo = idx->options(), fo = idx->options().dtype(at::kHalf);
		at::Tensor fe = at::empty({ A }, lo), order = at::empty({ A }, lo), tok = at::empty({ A }, lo);
		at::Tensor wts = at::empty({ A }, fo), count = at::empty({ E + 1 }, lo), start = at::empty({ E + 1 }, lo);
		at::Tensor inv = at::empty({ A }, lo), kind = at::empty({ E }, lo);
		int rc = amp_moe_prep((const long long *)idx->data_ptr(), (const float *)w->data_ptr(), (int)A, (int)K, (int)E, e0,
			(long long *)fe.data_ptr(), (long long *)order.data_ptr(), (long long *)tok.data_ptr(), wts.data_ptr(),
			(long long *)count.data_ptr(), (long long *)start.data_ptr(), (long long *)inv.data_ptr(),
			(long long *)kind.data_ptr(), c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "moe_prep"))
			return NULL;

		uc_value_t *r = ucv_array_new(vm);

		for (at::Tensor *t : { &fe, &order, &tok, &wts, &count, &start, &inv, &kind })
			ucv_array_push(r, tensor_wrap(vm, std::move(*t), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" int amp_fp4_dequant(const void *w, const void *s, void *out, int N, int K, cudaStream_t stream);
extern "C" int amp_fp4_moe_gemv(const void *x, const void *ids, const void *wp, const void *sp, void *out,
                                int slots, int K, int N, int xdiv, cudaStream_t stream);

/* torch.fp4_dequant(w, s): w (N, K/2) int8 (two e2m1 per byte, low nibble first), s (N, K/32) float8_e8m0fnu
 * -> (N, K) fp16 */
static uc_value_t *
torch_fp4_dequant(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *w, *s;

	if (nargs != 2 || !(w = tensor_arg(vm, uc_fn_arg(0))) || !(s = tensor_arg(vm, uc_fn_arg(1))))
		return NULL;

	if (w->dim() != 2 || s->dim() != 2 || w->scalar_type() != at::kChar ||
	    s->scalar_type() != at::kFloat8_e8m0fnu || !w->is_contiguous() || !s->is_contiguous() ||
	    (w->size(1) % 16) != 0 || s->size(0) != w->size(0) || s->size(1) != w->size(1) / 16 ||
	    w->get_device() != s->get_device())
		RAISE("fp4_dequant: w (N, K/2) int8, s (N, K/32) e8m0, contiguous, one device");

	TensorDeviceGuard dg(*w);

	try {
		int N = (int)w->size(0), K = (int)w->size(1) * 2;
		at::Tensor out = at::empty({ N, K }, w->options().dtype(at::kHalf));
		int rc = amp_fp4_dequant(w->data_ptr(), s->data_ptr(), out.data_ptr(), N, K,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "fp4_dequant"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* torch.fp4_moe_gemv(x, ids, wp, sp, N, xdiv): x (R, K) fp16; ids (S) int64 expert per slot; wp / sp (E) int64
 * device addresses of each expert's (N, K/2) int8 weight and (N, K/32) e8m0 scale -> (S, N) fp32, slot s using
 * row s / xdiv of x. */
static uc_value_t *
torch_fp4_moe_gemv(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *x, *ids, *wp, *sp;

	if (nargs != 6 || !(x = tensor_arg(vm, uc_fn_arg(0))) || !(ids = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(wp = tensor_arg(vm, uc_fn_arg(2))) || !(sp = tensor_arg(vm, uc_fn_arg(3))) ||
	    ucv_type(uc_fn_arg(4)) != UC_INTEGER || ucv_type(uc_fn_arg(5)) != UC_INTEGER)
		RAISE("fp4_moe_gemv: (x, ids, wp, sp, N, xdiv)");

	long N = ucv_int64_get(uc_fn_arg(4)), xdiv = ucv_int64_get(uc_fn_arg(5));

	if (x->dim() != 2 || x->scalar_type() != at::kHalf || !x->is_contiguous() || ids->scalar_type() != at::kLong ||
	    !ids->is_contiguous() || wp->scalar_type() != at::kLong || sp->scalar_type() != at::kLong ||
	    N < 1 || xdiv < 1 || x->size(0) * xdiv < ids->numel() || (x->size(1) % 32) != 0 ||
	    x->get_device() != ids->get_device() || x->get_device() != wp->get_device() ||
	    x->get_device() != sp->get_device())
		RAISE("fp4_moe_gemv: x (R, K) fp16, ids / wp / sp int64 on the same device, xdiv * R >= slots");

	TensorDeviceGuard dg(*x);

	try {
		int S = (int)ids->numel(), K = (int)x->size(1);
		at::Tensor out = at::empty({ S, N }, x->options().dtype(at::kFloat));
		int rc = amp_fp4_moe_gemv(x->data_ptr(), ids->data_ptr(), wp->data_ptr(), sp->data_ptr(),
			out.data_ptr(), S, K, (int)N, (int)xdiv, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "fp4_moe_gemv"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" long amp_v41_attn_ws_floats(int T, int H, int nsplit);
extern "C" int amp_v41_attn(const void *q, const void *kvw, long wbase, int Nw, const void *kvc, int Nc,
                            const void *idx, int Kc, const float *sinks, const float *inv_freq, void *out,
                            float *ws, int T, int H, int W, int nsplit, float scale, long pos0,
                            const long long *pos_ptr, int ring, const void *kvq, const void *kvs,
                            cudaStream_t stream);

/* torch.v41_attn(q, kvw, wbase, kvc|null, idx|null, sinks, inv_freq, pos0, scale, W, nsplit?, pos_t?, ring?):
 * DeepSeek-V4.1 sparse MQA attention (see amp_kernels.cu). q (T, H, 512) fp16; kvw (Nw, 512) fp16
 * window rows, row r = absolute position wbase + r; kvc (Nc, 512) fp16 and idx (T, Kc) int32 the
 * compressed picks; sinks (H) fp32; inv_freq (32) fp32 (the rope slice of the output is rotated back
 * by pos0 + t). pos_t: optional one-element int64 device tensor holding pos0 (read by the kernels:
 * CUDA-graph decode); ring: when > 0, kvw is a ring of that many rows holding position a at row a % ring
 * (the window then needs no gathering). Returns (T, H, 512) fp16. */
static uc_value_t *
torch_v41_attn(uc_vm_t *vm, size_t nargs)
{
	if (nargs < 10 || nargs > 13) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, kvw, wbase, kvc, idx, sinks, inv_freq, pos0, scale, W, nsplit?, pos_t?, ring?)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *kvw = q ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	int64_t wbase, pos0, W, nsplit = 0;
	double scale;

	if (!kvw)
		return NULL;

	/* kvc: a (Nc, 512) fp16 tensor, or [codes (Nc, 512) int8, scales (Nc, 16) fp16] (a Q8 pool, one scale per 32) */
	at::Tensor *kvc = NULL, *kvq = NULL, *kvs = NULL;
	uc_value_t *kva = uc_fn_arg(3);

	if (kva && ucv_type(kva) == UC_ARRAY) {
		if (ucv_array_length(kva) != 2 || !(kvq = tensor_arg(vm, ucv_array_get(kva, 0))) ||
		    !(kvs = tensor_arg(vm, ucv_array_get(kva, 1))))
			return NULL;
	}
	else {
		kvc = tensor_arg_opt(vm, kva);
	}

	at::Tensor *idx = tensor_arg_opt(vm, uc_fn_arg(4));
	at::Tensor *sinks = tensor_arg(vm, uc_fn_arg(5));
	at::Tensor *fr = sinks ? tensor_arg(vm, uc_fn_arg(6)) : NULL;

	if (!fr || !arg_int(vm, uc_fn_arg(2), &wbase) || !arg_int(vm, uc_fn_arg(7), &pos0) ||
	    !arg_number(vm, uc_fn_arg(8), &scale) || !arg_int(vm, uc_fn_arg(9), &W) ||
	    (nargs >= 11 && !arg_int(vm, uc_fn_arg(10), &nsplit)))
		return NULL;

	at::Tensor *pos_t = (nargs >= 12) ? tensor_arg_opt(vm, uc_fn_arg(11)) : NULL;
	int64_t ring = 0;

	if (nargs >= 13 && !arg_int(vm, uc_fn_arg(12), &ring))
		return NULL;

	if (pos_t && (pos_t->scalar_type() != at::kLong || pos_t->numel() != 1 || !pos_t->is_cuda() ||
	              pos_t->get_device() != q->get_device()))
		RAISE("v41_attn: pos_t must be a one-element int64 tensor on q's device");

	if (ring < 0 || (ring > 0 && ring < W))
		RAISE("v41_attn: ring must be 0 or at least the window");

	if (q->dim() != 3 || q->size(2) != 512 || q->scalar_type() != at::kHalf || !q->is_contiguous() ||
	    kvw->dim() != 2 || kvw->size(1) != 512 || kvw->scalar_type() != at::kHalf || !kvw->is_contiguous() ||
	    sinks->scalar_type() != at::kFloat || sinks->numel() != q->size(1) || !sinks->is_contiguous() ||
	    fr->scalar_type() != at::kFloat || fr->numel() != 32 || !fr->is_contiguous() || ((!kvc && !kvq) != !idx) ||
	    (kvc && (kvc->dim() != 2 || kvc->size(1) != 512 || kvc->scalar_type() != at::kHalf ||
	             !kvc->is_contiguous())) ||
	    (kvq && (kvq->dim() != 2 || kvq->size(1) != 512 || kvq->scalar_type() != at::kChar || !kvq->is_contiguous() ||
	             kvs->dim() != 2 || kvs->size(0) != kvq->size(0) || kvs->size(1) != 16 ||
	             kvs->scalar_type() != at::kHalf || !kvs->is_contiguous() || kvs->device() != kvq->device())) ||
	    (idx && (idx->dim() != 2 || idx->size(0) != q->size(0) || idx->scalar_type() != at::kInt ||
	             !idx->is_contiguous())) ||
	    q->size(1) % 8 || W < 1)
		RAISE("v41_attn: q (T,H,512) fp16, kvw (Nw,512) fp16, kvc (Nc,512) fp16 + idx (T,Kc) int32, sinks (H) fp32, inv_freq (32) fp32");

	int T = (int)q->size(0), H = (int)q->size(1), Kc = idx ? (int)idx->size(1) : 0;

	if (nsplit <= 0)
		nsplit = (T >= 64) ? 1 : (T >= 16) ? 2 : (T >= 4) ? 4 : 8;

	int total = (int)W + Kc;

	if (nsplit > (total + 15) / 16)
		nsplit = (total + 15) / 16;

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty_like(*q);
		at::Tensor ws = at::empty({ amp_v41_attn_ws_floats(T, H, (int)nsplit) }, q->options().dtype(at::kFloat));
		int rc = amp_v41_attn(q->data_ptr(), kvw->data_ptr(), (long)wbase, (int)kvw->size(0),
			kvc ? kvc->data_ptr() : NULL, kvc ? (int)kvc->size(0) : kvq ? (int)kvq->size(0) : 0,
			idx ? idx->data_ptr() : NULL, Kc,
			(const float *)sinks->data_ptr(), (const float *)fr->data_ptr(), out.data_ptr(),
			(float *)ws.data_ptr(), T, H, (int)W, (int)nsplit, (float)scale, (long)pos0,
			pos_t ? (const long long *)pos_t->data_ptr() : NULL, (int)ring,
			kvq ? kvq->data_ptr() : NULL, kvs ? kvs->data_ptr() : NULL,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "v41_attn"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" int
amp_skinny_f32(const float *x, const float *w, float *out, float *part, int *cnt,
              int M, int K, int N, long part_floats, int cnt_len, cudaStream_t stream);

/* torch.skinny(x, w): x (M, K) fp32, w (K, N) fp32 -> (M, N) fp32 for
 * M <= 8 (decode / verify rows); deterministic split-K, graph-capturable.
 * Replaces torch.matmul where cuBLAS falls back to a slow generic GEMM
 * for 2..8 rows. */
static uc_value_t *
torch_skinny(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 2) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (x, w)");
		return NULL;
	}

	at::Tensor *x = f32_arg(vm, uc_fn_arg(0), "skinny: x");
	at::Tensor *w = x ? f32_arg(vm, uc_fn_arg(1), "skinny: w") : NULL;

	if (!w)
		return NULL;

	if (x->dim() != 2 || w->dim() != 2 || x->size(1) != w->size(0) ||
	    x->size(0) < 1 || x->size(0) > 8 || !x->is_contiguous() ||
	    !w->is_contiguous() || x->get_device() != w->get_device())
		RAISE("skinny: x (M<=8, K) and w (K, N) must be contiguous fp32 on one device");

	int M = (int)x->size(0), K = (int)x->size(1), N = (int)w->size(1);

	TensorDeviceGuard dg(*x);

	try {
		/* per-device split-K workspace + tile counters (persist: graphs
		 * replay with these addresses; the reducer re-zeroes the counters) */
		static std::vector<at::Tensor> part_ws(64), cnt_ws(64);
		int dev = x->get_device();
		const long part_n = 32L * 1024 * 1024 / 4;     // 32 MiB
		const int cnt_n = 4096;

		if (!part_ws[dev].defined()) {
			part_ws[dev] = at::zeros({ part_n }, x->options());
			cnt_ws[dev] = at::zeros({ cnt_n }, x->options().dtype(at::kInt));
			at::cuda::device_synchronize();
		}

		at::Tensor out = at::empty({ M, N }, x->options());
		int rc = amp_skinny_f32((const float *)x->data_ptr(), (const float *)w->data_ptr(),
			(float *)out.data_ptr(), (float *)part_ws[dev].data_ptr(),
			(int *)cnt_ws[dev].data_ptr(), M, K, N, part_n, cnt_n,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "skinny"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* torch.moe_concurrency(dev): expert groups the fused MoE kernel runs at once. */
static uc_value_t *
torch_moe_concurrency(uc_vm_t *vm, size_t nargs)
{
	int dev = (nargs >= 1 && ucv_type(uc_fn_arg(0)) == UC_INTEGER) ?
		(int)ucv_int64_get(uc_fn_arg(0)) : 0;

	try {
		return ucv_int64_new(exl3_moe_max_concurrency(dev));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/* torch.moe_fused(y, out, count, tok_sorted, w_sorted, cfg, num_active, lo, hi, m_tile
 *                 [, scratch, slot_base])
 *
 * Token-sorted fused MoE (exl3_moe): experts with lo <= rows <= hi run
 * gate/up/act/down on their gathered rows and atomically accumulate
 * weight * result into out (fp32, zero-initialised, same shape as y) — in
 * arrival order, so the sum is not bit-reproducible run to run. With scratch
 * (fp32 [slots, hidden]) and slot_base (int64 per expert) each assignment
 * owns a slot instead and torch.moe_gather sums them in a fixed order.
 * cfg: pointer tables g_/u_/d_ {trellis,suh,svh}, temps ts_g ts_u ti_g ti_u,
 * numbers Kg Ku Kd act act_limit, booleans mcg mul1. */
static uc_value_t *
torch_moe_fused(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *y, *out, *cnt, *tok, *w;
	uc_value_t *cfg = (nargs >= 6) ? uc_fn_arg(5) : NULL;

	if ((nargs != 10 && nargs != 12) || !(y = tensor_arg(vm, uc_fn_arg(0))) ||
	    !(out = tensor_arg(vm, uc_fn_arg(1))) ||
	    !(cnt = tensor_arg(vm, uc_fn_arg(2))) ||
	    !(tok = tensor_arg(vm, uc_fn_arg(3))) ||
	    !(w = tensor_arg(vm, uc_fn_arg(4))))
		return NULL;

	at::Tensor *scr = NULL, *sbase = NULL;

	if (nargs == 12 && (!(scr = tensor_arg(vm, uc_fn_arg(10))) || !(sbase = tensor_arg(vm, uc_fn_arg(11)))))
		return NULL;

	if (ucv_type(cfg) != UC_OBJECT)
		RAISE("moe_fused: cfg must be an object");

	std::map<std::string, at::Tensor *> t;
	static const char *names[] = { "g_trellis", "g_suh", "g_svh", "u_trellis",
		"u_suh", "u_svh", "d_trellis", "d_suh", "d_svh", "ts_g", "ts_u",
		"ti_g", "ti_u" };

	for (const char *n : names) {
		at::Tensor *tt = tensor_arg_opt(vm, ucv_object_get(cfg, n, NULL));

		if (!tt)
			RAISE((std::string("moe_fused: cfg.") + n + " must be a tensor").c_str());

		t[n] = tt;
	}

	auto num = [&](const char *k) {
		uc_value_t *v = ucv_object_get(cfg, k, NULL);

		return (v && ucv_type(v) == UC_DOUBLE) ? ucv_double_get(v) :
		       (v && ucv_type(v) == UC_INTEGER) ? (double)ucv_int64_get(v) : 0.0;
	};
	auto flag = [&](const char *k) {
		uc_value_t *v = ucv_object_get(cfg, k, NULL);

		return v && ucv_type(v) == UC_BOOLEAN && ucv_boolean_get(v);
	};
	auto arg_i = [&](size_t i) {
		uc_value_t *v = uc_fn_arg(i);

		return (v && ucv_type(v) == UC_INTEGER) ? (int)ucv_int64_get(v) : 0;
	};

	/* mixed bit widths: uint8 tables g_bits / u_bits / d_bits (one entry per expert) */
	c10::optional<at::Tensor> fbg, fbu, fbd;
	{
		at::Tensor *tg = tensor_arg_opt(vm, ucv_object_get(cfg, "g_bits", NULL));
		at::Tensor *tu = tensor_arg_opt(vm, ucv_object_get(cfg, "u_bits", NULL));
		at::Tensor *td = tensor_arg_opt(vm, ucv_object_get(cfg, "d_bits", NULL));

		if (tg) fbg = *tg;
		if (tu) fbu = *tu;
		if (td) fbd = *td;
	}

	try {
		exl3_moe(*y, *out, *cnt, *tok, *w,
			*t["ts_g"], *t["ts_u"], *t["ti_g"], *t["ti_u"],
			(int)num("act"), (float)num("Kg"), (float)num("Ku"), (float)num("Kd"),
			*t["g_trellis"], *t["g_suh"], *t["g_svh"],
			*t["u_trellis"], *t["u_suh"], *t["u_svh"],
			*t["d_trellis"], *t["d_suh"], *t["d_svh"],
			flag("mcg"), flag("mul1"), flag("mcg"), flag("mul1"), flag("mcg"), flag("mul1"),
			(float)num("act_limit"), arg_i(6),
			scr ? c10::optional<at::Tensor>(*scr) : c10::nullopt,
			sbase ? c10::optional<at::Tensor>(*sbase) : c10::nullopt,
			arg_i(7), arg_i(8), arg_i(9), fbg, fbu, fbd);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* torch.moe_gather(out, scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, w_sorted)
 *
 * The fixed-order reduction of torch.moe_fused's per-assignment slots: every token's top-k slots
 * are summed in k order and added to its row of out (fp32). flat_expert: (T * K) local expert
 * id per assignment; inv_order: position of each assignment in the expert-sorted list;
 * expert_start / slot_base / slot_kind: per-expert tables (kind 1 = slots hold weighted outputs). */
static uc_value_t *
torch_moe_gather(uc_vm_t *vm, size_t nargs)
{
	at::Tensor *t[8];

	if (nargs != 8)
		RAISE("moe_gather: expected 8 tensors");

	for (size_t i = 0; i < 8; i++)
		if (!(t[i] = tensor_arg(vm, uc_fn_arg(i))))
			return NULL;

	try {
		exl3_moe_gather(*t[0], *t[1], *t[2], *t[3], *t[4], *t[5], *t[6], *t[7]);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

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

struct fork_slot {
	bool init = false;
	std::optional<c10::cuda::CUDAStream> side, origin;
	int dev = 0;
	cudaEvent_t ev_start = NULL, ev_done = NULL;
};

static fork_slot fork_slots[AMP_FORK_MAXD][AMP_FORK_MAXK];

static fork_slot *
fork_slot_arg(uc_vm_t *vm, size_t nargs)
{
	int64_t dev, k;

	if (nargs != 2 || !arg_int(vm, uc_fn_arg(0), &dev) || !arg_int(vm, uc_fn_arg(1), &k) ||
	    dev < 0 || dev >= AMP_FORK_MAXD || dev >= at::cuda::device_count() ||
	    k < 0 || k >= AMP_FORK_MAXK) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (device index, branch slot 0..3)");
		return NULL;
	}

	fork_slot *f = &fork_slots[dev][k];

	if (!f->init) {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)dev);

		/* slots 2..3: communication (the double-buffered prefill's
		 * all-reduces) — high priority, so the reduction kernel gets the
		 * next free SM slots ahead of the compute stream's pending blocks */
		f->side = c10::cuda::getStreamFromPool(k >= 2, (c10::DeviceIndex)dev);
		f->dev = (int)dev;
		cudaEventCreateWithFlags(&f->ev_start, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&f->ev_done, cudaEventDisableTiming);
		f->init = true;
	}

	return f;
}

static uc_value_t *
torch_stream_side(uc_vm_t *vm, size_t nargs)
{
	fork_slot *f = fork_slot_arg(vm, nargs);

	if (!f)
		return NULL;

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)f->dev);

		f->origin = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)f->dev);
		cudaEventRecord(f->ev_start, f->origin->stream());
		cudaStreamWaitEvent(f->side->stream(), f->ev_start, 0);
		c10::cuda::setCurrentCUDAStream(*f->side);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

static uc_value_t *
torch_stream_main(uc_vm_t *vm, size_t nargs)
{
	fork_slot *f = fork_slot_arg(vm, nargs);

	if (!f)
		return NULL;

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)f->dev);

		cudaEventRecord(f->ev_done, f->side->stream());
		c10::cuda::setCurrentCUDAStream(*f->origin);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

static uc_value_t *
torch_stream_wait(uc_vm_t *vm, size_t nargs)
{
	fork_slot *f = fork_slot_arg(vm, nargs);

	if (!f)
		return NULL;

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)f->dev);

		cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)f->dev).stream(),
			f->ev_done, 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* --- expert exchange (expert_dyn.uc) --------------------------------------- */

extern "C" int amp_poke(void *const *p, const long long *v, const int *sz, int n, cudaStream_t stream);

/* the byte-for-byte copy pairs of one direction: equal sizes, contiguous, the host side pinned */
static bool
exchange_pairs(uc_vm_t *vm, uc_value_t *dv, uc_value_t *sv, bool h2d, int dev, std::vector<at::Tensor> &dst,
               std::vector<at::Tensor> &src)
{
	if (!tensor_list_arg(vm, dv, dst, "expert_exchange") || !tensor_list_arg(vm, sv, src, "expert_exchange"))
		return false;

	if (dst.size() != src.size()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expert_exchange: %zu destinations for %zu sources", dst.size(),
			src.size());
		return false;
	}

	for (size_t i = 0; i < dst.size(); i++) {
		at::Tensor &d = h2d ? dst[i] : src[i], &h = h2d ? src[i] : dst[i];

		if (!d.is_cuda() || d.get_device() != dev || !h.is_pinned() || !d.is_contiguous() || !h.is_contiguous() ||
		    d.nbytes() != h.nbytes()) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"expert_exchange: copy %zu: a contiguous tensor on device %d and a pinned one of the same byte size", i,
				dev);
			return false;
		}
	}

	return true;
}

/*
 * expert_exchange(dev, slot, up_dst, up_src, down_dst, down_src, d2h_dst, d2h_src, pokes): null
 *
 * One exchange of dynamic expert residency, queued on the current stream of `dev` as expert_dyn.uc did op by op:
 * upload i (pinned up_src[i] -> up_dst[i] on the device) on the current stream, and behind each upload the download
 * i (device down_src[i] -> pinned down_dst[i]) on side stream `slot` (the link is full duplex; download i may
 * overwrite the bytes upload i has read); the current stream then waits for the side stream, copies d2h_src[i] ->
 * pinned d2h_dst[i], and writes `pokes` ([[address, value, 4 or 8], ...], at most 32: pointer-table entries, map
 * and mask words) in one kernel launch. Tensors must stay referenced until the stream has run the copies.
 */
static uc_value_t *
torch_expert_exchange(uc_vm_t *vm, size_t nargs)
{
	int64_t dev, slot;
	std::vector<at::Tensor> ud, us, dd, ds, xd, xs;
	std::vector<void *> pp;
	std::vector<long long> pv;
	std::vector<int> ps;

	if (nargs != 9 || !arg_int(vm, uc_fn_arg(0), &dev) || !arg_int(vm, uc_fn_arg(1), &slot) || dev < 0 ||
	    dev >= AMP_FORK_MAXD || dev >= at::cuda::device_count() || slot < 0 || slot >= AMP_FORK_MAXK) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (device, slot 0..3, up_dst, up_src, down_dst, down_src, d2h_dst, d2h_src, pokes)");
		return NULL;
	}

	if (!exchange_pairs(vm, uc_fn_arg(2), uc_fn_arg(3), true, (int)dev, ud, us) ||
	    !exchange_pairs(vm, uc_fn_arg(4), uc_fn_arg(5), false, (int)dev, dd, ds) ||
	    !exchange_pairs(vm, uc_fn_arg(6), uc_fn_arg(7), false, (int)dev, xd, xs))
		return NULL;

	if (dd.size() > ud.size()) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expert_exchange: more downloads than uploads");
		return NULL;
	}

	uc_value_t *pk = uc_fn_arg(8);

	if (ucv_type(pk) != UC_ARRAY || ucv_array_length(pk) > 32) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expert_exchange: pokes: expected an array of at most 32");
		return NULL;
	}

	for (size_t i = 0; i < ucv_array_length(pk); i++) {
		uc_value_t *e = ucv_array_get(pk, i);
		int64_t a, v, s;

		if (ucv_type(e) != UC_ARRAY || ucv_array_length(e) != 3 || !arg_int(vm, ucv_array_get(e, 0), &a) ||
		    !arg_int(vm, ucv_array_get(e, 1), &v) || !arg_int(vm, ucv_array_get(e, 2), &s) || !a ||
		    (s != 4 && s != 8) || a % s) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"expert_exchange: poke %zu: expected [aligned address, value, 4 or 8]", i);
			return NULL;
		}

		pp.push_back((void *)(uintptr_t)a);
		pv.push_back((long long)v);
		ps.push_back((int)s);
	}

	try {
		c10::cuda::CUDAGuard g((c10::DeviceIndex)dev);
		fork_slot *f = &fork_slots[dev][slot];

		// (the slot's side stream and events, created as stream_side would)
		if (!f->init) {
			f->side = c10::cuda::getStreamFromPool(slot >= 2, (c10::DeviceIndex)dev);
			f->dev = (int)dev;
			cudaEventCreateWithFlags(&f->ev_start, cudaEventDisableTiming);
			cudaEventCreateWithFlags(&f->ev_done, cudaEventDisableTiming);
			f->init = true;
		}

		cudaStream_t st = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)dev).stream();
		cudaStream_t side = f->side->stream();
		cudaError_t e = cudaSuccess;

		for (size_t i = 0; i < ud.size() && e == cudaSuccess; i++) {
			e = cudaMemcpyAsync(ud[i].data_ptr(), us[i].data_ptr(), ud[i].nbytes(), cudaMemcpyHostToDevice, st);

			if (e == cudaSuccess && i < dd.size()) {
				// (an event re-recorded after the wait was queued: the wait keeps the record it saw)
				e = cudaEventRecord(f->ev_start, st);

				if (e == cudaSuccess)
					e = cudaStreamWaitEvent(side, f->ev_start, 0);

				if (e == cudaSuccess)
					e = cudaMemcpyAsync(dd[i].data_ptr(), ds[i].data_ptr(), dd[i].nbytes(), cudaMemcpyDeviceToHost,
						side);
			}
		}

		if (e == cudaSuccess && dd.size()) {
			e = cudaEventRecord(f->ev_done, side);

			if (e == cudaSuccess)
				e = cudaStreamWaitEvent(st, f->ev_done, 0);
		}

		for (size_t i = 0; i < xd.size() && e == cudaSuccess; i++)
			e = cudaMemcpyAsync(xd[i].data_ptr(), xs[i].data_ptr(), xd[i].nbytes(), cudaMemcpyDeviceToHost, st);

		if (e != cudaSuccess)
			RAISE(cudaGetErrorString(e));

		int rc = amp_poke(pp.data(), pv.data(), ps.data(), (int)pp.size(), st);

		if (rc)
			RAISEF("expert_exchange: poke: %s", cudaGetErrorString((cudaError_t)rc));
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* --- completion notifier (eventfd + cudaLaunchHostFunc) ------------------- */

/* The shared state behind a notifier: the eventfd and the completion
 * counter live in a shared_ptr-held core so that an arm() whose
 * callbacks are still pending keeps them alive even when the notifier
 * resource is garbage-collected in between — the fd closes only after
 * the last pending callback ran (on a CUDA driver thread, where closing
 * it is safe: no CUDA API is touched there).
 *
 * Callback concurrency: each arm() launches one callback per device
 * stream; every callback decrements `remaining` exactly once and the
 * unique callback observing the 1 -> 0 transition bumps `seq`, writes
 * the eventfd and frees the arm. `drained` is touched from the VM
 * thread only. */
struct notifier_core {
	int efd;
	std::atomic<uint64_t> seq;	/* arms completed (all devices) */
	uint64_t drained;		/* wake-ups consumed by drain() */

	notifier_core() : efd(-1), seq(0), drained(0) {}

	~notifier_core()
	{
		if (efd >= 0)
			close(efd);
	}
};

struct notifier_arm {
	std::shared_ptr<notifier_core> core;
	std::atomic<int> remaining;

	notifier_arm(std::shared_ptr<notifier_core> c, int n) :
		core(std::move(c)), remaining(n) {}
};

/* Runs on a CUDA driver thread: no CUDA calls, no VM access here. */
static void
amp_notifier_callback(void *v)
{
	auto *a = static_cast<notifier_arm *>(v);

	if (a->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
		a->core->seq.fetch_add(1, std::memory_order_release);

		uint64_t one = 1;
		ssize_t rc = write(a->core->efd, &one, sizeof(one));

		(void)rc;	/* counter saturation loses one wake-up:
				 * one is already pending anyway */

		delete a;	/* may drop the last core reference */
	}
}

struct notifier_state {
	std::shared_ptr<notifier_core> core;
};

static void
notifier_free(void *data)
{
	delete static_cast<notifier_state *>(data);
}

static notifier_state *
notifier_this(uc_vm_t *vm)
{
	auto *ns = static_cast<notifier_state *>(uc_fn_thisval(NOTIFIER_TYPE));

	if (!ns)
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"called on non-notifier value");

	return ns;
}

/*
 * notifier(): torch.notifier
 *
 * A wake-up source for the ucode event loop. arm(devices) enqueues a
 * host-function callback on each device's current stream; when a
 * stream reaches it (all previously enqueued work on that stream
 * done), the callback bumps the completion counter and — on the last
 * device — writes the eventfd, which uloop reports as readable.
 *
 *   let n = torch.notifier();
 *   uloop.handle(n, function() { n.drain(); ...continue... },
 *                uloop.ULOOP_READ);
 *   ...enqueue work...
 *   n.arm([0, 1]);   // one wake-up when BOTH streams drained
 *
 * The callback must drain() the eventfd (uloop is level-triggered).
 * A wake-up can also be consumed without uloop: done() reports whether
 * an armed completion fired since the last drain(), seq() returns the
 * raw completion counter.
 */
static uc_value_t *
torch_notifier(uc_vm_t *vm, size_t nargs)
{
	auto *ns = new (std::nothrow) notifier_state();

	if (!ns)
		return NULL;

	ns->core = std::make_shared<notifier_core>();
	ns->core->efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);

	if (ns->core->efd < 0) {
		int err = errno;

		delete ns;
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "eventfd: %s",
			strerror(err));
		return NULL;
	}

	return uc_resource_new(ucv_resource_type_lookup(vm, NOTIFIER_TYPE), ns);
}

/*
 * fileno(): this — the eventfd number. The name matches what
 * uloop.handle() probes for, so the notifier registers as a handle
 * source directly.
 */
static uc_value_t *
notifier_fileno(uc_vm_t *vm, size_t nargs)
{
	auto *ns = notifier_this(vm);

	if (!ns)
		return NULL;

	return ucv_int64_new(ns->core->efd);
}

/*
 * arm(devices?): this
 *
 * Enqueues one callback per given device on its CURRENT stream — the
 * wake-up fires when the work enqueued so far on every listed device
 * has completed. devices: a device index, a "cuda:N" string, an array
 * of those, or null/absent for the current device. Repeats in the
 * array count separately (each needs its own callback).
 *
 * Raises during an active graph capture (a captured host function
 * would re-run at every replay with stale callback state).
 */
static uc_value_t *
torch_notifier_arm(uc_vm_t *vm, size_t nargs)
{
	auto *ns = notifier_this(vm);
	uc_value_t *arg = (nargs >= 1) ? uc_fn_arg(0) : NULL;
	std::vector<at::Device> devs;

	if (!ns)
		return NULL;

	if (!arg || ucv_type(arg) == UC_NULL) {
		devs.emplace_back(at::kCUDA,
			(c10::DeviceIndex)c10::cuda::current_device());
	}
	else if (ucv_type(arg) == UC_ARRAY) {
		for (size_t i = 0; i < ucv_array_length(arg); i++) {
			at::Device dev(at::kCPU);

			if (!parse_target_device(vm, ucv_array_get(arg, i), &dev))
				return NULL;

			devs.push_back(dev);
		}
	}
	else {
		at::Device dev(at::kCPU);

		if (!parse_target_device(vm, arg, &dev))
			return NULL;

		devs.push_back(dev);
	}

	if (devs.empty())
		RAISE("arm: empty device list");

	/* validate everything (and reject capture) BEFORE allocating the
	 * arm: once the first callback is enqueued it owns its state */
	std::vector<cudaStream_t> streams;

	for (auto &dev : devs) {
		if (dev.type() != at::kCUDA || dev.index() < 0 ||
		    dev.index() >= at::cuda::device_count())
			RAISE("arm: not a CUDA device");

		cudaStream_t st = c10::cuda::getCurrentCUDAStream(
			(c10::DeviceIndex)dev.index()).stream();
		cudaStreamCaptureStatus cs;
		cudaError_t e = cudaStreamIsCapturing(st, &cs);

		if (e != cudaSuccess)
			RAISE(cudaGetErrorString(e));

		if (cs != cudaStreamCaptureStatusNone)
			RAISE("arm: cannot arm during graph capture");

		streams.push_back(st);
	}

	auto *arm = new notifier_arm(ns->core, (int)streams.size());

	for (size_t si = 0; si < streams.size(); si++) {
		/* the default (null) stream handle is relative to the CURRENT
		 * device: enqueue under the stream's own device or the callback
		 * lands on device 0's queue and fires early */
		c10::cuda::CUDAGuard sg((c10::DeviceIndex)devs[si].index());
		cudaStream_t st = streams[si];
		cudaError_t e = cudaLaunchHostFunc(st, amp_notifier_callback, arm);

		if (e != cudaSuccess) {
			/* callbacks already enqueued keep the arm (and the fd)
			 * alive but the countdown can no longer complete */
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
				"arm: cudaLaunchHostFunc failed: %s",
				cudaGetErrorString(e));
			return NULL;
		}
	}

	return ucv_get(_uc_fn_this_res(vm));
}

/*
 * seq(): int — number of arm() completions fired so far (monotonic).
 * A completion fires only once EVERY device listed in the arm() call
 * reached its callback.
 */
static uc_value_t *
notifier_seq(uc_vm_t *vm, size_t nargs)
{
	auto *ns = notifier_this(vm);

	if (!ns)
		return NULL;

	return ucv_int64_new((int64_t)ns->core->seq.load(std::memory_order_acquire));
}

/*
 * done(): bool — true when at least one armed completion has fired
 * since the last drain() (or since creation).
 */
static uc_value_t *
notifier_done(uc_vm_t *vm, size_t nargs)
{
	auto *ns = notifier_this(vm);

	if (!ns)
		return NULL;

	return ucv_boolean_new(ns->core->seq.load(std::memory_order_acquire) !=
		ns->core->drained);
}

/*
 * drain(): int — consumes the pending wake-ups (a non-blocking read of
 * the eventfd) and returns how many there were; 0 when none. Must be
 * called from the uloop callback: the handle is level-triggered and
 * keeps firing while the eventfd is readable.
 */
static uc_value_t *
notifier_drain(uc_vm_t *vm, size_t nargs)
{
	auto *ns = notifier_this(vm);
	uint64_t val = 0;
	ssize_t rc;

	if (!ns)
		return NULL;

	do {
		rc = read(ns->core->efd, &val, sizeof(val));
	} while (rc < 0 && errno == EINTR);

	if (rc < 0) {
		if (errno == EAGAIN)
			return ucv_int64_new(0);

		int err = errno;
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "drain: %s",
			strerror(err));
		return NULL;
	}

	ns->core->drained += val;

	return ucv_int64_new((int64_t)val);
}

static uc_value_t *
notifier_free_method(uc_vm_t *vm, size_t nargs)
{
	void **slot = ucv_resource_dataptr(_uc_fn_this_res(vm), NOTIFIER_TYPE);

	if (slot && *slot) {
		notifier_free(static_cast<notifier_state *>(*slot));
		*slot = NULL;
	}

	return NULL;
}

static const uc_function_list_t notifier_methods[] = {
	{ "fileno",	notifier_fileno },
	{ "arm",		torch_notifier_arm },
	{ "seq",		notifier_seq },
	{ "done",	notifier_done },
	{ "drain",	notifier_drain },
	{ "free",	notifier_free_method },
};

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

	/*
	 * Read-ahead: one pread() at a time keeps an NVMe drive at queue depth 1 (350 MB/s on the 2.50bpw DeepSeek pack
	 * against 790 MB/s for eight parallel readers; the cold start of a 160 GB load was 10 minutes), and
	 * posix_fadvise(WILLNEED) starts no I/O on that array. A few helper threads read the entries ahead of the consumer
	 * into the page cache, at most WINDOW bytes past it, so the preads below hit memory.
	 */
	std::atomic<size_t> consumed(0), ra_next(0);
	std::atomic<bool> stop_ra(false);
	std::vector<size_t> prefix(ents.size() + 1, 0);

	for (size_t j = 0; j < ents.size(); j++)
		prefix[j + 1] = prefix[j] + ents[j].nbytes;

	auto ra_work = [&]() {
		const size_t WINDOW = (size_t)3 << 30;
		std::vector<uint8_t> scratch((size_t)4 << 20);

		for (;;) {
			size_t k = ra_next.fetch_add(1);

			if (k >= ents.size())
				return;

			// wait until the consumer is within WINDOW of this entry
			while (!stop_ra.load() && prefix[k] > prefix[std::min(consumed.load(), ents.size())] + WINDOW)
				usleep(2000);

			if (stop_ra.load())
				return;

			size_t done = 0;

			while (done < ents[k].nbytes && !stop_ra.load()) {
				size_t len = std::min(scratch.size(), ents[k].nbytes - done);
				ssize_t got = pread(ents[k].f->fd, scratch.data(), len, (off_t)(ents[k].file_off + done));

				if (got <= 0)
					break;

				done += (size_t)got;
			}
		}
	};
	std::vector<std::thread> ra_th;

	if (ents.size() > 1)
		for (int t = 0; t < 4; t++)
			ra_th.emplace_back(ra_work);

	struct ra_join {
		std::atomic<bool> &stop; std::vector<std::thread> &t;
		~ra_join() { stop.store(true); for (auto &x : t) if (x.joinable()) x.join(); }
	} ra_guard{stop_ra, ra_th};

	try {
		c10::cuda::CUDAGuard dev_guard(dev);
		h2d_ring ring;
		std::string err;

		if (!ring.init(dev, &err))
			throw std::runtime_error(err);

		for (i = 0; i < n; i++) {
			src_ent &ent = ents[i];

			consumed.store(i);

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
 * read_into(items: array, threads?): null
 *
 * Reads safetensors tensors straight into CPU memory the caller owns (the pinned arenas of host-resident experts).
 * items is an array of [stfile, name, dst] with dst a contiguous CPU tensor of the stored byte size; the bytes are
 * read with pread() by `threads` workers (default 8: one reader does not keep an NVMe array busy), not through the
 * mmap page-fault path (L40). Raises when any read fails or a destination does not fit.
 */
static uc_value_t *
torch_read_into(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *items = uc_fn_arg(0);
	int nthreads = 8;

	if (nargs < 1 || nargs > 2 || ucv_type(items) != UC_ARRAY) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (array of [stfile, name, dst], threads?)");
		return NULL;
	}

	if (nargs == 2) {
		uc_value_t *tv = uc_fn_arg(1);

		if (ucv_type(tv) != UC_INTEGER || ucv_int64_get(tv) < 1 || ucv_int64_get(tv) > 64) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE, "threads: expected an integer 1..64");
			return NULL;
		}

		nthreads = (int)ucv_int64_get(tv);
	}

	struct job { int fd; size_t file_off; size_t nbytes; uint8_t *dst; };
	std::vector<job> jobs;
	size_t n = ucv_array_length(items), i;

	jobs.reserve(n);

	for (i = 0; i < n; i++) {
		uc_value_t *it = ucv_array_get(items, i);

		if (ucv_type(it) != UC_ARRAY || ucv_array_length(it) != 3) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE, "item %zu: expected [stfile, name, dst]",
				(unsigned long)i);
			return NULL;
		}

		uc_value_t *fv = ucv_array_get(it, 0), *nv = ucv_array_get(it, 1), *dv = ucv_array_get(it, 2);
		st_file_t *f = NULL;

		if (ucv_type(fv) == UC_RESOURCE && res_is_type(fv, STFILE_TYPE))
			f = static_cast<st_file_t *>(ucv_resource_data(fv, STFILE_TYPE));

		if (!f || ucv_type(nv) != UC_STRING) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE, "item %zu: expected [stfile, name, dst]",
				(unsigned long)i);
			return NULL;
		}

		at::Tensor *dst = tensor_arg(vm, dv);

		if (!dst)
			return NULL;

		const char *dtype;
		int64_t *shape;
		size_t ndim, offset, nbytes;
		int rc = st_lookup(f, ucv_string_get(nv), &dtype, &shape, &ndim, &offset, &nbytes);

		if (rc < 0) {
			uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "item %zu: no such (or malformed) tensor '%s'",
				(unsigned long)i, ucv_string_get(nv));
			return NULL;
		}

		free(shape);

		if (dst->is_cuda() || !dst->is_contiguous() ||
		    (size_t)dst->numel() * dst->element_size() != nbytes) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"item %zu: dst must be a contiguous CPU tensor of %zu bytes", (unsigned long)i, nbytes);
			return NULL;
		}

		jobs.push_back({f->fd, f->data_off + offset, nbytes, (uint8_t *)dst->data_ptr()});
	}

	std::atomic<size_t> next(0);
	std::atomic<bool> failed(false);

	try {
		auto work = [&]() {
			for (;;) {
				size_t k = next.fetch_add(1);

				if (k >= jobs.size() || failed.load())
					return;

				job &j = jobs[k];
				size_t done = 0;

				while (done < j.nbytes) {
					ssize_t got = pread(j.fd, j.dst + done, j.nbytes - done, (off_t)(j.file_off + done));

					if (got <= 0) {
						failed.store(true);
						return;
					}

					done += (size_t)got;
				}
			}
		};
		std::vector<std::thread> th;

		for (int t = 1; t < nthreads && (size_t)t < jobs.size(); t++)
			th.emplace_back(work);

		work();

		for (auto &t : th)
			t.join();
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	if (failed.load()) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "read_into: short read from safetensors file");
		return NULL;
	}

	return NULL;
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

void ngram_gather_cpu(int64_t fd, int64_t base_offset, int64_t row_bytes, const at::Tensor& uids,
                      int64_t uid_base, at::Tensor out);

/* async row gathers (stfile.gather_async): one worker thread, a ticket per job */
struct gather_job {
	int fd = -1;
	int64_t base = 0, row_bytes = 0;
	at::Tensor rows, out;
	std::mutex m;
	std::condition_variable cv;
	bool done = false;
	std::string err;
};

/* heap-allocated and never destroyed: the detached worker waits on them while the process exits */
static std::mutex &g_gather_mu = *new std::mutex;      /* one gather at a time (the vendored pool is not re-entrant) */
static std::mutex &g_gq_mu = *new std::mutex;
static std::condition_variable &g_gq_cv = *new std::condition_variable;
static std::queue<std::shared_ptr<gather_job>> &g_gq = *new std::queue<std::shared_ptr<gather_job>>;
static std::map<int64_t, std::shared_ptr<gather_job>> &g_tickets = *new std::map<int64_t, std::shared_ptr<gather_job>>;
static int64_t g_ticket_next = 1;
static bool g_gq_started = false;

/* The rows through a read-only mapping of the file: a resident page costs a memcpy, not a pread syscall (the Engram
 * tables of a chunk are ~50k rows; a cold page faults in the worker, which is where it should wait). Several threads
 * split the rows. */
static void *
gather_map(int fd, size_t *len)
{
	static std::map<int, std::pair<void *, size_t>> &maps = *new std::map<int, std::pair<void *, size_t>>;
	static std::mutex &mu = *new std::mutex;
	std::lock_guard<std::mutex> lk(mu);
	auto it = maps.find(fd);

	if (it != maps.end()) {
		*len = it->second.second;
		return it->second.first;
	}

	struct stat sb;

	if (fstat(fd, &sb) != 0)
		return NULL;

	void *p = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);

	if (p == MAP_FAILED)
		return NULL;

	madvise(p, (size_t)sb.st_size, MADV_RANDOM);
	maps[fd] = std::make_pair(p, (size_t)sb.st_size);
	*len = (size_t)sb.st_size;

	return p;
}

static void __attribute__((unused))
gather_mmap(gather_job *job)
{
	size_t len = 0;
	const unsigned char *base = (const unsigned char *)gather_map(job->fd, &len);

	if (!base)
		throw std::runtime_error("gather_async: cannot map the table file");

	const int64_t *rows = (const int64_t *)job->rows.data_ptr();
	unsigned char *out = (unsigned char *)job->out.data_ptr();
	int64_t n = job->rows.numel(), rb = job->row_bytes;
	int nt = (n >= 2048) ? 8 : 1;
	std::vector<std::thread> th;
	std::atomic<bool> bad(false);

	auto part = [&](int64_t a, int64_t b) {
		for (int64_t i = a; i < b; i++) {
			size_t off = (size_t)job->base + (size_t)rows[i] * (size_t)rb;

			if (off + (size_t)rb > len) {
				bad = true;
				return;
			}

			memcpy(out + (size_t)i * (size_t)rb, base + off, (size_t)rb);
		}
	};

	for (int t = 1; t < nt; t++)
		th.emplace_back(part, n * t / nt, n * (t + 1) / nt);

	part(0, n / nt);

	for (auto &x : th)
		x.join();

	if (bad)
		throw std::runtime_error("gather_async: row out of the file");
}

static void
gather_worker(void)
{
	for (;;) {
		std::shared_ptr<gather_job> job;

		{
			std::unique_lock<std::mutex> lk(g_gq_mu);

			g_gq_cv.wait(lk, [] { return !g_gq.empty(); });
			job = g_gq.front();
			g_gq.pop();
		}

		std::string err;

		try {
			std::lock_guard<std::mutex> lk(g_gather_mu);

			// the vendored pread pool keeps many cold reads in flight; a mapping (gather_mmap) faults one page per
			// thread at a time and is slower on a cold table
			ngram_gather_cpu(job->fd, job->base, job->row_bytes, job->rows, 0, job->out);
		}
		catch (const std::exception &e) {
			err = e.what();
		}

		{
			std::lock_guard<std::mutex> lk(job->m);

			job->err = err;
			job->done = true;
		}

		job->cv.notify_all();
	}
}

static int64_t
gather_submit(std::shared_ptr<gather_job> job)
{
	std::lock_guard<std::mutex> lk(g_gq_mu);
	int64_t t = g_ticket_next++;

	if (!g_gq_started) {
		g_gq_started = true;
		std::thread(gather_worker).detach();
	}

	g_tickets[t] = job;
	g_gq.push(job);
	g_gq_cv.notify_one();

	return t;
}

static bool
gather_wait(int64_t t, std::string &err)
{
	std::shared_ptr<gather_job> job;

	{
		std::lock_guard<std::mutex> lk(g_gq_mu);
		auto it = g_tickets.find(t);

		if (it == g_tickets.end()) {
			err = "gather_wait: unknown ticket";
			return false;
		}

		job = it->second;
		g_tickets.erase(it);
	}

	std::unique_lock<std::mutex> lk(job->m);

	job->cv.wait(lk, [&] { return job->done; });

	if (!job->err.empty()) {
		err = job->err;
		return false;
	}

	return true;
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

static uc_value_t *torch_st_gather(uc_vm_t *vm, size_t nargs);
static uc_value_t *torch_st_gather_async(uc_vm_t *vm, size_t nargs);
static uc_value_t *torch_st_prefetch(uc_vm_t *vm, size_t nargs);

static const uc_function_list_t stfile_methods[] = {
	{ "list",	torch_st_list },
	{ "info",	torch_st_info },
	{ "gather",	torch_st_gather },
	{ "gather_async",	torch_st_gather_async },
	{ "prefetch",	torch_st_prefetch },
	{ "tensor",	torch_st_tensor },
	{ "free",	stfile_free_method },
};

/*
 * gelu_tanh(): torch.tensor — GELU with the tanh approximation
 * (hidden_act "gelu_pytorch_tanh", the vision tower's MLP activation).
 */
static uc_value_t *
tensor_gelu_tanh(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "gelu_tanh",
		[](const at::Tensor &x) { return at::gelu(x, "tanh"); });
}

/* gelu(): torch.tensor — exact (erf) GELU, the GLM-5 vision merger's
 * activation (GELU "approximate none"). */
static uc_value_t *
tensor_gelu(uc_vm_t *vm, size_t nargs)
{
	return tensor_unary(vm, nargs, "gelu",
		[](const at::Tensor &x) { return at::gelu(x, "none"); });
}

/*
 * layer_norm(x, weight, bias, eps): torch.tensor
 *
 * LayerNorm over the last dim. x, weight, bias must share a dtype
 * (the vision tower keeps its residual stream in fp32 and casts the
 * normalized result down for the fp16 linears).
 */
static uc_value_t *
torch_layer_norm(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs != 4 || !arg_number(vm, uc_fn_arg(3), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, weight, bias, eps)");
		return NULL;
	}

	at::Tensor *x = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *w = x ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *b = w ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!b)
		return NULL;

	try {
		at::Tensor r = at::layer_norm(*x, { w->size(0) }, *w, *b, eps);

		return tensor_wrap(vm, std::move(r), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * image_prepare(data: string, opts): object
 *
 * Decodes an encoded image (PNG/JPEG/BMP/GIF/TGA/PSD/PNM bytes) and
 * produces the vision tower's input: smart-resized (PIL-compatible
 * bicubic), normalized, flattened patches. opts: { patch_size,
 * merge_size, temporal_patch_size, min_pixels, max_pixels,
 * image_mean: [3], image_std: [3] }.
 *
 * Returns { patches: fp16 CPU tensor (grid_h * grid_w, 3 * tp * ps * ps),
 * grid: [1, grid_h, grid_w], width, height, new_width, new_height }.
 */
static bool
opt_number(uc_vm_t *vm, uc_value_t *opts, const char *key, double *out)
{
	uc_value_t *v = ucv_object_get(opts, key, NULL);

	if (!v) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"image_prepare: missing option '%s'", key);
		return false;
	}

	return arg_number(vm, v, out);
}

/* image_info(data): { width, height } of an encoded image */
static uc_value_t *
torch_image_info(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *data = uc_fn_arg(0);
	int w, h;

	if (nargs != 1 || ucv_type(data) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (data)");
		return NULL;
	}

	if (amp_image_info((const unsigned char *)ucv_string_get(data), ucv_string_length(data), &w, &h))
		RAISE("image_info: unsupported or corrupt image");

	uc_value_t *res = ucv_object_new(vm);

	ucv_object_add(res, "width", ucv_int64_new(w));
	ucv_object_add(res, "height", ucv_int64_new(h));

	return res;
}

static uc_value_t *
torch_image_prepare(uc_vm_t *vm, size_t nargs)
{
	uc_value_t *data = uc_fn_arg(0), *opts = uc_fn_arg(1);

	if (nargs != 2 || ucv_type(data) != UC_STRING ||
	    ucv_type(opts) != UC_OBJECT) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (data, opts)");
		return NULL;
	}

	struct amp_image_params p;
	double d;

	memset(&p, 0, sizeof(p));

	if (!opt_number(vm, opts, "patch_size", &d)) return NULL;
	p.patch_size = (int)d;
	if (!opt_number(vm, opts, "merge_size", &d)) return NULL;
	p.merge_size = (int)d;
	if (!opt_number(vm, opts, "temporal_patch_size", &d)) return NULL;
	p.temporal_patch_size = (int)d;
	if (ucv_object_get(opts, "min_tokens", NULL) || ucv_object_get(opts, "target_w", NULL)) {
		/* canvas / target modes: the pixel limits are not used */
	}
	else {
		if (!opt_number(vm, opts, "min_pixels", &d)) return NULL;
		p.min_pixels = (long)d;
		if (!opt_number(vm, opts, "max_pixels", &d)) return NULL;
		p.max_pixels = (long)d;
	}

	/* GLM-5 canvas mode: min_tokens / max_tokens instead of min/max pixels */
	{
		uc_value_t *mt = ucv_object_get(opts, "min_tokens", NULL);

		if (mt) {
			if (!arg_number(vm, mt, &d)) return NULL;
			p.min_tokens = (long)d;
			if (!opt_number(vm, opts, "max_tokens", &d)) return NULL;
			p.max_tokens = (long)d;
			p.max_pixels = 1;
		}
	}

	/* DeepSeek-V4 mode: the final size comes from the caller */
	{
		uc_value_t *tw = ucv_object_get(opts, "target_w", NULL);

		if (tw) {
			if (!arg_number(vm, tw, &d)) return NULL;
			p.target_w = (int)d;
			if (!opt_number(vm, opts, "target_h", &d)) return NULL;
			p.target_h = (int)d;
			p.pad = ucv_is_truish(ucv_object_get(opts, "pad", NULL)) ? 1 : 0;
			p.max_pixels = 1;

			if (p.target_w < p.patch_size || p.target_h < p.patch_size ||
			    p.target_w % p.patch_size || p.target_h % p.patch_size)
				RAISE("image_prepare: target size must be a multiple of the patch size");
		}
	}

	p.hwc = ucv_is_truish(ucv_object_get(opts, "hwc", NULL)) ? 1 : 0;

	if (p.patch_size < 1 || p.merge_size < 1 || p.temporal_patch_size < 1 ||
	    p.max_pixels < 1)
		RAISE("image_prepare: invalid geometry options");

	const char *keys[2] = { "image_mean", "image_std" };

	for (int k = 0; k < 2; k++) {
		uc_value_t *a = ucv_object_get(opts, keys[k], NULL);

		if (!a || ucv_type(a) != UC_ARRAY || ucv_array_length(a) != 3) {
			uc_vm_raise_exception(vm, EXCEPTION_TYPE,
				"image_prepare: '%s' must be an array of 3 numbers", keys[k]);
			return NULL;
		}

		for (int c = 0; c < 3; c++) {
			if (!arg_number(vm, ucv_array_get(a, c), &d))
				return NULL;

			(k ? p.std : p.mean)[c] = (float)d;
		}
	}

	struct amp_image_result r;
	char err[256] = "";

	if (amp_image_prepare((const unsigned char *)ucv_string_get(data),
	                     ucv_string_length(data), &p, &r, err, sizeof(err))) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME,
			"image_prepare: %s", err);
		return NULL;
	}

	uc_value_t *res = NULL;

	try {
		int64_t rows = (int64_t)r.grid_h * r.grid_w;
		int64_t cols = (int64_t)3 * p.temporal_patch_size *
			p.patch_size * p.patch_size;
		at::Tensor f = at::from_blob(r.patches, { rows, cols },
			at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
		at::Tensor h = f.to(at::kHalf).contiguous();

		free(r.patches);
		r.patches = NULL;

		res = ucv_object_new(vm);
		ucv_object_add(res, "patches", tensor_wrap(vm, std::move(h), 0));

		uc_value_t *grid = ucv_array_new(vm);

		ucv_array_push(grid, ucv_int64_new(1));
		ucv_array_push(grid, ucv_int64_new(r.grid_h));
		ucv_array_push(grid, ucv_int64_new(r.grid_w));
		ucv_object_add(res, "grid", grid);
		ucv_object_add(res, "width", ucv_int64_new(r.width));
		ucv_object_add(res, "height", ucv_int64_new(r.height));
		ucv_object_add(res, "new_width", ucv_int64_new(r.new_width));
		ucv_object_add(res, "new_height", ucv_int64_new(r.new_height));

		return res;
	}
	catch (const std::exception &e) {
		free(r.patches);
		ucv_put(res);
		RAISE(e.what());
	}
}

/*
 * hc_mix(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters):
 *   [post, comb, collapsed]
 *
 * mHC HyperConnection mix for one sublayer site (GLM-5.3 / DeepSeek-V4):
 * streams (R, 4, D) fp32 contiguous -> post (R, 4) fp32, comb (R, 4, 4)
 * fp32 (Sinkhorn-normalized) and collapsed (R, D) fp16 (the sublayer
 * input before its norm). fn (24, 4*D), base (24), scale (3) are fp32
 * (an fp16 fn is accepted too: the decode-size traffic saver).
 */
static uc_value_t *
torch_hc_mix(uc_vm_t *vm, size_t nargs)
{
	double rms_eps, hc_eps, iters;

	if (nargs != 7 || !arg_number(vm, uc_fn_arg(4), &rms_eps) ||
	    !arg_number(vm, uc_fn_arg(5), &hc_eps) ||
	    !arg_number(vm, uc_fn_arg(6), &iters)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters)");
		return NULL;
	}

	at::Tensor *st = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *fn = st ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *base = fn ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *scale = base ? tensor_arg(vm, uc_fn_arg(3)) : NULL;

	if (!scale)
		return NULL;

	if (st->dim() != 3 || st->size(1) != 4 || st->scalar_type() != at::kFloat ||
	    !st->is_contiguous() || st->size(2) % 4 || !st->is_cuda())
		RAISE("hc_mix: streams must be a contiguous fp32 CUDA tensor (R, 4, D), D % 4 == 0");

	if (fn->size(0) != 24 || fn->size(1) != 4 * st->size(2) || !fn->is_contiguous() ||
	    base->numel() != 24 || scale->numel() != 3)
		RAISE("hc_mix: fn must be (24, 4*D), base (24), scale (3)");

	try {
		TensorDeviceGuard dg(*st);

		int64_t R = st->size(0), D = st->size(2);
		int chunks = hc_mix_num_chunks((int)R, (int)(4 * D));
		auto f32 = st->options();
		at::Tensor partials = at::empty({ R, chunks, 24 + 1 }, f32);
		at::Tensor post = at::empty({ R, 4 }, f32);
		at::Tensor comb = at::empty({ R, 4, 4 }, f32);
		at::Tensor coll = at::empty({ R, D }, st->options().dtype(at::kHalf));

		hc_mix(*st, *fn, *base, *scale, rms_eps, hc_eps, (int64_t)iters,
			partials, post, comb, coll);

		uc_value_t *r = ucv_array_new(vm);

		ucv_array_push(r, tensor_wrap(vm, std::move(post), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(comb), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(coll), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * hc_mix2(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, pre_in): [post, comb, collapsed, pre]
 *
 * hc_mix for DeepSeek-V4.1's "one site late" collapse: the sublayer input is the streams collapsed with
 * pre_in (R, 4) fp32 (the previous site's pre), clamped to the fp16 range; also returns this site's pre
 * (R, 4) fp32 (sigmoid(...) + hc_eps) for the next collapse.
 */
static uc_value_t *
torch_hc_mix2(uc_vm_t *vm, size_t nargs)
{
	double rms_eps, hc_eps, iters;

	if (nargs != 8 || !arg_number(vm, uc_fn_arg(4), &rms_eps) ||
	    !arg_number(vm, uc_fn_arg(5), &hc_eps) ||
	    !arg_number(vm, uc_fn_arg(6), &iters)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, pre_in)");
		return NULL;
	}

	at::Tensor *st = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *fn = st ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *base = fn ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *scale = base ? tensor_arg(vm, uc_fn_arg(3)) : NULL;
	at::Tensor *pre = scale ? tensor_arg(vm, uc_fn_arg(7)) : NULL;

	if (!pre)
		return NULL;

	if (st->dim() != 3 || st->size(1) != 4 || st->scalar_type() != at::kFloat ||
	    !st->is_contiguous() || st->size(2) % 4 || !st->is_cuda())
		RAISE("hc_mix2: streams must be a contiguous fp32 CUDA tensor (R, 4, D), D % 4 == 0");

	if (fn->size(0) != 24 || fn->size(1) != 4 * st->size(2) || !fn->is_contiguous() ||
	    base->numel() != 24 || scale->numel() != 3 || pre->scalar_type() != at::kFloat ||
	    !pre->is_contiguous() || pre->numel() != st->size(0) * 4 || pre->device() != st->device())
		RAISE("hc_mix2: fn must be (24, 4*D), base (24), scale (3), pre_in (R, 4) fp32 on the same device");

	try {
		TensorDeviceGuard dg(*st);

		int64_t R = st->size(0), D = st->size(2);
		int chunks = hc_mix_num_chunks((int)R, (int)(4 * D));
		auto f32 = st->options();
		at::Tensor partials = at::empty({ R, chunks, 24 + 1 }, f32);
		at::Tensor post = at::empty({ R, 4 }, f32);
		at::Tensor comb = at::empty({ R, 4, 4 }, f32);
		at::Tensor pre_out = at::empty({ R, 4 }, f32);
		at::Tensor coll = at::empty({ R, D }, st->options().dtype(at::kHalf));

		hc_mix2(*st, *fn, *base, *scale, rms_eps, hc_eps, (int64_t)iters, *pre,
			partials, post, comb, pre_out, coll);

		uc_value_t *r = ucv_array_new(vm);

		ucv_array_push(r, tensor_wrap(vm, std::move(post), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(comb), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(coll), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(pre_out), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * hc_apply(x, y, post, comb): x
 *
 * Residual update of one sublayer site, IN PLACE on the streams:
 * x <- post (x) y + comb^T x. x (R, 4, D) fp32, y (R, D) fp16 or fp32,
 * post (R, 4) fp32, comb (R, 4, 4) fp32. Returns x.
 */
static uc_value_t *
torch_hc_apply(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 4) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (x, y, post, comb)");
		return NULL;
	}

	at::Tensor *x = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *y = x ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *post = y ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *comb = NULL;

	/* comb may be null: the gated-residual form (Qwen3.8-Flash-Next),
	 * x[h] += post[h] * y without a stream-mixing matrix */
	if (post && ucv_type(uc_fn_arg(3)) != UC_NULL)
		comb = tensor_arg(vm, uc_fn_arg(3));

	if (!post || (!comb && ucv_type(uc_fn_arg(3)) != UC_NULL))
		return NULL;

	if (x->dim() != 3 || x->size(1) != 4 || x->scalar_type() != at::kFloat ||
	    !x->is_contiguous() || x->size(2) % 4 || !x->is_cuda() ||
	    !y->is_contiguous() || !post->is_contiguous() ||
	    (comb && !comb->is_contiguous()) ||
	    (y->scalar_type() != at::kHalf && y->scalar_type() != at::kFloat) ||
	    post->scalar_type() != at::kFloat ||
	    (comb && comb->scalar_type() != at::kFloat) ||
	    y->numel() != x->size(0) * x->size(2) || post->numel() != x->size(0) * 4 ||
	    (comb && comb->numel() != x->size(0) * 16))
		RAISE("hc_apply: bad shapes/dtypes (x (R,4,D) fp32, y (R,D), post (R,4), comb (R,4,4) or null)");

	try {
		TensorDeviceGuard dg(*x);
		c10::optional<at::Tensor> co;

		if (comb)
			co = *comb;

		hc_apply(*x, *y, *post, co);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(uc_fn_arg(0));
}

/* Qwen3.8-Flash-Next kernels (vendor/exl3/exllamav3_ext/ngram.cu, ple.cu) */
int64_t ngram_hash_cpu
(
	const at::Tensor& ids, int64_t seq_len, const at::Tensor& multipliers,
	const at::Tensor& offsets, const at::Tensor& sizes, int64_t heads_per_ngram,
	int64_t eos_token, at::Tensor uids, at::Tensor inverse, at::Tensor heads
);

void ngram_gather_cpu
(
	int64_t fd, int64_t base_offset, int64_t row_bytes, const at::Tensor& uids,
	int64_t uid_base, at::Tensor out
);

void ngram_dequant
(
	const at::Tensor& packed, int64_t K, const at::Tensor& heads,
	const at::Tensor& bias, at::Tensor out
);

void ple_forward_streams
(
	const at::Tensor& streams, const at::Tensor& emb, const at::Tensor& key_w,
	const at::Tensor& value_w, const at::Tensor& norm_key_w,
	const at::Tensor& norm_query_w, const at::Tensor& norm_conv_w,
	const at::Tensor& conv_w, const c10::optional<at::Tensor>& conv_state,
	double rms_eps, double gate_scale, int64_t conv_dilation,
	at::Tensor delta, at::Tensor conv_stream
);

/*
 * ngram_hash(ids, out_len, multipliers, offsets, sizes, heads_per_ngram, eos,
 *            uids, inverse, heads): int
 *
 * Hash a (1, ctx + out_len) int64 CPU token history into the n-gram table
 * row ids: fills the sorted unique row ids (uids), the inverse map
 * (out_len * heads entries) and each unique row's hash head, all CPU
 * tensors (pinned or not), and returns the unique count.
 */
static uc_value_t *
torch_ngram_hash(uc_vm_t *vm, size_t nargs)
{
	int64_t out_len, hpn, eos;

	if (nargs != 10 || !arg_int(vm, uc_fn_arg(1), &out_len) ||
	    !arg_int(vm, uc_fn_arg(5), &hpn) || !arg_int(vm, uc_fn_arg(6), &eos)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (ids, out_len, multipliers, offsets, sizes, heads_per_ngram, eos, uids, inverse, heads)");
		return NULL;
	}

	at::Tensor *t[7];
	static const int idx[7] = { 0, 2, 3, 4, 7, 8, 9 };

	for (int i = 0; i < 7; i++) {
		t[i] = tensor_arg(vm, uc_fn_arg(idx[i]));

		if (!t[i])
			return NULL;

		if (t[i]->is_cuda())
			RAISE("ngram_hash: tensors must be on the CPU");
	}

	try {
		int64_t u = ngram_hash_cpu(*t[0], out_len, *t[1], *t[2], *t[3], hpn,
		                           eos, *t[4], *t[5], *t[6]);

		return ucv_int64_new(u);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * stfile.gather(name, rows, out): out
 *
 * Read the rows of tensor `name` (row-major, first dimension = row) listed in `rows` (int64 CPU,
 * ascending) with threaded preads from the file itself into out (len(rows), row bytes) CPU
 * tensor. For tables far larger than RAM (the DeepSeek-V4.1 Engram embeddings): the page
 * cache holds the hot rows, cold ones cost one disk read each, all in flight together.
 */
static uc_value_t *
torch_st_gather(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);
	uc_value_t *name = uc_fn_arg(0);
	const char *dtype;
	int64_t *shape;
	size_t ndim, offset, nbytes;

	if (!f)
		return NULL;

	if (nargs != 3 || ucv_type(name) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (name, rows, out)");
		return NULL;
	}

	at::Tensor *rows = tensor_arg(vm, uc_fn_arg(1));
	at::Tensor *out = rows ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!out)
		return NULL;

	if (st_lookup(f, ucv_string_get(name), &dtype, &shape, &ndim, &offset, &nbytes) == ST_NOT_FOUND) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "no such tensor");
		return NULL;
	}

	int64_t nrows = ndim ? shape[0] : 0;

	free(shape);

	if (nrows < 1 || nbytes % (size_t)nrows)
		RAISE("gather: not a row-major table");

	int64_t row_bytes = (int64_t)(nbytes / (size_t)nrows);

	if (rows->scalar_type() != at::kLong || rows->is_cuda() || !rows->is_contiguous() ||
	    out->is_cuda() || !out->is_contiguous() || out->dim() != 2 || out->size(0) < rows->numel() ||
	    out->size(1) * out->element_size() != row_bytes)
		RAISE("gather: rows must be contiguous int64 CPU, out a contiguous CPU (n, row bytes) tensor");

	if (rows->numel() > 0) {
		int64_t lo = rows->min().item<int64_t>(), hi = rows->max().item<int64_t>();

		if (lo < 0 || hi >= nrows)
			RAISE("gather: row index out of range");
	}

	try {
		std::lock_guard<std::mutex> lk(g_gather_mu);

		ngram_gather_cpu(f->fd, (int64_t)(f->data_off + offset), row_bytes, *rows, 0, *out);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(uc_fn_arg(2));
}

/*
 * stfile.gather_async(name, rows, out): int ticket
 *
 * The same read as gather(), done by a worker thread: returns at once, torch.gather_wait(ticket) blocks until the
 * rows are in `out` (rows and out stay referenced until then). The Engram rows of a prefill chunk are known when
 * the chunk is admitted: the reads run while the host issues the work of the other chunks.
 */
static uc_value_t *
torch_st_gather_async(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);
	uc_value_t *name = uc_fn_arg(0);
	const char *dtype;
	int64_t *shape;
	size_t ndim, offset, nbytes;

	if (!f)
		return NULL;

	if ((nargs != 3 && nargs != 4) || ucv_type(name) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (name, rows, out, total_rows?)");
		return NULL;
	}

	at::Tensor *rows = tensor_arg(vm, uc_fn_arg(1));
	at::Tensor *out = rows ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!out)
		return NULL;

	if (st_lookup(f, ucv_string_get(name), &dtype, &shape, &ndim, &offset, &nbytes) == ST_NOT_FOUND) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "no such tensor");
		return NULL;
	}

	int64_t nrows = ndim ? shape[0] : 0;

	free(shape);

	if (nrows < 1 || nbytes % (size_t)nrows)
		RAISE("gather_async: not a row-major table");

	int64_t row_bytes = (int64_t)(nbytes / (size_t)nrows);

	/* total_rows: the table is stored as consecutive equal shards (name = the first one): the rows run on through
	 * the file past the first tensor */
	if (nargs == 4) {
		int64_t total;

		if (!arg_int(vm, uc_fn_arg(3), &total) || total < nrows)
			RAISE("gather_async: total_rows must be an integer >= the first shard's rows");

		struct stat sb;

		if (fstat(f->fd, &sb) != 0 || (int64_t)(f->data_off + offset) + total * row_bytes > (int64_t)sb.st_size)
			RAISE("gather_async: total_rows run past the end of the file");

		nrows = total;
	}

	if (rows->scalar_type() != at::kLong || rows->is_cuda() || !rows->is_contiguous() ||
	    out->is_cuda() || !out->is_contiguous() || out->dim() != 2 || out->size(0) < rows->numel() ||
	    out->size(1) * out->element_size() != row_bytes)
		RAISE("gather_async: rows must be contiguous int64 CPU, out a contiguous CPU (n, row bytes) tensor");

	if (rows->numel() > 0) {
		int64_t lo = rows->min().item<int64_t>(), hi = rows->max().item<int64_t>();

		if (lo < 0 || hi >= nrows)
			RAISE("gather_async: row index out of range");
	}

	auto st = std::make_shared<gather_job>();

	st->fd = f->fd;
	st->base = (int64_t)(f->data_off + offset);
	st->row_bytes = row_bytes;
	st->rows = *rows;
	st->out = *out;

	return ucv_int64_new(gather_submit(st));
}

/* torch.gather_wait(ticket): block until the async gather finished; raises its error */
static uc_value_t *
torch_gather_wait(uc_vm_t *vm, size_t nargs)
{
	if (nargs != 1 || ucv_type(uc_fn_arg(0)) != UC_INTEGER)
		RAISE("gather_wait: expected (ticket)");

	std::string err;

	if (!gather_wait(ucv_int64_get(uc_fn_arg(0)), err))
		RAISE(err.c_str());

	return NULL;
}

/*
 * stfile.prefetch(name, rows): null
 *
 * Ask the kernel to read the rows of tensor `name` (first dimension = row, int64 CPU) into the page cache
 * (POSIX_FADV_WILLNEED); returns at once. For tables far larger than RAM whose rows are known before they
 * are needed (the Engram hashes depend on the token ids only).
 */
static uc_value_t *
torch_st_prefetch(uc_vm_t *vm, size_t nargs)
{
	st_file_t *f = stfile_this(vm);
	uc_value_t *name = uc_fn_arg(0);
	const char *dtype;
	int64_t *shape;
	size_t ndim, offset, nbytes;

	if (!f)
		return NULL;

	if (nargs != 2 || ucv_type(name) != UC_STRING) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE, "expected (name, rows)");
		return NULL;
	}

	at::Tensor *rows = tensor_arg(vm, uc_fn_arg(1));

	if (!rows)
		return NULL;

	if (st_lookup(f, ucv_string_get(name), &dtype, &shape, &ndim, &offset, &nbytes) == ST_NOT_FOUND) {
		uc_vm_raise_exception(vm, EXCEPTION_RUNTIME, "no such tensor");
		return NULL;
	}

	int64_t nrows = ndim ? shape[0] : 0;

	free(shape);

	if (nrows < 1 || nbytes % (size_t)nrows || rows->scalar_type() != at::kLong || rows->is_cuda() ||
	    !rows->is_contiguous())
		RAISE("prefetch: rows must be contiguous int64 CPU, the tensor a row-major table");

	int64_t row_bytes = (int64_t)(nbytes / (size_t)nrows);
	const int64_t *r = (const int64_t *)rows->data_ptr();
	int64_t n = rows->numel();

	/* hundreds of thousands of fadvise calls take seconds: on a detached thread, off the launch path */
	std::vector<int64_t> list(r, r + n);
	int fd = f->fd;
	off_t base = (off_t)(f->data_off + offset);

	std::thread([list = std::move(list), fd, base, nrows, row_bytes]() {
		for (int64_t row : list)
			if (row >= 0 && row < nrows)
				posix_fadvise(fd, base + (off_t)((size_t)row * (size_t)row_bytes), (off_t)row_bytes,
					POSIX_FADV_WILLNEED);
	}).detach();

	return NULL;
}

/*
 * ngram_dequant(packed, K, heads, bias, out): out
 *
 * Decode packed n-gram rows (U, 1 + 10*K) int16 on the GPU into out
 * (U, 160) fp16: mul1 codebook * row scale + the per-hash-head bias row.
 */
static uc_value_t *
torch_ngram_dequant(uc_vm_t *vm, size_t nargs)
{
	int64_t K;

	if (nargs != 5 || !arg_int(vm, uc_fn_arg(1), &K)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (packed, K, heads, bias, out)");
		return NULL;
	}

	at::Tensor *packed = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *heads = packed ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *bias = heads ? tensor_arg(vm, uc_fn_arg(3)) : NULL;
	at::Tensor *out = bias ? tensor_arg(vm, uc_fn_arg(4)) : NULL;

	if (!out)
		return NULL;

	if (!out->is_cuda() || !packed->is_cuda())
		RAISE("ngram_dequant: packed and out must be CUDA tensors");

	try {
		TensorDeviceGuard dg(*out);

		ngram_dequant(*packed, K, *heads, *bias, *out);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return ucv_get(uc_fn_arg(4));
}

/*
 * ple_forward(streams, emb, key_w, value_w, norm_key, norm_query, norm_conv,
 *             conv_w, conv_state|null, eps, gate_scale, dilation, delta,
 *             conv_stream)
 *
 * Fused PLE layer for a stream stack (R rows): streams (1, R, 4, D) fp32,
 * emb (1, R, E) fp16 -> delta (1, R, 4, D) fp32 (to be added to the
 * streams by the caller) and conv_stream (1, 4D, state + R) fp16 (its
 * trailing columns are the next conv state).
 */
static uc_value_t *
torch_ple_forward(uc_vm_t *vm, size_t nargs)
{
	double eps, gs;
	int64_t dil;

	if (nargs != 14 || !arg_number(vm, uc_fn_arg(9), &eps) ||
	    !arg_number(vm, uc_fn_arg(10), &gs) || !arg_int(vm, uc_fn_arg(11), &dil)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (streams, emb, key_w, value_w, norm_key, norm_query, norm_conv, conv_w, conv_state, eps, gate_scale, dilation, delta, conv_stream)");
		return NULL;
	}

	at::Tensor *t[8];

	for (int i = 0; i < 8; i++) {
		t[i] = tensor_arg(vm, uc_fn_arg(i));

		if (!t[i])
			return NULL;
	}

	at::Tensor *cs = NULL;

	if (ucv_type(uc_fn_arg(8)) != UC_NULL) {
		cs = tensor_arg(vm, uc_fn_arg(8));

		if (!cs)
			return NULL;
	}

	at::Tensor *delta = tensor_arg(vm, uc_fn_arg(12));
	at::Tensor *cstream = delta ? tensor_arg(vm, uc_fn_arg(13)) : NULL;

	if (!cstream)
		return NULL;

	try {
		TensorDeviceGuard dg(*t[0]);
		c10::optional<at::Tensor> cso;

		if (cs)
			cso = *cs;

		ple_forward_streams(*t[0], *t[1], *t[2], *t[3], *t[4], *t[5], *t[6],
		                    *t[7], cso, eps, gs, dil, *delta, *cstream);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}

	return NULL;
}

/* Gated-residual mix, fused decode form (vendor hc_mix.cu gr_mix) */
void gr_mix
(
	const at::Tensor& streams, const at::Tensor& fn, const at::Tensor& upt,
	const at::Tensor& w, double rms_eps, at::Tensor dots,
	c10::optional<at::Tensor> post, at::Tensor mixed
);

/*
 * gr_mix(streams, fn, upx, w, eps, inject): [post | null, mixed]
 *
 * Qwen3.8-Flash-Next gated-residual mix for R <= 8 rows (decode / verify):
 * streams (R, 4, D) fp32; fn (rank [+ 4], 4*D) fp16 = cat(down, inject) * w
 * (folded); upx (4, D/4, rank, 4) fp16 = the up projection repacked for the
 * kernel's rank loop; w (4*D) fp16 = hc_norm + 1. Returns post (R, 4) fp32
 * (the injection gates, when inject) and the mixed sublayer input (R, D)
 * fp16. Allocates its outputs (capture-safe: graph pool).
 */
static uc_value_t *
torch_gr_mix(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs != 6 || !arg_number(vm, uc_fn_arg(4), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (streams, fn, upx, w, eps, inject)");
		return NULL;
	}

	at::Tensor *st = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *fn = st ? tensor_arg(vm, uc_fn_arg(1)) : NULL;
	at::Tensor *upx = fn ? tensor_arg(vm, uc_fn_arg(2)) : NULL;
	at::Tensor *w = upx ? tensor_arg(vm, uc_fn_arg(3)) : NULL;
	bool inject = ucv_is_truish(uc_fn_arg(5));

	if (!w)
		return NULL;

	if (st->dim() != 3 || st->size(1) != 4 || st->scalar_type() != at::kFloat ||
	    !st->is_contiguous() || st->size(2) % 128 || !st->is_cuda() || st->size(0) > 8)
		RAISE("gr_mix: streams must be a contiguous fp32 CUDA tensor (R <= 8, 4, D), D % 128 == 0");

	try {
		TensorDeviceGuard dg(*st);

		int64_t R = st->size(0), D = st->size(2);
		int64_t M = fn->size(0) + 1;
		at::Tensor dots = at::empty({ R, M, 4 }, st->options());
		at::Tensor post = at::empty({ R, 4 }, st->options());
		at::Tensor mixed = at::empty({ R, D }, st->options().dtype(at::kHalf));
		c10::optional<at::Tensor> po;

		if (inject)
			po = post;

		gr_mix(*st, *fn, *upx, *w, eps, dots, po, mixed);

		uc_value_t *r = ucv_array_new(vm);

		ucv_array_push(r, inject ? tensor_wrap(vm, std::move(post), 0) : NULL);
		ucv_array_push(r, tensor_wrap(vm, std::move(mixed), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

extern "C" long
amp_qsa_attn_ws_floats(int R, int Hq, int Hk, int E, int ns);

extern "C" int
amp_qsa_attn_fp16(const void *q, const void *kc, const void *vc, const void *ks, const void *vs, const int *idx,
                 void *ws, void *out, int R, int Hq, int Hk, int E, int Kt,
                 int ns, float scale, cudaStream_t stream);

/*
 * qsa_attn(q, k_cache, v_cache, idx, scale, splits?): out
 *
 * Gathered GQA attention (Qwen3.8-Flash-Next QSA): q (R, Hq, E) fp16, the K/V
 * caches (max_seq, Hk, E) fp16 — or each [codes (max_seq, Hk, E) int8, scales
 * (max_seq, Hk, E / 32) fp16], a Q8 cache dequantized while the kernel stages
 * its tiles — idx (R, Kt) int32 — per query row the cache
 * rows it attends to, -1 = padding. Returns (R, Hq, E) fp16. splits: KV
 * splits per (row, kv head); default sized so the launch fills the GPU.
 */
static uc_value_t *
torch_qsa_attn(uc_vm_t *vm, size_t nargs)
{
	double scale;

	if (nargs < 5 || nargs > 6 || !arg_number(vm, uc_fn_arg(4), &scale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k_cache, v_cache, idx, scale, splits?)");
		return NULL;
	}

	at::Tensor *q = tensor_arg(vm, uc_fn_arg(0));
	at::Tensor *kc = NULL, *vc = NULL, *ksc = NULL, *vsc = NULL;
	uc_value_t *ka = q ? uc_fn_arg(1) : NULL, *va = q ? uc_fn_arg(2) : NULL;

	/* each cache: a tensor, or [codes, scales] */
	if (ka && ucv_type(ka) == UC_ARRAY) {
		if (ucv_array_length(ka) != 2 || !(kc = tensor_arg(vm, ucv_array_get(ka, 0))) ||
		    !(ksc = tensor_arg(vm, ucv_array_get(ka, 1))))
			return NULL;
	}
	else if (ka)
		kc = tensor_arg(vm, ka);

	if (kc && va && ucv_type(va) == UC_ARRAY) {
		if (ucv_array_length(va) != 2 || !(vc = tensor_arg(vm, ucv_array_get(va, 0))) ||
		    !(vsc = tensor_arg(vm, ucv_array_get(va, 1))))
			return NULL;
	}
	else if (kc && va)
		vc = tensor_arg(vm, va);

	at::Tensor *idx = vc ? tensor_arg(vm, uc_fn_arg(3)) : NULL;

	if (!idx)
		return NULL;

	bool q8 = (ksc != NULL);
	at::ScalarType cdt = q8 ? at::kChar : at::kHalf;

	if ((ksc == NULL) != (vsc == NULL) ||
	    (q8 && (ksc->scalar_type() != at::kHalf || vsc->scalar_type() != at::kHalf || !ksc->is_contiguous() ||
	            !vsc->is_contiguous() || ksc->dim() != 3 || vsc->dim() != 3 || ksc->size(2) * 32 != kc->size(2) ||
	            vsc->size(2) * 32 != kc->size(2) || ksc->size(0) != kc->size(0) || vsc->size(0) != kc->size(0) ||
	            ksc->size(1) != kc->size(1) || vsc->size(1) != kc->size(1))))
		RAISE("qsa_attn: a Q8 cache is [codes (max_seq, Hk, E) int8, scales (max_seq, Hk, E / 32) fp16] for K and V");

	if (q->dim() != 3 || kc->dim() != 3 || vc->dim() != 3 || idx->dim() != 2 ||
	    q->scalar_type() != at::kHalf || kc->scalar_type() != cdt ||
	    vc->scalar_type() != cdt || idx->scalar_type() != at::kInt ||
	    !q->is_contiguous() || !kc->is_contiguous() || !vc->is_contiguous() ||
	    !idx->is_contiguous() || !q->is_cuda() || q->size(0) != idx->size(0) ||
	    kc->size(1) != vc->size(1) || kc->size(2) != q->size(2) ||
	    vc->size(2) != q->size(2) || q->size(1) % kc->size(1))
		RAISE("qsa_attn: q (R, Hq, E) fp16, caches (max_seq, Hk, E) fp16 (or Q8), idx (R, Kt) int32, all contiguous CUDA");

	try {
		TensorDeviceGuard dg(*q);

		int R = (int)q->size(0), Hq = (int)q->size(1), E = (int)q->size(2);
		int Hk = (int)kc->size(1), Kt = (int)idx->size(1);
		int64_t sp = 0;

		if (nargs == 6 && !arg_int(vm, uc_fn_arg(5), &sp))
			return NULL;

		if (sp <= 0) {
			sp = (R * Hk >= 256) ? 1 : (256 + R * Hk - 1) / (R * Hk);

			if (sp > 16)
				sp = 16;
		}

		at::Tensor ws = at::empty({ amp_qsa_attn_ws_floats(R, Hq, Hk, E, (int)sp) },
			q->options().dtype(at::kFloat));
		at::Tensor out = at::empty({ R, Hq, E }, q->options());

		if (!launch_ok(vm, amp_qsa_attn_fp16(q->data_ptr(), kc->data_ptr(),
		    vc->data_ptr(), q8 ? ksc->data_ptr() : NULL, q8 ? vsc->data_ptr() : NULL,
		    (const int *)idx->data_ptr(), ws.data_ptr(), out.data_ptr(),
		    R, Hq, Hk, E, Kt, (int)sp, (float)scale,
		    c10::cuda::getCurrentCUDAStream().stream()), "qsa_attn"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * kda_scan(q, k, v, beta, g, state, scale, states_out?): torch.tensor
 *
 * Kimi delta attention scan (GLM-5.3): gated delta rule with per-key-
 * channel log-decay g. q, k (S, H, DK) fp32 (l2-normalized), v (S, H, dv)
 * fp32, beta (S, H) fp32, g (S, H, DK) fp32, state (H, DK, dv) fp32
 * (updated in place). Returns out (S, H, dv) fp32 = scale * S^T q.
 */
static uc_value_t *
torch_kda_scan(uc_vm_t *vm, size_t nargs)
{
	double scale;

	if ((nargs != 7 && nargs != 8) || !arg_number(vm, uc_fn_arg(6), &scale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (q, k, v, beta, g, state, scale, states_out?)");
		return NULL;
	}

	at::Tensor *q = f32_arg(vm, uc_fn_arg(0), "kda_scan: q");
	at::Tensor *k = q ? f32_arg(vm, uc_fn_arg(1), "kda_scan: k") : NULL;
	at::Tensor *v = k ? f32_arg(vm, uc_fn_arg(2), "kda_scan: v") : NULL;
	at::Tensor *beta = v ? f32_arg(vm, uc_fn_arg(3), "kda_scan: beta") : NULL;
	at::Tensor *g = beta ? f32_arg(vm, uc_fn_arg(4), "kda_scan: g") : NULL;
	at::Tensor *st = g ? f32_arg(vm, uc_fn_arg(5), "kda_scan: state") : NULL;

	if (!st)
		return NULL;

	at::Tensor *so = NULL;
	uc_value_t *sov = uc_fn_arg(7);

	if (nargs == 8 && ucv_type(sov) != UC_NULL) {
		if (!(so = f32_arg(vm, sov, "kda_scan: states_out")))
			return NULL;

		if (so->dim() != 4 || so->size(0) > q->size(0) ||
		    so->size(1) != st->size(0) || so->size(2) != st->size(1) ||
		    so->size(3) != st->size(2) || so->get_device() != st->get_device())
			RAISE("kda_scan: states_out must be (n <= S, H, DK, dv) on the state's device");
	}

	if (q->dim() != 3 || k->sizes() != q->sizes() || g->sizes() != q->sizes() ||
	    v->dim() != 3 || v->size(0) != q->size(0) || v->size(1) != q->size(1) ||
	    st->dim() != 3 || st->size(0) != q->size(1) || st->size(1) != q->size(2) ||
	    st->size(2) != v->size(2) || beta->numel() != q->size(0) * q->size(1))
		RAISE("kda_scan: incompatible shapes");

	int dev = q->get_device();

	if (k->get_device() != dev || v->get_device() != dev ||
	    beta->get_device() != dev || g->get_device() != dev ||
	    st->get_device() != dev)
		RAISE("kda_scan: all tensors must be on the same device");

	TensorDeviceGuard dg(*q);

	try {
		at::Tensor out = at::empty({ v->size(0), v->size(1), v->size(2) },
			v->options());

		int rc = amp_kda_scan_f32(q->data_ptr(), k->data_ptr(), v->data_ptr(),
			beta->data_ptr(), g->data_ptr(), st->data_ptr(), out.data_ptr(),
			(int)q->size(0), (int)q->size(1), (int)q->size(2),
			(int)v->size(2), (float)scale, so ? so->data_ptr() : NULL,
			so ? (int)so->size(0) : 0,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kda_scan"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * kda_gates(f, dt_bias, decay, braw, lower_bound, beta_scale): [g, beta]
 *
 * f (S, H*DK) fp32 forget-gate logits, dt_bias (H*DK) fp32, decay (H) fp32
 * (= exp(A_log)), braw (S, H) fp32 beta logits. Returns g (S, H, DK) =
 * lower_bound * sigmoid(decay * (f + dt_bias)) and beta (S, H) =
 * sigmoid(braw * beta_scale), both fp32.
 */
static uc_value_t *
torch_kda_gates(uc_vm_t *vm, size_t nargs)
{
	double lb, bscale;

	if (nargs != 6 || !arg_number(vm, uc_fn_arg(4), &lb) ||
	    !arg_number(vm, uc_fn_arg(5), &bscale)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (f, dt_bias, decay, braw, lower_bound, beta_scale)");
		return NULL;
	}

	at::Tensor *f = f32_arg(vm, uc_fn_arg(0), "kda_gates: f");
	at::Tensor *dt = f ? f32_arg(vm, uc_fn_arg(1), "kda_gates: dt_bias") : NULL;
	at::Tensor *dec = dt ? f32_arg(vm, uc_fn_arg(2), "kda_gates: decay") : NULL;
	at::Tensor *br = dec ? f32_arg(vm, uc_fn_arg(3), "kda_gates: braw") : NULL;

	if (!br)
		return NULL;

	int64_t S = f->size(0), H = dec->numel();

	if (f->dim() != 2 || dt->numel() != f->size(1) || H <= 0 ||
	    f->size(1) % H || br->numel() != S * H)
		RAISE("kda_gates: incompatible shapes");

	int64_t DK = f->size(1) / H;

	TensorDeviceGuard dg(*f);

	try {
		at::Tensor g = at::empty({ S, H, DK }, f->options());
		at::Tensor beta = at::empty({ S, H }, f->options());

		int rc = amp_kda_gates(f->data_ptr(), dt->data_ptr(), dec->data_ptr(),
			br->data_ptr(), g.data_ptr(), beta.data_ptr(), (long)S, (int)H,
			(int)DK, (float)lb, (float)bscale,
			c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kda_gates"))
			return NULL;

		uc_value_t *r = ucv_array_new(vm);

		ucv_array_push(r, tensor_wrap(vm, std::move(g), 0));
		ucv_array_push(r, tensor_wrap(vm, std::move(beta), 0));

		return r;
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

/*
 * kda_out_norm(core, z, weight, eps): torch.tensor
 *
 * core (S, H, dv) fp32, z (S, H*dv) fp32 gate logits, weight (dv) fp16 or
 * fp32. Returns (1, S, H*dv) fp16 = weight * rmsnorm(core) * sigmoid(z).
 */
static uc_value_t *
torch_kda_out_norm(uc_vm_t *vm, size_t nargs)
{
	double eps;

	if (nargs != 4 || !arg_number(vm, uc_fn_arg(3), &eps)) {
		uc_vm_raise_exception(vm, EXCEPTION_TYPE,
			"expected (core, z, weight, eps)");
		return NULL;
	}

	at::Tensor *core = f32_arg(vm, uc_fn_arg(0), "kda_out_norm: core");
	at::Tensor *z = core ? f32_arg(vm, uc_fn_arg(1), "kda_out_norm: z") : NULL;
	at::Tensor *w = z ? tensor_arg(vm, uc_fn_arg(2)) : NULL;

	if (!w)
		return NULL;

	bool wh = w->scalar_type() == at::kHalf;

	if (core->dim() != 3 || z->numel() != core->numel() ||
	    w->numel() != core->size(2) || !w->is_cuda() || !w->is_contiguous() ||
	    (!wh && w->scalar_type() != at::kFloat))
		RAISE("kda_out_norm: incompatible shapes/dtypes");

	TensorDeviceGuard dg(*core);

	try {
		int64_t S = core->size(0), H = core->size(1), dv = core->size(2);
		at::Tensor out = at::empty({ 1, S, H * dv },
			core->options().dtype(at::kHalf));

		int rc = amp_kda_out_norm(core->data_ptr(), z->data_ptr(),
			w->data_ptr(), wh ? 1 : 0, out.data_ptr(), (long)(S * H), (int)dv,
			(float)eps, c10::cuda::getCurrentCUDAStream().stream());

		if (!launch_ok(vm, rc, "kda_out_norm"))
			return NULL;

		return tensor_wrap(vm, std::move(out), 0);
	}
	catch (const std::exception &e) {
		RAISE(e.what());
	}
}

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
	{ "view_dtype",	tensor_view_dtype },
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
	{ "index_add_",	tensor_index_add_ },
	{ "add",	tensor_add },
	{ "add_",	tensor_add_ },
	{ "mul",	tensor_mul },
	{ "mul_",	tensor_mul_ },
	{ "div",	tensor_div },
	{ "sub",	tensor_sub },
	{ "pow",	tensor_pow },
	{ "clamp",	tensor_clamp },
	{ "sqrt",	tensor_sqrt },
	{ "abs",	tensor_abs },
	{ "rsqrt",	tensor_rsqrt },
	{ "neg",	tensor_neg },
	{ "le",	tensor_le },
	{ "exp",	tensor_exp },
	{ "log",	tensor_log },
	{ "cos",	tensor_cos },
	{ "sin",	tensor_sin },
	{ "silu",	tensor_silu },
	{ "sigmoid",	tensor_sigmoid },
	{ "gelu_tanh",	tensor_gelu_tanh },
	{ "gelu",	tensor_gelu },
	{ "softplus",	tensor_softplus },
	{ "softmax",	tensor_softmax },
	{ "cumsum",	tensor_cumsum },
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
	{ "pinned_empty",	torch_pinned_empty },
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
	{ "idx_score",	torch_idx_score },
	{ "dsa_select",	torch_dsa_select },
	{ "mem_info",	torch_mem_info },
	{ "device_props",	torch_device_props },
	{ "empty_cache",	torch_empty_cache },
	{ "kv_q8_put",	torch_kv_q8_put },
	{ "kv_q8_get",	torch_kv_q8_get },
	{ "kv_q4_put",	torch_kv_q4_put },
	{ "kv_q4_get",	torch_kv_q4_get },
	{ "layer_norm",	torch_layer_norm },
	{ "hc_mix",	torch_hc_mix },
	{ "hc_apply",	torch_hc_apply },
	{ "gr_mix",	torch_gr_mix },
	{ "qsa_attn",	torch_qsa_attn },
	{ "ngram_hash",	torch_ngram_hash },
	{ "ngram_dequant",	torch_ngram_dequant },
	{ "ple_forward",	torch_ple_forward },
	{ "image_prepare",	torch_image_prepare },
	{ "gdn_scan",	torch_gdn_scan },
	{ "kda_scan",	torch_kda_scan },
	{ "kda_gates",	torch_kda_gates },
	{ "kda_out_norm",	torch_kda_out_norm },
	{ "flash_attn_kvcache",	torch_flash_attn_kvcache },
	{ "p2p_comm",	torch_p2p_comm },
	{ "device_count",	torch_device_count },
	{ "common_prefix",	torch_common_prefix },
	{ "copy_many",	torch_copy_many },
	{ "copy_async",	torch_copy_async },
	{ "copy_h2d_async",	torch_copy_h2d_async },
	{ "fd_write",	torch_fd_write },
	{ "fd_read",	torch_fd_read },
	{ "fd_send_all",	torch_fd_send_all },
	{ "fd_recv_all",	torch_fd_recv_all },
	{ "moe_coop",	torch_moe_coop },
	{ "skinny",	torch_skinny },
	{ "fp8_gemv",	torch_fp8_gemv },
	{ "v41_attn",	torch_v41_attn },
	{ "fp8_dequant",	torch_fp8_dequant },
	{ "fp8_gemv_grouped",	torch_fp8_gemv_grouped },
	{ "hc_mix2",	torch_hc_mix2 },
	{ "gather_wait",	torch_gather_wait },
	{ "v41_rope",	torch_v41_rope },
	{ "v41_route",	torch_v41_route },
	{ "moe_prep",	torch_moe_prep },
	{ "fp4_dequant",	torch_fp4_dequant },
	{ "fp4_moe_gemv",	torch_fp4_moe_gemv },
	{ "stream_side",	torch_stream_side },
	{ "stream_main",	torch_stream_main },
	{ "stream_wait",	torch_stream_wait },
	{ "expert_exchange",	torch_expert_exchange },
	{ "moe_cpu_threads",	torch_moe_cpu_threads },
	{ "moe_cpu_prof",	torch_moe_cpu_prof },
	{ "moe_fused",	torch_moe_fused },
	{ "moe_gather",	torch_moe_gather },
	{ "moe_concurrency",	torch_moe_concurrency },
	{ "notifier",	torch_notifier },
	{ "attn_decode",	torch_attn_decode },
	{ "qk_norm_rope",	torch_qk_norm_rope },
	{ "dsv4_qkv",	torch_dsv4_qkv },
	{ "dsv4_comp",	torch_dsv4_comp },
	{ "dsv4_attn",	torch_dsv4_attn },
	{ "dsv4_derot",	torch_dsv4_derot },
	{ "dsv4_mask",	torch_dsv4_mask },
	{ "image_info",	torch_image_info },
	{ "attn_gate",	torch_attn_gate },
	{ "gdn_chunk",	torch_gdn_chunk },
	{ "gdn_prep",	torch_gdn_prep },
	{ "gdn_gates",	torch_gdn_gates },
	{ "gdn_ab_gates",	torch_gdn_ab_gates },
	{ "dflash_conv",	torch_dflash_conv },
	{ "dflash_select",	torch_dflash_select },
	{ "rms_norm_f32",	torch_rms_norm_f32 },
	{ "norm32",	torch_norm32 },
	{ "dflash_conv_res",	torch_dflash_conv_res },
	{ "gdn_out_norm",	torch_gdn_out_norm },
	{ "flash_attn",	torch_flash_attn },
	{ "rms_norm_res",	torch_rms_norm_res },
	{ "decode_attn",	torch_decode_attn },
	{ "argmax_advance",	torch_argmax_advance },
	{ "mgemm",	torch_mgemm },
	{ "ptr_array",	torch_ptr_array },
	{ "h2d",		torch_h2d },
	{ "h2d_file",	torch_h2d_file },
	{ "read_into",	torch_read_into },
	{ "profile",	torch_profile },
	{ "moe_cpu_make_layer",	torch_moe_cpu_make_layer },
	{ "moe_cpu_free_layer",	torch_moe_cpu_free_layer },
	{ "moe_cpu_forward",	torch_moe_cpu_forward },
	{ "moe_cpu_enqueue",	torch_moe_cpu_enqueue },
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
	uc_type_declare(vm, COMM_TYPE, comm_methods, comm_free);
	uc_type_declare(vm, NOTIFIER_TYPE, notifier_methods, notifier_free);
	uc_function_list_register(scope, torch_fns);
	amp_tokenizer_init(vm, scope);
}