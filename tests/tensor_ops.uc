// Phase 1 test: tensor API surface — constructors, views, elementwise ops,
// reductions, structural ops, integer dtypes.
// Run via CTest or manually:
//   ucode -L <build-dir> tests/tensor_ops.uc

function abs(x) {
	return x < 0 ? -x : x;
}

// --- constructors ---------------------------------------------------------

let a = torch.from_array([[1.0, 2.0], [3.0, 4.0]]);
let b = torch.from_array([[10.0, 20.0], [30.0, 40.0]]);

assert(a.shape()[0] == 2 && a.shape()[1] == 2, 'from_array shape');
assert(a.numel() == 4, 'numel');
assert(a.dtype() == 'float32', 'from_array dtype');

let z = torch.zeros([2, 3], 'float16');
assert(z.dtype() == 'float16' && z.numel() == 6, 'zeros dtype/numel');
assert(z.tolist()[1][2] == 0.0, 'zeros content');

let o = torch.ones([3]);
assert(o.tolist()[2] == 1.0, 'ones content');

let f = torch.full([2, 2], 2.5, 'float16');
assert(f.tolist()[0][1] == 2.5, 'full content');

let ar = torch.arange(5);
assert(ar.tolist()[4] == 4.0, 'arange content');

// --- elementwise arithmetic ------------------------------------------------

assert(a.add(1.5).tolist()[0][0] == 2.5, 'add scalar');
assert(a.mul(b).tolist()[1][1] == 160.0, 'mul tensor');
assert(a.div(2.0).tolist()[1][0] == 1.5, 'div scalar');
assert(a.sub(0.5).tolist()[0][1] == 1.5, 'sub scalar');
assert(a.add(a).tolist()[1][1] == 8.0, 'add tensor');
assert(torch.matmul(a, b).tolist()[1][1] == 220.0, 'matmul');

// in-place
let x = torch.zeros([2, 2]);
x.copy_(torch.ones([2, 2]));
x.add_(1.0);
x.mul_(2.0);
assert(x.tolist()[1][0] == 4.0, 'copy_/add_/mul_ chain');

// fill_ still works
let y = torch.zeros([2]);
y.fill_(3.0);
assert(y.tolist()[1] == 3.0, 'fill_');

// --- unary math -------------------------------------------------------------

assert(torch.from_array([4.0]).sqrt().tolist()[0] == 2.0, 'sqrt');
assert(torch.from_array([4.0]).rsqrt().tolist()[0] == 0.5, 'rsqrt');
assert(torch.from_array([2.0, -3.0]).neg().tolist()[1] == 3.0, 'neg');
assert(torch.from_array([0.0]).exp().tolist()[0] == 1.0, 'exp');
assert(torch.from_array([1.0]).log().tolist()[0] == 0.0, 'log');
assert(torch.from_array([2.0, 4.0]).pow(2).tolist()[1] == 16.0, 'pow');

// silu(x) = x * sigmoid(x)
let s = torch.from_array([0.5, -0.5]).silu().tolist();
assert(abs(s[0] - 0.31123) < 1e-4 && abs(s[1] + 0.18877) < 1e-4, 'silu');

// --- reductions --------------------------------------------------------------

let sm = a.mul(0.5).softmax(1).sum(1).tolist();
assert(abs(sm[0] - 1.0) < 1e-5 && abs(sm[1] - 1.0) < 1e-5, 'softmax rows sum to 1');

assert(a.sum().tolist() == 10.0, 'sum all');
assert(a.mean().tolist() == 2.5, 'mean all');
assert(a.sum(1).tolist()[0] == 3.0, 'sum dim');

// --- topk ---------------------------------------------------------------------

let tk = torch.from_array([3.0, 1.0, 4.0, 1.5]).topk(2);
assert(tk.values.tolist()[0] == 4.0 && tk.values.tolist()[1] == 3.0, 'topk values');
assert(tk.indices.tolist()[0] == 2 && tk.indices.tolist()[1] == 0, 'topk indices');

// --- views and shape ops --------------------------------------------------------

assert(a.transpose(0, 1).tolist()[0][1] == 3.0, 'transpose');
assert(a.t().tolist()[1][0] == 2.0, 't');
assert(a.narrow(1, 0, 1).tolist()[1][0] == 3.0, 'narrow');
assert(a.view(4).tolist()[3] == 4.0, 'view');
assert(a.reshape(1, 4).shape()[0] == 1 && a.reshape(1, 4).shape()[1] == 4, 'reshape');
assert(a.unsqueeze(0).shape()[0] == 1, 'unsqueeze');
assert(torch.stack([a, a], 0).shape()[0] == 2, 'stack');
assert(torch.cat([a, b], 1).tolist()[0][3] == 20.0, 'cat dim1');

// transposed views need contiguous() before kernels that require it
let tc = a.t().contiguous();
assert(tc.tolist()[0][1] == 3.0, 'contiguous after transpose');

// --- dtype casts -----------------------------------------------------------------

assert(a.to('float16').dtype() == 'float16', 'to float16');
assert(a.to('int32').tolist()[1][1] == 4, 'to int32 value');
assert(a.to('float64').dtype() == 'float64', 'to float64');

// --- integer tensors: exact ints ---------------------------------------------------

let idx = torch.arange(5, 'int64');
assert(idx.tolist()[3] == 3, 'int64 tolist exact');
assert(idx.dtype() == 'int64', 'int64 dtype');

// --- device round-trip --------------------------------------------------------------

let g = a.cuda().mul(2.0).cpu();
assert(g.tolist()[1][1] == 8.0, 'cuda round-trip with op');

// --- chained pipeline (llama-ish): rms-norm-ish + silu-mlp ---------------------------

let h = torch.from_array([[1.0, 2.0, 3.0, 4.0]]);
let normed = h.mul(h.pow(2).mean(1, true).add(1e-5).rsqrt());
assert(normed.shape()[0] == 1 && normed.shape()[1] == 4, 'rms-norm shape');
assert(abs(normed.tolist()[0][0] - 0.36515) < 1e-3, 'rms-norm value');

printf('tensor_ops: all assertions passed\n');