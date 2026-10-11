// Phase 0.2 test: tensor resource type basics.
// Run via CTest (requires UCODE_EXECUTABLE) or manually:
//   ucode -L <build-dir> tests/tensor_basics.uc

assert(length(torch.version()) > 0, 'version()');

// from_array: 1-D
let t = torch.from_array([1.0, 2.0, 3.0, 4.0]);
assert(length(t.shape()) == 1 && t.shape()[0] == 4, '1-D shape');
assert(t.dtype() == 'float32', 'dtype');
assert(t.tolist()[2] == 3.0, '1-D tolist');

// from_array: 2-D
let m = torch.from_array([[1.0, 2.0], [3.0, 4.0]]);
assert(length(m.shape()) == 2 && m.shape()[0] == 2 && m.shape()[1] == 2, '2-D shape');
assert(m.tolist()[1][0] == 3.0, '2-D tolist');

// zeros + fill_ chaining
let f = torch.zeros([2, 3]).fill_(7);
assert(f.tolist()[0][2] == 7.0 && f.tolist()[1][1] == 7.0, 'zeros+fill_');

// item()
assert(torch.from_array([42.0]).item() == 42, 'item()');

// device round-trip CPU -> CUDA -> CPU
let g = t.cuda();
assert(g.dtype() == 'float32' && g.tolist()[3] == 4.0, 'cuda()');
let c = g.cpu();
assert(c.tolist()[0] == 1.0, 'cpu() round-trip');

// explicit free, use-after-free raises, double free safe
let x = torch.from_array([1.0]);
x.free();
let caught = false;

try {
    x.item();
} catch (e) {
    caught = true;
}

assert(caught, 'use-after-free raises');
x.free();

// GC churn: finalizers must not corrupt the heap
for (let i = 0; i < 2000; i++)
    torch.from_array([i, i + 1, i + 2]);

printf('tensor_basics: all assertions passed\n');