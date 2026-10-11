// Phase 0.3 test: safetensors mmap reader + zero-copy tensor views.
// Fixture: tests/fixtures/st_test.safetensors (deterministic; regenerate
// with `.venv/bin/python tests/oracle/make_fixture.py`).
// Run via CTest or manually:
//   ucode -L <build-dir> tests/st_reader.uc

const FIXTURE = `${sourcepath(0, true)}/fixtures/st_test.safetensors`;

let f = torch.st(FIXTURE);

// tensor inventory
let names = f.list();
assert(length(names) == 3, 'tensor count');
assert(index(names, 'w') >= 0 && index(names, 'b') >= 0 &&
       index(names, 'flag') >= 0, 'tensor names');

// header info
let wi = f.info('w');
assert(wi.dtype == 'F16', 'w dtype');
assert(wi.shape[0] == 4 && wi.shape[1] == 8, 'w shape');
assert(wi.nbytes == 64, 'w nbytes');

// zero-copy views: values must match the oracle (bit-exact fp16)
let w = f.tensor('w');
assert(w.dtype() == 'float16' && w.shape()[0] == 4 && w.shape()[1] == 8, 'w tensor meta');

let row0 = w.tolist()[0];
let expect = [-2.0, -1.87109375, -1.7421875, -1.61328125,
              -1.484375, -1.35546875, -1.2255859375, -1.0966796875];

for (let i = 0; i < 8; i++)
    assert(row0[i] == expect[i], `w[0][${i}] == ${expect[i]}`);

// integer dtypes
let b = f.tensor('b');
let bl = b.tolist();

for (let i = 0; i < 8; i++)
    assert(bl[i] == i, `b[${i}] == ${i}`);

let flag = f.tensor('flag');
assert(flag.tolist()[2] == 1.0, 'flag[2]');

// view survives GC pressure (stfile pinned via resource value slot)
for (let i = 0; i < 500; i++)
    torch.from_array([i, i + 1]);

assert(w.tolist()[0][7] == expect[7], 'w view valid after GC churn');

// device round-trip from a mmap view
let wg = w.cuda();
assert(wg.tolist()[1][0] == -0.9677734375, 'cuda() from mmap view');

// free semantics
let x = f.tensor('b');
x.free();

let caught = false;

try {
    x.tolist();
} catch (e) {
    caught = true;
}

assert(caught, 'freed view raises');

// stfile free + use-after-free
f.free();
caught = false;

try {
    f.list();
} catch (e) {
    caught = true;
}

assert(caught, 'freed stfile raises');

printf('st_reader: all assertions passed\n');