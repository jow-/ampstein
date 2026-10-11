// Phase 1 exit gate: 2-layer MLP forward pass on the GPU, written in pure
// ucode against the torch module.
//
//   SwiGLU MLP (llama-style):
//     h  = x @ W_gate.T  (gate projection)
//     u  = x @ W_up.T    (up projection)
//     act = silu(h) * u  (elementwise)
//     y  = act @ W_down.T (down projection)
//
// Run:
//   ucode -L <build-dir> examples/mlp.uc

let torch = require('ampstein.torch');
global.torch = torch;

function sqrt(x) {
	// Newton iteration; the MLP gate only needs a rough magnitude check
	let r = x > 1.0 ? x : 1.0;

	for (let i = 0; i < 40; i++)
		r = (r + x / r) / 2.0;

	return r;
}

const D_MODEL = 256, D_FF = 1024, BATCH = 8;

// deterministic pseudo-random weight init (Xavier-ish scale)
function randn(rows, cols, scale) {
	let t = torch.zeros([rows, cols]);
	let data = [];
	let seed = 12345;

	for (let i = 0; i < rows * cols; i++) {
		// xorshift-ish LCG; sum of 3 uniforms approximates a gaussian
		seed = (seed * 1103515245 + 12345) % 2147483648;
		let u1 = seed / 2147483648.0;
		seed = (seed * 1103515245 + 12345) % 2147483648;
		let u2 = seed / 2147483648.0;
		seed = (seed * 1103515245 + 12345) % 2147483648;
		let u3 = seed / 2147483648.0;
		push(data, (u1 + u2 + u3 - 1.5) * 2.0 * scale);
	}

	// fill_ only writes scalars; build via reshape of a 1-D tensor
	let flat = torch.from_array(data);
	return flat.reshape([rows, cols]);
}

// --- weights (CPU, then moved to GPU once) ---------------------------------

let W_gate = randn(D_FF, D_MODEL, 0.02).cuda();
let W_up   = randn(D_FF, D_MODEL, 0.02).cuda();
let W_down = randn(D_MODEL, D_FF, 0.02).cuda();

// --- input ------------------------------------------------------------------

let x = randn(BATCH, D_MODEL, 1.0).cuda();

// --- forward ------------------------------------------------------------------

function mlp(x) {
	let h = torch.matmul(x, W_gate.t());
	let u = torch.matmul(x, W_up.t());
	let act = h.silu().mul(u);
	return torch.matmul(act, W_down.t());
}

let y = mlp(x);

// --- checks ---------------------------------------------------------------------

let ys = y.shape();
assert(ys[0] == BATCH && ys[1] == D_MODEL, 'output shape');
assert(y.dtype() == 'float32', 'output dtype');

let rows = y.cpu().tolist();
let sum = 0.0;

for (let i = 0; i < BATCH; i++)
	for (let j = 0; j < D_MODEL; j++)
		sum += rows[i][j] * rows[i][j];

let rms = sqrt(sum / (BATCH * D_MODEL));

// a sane forward pass: activations neither exploded nor vanished
assert(rms > 1e-4 && rms < 100.0, `output RMS in sane range, got ${rms}`);

// determinism: same input, same output
let y2 = mlp(x);
let r2 = y2.cpu().tolist();

for (let i = 0; i < BATCH; i++)
	for (let j = 0; j < D_MODEL; j++)
		assert(rows[i][j] == r2[i][j], 'deterministic forward');

printf('mlp: 2-layer SwiGLU forward on GPU ok (output RMS %.4f)\n', rms);