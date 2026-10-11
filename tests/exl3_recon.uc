// Test: torch.exl3_gemm_recon (dequantize-then-cuBLAS prefill path) vs
// torch.exl3_gemm (on-the-fly trellis decode) on a real EXL3 layer
// (fixture from tests/oracle/make_exl3_real_fixture.py: Llama-3.2-1B
// layer-0 q_proj, 2048x2048, K=4). Both recon paths are exercised:
// rows < 1024 (input/output Hadamard around a rotated-basis weight) and
// rows >= 1024 (fused original-basis reconstruct).
//
//   ucode -L <build-dir> tests/exl3_recon.uc

import { stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/exl3_real_layer.safetensors`;

if (!stat(FIXTURE)) {
	warn('skip: fixture not found: ' + FIXTURE + '\n');
	exit(125);
}

let f = torch.st(FIXTURE);
let trellis = f.tensor('trellis').cuda();
let suh = f.tensor('suh').cuda();
let svh = f.tensor('svh').cuda();
let X = f.tensor('X').cuda();                 // (m, 2048) fp16
let K = trellis.shape()[2] / 16.0;
let k = trellis.shape()[0] * 16, n = trellis.shape()[1] * 16;

/* rows x k input built by tiling the fixture's activations */
function input(rows) {
	let m = X.shape()[0];
	let parts = [];

	for (let r = 0; r < rows; r += m)
		push(parts, X.narrow(0, 0, (rows - r < m) ? rows - r : m));

	return torch.cat(parts, 0).contiguous();
}

for (let rows in [ 200, 1100 ]) {
	let x = input(rows);
	let ref = torch.zeros([rows, n], 'float16', 'cuda');

	torch.exl3_gemm(x, trellis, ref, suh, svh);

	let y = torch.exl3_gemm_recon(x, trellis, suh, svh, K);
	let rf = ref.to('float32'), yf = y.to('float32');
	let rel = yf.sub(rf).pow(2).mean().sqrt().item() /
		rf.pow(2).mean().sqrt().item();

	assert(rel < 5e-3, `rows ${rows}: rel-rms ${rel} vs exl3_gemm`);
	printf('exl3_gemm_recon rows %d (%s path): rel-rms %.2e vs exl3_gemm\n',
		rows, rows >= 1024 ? 'fused' : 'had', rel);
}

print('exl3_recon: PASS\n');
