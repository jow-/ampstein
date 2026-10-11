// tests/exl3_gemv_rows.uc — EXL3 small-m products (MTP verify windows:
// m = 1..4 rows) on real qwen35 linears: every m checked against the
// dequantize + cuBLAS reference (torch.exl3_gemm_recon), and timed.
// Measured (A5000, 6 bpw): 4 rows cost ~1.4x one row; an int8 per-slice
// kernel for 3-4 rows was slower than the tensor-core path (docs/LESSONS.md
// L70). Model: $AMP_MODEL_QWEN35.

import * as weights from 'ampstein.weights';
import * as models from 'ampstein.models';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

if (!stat(MODEL_DIR) || torch.device_count() < 1) {
	warn('skip: model dir not found or no GPU\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

let ws = weights.open(MODEL_DIR);
let L = 'model.language_model.layers.';
let keys = [
	L + '8.mlp.gate_proj', L + '8.mlp.down_proj',
	L + '8.linear_attn.in_proj_qkv', L + '8.linear_attn.out_proj',
	L + '11.self_attn.q_proj', L + '11.self_attn.o_proj'
];

let total = [ 0, 0, 0, 0 ];

for (let key in keys) {
	let lin = models.linear(ws, key, true, 0);

	if (!lin || !lin.trellis)
		continue;

	let line = sprintf('%-46s K=%d %5dx%-5d', substr(key, length(L)), lin.K, lin.in_features, lin.out_features);

	for (let m = 1; m <= 4; m++) {
		let x = torch.arange(m * lin.in_features, 'float32', 'cuda:0').mul(0.0137).sin()
			.to('float16').view([m, lin.in_features]);
		let ref = torch.exl3_gemm_recon(x, lin.trellis, lin.suh, lin.svh, lin.K, lin.mul1, lin.mcg);
		let y = lin.forward(x);
		let rf = ref.to('float32');
		let rel = math.sqrt(y.to('float32').sub(rf).pow(2).mean().item() / rf.pow(2).mean().item());

		assert(rel < 0.03, `${key} m=${m}: rel-rms ${rel} vs reference`);

		for (let i = 0; i < 5; i++)
			lin.forward(x);

		torch.synchronize();

		let t0 = now_ms(), R = 200;

		for (let i = 0; i < R; i++)
			lin.forward(x);

		torch.synchronize();

		let us = (now_ms() - t0) * 1000 / R;

		total[m - 1] += us;
		line += sprintf('  m%d %6.1f us (%.0e)', m, us, rel);
	}

	print(line, '\n');
}

printf('sum over tensors: m1 %.1f  m2 %.1f  m3 %.1f  m4 %.1f us\n',
	total[0], total[1], total[2], total[3]);
print('exl3_gemv_rows: PASS\n');
