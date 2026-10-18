// tests/v41_rope.uc — torch.v41_rope (fused rotary embedding of V4.1 rows) against the composite rope_il form,
// with int64 positions read through a strided column view (the decode meta rows) and fp32 positions.

import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

function max_abs(a, b) {
	return a.to('float32').sub(b.to('float32')).abs().max(-1).max(-1).max(-1).item();
}

function check(T, H, D, rd, theta) {
	let inv = [];

	for (let i = 0; i < rd / 2; i++)
		push(inv, math.pow(theta, -2.0 * i / rd));

	let inv_t = torch.from_array(inv, 'float32', dv);
	let x = torch.rand([T, H, D], 'float16', dv).sub(0.5).mul(8.0);
	let meta = [], pf = [];

	for (let t = 0; t < T; t++) {
		let p = 70000 + t * 3;

		push(meta, p, 0, 0, 0, 0, 0, 0, 0);
		push(pf, p * 1.0);
	}

	let mt = torch.from_array(meta, 'int64', dv).view([T, 8]);
	let col = mt.narrow(1, 0, 1);                                  // (T, 1), element stride 8
	let ref = dsv41.rope_il(x.to('float32'), dsv41.rope_cs_rows(inv_t, col.to('float32')), rd).to('float16');
	let a = torch.v41_rope(x, col, inv_t);
	let b = torch.v41_rope(x, torch.from_array(pf, 'float32', dv).view([T, 1]), inv_t);
	let e = [ max_abs(a, ref), max_abs(b, ref) ];

	printf('T=%d H=%d D=%d rd=%d: max abs diff %.2e (int64 col) %.2e (fp32)\n', T, H, D, rd, ...e);
	assert(e[0] < 2e-2 && e[1] < 2e-2, 'fused rope differs from the composite form');
}

check(1, 64, 512, 64, 10000.0);
check(5, 64, 512, 64, 160000.0);
check(5, 32, 128, 64, 160000.0);
check(7, 1, 512, 64, 10000.0);
printf('v41_rope: all ok\n');
