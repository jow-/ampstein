// tests/v41_route.uc — torch.v41_route (fused sqrt-softplus router) against the composite form.

import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

function check(T, E, k, scale) {
	let lg = torch.rand([T, E], 'float32', dv).sub(0.5).mul(12.0);
	let bias = torch.rand([1, E], 'float32', dv).sub(0.5).mul(0.2);
	let r = torch.v41_route(lg, bias, k, scale);
	let sc = lg.softplus().sqrt();
	let sel = sc.add(bias).topk(k, -1).indices;
	let sw = sc.gather(1, sel);
	let w = sw.div(sw.sum(1, true).add(1e-20)).mul(scale);
	let same = r[0].to('float32').sub(sel.to('float32')).abs().max(-1).max(-1).item();
	let dw = r[1].sub(w).abs().max(-1).max(-1).item();

	printf('T=%d E=%d k=%d: index mismatch %g, weight diff %.1e\n', T, E, k, same, dw);
	assert(same == 0 && dw < 1e-6, 'fused router differs from the composite form');
}

check(1, 384, 6, 2.5);
check(5, 384, 6, 2.5);
check(300, 384, 6, 1.5);
check(5, 128, 3, 2.5);
printf('v41_route: all ok\n');
