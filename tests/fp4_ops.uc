// tests/fp4_ops.uc — e2m1 block-scaled weights (DSpark drafter experts): torch.fp4_dequant and
// torch.fp4_moe_gemv against a composite reference built from the same nibbles.

import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

const LUT = [ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 ];

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

/* {w: (N, K/2) int8, s: (N, K/32) e8m0, ref: (N, K) fp32}; nibbles from a deterministic pattern */
function make(N, K, seed) {
	let vals = [], bytes = [];

	for (let n = 0; n < N; n++) {
		for (let k = 0; k < K; k += 2) {
			let lo = (n * 7 + k * 3 + seed + int(k / 32)) % 16, hi = (n * 5 + k * 11 + seed * 3 + 1) % 16;
			let b = lo + 16 * hi;

			push(bytes, b > 127 ? b - 256 : b);
			push(vals, LUT[lo], LUT[hi]);
		}
	}

	let sv = [];

	for (let i = 0; i < N * (K / 32); i++)
		push(sv, math.pow(2.0, (i * 5 + seed) % 6 - 4));

	let w = torch.from_array(bytes, 'int8', dv).view([N, int(K / 2)]);
	let s = torch.from_array(sv, 'float32', dv).view([N, int(K / 32)]).to('float8_e8m0fnu').contiguous();
	let v = torch.from_array(vals, 'float32', dv).view([N, K]);
	let ref = v.view([N, int(K / 32), 32]).mul(s.to('float32').view([N, int(K / 32), 1])).view([N, K]);

	return {w: w, s: s, ref: ref};
}

function check(N, K, E, R, topk) {
	let ex = [];

	for (let e = 0; e < E; e++)
		push(ex, make(N, K, e));

	let wd = torch.fp4_dequant(ex[0].w, ex[0].s);

	assert(rel_rms(wd, ex[0].ref) < 1e-6, `dequant ${N}x${K}`);

	let wp = torch.from_array(map(ex, (e) => e.w.data_ptr()), 'int64', dv);
	let sp = torch.from_array(map(ex, (e) => e.s.data_ptr()), 'int64', dv);
	let ids = [];

	for (let i = 0; i < R * topk; i++)
		push(ids, (i * 7 + 3) % E);

	let x = torch.rand([R, K], 'float16', dv).sub(0.5).mul(4.0);
	let y = torch.fp4_moe_gemv(x, torch.from_array(ids, 'int64', dv), wp, sp, N, topk);

	for (let i = 0; i < R * topk; i++) {
		let yr = torch.matmul(x.narrow(0, int(i / topk), 1).to('float32'), ex[ids[i]].ref.t());
		let r = rel_rms(y.narrow(0, i, 1), yr);

		assert(r < 1e-4, `moe_gemv ${N}x${K} slot ${i}: rel rms ${r}`);
	}

	printf('ok %dx%d E=%d slots=%d\n', N, K, E, R * topk);
}

check(96, 5120, 4, 5, 3);
check(160, 2304, 4, 15, 1);
printf('fp4_ops: all ok\n');
