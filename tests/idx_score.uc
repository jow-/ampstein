// tests/idx_score.uc — the fused DSA indexer scoring kernel (torch.idx_score)
// against an fp32 composite reference: ragged shapes (R not a multiple of 8,
// n_pools not a multiple of 64), the visibility mask, wrong-argument errors;
// a device-side position (int64 tensor, the CUDA-graph decode form) must give
// the same result bit for bit, including tiles skipped as all-mask.
import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);

function noise(rows, cols, lo, hi, dtype) {
	let a = [];

	for (let r = 0; r < rows; r++) {
		let row = [];

		for (let c = 0; c < cols; c++)
			push(row, math.rand(lo, hi));

		push(a, row);
	}

	return torch.from_array(a, 'float32').to(dtype).cuda(DEV);
}

function raises(f) {
	try { f(); } catch (e) { return true; }

	return false;
}

function check(R, n, pos0, P, HI) {
	HI = HI || 32;

	let q = noise(R * HI, 128, -1.0, 1.0, 'float16').view([R, HI, 128]);
	let pk = noise(n, 128, -1.0, 1.0, 'float16');
	let w = noise(R, HI, 0.0, 0.3 * 32 / HI, 'float32');
	let scale = math.pow(128.0, -0.5);
	let out = torch.idx_score(q, pk, w, scale, pos0, P);

	assert(out.shape()[0] == R && out.shape()[1] == n, 'output shape');

	// reference in fp32
	let s = torch.matmul(q.to('float32').reshape([R * HI, 128]), pk.to('float32').t())
		.view([R, HI, n]).clamp(0, null).mul(scale);
	let ref = torch.matmul(w.view([R, 1, HI]), s).view([R, n]);

	// visibility: pool p of query r selectable when p * P + P - 1 <= pos0 + r
	let pend = torch.arange(n, 'float32', 'cuda:' + DEV).mul(P * 1.0).add(P - 1.0);
	let qpos = torch.arange(R, 'float32', 'cuda:' + DEV).add(pos0 * 1.0).view([R, 1]);
	let ok = pend.view([1, n]).le(qpos);
	let got = out.to('float32');
	let okf = ok.to('float32');
	let diff = got.sub(ref).abs().mul(okf).max().item();
	let amax = ref.abs().mul(okf).max().item();
	let masked = got.add(65504.0).abs().mul(okf.neg().add(1.0)).max().item();
	let nvalid = okf.sum().item();

	printf('H=%d R=%d n=%d pos0=%d: valid %d/%d, max |err| %.5f (max |ref| %.3f), masked-entry error %.1f\n',
		HI, R, n, pos0, nvalid, R * n, diff, amax, masked);

	assert(diff <= 0.003 * amax + 1e-3, 'scores match the fp32 reference');
	assert(masked == 0.0, 'invisible pools carry -65504');

	let pt = torch.from_array([ pos0 ], 'int64', 'cuda:' + DEV);
	let dout = torch.idx_score(q, pk, w, scale, pt, P);

	assert(dout.sub(out).abs().max().item() == 0, 'a device-side pos0 gives the same scores');
}

check(40, 1000, 3000, 4);     // ragged R and n_pools, part of the pools masked
check(32, 4096, 100000, 4);   // everything visible
check(5, 70, 100, 4);         // tiny, most masked
check(1, 64, 300, 4);         // single query, exactly one tile
check(40, 1000, 3000, 4, 64); // DeepSeek-V4: 64 indexer heads in one pass
check(7, 4100, 100000, 4, 64);
check(5, 300, 700, 128, 64);  // HCA-sized pool rate
check(9, 130, 500, 4, 96);    // 3 head groups
check(4, 65537, 3, 4);        // GLM decode at ctx 262144: almost every tile skipped
check(4, 65537, 87000, 4);    // a third of the pool capacity visible
check(4, 65537, 262140, 4);   // the whole capacity

let q = noise(8 * 32, 128, -1.0, 1.0, 'float16').view([8, 32, 128]);
let pk = noise(100, 128, -1.0, 1.0, 'float16');
let w = noise(8, 32, 0.0, 0.3, 'float32');

assert(raises(() => torch.idx_score(q, pk, w.to('float16'), 0.1, 0, 4)), 'w must be fp32');
assert(raises(() => torch.idx_score(q.view([8 * 32, 128]), pk, w, 0.1, 0, 4)), 'q must be 3-D');
assert(raises(() => torch.idx_score(q, pk.narrow(1, 0, 64), w, 0.1, 0, 4)), 'pool dim must be 128');
assert(raises(() => torch.idx_score(q, pk, w, 0.1, 0, 0)), 'P must be positive');
assert(raises(() => torch.idx_score(q, pk, w, 0.1, torch.from_array([ 5 ], 'int32', 'cuda:' + DEV), 4)), 'a pos0 tensor must be int64');
assert(raises(() => torch.idx_score(q, pk, w, 0.1, torch.from_array([ 5, 6 ], 'int64', 'cuda:' + DEV), 4)), 'a pos0 tensor has one element');
assert(raises(() => torch.idx_score(noise(8 * 48, 128, -1.0, 1.0, 'float16').view([8, 48, 128]), pk, noise(8, 48, 0.0, 0.3, 'float32'), 0.1, 0, 4)), 'heads must be a multiple of 32');

print('idx_score: ok\n');
