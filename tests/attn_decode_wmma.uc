// Test: the WMMA decode attention kernel (ROCm; head dim 256, up to 32 query rows per kv head) against the scalar split-KV kernel, for the fp16
// cache (torch.attn_decode, mma = 1 vs 0) and the Q8 cache (torch.attn_decode_q8, mma = 1 vs 0): S = 1 .. 5, key counts around the 32-key tile
// and the split boundaries, a causal row limit per query row, run-to-run determinism. Synthetic data. Skipped (exit 125) where the kernel is not built.
//
//   ucode -L <build-dir> -L <src>/ucode tests/attn_decode_wmma.uc

import * as math from 'math';

if (torch.device_props(0).backend != 'rocm') {
	print('attn_decode_wmma: skipped (ROCm only)\n');
	exit(125);
}

let seed = 7;

function rnd() {
	seed = (seed * 1103515245 + 12345) % 2147483648;

	return (seed / 1073741824.0) - 1.0;
}

function tensor(shape, lo, hi) {
	let n = 1;

	for (let d in shape)
		n *= d;

	let vals = [];

	for (let i = 0; i < n; i++)
		push(vals, lo + (rnd() + 1.0) * 0.5 * (hi - lo));

	return torch.from_array(vals).view(shape).to('float16').cuda(0);
}

function maxdiff(a, b) {
	return a.to('float32').sub(b.to('float32')).abs().max().item();
}

const E = 256, Hk = 4, Hq = 24, max_seq = 2048;
let K = tensor([ Hk * max_seq, E ], -2, 2), V = tensor([ Hk * max_seq, E ], -2, 2);
let kq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), ks = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');
let vq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), vs = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');

torch.kv_q8_put(K, kq, ks, 0);
torch.kv_q8_put(V, vq, vs, 0);

let Kh = K.view([ Hk, max_seq, E ]), Vh = V.view([ Hk, max_seq, E ]);
let fails = 0;

for (let S in [ 1, 2, 3, 4, 5 ]) {
	for (let len in [ 1, 5, 31, 32, 33, 64, 65, 100, 777, 1500, 2048 ]) {
		for (let splits in [ 1, 7, 48 ]) {
			let q = tensor([ S, Hq, E ], -1, 1);
			let l = torch.from_array([ len ], 'int32').cuda(0);
			let sc = 1.0 / math.sqrt(E);
			let ref = torch.attn_decode(q, Kh, Vh, l, sc, splits, 0);
			let got = torch.attn_decode(q, Kh, Vh, l, sc, splits, 1);
			let again = torch.attn_decode(q, Kh, Vh, l, sc, splits, 1);
			let ref8 = torch.attn_decode_q8(q, kq, ks, vq, vs, l, Hk, sc, splits, 0);
			let got8 = torch.attn_decode_q8(q, kq, ks, vq, vs, l, Hk, sc, splits, 1);
			let d = maxdiff(got, ref), d8 = maxdiff(got8, ref8);
			let det = maxdiff(got, again) == 0;
			let bad = !(d <= 1.5e-3) || !(d8 <= 1.5e-3) || !det;       // (written so that a NaN difference fails)

			if (bad || (len == 777 && splits == 7))
				printf('S=%d len=%4d splits=%2d: fp16 %.2e, q8 %.2e%s%s\n', S, len, splits, d, d8, det ? '' : ' NOT DETERMINISTIC', bad ? '  <-- FAIL' : '');

			if (bad)
				fails++;
		}
	}
}

assert(fails == 0, `${fails} WMMA attention cases out of tolerance`);
print('attn_decode_wmma: ok\n');
