// Test: torch.attn_decode_q8 (decode attention over an int8 KV cache, scales per 32 values) against torch.attn_decode on the same cache
// dequantized to fp16 (must agree to fp16 noise), and against the unquantized cache (the cost of Q8, reported and bounded). Synthetic data.
//
//   ucode -L <build-dir> -L <src>/ucode tests/attn_decode_q8.uc

import * as math from 'math';

let seed = 99;

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

let fails = 0;

for (let E in [ 256, 128, 64 ]) {
	let Hk = 4, Hq = 24, max_seq = 1024;
	let K = tensor([ Hk * max_seq, E ], -2, 2), V = tensor([ Hk * max_seq, E ], -2, 2);
	let kq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), ks = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');
	let vq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), vs = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');

	torch.kv_q8_put(K, kq, ks, 0);
	torch.kv_q8_put(V, vq, vs, 0);

	let Kd = torch.kv_q8_get(kq, ks, 0, Hk * max_seq).view([ Hk, max_seq, E ]);
	let Vd = torch.kv_q8_get(vq, vs, 0, Hk * max_seq).view([ Hk, max_seq, E ]);
	let K0 = K.view([ Hk, max_seq, E ]), V0 = V.view([ Hk, max_seq, E ]);

	for (let S in [ 1, 3, 4 ]) {
		for (let len in [ 1, 37, 300, 1024 ]) {
			let q = tensor([ S, Hq, E ], -1, 1);
			let l = torch.from_array([ len ], 'int32').cuda(0);
			let sc = 1.0 / math.sqrt(E);
			let ref = torch.attn_decode(q, Kd, Vd, l, sc, null, 0);
			let ref0 = torch.attn_decode(q, K0, V0, l, sc, null, 0);
			let got = torch.attn_decode_q8(q, kq, ks, vq, vs, l, Hk, sc);
			let again = torch.attn_decode_q8(q, kq, ks, vq, vs, l, Hk, sc);
			let d = maxdiff(got, ref), dq = maxdiff(got, ref0), scale0 = ref0.to('float32').abs().max().item();

			printf('E=%d S=%d len=%4d: vs dequantized fp16 %.2e, vs unquantized %.2e (%.2f%% of max |o| %.2f)%s\n', E, S, len, d, dq,
				100 * dq / scale0, scale0, maxdiff(got, again) == 0 ? '' : ' NOT DETERMINISTIC');

			if (d > 2e-3 || dq > 0.03 * scale0 || maxdiff(got, again) != 0)
				fails++;
		}
	}
}

// bad input is rejected
let ok = false;

try {
	torch.attn_decode_q8(1, 2, 3);
}
catch (e) {
	ok = true;
}

assert(ok, 'attn_decode_q8 accepts bad arguments');
assert(fails == 0, `${fails} Q8 attention cases out of tolerance`);
print('attn_decode_q8: ok\n');
