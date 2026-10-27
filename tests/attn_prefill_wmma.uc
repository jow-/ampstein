// Test: torch.attn_prefill / attn_prefill_q8 (the ROCm WMMA prefill attention, head dim 256) against torch.flash_attn (ATen) on the same keys: chunks
// of S query positions at several start positions, GQA 24 / 4, fp16 caches and Q8 caches (against FlashAttention over the dequantized cache).
// Skipped (exit 125) where the kernel is not built.
//
//   ucode -L <build-dir> -L <src>/ucode tests/attn_prefill_wmma.uc

import * as math from 'math';

if (torch.device_props(0).backend != 'rocm') {
	print('attn_prefill_wmma: skipped (ROCm only)\n');
	exit(125);
}

let seed = 11;

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

const E = 256, Hk = 4, Hq = 24, max_seq = 4096;
let K = tensor([ Hk * max_seq, E ], -2, 2), V = tensor([ Hk * max_seq, E ], -2, 2);
let kq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), ks = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');
let vq = torch.zeros([ Hk * max_seq, E ], 'int8', 'cuda:0'), vs = torch.zeros([ Hk * max_seq, E / 32 ], 'float16', 'cuda:0');

torch.kv_q8_put(K, kq, ks, 0);
torch.kv_q8_put(V, vq, vs, 0);

let Kh = K.view([ 1, Hk, max_seq, E ]), Vh = V.view([ 1, Hk, max_seq, E ]);
let Kd = torch.kv_q8_get(kq, ks, 0, Hk * max_seq).view([ 1, Hk, max_seq, E ]);
let Vd = torch.kv_q8_get(vq, vs, 0, Hk * max_seq).view([ 1, Hk, max_seq, E ]);
let fails = 0;

for (let S in [ 1, 7, 10, 11, 64, 100, 1000 ]) {
	for (let pos in [ 0, 5, 100, 3000 ]) {
		let len = pos + S;

		if (len > max_seq)
			continue;

		let q = tensor([ S, Hq, E ], -1, 1);
		let l = torch.from_array([ len ], 'int32').cuda(0);
		let sc = 1.0 / math.sqrt(E);
		let qh = q.transpose(0, 1).contiguous().view([ 1, Hq, S, E ]);
		let ref = torch.flash_attn(qh, Kh.narrow(2, 0, len), Vh.narrow(2, 0, len), true).view([ Hq, S, E ]).transpose(0, 1);
		let ref8 = torch.flash_attn(qh, Kd.narrow(2, 0, len), Vd.narrow(2, 0, len), true).view([ Hq, S, E ]).transpose(0, 1);
		let got = torch.attn_prefill(q, Kh.view([ Hk, max_seq, E ]), Vh.view([ Hk, max_seq, E ]), l, sc);
		let got8 = torch.attn_prefill_q8(q, kq, ks, vq, vs, l, Hk, sc);
		let again = torch.attn_prefill(q, Kh.view([ Hk, max_seq, E ]), Vh.view([ Hk, max_seq, E ]), l, sc);
		let d = maxdiff(got, ref), d8 = maxdiff(got8, ref8), det = maxdiff(got, again) == 0;
		let bad = !(d <= 2e-3) || !(d8 <= 2e-3) || !det;

		printf('S=%4d pos=%4d: fp16 %.2e, q8 %.2e%s%s\n', S, pos, d, d8, det ? '' : ' NOT DETERMINISTIC', bad ? '  <-- FAIL' : '');

		if (bad)
			fails++;
	}
}

assert(fails == 0, `${fails} WMMA prefill attention cases out of tolerance`);
print('attn_prefill_wmma: ok\n');
