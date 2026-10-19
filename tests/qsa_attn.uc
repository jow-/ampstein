// tests/qsa_attn.uc — torch.qsa_attn (gathered GQA attention with per-query token
// lists, Qwen3.8-Flash-Next QSA) against an ATen reference: random q / K / V
// caches, per-row index lists with -1 padding and a row of all padding.

import * as math from 'math';

const DEV = 'cuda:' + (getenv('AMP_QSA_DEV') || 0);
let R = 37, MS = 6000, HK = 2, HQ = 24, E = 256, KT = 2052;
let rep = HQ / HK;

let q = torch.rand([R, HQ, E], 'float16', DEV).sub(0.5);
let kc = torch.rand([MS, HK, E], 'float16', DEV).sub(0.5);
let vc = torch.rand([MS, HK, E], 'float16', DEV).sub(0.5);

// index lists: random cache rows, ~30% padding, row 5 entirely padding
let idx_l = [];

for (let r = 0; r < R; r++)
	for (let t = 0; t < KT; t++)
		push(idx_l, (r == 5 || math.rand(0, 100) < 30) ? -1 : int(math.rand(0, MS)));

let idx = torch.from_array(idx_l, 'int32', DEV).view([R, KT]);
let scale = 1.0 / math.sqrt(E * 1.0);

function rel(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / b.to('float32').mul(b.to('float32')).mean().item());
}

function ref(q, idx) {
	let i64 = idx.to('int64');
	let pad = i64.le(-1);                                       // true = padding
	let tok = torch.where(pad, torch.zeros([1], 'int64', DEV), i64).contiguous();
	let flat = tok.view([R * KT]);
	let Kg = kc.view([MS, HK * E]).index(flat).view([R, KT, HK, E]).transpose(1, 2).contiguous();
	let Vg = vc.view([MS, HK * E]).index(flat).view([R, KT, HK, E]).transpose(1, 2).contiguous();
	let s = torch.matmul(q.view([R, HK, rep, E]), Kg.transpose(2, 3)).to('float32').mul(scale);

	s = torch.where(pad.view([R, 1, 1, KT]), torch.full([1], -1e30, 'float32', DEV), s).softmax(-1);

	return torch.matmul(s.to('float16'), Vg).view([R, HQ, E]);
}

let want = ref(q, idx);
let ok = true;

for (let sp in [ 0, 1, 4, 16 ]) {
	let got = torch.qsa_attn(q, kc, vc, idx, scale, sp);

	// compare the rows that have valid tokens (row 5 has none: the kernel gives 0,
	// the reference a uniform average of the padding)
	let r = rel(got.narrow(0, 6, R - 6), want.narrow(0, 6, R - 6));
	let r0 = rel(got.narrow(0, 0, 5), want.narrow(0, 0, 5));

	printf('splits %2d: rel-rms vs reference %.3e (rows 0-4 %.3e); row 5 max |out| %.3g\n', sp, r, r0,
		got.narrow(0, 5, 1).abs().max().item());

	if (r > 0.02 || r0 > 0.02)
		ok = false;
}

// Q8 caches (int8 codes + a fp16 scale per 32 values, written by kv_q8_put): the kernel dequantizes while staging, so
// the result must equal the fp16 kernel run on the dequantized caches bit for bit
{
	let kq = torch.zeros([MS, HK * E], 'int8', DEV), ks = torch.zeros([MS, HK * E / 32], 'float16', DEV);
	let vq = torch.zeros([MS, HK * E], 'int8', DEV), vs = torch.zeros([MS, HK * E / 32], 'float16', DEV);

	torch.kv_q8_put(kc.view([MS, HK * E]), kq, ks, 0);
	torch.kv_q8_put(vc.view([MS, HK * E]), vq, vs, 0);

	let kd = torch.kv_q8_get(kq, ks, 0, MS).view([MS, HK, E]), vd = torch.kv_q8_get(vq, vs, 0, MS).view([MS, HK, E]);

	for (let sp in [ 0, 4 ]) {
		let want8 = torch.qsa_attn(q, kd, vd, idx, scale, sp);
		let got8 = torch.qsa_attn(q, [ kq.view([MS, HK, E]), ks.view([MS, HK, E / 32]) ],
			[ vq.view([MS, HK, E]), vs.view([MS, HK, E / 32]) ], idx, scale, sp);
		let d = got8.sub(want8).abs().max().item();

		printf('Q8 splits %d: max |diff| vs the fp16 kernel on dequantized caches %g; rel-rms vs fp16 caches %.3e\n', sp, d,
			rel(got8, torch.qsa_attn(q, kc, vc, idx, scale, sp)));

		if (d != 0)
			ok = false;
	}
}

// determinism
let a = torch.qsa_attn(q, kc, vc, idx, scale, 4), b = torch.qsa_attn(q, kc, vc, idx, scale, 4);

if (a.sub(b).abs().max().item() != 0) {
	print('not deterministic\n');
	ok = false;
}

// timing: a prefill-slab shaped launch
torch.synchronize();

let c0 = clock(true);

for (let i = 0; i < 20; i++)
	torch.qsa_attn(q, kc, vc, idx, scale);

torch.synchronize();

let c1 = clock(true);

printf('%d queries x %d tokens: %.2f ms per launch\n', R, KT,
	((c1[0] - c0[0]) + (c1[1] - c0[1]) / 1e9) * 1000 / 20);

print(ok ? 'qsa_attn ok\n' : 'qsa_attn FAILED\n');
exit(ok ? 0 : 1);
