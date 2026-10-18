// tests/v41_attn.uc — torch.v41_attn (DeepSeek-V4.1 sparse MQA attention) against a composite
// reference: window rows + picked compressed rows, sink in the denominator, V = K, rope slice
// rotated back by the position.

import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

function run(T, H, pos0, Nc, Kc, nsplit) {
	const W = 128, D = 512, rd = 64;
	let wbase = max(0, pos0 - (W - 1));
	let Nw = pos0 + T - wbase;
	let q = torch.rand([T, H, D], 'float16', dv).sub(0.5).mul(2.0);
	let kvw = torch.rand([Nw, D], 'float16', dv).sub(0.5).mul(2.0);
	let kvc = Nc ? torch.rand([Nc, D], 'float16', dv).sub(0.5).mul(2.0) : null;
	let sinks = torch.rand([H], 'float32', dv).sub(0.5);
	let fr = [];

	for (let i = 0; i < rd / 2; i++)
		push(fr, math.pow(10000.0, -(2.0 * i) / rd));

	let inv = torch.from_array(fr, 'float32', dv);
	let ix = null, idl = [];

	if (Nc) {
		for (let t = 0; t < T; t++)
			for (let k = 0; k < Kc; k++)
				push(idl, (k * 3 + t) % (Nc + 3) < Nc ? (k * 3 + t) % (Nc + 3) : -1);

		ix = torch.from_array(idl, 'int32', dv).view([T, Kc]);
	}

	let scale = math.pow(D * 1.0, -0.5);
	let o = torch.v41_attn(q, kvw, wbase, kvc, ix, sinks, inv, pos0, scale, W, nsplit);

	// a Q8 pool (codes + a scale per 32, kv_q8_put): the kernel's in-flight dequantization must equal the dequantized
	// fp16 pool bit for bit, and stay within quantization noise of the fp16 original
	if (Nc) {
		let qc = torch.zeros([Nc, D], 'int8', dv), qs = torch.zeros([Nc, D / 32], 'float16', dv);

		torch.kv_q8_put(kvc, qc, qs, 0);

		let kvd = torch.kv_q8_get(qc, qs, 0, Nc);
		let od = torch.v41_attn(q, kvw, wbase, kvd, ix, sinks, inv, pos0, scale, W, nsplit);
		let oq = torch.v41_attn(q, kvw, wbase, [ qc, qs ], ix, sinks, inv, pos0, scale, W, nsplit);
		let e8 = rel_rms(oq, od), en = rel_rms(oq, o);

		assert(e8 < 1e-6, `Q8 pool: dequantized vs in-kernel rel rms ${e8}`);
		assert(en < 2e-2, `Q8 pool vs fp16 pool: rel rms ${en}`);
		printf('   q8 pool: in-kernel vs dequantized %.1e, vs fp16 %.1e\n', e8, en);
	}

	// reference, query by query
	let ref = [];

	for (let t = 0; t < T; t++) {
		let p = pos0 + t;
		let lo = max(0, p - W + 1) - wbase, n = p - wbase - lo + 1;
		let keys = kvw.narrow(0, lo, n).to('float32');

		if (Nc) {
			let sel = [];

			for (let k = 0; k < Kc; k++)
				if (idl[t * Kc + k] >= 0)
					push(sel, idl[t * Kc + k]);

			if (length(sel))
				keys = torch.cat([ keys, kvc.index(torch.from_array(sel, 'int64', dv)).to('float32') ], 0);
		}

		let qt = q.narrow(0, t, 1).view([H, D]).to('float32');
		let s = torch.matmul(qt, keys.t()).mul(scale);
		let all = torch.cat([ s, sinks.view([H, 1]) ], 1);
		let pr = all.softmax(1).narrow(1, 0, keys.shape()[0]);
		let r = torch.matmul(pr, keys);        // (H, D)
		let ang = inv.mul(p * 1.0);
		let c = ang.cos().view([1, 32]), sn = ang.sin().view([1, 32]);
		let tail = r.narrow(1, D - rd, rd).reshape([H, 32, 2]);
		let a = tail.narrow(2, 0, 1).view([H, 32]), b = tail.narrow(2, 1, 1).view([H, 32]);
		let ra = a.mul(c).add(b.mul(sn)), rb = b.mul(c).sub(a.mul(sn));

		push(ref, torch.cat([ r.narrow(1, 0, D - rd), torch.cat([ ra.view([H, 32, 1]), rb.view([H, 32, 1]) ], 2).view([H, rd]) ], 1).view([1, H, D]));
	}

	let rr = torch.cat(ref, 0);
	// the same through a window ring and a device-side position
	let RING = 1024, slots = [];

	for (let r = 0; r < Nw; r++)
		push(slots, (wbase + r) % RING);

	let ringt = torch.zeros([RING, D], 'float16', dv);

	ringt.index_copy_(0, torch.from_array(slots, 'int64', dv), kvw);

	let pt = torch.from_array([ pos0 ], 'int64', dv);
	let o2 = torch.v41_attn(q, ringt, 0, kvc, ix, sinks, inv, 0, scale, W, nsplit, pt, RING);
	let er = rel_rms(o2, o);

	assert(er < 1e-6, `ring / device position: rel rms ${er}`);

	let e = rel_rms(o, rr);

	assert(e < 3e-3, `T=${T} pos0=${pos0} Nc=${Nc} Kc=${Kc} nsplit=${nsplit}: rel rms ${e}`);
	printf('ok T=%d pos0=%d Nc=%d Kc=%d nsplit=%d (rel %.2e)\n', T, pos0, Nc, Kc, nsplit, e);
}

run(1, 64, 0, 0, 0, 1);
run(5, 64, 0, 0, 0, 2);
run(3, 64, 300, 40, 24, 4);
run(1, 64, 5000, 700, 512, 8);
run(9, 64, 700, 300, 100, 1);
run(33, 64, 130, 70, 33, 0);
printf('v41_attn: all ok\n');
