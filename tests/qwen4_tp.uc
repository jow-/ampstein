// tests/qwen4_tp.uc — tensor/expert-parallel Qwen3.8-Flash-Next vs the exllamav3
// oracle (tests/oracle/make_qwen4_oracle.py): residual streams after each block,
// logits, a cached greedy continuation, and (QWEN4_GRAPH != 0) the same decode
// through the multi-GPU graphs. Needs all six GPUs free.
// Env: AMP_MODEL_QWEN4, QWEN4_FIX, AMP_QWEN4_LAYERS=N (first N layers only, per-block
// compare), QWEN4_TP_DEVS (rank count, default all), AMP_QWEN4_TRACE=0, AMP_QWEN4_GRAPH=0.

import * as qwen4_tp from 'ampstein.qwen4_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') || '/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
const FIX = getenv('AMP_QWEN4_FIX') || `${sourcepath(0, true)}/fixtures/qwen4_oracle.safetensors`;

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: Qwen4 model dir or oracle fixture missing\n');
	exit(125);
}

function elapsed(c0) {
	let c1 = clock(true);

	return (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1000000000.0;
}

function rel_rms(a, b) {
	let x = a.to('float32'), y = b.to('float32');
	let d = x.sub(y);

	return math.sqrt(d.mul(d).mean().item() / y.mul(y).mean().item());
}

const NL = +(getenv('AMP_QWEN4_LAYERS') || 0);
let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let n = length(ids);

let t0 = clock(true);
let tp = qwen4_tp.load(MODEL, { devices: +(getenv('AMP_QWEN4_TP_DEVS') || 0) || null, layers: NL || null,
	stages: +(getenv('AMP_QWEN4_TP_STAGES') || 1) });

printf('loaded in %.1fs on %d ranks, %d stage(s)\n', elapsed(t0), tp.G, tp.NS);

let cfg = tp.cfg;
let cache = tp.new_cache(n + 64);
let ok = true;

if (getenv('AMP_QWEN4_TRACE') != '0')
	cache.trace = { blocks: [] };

let r = tp.forward(ids, cache, null);

if (cache.trace) {
	let worst = 0.0, worst_i = -1;

	for (let i = 0; i < tp.nl; i++) {
		let rr = rel_rms(cache.trace.blocks[i], fx.tensor(`b${i}_x`).reshape([n, cfg.hc_mult, cfg.hidden_size]));

		if (i < 6 || i % 8 == 7 || NL)
			printf('  block %2d (%s) rel-rms %.2e\n', i, cfg.layer_types[i], rr);

		if (rr > worst) { worst = rr; worst_i = i; }
	}

	printf('worst block %d rel-rms %.2e\n', worst_i, worst);

	if (worst > 0.1)
		ok = false;
}

if (NL) {
	try { torch.synchronize(); } catch (e) {}

	print(ok ? 'qwen4_tp (partial) ok\n' : 'qwen4_tp (partial) FAILED\n');
	exit(ok ? 0 : 1);
}

let lg = r.logits.view([n, -1]);
let NC = min(n, 64);
let ref = fx.tensor('logits').narrow(0, n - NC, NC).cuda(tp.last.devs[0]);
let got = lg.narrow(0, n - NC, NC);
let rl = rel_rms(got, ref);
let am = got.argmax(-1).tolist(), rm = ref.argmax(-1).tolist();
let agree = 0;

for (let i = 0; i < NC; i++)
	if (am[i] == rm[i]) agree++;

printf('logits (last %d) rel-rms %.3e, argmax %d/%d, last-pos argmax %d vs %d\n', NC, rl, agree, NC,
	am[NC - 1], rm[NC - 1]);

if (rl > 0.15 || am[NC - 1] != rm[NC - 1])
	ok = false;

// cached greedy continuation (eager single-token decode)
let cont = fx.tensor('cont').tolist();
let steps = min(length(cont), 12);
let tok = am[NC - 1], gt = [];
let tt = clock(true);

for (let i = 0; i < steps; i++) {
	push(gt, tok);
	tok = tp.forward([tok], cache, null).logits.view([1, -1]).argmax(-1).tolist()[0];
}

torch.synchronize();
printf('eager TP decode: %.1f ms/token\n', elapsed(tt) * 1000.0 / steps);

let same = 0;

for (let i = 0; i < steps; i++)
	if (gt[i] == cont[i]) same++;

printf('continuation %d/%d identical: %J\n', same, steps, gt);

if (same < steps - 2)
	ok = false;

// multi-GPU CUDA-graph decode
if (tp.graphs && getenv('AMP_QWEN4_GRAPH') != '0') {
	let gc2 = tp.new_cache(n + 64);
	let g = tp.graphs(gc2, [ 1 ]);

	tp.reset(gc2);

	tp.forward(ids, gc2, 0);

	let tk = lg.narrow(0, n - 1, 1).argmax(-1).tolist()[0];
	let gg = [];
	let t2 = clock(true);

	for (let i = 0; i < steps; i++) {
		push(gg, tk);
		tk = g.run([tk], gc2.length).logits.view([1, -1]).argmax(-1).tolist()[0];
	}

	torch.synchronize();
	printf('TP graph decode: %.2f ms/token\n', elapsed(t2) * 1000.0 / steps);

	let s2 = 0;

	for (let i = 0; i < steps; i++)
		if (gg[i] == cont[i]) s2++;

	printf('graph continuation %d/%d identical: %J\n', s2, steps, gg);

	if (s2 < steps - 2)
		ok = false;
}

try { torch.synchronize(); } catch (e) {}

print(ok ? 'qwen4_tp ok\n' : 'qwen4_tp FAILED\n');
exit(ok ? 0 : 1);
