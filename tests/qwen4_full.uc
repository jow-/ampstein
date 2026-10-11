// tests/qwen4_full.uc — Qwen3.8-Flash-Next whole model vs the exllamav3 oracle
// (tests/oracle/make_qwen4_oracle.py): PLE output, residual streams after every
// block, final logits and a cached greedy continuation (decode path).
// Needs all six GPUs free (~100 GB of weights). Env: AMP_MODEL_QWEN4,
// QWEN4_FIX (fixture path), AMP_QWEN4_TRACE=0 to skip the per-block comparison.

import * as qwen4 from 'ampstein.qwen4';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') ||
	'/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
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

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let n = length(ids);

let t0 = time();
const NL = +(getenv('AMP_QWEN4_LAYERS') || 0);       // >0: first NL layers only, 1 GPU
let model = qwen4.load(MODEL, NL ? { layers: NL, devices: 1 } : { });

printf('loaded in %.1fs\n', time() - t0);

let cfg = model.cfg;
let cache = model.new_cache(n + 64);

if (getenv('AMP_QWEN4_TRACE') != '0')
	cache.trace = { blocks: [] };

const PIPE = (getenv('AMP_QWEN4_PIPE') == '1');     // prefill through the wavefront runner
let logits, tp0 = clock(true);

if (PIPE) {
	let r = model.prefill_pipeline(cache, ids, { chunk: +(getenv('AMP_QWEN4_CHUNK') || 1024) });

	while (r.step(2));

	logits = r.logits();
	torch.synchronize();
	printf('pipelined prefill: %d tokens in %.2f s (%.0f tok/s)\n', n, elapsed(tp0), n / elapsed(tp0));
}
else
	logits = model.forward_chunk(ids, cache);

let ok = true;

if (cache.trace && cache.trace.ple) {
	let r = rel_rms(cache.trace.ple, fx.tensor('ple_out').reshape([n, cfg.hc_mult, cfg.hidden_size]));

	printf('PLE out rel-rms %.2e\n', r);

	if (r > 0.02)
		ok = false;
}

if (getenv('AMP_QWEN4_TRACE') != '0') {
	let worst = 0.0, worst_i = -1;

	for (let i = 0; i < model.nl; i++) {
		let r = rel_rms(cache.trace.blocks[i],
			fx.tensor(`b${i}_x`).reshape([n, cfg.hc_mult, cfg.hidden_size]));

		if (i < 6 || i % 8 == 7 || NL)
			printf('  block %2d (%s) rel-rms %.2e\n', i, cfg.layer_types[i], r);

		if (r > worst) { worst = r; worst_i = i; }
	}

	printf('worst block %d rel-rms %.2e\n', worst_i, worst);

	if (worst > 0.1)
		ok = false;
}

if (NL) {
	try { torch.synchronize(); } catch (e) {}

	print(ok ? 'qwen4_full (partial) ok\n' : 'qwen4_full (partial) FAILED\n');
	exit(ok ? 0 : 1);
}

// compare the last NC positions (the full (n, vocab) fp32 copies of a long prompt do not fit)
let NC = PIPE ? 1 : min(n, 64);
let lg = PIPE ? logits.view([1, -1]) : logits.view([n, -1]).narrow(0, n - NC, NC);
let ref = fx.tensor('logits').narrow(0, n - NC, NC).cuda(model.mixer.dev);
let rl = rel_rms(lg, ref);
let am = lg.argmax(-1).tolist(), rm = ref.argmax(-1).tolist();
let agree = 0;

for (let i = 0; i < NC; i++)
	if (am[i] == rm[i]) agree++;

printf('logits (last %d) rel-rms %.3e, argmax %d/%d, last-pos argmax %d vs %d\n', NC, rl, agree, NC,
	am[NC - 1], rm[NC - 1]);

if (rl > 0.15 || am[NC - 1] != rm[NC - 1])
	ok = false;

// cached greedy continuation (single-token decode path)
let cont = fx.tensor('cont').tolist();
let got = [];
let tok = am[NC - 1];
let steps = min(length(cont), 12);

for (let i = 0; i < steps; i++) {
	push(got, tok);

	let l = model.forward_chunk([tok], cache);

	tok = l.view([1, -1]).argmax(-1).tolist()[0];
}

let same = 0;

for (let i = 0; i < steps; i++)
	if (got[i] == cont[i]) same++;

printf('continuation %d/%d identical: %J\n', same, steps, got);

if (same < steps - 2)
	ok = false;

// eager decode timing (same cache, continues the sequence)
{
	let t = clock(true);
	let nt = 8, tk = got[steps - 1];

	for (let i = 0; i < nt; i++)
		tk = model.forward_chunk([tk], cache).view([1, -1]).argmax(-1).tolist()[0];

	torch.synchronize();
	printf('eager decode: %.1f ms/token\n', elapsed(t) * 1000.0 / nt);
}

// CUDA-graph decode: fresh cache, graphs built (and reset) before the prefill
if (getenv('AMP_QWEN4_GRAPH') != '0') {
	let gc = model.new_cache(n + 64);
	let g = model.graphs(gc, [ 1 ]);

	for (let cl in gc.layers)
		if (cl.rec_state) {
			cl.rec_state.fill_(0);
			cl.conv_state.fill_(0);
		}

	model.aux_reset(gc);
	gc.length = 0;

	let l0 = model.forward_chunk(ids, gc);
	let tk = l0.view([n, -1]).narrow(0, n - 1, 1).argmax(-1).tolist()[0];
	let gt = [];
	let t = clock(true);

	for (let i = 0; i < steps; i++) {
		push(gt, tk);
		tk = g.run([tk], gc.length).logits.view([1, -1]).argmax(-1).tolist()[0];
	}

	torch.synchronize();
	printf('graph decode: %.1f ms/token\n', elapsed(t) * 1000.0 / steps);

	let gs = 0;

	for (let i = 0; i < steps; i++)
		if (gt[i] == cont[i]) gs++;

	printf('graph continuation %d/%d identical: %J\n', gs, steps, gt);

	if (gs < steps - 2)
		ok = false;
}

try { torch.synchronize(); } catch (e) {}

print(ok ? 'qwen4_full ok\n' : 'qwen4_full FAILED\n');
exit(ok ? 0 : 1);
