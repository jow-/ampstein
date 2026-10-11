// tests/dsv4_full.uc — DeepSeek-V4-Flash vs the exllamav3 oracle
// (tests/oracle/make_dsv4_oracle.py): attention / ffn outputs and the residual
// streams after every block, final logits and (short prompt) a cached greedy
// continuation through the decode path. Needs all six GPUs free.
// Env: AMP_MODEL_DSV4, DSV4_FIX (fixture file; default the short one),
// AMP_DSV4_LAYERS=N (first N layers only, 1 GPU), DSV4_DEVICES (default 6),
// DSV4_CHUNK (prefill chunk, default whole prompt), AMP_DSV4_CONT=0 (skip decode).

import * as dsv4 from 'ampstein.dsv4';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') ||
	'/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const FIX = getenv('AMP_DSV4_FIX') || `${sourcepath(0, true)}/fixtures/dsv4_oracle_short.safetensors`;

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: DSV4 model dir or oracle fixture missing\n');
	exit(125);
}

function rel_rms(a, b) {
	let x = a.to('float32'), y = b.to('float32');
	let d = x.sub(y);

	return math.sqrt(d.mul(d).mean().item() / y.mul(y).mean().item());
}

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let n = length(ids);
let keep = min(n, 128);
let short = (n <= 128);

const NL = +(getenv('AMP_DSV4_LAYERS') || 0);

let t0 = time();
let model = dsv4.load(MODEL, NL ? { layers: NL, devices: +(getenv('AMP_DSV4_DEVICES') || 1) } : { devices: +(getenv('AMP_DSV4_DEVICES') || 6) });

printf('loaded in %.1fs, devices %J\n', time() - t0, model.devs);

let cfg = model.cfg;
let nl = cfg.num_hidden_layers;
let cache = model.new_cache(n + 64);
let trace = [];
let chunk = +(getenv('AMP_DSV4_CHUNK') || n);
let ok = true;
let tp0 = clock(true);

// the trace is captured on the last chunk only (its last `keep` rows)
const NOTRACE = getenv('AMP_DSV4_NOTRACE') ? true : false;       // the fused prefill path (no per-block trace)
let logits = model.prefill(ids, cache, chunk, short ? n : 64, NOTRACE ? null : trace, keep);

torch.synchronize();

let tp1 = clock(true);

printf('prefill: %d tokens in %.2f s\n', n, (tp1[0] - tp0[0]) + (tp1[1] - tp0[1]) / 1e9);

let worst = 0.0, worst_i = -1;
let H = cfg.hidden_size;
let nblocks = NOTRACE ? 0 : nl;
let tail = (n == length(ids) && chunk >= n);       // trace rows == last `keep` rows of the prompt

for (let i = 0; i < nblocks; i++) {
	let tr = trace[i];
	let rows = tr.x.shape()[0];
	let rx = rel_rms(tr.x, fx.tensor(`b${i}_x`).reshape([rows, 4, H]));
	let ra = rel_rms(tr.attn_out, fx.tensor(`b${i}_attn_out`).reshape([rows, H]));
	let rm = rel_rms(tr.mlp_out, fx.tensor(`b${i}_mlp_out`).reshape([rows, H]));
	let ri = rel_rms(tr.attn_in, fx.tensor(`b${i}_attn_in`).reshape([rows, H]));
	let rf = rel_rms(tr.mlp_in, fx.tensor(`b${i}_mlp_in`).reshape([rows, H]));

	let hx = '';

	if (tr.xh) {
		try {
			let ref = fx.tensor(`b${i}_x@h`).reshape([-1, 4, H]);

			hx = sprintf('  head rows: r0 %.1e r1 %.1e (|x| ours %.0f / ref %.0f)', rel_rms(tr.xh.narrow(0, 0, 1), ref.narrow(0, 0, 1)),
				rel_rms(tr.xh.narrow(0, 1, 1), ref.narrow(0, 1, 1)), tr.xh.narrow(0, 1, 1).abs().max().item(), ref.narrow(0, 1, 1).abs().max().item());
		}
		catch (e) {}
	}

	if (i < 8 || i % 6 == 5 || NL || rx > 0.05 || (hx != '' && i >= 8)) {
		printf('  block %2d (%-7s) x %.2e  attn_in %.2e attn %.2e  ffn_in %.2e ffn %.2e\n', i, cfg.layer_types[i], rx, ri, ra, rf, rm);

		if (hx != '')
			print(hx + '\n');
	}

	if (rx > worst) { worst = rx; worst_i = i; }
}

printf('worst block %d rel-rms %.2e\n', worst_i, worst);

if (worst > 0.1)
	ok = false;

if (NL) {
	try { torch.synchronize(); } catch (e) {}

	print(ok ? 'dsv4_full (partial) ok\n' : 'dsv4_full (partial) FAILED\n');
	exit(ok ? 0 : 1);
}

let NC = short ? n : 64;
let lg = logits.view([NC, -1]);
let ref = fx.tensor('logits').reshape([NC, -1]).cuda(model.devs[nl - 1]);
let rl = rel_rms(lg, ref);
let am = lg.argmax(-1).tolist(), rm = ref.argmax(-1).tolist();
let agree = 0;

for (let i = 0; i < NC; i++)
	if (am[i] == rm[i]) agree++;

printf('logits (last %d) rel-rms %.3e, argmax %d/%d, last-pos argmax %d vs %d\n', NC, rl, agree, NC,
	am[NC - 1], rm[NC - 1]);

if (rl > 0.15 || am[NC - 1] != rm[NC - 1])
	ok = false;

if (short && getenv('AMP_DSV4_CONT') != '0') {
	// teacher-forced decode: feed the oracle's continuation through the cached
	// single-token path and check each step's argmax (L88: near-ties flip a
	// free-running continuation; the margin shows whether one did)
	let cont = fx.tensor('cont').tolist();
	let steps = min(length(cont), 24);
	let same = 0;
	let cur = am[NC - 1];                    // argmax after the prompt
	let lrow = lg.narrow(0, NC - 1, 1).view([-1]);

	for (let i = 0; i < steps; i++) {
		let lv = lrow.to('float32');
		let top = lv.argmax(-1).item();
		let margin = lv.narrow(0, top, 1).item() - lv.narrow(0, cont[i], 1).item();

		if (top == cont[i])
			same++;
		else
			printf('  step %2d: argmax %d vs oracle %d (logit margin %.3f)\n', i, top, cont[i], margin);

		lrow = model.forward_chunk([cont[i]], cache).view([-1]);
	}

	printf('teacher-forced continuation %d/%d argmax-identical\n', same, steps);

	if (same < steps - 2)
		ok = false;
}

try { torch.synchronize(); } catch (e) {}

print(ok ? 'dsv4_full ok\n' : 'dsv4_full FAILED\n');
exit(ok ? 0 : 1);
