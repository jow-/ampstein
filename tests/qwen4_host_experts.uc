// tests/qwen4_host_experts.uc — Qwen3.8-Flash-Next expert-parallel with part of the experts in pinned host
// memory (qwen4_tp host_experts): the residual streams after every block of a prefill chunk and of a few
// eager decode steps must agree with the all-in-VRAM model to its own run-to-run noise (a host pointer
// changes where the kernels read the weights, never the arithmetic; the kernel level is checked bit for
// bit by examples/bench_zerocopy_qwen4.uc). Needs two GPUs free.
// Env: AMP_MODEL_QWEN4, AMP_QWEN4_LAYERS=N (default 8), AMP_QWEN4_HOST_FRAC (resident fraction, default 0.4).

import * as qwen4_tp from 'ampstein.qwen4_tp';
import * as glm5 from 'ampstein.glm5';
import * as math from 'math';
import { stat, writefile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') || '/data/models/Qwen3.8-Flash-Next-exl3-2.05bpw';
const NL = +(getenv('AMP_QWEN4_LAYERS') || 8);
const FRAC = +(getenv('AMP_QWEN4_HOST_FRAC') || 0.4);

if (!stat(MODEL) || torch.device_count() < 2) {
	warn('skip: Qwen4 model dir missing or fewer than two GPUs\n');
	exit(125);
}

// a deterministic synthetic profile: no two experts of a layer tie
let counts = [];

for (let l = 0; l < 48; l++) {
	let row = [];

	for (let e = 0; e < 512; e++)
		push(row, ((l * 7919 + e * 104729) % 4093) + e / 1024.0);

	push(counts, row);
}

const PROFILE = (getenv('TMPDIR') || '/tmp') + `/qwen4_host_experts_profile.${time()}.json`;

writefile(PROFILE, sprintf('%J', { counts: counts }));

let tk = torch.tokenizer(MODEL);
let text = 'The quick brown fox jumps over the lazy dog.';

for (let i = 0; i < 60; i++)
	text += ' token' + i;

let ids = tk.encode(text + ' Expert streaming keeps cold experts in host memory.', false);
const DECODE = [101, 202, 303, 404, 505];

let exact = true, planned = 0;

// one MoE layer on a one-row input: the decode kernel is deterministic, so exchanging experts between VRAM and
// host memory (the same bytes, other addresses) must not change one bit of its output
function layer_out(m, L, r) {
	let M = m.layers[L][r].moe, dv = 'cuda:' + M.dev, H = m.cfg.hidden_size;
	let xs = [];

	for (let i = 0; i < H; i++)
		push(xs, ((i * 2654435761 + L * 97) % 1000) / 20000.0 - 0.025);

	let y = torch.from_array(xs, 'float32', dv).to('float16').reshape([1, H]);

	return glm5.moe_finish(m.cfg, M, glm5.moe_begin(m.cfg, M, y)).cpu();
}

// exchange the first cold expert with the first resident one of every layer and rank
function swap_all(m) {
	for (let L = 0; L < NL; L++)
		for (let r = 0; r < m.Gs; r++) {
			let ex = m.layers[L][r].moe.experts, jc = -1, jh = -1;

			for (let j = 0; j < length(ex); j++) {
				if (ex[j].slot != null && jc < 0)
					jc = j;
				else if (ex[j].slot == null && jh < 0)
					jh = j;
			}

			if (jc < 0 || jh < 0)
				continue;

			// the experts the layer's one-row input picks matter little: force the exchange of the first pair and
			// of the pair it was made from, then check that nothing moved
			let before = layer_out(m, L, r);

			m.dyn.force(L, r, jc, jh);

			if (layer_out(m, L, r).sub(before).abs().max().item() != 0)
				exact = false;

			m.dyn.force(L, r, jh, jc);      // back: the original resident expert returns, the cold one leaves

			if (layer_out(m, L, r).sub(before).abs().max().item() != 0)
				exact = false;

			m.dyn.force(L, r, jc, jh);
		}
}

function run(host) {
	let m = qwen4_tp.load(MODEL, { devices: 2, layers: NL, host_experts: host });
	let out = [];

	for (let pass = 0; pass < 2; pass++) {
		let cache = m.new_cache(length(ids) + 64);

		cache.trace = { blocks: [] };
		m.forward(ids, cache, null);

		if (m.dyn && pass == 0)
			swap_all(m);      // the second pass prefills on the exchanged (and re-staged) experts

		for (let t in DECODE) {
			m.forward([t], cache, null);

			if (m.dyn)
				m.dyn.after(1);
		}

		out = [ ...out, ...map(cache.trace.blocks, (b) => b.cpu()) ];
	}

	torch.synchronize();

	if (m.dyn) {
		printf('planned exchanges: %d in %d plans\n', m.dyn.stats.swaps, m.dyn.stats.plans);
		planned = m.dyn.stats.swaps;
	}

	return out;
}

function rel_rms(a, b) {
	let x = a.to('float32'), y = b.to('float32');
	let d = x.sub(y);

	return math.sqrt(d.mul(d).mean().item() / y.mul(y).mean().item());
}

function worst_rel(a, b) {
	let w = 0.0;

	for (let i = 0; i < length(a); i++) {
		let r = rel_rms(a[i], b[i]);

		if (r > w)
			w = r;
	}

	return w;
}

// the all-VRAM model is not bit-reproducible run to run (the prefill MoE accumulates with atomics in
// arrival order), so the hosted model is held to that run-to-run noise floor
let base = run(null);

gc();

let base2 = run(null);

gc();

let hosted = run({ frac: FRAC, profile: PROFILE });

gc();

// dynamic residency: exchanges forced in every layer, then a plan after every decode step
// (a near-zero prior: whatever the steps pick beats the resident experts, so planned exchanges do happen)
let dynr = run({ frac: FRAC, profile: PROFILE, swaps: 2, interval: 1, prior: 0.001 });
let ok = length(base) == length(hosted) && length(base) == length(base2) && length(base) == length(dynr) && length(base) > 0;
let floor = ok ? worst_rel(base, base2) : 1.0;

// the first two blocks of each forward have not amplified the atomics' rounding noise yet
function early_of(a, b) {
	let w = 0.0;

	for (let i = 0; i < length(a); i++)
		if (i % NL < 2 && rel_rms(a[i], b[i]) > w)
			w = rel_rms(a[i], b[i]);

	return w;
}

for (let v in [ [ 'static host experts', hosted ], [ 'dynamic residency', dynr ] ]) {
	let diff = ok ? worst_rel(base, v[1]) : 1.0;
	let early = ok ? early_of(base, v[1]) : 1.0;

	printf('%s: %d blocks, %d layers, resident fraction %.2f: run-to-run rel-rms %.2e, vs VRAM %.2e, first two blocks of each forward %.2e\n',
		v[0], length(base), NL, FRAC, floor, diff, early);

	ok = ok && diff <= 3 * floor + 1e-6 && early < 1e-3;
}

printf('layer output unchanged by exchanges (bit for bit): %s\n', exact ? 'yes' : 'NO');

ok = ok && exact && planned > 0;
print(ok ? 'qwen4_host_experts ok\n' : 'qwen4_host_experts FAILED\n');
exit(ok ? 0 : 1);
