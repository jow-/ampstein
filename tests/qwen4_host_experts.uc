// tests/qwen4_host_experts.uc — Qwen3.8-Flash-Next expert-parallel with part of the experts in pinned host
// memory (qwen4_tp host_experts): the residual streams after every block of a prefill chunk and of a few
// eager decode steps must agree with the all-in-VRAM model to its own run-to-run noise (a host pointer
// changes where the kernels read the weights, never the arithmetic; the kernel level is checked bit for
// bit by examples/bench_zerocopy_qwen4.uc). Needs two GPUs free.
// Env: AMP_MODEL_QWEN4, AMP_QWEN4_LAYERS=N (default 8), AMP_QWEN4_HOST_FRAC (resident fraction, default 0.4).

import * as qwen4_tp from 'ampstein.qwen4_tp';
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

function run(host) {
	let m = qwen4_tp.load(MODEL, { devices: 2, layers: NL, host_experts: host });
	let cache = m.new_cache(length(ids) + 64);

	cache.trace = { blocks: [] };
	m.forward(ids, cache, null);

	for (let t in DECODE)
		m.forward([t], cache, null);

	let out = map(cache.trace.blocks, (b) => b.cpu());

	torch.synchronize();

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
let ok = length(base) == length(hosted) && length(base) == length(base2) && length(base) > 0;
let floor = ok ? worst_rel(base, base2) : 1.0;
let diff = ok ? worst_rel(base, hosted) : 1.0;

printf('%d blocks compared, %d layers, resident fraction %.2f: run-to-run rel-rms %.2e, hosted vs VRAM %.2e\n',
	length(base), NL, FRAC, floor, diff);

if (getenv("AMP_TEST_VERBOSE"))
	for (let i = 0; i < length(base); i++)
		printf("  block %2d floor %.2e hosted %.2e\n", i, rel_rms(base[i], base2[i]), rel_rms(base[i], hosted[i]));

// the first two blocks of each forward have not amplified the atomics' rounding noise yet
let early = 0.0;

for (let i = 0; ok && i < length(base); i++)
	if (i % NL < 2 && rel_rms(base[i], hosted[i]) > early)
		early = rel_rms(base[i], hosted[i]);

printf('first two blocks of each forward: worst rel-rms %.2e\n', early);

ok = ok && diff <= 3 * floor + 1e-6 && early < 1e-3;
print(ok ? 'qwen4_host_experts ok\n' : 'qwen4_host_experts FAILED\n');
exit(ok ? 0 : 1);
