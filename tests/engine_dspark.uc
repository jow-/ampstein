// tests/engine_dspark.uc — DSpark speculative decoding through the engine (dsv4_tp + dspark_tp):
// a greedy job with the drafter must deliver the plain greedy tokens (fp near-ties may flip a
// late token), a sampled job must run, and the usage / stop handling must be consistent.
// Env: DSV4_LAYERS (0 = the full model; a subset checks the plumbing only — the taps come from
// layers 40-42, so the drafter is meaningless there but every code path runs).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as dspark_tp from 'ampstein.dspark_tp';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const NL = +(getenv('AMP_DSV4_LAYERS') || 0);
const N = +(getenv('AMP_ED_GEN') || 48);

if (!stat(MODEL)) {
	warn('skip: DSV4 model dir missing\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

uloop.init();

let tok = torch.tokenizer(MODEL);
let model = dsv4_tp.load(MODEL, { layers: NL || null, devices: 6, stages: 3, drafter: true, prefill_rows: 512 });

model.dspark = dspark_tp.load(model, { max_seq: 4096 });

let eng = engine.create(model, { max_ctx: 4096, prefill_chunk: 512, dspark_window: 3, warmup: false });
let prompt = tok.encode('<｜begin▁of▁sentence｜><｜User｜>Write a Python function that merges two sorted lists into one sorted list.<｜Assistant｜></think>', true);
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

function run_job(opts) {
	let toks = [], info = null;
	let t0 = now_ms();

	eng.submit({
		ids: prompt, snap_at: 0, max_new: N, stop_ids: [], no_dspark: opts.plain,
		sampling: opts.sampling,
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	let deadline = now_ms() + 600000;

	while (info == null && now_ms() < deadline)
		uloop.run(20);

	if (info == null)
		die('job timed out');

	return { toks: toks, info: info, ms: now_ms() - t0 };
}

let plain = run_job({ plain: true, sampling: { temperature: 0 } });
let spec = run_job({ plain: false, sampling: { temperature: 0 } });
let same = 0;

while (same < length(plain.toks) && same < length(spec.toks) && plain.toks[same] == spec.toks[same])
	same++;

printf('plain: %d tokens in %.0f ms; speculative: %d tokens in %.0f ms; identical prefix %d; rounds %J\n',
	length(plain.toks), plain.ms, length(spec.toks), spec.ms, same, spec.info.mtp);

check('plain job finished by length', plain.info.reason == 'length');
check('speculative job finished by length', spec.info.reason == 'length');
check('both deliver max_new tokens', length(plain.toks) == N && length(spec.toks) == N);
check('speculative greedy output follows plain greedy', same >= (NL ? 1 : 16), `${same} of ${N}`);
check('usage counts the delivered tokens', spec.info.completion_tokens == N);
check('the drafter ran', spec.info.mtp != null && spec.info.mtp.rounds > 0);

let smp = run_job({ plain: false, sampling: { temperature: 0.8, top_p: 0.95, seed: 7 } });

check('a sampled job delivers max_new tokens', length(smp.toks) == N && smp.info.reason == 'length');

// stop inside a round: stop on the 5th plain token of the base run
let stop_tok = plain.toks[4];
let stopped = null, stoks = [];

eng.submit({
	ids: prompt, snap_at: 0, max_new: N, stop_ids: [ stop_tok ], sampling: { temperature: 0 },
	alive: () => true, on_progress: () => true,
	on_token: function (t) { push(stoks, t); return true; },
	on_done: function (i) { stopped = i; }, on_error: function (m) { stopped = { reason: 'error' }; }
});

while (stopped == null)
	uloop.run(20);

let first_stop = 0;

while (first_stop < N && plain.toks[first_stop] != stop_tok)
	first_stop++;

check('stop token ends the speculative job at the same place', stopped.reason == 'stop' && length(stoks) == first_stop,
	`${stopped.reason}, ${length(stoks)} tokens, expected ${first_stop}`);

// the cache after the stop must still serve a following request correctly
let again = run_job({ plain: false, sampling: { temperature: 0 } });

// (a layer slice generates noise-level text where near-ties flip with the verify window: 4 tokens)
let nchk = NL ? 4 : 8;

check('a request after the stop reproduces the output', join(',', slice(again.toks, 0, nchk)) == join(',', slice(plain.toks, 0, nchk)),
	join(',', slice(again.toks, 0, nchk)));

print(fails ? 'engine_dspark FAILED\n' : 'engine_dspark ok\n');
exit(fails ? 1 : 0);
