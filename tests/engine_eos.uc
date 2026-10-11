// tests/engine_eos.uc — the end-of-turn hook (job.on_eos): when the model
// samples a stop id the consumer may let it end the generation (true),
// redraw the position with the stop ids' logits lowered (a number) or force
// tokens (an array). Greedy jobs, MTP window off and on (a stop id accepted
// as a draft mid-round must end the round there, rolled back consistently).
//
// The stop id is a token the model emits naturally (g[k] of a plain run), so
// the sequence is deterministic:
//   accept   -> g[0 .. k-1], reason 'stop'
//   bias     -> g[0 .. k-1], then a token != g[k] that is the runner-up
//   force F  -> g[0 .. k-1], then the continuation of prompt + g[0 .. k-1] + F
// Needs a Qwen3.5-family checkpoint (AMP_MODEL_QWEN35) and two GPUs.
import * as uloop from 'uloop';
import * as math from 'math';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-6.00bpw';
const TOL = 6, K = 4, N = 14;

if (!stat(MODEL) || torch.device_count() < 2) {
	warn('skip: needs the Qwen model and two GPUs\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

function run_until(cond, timeout_ms) {
	let deadline = now_ms() + timeout_ms;

	while (!cond()) {
		if (now_ms() > deadline)
			return false;

		uloop.run(20);
	}

	return true;
}

uloop.init();

let tok = torch.tokenizer(MODEL);
let raw = json(readfile(MODEL + '/config.json'));
let nl = (raw.text_config || raw).num_hidden_layers;
let per = int((nl + 1) / 2), devs = [];

for (let i = 0; i < nl; i++)
	push(devs, (int(i / per) >= 2) ? 1 : int(i / per));

let model = qwen35.load(MODEL, { devices: devs, mtp: true });
let prompt = tok.encode('The three largest cities of France are Paris, Marseille and', false);
let forced = tok.encode(' Lyon, and the capital is', false);
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

function run_job(eng, ids, max_new, stop_ids, on_eos, eos_max) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: 0, max_new: max_new, stop_ids: stop_ids,
		sampling: { temperature: 0 },
		on_eos: on_eos, eos_max: eos_max,
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	if (!run_until(() => info != null, 300000))
		die('job timed out');

	return { toks: toks, info: info };
}

function same(a, b, n) {
	for (let i = 0; i < n; i++)
		if (a[i] != b[i])
			return false;

	return length(a) >= n && length(b) >= n;
}

for (let W in [ 0, 3 ]) {
	let eng = engine.create(model, { max_ctx: 4096, prefill_chunk: 256, mtp_window: W,
		warmup: false });
	let base = run_job(eng, prompt, 24, [], null);

	for (let k in [ 2, 5, 9 ]) {
		let g = slice(base.toks, 0, k);
		let stop = base.toks[k];
		let tag = `W=${W} k=${k}`;

		// a stop id that also occurs earlier would end the run sooner
		if (index(g, stop) >= 0) {
			warn(`skip ${tag}: stop id repeats earlier\n`);
			continue;
		}

		// no hook, hook accepting
		let r = run_job(eng, prompt, 24, [ stop ], null);

		check(`${tag}: no hook: ends at the stop id`, join(',', r.toks) == join(',', g) && r.info.reason == 'stop',
			join(',', r.toks) + ' ' + r.info.reason);

		let asked = [];

		r = run_job(eng, prompt, 24, [ stop ], (n) => { push(asked, n); return true; });
		check(`${tag}: hook returning true: ends at the stop id`,
			join(',', r.toks) == join(',', g) && r.info.reason == 'stop' && join(',', asked) == '0',
			join(',', r.toks) + ' ' + r.info.reason + ' asked ' + join(',', asked));

		// junk verdicts are accepts
		for (let v in [ null, false, '', 0, -3, {} ]) {
			r = run_job(eng, prompt, 24, [ stop ], () => v);
			check(`${tag}: verdict ${sprintf('%J', v)} is an accept`, join(',', r.toks) == join(',', g) && r.info.reason == 'stop');
		}

		// a huge bias: the runner-up replaces the stop id; the cap ends the second consultation
		r = run_job(eng, prompt, 24, [ stop ], (n) => 1000, 1);
		check(`${tag}: bias: the position is redrawn`, length(r.toks) > k && same(r.toks, g, k) && r.toks[k] != stop,
			join(',', r.toks));
		check(`${tag}: bias: the generation went on`, length(r.toks) >= k + 4, `${length(r.toks)} tokens, ${r.info.reason}`);

		// force
		r = run_job(eng, prompt, k + length(forced) + 10, [ stop ], (n) => (n == 0) ? forced : true);

		// (the stop id may be the very next token: the reference ends there too)
		let ref = run_job(eng, [ ...prompt, ...g, ...forced ], 10, [ stop ], null);
		let tail = slice(r.toks, k);
		let n = (length(ref.toks) < TOL) ? length(ref.toks) : TOL;

		check(`${tag}: force: tokens before are the plain ones`, same(r.toks, g, k));
		check(`${tag}: force: the continuation equals the extended-prompt reference (first ${TOL}, or up to the stop)`,
			length(tail) >= n && same(tail, ref.toks, n) &&
			(ref.info.reason != 'stop' || length(tail) == length(ref.toks)),
			join(' ', slice(tail, 0, 8)) + ' | ' + join(' ', slice(ref.toks, 0, 8)) + ' ' + ref.info.reason);
		check(`${tag}: force: usage counts the forced tokens`,
			r.info.completion_tokens == k + length(forced) + length(tail),
			`${r.info.completion_tokens} vs ${k + length(forced) + length(tail)}`);

		if (W)
			check(`${tag}: force: speculative rounds go on afterwards`, r.info.mtp && r.info.mtp.rounds > 0);
	}
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_eos: ok\n');
