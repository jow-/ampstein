// tests/engine_inject.uc — forced tokens through the engine's force() hook
// (the server's reasoning budget): at a step boundary the generation
// continues as if the prompt had been extended by the forced tokens.
//
//   plain  job:  prompt                      -> g[0 ..]            (greedy)
//   forced job:  prompt, force F after k     -> g[0 .. k-1], then the continuation
//   ref    job:  prompt + g[0 .. k-1] + F    -> the continuation expected
//
// The forced job's continuation must equal the reference job's (the same
// computation through the live-reuse prefill; fp noise between a decode step
// and a prefill row can flip a late near-tie, so the first TOL tokens must
// match exactly). Run with the MTP window off and on: with MTP the draft
// cache must follow the injection (rounds go on afterwards). Needs a
// Qwen3.5-family checkpoint (AMP_MODEL_QWEN35) and two GPUs.
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

/* run one greedy job to completion; force_at: emit `forced` once that many
 * tokens were delivered (null: never). Returns { toks, info }. */
function run_job(eng, ids, max_new, force_at) {
	let toks = [], info = null, fired = false, at = -1;

	eng.submit({
		ids: ids, snap_at: 0, max_new: max_new, stop_ids: [],
		sampling: { temperature: 0 },
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		force: (force_at == null) ? null : function () {
			if (fired || length(toks) < force_at)
				return null;

			fired = true;
			at = length(toks);

			return forced;
		},
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	if (!run_until(() => info != null, 300000))
		die('job timed out');

	return { toks: toks, info: info, at: at };
}

for (let W in [ 0, 3 ]) {
	let eng = engine.create(model, { max_ctx: 4096, prefill_chunk: 256, mtp_window: W,
		warmup: false });
	let base = run_job(eng, prompt, K + N, null);
	let inj = run_job(eng, prompt, K + length(forced) + N, K);
	let g = slice(base.toks, 0, inj.at);
	let ref = run_job(eng, [ ...prompt, ...g, ...forced ], N, null);

	// delivered tokens of the forced job: K plain ones, then the continuation
	let head = slice(inj.toks, 0, inj.at);
	let tail = slice(inj.toks, inj.at);
	let match = 0;

	while (match < length(tail) && match < length(ref.toks) && tail[match] == ref.toks[match])
		match++;

	printf('W=%d: forced job delivered %d tokens (%d forced, %d after), reference continuation matches the first %d of %d\n',
		W, length(inj.toks), length(forced), length(tail), match, length(ref.toks));
	printf('     forced : %s\n     ref    : %s\n', join(' ', slice(tail, 0, 10)), join(' ', slice(ref.toks, 0, 10)));

	check(`W=${W}: tokens before the injection are the plain greedy ones`,
		join(',', head) == join(',', g));
	check(`W=${W}: usage counts the forced tokens`, inj.info.completion_tokens == inj.at + length(forced) + length(tail),
		`${inj.info.completion_tokens} vs ${inj.at + length(forced) + length(tail)}`);
	check(`W=${W}: the continuation equals the extended-prompt reference (first ${TOL})`,
		match >= TOL && length(tail) >= TOL);

	if (W)
		check('W=3: speculative rounds go on after the injection', inj.info.mtp && inj.info.mtp.rounds > 0,
			sprintf('%J', inj.info.mtp));
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_inject: ok\n');
