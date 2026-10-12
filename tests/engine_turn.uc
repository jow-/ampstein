// tests/engine_turn.uc — the engine's turn checkpoint.
//
// A conversation is P (up to the end of the last user message, the turn
// boundary T = |P|) followed by a tool loop L. When the next request carries
// a NEW user message, the template re-renders the loop differently (earlier
// reasoning dropped), so the prompt P + L' + U diverges from the live state
// right after P — before the end-of-prompt snapshot, which a recurrent
// model cannot rewind to. The turn checkpoint taken at T (job.turn_at) lets
// the engine resume there:
//
//   in engine : P+L (turn_at = T), P+L+L2 (live), P+L'+U -> reuse 'turn',
//               cached == T
//   parked    : the same, with an unrelated conversation in between (the
//               state and its checkpoint travel through host RAM) -> 'ram',
//               cached == T
//
// The continuation must equal a fresh engine's (first TOL tokens: fp noise
// can flip a late near-tie). Needs a Qwen3.5-family checkpoint
// (AMP_MODEL_QWEN35) and two GPUs.
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const TOL = 6, N = 12;

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
let devs = [];

for (let i = 0; i < nl; i++)
	push(devs, int(i * 2 / nl));

let model = qwen35.load(MODEL, { devices: devs, mtp: true });
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

function text(seed, n) {
	let s = '';

	for (let i = 0; i < n; i++)
		s += `Item ${seed}-${i}: the ${seed} committee reviewed record number ${i * 7 + length(seed)} and noted nothing unusual. `;

	return s;
}

function enc(s) {
	return tok.encode(s, false);
}

function run_job(eng, ids, snap_at, turn_at) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: snap_at, turn_at: turn_at, max_new: N, stop_ids: [],
		sampling: { temperature: 0 },
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	if (!run_until(() => info != null, 600000))
		die('job timed out');

	return { toks: toks, info: info };
}

function same_head(a, b) {
	let k = 0;

	while (k < length(a) && k < length(b) && a[k] == b[k])
		k++;

	return k;
}

let P = enc(text('alpha', 120));          // up to the end of the user message
let L = enc(text('loop-one', 40));        // the tool loop as the first request saw it
let L2 = enc(text('loop-two', 15));       // more of the loop (live continuation)
let Lp = enc(text('rewritten', 40));      // the same turns re-rendered (diverges at |P|)
let U = enc(' Actually stop and summarise in one sentence:');
let B = enc(text('bravo', 120));
let T = length(P);

printf('P %d tokens, L %d, L2 %d, Lp %d\n', T, length(L), length(L2), length(Lp));

let X = [ ...P, ...Lp, ...U ];            // the interrupting request

for (let W in [ 0, 3 ]) {
	let mk = () => engine.create(model, { max_ctx: 8192, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 6 });

	// reference: a fresh engine
	let eng = mk();
	let ref = run_job(eng, X, 0, 0);

	// in engine: P+L, P+L+L2 (live continuation), then X
	eng = mk();

	let a = run_job(eng, [ ...P, ...L ], T + length(L), T);

	check(`W=${W}: first request starts fresh`, a.info.reuse == 'fresh', a.info.reuse);

	let b = run_job(eng, [ ...P, ...L, ...a.toks, ...L2 ], T + length(L) + length(a.toks) + length(L2), T);

	check(`W=${W}: the loop continues from the live state or the snapshot`,
		b.info.reuse == 'live' || b.info.reuse == 'snapshot', b.info.reuse);

	let x = run_job(eng, X, length(X) - length(U), T);

	printf('W=%d: X reuse %s, cached %d of %d\n', W, x.info.reuse, x.info.cached_tokens, length(X));
	check(`W=${W}: X resumes at the turn checkpoint`,
		x.info.reuse == 'turn' && x.info.cached_tokens == T,
		`${x.info.reuse} ${x.info.cached_tokens} ${sprintf('%J', x.info.reuse_detail)}`);
	check(`W=${W}: X continuation equals the fresh reference`, same_head(x.toks, ref.toks) >= TOL,
		`${same_head(x.toks, ref.toks)} of ${length(ref.toks)}`);

	// parked: P+L, then an unrelated conversation (the state goes to RAM with its
	// checkpoint), then X
	eng = mk();
	run_job(eng, [ ...P, ...L ], T + length(L), T);

	let o = run_job(eng, B, length(B) - 10, length(B) - 10);

	check(`W=${W}: the unrelated conversation parks the first`,
		o.info.reuse == 'fresh' && o.info.reuse_detail && o.info.reuse_detail.stored,
		sprintf('%J', o.info.reuse_detail));

	let y = run_job(eng, X, length(X) - length(U), T);

	printf('W=%d: parked X reuse %s, cached %d of %d\n', W, y.info.reuse, y.info.cached_tokens, length(X));
	check(`W=${W}: parked X resumes at the checkpoint from RAM`,
		y.info.reuse == 'ram' && y.info.cached_tokens == T,
		`${y.info.reuse} ${y.info.cached_tokens}`);
	check(`W=${W}: parked X continuation equals the fresh reference`, same_head(y.toks, ref.toks) >= TOL,
		`${same_head(y.toks, ref.toks)} of ${length(ref.toks)}`);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_turn: ok\n');
