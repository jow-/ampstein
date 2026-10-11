// tests/engine_store.uc — the engine's host-RAM prompt-cache store.
//
//   fresh  : A2 on an empty engine                          -> reference tokens
//   run A, run B (an unrelated prompt: A's state moves to RAM), run A2 (A
//   extended): the engine swaps A's state back ('ram') and prefills only the
//   tail; its greedy output must equal the reference (first TOL tokens: fp noise
//   between a prefill row and a decode step can flip a late near-tie).
//
// Both reuse points of a stored state are exercised: A2 = A + A's generation +
// more (the live state) and A3 = A + other text (the snapshot taken at snap_at).
// Needs a Qwen3.5-family checkpoint (AMP_MODEL_QWEN35) and two GPUs.
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-6.00bpw';
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
let G = +(getenv('AMP_STORE_GPUS') || 2), devs = [];

for (let i = 0; i < nl; i++)
	push(devs, int(i * G / nl));

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

let A = tok.encode(text('alpha', 150), false);
let B = tok.encode(text('bravo', 150), false);
let TAIL = tok.encode(' Summarize the findings in one sentence:', false);

printf('prompt tokens: A %d, B %d\n', length(A), length(B));

function run_job(eng, ids, snap_at) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: snap_at, max_new: N, stop_ids: [],
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

for (let W in [ 0, 3 ]) {
	let mk = () => engine.create(model, { max_ctx: 8192, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 6 });

	// A's generation defines the A2 / A3 prompts
	let eng = mk();
	let a1 = run_job(eng, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];        // extends A's live state
	let A3 = [ ...slice(A, 0, length(A) - 20), ...TAIL, ...TAIL ];   // extends the snapshot only

	// independent references: fresh engine states
	eng = mk();
	let r2 = run_job(eng, A2, 0);
	eng = mk();
	let r3 = run_job(eng, A3, 0);

	// the scenario: A, then B (A moves to RAM), then A2 / A3 from RAM
	eng = mk();
	run_job(eng, A, length(A) - 20);
	let b = run_job(eng, B, length(B) - 20);

	check(`W=${W}: B starts fresh and A's state goes to RAM`,
		b.info.reuse == 'fresh' && b.info.reuse_detail && b.info.reuse_detail.stored,
		sprintf('%J', b.info.reuse_detail));
	check(`W=${W}: the store holds one entry`, eng.cached().stored == 1, sprintf('%J', eng.cached()));

	let x2 = run_job(eng, A2, 0);

	printf('W=%d: A2 reuse %s, cached %d of %d\n', W, x2.info.reuse, x2.info.cached_tokens, length(A2));
	check(`W=${W}: A2 swaps A's live state back from RAM`,
		x2.info.reuse == 'ram' && x2.info.cached_tokens >= length(A),
		`${x2.info.reuse} ${x2.info.cached_tokens}`);
	check(`W=${W}: A2 continuation equals the fresh reference`, same_head(x2.toks, r2.toks) >= TOL,
		`${same_head(x2.toks, r2.toks)} of ${length(r2.toks)}`);

	// A2's own state is now live; B's moved to RAM; A3 needs A's snapshot, which
	// travelled in A's entry (the entry was consumed by A2: it is gone) — so the
	// second reuse point is exercised on a fresh sequence: A, B, A3
	eng = mk();
	run_job(eng, A, length(A) - 20);
	run_job(eng, B, length(B) - 20);

	let x3 = run_job(eng, A3, 0);

	printf('W=%d: A3 reuse %s, cached %d of %d\n', W, x3.info.reuse, x3.info.cached_tokens, length(A3));
	check(`W=${W}: A3 restores A's snapshot from RAM`,
		x3.info.reuse == 'ram' && x3.info.cached_tokens == length(A) - 20,
		`${x3.info.reuse} ${x3.info.cached_tokens}`);
	check(`W=${W}: A3 continuation equals the fresh reference`, same_head(x3.toks, r3.toks) >= TOL,
		`${same_head(x3.toks, r3.toks)} of ${length(r3.toks)}`);

	// and B is still reachable afterwards (it was stored when A3 took over)
	let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

	check(`W=${W}: B comes back from RAM too`, b2.info.reuse == 'ram' && b2.info.cached_tokens >= length(B) - 20,
		`${b2.info.reuse} ${b2.info.cached_tokens} ${sprintf('%J', b2.info.reuse_detail)} ${sprintf('%J', eng.cached())}`);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store: ok\n');
