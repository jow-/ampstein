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
// The GGUF Qwen3.8-27B (AMP_MODEL_QWEN35_GGUF) on one GPU (the ROCm host), with the fp16 and the Q8 KV cache
// (--cache-mode Q8: int8 codes + scales parked as they are). AMP_STORE_Q8=1, AMP_STORE_W (MTP window).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35_GGUF') || '/data/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp';
const TOL = 6, N = 12;

if (!stat(MODEL) || torch.device_count() < 1) {
	warn('skip: needs the Qwen3.8 GGUF model and a GPU\n');
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
let G = 1, devs = [];

for (let i = 0; i < nl; i++)
	push(devs, int(i * G / nl));

let model, fails = 0;

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

{
	let Q8 = getenv('AMP_STORE_Q8') == '1';

	model = qwen35.load(MODEL, { devices: devs, mtp: true, kv_q8: Q8 });

{
	let W = +(getenv('AMP_STORE_W') || 0);

	let eng;

	// (one 24 GB card: an engine from an earlier run must be collected before the next one allocates its caches;
	// one cache mode and window per process, a second model or window does not fit next to the garbage)
	let mk = () => { eng = null; gc(); return engine.create(model, { max_ctx: 8192, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 6 }); };

	// A's generation defines the A2 / A3 prompts
	eng = mk();
	let a1 = run_job(eng, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];        // extends A's live state
	let A3 = [ ...slice(A, 0, length(A) - 20), ...TAIL, ...TAIL ];   // extends the snapshot only

	// independent references: fresh engine states
	eng = mk();
	let r2 = run_job(eng, A2, 0);
	eng = mk();
	let r3 = run_job(eng, A3, 0);

	// the live continuation of A (no parking): what a lossless restore must reproduce
	eng = mk();
	run_job(eng, A, length(A) - 20);

	let l2 = run_job(eng, A2, 0);

	// the scenario: A, then B (A moves to RAM), then A2 / A3 from RAM
	eng = mk();
	run_job(eng, A, length(A) - 20);
	let b = run_job(eng, B, length(B) - 20);

	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: B starts fresh and A's state goes to RAM`,
		b.info.reuse == 'fresh' && b.info.reuse_detail && b.info.reuse_detail.stored,
		sprintf('%J', b.info.reuse_detail));
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: the store holds one entry`, eng.cached().stored == 1, sprintf('%J', eng.cached()));

	let x2 = run_job(eng, A2, 0);

	printf('W=%d: A2 reuse %s, cached %d of %d\n', W, x2.info.reuse, x2.info.cached_tokens, length(A2));
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: A2 swaps A's live state back from RAM`,
		x2.info.reuse == 'ram' && x2.info.cached_tokens >= length(A),
		`${x2.info.reuse} ${x2.info.cached_tokens}`);
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: A2 continuation equals the live (unparked) continuation`,
		same_head(x2.toks, l2.toks) == length(l2.toks), `${same_head(x2.toks, l2.toks)} of ${length(l2.toks)}`);
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: A2 continuation equals the fresh reference`, same_head(x2.toks, r2.toks) >= TOL,
		`${same_head(x2.toks, r2.toks)} of ${length(r2.toks)}`);

	// A2's own state is now live; B's moved to RAM; A3 needs A's snapshot, which
	// travelled in A's entry (the entry was consumed by A2: it is gone) — so the
	// second reuse point is exercised on a fresh sequence: A, B, A3
	eng = mk();
	run_job(eng, A, length(A) - 20);
	run_job(eng, B, length(B) - 20);

	let x3 = run_job(eng, A3, 0);

	printf('W=%d: A3 reuse %s, cached %d of %d\n', W, x3.info.reuse, x3.info.cached_tokens, length(A3));
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: A3 restores A's snapshot from RAM`,
		x3.info.reuse == 'ram' && x3.info.cached_tokens == length(A) - 20,
		`${x3.info.reuse} ${x3.info.cached_tokens}`);
	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: A3 continuation equals the fresh reference`, same_head(x3.toks, r3.toks) >= TOL,
		`${same_head(x3.toks, r3.toks)} of ${length(r3.toks)}`);

	// and B is still reachable afterwards (it was stored when A3 took over)
	let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

	check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: B comes back from RAM too`, b2.info.reuse == 'ram' && b2.info.cached_tokens >= length(B) - 20,
		`${b2.info.reuse} ${b2.info.cached_tokens} ${sprintf('%J', b2.info.reuse_detail)} ${sprintf('%J', eng.cached())}`);

	// hot reload: A's live state goes to a file and comes back in a new engine (without the MTP draft cache, whose
	// state is the same fp16 one in both modes: a second engine with it does not fit next to the 27B)
	if (W == 0) {
		let path = '/tmp/engine_store_gguf_state.st';

		eng = mk();

		let h1 = run_job(eng, A, length(A) - 20);
		let H2 = [ ...A, ...h1.toks, ...TAIL ];
		let n = eng.save_state(path);

		check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: hot reload: the state is saved`, n >= length(A), n);

		let hr = run_job(eng, H2, 0);

		eng = mk();
		check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: hot reload: the state is restored`, eng.restore_state(path) == n);

		let hx = run_job(eng, H2, 0);

		check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: hot reload: H2 continues the restored live state`,
			hx.info.reuse == 'live' && hx.info.cached_tokens >= length(A), `${hx.info.reuse} ${hx.info.cached_tokens}`);
		check(`${Q8 ? 'Q8' : 'fp16'} W=${W}: hot reload: the continuation equals the original engine's`,
			same_head(hx.toks, hr.toks) == length(hr.toks), `${same_head(hx.toks, hr.toks)} of ${length(hr.toks)}`);
	}
}
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store_gguf: ok\n');
