// tests/engine_store_gemma4.uc — the engine's snapshot / host-RAM store / hot reload for Gemma 4: the sliding layers' ring is
// described by its last 1024 rows (the model's aux state), the global layers by their rows [0, n). Run A, run B (A's state
// moves to RAM), run A2 (A's live state), A3 (A's snapshot, after A generated enough tokens to wrap the rings), then a
// save_state / restore_state round trip. Continuations must follow the engine's own reuse without the store.
//
// Env: AMP_MODEL_GEMMA4, AMP_GEMMA4_GPUS (2), AMP_GEMMA4_TP (1: tensor parallel).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as gemma4 from 'ampstein.gemma4';
import * as gemma4_tp from 'ampstein.gemma4_tp';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GEMMA4') || '/data/models/gemma-4-26B-A4B-it-exl3-6.10bpw';
const NGPU = +(getenv('AMP_GEMMA4_GPUS') || 2);
const TP = !!getenv('AMP_GEMMA4_TP');
const TOL = 3, N = 12;

if (!stat(MODEL) || torch.device_count() < NGPU) {
	warn('skip: needs the Gemma 4 model and ' + NGPU + ' GPUs\n');
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
let model = gemma4.load(MODEL, { devices: NGPU, prefill_rows: 1024 });

if (TP) {
	let ranks = [];

	for (let g = 0; g < NGPU; g++)
		push(ranks, g);

	model = gemma4_tp.shard(model, { devices: ranks, prefill_rows: 1024 });
}

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

function run_job(eng, ids, snap_at, max_new) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: snap_at, max_new: max_new || N, stop_ids: [],
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

let mk = () => engine.create(model, { max_ctx: 8192, prefill_chunk: 1024, mtp_window: 0, warmup: false, cache_ram: 8 });

// A's generation defines the A2 prompt; the same engine then serves it from its live state (no store: B never intervenes)
let ref = mk();
let a1 = run_job(ref, A, length(A) - 20, 1200);          // 1200 generated tokens wrap the sliding rings
let A2 = [ ...A, ...a1.toks, ...TAIL ];
let A3 = [ ...slice(A, 0, length(A) - 20), ...TAIL, ...TAIL ];   // extends the snapshot only
let r2 = run_job(ref, A2, 0);

check('reference A2 is a live continuation', r2.info.reuse == 'live', r2.info.reuse);

// the snapshot reuse on the same engine: the rings were overwritten by the generation, the aux state brings the window back
let r3 = run_job(ref, A3, 0);

check('reference A3 is a snapshot continuation', r3.info.cached_tokens == length(A) - 20, `${r3.info.reuse} ${r3.info.cached_tokens}`);

// a fresh prefill of A3 (no reuse at all) must agree with the snapshot continuation
let fresh = run_job(mk(), A3, 0);

check('A3 snapshot continuation equals a fresh prefill', same_head(r3.toks, fresh.toks) >= TOL,
	`${same_head(r3.toks, fresh.toks)} of ${length(fresh.toks)}`);

// the scenario: A (with the long generation), then B (A moves to RAM), then A2 from RAM. The tensor-parallel kernels are not
// bit-reproducible run to run (greedy near-ties flip), so A2 is built from THIS engine's own generation and judged against
// a fresh prefill of it
let eng = mk();
let ax = run_job(eng, A, length(A) - 20, 1200);
let A2x = [ ...A, ...ax.toks, ...TAIL ];

let b = run_job(eng, B, length(B) - 20);

check('B starts fresh and A\'s state goes to RAM', b.info.reuse == 'fresh' && b.info.reuse_detail && b.info.reuse_detail.stored,
	sprintf('%J', b.info.reuse_detail));

let x2 = run_job(eng, A2x, 0);

printf('A2 reuse %s, cached %d of %d\n', x2.info.reuse, x2.info.cached_tokens, length(A2x));
check('A2 swaps A\'s live state back from RAM', x2.info.reuse == 'ram' && x2.info.cached_tokens >= length(A2x) - length(TAIL) - 12,
	`${x2.info.reuse} ${x2.info.cached_tokens}`);

let f2 = run_job(mk(), A2x, 0);

check('A2 continuation equals a fresh prefill', same_head(x2.toks, f2.toks) >= TOL, `${same_head(x2.toks, f2.toks)} of ${length(f2.toks)}`);

// A, B, A3: A's snapshot travelled in its entry
eng = mk();
run_job(eng, A, length(A) - 20, 1200);
run_job(eng, B, length(B) - 20);

let x3 = run_job(eng, A3, 0);

printf('A3 reuse %s, cached %d of %d\n', x3.info.reuse, x3.info.cached_tokens, length(A3));
check('A3 restores A\'s snapshot from RAM', x3.info.reuse == 'ram' && x3.info.cached_tokens == length(A) - 20,
	`${x3.info.reuse} ${x3.info.cached_tokens}`);
check('A3 continuation equals the snapshot-reuse reference', same_head(x3.toks, r3.toks) >= TOL,
	`${same_head(x3.toks, r3.toks)} of ${length(r3.toks)}`);

let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

check('B comes back from RAM too', b2.info.reuse == 'ram' && b2.info.cached_tokens >= length(B) - 20,
	`${b2.info.reuse} ${b2.info.cached_tokens}`);

// hot reload: the live state of A goes to a file and comes back in a new engine
{
	let path = '/tmp/engine_store_gemma4_state' + (TP ? '_tp' : '') + '.st';
	let e1 = mk();
	let a1 = run_job(e1, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];
	let n = e1.save_state(path);

	check('hot reload: the state is saved', n >= length(A), n);

	let e2 = mk();

	check('hot reload: the state is restored', e2.restore_state(path) == n);

	let r = run_job(e1, A2, 0);
	let x = run_job(e2, A2, 0);

	check('hot reload: A2 continues the restored live state', x.info.reuse == 'live' && x.info.cached_tokens >= length(A),
		`${x.info.reuse} ${x.info.cached_tokens}`);
	check('hot reload: the continuation equals the original engine\'s', same_head(x.toks, r.toks) >= TOL,
		`${same_head(x.toks, r.toks)} of ${length(r.toks)}`);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store_gemma4: ok\n');
