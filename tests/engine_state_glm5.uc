// tests/engine_state_glm5.uc — hot reload for GLM-5.3-Flash: save_state() after A, a fresh
// engine restore_state()s it (a server restart), and A2 (extends A's live state) and A3
// (extends A's snapshot only) continue as on the engine that never stopped.
//
// Env: AMP_MODEL_GLM5, AMP_GLM5_GPUS (6), AMP_GLM5_STAGES (1, tensor-parallel stages),
// AMP_GLM5_SPLIT=1 (layer split on glm5.load instead of TP), AMP_GLM5_KV_BITS (0 | 8).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as glm5 from 'ampstein.glm5';
import * as glm5_tp from 'ampstein.glm5_tp';
import { stat, unlink } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const NGPU = +(getenv('AMP_GLM5_GPUS') || 6);
const KVB = +(getenv('AMP_GLM5_KV_BITS') || 0) || null;
const TOL = 8, N = 12;

if (!stat(MODEL) || torch.device_count() < NGPU) {
	warn('skip: needs the GLM-5.3 model and ' + NGPU + ' GPUs\n');
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
let model = getenv('AMP_GLM5_SPLIT') ?
	glm5.load(MODEL, { devices: NGPU, max_seq: 4096, kv_bits: KVB }) :
	glm5_tp.load(MODEL, { devices: NGPU, stages: +(getenv('AMP_GLM5_STAGES') || 1), max_seq: 4096, kv_bits: KVB });
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

let tmp = '/tmp/ampstein-test-glm5-state.safetensors';

for (let W in [ 3 ]) {
	let mk = () => engine.create(model, { max_ctx: 4096, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 0, slots: 0 });

	// the engine that never stops: A, then its own live / snapshot continuations
	let ref = mk();
	let a1 = run_job(ref, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];
	let A3 = [ ...slice(A, 0, length(A) - 20), ...TAIL, ...TAIL ];

	// the server that stops after A and starts again
	let first = mk();

	run_job(first, A, length(A) - 20);

	let n = first.save_state(tmp);

	check(`W=${W}: save_state reports the tokens`, n >= length(A), `${n} vs ${length(A)}`);

	let second = mk();
	let m = second.restore_state(tmp);

	check(`W=${W}: restore_state reports the tokens`, m == n, `${m} vs ${n}`);

	let r2 = run_job(ref, A2, 0);
	let x2 = run_job(second, A2, 0);

	check(`W=${W}: A2 is a live continuation after the restart`, x2.info.reuse == 'live' &&
		x2.info.cached_tokens >= length(A), `${x2.info.reuse} ${x2.info.cached_tokens}`);
	check(`W=${W}: A2 continuation equals the never-stopped engine`, same_head(x2.toks, r2.toks) >= TOL,
		`${same_head(x2.toks, r2.toks)} of ${length(r2.toks)}`);

	// the snapshot travelled too: a second restart, then the prompt that extends it only
	let third = mk();

	third.restore_state(tmp);

	let r3 = run_job(ref, A3, 0);
	let x3 = run_job(third, A3, 0);

	check(`W=${W}: A3 rewinds to A's snapshot after the restart`, x3.info.cached_tokens == length(A) - 20,
		`${x3.info.reuse} ${x3.info.cached_tokens}`);
	check(`W=${W}: A3 continuation equals the never-stopped engine`, same_head(x3.toks, r3.toks) >= TOL,
		`${same_head(x3.toks, r3.toks)} of ${length(r3.toks)}`);

	unlink(tmp);
	unlink(tmp + '.tree');
}

if (fails)
	die(`${fails} check(s) failed`);
