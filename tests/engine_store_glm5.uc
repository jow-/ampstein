// tests/engine_store_glm5.uc — the engine's host-RAM prompt-cache store for GLM-5.3-Flash:
// MLA latent rows + indexer plane / pools (fp16 or Q8, age tiers), KDA states, the MTP draft
// cache. As engine_store.uc: run A, run B (A's state moves to RAM), run A2 (A's live state)
// and A3 (A's snapshot); the continuation must follow a fresh-engine reference.
//
// Env: AMP_MODEL_GLM5, AMP_GLM5_GPUS (6), AMP_GLM5_STAGES (1, tensor-parallel stages),
// AMP_GLM5_SPLIT=1 (layer split on glm5.load instead of TP), AMP_GLM5_KV_BITS (0 | 8).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as glm5 from 'ampstein.glm5';
import * as glm5_tp from 'ampstein.glm5_tp';
import { stat } from 'fs';

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

// The reference is the engine's own reuse without the store (live continuation for A2, the
// snapshot for A3): a fresh prefill of the same prompt is not bit-identical run to run (a
// near-tie token flips), a restored state is. Engines hold GPU memory for the process
// lifetime (a 6-GPU, 3-bit GLM-5.3 fits about six), so the scenarios share them.
// (only the speculative variant: it adds the draft cache to everything the plain one stores)
for (let W in [ 3 ]) {
	let mk = () => engine.create(model, { max_ctx: 4096, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 8, slots: 0 });

	// A's generation defines the A2 / A3 prompts; the same engine then serves them from its
	// live / snapshot state (no store: B never intervenes)
	let ref = mk();
	let eng;
	let a1 = run_job(ref, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];        // extends A's live state
	let A3 = [ ...slice(A, 0, length(A) - 20), ...TAIL, ...TAIL ];   // extends the snapshot only
	let r2 = run_job(ref, A2, 0);

	check(`W=${W}: reference A2 is a live continuation`, r2.info.reuse == 'live', r2.info.reuse);

	// the scenario: A, then B (A moves to RAM), then A2 from RAM
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
	check(`W=${W}: A2 continuation equals the live-reuse reference`, same_head(x2.toks, r2.toks) >= TOL,
		`${same_head(x2.toks, r2.toks)} of ${length(r2.toks)}`);

	// the second reuse point: A3 extends A's snapshot only. The reference is the snapshot
	// reuse on the engine that served A and A2 (its snapshot of A is still there)
	let r3 = run_job(ref, A3, 0);

	check(`W=${W}: reference A3 is a snapshot continuation`, r3.info.cached_tokens == length(A) - 20,
		`${r3.info.reuse} ${r3.info.cached_tokens}`);

	// on a fresh sequence: A, B, A3 (A's snapshot travelled in its entry)
	eng = mk();
	run_job(eng, A, length(A) - 20);
	run_job(eng, B, length(B) - 20);

	let x3 = run_job(eng, A3, 0);

	printf('W=%d: A3 reuse %s, cached %d of %d\n', W, x3.info.reuse, x3.info.cached_tokens, length(A3));
	check(`W=${W}: A3 restores A's snapshot from RAM`,
		x3.info.reuse == 'ram' && x3.info.cached_tokens == length(A) - 20,
		`${x3.info.reuse} ${x3.info.cached_tokens}`);
	check(`W=${W}: A3 continuation equals the snapshot-reuse reference`, same_head(x3.toks, r3.toks) >= TOL,
		`${same_head(x3.toks, r3.toks)} of ${length(r3.toks)}`);

	// and B is still reachable afterwards (it was stored when A3 took over)
	let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

	check(`W=${W}: B comes back from RAM too`, b2.info.reuse == 'ram' && b2.info.cached_tokens >= length(B) - 20,
		`${b2.info.reuse} ${b2.info.cached_tokens} ${sprintf('%J', b2.info.reuse_detail)}`);
}

// resident device slots: A's state parks on the device and comes back from there; with one
// slot, a second parked state demotes the first to host RAM, from where it still restores
{
	let eng = engine.create(model, { max_ctx: 4096, prefill_chunk: 512, mtp_window: 3,
		warmup: false, cache_ram: 8, slots: 1 });

	check('slots: one slot allocated', eng.cached().slots == 1, sprintf('%J', eng.cached()));

	let a1 = run_job(eng, A, length(A) - 20);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];

	run_job(eng, B, length(B) - 20);

	let x = run_job(eng, A2, 0);

	check('slots: A2 swaps A back from a device slot',
		x.info.reuse == 'ram' && x.info.cached_tokens >= length(A) && x.info.reuse_detail.ram_tier == 'device',
		sprintf('%s %d %J', x.info.reuse, x.info.cached_tokens, x.info.reuse_detail));

	// B sits in the slot (parked when A2 took over); parking A2 demotes B to host RAM
	run_job(eng, [ ...A, ...TAIL, ...TAIL ], 0);

	let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

	check('slots: a demoted entry restores from host RAM', b2.info.reuse == 'ram' &&
		b2.info.cached_tokens >= length(B) - 20, sprintf('%s %d %J', b2.info.reuse, b2.info.cached_tokens, b2.info.reuse_detail));
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store_glm5: ok\n');
