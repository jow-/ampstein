// tests/engine_store_dsv4.uc — the engine's host-RAM prompt-cache store for DeepSeek-V4
// (dsv4_tp + dspark_tp; engine hooks model.cache_export / cache_import).
//
//   run A, run B (an unrelated prompt: A's state moves to RAM), run A2 (A extended):
//   the engine swaps A's live state back (reuse 'ram') and prefills only the tail; its
//   greedy output must follow the same request served from the live state without the store.
//
// Both reuse points of a stored state are exercised: A2 = A + A's generation + more (the
// live state) and A3 = A + other text (the snapshot taken at snap_at). The drafter's main-kv
// rings travel with the entry, so the speculative run (dw=3) must also accept drafts again.
//
// The engine's prefill is not bit-deterministic run to run (the same prompt prefilled twice
// differs by fp noise, a near-tie token can flip), so the continuation checks are loose; the
// numeric equivalence of a restored state to a fresh one is tests/dsv4_store_state.uc.
//
// Env: AMP_DSV4_LAYERS (0 = the full model; a subset needs enough layers for every stage),
// AMP_STORE_DEVICES (6), AMP_STORE_STAGES (3).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as dspark_tp from 'ampstein.dspark_tp';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const NL = +(getenv('AMP_DSV4_LAYERS') || 0);
const ND = +(getenv('AMP_STORE_DEVICES') || 6);
const NS = +(getenv('AMP_STORE_STAGES') || 3);
const TOL = NL ? 2 : 4, N = 12;
const BOS = '<｜begin▁of▁sentence｜>';

if (!stat(MODEL) || torch.device_count() < ND) {
	warn('skip: needs the DSV4 model and ' + ND + ' GPUs\n');
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
let model = dsv4_tp.load(MODEL, { layers: NL || null, devices: ND, stages: NS, drafter: true, prefill_rows: 512 });

model.dspark = dspark_tp.load(model, { max_seq: 8192 });

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
	let s = BOS;

	for (let i = 0; i < n; i++)
		s += `Item ${seed}-${i}: the ${seed} committee reviewed record number ${i * 7 + length(seed)} and noted nothing unusual. `;

	return s;
}

let A = tok.encode(text('alpha', 150), true);
let B = tok.encode(text('bravo', 150), true);
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

	if (!run_until(() => info != null, 900000))
		die('job timed out');

	return { toks: toks, info: info };
}

function same_head(a, b) {
	let k = 0;

	while (k < length(a) && k < length(b) && a[k] == b[k])
		k++;

	return k;
}

for (let DW in [ 0, 3 ]) {
	// (slots: 0 — this test is about the host tier; the device tier is engine_slots_dsv4.uc, and
	// several engines each auto-allocating slots would not fit)
	let mk = (ram) => engine.create(model, { max_ctx: 8192, prefill_chunk: 512, dspark_window: DW,
		warmup: false, cache_ram: ram, slots: 0 });
	let SNAP = length(A) - 20;

	// the scenario with the store: A, then B (A moves to RAM), then A2 (A's live state swaps back)
	let eng = mk(6);
	let a1 = run_job(eng, A, SNAP);
	let A2 = [ ...A, ...a1.toks, ...TAIL ];        // extends A's live state
	let A3 = [ ...slice(A, 0, SNAP), ...TAIL, ...TAIL ];   // extends the snapshot only
	let b = run_job(eng, B, length(B) - 20);

	check(`dw=${DW}: B starts fresh and A's state goes to RAM`,
		b.info.reuse == 'fresh' && b.info.reuse_detail && b.info.reuse_detail.stored,
		sprintf('%J', b.info.reuse_detail));
	check(`dw=${DW}: the store holds one entry`, eng.cached().stored == 1, sprintf('%J', eng.cached()));

	let x2 = run_job(eng, A2, 0);

	printf('dw=%d: A2 reuse %s, cached %d of %d, stored %.3f GiB\n', DW, x2.info.reuse, x2.info.cached_tokens, length(A2),
		eng.cached().stored_bytes / 1073741824.0);
	check(`dw=${DW}: A2 swaps A's live state back from RAM`,
		x2.info.reuse == 'ram' && x2.info.cached_tokens >= length(A),
		`${x2.info.reuse} ${x2.info.cached_tokens}`);

	if (DW > 0)
		check(`dw=${DW}: the drafter ran after the swap-in`, x2.info.mtp != null && x2.info.mtp.rounds > 0,
			sprintf('%J', x2.info.mtp));

	// the same sequence without the store: A2 continues the live state in place. The
	// swapped-in state is a bit copy of it, so the outputs must be IDENTICAL.
	let e0 = mk(0);
	let c1 = run_job(e0, A, SNAP);
	let y2 = run_job(e0, A2, 0);

	check(`dw=${DW}: A2 without the store continues the live state`, y2.info.reuse == 'live', y2.info.reuse);
	// the MoE accumulates in a fixed order (AMP_MOE_ATOMIC unset), so without the drafter a restored
	// state continues exactly like the live one; with it the verify windows follow the draft
	// acceptance, whose drafter state is only approximately restored (loose)
	check(`dw=${DW}: A2 via RAM ${DW ? 'follows' : 'equals'} A2 via the live state`,
		same_head(x2.toks, y2.toks) >= (DW ? TOL : N),
		`${same_head(x2.toks, y2.toks)} of ${N} (generation repeat: ${same_head(a1.toks, c1.toks)} of ${N})`);

	// snapshot reuse point: A, B, then A3 (RAM) vs A, A3 (the device snapshot)
	eng = mk(6);
	run_job(eng, A, SNAP);
	run_job(eng, B, length(B) - 20);

	let x3 = run_job(eng, A3, 0);

	e0 = mk(0);
	run_job(e0, A, SNAP);

	let y3 = run_job(e0, A3, 0);

	printf('dw=%d: A3 reuse %s, cached %d of %d; without store: %s, cached %d\n', DW, x3.info.reuse,
		x3.info.cached_tokens, length(A3), y3.info.reuse, y3.info.cached_tokens);
	check(`dw=${DW}: A3 restores A's snapshot from RAM`,
		x3.info.reuse == 'ram' && x3.info.cached_tokens == SNAP, `${x3.info.reuse} ${x3.info.cached_tokens}`);
	check(`dw=${DW}: A3 without the store uses the device snapshot`, y3.info.reuse == 'snapshot', y3.info.reuse);
	check(`dw=${DW}: A3 via RAM ${DW ? 'follows' : 'equals'} A3 via the device snapshot`,
		same_head(x3.toks, y3.toks) >= (DW ? TOL : N), `${same_head(x3.toks, y3.toks)} of ${N}`);

	// fresh-prefill references (loose: a chunked / resumed prefill differs from a one-shot one by fp noise)
	e0 = mk(0);
	let r2 = run_job(e0, A2, 0);
	e0 = mk(0);
	let r3 = run_job(e0, A3, 0);

	printf('dw=%d: vs fresh prefill: A2 %d of %d, A3 %d of %d identical tokens\n', DW,
		same_head(x2.toks, r2.toks), N, same_head(x3.toks, r3.toks), N);
	// (no assertion against the fresh references: the first token of a resumed vs a one-shot
	// prefill can flip on a near-tie; the numeric comparison is tests/dsv4_store_state.uc)

	// B is still reachable (it was stored when A3 took over)
	let b2 = run_job(eng, [ ...B, ...TAIL ], 0);

	check(`dw=${DW}: B comes back from RAM too`, b2.info.reuse == 'ram' && b2.info.cached_tokens >= length(B) - 20,
		`${b2.info.reuse} ${b2.info.cached_tokens} ${sprintf('%J', eng.cached())}`);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store_dsv4: ok\n');
