// tests/engine_slots_dsv4.uc — resident device slots in front of the host-RAM store (DeepSeek-V4).
//
// Three conversations A, B, C on an engine with 2 device slots + the host store:
//   A, B, C      : A -> slot, B -> slot (C live)
//   A2 (A ext.)  : swaps A back from its device slot; C needs a slot, so B is demoted to host
//   B2 (B ext.)  : swaps B back from host RAM (the older tier)
//   C2 (C ext.)  : swaps C back from a device slot
// Every continuation must follow the same request served live without any store (fp noise of
// the non-deterministic prefill allows a late near-tie flip), and the device swap-in must be
// faster than the host one.
//
// Env: AMP_STORE_DEVICES (6), AMP_STORE_STAGES (3), AMP_DSV4_LAYERS (0 = the full model).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as dspark_tp from 'ampstein.dspark_tp';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const NL = +(getenv('AMP_DSV4_LAYERS') || 0);
const ND = +(getenv('AMP_STORE_DEVICES') || 6);
const NS = +(getenv('AMP_STORE_STAGES') || 3);
const N = 12, TOL = 2;
const BOS = '<｜begin▁of▁sentence｜>';

if (!stat(MODEL) || torch.device_count() < ND) {
	warn('skip: needs the DSV4 model and ' + ND + ' GPUs\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
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
		warn('ok: ' + name + (detail != null ? ' (' + detail + ')' : '') + '\n');
}

function text(seed, n) {
	let s = BOS;

	for (let i = 0; i < n; i++)
		s += `Item ${seed}-${i}: the ${seed} committee reviewed record number ${i * 7 + length(seed)} and noted nothing unusual. `;

	return s;
}

let A = tok.encode(text('alpha', 150), true);
let B = tok.encode(text('bravo', 150), true);
let C = tok.encode(text('charlie', 150), true);
let TAIL = tok.encode(' Summarize the findings in one sentence:', false);

function run_job(eng, ids) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: length(ids) - 20, max_new: N, stop_ids: [],
		sampling: { temperature: 0 },
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	let deadline = now_ms() + 900000;

	while (info == null && now_ms() < deadline)
		uloop.run(20);

	if (info == null)
		die('job timed out');

	return { toks: toks, info: info };
}

function same_head(a, b) {
	let k = 0;

	while (k < length(a) && k < length(b) && a[k] == b[k])
		k++;

	return k;
}

let mk = (o) => engine.create(model, { max_ctx: 8192, prefill_chunk: 512, dspark_window: 3, warmup: false,
	cache_ram: 6, ...o });

// the slotted engine
let eng = mk({ slots: 2, slot_margin: 0 });

check('two resident slots allocated', eng.cached().slots == 2 && eng.cached().slots_free == 2, sprintf('%J', eng.cached()));

let a = run_job(eng, A);
let b = run_job(eng, B);

check('A went to a device slot', b.info.reuse_detail.stored && eng.cached().slots_free == 1, sprintf('%J', eng.cached()));

let c = run_job(eng, C);

check('B went to the second slot', c.info.reuse_detail.stored && eng.cached().slots_free == 0, sprintf('%J', eng.cached()));

let A2 = [ ...A, ...a.toks, ...TAIL ];
let B2 = [ ...B, ...b.toks, ...TAIL ];
let C2 = [ ...C, ...c.toks, ...TAIL ];

let xa = run_job(eng, A2);
let dev_ms = xa.info.reuse_detail.ram_ms;

check('A2 swaps in from a device slot', xa.info.reuse == 'ram' && xa.info.reuse_detail.ram_tier == 'device' &&
	xa.info.cached_tokens >= length(A), sprintf('%s %s %d', xa.info.reuse, xa.info.reuse_detail.ram_tier, xa.info.cached_tokens));
check('B was demoted to host, C holds a slot', eng.cached().stored == 2 && eng.cached().slots_free == 1,
	sprintf('%J', eng.cached()));

let xb = run_job(eng, B2);
let host_ms = xb.info.reuse_detail.ram_ms;

check('B2 swaps in from host RAM', xb.info.reuse == 'ram' && xb.info.reuse_detail.ram_tier == 'host' &&
	xb.info.cached_tokens >= length(B) - 20, sprintf('%s %s %d', xb.info.reuse, xb.info.reuse_detail.ram_tier, xb.info.cached_tokens));

let xc = run_job(eng, C2);
let dev2_ms = xc.info.reuse_detail.ram_ms;

check('C2 swaps in from a device slot', xc.info.reuse == 'ram' && xc.info.reuse_detail.ram_tier == 'device' &&
	xc.info.cached_tokens >= length(C) - 20, sprintf('%s %s %d', xc.info.reuse, xc.info.reuse_detail.ram_tier, xc.info.cached_tokens));

printf('swap-in: device %.0f ms / %.0f ms, host %.0f ms\n', dev_ms, dev2_ms, host_ms);
check('a device swap-in is faster than a host one', dev_ms < host_ms && dev2_ms < host_ms,
	sprintf('%.0f / %.0f vs %.0f ms', dev_ms, dev2_ms, host_ms));

// references: the same requests served live, no store, no slots
let ref = mk({ slots: 0, cache_ram: 0 });

run_job(ref, A);

let ya = run_job(ref, A2);

run_job(ref, B);

let yb = run_job(ref, B2);

run_job(ref, C);

let yc = run_job(ref, C2);

check('A2 follows the live reference', same_head(xa.toks, ya.toks) >= TOL, `${same_head(xa.toks, ya.toks)} of ${N}`);
check('B2 follows the live reference', same_head(xb.toks, yb.toks) >= TOL, `${same_head(xb.toks, yb.toks)} of ${N}`);
check('C2 follows the live reference', same_head(xc.toks, yc.toks) >= TOL, `${same_head(xc.toks, yc.toks)} of ${N}`);

if (fails)
	die(`${fails} check(s) failed`);

print('engine_slots_dsv4: ok\n');
