// tests/engine_slice.uc — time-sliced co-generation: a long generation yields to a waiting session.
//
// Two sessions A and B are submitted back to back with slice_ms small. A runs, is parked after a
// slice (its state moves into the store), B prefills and runs, is parked, A resumes, ... Both
// must complete with exactly the tokens they produce alone (greedy), the token streams must
// interleave, and a parked session whose client goes away must release its entry.
//
// Needs a Qwen3.5-family checkpoint (AMP_MODEL_QWEN35) and two GPUs (AMP_STORE_GPUS).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const N = +(getenv('AMP_SLICE_GEN') || 120);        // tokens generated per session
const SLICE = +(getenv('AMP_SLICE_TOK') || 32);     // tokens per slice

if (!stat(MODEL) || torch.device_count() < 2) {
	warn('skip: needs the Qwen model and two GPUs\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
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
		warn('ok: ' + name + (detail != null ? ' (' + detail + ')' : '') + '\n');
}

function text(seed, n) {
	let s = '';

	for (let i = 0; i < n; i++)
		s += `Item ${seed}-${i}: the ${seed} committee reviewed record number ${i * 7 + length(seed)} and noted nothing unusual. `;

	return s + ' Now continue the list with the next items:';
}

let PA = tok.encode(text('alpha', 30), false);
let PB = tok.encode(text('bravo', 30), false);

printf('prompt tokens: A %d, B %d; %d tokens each, slice %d tokens\n', length(PA), length(PB), N, SLICE);

// submit a job; returns a handle { toks, ts, info, alive }
function submit(eng, ids, alive_fn) {
	let h = { toks: [], ts: [], info: null };

	eng.submit({
		ids: ids, snap_at: 0, max_new: N, stop_ids: [],
		sampling: { temperature: 0 },
		alive: alive_fn || (() => true),
		on_progress: () => true,
		on_token: function (t) { push(h.toks, t); push(h.ts, now_ms()); return true; },
		on_done: function (i) { h.info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); h.info = { reason: 'error' }; }
	});

	return h;
}

function wait_all(hs, timeout_ms) {
	let deadline = now_ms() + timeout_ms;

	while (now_ms() < deadline) {
		let done = true;

		for (let h in hs)
			if (h.info == null)
				done = false;

		if (done)
			return true;

		uloop.run(20);
	}

	return false;
}

function same_head(a, b) {
	let k = 0;

	while (k < length(a) && k < length(b) && a[k] == b[k])
		k++;

	return k;
}

// switches between the sessions in the merged token stream
function switches(a, b) {
	let ev = [];

	for (let t in a.ts)
		push(ev, [ t, 'A' ]);

	for (let t in b.ts)
		push(ev, [ t, 'B' ]);

	ev = sort(ev, (x, y) => x[0] - y[0]);

	let n = 0, prev = null;

	for (let e in ev) {
		if (prev != null && e[1] != prev)
			n++;

		prev = e[1];
	}

	return n;
}

for (let W in [ 0, 3 ]) {
	let mk = (slice) => engine.create(model, { max_ctx: 8192, prefill_chunk: 512, mtp_window: W,
		warmup: false, cache_ram: 6, slots: 0, slice_tokens: slice });

	// references: each session alone, no slicing
	let eng = mk(0);
	let ra = submit(eng, PA);

	wait_all([ ra ], 600000);

	let rb = submit(eng, PB);

	wait_all([ rb ], 600000);

	// co-generation
	eng = mk(SLICE);

	let a = submit(eng, PA), b = submit(eng, PB);

	check(`W=${W}: both sessions complete`, wait_all([ a, b ], 900000) && a.info.reason == 'length' && b.info.reason == 'length',
		`${a.info ? a.info.reason : 'timeout'} ${b.info ? b.info.reason : 'timeout'}`);

	let pa = a.info.preemptions || 0, pb = b.info.preemptions || 0;
	let sw = switches(a, b);

	printf('W=%d: A parked %d x, B parked %d x, %d switches in the token stream, both finished %.1f s / %.1f s after submit\n',
		W, pa, pb, sw, (a.ts[length(a.ts) - 1] - a.ts[0]) / 1000.0, (b.ts[length(b.ts) - 1] - b.ts[0]) / 1000.0);
	check(`W=${W}: the sessions were interleaved`, pa + pb >= 2 && sw >= 2, `${pa + pb} parks, ${sw} switches`);
	check(`W=${W}: A delivers exactly its solo tokens`, same_head(a.toks, ra.toks) == N,
		`${same_head(a.toks, ra.toks)} of ${N} (${length(a.toks)} delivered)`);
	check(`W=${W}: B delivers exactly its solo tokens`, same_head(b.toks, rb.toks) == N,
		`${same_head(b.toks, rb.toks)} of ${N} (${length(b.toks)} delivered)`);
	check(`W=${W}: nothing stays parked`, eng.cached().parked == 0, sprintf('%J', eng.cached()));

	// a parked session whose client went away releases its entry and the engine carries on
	let gone = false;
	let c = submit(eng, PA, () => !gone);
	let d = submit(eng, PB);

	// let c run and be parked once, then drop it
	let t0 = now_ms();

	while (now_ms() - t0 < 600000 && !(c.info || (c.info == null && eng.cached().parked > 0 && length(c.toks) > 0)))
		uloop.run(20);

	gone = true;

	check(`W=${W}: the other session still completes`, wait_all([ d ], 600000) && d.info.reason == 'length', d.info ? d.info.reason : 'timeout');
	wait_all([ c ], 600000);
	check(`W=${W}: the dropped session ends cancelled or finished, nothing parked`, c.info != null &&
		(c.info.reason == 'cancelled' || c.info.reason == 'length') && eng.cached().parked == 0,
		`${c.info ? c.info.reason : 'timeout'} ${sprintf('%J', eng.cached())}`);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_slice: ok\n');
