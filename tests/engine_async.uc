// tests/engine_async.uc — engine.uc's notifier-driven sampling split:
//   - dist_begin/dist_finish reproduce dist() exactly (ids equal, probs
//     within fp32 rounding) across the parameter space: default top_k,
//     explicit top_k, min_p, top_p, both combined, presence/frequency
//     penalties, top_k 1
//   - the no-mask path fills `keep` host-side (no mask kernel, no copy)
//   - argmax_begin reproduces the greedy argmax
//   - every case wakes through the uloop handle (drain + uloop.end()),
//     exactly the engine's decode-loop mechanism
//
// Needs >= 1 GPU (skips otherwise). No model: the sampling helpers take
// the notifier, the pinned buffers and the logits directly.

import * as uloop from 'uloop';
import * as math from 'math';
import * as engine from 'ampstein.engine';

let ng = torch.device_count();

if (ng < 1) {
	warn('skip: needs a GPU\n');
	exit(125);
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

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

/* uloop.run(timeout) is ONE poll round capped at timeout, so waiting
 * for a wake-up means repeated short rounds until the condition holds
 * (see tests/notifier.uc). */
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

const V = 2000;
const DEV = [0];

let note = torch.notifier();
let bufs = {
	p: torch.zeros([1000], 'float32').pin(),
	idxs: torch.zeros([1000], 'int64').pin(),
	keep: torch.zeros([1], 'float32').pin(),
	tok: torch.zeros([1], 'int64').pin()
};

let woke = 0;

uloop.handle(note, function () {
	note.drain();
	woke++;
	uloop.end();
}, uloop.ULOOP_READ);

// deterministic varied logits, fp16 CUDA (1, V) — the model's shape
let base = torch.arange(V, 'float32', 'cuda:0');

function make_logits(shift) {
	return base.sin().mul(4.0).add(base.mul(0.013).cos())
		.add(shift).reshape([1, V]).to('float16');
}

// penalty counts: a (V) fp32 device tensor with a few seen tokens
function make_counts() {
	let c = torch.zeros([V], 'float32', 'cuda:0');

	c.narrow(0, 7, 1).add_(3.0);
	c.narrow(0, 900, 1).add_(1.0);
	c.narrow(0, 1500, 2).add_(2.0);

	return c;
}

/* One sampled case: run the sync reference and the async pair, compare. */
function case_dist(name, s, counts) {
	let logits = make_logits(0.5);
	let ref = engine.dist(logits, s, counts);

	woke = 0;

	let st = engine.dist_begin(note, bufs, DEV, logits, s, counts);

	check(name + ': wake-up arrived', run_until(() => woke == 1, 2000));
	engine.dist_finish(bufs, st);

	let ok_ids = true, ok_probs = true;

	if (length(st.ids) != length(ref.ids))
		ok_ids = false;
	else
		for (let i = 0; i < length(st.ids); i++) {
			if (st.ids[i] !== ref.ids[i])
				ok_ids = false;

			if (math.abs(st.probs[i] - ref.probs[i]) > 1e-6)
				ok_probs = false;
		}

	check(name + ': ids exact (' + length(st.ids) + ' kept)', ok_ids,
		length(st.ids) + ' vs ' + length(ref.ids));
	check(name + ': probs match', ok_probs);

	// probabilities sum to 1
	let total = 0.0;

	for (let i = 0; i < length(st.probs); i++)
		total += st.probs[i];

	check(name + ': probs normalized', math.abs(total - 1.0) < 1e-9, total);
}

/* --- 1. greedy ------------------------------------------------------------ */

{
	let logits = make_logits(0.5);
	let want = logits.reshape([-1]).argmax().item();

	woke = 0;
	engine.argmax_begin(note, bufs, DEV, logits);

	check('greedy: wake-up arrived', run_until(() => woke == 1, 2000));
	check('greedy: argmax exact', bufs.tok.item() == want,
		bufs.tok.item() + ' vs ' + want);
}

/* --- 2. sampled cases ------------------------------------------------------ */

case_dist('default top_k (no mask)', { temperature: 0.8 }, null);
case_dist('top_k 50', { temperature: 1.0, top_k: 50 }, null);
case_dist('top_k 1', { temperature: 0.7, top_k: 1 }, null);
case_dist('min_p 0.05', { temperature: 1.1, min_p: 0.05 }, null);
case_dist('top_p 0.9', { temperature: 0.9, top_p: 0.9 }, null);
case_dist('min_p + top_p', { temperature: 1.0, min_p: 0.02, top_p: 0.85 },
	null);
case_dist('penalties + top_k 100',
	{ temperature: 1.0, top_k: 100, presence_penalty: 0.5,
	  frequency_penalty: 0.3 }, make_counts());

// the no-mask path sets keep host-side: check the buffer directly
{
	let logits = make_logits(0.5);

	woke = 0;
	engine.dist_begin(note, bufs, DEV, logits, { temperature: 0.8 }, null);
	run_until(() => woke == 1, 2000);
	check('no-mask: keep filled host-side', bufs.keep.item() == 1000,
		bufs.keep.item());
}

/* --- 3. device-side MTP acceptance count ----------------------------------- */

// The async MTP round computes the accepted-prefix length as the first
// mismatch index: min over (i where vpred != c[1..W], W where equal).
// Matches may occur after a mismatch (repeated tokens), so the mask SUM
// is not the answer. Compare against the host-side walk the synchronous
// round uses, across accept/reject shapes incl. re-matches.

function accept_case(name, c, vpred) {
	let W = length(vpred);

	// host reference: the synchronous round's walk
	let a_ref = 0;

	while (a_ref < W && vpred[a_ref] == c[a_ref + 1])
		a_ref++;

	// device-side: the async round's math
	let cd = torch.from_array(c, 'int64', 'cuda:0');
	let vd = torch.from_array(vpred, 'int64', 'cuda:0');
	let cshift = cd.narrow(0, 1, W);
	let m = vd.le(cshift).to('float32')
		.mul(cshift.le(vd).to('float32'));
	let idx = torch.arange(W, 'float32', 'cuda:0');
	let cand = idx.add(m.mul(W));

	woke = 0;
	torch.copy_async(bufs.keep, cand.neg().max().neg().reshape([1]));
	note.arm(DEV);
	run_until(() => woke == 1, 2000);

	let a = int(bufs.keep.item());

	check(name, a == a_ref, a + ' vs ' + a_ref);
}

accept_case('accept: full match', [5, 7, 9, 11, 13], [7, 9, 11, 13]);
accept_case('accept: reject at 0', [5, 7, 9, 11, 13], [8, 9, 11, 13]);
accept_case('accept: reject at 1', [5, 7, 9, 11, 13], [7, 8, 11, 13]);
accept_case('accept: reject at 3', [5, 7, 9, 11, 13], [7, 9, 11, 12]);
accept_case('accept: re-match after reject', [5, 7, 9, 7, 9],
	[8, 9, 7, 9]);
accept_case('accept: W=1 full', [5, 7], [7]);
accept_case('accept: W=1 reject', [5, 7], [6]);

/* --- 3b. the two-device acceptance ---------------------------- */
// The async round's vpred comes from the lm_head's logits, which live on
// the head's device — under a layer split that is NOT devs[0], where the
// draft ids c_dev live. The fixed engine line is
// `vr.logits.view([W, -1]).argmax(-1).cuda(model.devs[0])`; compare
// against the host walk and lock in that the pre-fix shape (comparing
// the vpred on the logits' device against c_dev on devs[0]) raises
// instead of silently doing something. Needs a second GPU.
if (ng >= 2) {
	let W2 = 3, V2 = 512;
	let c2 = [11, 22, 33, 44, 55];
	let vpred2 = [22, 33, 77];            // accept 2, reject at 2
	// logits with the right argmax pattern on cuda:1
	let lg = torch.full([W2, V2], -2.0, 'float32', 'cuda:1');

	for (let i = 0; i < W2; i++)
		lg.narrow(0, i, 1).narrow(1, vpred2[i], 1).fill_(1.0);

	// the engine's fixed line: argmax on the head's device, move to
	// devs[0] before the comparison
	let vd2 = lg.view([W2, V2]).argmax(-1).cuda(0);
	let cd2 = torch.from_array(c2, 'int64', 'cuda:0');
	let csh2 = cd2.narrow(0, 1, W2);
	let m2 = vd2.le(csh2).to('float32')
		.mul(csh2.le(vd2).to('float32'));
	let cand2 = torch.arange(W2, 'float32', 'cuda:0').add(m2.mul(W2));

	woke = 0;
	torch.copy_async(bufs.keep, cand2.neg().max().neg().reshape([1]));
	note.arm(DEV);
	run_until(() => woke == 1, 2000);

	let a2_ref = 0;

	while (a2_ref < W2 && vpred2[a2_ref] == c2[a2_ref + 1])
		a2_ref++;

	check('accept: two-device (vpred moved to devs[0])',
		int(bufs.keep.item()) == a2_ref,
		bufs.keep.item() + ' vs ' + a2_ref);

	// pre-fix shape: the vpred left on the logits' device, compared
	// against c_dev on devs[0] — must raise, that is the bug
	let vd2b = lg.view([W2, V2]).argmax(-1);   // on cuda:1
	let raised = false;

	try {
		vd2b.le(cd2.narrow(0, 1, W2));
	}
	catch (e) {
		raised = true;
	}

	check('accept: two-device pre-fix shape raises', raised);
}

// best-effort drain of in-flight cudaLaunchHostFunc callbacks before
// teardown; under heavy co-tenant memory pressure the explicit device
// sync can itself report an async OOM — ignore that (the checks above
// already passed), do not mask them
try {
	torch.synchronize();
}
catch (e) {}

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('all green\n');