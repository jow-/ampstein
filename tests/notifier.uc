// tests/notifier.uc — torch.notifier (eventfd + cudaLaunchHostFunc) and
// torch.copy_async:
//   - the notifier registers as a uloop handle source (fileno() probe)
//     and its callback fires once every armed device's stream drained
//   - arm([0, 1]) with one device idle wakes only after the second
//     device's work completes (the wake-up is the AND of the streams)
//   - seq()/done()/drain() bookkeeping across several arms
//   - copy_async lands the results in a pinned host buffer without a
//     host sync, values exact, and a notifier armed on the same stream
//     after the copy fires only once the copy completed
//   - arm() during a graph capture raises
// Needs >= 1 GPU (skips otherwise); the multi-device case needs 2.
//
// Pattern: the handle callback drains, records and calls uloop.end()
// once satisfied; run_until() drives repeated single-round
// uloop.run(20) calls until the condition holds (uloop.run(timeout) is
// ONE poll round capped at timeout, not a wait of that length).

import * as uloop from 'uloop';

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

/* uloop.run(timeout) is ONE poll round capped at timeout (libubox
 * uloop_run_timeout loops only while timeout < 0), so waiting for a
 * wake-up means repeated short rounds until the condition holds. The
 * handle callback drains and calls uloop.end() once satisfied, which
 * cuts the current round short. */
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

/* --- 1. single device: arm -> wake-up -> drain ---------------------------- */

function test_single() {
	let n = torch.notifier();
	let fired = 0;

	let h = uloop.handle(n, function () {
		let drained = n.drain();

		check('single: drain returns 1', drained == 1, drained);
		check('single: seq advanced', n.seq() == 1, n.seq());
		fired++;
		uloop.end();
	}, uloop.ULOOP_READ);

	// a little GPU work so the callback is ordered behind something
	let x = torch.arange(1000, 'float32', 'cuda:0').mul(2.0);
	let host = x.cpu();          // sync copy, not under test

	n.arm('cuda:0');
	check('single: wake-up arrived', run_until(() => fired == 1, 500));

	check('single: callback fired', fired == 1, fired);
	check('single: done() false after drain', n.done() == false);

	h.delete();
}

test_single();

/* --- 2. two devices: the wake-up is the AND of both streams --------------- */

/* Deterministic instead of timing-based (the GPUs may be co-occupied,
 * docs/LESSONS.md L31): each device's stream increments a counter N times and
 * copy_asyncs it into a pinned host buffer BEFORE the arm; the callback
 * then reads both counters. A wake-up that fired before BOTH streams
 * drained would read stale/partial values. */
function test_two_devices() {
	if (ng < 2) {
		warn('skip: two-device case (needs 2 GPUs)\n');
		return;
	}

	let n = torch.notifier();
	let fired = 0;

	// device 1: a long-running in-place chain (~64 MiB x 20 ops), then
	// the D2H copy of the counter; device 0: a short chain
	let big = torch.zeros([8, 2048, 2048], 'float16', 'cuda:1');
	let c1 = torch.zeros([1], 'float32', 'cuda:1');

	for (let i = 0; i < 20; i++) {
		big.mul_(1.0001);
		c1.add_(1.0);
	}

	let host1 = torch.zeros([1], 'float32', 'cpu').pin();

	let c0 = torch.zeros([1], 'float32', 'cuda:0');

	for (let i = 0; i < 3; i++)
		c0.add_(1.0);

	let host0 = torch.zeros([1], 'float32', 'cpu').pin();

	torch.copy_async(host1, c1);
	torch.copy_async(host0, c0);

	let h = uloop.handle(n, function () {
		n.drain();
		fired++;
		uloop.end();
	}, uloop.ULOOP_READ);

	n.arm([0, 1]);
	check('two: wake-up arrived', run_until(() => fired == 1, 5000));

	check('two: one wake-up', fired == 1, fired);
	check('two: seq == 1', n.seq() == 1, n.seq());
	check('two: device 1 counter complete', host1.tolist()[0] == 20.0,
		host1.tolist()[0]);
	check('two: device 0 counter complete', host0.tolist()[0] == 3.0,
		host0.tolist()[0]);

	big.free();
	h.delete();
}

test_two_devices();

/* --- 3. several arms: seq counts, drain consumes all pending -------------- */

function test_multi_arm() {
	let n = torch.notifier();
	let fired = 0;

	let h = uloop.handle(n, function () {
		let k = n.drain();

		check('multi: drain > 0', k > 0, k);
		fired++;
		uloop.end();
	}, uloop.ULOOP_READ);

	for (let i = 0; i < 5; i++) {
		torch.arange(16, 'float32', 'cuda:0').mul(i + 1.0);
		n.arm(0);
	}

	check('multi: all 5 completions', run_until(() => n.seq() == 5, 500));

	// the loop may exit between rounds: consume the final wake-up (all
	// completions have fired by now, nothing new can arrive)
	n.drain();

	check('multi: seq == 5', n.seq() == 5, n.seq());
	check('multi: done() false after drain', n.done() == false);
	check('multi: fired at least once', fired >= 1, fired);

	h.delete();
}

test_multi_arm();

/* --- 4. copy_async: results land without a host sync ---------------------- */

function test_copy_async() {
	let n = torch.notifier();
	let fired = 0;

	let h = uloop.handle(n, function () {
		n.drain();
		fired++;
		uloop.end();
	}, uloop.ULOOP_READ);

	let dev = torch.arange(64, 'float32', 'cuda:0').mul(0.25);
	let host = torch.zeros([64], 'float32', 'cpu').pin();

	torch.copy_async(host, dev);
	n.arm(0);

	// WITHOUT the notifier this point would have to synchronize; here
	// the loop just runs until the callback fires
	check('copy_async: wake-up arrived', run_until(() => fired == 1, 500));

	check('copy_async: callback fired', fired == 1, fired);

	let got = host.tolist();

	let ok = true;

	for (let i = 0; i < 64; i++)
		if (got[i] != i * 0.25)
			ok = false;

	check('copy_async: values exact', ok);

	h.delete();
}

test_copy_async();

/* --- 5. copy_async ordered behind a long kernel: arm after the copy ------- */

function test_copy_async_ordering() {
	if (ng < 2) {
		warn('skip: copy ordering case (needs 2 GPUs)\n');
		return;
	}

	let n = torch.notifier();
	let fired = 0;

	let h = uloop.handle(n, function () {
		n.drain();
		fired++;
		uloop.end();
	}, uloop.ULOOP_READ);

	let big = torch.zeros([8, 2048, 2048], 'float16', 'cuda:1');

	for (let i = 0; i < 20; i++)
		big.mul_(1.0001);

	let dev = torch.full([4], 7.0, 'float32', 'cuda:1');
	let host = torch.zeros([4], 'float32', 'cpu').pin();

	torch.copy_async(host, dev);   // enqueued AFTER the big kernels
	n.arm(1);

	check('ordering: wake-up arrived', run_until(() => fired == 1, 5000));

	check('ordering: wake-up after the copy', fired == 1, fired);
	check('ordering: values exact', host.tolist()[0] == 7.0,
		host.tolist()[0]);

	big.free();
	h.delete();
}

test_copy_async_ordering();

/* --- 6. arm() during graph capture raises --------------------------------- */

function test_capture_reject() {
	let n = torch.notifier();
	let g = torch.graph();
	let raised = null;

	g.capture_begin(0);
	torch.arange(4, 'float32', 'cuda:0');   // non-empty capture

	try {
		n.arm(0);
	}
	catch (e) {
		raised = e.message;
	}

	g.capture_end();

	check('capture: arm raises', raised != null &&
		index(raised, 'capture') >= 0, raised);

	n.free();
	g.free();
}

test_capture_reject();

/* --- 7. notifier usable as a plain fileno() source ------------------------ */

function test_fileno() {
	let n = torch.notifier();

	check('fileno: positive fd', n.fileno() > 0, n.fileno());

	n.free();
}

test_fileno();

warn(fails ? `${fails} FAILURES\n` : 'all green\n');
exit(fails ? 1 : 0);