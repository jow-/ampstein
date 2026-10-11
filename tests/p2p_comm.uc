// tests/p2p_comm.uc — N-way P2P all-reduce (torch.p2p_comm) on every
// rank count 2..(visible GPUs), one-shot and two-phase: the result equals
// the fp32 sum of the partials rounded once to fp16, is bit-identical on
// every rank, survives many back-to-back calls (slot/flag reuse), sizes
// that do not divide evenly into N chunks, and replays inside one multi-
// device CUDA graph. Reports per-call latency for a decode-sized payload.
// Needs >= 2 GPUs (skips otherwise).

let ng = torch.device_count();

if (ng < 2) {
	warn('skip: needs >= 2 GPUs\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

/* deterministic partial of rank r: values in about [-1, 1] */
function partial(n, r, k, dev) {
	return torch.arange(n, 'float32', 'cuda:' + dev).mul(0.37 + 0.11 * r + 0.013 * k)
		.sin().to('float16');
}

function run_all(comm, xs) {
	let out = [];

	for (let r = 0; r < length(xs); r++)
		push(out, comm.allreduce(xs[r], r));

	return out;
}

function check(N, mode) {
	let devs = [];

	for (let r = 0; r < N; r++)
		push(devs, r);

	let cap = 8 * 1024 * 40;
	let comm = torch.p2p_comm(devs, cap, mode);

	for (let k = 0; k < 24; k++) {
		let n = [ 5120, 5128, 40, 4096 * 3, 5120 * 4, 8 * 1024 * 40, 5120 * 64 ][k % 7];
		let xs = map(devs, (d) => partial(n, d, k, d));
		let out = run_all(comm, xs);

		let ref = xs[0].to('float32');

		for (let r = 1; r < N; r++)
			ref = ref.add(xs[r].cuda(0).to('float32'));

		ref = ref.to('float16');

		let d0 = out[0].to('float32').sub(ref.to('float32')).abs().max().item();

		assert(d0 == 0, `N=${N} ${mode} n=${n}: differs from fp32-sum reference (${d0})`);

		for (let r = 1; r < N; r++) {
			let dr = out[r].cuda(0).to('float32').sub(out[0].to('float32')).abs().max().item();

			assert(dr == 0, `N=${N} ${mode} n=${n}: rank ${r} differs from rank 0`);
		}
	}

	// graph: 64 all-reduces on all ranks in one multi-device graph
	let n = 5120;
	let xs = map(devs, (d) => partial(n, d, 99, d));
	let cur = xs;
	let g = torch.graph();
	let extra = slice(devs, 1);

	run_all(comm, xs);
	torch.synchronize();
	g.capture_begin(0, extra);

	// same inputs every call (chaining would overflow fp16: x N per call)
	for (let i = 0; i < 64; i++)
		cur = run_all(comm, xs);

	g.capture_end();

	let held = cur;     // graph outputs (fixed addresses)

	for (let i = 0; i < 3; i++)
		g.replay();

	torch.synchronize();

	let t0 = now_ms(), R = 20;

	for (let i = 0; i < R; i++)
		g.replay();

	torch.synchronize();

	let us = (now_ms() - t0) * 1000 / (R * 64);

	for (let r = 1; r < N; r++)
		assert(held[r].cuda(0).to('float32').sub(held[0].to('float32')).abs().max().item() == 0,
			`N=${N} ${mode}: graph ranks differ`);

	printf('p2p_comm N=%d %-8s: exact, ranks identical; graph %.1f us per all-reduce (%d fp16)\n',
		N, comm.info().mode, us, n);
}

for (let N = 2; N <= min(ng, 8); N++) {
	check(N, 'oneshot');
	check(N, 'twophase');
}

print('p2p_comm: PASS\n');
