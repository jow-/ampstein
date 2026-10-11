// examples/bench_allreduce.uc — latency of the P2P all-reduce inside a CUDA
// graph (100 chained all-reduces of ROWS x 4096 fp16) for device groups.
//   ucode ... examples/bench_allreduce.uc -- 0,1,2 0,3 0,1,2,3,4,5 [--rows N] [--mode oneshot|twophase]
let torch = require('ampstein.torch');
global.torch = torch;

let groups = [], rows = 1, mode = null;
let argv = ARGV, st = index(argv, '--');

if (st >= 0) argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	if (argv[i] == '--rows') rows = int(argv[++i]);
	else if (argv[i] == '--mode') mode = argv[++i];
	else push(groups, map(split(argv[i], ','), (x) => int(x)));
}

const N_AR = 100;

for (let devs in groups) {
	let G = length(devs);

	// cross-device copies first (as every real workload has): see docs/LESSONS.md L87
	for (let a in devs)
		for (let b in devs)
			if (a != b)
				torch.ones([8], 'float16', 'cuda:' + a).cuda(b);

	torch.synchronize();

	let cap = (rows > 8 ? rows : 8) * 4096;
	let comm = mode ? torch.p2p_comm(devs, cap, mode) : torch.p2p_comm(devs, cap);
	let x0 = map(devs, (d) => torch.ones([rows, 4096], 'float16', 'cuda:' + d));
	let last = null;

	function body() {
		let xs = x0;

		for (let k = 0; k < N_AR; k++)
			xs = map(xs, (x, g) => comm.allreduce(x, g));

		last = xs;     // keep the chain's tensors alive for the replays
	}

	body();
	torch.synchronize();

	let gr = torch.graph();

	gr.capture_begin(devs[0], slice(devs, 1));
	body();
	gr.capture_end();
	torch.synchronize();

	for (let w = 0; w < 3; w++) {
		gr.replay();
		torch.synchronize();
	}

	let c0 = clock(true);

	for (let r = 0; r < 10; r++)
		gr.replay();

	torch.synchronize();

	let c1 = clock(true);
	let t = (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1000000000.0;

	printf('devices %s%s rows %d: %.1f us per all-reduce \n',
		join(',', devs), mode ? ' ' + mode : '', rows, t * 1e6 / (10 * N_AR));
}
