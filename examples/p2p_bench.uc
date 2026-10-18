// examples/p2p_bench.uc — all-reduce latency / bandwidth of every GPU pair (the P2P communicator the TP model uses)
// and the raw peer-copy bandwidth.   ucode ... examples/p2p_bench.uc [rows ...]
function now() {
	let c = clock(true);

	return c[0] * 1e6 + c[1] / 1e3;          // µs
}

const H = 5120;
let n = torch.device_count();
let rows = length(ARGV) ? map(ARGV, (x) => +x) : [ 1, 8, 1024 ];

for (let a = 0; a < n; a++) {
	for (let b = a + 1; b < n; b++) {
		let line = sprintf('GPU %d-%d:', a, b);

		for (let r in rows) {
			let cap = max(r, 8) * H;
			let comm = torch.p2p_comm([ a, b ], cap);
			let x0 = torch.rand([r, H], 'float16', 'cuda:' + a), x1 = torch.rand([r, H], 'float16', 'cuda:' + b);
			let iters = (r >= 1024) ? 40 : 400;

			for (let i = 0; i < 5; i++) {
				comm.allreduce(x0, 0);
				comm.allreduce(x1, 1);
			}

			torch.synchronize();

			let t0 = now();

			for (let i = 0; i < iters; i++) {
				comm.allreduce(x0, 0);
				comm.allreduce(x1, 1);
			}

			torch.synchronize();
			line += sprintf('  %d rows: %.0f us', r, (now() - t0) / iters);
		}

		let big = torch.rand([32 * 1024 * 1024], 'float16', 'cuda:' + a);

		big.cuda(b);
		torch.synchronize();

		let t0 = now();

		for (let i = 0; i < 10; i++)
			big.cuda(b);

		torch.synchronize();
		printf('%s  copy %.1f GB/s\n', line, 10 * 64.0 / 1024 / ((now() - t0) / 1e6));
	}
}
