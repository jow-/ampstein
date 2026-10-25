// examples/bench_gguf.uc — throughput of the GGUF decode kernels (torch.gguf_gemv / torch.gguf_moe_gemv) on random weights:
// microseconds per call and the bytes of weights read per second.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/bench_gguf.uc [device=0]

const DEV = 'cuda:' + (ARGV[0] || 0);
const SIZES = {Q8_0: [ 32, 34 ], Q5_K: [ 256, 176 ], IQ2_XXS: [ 256, 66 ], IQ2_XS: [ 256, 74 ], IQ1_M: [ 256, 56 ], IQ4_NL: [ 32, 18 ]};

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

function bench(fn, iters) {
	for (let i = 0; i < 3; i++)
		fn();

	torch.synchronize();

	let t0 = now();

	for (let i = 0; i < iters; i++)
		fn();

	torch.synchronize();

	return (now() - t0) / iters;
}

function rawrows(rows, K, ty) {
	let b = SIZES[ty];

	return torch.rand([rows, K / b[0] * b[1]], 'float16', DEV).view([rows, -1]).to('uint8');
}

printf('%-34s %9s %9s\n', 'dense gemv (M rows)', 'us', 'GB/s');

for (let c in [ [ 'Q5_K', 32768, 1280, 'wq_b' ], [ 'Q5_K', 5120, 8192, 'wo_b' ], [ 'Q8_0', 1280, 5120, 'wq_a' ], [ 'Q5_K', 129280, 5120, 'head' ] ]) {
	let ty = c[0], N = c[1], K = c[2];
	let w = rawrows(N, K, ty);
	let bytes = N * (K / SIZES[ty][0] * SIZES[ty][1]);

	for (let M in [ 1, 5 ]) {
		let x = torch.rand([M, K], 'float16', DEV);
		let t = bench(() => torch.gguf_gemv(x, w, ty), 50);

		printf('%-34s %9.1f %9.0f\n', sprintf('%s %s %dx%d M=%d', c[3], ty, N, K, M), t * 1e6, bytes / t / 1e9);
	}

	w = null;
}

printf('\n%-34s %9s %9s\n', 'routed experts (6 slots)', 'us', 'GB/s');

for (let ty in [ 'IQ2_XXS', 'IQ2_XS', 'IQ1_M' ]) {
	let E = 64, N = 2304, K = 5120;
	let w = torch.rand([E, N, K / 256 * SIZES[ty][1]], 'float16', DEV).to('uint8');
	let x = torch.rand([1, K], 'float16', DEV);
	let ids = torch.from_array([ 3, 9, 14, 20, 33, 60 ], 'int64', DEV);
	let bytes = 6 * N * (K / 256 * SIZES[ty][1]);
	let t = bench(() => torch.gguf_moe_gemv(x, ids, w, ty, 6), 50);

	printf('%-34s %9.1f %9.0f\n', sprintf('gate/up %s %dx%d', ty, N, K), t * 1e6, bytes / t / 1e9);

	let x2 = torch.rand([6, N], 'float16', DEV);
	let w2 = torch.rand([E, K, N / 256 * SIZES[ty][1]], 'float16', DEV).to('uint8');

	t = bench(() => torch.gguf_moe_gemv(x2, ids, w2, ty, 1), 50);
	printf('%-34s %9.1f %9.0f\n', sprintf('down %s %dx%d', ty, K, N), t * 1e6, 6 * K * (N / 256 * SIZES[ty][1]) / t / 1e9);
}
