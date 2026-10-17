// examples/bench_fp8_gemv.uc — torch.fp8_gemv on the shapes of the V4.1 decode: GB/s of the weight read.
function now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

for (let sh in [ [32768, 1280], [5120, 8192], [1280, 5120], [1024, 4096], [2304, 5120], [5120, 2304], [512, 5120], [4096, 1280] ]) {
	let N = sh[0], K = sh[1];
	let w = torch.rand([N, K], 'float32', 'cuda:0').to('float8_e4m3fn').contiguous();
	let s = torch.ones([int(N / 32), int(K / 32)], 'float32', 'cuda:0').to('float8_e8m0fnu').contiguous();
	let x = torch.rand([1, K], 'float16', 'cuda:0');

	torch.fp8_gemv(x, w, s);
	torch.synchronize();

	let t0 = now();

	for (let i = 0; i < 200; i++)
		torch.fp8_gemv(x, w, s);

	torch.synchronize();

	let ms = (now() - t0) / 200;

	printf('N=%5d K=%5d: %.1f us, %.0f GB/s\n', N, K, ms * 1000, N * K / ms / 1e6);
}
