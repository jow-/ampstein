// examples/bench_fp4_moe.uc — torch.fp4_moe_gemv at the drafter's shapes (128 experts, 15 slots).
//   ucode ... examples/bench_fp4_moe.uc [device]
const dv = 'cuda:' + (ARGV[0] || 0);
const E = 128;

function now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

function bench(N, K, slots, xdiv) {
	let w = [], s = [], keep = [];

	for (let e = 0; e < E; e++) {
		let wt = torch.rand([N, K / 2], 'float16', dv).to('int8');
		let st = torch.full([N, K / 32], 127.0, 'float32', dv).to('float8_e8m0fnu');

		push(keep, wt, st);
		push(w, wt.data_ptr());
		push(s, st.data_ptr());
	}

	let wp = torch.from_array(w, 'int64', dv), sp = torch.from_array(s, 'int64', dv);
	let ids = [];

	for (let i = 0; i < slots; i++)
		push(ids, (i * 37 + 5) % E);

	let ids_t = torch.from_array(ids, 'int64', dv);
	let x = torch.rand([int(slots / xdiv), K], 'float16', dv);

	torch.fp4_moe_gemv(x, ids_t, wp, sp, N, xdiv);
	torch.synchronize();

	// 50 launches inside one CUDA graph: device time without the host call overhead
	let g = torch.graph();
	let keepo = [];

	g.capture_begin(+(ARGV[0] || 0));

	for (let i = 0; i < 50; i++)
		push(keepo, torch.fp4_moe_gemv(x, ids_t, wp, sp, N, xdiv));

	g.capture_end();
	torch.synchronize();
	g.replay();
	torch.synchronize();

	let t0 = now();

	g.replay();
	torch.synchronize();

	let ms = (now() - t0) / 50;
	let bytes = slots * N * K / 2 + slots * N * K / 32;

	printf('N=%d K=%d slots=%d: %.3f ms, %.0f GB/s\n', N, K, slots, ms, bytes / ms / 1e6);
}

bench(2304, 5120, 15, 3);
bench(5120, 2304, 15, 1);
