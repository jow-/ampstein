// examples/bench_dsv41_moe.uc — one MoE layer of the V4.1 pack: routing, shared expert and the fused expert
// kernels per chunk size.   ucode ... examples/bench_dsv41_moe.uc <model_dir> [layer]
import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';
import * as glm5 from 'ampstein.glm5';

let dir = ARGV[0], layer = +(ARGV[1] || 10);
let cfg = dsv41.load_config(dir);
let ws = weights.open(dir);
let pre = `layers.${layer}.`;

ws.upload((k) => (index(k, pre) == 0 && index(k, 'engram.embed') < 0) ? 0 : null);

let M = dsv41.build_moe(ws, cfg, layer, 0);

glm5.set_fused_rows(2048);

function now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

function timeit(name, f, n) {
	f();
	torch.synchronize();

	let t0 = now();

	for (let i = 0; i < n; i++)
		f();

	torch.synchronize();
	printf('  %-28s %.2f ms\n', name, (now() - t0) / n);
}

for (let T in [ 256, 512, 1024, 2048 ]) {
	let y = torch.rand([T, 5120], 'float16', 'cuda:0').sub(0.5);

	printf('T=%d\n', T);
	timeit('moe_forward (total)', () => glm5.moe_forward(cfg, M, y), 5);
	timeit('  route', () => glm5.moe_route(cfg, M, y), 5);
	timeit('  shared expert', () => glm5.mlp_forward(cfg, M.shared, y), 5);
}
