// examples/moe_cpu_bench.uc — CPU experts (torch.moe_cpu_*) against the fused GPU kernels on one MoE layer of a
// DeepSeek-V4.1 EXL3 pack with a single codebook (mul1): accuracy of the CPU result and its speed per call.
//   ucode ... examples/moe_cpu_bench.uc <model_dir> [layer] [rows]
let torch = require('ampstein.torch');
global.torch = torch;
import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';
import * as glm5 from 'ampstein.glm5';

function math_sqrt(v) {
	return v ** 0.5;
}

let dir = ARGV[0], layer = +(ARGV[1] || 5), T = +(ARGV[2] || 1);
let cfg = dsv41.load_config(dir);
let ws = weights.open(dir);
let pre = `layers.${layer}.`;

ws.upload((k) => (index(k, pre) == 0 && index(k, 'engram.embed') < 0) ? 0 : null);

let M = dsv41.build_moe(ws, cfg, layer, 0);
let NE = cfg.n_routed_experts, K = cfg.num_experts_per_tok, H = cfg.hidden_size;

function now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

// CPU copies of all experts (the mmap'ed file pages are not what a server would read from)
function copy_of(t) {
	let c = torch.empty(t.shape(), t.dtype());

	c.copy_(t);

	return c;
}

let arr = {};

for (let m in ['w1', 'w3', 'w2'])
	for (let part in ['trellis', 'suh', 'svh']) {
		let a = [];

		for (let e = 0; e < NE; e++)
			push(a, copy_of(ws.tensor(`${pre}ffn.experts.${e}.${m}.${part}`)));

		arr[m + part] = a;
	}

let h = torch.moe_cpu_make_layer(arr.w1trellis, arr.w1suh, arr.w1svh, arr.w3trellis, arr.w3suh, arr.w3svh,
	arr.w2trellis, arr.w2suh, arr.w2svh, cfg.swiglu_limit + 0.0);

let xs = [], seed = 12345;

for (let i = 0; i < T * H; i++) {
	seed = (seed * 1103515245 + 12345) % 2147483648;
	push(xs, ((seed % 20000) - 10000) / 10000.0);
}

let y = torch.from_array(xs, 'float32', 'cuda:0').to('float16').reshape([T, H]);

glm5.moe_forward(cfg, M, y);       // builds M.coop
torch.synchronize();

let r = glm5.moe_route(cfg, M, y);
let w16 = r.w.to('float16').contiguous();

torch.moe_coop(y, r.idx.contiguous(), w16, M.coop, null);
torch.synchronize();

let ref = M.coop.out.narrow(0, 0, T).cpu();
let sel = r.idx.to('int32').cpu().contiguous();
let wh = w16.cpu();
let xh = y.cpu();
let out = torch.zeros([T, H], 'float32');

for (let th in [ 4, 8, 12, 16, 24 ]) {
	torch.moe_cpu_forward(h, xh, sel, wh, out, th);

	let t0 = now();

	for (let i = 0; i < 20; i++)
		torch.moe_cpu_forward(h, xh, sel, wh, out, th);

	printf('threads %2d: %.3f ms per call (%d rows, %d picks)\n', th, (now() - t0) / 20, T, T * K);
}

let d = out.sub(ref);
let num = d.mul(d).sum().item(), den = ref.mul(ref).sum().item();

printf('relative error (rms) CPU vs GPU: %.4f   (|ref| rms %.3f)\n', math_sqrt(num / den), math_sqrt(den / (T * H)));
