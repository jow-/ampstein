// examples/bench_zerocopy_qwen4.uc — bench_zerocopy.uc for Qwen3.8-Flash-Next (512 experts, top-10, 640 wide).
// experts' trellis data in VRAM vs. in pinned (UVA-mapped) host memory vs. half and half.
// docs/EXPERT_STREAMING.md.
//   ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/bench_zerocopy.uc [dev] [nexp]
import * as glm5 from 'ampstein.glm5';
import * as qwen4 from 'ampstein.qwen4';
import * as weights from 'ampstein.weights';

const MODEL = getenv('AMP_MODEL_QWEN4') || '/data/models/Qwen3.8-Flash-Next-exl3-2.05bpw';
let dev = int(ARGV[0] ?? 0);
let NE = int(ARGV[1] ?? 64);
let dv = 'cuda:' + dev;
let cfg = qwen4.load_config(MODEL);
let ws = weights.open(MODEL);
const LAYER = 3;
const ROWS = 8;
const K = cfg.num_experts_per_tok;

let dexp = [], hexp = [];
for (let e = 0; e < NE; e++) {
	let m = glm5.build_mlp(ws, `${qwen4.LP}${LAYER}.mlp.experts.${e}`, dev);
	let h = {};
	for (let n in ['gate', 'up', 'down'])
		h[n] = { trellis: m[n].trellis.cpu().pin(), suh: m[n].suh, svh: m[n].svh };
	push(dexp, m);
	push(hexp, h);
}
torch.synchronize();

function coop(ex, g0, u0, d0) {
	function table(name, field) {
		let a = [];
		for (let e in ex)
			push(a, e[name][field].data_ptr());
		return torch.from_array(a, 'int64', dv);
	}
	let Hi = g0.in_features, I = g0.out_features, Ho = d0.out_features;
	let slots = ROWS * K;
	let ctr_len = slots * int(I / 128) + ROWS * int(Ho / 128) + 2 + (slots + 1) + slots;
	return {
		g_trellis: table('gate', 'trellis'), g_suh: table('gate', 'suh'), g_svh: table('gate', 'svh'),
		u_trellis: table('up', 'trellis'), u_suh: table('up', 'suh'), u_svh: table('up', 'svh'),
		d_trellis: table('down', 'trellis'), d_suh: table('down', 'suh'), d_svh: table('down', 'svh'),
		had_g: torch.zeros([slots, Hi], 'float16', dv), had_u: torch.zeros([slots, Hi], 'float16', dv),
		gu_g: torch.zeros([slots, I], 'float16', dv), gu_u: torch.zeros([slots, I], 'float16', dv),
		act_out: torch.zeros([slots, I], 'float16', dv), d_out: torch.zeros([slots, Ho], 'float32', dv),
		ctr: torch.zeros([ctr_len], 'int32', dv), out: torch.zeros([ROWS, cfg.hidden_size], 'float32', dv),
		Hi: Hi, Kg: g0.K, Ku: u0.K, Kd: d0.K, mcg: g0.mcg ? true : false,
		mul1: g0.mul1 ? true : false, act: 0, act_limit: cfg.swiglu_limit + 0.0,
		min_expert: -1, max_expert: -1
	};
}

let g0 = dexp[0].gate, u0 = dexp[0].up, d0 = dexp[0].down;
let A = coop(dexp, g0, u0, d0);
let B = coop(hexp, g0, u0, d0);
// mixed: first half device, second half host
let mix = [];
for (let e = 0; e < NE; e++) push(mix, (e % 2) ? hexp[e] : dexp[e]);
let C = coop(mix, g0, u0, d0);

let H = cfg.hidden_size;
let xs = [];
for (let i = 0; i < ROWS * H; i++) push(xs, ((i * 2654435761) % 1000) / 20000.0 - 0.025);
let X = torch.from_array(xs, 'float32', dv).to('float16').reshape([ROWS, H]);

function sel(T, it) {
	let a = [];
	for (let t = 0; t < T; t++)
		for (let k = 0; k < K; k++)
			push(a, (it * T * K + t * K + k) % NE);
	return torch.from_array(a, 'int64', dv).reshape([T, K]);
}

function rwl(T) {
	let a = [];
	for (let i = 0; i < T * K; i++)
		push(a, 0.125);
	return a;
}

function bench(label, S, T, picker) {
	let iters = 40;
	let sels = [], rws = [];
	for (let i = 0; i < iters; i++) {
		push(sels, picker ? picker(T, i) : sel(T, i));
		push(rws, torch.from_array(rwl(T), 'float32', dv).to('float16').reshape([T, K]));
	}
	let x = X.narrow(0, 0, T).contiguous();
	for (let i = 0; i < 4; i++)
		torch.moe_coop(x, sels[i], rws[i], S, null);
	torch.synchronize();
	let c0 = clock(true);
	for (let i = 0; i < iters; i++)
		torch.moe_coop(x, sels[i], rws[i], S, null);
	torch.synchronize();
	let c1 = clock(true);
	let ms = ((c1[0] - c0[0]) + (c1[1] - c0[1]) / 1e9) * 1000 / iters;
	printf('%-28s T=%d: %.3f ms/layer-call\n', label, T, ms);
}

// the same inputs through the device, host and mixed tables must give identical bits
for (let T in [1, 3, 8]) {
	let x = X.narrow(0, 0, T).contiguous();
	let sl = sel(T, 5);
	let rw = torch.from_array(rwl(T), 'float32', dv).to('float16').reshape([T, K]);
	let o = [];

	for (let S in [A, B, C, A]) {
		torch.moe_coop(x, sl, rw, S, null);
		push(o, S.out.narrow(0, 0, T).cpu());
	}

	torch.synchronize();
	printf('T=%d: host == device: %s, mixed == device: %s, device repeat == device: %s\n', T,
		o[1].sub(o[0]).abs().max().item() == 0 ? 'identical' : 'DIFFERENT',
		o[2].sub(o[0]).abs().max().item() == 0 ? 'identical' : 'DIFFERENT',
		o[3].sub(o[0]).abs().max().item() == 0 ? 'identical' : 'DIFFERENT');
}

for (let T in [1, 3, 8]) {
	bench('device (VRAM)', A, T);
	bench('pinned host (zero-copy)', B, T);
	bench('half/half', C, T);
}
// even expert ids only = all hits in the mixed table
let even = (T, it) => {
	let a = [];
	for (let i = 0; i < T * K; i++)
		push(a, ((it * T * K + i) * 2) % NE);
	return torch.from_array(a, 'int64', dv).reshape([T, K]);
};
bench('mixed table, all hits', C, 1, even);
bench('mixed table, all hits', C, 3, even);
printf('bytes per expert (trellis): %d\n', g0.trellis.numel() * 2 * 3);
