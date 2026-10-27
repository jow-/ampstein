// tests/qwen4_gguf_moe_prefill.uc — the pointer-table prefill kernel of the GGUF routed experts (torch.gguf_moe_prefill_pt, resident and
// host-mapped experts in one launch) against an fp32 reference built from the dequantized weights, next to the per-expert
// "dequantize to fp16 + cuBLAS" loop it replaces. The kernel must be at least as close to the reference as the loop.
// Env: AMP_MODEL_QWEN4_GGUF (a GGUF bundle dir), AMP_QWEN4_LAYERS=a,b,c (default 0,13,26,39).

import * as qwen4 from 'ampstein.qwen4';
import * as glm5 from 'ampstein.glm5';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4_GGUF') || '/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS';
const LAYERS = map(split(getenv('AMP_QWEN4_LAYERS') || '0,13,26,39', ','), (x) => +x);
const T = 1024;

if (!stat(MODEL) || torch.device_count() < 1) {
	warn('skip: GGUF Qwen4 model dir missing or no GPU\n');
	exit(125);
}

qwen4.configure({resident: 0.5});

let m = qwen4.load(MODEL, {mtp: false});
let cfg = m.cfg, Hd = cfg.hidden_size, Kt = cfg.num_experts_per_tok;
let bad = 0;

function rel(a, b) {
	let d = a.sub(b);

	return d.mul(d).sum().tolist() / b.mul(b).sum().tolist();
}

for (let li in LAYERS) {
	let M = m.layers[li].moe, dv = 'cuda:' + M.dev, gq = M.gq, E = M.n_experts_total;
	let xs = [];

	for (let i = 0; i < T * Hd; i++)
		push(xs, (((i * 2654435761 + li * 97) % 2003) - 1001) / 800.0);

	let y = torch.from_array(xs, 'float32', dv).to('float16').reshape([T, Hd]);
	let r = glm5.moe_route(cfg, M, y);
	let pr = torch.moe_prep(r.idx.contiguous(), r.w.contiguous(), E, 0);
	let tok = pr[2], wts = pr[3].to('float32');
	let count = pr[4].tolist(), start = pr[5].tolist();
	let ref = torch.zeros([T, Hd], 'float32', dv), loop = torch.zeros([T, Hd], 'float32', dv);

	for (let e = 0; e < E; e++) {
		let n = count[e];

		if (n < 1)
			continue;

		let rs = M.res_of[e];
		let src = (rs >= 0) ? gq : M.gq_cold;
		let local = (rs >= 0) ? rs : M.cold_of[e];
		let rows = tok.narrow(0, start[e], n);
		let ye = y.index(rows);
		let w = wts.narrow(0, start[e], n).view([n, 1]);

		let mat = (t, wm, nr, dt) => torch.gguf_dequant(wm.narrow(0, local, 1).view([nr, -1]), t, dt);
		let g32 = torch.matmul(ye.to('float32'), mat(src.tw1, src.w1, gq.inter, 'float32').t());
		let u32 = torch.matmul(ye.to('float32'), mat(src.tw3, src.w3, gq.inter, 'float32').t());
		let d32 = torch.matmul(g32.silu().mul(u32), mat(src.tw2, src.w2, Hd, 'float32').t());

		ref.index_add_(0, rows, d32.mul(w));

		let h = glm5.act_mul(cfg, torch.matmul(ye, mat(src.tw1, src.w1, gq.inter, 'float16').t()),
			torch.matmul(ye, mat(src.tw3, src.w3, gq.inter, 'float16').t()));

		loop.index_add_(0, rows, torch.matmul(h, mat(src.tw2, src.w2, Hd, 'float16').t()).to('float32').mul(w));
	}

	let scratch = torch.gguf_moe_prefill_pt(y, pr[2], pr[3], pr[4], pr[5], M.pt1, M.pt3, M.pt2, gq.tw1, gq.tw2, gq.inter,
		cfg.swiglu_limit + 0.0);
	let sum = torch.zeros([T, Hd], 'float32', dv);

	torch.moe_gather(sum, scratch, pr[0], pr[6], pr[5].narrow(0, 0, E), pr[5].narrow(0, 0, E), pr[7].narrow(0, 0, E), pr[3]);

	let e_new = rel(sum, ref), e_loop = rel(loop, ref);

	printf('layer %d (%s/%s, %d cold): relative error^2 vs fp32  kernel %.3e  loop %.3e\n', li, gq.tw1, gq.tw2,
		length(filter(M.res_of, (x) => x < 0)), e_new, e_loop);

	if (!(e_new < 4 * e_loop + 1e-8)) {
		warn(`FAIL layer ${li}: the kernel is further from the reference than the loop\n`);
		bad++;
	}
}

exit(bad ? 1 : 0);
