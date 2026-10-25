// examples/gguf_expert_check.uc — are the GGUF's routed experts the same experts as the EXL3 pack's? For a few experts of a few layers:
// the output of the whole expert (down(act(gate(y), up(y)))) for the same input rows, from the GGUF (torch.gguf_moe_gemv) and from the
// EXL3 pack's expert. Two different quantizations of the same weights: cosines well above 0 (0.8 - 0.95 at 1.6 - 2 bpw) when the expert
// index and the gate / up / down roles agree, ~0 when not. Also checks the GGUF path against its own dequantized weights.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/gguf_expert_check.uc GGUF_DIR EXL3_DIR [layers=8] [gpus=4] [layer...]

import * as dsv41 from 'ampstein.dsv41';
import * as glm5 from 'ampstein.glm5';

let gdir = ARGV[0], edir = ARGV[1];
let NL = +(ARGV[2] || 8), NG = +(ARGV[3] || 4);
let which = (length(ARGV) > 4) ? map(slice(ARGV, 4), (x) => +x) : [ 0, 3, NL - 1 ];

let g = dsv41.load(gdir, {devices: NG, layers: NL, max_seq: 256, prefill_rows: 64});
let e = (edir == '-') ? null : dsv41.load(edir, {devices: NG, layers: NL, max_seq: 256, prefill_rows: 64});        // '-': the GGUF checks only
let cfg = g.cfg, H = cfg.hidden_size;

function cos(a, b) {
	a = a.to('float32').reshape([-1]).cpu();
	b = b.to('float32').reshape([-1]).cpu();

	return a.mul(b).sum().item() / ((a.mul(a).sum().item() * b.mul(b).sum().item()) ** 0.5);
}

let worst = 1.0;

for (let l in which) {
	let Mg = g.layers[l].moe, Me = e ? e.layers[l].moe : null, gq = Mg.gq;
	let dg = 'cuda:' + Mg.dev;
	let y = torch.rand([4, H], 'float32', dg).sub(0.5).mul(3.4).to('float16');          // ~ unit rms, like a normed hidden state
	let line = '';

	for (let ex in [ 0, 1, 5, 100, 255, 383 ]) {
		let ids = torch.from_array([ ex, ex, ex, ex ], 'int64', dg);
		let a = torch.gguf_moe_gemv(y, ids, gq.w1, gq.tw1, 1).to('float16'), b = torch.gguf_moe_gemv(y, ids, gq.w3, gq.tw3, 1).to('float16');
		let out_g = torch.gguf_moe_gemv(glm5.act_mul(cfg, a, b), ids, gq.w2, gq.tw2, 1);

		// the same from the dequantized matrices
		let W1 = torch.gguf_dequant(gq.w1.narrow(0, ex, 1).view([gq.inter, -1]), gq.tw1, 'float32');
		let W3 = torch.gguf_dequant(gq.w3.narrow(0, ex, 1).view([gq.inter, -1]), gq.tw3, 'float32');
		let W2 = torch.gguf_dequant(gq.w2.narrow(0, ex, 1).view([H, -1]), gq.tw2, 'float32');
		let ref = torch.matmul(glm5.act_mul(cfg, torch.matmul(y.to('float32'), W1.t()).to('float16'),
			torch.matmul(y.to('float32'), W3.t()).to('float16')).to('float32'), W2.t());
		let c_self = cos(out_g, ref);

		if (Me) {
			let out_e = glm5.mlp_forward(cfg, Me.experts[ex], y.cuda(Me.dev).to('float16'));
			let c_x = cos(out_g, out_e);
			let c_ctl = cos(out_g, glm5.mlp_forward(cfg, Me.experts[(ex + 1) % 384], y.cuda(Me.dev).to('float16')));      // another expert: the control

			line += sprintf(' e%d: self %.4f x-pack %.4f (ctl %.4f)', ex, c_self, c_x, c_ctl);
			worst = min(worst, c_x);
		}
		else {
			line += sprintf(' e%d: self %.4f', ex, c_self);
		}

		if (c_self < 0.995)
			warn(sprintf('layer %d expert %d: the GGUF kernels differ from the dequantized matrices (cos %.5f)\n', l, ex, c_self));
	}

	// the prefill path (tokens sorted by expert, dequantized matrices through cuBLAS) against the decode path (GEMV kernels)
	let yp = torch.rand([48, H], 'float32', dg).sub(0.5).mul(3.4).to('float16');
	let pre = glm5.moe_forward(cfg, Mg, yp);
	let dec = torch.cat(map([ 0, 1, 2, 3, 4, 5 ], (i) => glm5.moe_forward(cfg, Mg, yp.narrow(0, i * 8, 8).contiguous())), 0);
	let dd = pre.sub(dec);

	line += sprintf('  prefill vs decode path: rel rms %.2e', (dd.mul(dd).mean().item() / dec.mul(dec).mean().item()) ** 0.5);
	// the tensor-core prefill kernel against the exact reference (each active expert's matrices dequantized to fp16, cuBLAS)
	{
		let rt = glm5.moe_route(cfg, Mg, yp), E = gq.n_experts, I = gq.inter;
		let pr = torch.moe_prep(rt.idx.contiguous(), rt.w.contiguous(), E, 0);
		let got = torch.gguf_moe_prefill(yp, pr[2], pr[3], pr[4], pr[5], gq.w1, gq.w3, gq.w2, gq.tw1, gq.tw2, cfg.swiglu_limit + 0.0);
		let tok = pr[2], wt = pr[3].to('float32'), cnt = pr[4].tolist(), st = pr[5].tolist();
		let ref = torch.zeros([48, H], 'float32', dg);

		for (let ex = 0; ex < E; ex++) {
			if (cnt[ex] < 1)
				continue;

			let rows = tok.narrow(0, st[ex], cnt[ex]), ye = yp.index(rows);
			let h = glm5.act_mul(cfg, torch.matmul(ye, torch.gguf_dequant(gq.w1.narrow(0, ex, 1).view([I, -1]), gq.tw1, 'float16').t()),
				torch.matmul(ye, torch.gguf_dequant(gq.w3.narrow(0, ex, 1).view([I, -1]), gq.tw3, 'float16').t()));
			let d = torch.matmul(h, torch.gguf_dequant(gq.w2.narrow(0, ex, 1).view([H, -1]), gq.tw2, 'float16').t()).to('float32');

			ref.index_add_(0, rows, d.mul(wt.narrow(0, st[ex], cnt[ex]).view([cnt[ex], 1])));
		}

		let dd2 = got.sub(ref);

		line += sprintf('  tensor-core prefill vs dequant+cuBLAS: rel rms %.2e', (dd2.mul(dd2).mean().item() / ref.mul(ref).mean().item()) ** 0.5);
	}

	printf('layer %d (gate %s, up %s, down %s):%s\n', l, gq.tw1, gq.tw3, gq.tw2, line);
}

printf('worst cross-pack cosine %.4f\n', worst);
