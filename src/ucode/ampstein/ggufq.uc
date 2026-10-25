// ucode/ggufq.uc — layers over ggml-quantized (GGUF) weights: dense linears and the routed experts of DeepSeek-V4.1.
//
// The weights stay in their file format on the GPU (uint8 rows of blocks: Q8_0, Q5_K, IQ2_XXS, IQ2_XS, IQ1_M, ...; see
// ggufw.uc, src/torch/amp_gguf.cu). Up to 8 rows (decode, speculative verify) go through the matrix-vector kernels
// torch.gguf_gemv / torch.gguf_moe_gemv, which dequantize in registers; longer inputs (prefill) dequantize a slice of the
// matrix into fp16 and run cuBLAS.

const SCRATCH_ELEMS = +(getenv("AMP_GGUF_SLICE") ?? 8 * 1024 * 1024);     // dequantized slice of a prefill matmul, elements (fp16: 16 MiB)
const PF_OFF = getenv('AMP_GGUF_NO_PF') != null;                    // the per-expert dequantize + cuBLAS prefill instead of the grouped kernel
const PF_TYPES = {IQ2_XXS: true, IQ2_XS: true, IQ1_M: true, IQ1_S: true};
const FUSED_OFF = getenv('AMP_GGUF_NO_FUSED') != null || getenv('AMP_GGUF_FP16') != null;           // separate gate / up launches
const SHARED_PAR_OFF = getenv('AMP_GGUF_NO_SHARED_PAR') != null;                                       // the shared expert on the compute stream
const GROUP_OFF = getenv('AMP_GGUF_GROUP') == null;                                                    // verify steps: one slot per pick (AMP_GGUF_GROUP=1: grouped by expert, slower on random picks)
const ATOMIC = getenv('AMP_MOE_ATOMIC') != null;                                                       // prefill: sum the experts' outputs with atomic adds (faster, not reproducible)
const GROUPED_OFF = getenv('AMP_GGUF_NO_GROUPED') != null || getenv('AMP_GGUF_FP16') != null;     // per-group launches instead

/* x (M, K) fp16 @ dequant(raw[r0 .. r0 + n))^T -> (M, n) fp16, in row slices */
function matmul_rows(x, raw, qt, K, r0, n) {
	let per = max(1, int(SCRATCH_ELEMS / K));
	let parts = [];

	for (let a = 0; a < n; a += per) {
		let m = min(per, n - a);

		push(parts, torch.matmul(x, torch.gguf_dequant(raw.narrow(0, r0 + a, m), qt, 'float16').t()));
	}

	return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
}

/* elements and bytes of a block of each type (column slices of a quantized matrix are cut at block boundaries) */
const BLK = {Q8_0: [ 32, 34 ], IQ4_NL: [ 32, 18 ], Q5_K: [ 256, 176 ], IQ2_XXS: [ 256, 66 ], IQ2_XS: [ 256, 74 ], IQ1_M: [ 256, 56 ], IQ1_S: [ 256, 50 ]};

/* The linear over rows `raw` (N, row bytes) of type qt: forward(x) with x (M, K) fp16 -> (M, N) fp16 */
export function linear_from(raw, qt, dev) {
	let blk = BLK[qt];

	return {
		raw: raw, qtype: qt, dev: dev, out_features: raw.shape()[0], in_features: raw.shape()[1] / blk[1] * blk[0],

		forward: function (x) {
			if (x.shape()[0] <= 8)
				return torch.gguf_gemv(x.contiguous(), this.raw, this.qtype);

			return matmul_rows(x, this.raw, this.qtype, this.in_features, 0, this.out_features);
		}
	};
}

/* The linear under `key` (key + '.weight' a GGUF block-quantized matrix): forward(x) with x (M, K) fp16 -> (M, N) fp16 */
export function linear(ws, key, dev) {
	let sh = ws.info(key + '.weight').shape;
	let l = linear_from(ws.gpu(key + '.weight', dev), ws.qtype(key + '.weight'), dev);

	l.out_features = sh[0];
	l.in_features = sh[1];

	return l;
}

/* tensor-parallel shards (copies on dev): all of it, output rows [r0, r0 + n), input columns [c0, c0 + n) (a block multiple) */
export function linear_to(l, dev) {
	return linear_from(l.raw.cuda(dev), l.qtype, dev);
}

export function linear_rows(l, r0, n, dev) {
	return linear_from(l.raw.narrow(0, r0, n).contiguous().cuda(dev), l.qtype, dev);
}

export function linear_cols(l, c0, n, dev) {
	let blk = BLK[l.qtype];

	if (c0 % blk[0] || n % blk[0])
		die(`ggufq: columns [${c0}, ${c0 + n}) of a ${l.qtype} matrix are not block aligned`);

	return linear_from(l.raw.narrow(1, c0 / blk[0] * blk[1], n / blk[0] * blk[1]).contiguous().cuda(dev), l.qtype, dev);
}

/* A gated MLP over three linears (gate / up / down): when gate and up are of one type, for up to 8 rows gate, up and the clamped
 * SwiGLU are one launch (M.fused(x, limit) -> (M, inter) fp16, glm5.mlp_forward), then the down projection. */
export function mlp_from(gate, up, down) {
	return {
		gate: gate, up: up, down: down,
		fused: (FUSED_OFF || gate.qtype != up.qtype) ? null : function (x, lim) {
			return torch.gguf_gate_up(x.contiguous(), gate.raw, up.raw, gate.qtype, lim);
		}
	};
}

/* The shared expert (key + '.w1' / '.w3' / '.w2' = gate / up / down) */
export function mlp(ws, key, dev) {
	return mlp_from(linear(ws, key + '.w1', dev), linear(ws, key + '.w3', dev), linear(ws, key + '.w2', dev));
}

/* Grouped linear (the attention's wo_a): G blocks of R rows, block g sees columns [g * K, (g + 1) * K) of x only.
 * x: (M, G * K) fp16 -> (M, G * R). The matrix is stored (G * R, K). */
export function grouped_from(raw, qt, G, dev) {
	let blk = BLK[qt];

	return {
		raw: raw, qtype: qt, dev: dev, groups: G, rows: int(raw.shape()[0] / G), in_features: raw.shape()[1] / blk[1] * blk[0],

		forward: function (x) {
			let M = x.shape()[0], K = this.in_features, R = this.rows;
			let parts = [];

			// decode: all groups in one launch
			if (M <= 8 && !GROUPED_OFF && this.qtype != 'IQ4_NL')
				return torch.gguf_gemv_grouped(x.contiguous(), this.raw, this.qtype, this.groups);

			for (let g = 0; g < this.groups; g++) {
				let xg = x.narrow(1, g * K, K).contiguous();

				push(parts, (M <= 8) ? torch.gguf_gemv(xg, this.raw.narrow(0, g * R, R), this.qtype) :
					matmul_rows(xg, this.raw, this.qtype, K, g * R, R));
			}

			return torch.cat(parts, 1);
		}
	};
}

export function linear_grouped(ws, key, G, dev) {
	let raw = ws.gpu(key + '.weight', dev);
	let sh = ws.info(key + '.weight').shape;

	// (the file stores wo_a as (G, rows, K); the rows of all groups are one matrix (G * rows, K))
	if (length(sh) == 3)
		raw = raw.view([sh[0] * sh[1], -1]);

	return grouped_from(raw, ws.qtype(key + '.weight'), G, dev);
}

/* groups [g0, g0 + n) of a grouped linear, copied to dev */
export function grouped_groups(l, g0, n, dev) {
	return grouped_from(l.raw.narrow(0, g0 * l.rows, n * l.rows).contiguous().cuda(dev), l.qtype, n, dev);
}

/* The routed experts of a layer: the three stacked matrices (E, rows, row bytes) of gate (w1), up (w3) and down (w2) with
 * their types. A shard of experts [e0, e0 + n) is read from the file and uploaded alone (never the whole stack). */
export function experts(ws, key, dev, e0, n) {
	let gq = {dev: dev, e0: e0 ?? 0};

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let k = `${key}.exps.${m}`;

		gq[m] = (n != null) ? ws.tensor(k).narrow(0, e0, n).cuda(dev) : ws.gpu(k, dev);
		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = gq.w1.shape()[0];
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* Routed + shared experts of one layer for y (T, hidden) fp16 -> (T, hidden) fp32. H: the glm5 module's {route, act,
 * mlp} (passed in: glm5.uc cannot import this module). */
export function moe_forward(cfg, M, y, H) {
	let gq = M.gq;
	let T = y.shape()[0], Kt = cfg.num_experts_per_tok, Hd = cfg.hidden_size;
	let r = H.route(cfg, M, y);
	let out, sh = null;

	if (T <= 8) {
		// decode / verify: every (token, pick) slot is one expert; the kernels dequantize in registers. Gate and up are one
		// launch with the clamped SwiGLU in registers, the routing weight is applied by the down projection
		let ids = r.idx.reshape([-1]).contiguous();
		let yc = y.contiguous();
		let h;

		// 2 .. 8 rows (a verify step): the picks grouped by expert, so an expert's weights are read once for every row that chose it
		let grp = (T >= 2 && !GROUP_OFF && !FUSED_OFF) ? torch.gguf_moe_group(ids) : null;

		// the shared expert on a forked stream while the routed ones run (graph capture only: its kernels are short and read the DRAM the
		// routed ones leave idle); joined before the sum
		sh = (!SHARED_PAR_OFF && H.side) ? H.side(M.dev, 1, () => H.mlp(cfg, M.shared, y).to('float32')) : null;

		if (gq.tw1 == gq.tw3 && !FUSED_OFF) {
			h = torch.gguf_moe_gate_up(yc, ids, gq.w1, gq.w3, gq.tw1, Kt, cfg.swiglu_limit + 0.0, gq.e0, grp, T);
		}
		else {
			let g = torch.gguf_moe_gemv(yc, ids, gq.w1, gq.tw1, Kt, null, gq.e0).to('float16');
			let u = torch.gguf_moe_gemv(yc, ids, gq.w3, gq.tw3, Kt, null, gq.e0).to('float16');

			h = H.act(cfg, g, u);
		}

		out = torch.gguf_moe_gemv(h, ids, gq.w2, gq.tw2, 1, r.w.reshape([-1]).contiguous(), gq.e0, grp, T).view([T, Kt, Hd]).sum(1);
	}
	else {
		// prefill: tokens sorted by expert; an expert's three matrices are dequantized to fp16 and run through cuBLAS
		let E = gq.n_experts, I = gq.inter;
		let pr = torch.moe_prep(r.idx.contiguous(), r.w.contiguous(), E, gq.e0);

		if (!PF_OFF && gq.tw1 == gq.tw3 && PF_TYPES[gq.tw1] && PF_TYPES[gq.tw2]) {
			// tensor cores, weights dequantized into shared memory tiles (no host readback)
			if (!ATOMIC) {
				// every assignment's weighted output in its own scratch row, summed per token in a fixed order: reproducible to the bit
				// (an atomic sum rounds differently in every run, which the depth of the network amplifies into different tokens)
				let scratch = torch.gguf_moe_prefill(y.contiguous(), pr[2], pr[3], pr[4], pr[5], gq.w1, gq.w3, gq.w2, gq.tw1, gq.tw2,
					cfg.swiglu_limit + 0.0, true);
				let sum = torch.zeros([T, Hd], 'float32', 'cuda:' + M.dev);

				torch.moe_gather(sum, scratch, pr[0], pr[6], pr[5].narrow(0, 0, E), pr[5].narrow(0, 0, E), pr[7].narrow(0, 0, E), pr[3]);

				return sum.add(H.mlp(cfg, M.shared, y).to('float32'));
			}

			return torch.gguf_moe_prefill(y.contiguous(), pr[2], pr[3], pr[4], pr[5], gq.w1, gq.w3, gq.w2, gq.tw1, gq.tw2,
				cfg.swiglu_limit + 0.0).add(H.mlp(cfg, M.shared, y).to('float32'));
		}

		let tok = pr[2], wts = pr[3].to('float32');
		let count = pr[4].tolist(), start = pr[5].tolist();

		out = torch.zeros([T, Hd], 'float32', 'cuda:' + M.dev);

		for (let e = 0; e < E; e++) {
			let n = count[e];

			if (n < 1)
				continue;

			let rows = tok.narrow(0, start[e], n);
			let ye = y.index(rows);
			let w1 = torch.gguf_dequant(gq.w1.narrow(0, e, 1).view([I, -1]), gq.tw1, 'float16');
			let w3 = torch.gguf_dequant(gq.w3.narrow(0, e, 1).view([I, -1]), gq.tw3, 'float16');
			let h = H.act(cfg, torch.matmul(ye, w1.t()), torch.matmul(ye, w3.t()));
			let w2 = torch.gguf_dequant(gq.w2.narrow(0, e, 1).view([Hd, -1]), gq.tw2, 'float16');
			let d = torch.matmul(h, w2.t()).to('float32');

			out.index_add_(0, rows, d.mul(wts.narrow(0, start[e], n).view([n, 1])));
		}
	}

	if (sh) {
		H.join(M.dev, 1);

		return out.add(sh);
	}

	return out.add(H.mlp(cfg, M.shared, y).to('float32'));
}
