// ucode/ggufq.uc — layers over ggml-quantized (GGUF) weights: dense linears and the routed experts of DeepSeek-V4.1.
//
// The weights stay in their file format on the GPU (uint8 rows of blocks: Q8_0, Q5_K, IQ2_XXS, IQ2_XS, IQ1_M, ...; see
// ggufw.uc, src/torch/amp_gguf.cu). Up to 8 rows (decode, speculative verify) go through the matrix-vector kernels
// torch.gguf_gemv / torch.gguf_moe_gemv, which dequantize in registers; longer inputs (prefill) dequantize a slice of the
// matrix into fp16 and run cuBLAS.

const SCRATCH_ELEMS = +(getenv("AMP_GGUF_SLICE") ?? 8 * 1024 * 1024);     // dequantized slice of a prefill matmul, elements (fp16: 16 MiB)
const PF_OFF = getenv('AMP_GGUF_NO_PF') != null;                    // the per-expert dequantize + cuBLAS prefill instead of the grouped kernel
const PT_OFF = getenv('AMP_GGUF_NO_PT') != null;                    // the pointer-table prefill kernel off (per-expert loop for cold experts)
const PF_TYPES = {IQ2_XXS: true, IQ2_XS: true, IQ1_M: true, IQ1_S: true};
const I8_TYPES = {Q8_0: true, Q5_K: true, Q6_K: true, Q4_K: true, IQ4_XS: true, IQ4_NL: true, Q2_0: true, IQ2_S: true, IQ3_XXS: true, IQ3_S: true, IQ2_XXS: true, IQ2_XS: true, IQ1_M: true, IQ1_S: true};
const FUSED_OFF = getenv('AMP_GGUF_NO_FUSED') != null || getenv('AMP_GGUF_FP16') != null;           // separate gate / up launches
const SHARED_PAR_OFF = getenv('AMP_GGUF_NO_SHARED_PAR') != null;                                       // the shared expert on the compute stream
const GROUP_OFF = getenv('AMP_GGUF_GROUP') == null;                                                    // verify steps: one slot per pick (AMP_GGUF_GROUP=1: grouped by expert, slower on random picks)
const ATOMIC = getenv('AMP_MOE_ATOMIC') != null;                                                       // prefill: sum the experts' outputs with atomic adds (faster, not reproducible)
const GROUPED_OFF = getenv('AMP_GGUF_NO_GROUPED') != null || getenv('AMP_GGUF_FP16') != null;     // per-group launches instead

/* x (M, K) fp16 @ dequant(raw[r0 .. r0 + n))^T -> (M, n) fp16, in row slices */
function matmul_rows(x, raw, qt, K, r0, n) {
	// the slice height is a multiple of 128 rows: the BLAS gemm of a slice runs at 80 TFLOPS on a gfx1100 for 1536 output columns and
	// at 33 for 1638 (the tiles of the odd shape waste most of their work)
	let per = max(1, int(SCRATCH_ELEMS / K));

	if (per >= 192)
		per = int(per / 128 + 0.5) * 128;

	let parts = [];

	for (let a = 0; a < n; a += per) {
		let m = min(per, n - a);

		push(parts, torch.matmul(x, torch.gguf_dequant(raw.narrow(0, r0 + a, m), qt, 'float16').t()));
	}

	return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
}

/* elements and bytes of a block of each type (column slices of a quantized matrix are cut at block boundaries) */
const BLK = {
	Q4_0: [ 32, 18 ], Q8_0: [ 32, 34 ], IQ4_NL: [ 32, 18 ], Q5_K: [ 256, 176 ],
	IQ2_XXS: [ 256, 66 ], IQ2_XS: [ 256, 74 ], IQ1_M: [ 256, 56 ], IQ1_S: [ 256, 50 ],
	Q2_0: [ 64, 18 ], Q4_K: [ 256, 144 ], Q3_K: [ 256, 110 ], Q6_K: [ 256, 210 ],
	IQ2_S: [ 256, 82 ], IQ3_XXS: [ 256, 98 ], IQ3_S: [ 256, 110 ], IQ4_XS: [ 256, 136 ]
};

/* The linear over rows `raw` (N, row bytes) of type qt: forward(x) with x (M, K) fp16 -> (M, N) fp16 */
export function linear_from(raw, qt, dev) {
	let blk = BLK[qt];

	return {
		raw: raw, qtype: qt, dev: dev, out_features: raw.shape()[0], in_features: raw.shape()[1] / blk[1] * blk[0],

		/* x: (M, K) fp16, or (B, S, K) (flattened to B * S rows and reshaped back: the dense models' layers pass sequences) */
		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x : x.view([-1, this.in_features]);
			let out;

			if (flat.shape()[0] <= 8)
				out = torch.gguf_gemv(flat.contiguous(), this.raw, this.qtype);
			else
				out = matmul_rows(flat, this.raw, this.qtype, this.in_features, 0, this.out_features);

			return (length(s) == 2) ? out : out.view([s[0], s[1], this.out_features]);
		}
	};
}

/* The linear under `key` (key + '.weight' a GGUF block-quantized matrix): forward(x) with x (M, K) fp16 -> (M, N) fp16 */
export function linear(ws, key, dev) {
	let sh = ws.info(key + '.weight').shape;
	let raw = ws.gpu(key + '.weight', dev);

	if (!raw)
		die(`ggufq: no GPU tensor for ${key}.weight`);

	let l = linear_from(raw, ws.qtype(key + '.weight'), dev);

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
		fused: (FUSED_OFF || gate.qtype != up.qtype || !I8_TYPES[gate.qtype]) ? null : function (x, lim) {
			return torch.gguf_gate_up(x.contiguous(), gate.raw, up.raw, gate.qtype, lim);
		}
	};
}

/* silu(x gate^T) * (x up^T) of two GGUF linears of one type as ONE launch (decode / verify rows: up to 8) -- null when they do not qualify.
 * x: (M, K) or (B, S, K) fp16; the result has the shape of x with the last dimension the gate's out_features. */
export function gate_up(gate, up, x) {
	if (FUSED_OFF || !gate.raw || !up.raw || gate.qtype != up.qtype || !I8_TYPES[gate.qtype])
		return null;

	let s = x.shape();
	let rows = (length(s) == 3) ? s[0] * s[1] : s[0];

	if (rows > 8)
		return null;

	let h = torch.gguf_gate_up(x.contiguous().view([ rows, -1 ]), gate.raw, up.raw, gate.qtype, 0);

	return (length(s) == 3) ? h.view([ s[0], s[1], gate.out_features ]) : h;
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

		if (n == null || (e0 == 0 && n == ws.info(k).shape[0]))
			gq[m] = ws.gpu(k, dev);
		else
			gq[m] = ws.tensor(k).narrow(0, e0, n).cuda(dev);

		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = gq.w1.shape()[0];
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* the cold part of an expert stack, pinned on the host: rows [e0, e0 + n), local index 0 .. n-1 */
export function experts_host(ws, key, e0, n) {
	let gq = {host: true, e0: 0};

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let k = `${key}.exps.${m}`;

		gq[m] = ws.tensor(k).narrow(0, e0, n);
		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = gq.w1.shape()[0];
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* A selected set of routed experts, in the order of `ids`: the three stacked matrices (n, rows, row bytes) with
 * their types. Used for frequency-based placement, where the resident set is not a contiguous prefix. */
export function experts_pick(ws, key, dev, ids) {
	let gq = {dev: dev, e0: 0};

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let k = `${key}.exps.${m}`;
		let raw = ws.tensor(k), parts = [];

		for (let i in ids)
			push(parts, raw.narrow(0, i, 1));

		gq[m] = torch.cat(parts, 0).cuda(dev);
		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = length(ids);
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* The cold complement of `experts_pick`, in CUDA-mapped pinned host memory. The tensor is CUDA and its rows are
 * read over PCIe by the dequant/gemv kernels without staging copies. Local index 0 .. n-1. */
export function experts_host_mapped(ws, key, ids, dev) {
	let gq = {dev: dev, e0: 0};

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let k = `${key}.exps.${m}`;
		let raw = ws.tensor(k), parts = [];

		for (let i in ids)
			push(parts, raw.narrow(0, i, 1));

		gq[m] = torch.host_mapped(torch.cat(parts, 0), dev);
		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = length(ids);
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* Exchange the cold expert `a` (a row of the mapped host stack) with the resident expert `b` (a row of the VRAM stack) in a layer
 * of qwen4.build_moe: the three matrices move on the compute stream (a decode graph replayed later sees the finished copies), the
 * placement tables res_of / cold_of and the host mirrors of the pointer tables follow. Call sync_tables(M) after a batch. */
export function swap_experts(M, a, b) {
	let r = M.res_of[b], c = M.cold_of[a];

	if (r == null || c == null || r < 0 || c < 0)
		return false;

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let dst = M.gq[m].narrow(0, r, 1), src = M.gq_cold[m].narrow(0, c, 1);
		let tmp = torch.empty(dst.shape(), 'uint8', 'cuda:' + M.dev);
		let sz = M.gq[m].shape()[1] * M.gq[m].shape()[2];
		let pt = M.pt_host['pt' + substr(m, 1)];

		tmp.copy_(dst);
		dst.copy_(src);
		src.copy_(tmp);

		pt[a] = M.gq[m].data_ptr() + r * sz;
		pt[b] = M.gq_cold[m].data_ptr() + c * sz;
	}

	M.res_of[a] = r;
	M.cold_of[a] = -1;
	M.res_of[b] = -1;
	M.cold_of[b] = c;

	return true;
}

export function sync_tables(M) {
	for (let t in [ 'pt1', 'pt3', 'pt2' ])
		M[t].copy_(torch.from_array(M.pt_host[t], 'int64', 'cuda:' + M.dev));
}

/* The cold complement of `experts_pick`, pinned on the host, local index 0 .. n-1. */
export function experts_host_pick(ws, key, ids) {
	let gq = {host: true, e0: 0};

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		let k = `${key}.exps.${m}`;
		let raw = ws.tensor(k), parts = [];

		for (let i in ids)
			push(parts, raw.narrow(0, i, 1));

		gq[m] = torch.cat(parts, 0);
		gq['t' + m] = ws.qtype(k);
	}

	gq.n_experts = length(ids);
	gq.inter = gq.w1.shape()[1];

	return gq;
}

/* Routed + shared experts of one layer for y (T, hidden) fp16 -> (T, hidden) fp32. H: the glm5 module's {route, act,
 * mlp} (passed in: glm5.uc cannot import this module). */
export function moe_forward(cfg, M, y, H) {
	let gq = M.gq;
	let T = y.shape()[0], Kt = cfg.num_experts_per_tok, Hd = cfg.hidden_size;
	let r = H.route(cfg, M, y);

	if (M.prof)
		M.prof(r.idx);                 // the picks of this step, for the routing profile (qwen4_prof.uc)
	let out, sh = null;

	function shared() {
		let s = H.mlp(cfg, M.shared, y).to('float32');

		if (M.shared_gate) {
			let sg = torch.matmul(y.to('float32'), M.shared_gate);
			let sh = sg.shape();

			if (length(sh) == 1)
				sg = sg.view([sh[0], 1]);

			s = s.mul(sg.sigmoid());
		}

		return s;
	}

	// decode: one grouped gemv per matrix over the device pointer table, so resident and cold experts share a
	// single launch and no host-side split or staging copy.
	if (T <= 8 && M.pt1) {
		let S = T * Kt;
		let xt = M.slot_rows.narrow(0, 0, S);
		let xg = y.index(xt).contiguous();
		let ids = r.idx.reshape([-1]).contiguous();

		let h;

		if (!FUSED_OFF && gq.tw1 == gq.tw3 && I8_TYPES[gq.tw1]) {
			// gate and up in one pointer-table launch (the same fused kernel as the non-pointer i8 path)
			h = torch.gguf_moe_gate_up_pt(xg, ids, M.pt1, M.pt3, gq.tw1, 1, gq.inter, cfg.swiglu_limit + 0.0);
		}
		else {
			let g = torch.gguf_moe_gemv_pt(xg, ids, M.pt1, gq.tw1, 1, gq.inter).to('float16');
			let u = torch.gguf_moe_gemv_pt(xg, ids, M.pt3, gq.tw3, 1, gq.inter).to('float16');

			h = H.act(cfg, g, u);
		}

		let d = torch.gguf_moe_gemv_pt(h, ids, M.pt2, gq.tw2, 1, Hd);

		let wflat = r.w.reshape([-1]);
		let out = d.mul(wflat.view([S, 1])).view([T, Kt, Hd]).sum(1).to('float32');

		return out.add(shared());
	}

	// prefill: one tensor-core launch per stage over the pointer tables (resident and cold experts together, any type), the
	// routing-weighted outputs summed per token in a fixed order
	if (T > 8 && M.pt1 && !PF_OFF && !PT_OFF && gq.tw1 == gq.tw3) {
		let E = M.n_experts_total;
		let pr = torch.moe_prep(r.idx.contiguous(), r.w.contiguous(), E, 0);
		let scratch = torch.gguf_moe_prefill_pt(y.contiguous(), pr[2], pr[3], pr[4], pr[5], M.pt1, M.pt3, M.pt2, gq.tw1, gq.tw2,
			gq.inter, cfg.swiglu_limit + 0.0);
		let sum = torch.zeros([T, Hd], 'float32', 'cuda:' + M.dev);

		torch.moe_gather(sum, scratch, pr[0], pr[6], pr[5].narrow(0, 0, E), pr[5].narrow(0, 0, E), pr[7].narrow(0, 0, E), pr[3]);

		return sum.add(shared());
	}

	if (M.gq_cold) {
		// expert offload: the resident stack holds experts [0, R); the cold stack is on the host, local ids [0, C).
		// A prefill chunk sorts the assignments by expert; each chosen cold expert's three rows are transferred once
		// and its matrix is reused for every token that picked it (not one transfer per token).
		if (T > 8) {
			let E = M.n_experts_total;
			let pr = torch.moe_prep(r.idx.contiguous(), r.w.contiguous(), E, 0);
			let tok = pr[2], wts = pr[3].to('float32');
			let count = pr[4].tolist(), start = pr[5].tolist();
			let out = torch.zeros([T, Hd], 'float32', 'cuda:' + M.dev);

			for (let e = 0; e < E; e++) {
				let n = count[e];

				if (n < 1)
					continue;

				let rs = M.res_of[e];
				let src = (rs >= 0) ? gq : M.gq_cold;
				let local = (rs >= 0) ? rs : M.cold_of[e];
				let rows = tok.narrow(0, start[e], n);
				let ye = y.index(rows);

				let w1 = src.w1.narrow(0, local, 1);
				let w3 = src.w3.narrow(0, local, 1);
				let w2 = src.w2.narrow(0, local, 1);

				let w1f = torch.gguf_dequant(w1.view([gq.inter, -1]), src.tw1, 'float16');
				let w3f = torch.gguf_dequant(w3.view([gq.inter, -1]), src.tw3, 'float16');
				let h = H.act(cfg, torch.matmul(ye, w1f.t()), torch.matmul(ye, w3f.t()));
				let w2f = torch.gguf_dequant(w2.view([Hd, -1]), src.tw2, 'float16');
				let d = torch.matmul(h, w2f.t()).to('float32');

				out.index_add_(0, rows, d.mul(wts.narrow(0, start[e], n).view([n, 1])));
			}

			return out.add(shared());
		}

		// decode / verify: split the picks, transfer only the cold rows actually chosen, and combine the slot outputs.
		let ids_all = r.idx.reshape([-1]).tolist();
		let S = T * Kt;
		let res_slots = [], res_ids = [], cold_slots = [], cold_ids = [];

		for (let s = 0; s < S; s++) {
			let e = ids_all[s];

			if (e < 0)
				continue;

			let rs = M.res_of[e];

			if (rs >= 0) {
				push(res_slots, s);
				push(res_ids, rs);
			}
			else {
				push(cold_slots, s);
				push(cold_ids, M.cold_of[e]);
			}
		}

		let out = torch.zeros([S, Hd], 'float32', 'cuda:' + M.dev);
		let wflat = r.w.reshape([-1]);

		let add_group = function (gqg, slots, ids, cold) {
			if (!length(slots))
				return;

			let st = torch.from_array(slots, 'int64').cuda(M.dev, true);
			let ti = [];

			for (let s in slots)
				push(ti, int(s / Kt));

			let xt = torch.from_array(ti, 'int64').cuda(M.dev, true);
			let xg = y.index(xt).contiguous();
			// cold: `gqg` is already a CUDA tensor backed by mapped pinned host memory; the grouped kernel reads
			// the selected expert rows over PCIe directly (no per-token staging copy).
			let wg = gqg;
			let idg = torch.from_array(ids, 'int64').cuda(M.dev, true);

			let g = torch.gguf_moe_gemv(xg, idg, wg.w1, wg.tw1, 1, null, 0).to('float16');
			let u = torch.gguf_moe_gemv(xg, idg, wg.w3, wg.tw3, 1, null, 0).to('float16');
			let h = H.act(cfg, g, u);
			let d = torch.gguf_moe_gemv(h, idg, wg.w2, wg.tw2, 1, null, 0);

			let wslt = wflat.view([S, 1]).index(st);
			d = d.mul(wslt);
			out.index_copy_(0, st, d);
		};

		add_group(gq, res_slots, res_ids, false);
		add_group(M.gq_cold, cold_slots, cold_ids, true);

		out = out.view([T, Kt, Hd]).sum(1);

		return out.add(shared());
	}

	if (T <= 8) {
		// decode / verify: every (token, pick) slot is one expert; the kernels dequantize in registers. Gate and up are one
		// launch with the clamped SwiGLU in registers, the routing weight is applied by the down projection
		let ids = r.idx.reshape([-1]).contiguous();
		let yc = y.contiguous();
		let h;

		// 2 .. 8 rows (a verify step): the picks grouped by expert, so an expert's weights are read once for every row that chose it
		let grp = (T >= 2 && !GROUP_OFF && !FUSED_OFF && I8_TYPES[gq.tw1] && I8_TYPES[gq.tw2]) ? torch.gguf_moe_group(ids) : null;

		// the shared expert on a forked stream while the routed ones run (graph capture only: its kernels are short and read the DRAM the
		// routed ones leave idle); joined before the sum
		sh = (!SHARED_PAR_OFF && H.side) ? H.side(M.dev, 1, () => shared()) : null;

		if (gq.tw1 == gq.tw3 && !FUSED_OFF && I8_TYPES[gq.tw1]) {
			h = torch.gguf_moe_gate_up(yc, ids, gq.w1, gq.w3, gq.tw1, Kt, cfg.swiglu_limit + 0.0, gq.e0, grp, T);
		}
		else {
			let g = torch.gguf_moe_gemv(yc, ids, gq.w1, gq.tw1, Kt, null, gq.e0).to('float16');
			let u = torch.gguf_moe_gemv(yc, ids, gq.w3, gq.tw3, Kt, null, gq.e0).to('float16');

			h = H.act(cfg, g, u);
		}

		let scale = I8_TYPES[gq.tw2] ? r.w.reshape([-1]).contiguous() : null;
		let d = torch.gguf_moe_gemv(h, ids, gq.w2, gq.tw2, 1, scale, gq.e0, grp, T);

		if (!I8_TYPES[gq.tw2])
			d = d.mul(r.w.reshape([-1]).view([T * Kt, 1]));

		out = d.view([T, Kt, Hd]).sum(1);
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

				return sum.add(shared());
			}

			return torch.gguf_moe_prefill(y.contiguous(), pr[2], pr[3], pr[4], pr[5], gq.w1, gq.w3, gq.w2, gq.tw1, gq.tw2,
				cfg.swiglu_limit + 0.0).add(shared());
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

	return out.add(shared());
}
