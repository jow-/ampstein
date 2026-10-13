// ucode/dflash.uc — DFlash 2 block-diffusion drafter for the Qwen3.5 / 3.8
// tensor-parallel model (docs/DFLASH.md).
//
// One draft round, in a single forward over a block of W + 1 rows (the
// anchor token + W mask tokens), proposes W tokens:
//   - the target's hidden states after the layers cfg.target_layer_ids
//     (rows of the last verify, or of the prompt) are fused by `fc` and
//     `hidden_norm` and serve as extra keys / values in every draft layer
//     (the context), cached per layer in a ring of positions;
//   - each layer is a Qwen3 block (non-causal sliding-window attention
//     over context + block) wrapped in two-tap grouped dynamic causal
//     convolutions before and after both the attention and the MLP;
//   - the final hidden of rows 1..W is scored by the TARGET's lm_head
//     (vocabulary-sharded here), the top-16 candidates per row are kept
//     and the candidate selector walks one path through them (unary
//     score + a low-rank predecessor/successor term).
// Drafts are the greedy walk, i.e. a deterministic one-hot proposal, so
// verification stays lossless for greedy and sampled requests alike.
//
// Rows of the ring: slot = position % CAP; posbuf[slot] is the absolute
// position stored there (-1: empty). A key is visible to a block row when
// |q - k| < sliding_window, it is not past the block's last row, and the
// slot is written. Context rows beyond the accepted prefix (the verify
// produced W rows, only a + 1 are real) land on slots the block rows
// overwrite in the same pass.
//
// Usage (engine):
//   let df = dflash.load(dir, tp, {window: W, max_seq: N});
//   df.build(feat_of)            // capture the round's CUDA graph
//   let ids = df.draft(seed, p_prev, start)   // W token ids
//   df.prefill_rows(feats, pos)  // context rows of a prefill chunk

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as qwen35_tp from './qwen35_tp.uc';
import * as math from 'math';
import { readfile } from 'fs';

/* o_proj / down_proj outputs are produced 1/OSCALE (their svh is scaled at
 * load), the residual add scales them back in fp32 */
const OSCALE = 16.0;
const CAP = 4096;       // ring slots: > sliding window + block + slack
const TRACE = getenv('AMP_DFLASH_TRACE') != null;

function tr(name, t) {
	if (TRACE)
		printf('dflash: %-14s absmax %.4g  mean|x| %.4g\n', name,
			t.to('float32').abs().max().item(), t.to('float32').abs().mean().item());
}

function rms(x, w, eps) {
	return torch.rms_norm(x.contiguous(), w, eps);
}

export function load(dir, tp, opts) {
	opts = opts || {};

	let cj = json(readfile(dir + '/config.json'));
	let dc = cj.dflash_config || {};
	let H = cj.hidden_size;
	let nh = cj.num_attention_heads, nkv = cj.num_key_value_heads, hd = cj.head_dim;
	let NL = cj.num_hidden_layers;
	let eps = cj.rms_norm_eps;
	let theta = cj.rope_parameters ? cj.rope_parameters.rope_theta : cj.rope_theta;
	let window = cj.sliding_window || 2048;
	let group = dc.conv_group_size || 16;
	let ksz = dc.conv_kernel_size || 2;
	let top_k = dc.selector_top_k || 16;
	let rank = dc.selector_rank || 256;
	let layer_ids = dc.target_layer_ids;
	let mask_id = dc.mask_token_id;
	let block = dc.block_size || 8;

	if (ksz != 2)
		die(`dflash: conv_kernel_size ${ksz} not supported (2 taps)`);

	if (cj.is_causal)
		die('dflash: causal drafter attention not supported');

	if (hd != 128 || nh % nkv)
		die(`dflash: unsupported attention shape (hd ${hd}, heads ${nh}/${nkv})`);

	let W = opts.window || (block - 1);

	if (W < 1 || W > block - 1)
		die(`dflash: window ${W} outside 1..${block - 1}`);

	let G = tp.G;
	let devs = tp.devs;
	let dev = devs[0];
	let dv = 'cuda:' + dev;
	let ws = weights.open(dir);
	let rep = nh / nkv;

	// tensor parallel like the target (qwen35_tp): attention by kv-head
	// groups (q follows), MLP by 256-column ranges, o / down / fc by rows
	// with an all-reduce of the partials; the fp32 residual stream, the
	// norms and the convolutions' kernel projections are replicated
	let wts = tp.plan.weights;
	let p_kv = qwen35_tp.split_n(nkv, 1, wts);
	let p_mlp = qwen35_tp.split_n(cj.intermediate_size, 256, wts);
	let p_fc = qwen35_tp.split_n(length(layer_ids) * H, 128, wts);

	for (let g = 0; g < G; g++)
		if (!p_kv[g][1] || !p_mlp[g][1] || !p_fc[g][1])
			die(`dflash: rank ${g} gets no share of the drafter (${G} ranks)`);

	let lin = (key) => models.linear(ws, key, true, dev);

	let L = [];

	for (let i = 0; i < NL; i++) {
		let k = `layers.${i}.`;
		let q = lin(k + 'self_attn.q_proj'), kp = lin(k + 'self_attn.k_proj');
		let vp = lin(k + 'self_attn.v_proj'), op = lin(k + 'self_attn.o_proj');
		let gp = lin(k + 'mlp.gate_proj'), up = lin(k + 'mlp.up_proj');
		let dp = lin(k + 'mlp.down_proj');
		let R = [];

		op.svh = op.svh.mul(1.0 / OSCALE);
		dp.svh = dp.svh.mul(1.0 / OSCALE);

		for (let g = 0; g < G; g++) {
			let d = devs[g], kv0 = p_kv[g][0], kvn = p_kv[g][1], mi = p_mlp[g];

			push(R, {
				kvh: kvn,
				in_ln: ws.gpu16(k + 'input_layernorm.weight', d),
				post_ln: ws.gpu16(k + 'post_attention_layernorm.weight', d),
				q_proj: models.cols(q, [ [ kv0 * rep * hd, kvn * rep * hd ] ], d),
				k_proj: models.cols(kp, [ [ kv0 * hd, kvn * hd ] ], d),
				v_proj: models.cols(vp, [ [ kv0 * hd, kvn * hd ] ], d),
				o_proj: models.rows(op, kv0 * rep * hd, kvn * rep * hd, d),
				q_norm: ws.gpu16(k + 'self_attn.q_norm.weight', d),
				k_norm: ws.gpu16(k + 'self_attn.k_norm.weight', d),
				gate_proj: models.cols(gp, [ mi ], d),
				up_proj: models.cols(up, [ mi ], d),
				down_proj: models.rows(dp, mi[0], mi[1], d),
				a_kp: models.linear(ws, k + 'attention_conv.kernel_projection', true, d),
				a_base: ws.gpu16(k + 'attention_conv.base_kernel', d).contiguous(),
				m_kp: models.linear(ws, k + 'mlp_conv.kernel_projection', true, d),
				m_base: ws.gpu16(k + 'mlp_conv.base_kernel', d).contiguous(),
				kc: torch.zeros([kvn, CAP, hd], 'float16', 'cuda:' + d),
				vc: torch.zeros([kvn, CAP, hd], 'float16', 'cuda:' + d)
			});
		}

		push(L, R);
		gc();
	}

	let fc_full = lin('fc');
	let fc = map(devs, (d, g) => models.rows(fc_full, p_fc[g][0], p_fc[g][1], d));
	let hidden_norm = map(devs, (d) => ws.gpu16('hidden_norm.weight', d));
	let norm = map(devs, (d) => ws.gpu16('norm.weight', d));
	let sel_hp = ws.gpu16('candidate_selector.hidden_projection.weight', dev).contiguous();
	let sel_pred = ws.gpu16('candidate_selector.predecessor_codebook', dev).contiguous();
	let sel_succ = ws.gpu16('candidate_selector.successor_codebook', dev).contiguous();

	fc_full = null;
	gc();

	// rotary: inv_freq[i] = theta ^ (-2i / hd); cos / sin built per call
	let half = hd / 2;
	let invf = [];

	for (let i = 0; i < half; i++)
		push(invf, math.pow(theta, -2.0 * i / hd));

	let freq = map(devs, (d) => torch.from_array(invf).cuda(d).view([1, half]));

	let n = W + 1;          // block rows (anchor + masks)
	let Cn = W;             // context rows per round (the verify rows)

	// persistent per-round inputs / state (graph-visible), one per rank;
	// the host writes rank 0's meta, the round copies it to the others
	// (in the graph, ordered with the replay)
	let meta = map(devs, (d) => torch.zeros([2 * Cn + 2 * n + 1], 'int64', 'cuda:' + d));
	let ids_buf = torch.full([n], mask_id, 'int64', dv);
	let posbuf = map(devs, (d) => torch.full([CAP], -1, 'int64', 'cuda:' + d));
	let out = null;
	let featf = null;       // function () -> 5 x [per rank (1, Cn, H)] of the last verify
	let graph = null;

	let df = {
		dev: dev, W: W, window: window, layer_ids: layer_ids, cap: CAP,
		end: 0           // ring is valid for the sequence up to this position
	};

	/* ring write of rows (slots) of one layer shard */
	function ring_write(l, slots, k_rows, v_rows) {
		// k_rows / v_rows: (R, kvh, hd) -> (kvh, R, hd)
		l.kc.index_copy_(1, slots, k_rows.transpose(0, 1));
		l.vc.index_copy_(1, slots, v_rows.transpose(0, 1));
	}

	function cos_sin(pos, g) {
		let ang = pos.to('float32').unsqueeze(1).mul(freq[g]);

		return [ ang.cos().contiguous(), ang.sin().contiguous() ];
	}

	/* fused context features of `feats` (5 x [per rank (1, R, H)]) ->
	 * per rank (R, H): each rank multiplies its input rows of fc */
	function fuse(feats, R) {
		let p = [];

		for (let g = 0; g < G; g++) {
			let x = torch.cat(map(feats, (f) => f[g]), -1).view([R, length(feats) * H])
				.narrow(1, p_fc[g][0], p_fc[g][1]).contiguous();

			push(p, fc[g].forward(x));
		}

		let s = tp.allreduce(p);

		return map(s, (x, g) => rms(x, hidden_norm[g], eps));
	}

	/* the draft block: ctx rows (positions ctx_pos / slots ctx_slot, fused
	 * features f, per rank) and block rows (blk_pos / blk_slot, emb). The
	 * per-rank arguments are arrays. Returns the final-norm hidden (n, H)
	 * per rank. */
	function forward(f, Cr, ctx_pos, ctx_slot, blk_pos, blk_slot, emb) {
		let cs_c = [], cs_b = [], addm = [], h = [];
		let scale = 1.0 / math.sqrt(hd);

		for (let g = 0; g < G; g++) {
			push(cs_c, cos_sin(ctx_pos[g], g));
			push(cs_b, cos_sin(blk_pos[g], g));

			// positions -> ring metadata, then the visibility mask (n, CAP)
			let pb = posbuf[g];

			pb.index_copy_(0, ctx_slot[g], ctx_pos[g]);
			pb.index_copy_(0, blk_slot[g], blk_pos[g]);

			let diff = blk_pos[g].unsqueeze(1).sub(pb.unsqueeze(0)).abs();
			let near = diff.le(window - 1).to('float32');
			let written = pb.neg().le(0).to('float32');
			let upto = pb.le(blk_pos[g].narrow(0, n - 1, 1)).to('float32');
			let vis = near.mul(written.unsqueeze(0)).mul(upto.unsqueeze(0));

			// additive, rows expanded over the query-head group (row i * rep + r)
			push(addm, vis.sub(1.0).mul(1e30).unsqueeze(1).expand([n, rep, CAP])
				.reshape([n * rep, CAP]).contiguous());
			push(h, ((g == 0) ? emb : emb.cuda(devs[g])).to('float32').contiguous());
		}

		tr('emb', emb);

		for (let li = 0; li < NL; li++) {
			let o = [], dyn = [];

			// ---- attention (each rank: its kv-head group)
			for (let g = 0; g < G; g++) {
				let l = L[li][g], kvh = l.kvh, qh = kvh * rep;
				let x = torch.rms_norm_f32(h[g], l.in_ln, eps);

				dyn[g] = l.a_kp.forward(x);

				let xc = torch.dflash_conv(x, dyn[g], l.a_base, 0, group);
				let q = l.q_proj.forward(xc).view([1, n, qh, hd]);
				let kn = l.k_proj.forward(xc).view([1, n, kvh, hd]);
				let vn = l.v_proj.forward(xc).view([n, kvh, hd]);
				let kcx = l.k_proj.forward(f[g]).view([1, Cr, kvh, hd]);
				let vcx = l.v_proj.forward(f[g]).view([Cr, kvh, hd]);

				q = torch.qk_norm_rope(q, l.q_norm, cs_b[g][0], cs_b[g][1], hd, eps);
				kn = torch.qk_norm_rope(kn, l.k_norm, cs_b[g][0], cs_b[g][1], hd, eps);
				kcx = torch.qk_norm_rope(kcx, l.k_norm, cs_c[g][0], cs_c[g][1], hd, eps);

				ring_write(l, ctx_slot[g], kcx.view([Cr, kvh, hd]), vcx);
				ring_write(l, blk_slot[g], kn.view([n, kvh, hd]), vn);

				// grouped-query attention over the ring: (kvh, rep * n, hd)
				let qg = q.view([n, kvh, rep, hd]).transpose(0, 1).contiguous()
					.view([kvh, n * rep, hd]);
				let sc = torch.matmul(qg, l.kc.transpose(1, 2)).to('float32')
					.mul(scale).add(addm[g]);
				let p = sc.softmax(-1).to('float16');
				let ao = torch.matmul(p, l.vc).view([kvh, n, rep, hd])
					.transpose(0, 1).contiguous().view([n, qh * hd]);

				if (g == 0) {
					tr(`L${li} q`, q);
					tr(`L${li} attn`, ao);
				}

				push(o, l.o_proj.forward(ao));
			}

			o = tp.allreduce(o);
			tr(`L${li} o`, o[0]);

			for (let g = 0; g < G; g++) {
				let l = L[li][g];

				torch.dflash_conv_res(h[g], o[g], dyn[g], l.a_base, 1, group, OSCALE);
			}

			// ---- MLP (each rank: its column range)
			let m = [];

			for (let g = 0; g < G; g++) {
				let l = L[li][g];
				let x = torch.rms_norm_f32(h[g], l.post_ln, eps);

				dyn[g] = l.m_kp.forward(x);

				let xc = torch.dflash_conv(x, dyn[g], l.m_base, 0, group);

				push(m, l.down_proj.forward(torch.silu_mul(l.gate_proj.forward(xc),
					l.up_proj.forward(xc))));
			}

			m = tp.allreduce(m);

			for (let g = 0; g < G; g++) {
				let l = L[li][g];

				torch.dflash_conv_res(h[g], m[g], dyn[g], l.m_base, 1, group, OSCALE);
			}

			tr(`L${li} h`, h[0]);
		}

		tr('final h', h[0]);

		return map(h, (x, g) => torch.rms_norm_f32(x, norm[g], eps));
	}

	/* the rest of the round after the hidden: sharded head on every rank's
	 * own copy of the hidden, top-k per shard, merge on rank 0, selector
	 * walk. hn: per rank (W, H); returns ids (W) on rank 0. */
	function propose(hn, anchor) {
		let vals = [], idxs = [];

		for (let g = 0; g < G; g++) {
			let lg = tp.lm_heads[g].forward(hn[g].view([1, W, H])).view([W, -1]);
			let t = lg.topk(top_k, -1);
			let iv = t.values.to('float32');
			let ii = t.indices.add(torch.full([1], tp.plan.vocab[g][0], 'int64', 'cuda:' + devs[g]));

			push(vals, (g == 0) ? iv : iv.cuda(dev));
			push(idxs, (g == 0) ? ii : ii.cuda(dev));
		}

		let v = (G == 1) ? vals[0] : torch.cat(vals, 1);
		let ix = (G == 1) ? idxs[0] : torch.cat(idxs, 1);
		let t2 = v.topk(top_k, -1);
		let unary = t2.values.contiguous();
		let cand = ix.gather(1, t2.indices).contiguous();
		let hp = torch.matmul(hn[0], sel_hp.t()).contiguous();

		return torch.dflash_select(unary, cand, hp, anchor, sel_pred, sel_succ);
	}

	function body() {
		for (let g = 1; g < G; g++)
			meta[g].copy_(meta[0]);

		let part = (o, len) => map(meta, (mt) => mt.narrow(0, o, len));
		let ctx_pos = part(0, Cn);
		let ctx_slot = part(Cn, Cn);
		let blk_pos = part(2 * Cn, n);
		let blk_slot = part(2 * Cn + n, n);
		let seed = meta[0].narrow(0, 2 * Cn + 2 * n, 1);

		ids_buf.narrow(0, 0, 1).copy_(seed);

		let emb = tp.embed_tokens.index(ids_buf).view([n, H]);
		let f = fuse(featf(), Cn);

		tr('ctx fused', f[0]);
		let hn = forward(f, Cn, ctx_pos, ctx_slot, blk_pos, blk_slot, emb);

		out = propose(map(hn, (x) => x.narrow(0, 1, W).contiguous()), seed);
	}

	function set_meta(seed, p_prev, start) {
		let v = [];

		for (let k = 0; k < Cn; k++)
			push(v, p_prev + k);

		for (let k = 0; k < Cn; k++)
			push(v, (p_prev + k) % CAP);

		for (let k = 0; k < n; k++)
			push(v, start + k);

		for (let k = 0; k < n; k++)
			push(v, (start + k) % CAP);

		push(v, seed);
		meta[0].copy_(torch.from_array(v, 'int64', dv));
	}

	/* Capture the round. feat_of() returns the five (1, Cn, H) feature
	 * tensors the verify graph writes (persistent buffers). Also runs one
	 * eager round first (allocations, autotune). */
	df.build = function (feat_of, use_graph) {
		featf = feat_of;

		// warm-up: positions far from anything real, then forget them
		set_meta(mask_id, 0, Cn);
		body();
		torch.synchronize();

		if (use_graph != false) {
			graph = torch.graph();
			graph.capture_begin(dev, slice(devs, 1));
			body();
			graph.capture_end();
			torch.synchronize();
		}

		df.reset();
	};

	df.reset = function () {
		for (let pb in posbuf)
			pb.fill_(-1);

		df.end = 0;
	};

	/* One round. seed: the anchor token (the sampled token, not yet in
	 * the target cache); start: its position; p_prev: position of the
	 * first of the last verify's rows, or null when there are none to
	 * add (they are then pointed at `start`, where the block overwrites
	 * them). Returns W token ids. */
	df.draft = function (seed, p_prev, start) {
		set_meta(seed, (p_prev == null) ? start : p_prev, start);

		if (graph)
			graph.replay();
		else
			body();

		let r = out.tolist();

		df.end = start;

		return r;
	};

	/* The same round without the host read: the W drafts are copied into
	 * `dst` (a (W) int64 tensor on the drafter's device — the engine's
	 * round buffer) and the call returns as soon as the work is queued. */
	df.draft_async = function (seed, p_prev, start, dst) {
		set_meta(seed, (p_prev == null) ? start : p_prev, start);

		if (graph)
			graph.replay();
		else
			body();

		dst.copy_(out);
		df.end = start;
	};

	/* Context rows of a prefill chunk (eager): feats = 5 x [per rank (1,
	 * R, H)] for positions pos .. pos + R - 1. Only the newest rows that
	 * fit the window are written. */
	df.prefill_rows = function (feats, pos) {
		let R = feats[0][0].shape()[1];
		let T = (R > window + 16) ? window + 16 : R;
		let off = R - T;

		if (off > 0)
			feats = map(feats, (x) => map(x, (y) => y.narrow(1, off, T).contiguous()));

		let p0 = pos + off;
		let vp = [], vs = [];

		for (let k = 0; k < T; k++) {
			push(vp, p0 + k);
			push(vs, (p0 + k) % CAP);
		}

		let f = fuse(feats, T);

		for (let g = 0; g < G; g++) {
			let d = 'cuda:' + devs[g];
			let ppos = torch.from_array(vp, 'int64', d);
			let slots = torch.from_array(vs, 'int64', d);
			let cs = cos_sin(ppos, g);

			for (let li = 0; li < NL; li++) {
				let l = L[li][g];
				let kx = l.k_proj.forward(f[g]).view([1, T, l.kvh, hd]);
				let vx = l.v_proj.forward(f[g]).view([T, l.kvh, hd]);

				kx = torch.qk_norm_rope(kx, l.k_norm, cs[0], cs[1], hd, eps);
				ring_write(l, slots, kx.view([T, l.kvh, hd]), vx);
			}

			posbuf[g].index_copy_(0, slots, ppos);
		}

		df.end = pos + R;
	};

	return df;
}
