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

	let lin = (key) => models.linear(ws, key, true, dev);

	let L = [];

	for (let i = 0; i < NL; i++) {
		let k = `layers.${i}.`;

		push(L, {
			in_ln: ws.gpu16(k + 'input_layernorm.weight', dev),
			post_ln: ws.gpu16(k + 'post_attention_layernorm.weight', dev),
			q_proj: lin(k + 'self_attn.q_proj'),
			k_proj: lin(k + 'self_attn.k_proj'),
			v_proj: lin(k + 'self_attn.v_proj'),
			o_proj: lin(k + 'self_attn.o_proj'),
			q_norm: ws.gpu16(k + 'self_attn.q_norm.weight', dev),
			k_norm: ws.gpu16(k + 'self_attn.k_norm.weight', dev),
			gate_proj: lin(k + 'mlp.gate_proj'),
			up_proj: lin(k + 'mlp.up_proj'),
			down_proj: lin(k + 'mlp.down_proj'),
			a_kp: lin(k + 'attention_conv.kernel_projection'),
			a_base: ws.gpu16(k + 'attention_conv.base_kernel', dev).contiguous(),
			m_kp: lin(k + 'mlp_conv.kernel_projection'),
			m_base: ws.gpu16(k + 'mlp_conv.base_kernel', dev).contiguous(),
			kc: torch.zeros([nkv, CAP, hd], 'float16', dv),
			vc: torch.zeros([nkv, CAP, hd], 'float16', dv)
		});
	}

	for (let l in L) {
		l.o_proj.svh = l.o_proj.svh.mul(1.0 / OSCALE);
		l.down_proj.svh = l.down_proj.svh.mul(1.0 / OSCALE);
	}

	let fc = lin('fc');
	let hidden_norm = ws.gpu16('hidden_norm.weight', dev);
	let norm = ws.gpu16('norm.weight', dev);
	let sel_hp = ws.gpu16('candidate_selector.hidden_projection.weight', dev).contiguous();
	let sel_pred = ws.gpu16('candidate_selector.predecessor_codebook', dev).contiguous();
	let sel_succ = ws.gpu16('candidate_selector.successor_codebook', dev).contiguous();

	// rotary: inv_freq[i] = theta ^ (-2i / hd); cos / sin built per call
	let half = hd / 2;
	let invf = [];

	for (let i = 0; i < half; i++)
		push(invf, math.pow(theta, -2.0 * i / hd));

	let freq = torch.from_array(invf).cuda(dev).view([1, half]);

	let n = W + 1;          // block rows (anchor + masks)
	let Cn = W;             // context rows per round (the verify rows)

	// persistent per-round inputs / state (graph-visible)
	let meta = torch.zeros([2 * Cn + 2 * n + 1], 'int64', dv);
	let ids_buf = torch.full([n], mask_id, 'int64', dv);
	let posbuf = torch.full([CAP], -1, 'int64', dv);
	let out = null;
	let featf = null;       // function () -> 5 tensors (1, Cn, H) of the last verify
	let graph = null;

	let df = {
		dev: dev, W: W, window: window, layer_ids: layer_ids, cap: CAP,
		end: 0           // ring is valid for the sequence up to this position
	};

	/* ring + posbuf write of rows (positions, slots) of one layer */
	function ring_write(l, slots, k_rows, v_rows) {
		// k_rows / v_rows: (R, nkv, hd) -> (nkv, R, hd)
		l.kc.index_copy_(1, slots, k_rows.transpose(0, 1));
		l.vc.index_copy_(1, slots, v_rows.transpose(0, 1));
	}

	function cos_sin(pos) {
		let ang = pos.to('float32').unsqueeze(1).mul(freq);

		return [ ang.cos().contiguous(), ang.sin().contiguous() ];
	}

	/* fused context features of `feats` (5 x (1, R, H)) -> (R, H) */
	function fuse(feats, R) {
		let x = torch.cat(feats, -1).view([R, length(feats) * H]);

		return rms(fc.forward(x), hidden_norm, eps);
	}

	/* the draft block: ctx rows (positions ctx_pos / slots ctx_slot, fused
	 * features f) and block rows (blk_pos / blk_slot, ids). Returns the
	 * final-norm hidden (n, H). */
	function forward(f, Cr, ctx_pos, ctx_slot, blk_pos, blk_slot, emb) {
		let cs_c = cos_sin(ctx_pos), cs_b = cos_sin(blk_pos);

		// positions -> ring metadata, then the visibility mask (n, CAP)
		posbuf.index_copy_(0, ctx_slot, ctx_pos);
		posbuf.index_copy_(0, blk_slot, blk_pos);

		let diff = blk_pos.unsqueeze(1).sub(posbuf.unsqueeze(0)).abs();
		let near = diff.le(window - 1).to('float32');
		let written = posbuf.neg().le(0).to('float32');
		let upto = posbuf.le(blk_pos.narrow(0, n - 1, 1)).to('float32');
		let vis = near.mul(written.unsqueeze(0)).mul(upto.unsqueeze(0));
		// additive, rows expanded over the query-head group (row i * rep + r)
		let rep = nh / nkv;
		let addm = vis.sub(1.0).mul(1e30).unsqueeze(1).expand([n, rep, CAP])
			.reshape([n * rep, CAP]).contiguous();
		let scale = 1.0 / math.sqrt(hd);

		let h = emb.to('float32').contiguous();

		tr('emb', emb);

		for (let li = 0; li < NL; li++) {
			let l = L[li];

			// ---- attention
			let x = torch.rms_norm_f32(h, l.in_ln, eps);
			let dyn = l.a_kp.forward(x);
			let xc = torch.dflash_conv(x, dyn, l.a_base, 0, group);

			let q = l.q_proj.forward(xc).view([1, n, nh, hd]);
			let kn = l.k_proj.forward(xc).view([1, n, nkv, hd]);
			let vn = l.v_proj.forward(xc).view([n, nkv, hd]);
			let kcx = l.k_proj.forward(f).view([1, Cr, nkv, hd]);
			let vcx = l.v_proj.forward(f).view([Cr, nkv, hd]);

			q = torch.qk_norm_rope(q, l.q_norm, cs_b[0], cs_b[1], hd, eps);
			kn = torch.qk_norm_rope(kn, l.k_norm, cs_b[0], cs_b[1], hd, eps);
			kcx = torch.qk_norm_rope(kcx, l.k_norm, cs_c[0], cs_c[1], hd, eps);

			ring_write(l, ctx_slot, kcx.view([Cr, nkv, hd]), vcx);
			ring_write(l, blk_slot, kn.view([n, nkv, hd]), vn);

			// grouped-query attention over the ring: (nkv, rep * n, hd)
			let qg = q.view([n, nkv, rep, hd]).transpose(0, 1).contiguous()
				.view([nkv, n * rep, hd]);
			let sc = torch.matmul(qg, l.kc.transpose(1, 2)).to('float32')
				.mul(scale).add(addm);
			let p = sc.softmax(-1).to('float16');
			let ao = torch.matmul(p, l.vc).view([nkv, n, rep, hd])
				.transpose(0, 1).contiguous().view([n, nh * hd]);

			let o = l.o_proj.forward(ao);

			tr(`L${li} q`, q);
			tr(`L${li} attn`, ao);
			tr(`L${li} o`, o);

			torch.dflash_conv_res(h, o, dyn, l.a_base, 1, group, OSCALE);

			// ---- MLP
			x = torch.rms_norm_f32(h, l.post_ln, eps);
			dyn = l.m_kp.forward(x);
			xc = torch.dflash_conv(x, dyn, l.m_base, 0, group);

			let m = l.down_proj.forward(torch.silu_mul(l.gate_proj.forward(xc),
				l.up_proj.forward(xc)));

			torch.dflash_conv_res(h, m, dyn, l.m_base, 1, group, OSCALE);

			tr(`L${li} h`, h);
		}

		tr('final h', h);

		return torch.rms_norm_f32(h, norm, eps);
	}

	/* the rest of the round after the hidden: sharded head, top-k per
	 * shard, merge, selector walk. hn (W, H) on dev; returns ids (W). */
	function propose(hn, anchor) {
		let vals = [], idxs = [];

		for (let g = 0; g < G; g++) {
			let hl = (g == 0) ? hn : hn.cuda(devs[g]);
			let lg = tp.lm_heads[g].forward(hl.view([1, W, H])).view([W, -1]);
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
		let hp = torch.matmul(hn, sel_hp.t()).contiguous();

		return torch.dflash_select(unary, cand, hp, anchor, sel_pred, sel_succ);
	}

	function body() {
		let ctx_pos = meta.narrow(0, 0, Cn);
		let ctx_slot = meta.narrow(0, Cn, Cn);
		let blk_pos = meta.narrow(0, 2 * Cn, n);
		let blk_slot = meta.narrow(0, 2 * Cn + n, n);
		let seed = meta.narrow(0, 2 * Cn + 2 * n, 1);

		ids_buf.narrow(0, 0, 1).copy_(seed);

		let emb = tp.embed_tokens.index(ids_buf).view([n, H]);
		let f = fuse(featf(), Cn);

		tr('ctx fused', f);
		let hn = forward(f, Cn, ctx_pos, ctx_slot, blk_pos, blk_slot, emb);

		out = propose(hn.narrow(0, 1, W).contiguous(), seed);
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
		meta.copy_(torch.from_array(v, 'int64', dv));
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
		posbuf.fill_(-1);
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

	/* Context rows of a prefill chunk (eager): feats = 5 x (1, R, H) for
	 * positions pos .. pos + R - 1. Only the newest rows that fit the
	 * window are written. */
	df.prefill_rows = function (feats, pos) {
		let R = feats[0].shape()[1];
		let T = (R > window + 16) ? window + 16 : R;
		let off = R - T;

		if (off > 0)
			feats = map(feats, (x) => x.narrow(1, off, T).contiguous());

		let p0 = pos + off;
		let vp = [], vs = [];

		for (let k = 0; k < T; k++) {
			push(vp, p0 + k);
			push(vs, (p0 + k) % CAP);
		}

		let ppos = torch.from_array(vp, 'int64', dv);
		let slots = torch.from_array(vs, 'int64', dv);
		let f = fuse(feats, T);
		let cs = cos_sin(ppos);

		for (let li = 0; li < NL; li++) {
			let l = L[li];
			let kx = l.k_proj.forward(f).view([1, T, nkv, hd]);
			let vx = l.v_proj.forward(f).view([T, nkv, hd]);

			kx = torch.qk_norm_rope(kx, l.k_norm, cs[0], cs[1], hd, eps);
			ring_write(l, slots, kx.view([T, nkv, hd]), vx);
		}

		posbuf.index_copy_(0, slots, ppos);
		df.end = pos + R;
	};

	return df;
}
