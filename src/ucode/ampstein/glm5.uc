// ucode/glm5.uc — GLM-5.3-Flash (Glm5NextForConditionalGeneration), text.
//
// See docs/GLM5.md for the architecture. The residual is a stack of 4 fp32
// streams (mHC hyper-connections); every sublayer site is
//
//     (post, comb, c) = hc_mix(streams)        c: fp16 collapsed input
//     y               = sublayer(norm(c))
//     streams         = post (x) y + comb^T streams        (hc_apply, in place)
//
// This file holds the config, the mHC wrappers, the (clamped) SwiGLU MLP
// and — as milestones land — the KDA / MLA / MoE layers and the model.

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as qwen35 from './qwen35.uc';
import * as expert_host from './expert_host.uc';
import {readfile, stat} from 'fs';

export const LP = 'model.language_model.layers.';

/* Intra-graph concurrency: while PAR is set (CUDA-graph capture of decode
 * steps) independent op chains run on a forked side stream per device. */
let PAR = false;

export function set_par(b) {
	PAR = b ? true : false;
}

function side(dev, k, fn) {
	if (!PAR)
		return fn();

	torch.stream_side(dev, k);

	let r = fn();

	torch.stream_main(dev, k);

	return r;
}

function join(dev, k) {
	if (PAR)
		torch.stream_wait(dev, k);
}

/* fp32 (T, K) @ (K, N): the skinny kernel for decode / verify rows (cuBLAS
 * turns 2..8 rows into a slow generic GEMM), cuBLAS above */
function mm32(x, w) {
	return (x.shape()[0] <= 8 && !getenv('AMP_GLM5_NO_SKINNY')) ? torch.skinny(x, w) : torch.matmul(x, w);
}

/* ------------------------------------------------------------------ */
/* config                                                              */

export function load_config(model_dir) {
	let raw = json(readfile(model_dir + '/config.json'));
	let t = raw.text_config || raw;

	let cfg = {
		model_dir: model_dir,
		hidden_size: t.hidden_size,
		num_hidden_layers: t.num_hidden_layers,
		vocab_size: t.vocab_size,
		rms_norm_eps: t.rms_norm_eps,
		layer_types: t.layer_types,
		mlp_layer_types: t.mlp_layer_types,
		indexer_types: t.indexer_types,

		// mHC
		hc_mult: t.hc_mult,
		hc_eps: t.hc_eps,
		hc_sinkhorn_iters: t.hc_sinkhorn_iters,

		// MLP / MoE
		intermediate_size: t.intermediate_size,
		moe_intermediate_size: t.moe_intermediate_size,
		n_routed_experts: t.n_routed_experts,
		n_shared_experts: t.n_shared_experts,
		num_experts_per_tok: t.num_experts_per_tok,
		routed_scaling_factor: t.routed_scaling_factor,
		swiglu_limit: t.swiglu_limit,

		// MLA
		num_q_heads: t.num_attention_heads,
		q_lora_rank: t.q_lora_rank,
		kv_lora_rank: t.kv_lora_rank,
		qk_nope_head_dim: t.qk_nope_head_dim,
		qk_rope_head_dim: t.qk_rope_head_dim,
		v_head_dim: t.v_head_dim,

		// indexer
		index_n_heads: t.index_n_heads,
		index_head_dim: t.index_head_dim,
		index_topk: t.index_topk,
		index_kpool: t.index_kpool,

		// KDA
		kda: t.linear_attn_config,

		eos_token_id: t.eos_token_id,

		// vision (the tower itself is glm5_vision.uc)
		image_token_id: raw.image_token_id,
		has_vision: raw.vision_config ? true : false
	};

	if (cfg.hc_mult != 4)
		die('glm5: only hc_mult == 4 is supported');

	if (cfg.qk_rope_head_dim != 0)
		die('glm5: expected NoPE attention (qk_rope_head_dim == 0)');

	for (let i = 0; i < length(cfg.layer_types); i++)
		if (cfg.layer_types[i] != 'linear_attention' &&
		    cfg.layer_types[i] != 'deepseek_sparse_attention')
			die(`glm5: unknown layer type '${cfg.layer_types[i]}'`);

	cfg.sm_scale = math.pow(cfg.qk_nope_head_dim * 1.0, -0.5);

	return cfg;
}

/* Image spans of a prompt for the embedding overlay. images: [{start,
 * emb ((n, hidden) fp16), ...}] in prompt order; the text model is NoPE,
 * so a span is just rows to overwrite (qwen35.embed_rows does the
 * overlay: span fields start / n / emb). Returns null for no images. */
export function mm_build(cfg, images) {
	if (!images || !length(images))
		return null;

	let spans = [], prev_end = 0;

	for (let im in images) {
		let n = im.emb ? im.emb.shape()[0] : im.n;

		if (im.start < prev_end)
			die('glm5: overlapping or unordered image spans');

		push(spans, {start: im.start, n: n, emb: im.emb});
		prev_end = im.start + n;
	}

	return {spans: spans};
}

/* ------------------------------------------------------------------ */
/* mHC hyper-connections                                               */

/* The raw fp32 tensors of one site: tag is 'attn' or 'ffn'. */
export function build_hc(ws, i, tag, dev) {
	let k = `${LP}${i}.hc_${tag}`;

	return {
		fn: ws.gpu(k + '_fn', dev).to('float32').contiguous(),
		base: ws.gpu(k + '_base', dev).to('float32').contiguous(),
		scale: ws.gpu(k + '_scale', dev).to('float32').contiguous()
	};
}

/* streams: (R, 4, D) fp32 contiguous. Returns [post, comb, collapsed
 * (R, D) fp16]. */
export function hc_mix(cfg, hc, streams) {
	return torch.hc_mix(streams, hc.fn, hc.base, hc.scale,
		cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters);
}

/* In-place residual update of the streams with the sublayer output y
 * (R, D) fp16/fp32: streams <- post (x) y + comb^T streams. */
export function hc_apply(streams, y, post, comb) {
	return torch.hc_apply(streams, y, post, comb);
}

/* ------------------------------------------------------------------ */
/* MLP                                                                 */

/* Clamped SwiGLU (swiglu_limit): silu(min(g, L)) * clamp(u, -L, L). */
export function act_mul(cfg, g, u) {
	if (cfg.act == 'gelu')                      // Gemma 4: gelu (tanh form) gated
		return g.gelu_tanh().mul(u);

	let lim = cfg.swiglu_limit;

	if (lim && lim != 0)
		return torch.silu_mul(g.clamp(null, lim), u.clamp(-lim, lim));

	return torch.silu_mul(g, u);
}

/* Gated MLP with separate gate/up/down EXL3 linears. */
export function build_mlp(ws, key, dev) {
	return {
		gate: models.linear(ws, key + '.gate_proj', false, dev),
		up: models.linear(ws, key + '.up_proj', false, dev),
		down: models.linear(ws, key + '.down_proj', false, dev)
	};
}

/* x: (R, hidden) fp16. Returns (R, hidden) fp16. */
export function mlp_forward(cfg, M, x) {
	let g, u;

	if (M.gu && x.shape()[0] == 1) {
		// gate and up as ONE batched launch (DeepSeek-V4 decode, models.multi)
		M.gu.run(x.view([1, 1, x.shape()[1]]));
		g = M.gu.out.narrow(0, 0, 1).view([1, -1]);
		u = M.gu.out.narrow(0, 1, 1).view([1, -1]);
	}
	else {
		g = M.gate.forward(x);
		u = M.up.forward(x);
	}

	return M.down.forward(act_mul(cfg, g, u));
}

/* ------------------------------------------------------------------ */
/* KDA (Kimi delta attention) layers                                   */

/* fp32 copy of a small projection weight, transposed to (in, out) so
 * x_fp32 @ W matmuls run in fp32 (the gates feed the decay chain, where
 * fp16 rounding compounds). */
function small_t(ws, key, dev) {
	return ws.gpu(key + '.weight', dev).to('float32').t().contiguous();
}

export function build_kda(ws, cfg, i, dev) {
	let key = `${LP}${i}.self_attn`;
	let la = cfg.kda;
	let H = la.num_heads, hd = la.head_dim;

	return {
		type: 'kda', dev: dev, heads: H, head_dim: hd,
		kernel: la.short_conv_kernel_size,
		lower_bound: la.gate_lower_bound,
		qkv: models.linear(ws, key + '.qkv_proj', false, dev),
		o: models.linear(ws, key + '.o_proj', false, dev),
		b_t: small_t(ws, key + '.b_proj', dev),
		fa_t: small_t(ws, key + '.f_a_proj', dev),
		fb_t: small_t(ws, key + '.f_b_proj', dev),
		ga_t: small_t(ws, key + '.g_a_proj', dev),
		gb_t: small_t(ws, key + '.g_b_proj', dev),
		conv_w: ws.gpu16(key + '.conv1d.weight', dev).squeeze(1).contiguous(),
		decay: ws.gpu(key + '.A_log', dev).to('float32').reshape([-1]).exp().contiguous(),
		dt_bias: ws.gpu(key + '.dt_bias', dev).to('float32').reshape([-1]).contiguous(),
		o_norm: ws.gpu(key + '.o_norm.weight', dev).to('float32').contiguous()
	};
}

/* Recurrent state of one KDA layer: conv taps (1, D, K-1) fp16 and the
 * per-head delta-rule state (H, dk, dv) fp32. */
export function new_kda_state(cfg, K, dev) {
	let la = cfg.kda;
	let D = 3 * la.num_heads * la.head_dim;
	let dv = 'cuda:' + dev;

	return {
		conv_state: torch.zeros([1, D, la.short_conv_kernel_size - 1], 'float16', dv),
		rec_state: torch.zeros([la.num_heads, la.head_dim, la.head_dim], 'float32', dv),
		dev: dev
	};
}

/* x: (T, hidden) fp16 (the normed, collapsed site input); cl: the layer's
 * recurrent state. Returns (T, hidden) fp16. */
export function kda_forward(cfg, K, x, cl) {
	let T = x.shape()[0];
	let H = K.heads, hd = K.head_dim;
	let D = 3 * H * hd;

	// gate / decay / output-gate projections (fp32): independent of the
	// qkv product, so they run beside it
	let gg = side(K.dev, 1, function () {
		let xf = x.to('float32');
		let f = mm32(mm32(xf, K.fa_t), K.fb_t);                      // (T, H*hd)
		let braw = mm32(xf, K.b_t);                                  // (T, H)
		let gb = torch.kda_gates(f, K.dt_bias, K.decay, braw, K.lower_bound, 1.0);
		let z = mm32(mm32(xf, K.ga_t), K.gb_t);                      // (T, H*hd)

		return {gb: gb, z: z};
	});

	let qkv = K.qkv.forward(x);                                  // (T, D) fp16

	// speculative verify (the engine set cl.spec_w = W and this is the
	// W-token verify step): keep the state after each of the first W-1
	// tokens — the conv window per candidate here, before gdn_prep
	// advances it, the recurrent state inside the scan — so a rejection
	// after token a restores them with one batched copy per GPU
	let spec = cl.spec_w && T == cl.spec_w;

	if (spec) {
		let km1 = cl.conv_state.shape()[2];
		let win = torch.cat([ cl.conv_state, qkv.view([1, T, D]).transpose(1, 2) ], 2);

		for (let t = 0; t < T - 1; t++)
			cl.spec_conv[t].copy_(win.narrow(2, t + 1, km1));
	}

	let p = torch.gdn_prep(qkv.view([1, T, D]), cl.conv_state, K.conv_w, H, H, hd);

	join(K.dev, 1);

	let core = spec ?
		torch.kda_scan(p[0], p[1], p[2], gg.gb[1], gg.gb[0], cl.rec_state,
			math.pow(hd * 1.0, -0.5), cl.spec_states) :
		torch.kda_scan(p[0], p[1], p[2], gg.gb[1], gg.gb[0], cl.rec_state,
			math.pow(hd * 1.0, -0.5));
	let o = torch.kda_out_norm(core, gg.z, K.o_norm, cfg.rms_norm_eps);

	return K.o.forward(o.view([T, H * hd]));
}

/* ------------------------------------------------------------------ */
/* NoPE MLA (the "deepseek_sparse_attention" layers), dense form        */
/*                                                                     */
/* Absorbed latent attention: the cache holds one 512-wide latent per   */
/* token (kv_a_layernorm(kv_a(x))); all 64 query heads attend to it as  */
/* one shared key/value stream:                                         */
/*   q_lat[h] = q_nope[h] W_UK[h]         (256 -> 512)                  */
/*   s        = softmax(q_lat . ckv^T * 256^-0.5)   (causal)            */
/*   o[h]     = (s . ckv) W_UV[h]^T       (512 -> 256)                  */
/* Exact while the context is within index_topk (the indexer then       */
/* selects everything); the k-pool selection for longer contexts is a   */
/* later milestone.                                                     */

/* kpool_aux.safetensors (indexer pool parameters; not in the weight map). */
let aux_cache = {dir: null, st: null};

function kpool_aux(cfg) {
	if (aux_cache.dir != cfg.model_dir) {
		aux_cache.dir = cfg.model_dir;
		aux_cache.st = torch.st(cfg.model_dir + '/kpool_aux.safetensors');
	}

	return aux_cache.st;
}

export function build_mla(ws, cfg, i, dev) {
	let key = `${LP}${i}.self_attn`;
	let H = cfg.num_q_heads;
	let nope = cfg.qk_nope_head_dim, vd = cfg.v_head_dim, dc = cfg.kv_lora_rank;

	// kv_b_proj: (H * (nope + v), dc) -> per head [W_UK (nope, dc); W_UV (v, dc)]
	let w = ws.gpu16(key + '.kv_b_proj.weight', dev).view([H, nope + vd, dc]);

	let ik = key + '.indexer';
	let akey = `${LP}${i}.self_attn.indexer`;

	// the pool parameters of the trunk layers live in kpool_aux.safetensors;
	// the MTP layer carries its own next to its other indexer tensors
	let aux = ws.has(akey + '.index_kpool_compress_gate') ? {
		tensor: (k) => ws.tensor(k)
	} : kpool_aux(cfg);

	return {
		type: 'mla', dev: dev,
		// k-pool DSA indexer (queries from the q_a latent, keys from x)
		idx_wq_b: models.linear(ws, ik + '.wq_b', false, dev),
		idx_wk_t: ws.gpu16(ik + '.wk.weight', dev).t().contiguous(),            // (hid, 128)
		idx_w_t: ws.gpu16(ik + '.weights_proj.weight', dev).t().contiguous(),   // (hid, 32)
		idx_norm_w: ws.gpu16(ik + '.k_norm.weight', dev).to('float32'),
		idx_norm_b: ws.gpu16(ik + '.k_norm.bias', dev).to('float32'),
		idx_gate_t: aux.tensor(akey + '.index_kpool_compress_gate').to('float16')
			.cuda(dev).t().contiguous(),                                     // (hid, 128)
		idx_ape: aux.tensor(akey + '.index_kpool_compress_ape').to('float32')
			.cuda(dev).contiguous(),                                         // (P, 128)
		q_a: models.linear(ws, key + '.q_a_proj', false, dev),
		q_a_norm: ws.gpu16(key + '.q_a_layernorm.weight', dev),
		q_b: models.linear(ws, key + '.q_b_proj', false, dev),
		kv_a: models.linear(ws, key + '.kv_a_proj_with_mqa', false, dev),
		kv_a_norm: ws.gpu16(key + '.kv_a_layernorm.weight', dev),
		o: models.linear(ws, key + '.o_proj', false, dev),
		w_uk: w.narrow(1, 0, nope).contiguous(),                    // (H, nope, dc)
		w_uv_t: w.narrow(1, nope, vd).transpose(1, 2).contiguous()  // (H, dc, v)
	};
}

/* Latent cache of one MLA layer: (max_seq, kv_lora_rank) fp16, or with
 * kv_bits = 8 the Q8 form (int8 codes `qk` + fp16 group scales `qs`, 32
 * values per scale; 544 instead of 1024 bytes per token). The indexer
 * plane and pools stay fp16 (they feed every selection, and are small). */
export function new_mla_cache(cfg, max_seq, dev, kv_bits) {
	let P = cfg.index_kpool;
	let dv = 'cuda:' + dev;
	let dc = cfg.kv_lora_rank;
	let cl = {
		// per-token indexer plane [key || pool gate] and the pooled keys
		plane: torch.zeros([max_seq, 2 * cfg.index_head_dim], 'float16', dv),
		pools: torch.zeros([int(max_seq / P) + 1, cfg.index_head_dim], 'float16', dv),
		dev: dev
	};

	if (kv_bits == 8) {
		cl.qk = torch.zeros([max_seq, dc], 'int8', dv);
		cl.qs = torch.zeros([max_seq, dc / 32], 'float16', dv);
	}
	else if (kv_bits) {
		die(`new_mla_cache: unsupported kv_bits ${kv_bits} (8 or none)`);
	}
	else {
		cl.ckv = torch.zeros([max_seq, dc], 'float16', dv);
	}

	return cl;
}

/* Latent cache access (fp16 or Q8): write the rows of ckv at positions pos
 * (an integer start or an int64 index tensor), read n rows from start, or
 * gather rows idx (1-D int64) — always as fp16 (n, dc). */
function ckv_put(cl, pos, ckv) {
	if (cl.qk)
		torch.kv_q8_put(ckv, cl.qk, cl.qs, pos);
	else if (type(pos) == 'int')
		cl.ckv.narrow(0, pos, ckv.shape()[0]).copy_(ckv);
	else
		cl.ckv.index_copy_(0, pos, ckv);
}

function ckv_rows(cl, start, n) {
	return cl.qk ? torch.kv_q8_get(cl.qk, cl.qs, start, n) : cl.ckv.narrow(0, start, n);
}

function ckv_gather(cl, idx) {
	return cl.qk ? torch.kv_q8_get(cl.qk, cl.qs, idx) : cl.ckv.index(idx);
}

/* Additive causal mask (T, pos + T) fp16 for a chunk at position pos. */
function causal_add_mask(T, pos, dev) {
	let len = pos + T;
	let cols = torch.arange(len, 'int64', 'cuda:' + dev);
	let rows = cols.narrow(0, pos, T).view([T, 1]);
	let ok = cols.view([1, len]).le(rows);

	return torch.where(ok, torch.zeros([1], 'float16', 'cuda:' + dev),
		torch.full([1], -65504.0, 'float16', 'cuda:' + dev));
}

/* Indexer keys of the chunk -> the per-token plane [k || gate], and the
 * pooled keys of the pools this chunk completes. A pool key is the
 * softmax(gate + APE)-weighted mean of its P members' keys (per channel),
 * read back from the plane so pools straddling chunk boundaries resolve. */
function mla_index_update(cfg, A, x, cl, pos) {
	let T = x.shape()[0];
	let P = cfg.index_kpool, D = cfg.index_head_dim;

	let k = torch.layer_norm(torch.matmul(x, A.idx_wk_t).to('float32'),
		A.idx_norm_w, A.idx_norm_b, 1e-6).to('float16');
	let gate = torch.matmul(x, A.idx_gate_t);

	cl.plane.narrow(0, pos, T).copy_(torch.cat([k, gate], 1));

	let first = int(pos / P), last = int((pos + T) / P);

	if (last > first) {
		let n = last - first;
		let mem = cl.plane.narrow(0, first * P, n * P).view([n, P, 2 * D])
			.to('float32');
		let probs = mem.narrow(2, D, D).add(A.idx_ape.unsqueeze(0)).softmax(1);

		cl.pools.narrow(0, first, n).copy_(
			probs.mul(mem.narrow(2, 0, D)).sum(1).to('float16'));
	}
}

/* Sparse (DSA) attention for context > index_topk: per query the indexer
 * scores the pooled keys, selects the best index_topk/P pools, expands them
 * to raw token indices (+ the incomplete tail pool, raw), and attention runs
 * over the gathered latent rows only. q_lat: (H, T, dc); qr: the normed q_a
 * latent (T, q_lora). Returns o_lat (H, T, dc). Processed in query slabs to
 * bound the (rows x pools) score matrices. */
function mla_sparse(cfg, A, x, qr, q_lat, cl, pos) {
	let T = x.shape()[0];
	let H = cfg.num_q_heads, dc = cfg.kv_lora_rank;
	let P = cfg.index_kpool, Hi = cfg.index_n_heads, Di = cfg.index_head_dim;
	let len = pos + T;
	let n_pools = int(len / P);
	let kp = (int(cfg.index_topk / P) < n_pools) ? int(cfg.index_topk / P) : n_pools;
	let dv = 'cuda:' + cl.dev;
	const ASLAB = 64;                    // rows per gathered-attention sub-slab

	let q_idx = A.idx_wq_b.forward(qr).view([T, Hi, Di]);
	let w = torch.matmul(x, A.idx_w_t).mul(math.pow(Hi * 1.0, -0.5));    // (T, Hi)
	let pk = cl.pools.narrow(0, 0, n_pools);
	// the fused scoring kernel (32 heads x 128 dims); AMP_GLM5_NO_IDX_KERNEL=1: composite ATen path
	let fast = (Hi == 32 && Di == 128 && !getenv('AMP_GLM5_NO_IDX_KERNEL'));
	let wf = fast ? w.to('float32').contiguous() : null;
	let pend = torch.arange(n_pools, 'float32', dv).mul(P * 1.0).add(P - 1.0);
	let offs = torch.arange(P, 'float32', dv).view([1, 1, P]);
	let neg = torch.full([1], -65504.0, 'float16', dv);

	// Scoring / top-k / index slabs. With the fused kernel the score matrix
	// is only (R, n_pools), so big slabs keep the launch count small — the
	// single host thread that drives all GPUs is the bottleneck of a
	// pipelined prefill (232k launches in 9.5 s, 85% of them this loop at
	// 32-row slabs); the composite path needs small ones.
	let SLAB = fast ? 256 : 32;
	let m1 = torch.full([1], -1, 'int64', dv);
	let z1 = torch.zeros([1], 'int64', dv);
	let zh = torch.zeros([1], 'float16', dv);
	let arT = torch.arange(P - 1, 'float32', dv).view([1, P - 1]);
	let outs = [];

	for (let r0 = 0; r0 < T; r0 += SLAB) {
		let R = (T - r0 < SLAB) ? T - r0 : SLAB;

		// pooled scores: sum_h w_h * relu(q_h . pool * Di^-0.5); a pool is
		// selectable once all of its tokens are visible
		let sc;

		if (fast) {
			sc = torch.idx_score(q_idx.narrow(0, r0, R).contiguous(), pk,
				wf.narrow(0, r0, R).contiguous(), math.pow(Di * 1.0, -0.5), pos + r0, P);
		}
		else {
			let s = torch.matmul(q_idx.narrow(0, r0, R).reshape([R * Hi, Di]), pk.t())
				.view([R, Hi, n_pools]).mul(math.pow(Di * 1.0, -0.5)).clamp(0, null);

			sc = torch.matmul(w.narrow(0, r0, R).view([R, 1, Hi]), s)
				.view([R, n_pools]);

			let qpos = torch.arange(R, 'float32', dv).add(pos + r0 * 1.0).view([R, 1]);
			let ok = pend.view([1, n_pools]).le(qpos);

			sc = torch.where(ok, sc, neg);
		}

		let tk = sc.topk(kp, -1);
		let dead = tk.values.le(-60000.0).view([R, kp, 1]);       // masked-out picks

		// pools -> raw token indices, -1 where the pick was masked out
		let exp = tk.indices.to('float32').view([R, kp, 1]).mul(P * 1.0).add(offs)
			.to('int64');

		exp = torch.where(dead.expand([R, kp, P]), m1, exp).view([R, kp * P]);

		// the incomplete tail pool of each query, raw: the tokens
		// [floor((qpos + 1) / P) * P, qpos] (+ 1.5: robust to rounding)
		let qp = torch.arange(R, 'float32', dv).add(pos + r0 * 1.0).view([R, 1]);
		let tf = qp.add(1.5).mul(1.0 / P).to('int64').to('float32').mul(P * 1.0).add(arT);
		let tail = torch.where(tf.le(qp), tf.to('int64'), m1);

		let idx = torch.cat([exp, tail], 1);
		let K = idx.shape()[1];

		if (cl.dbg)
			push(cl.dbg, idx);               // tests: the selected indices

		let pad = idx.le(-1);
		let safe = torch.where(pad, z1, idx);
		let padneg = torch.where(pad, neg, zh);                              // (R, K) additive mask

		// gathered attention over the selected latent rows, in sub-slabs
		for (let a0 = 0; a0 < R; a0 += ASLAB) {
			let Ra = (R - a0 < ASLAB) ? R - a0 : ASLAB;
			let g = ckv_gather(cl, safe.narrow(0, a0, Ra).reshape([-1])).view([Ra, K, dc]);
			let ql = q_lat.narrow(1, r0 + a0, Ra).transpose(0, 1).contiguous();   // (Ra, H, dc)
			let a = torch.matmul(ql, g.transpose(1, 2)).mul(cfg.sm_scale);         // (Ra, H, K)

			a = a.add(padneg.narrow(0, a0, Ra).view([Ra, 1, K]));
			a = a.softmax(-1);

			push(outs, torch.matmul(a, g).transpose(0, 1));                       // (H, Ra, dc)
		}
	}

	return (length(outs) == 1) ? outs[0] : torch.cat(outs, 1);
}

/* x: (T, hidden) fp16; cl: the layer's latent + indexer cache; pos: tokens
 * already cached. Appends the chunk and returns (T, hidden) fp16. Dense
 * latent attention while the context fits index_topk, sparse after. */
export function mla_forward(cfg, A, x, cl, pos) {
	let T = x.shape()[0];
	let H = cfg.num_q_heads;
	let nope = cfg.qk_nope_head_dim, dc = cfg.kv_lora_rank;
	let eps = cfg.rms_norm_eps;
	let len = pos + T;

	// queries: (T, H, nope) -> (H, T, nope)
	let qr = torch.rms_norm(A.q_a.forward(x), A.q_a_norm, eps);
	let q = A.q_b.forward(qr).view([T, H, nope]).transpose(0, 1);

	// latent K/V of this chunk, cached; indexer keys/pools always kept
	// current so the selection has the full history once it activates
	let ckv = torch.rms_norm(A.kv_a.forward(x), A.kv_a_norm, eps);   // (T, dc)

	ckv_put(cl, pos, ckv);
	mla_index_update(cfg, A, x, cl, pos);

	// absorb W_UK: (H, T, nope) x (H, nope, dc) -> (H, T, dc)
	let q_lat = torch.matmul(q, A.w_uk);
	let o_lat;

	if (len > cfg.index_topk) {
		o_lat = mla_sparse(cfg, A, x, qr, q_lat, cl, pos);
	}
	else {
		let kv = ckv_rows(cl, 0, len);                               // (len, dc)

		// scores over the whole cached latent stream (shared by all heads)
		let s = torch.matmul(q_lat.reshape([H * T, dc]), kv.t())      // (H*T, len)
			.view([H, T, len]).mul(cfg.sm_scale);

		s = s.add(causal_add_mask(T, pos, A.dev));
		s = s.softmax(-1);

		o_lat = torch.matmul(s.view([H * T, len]), kv).view([H, T, dc]);
	}

	let o = torch.matmul(o_lat, A.w_uv_t);                            // (H, T, v)

	return A.o.forward(o.transpose(0, 1).reshape([T, H * cfg.v_head_dim]));
}

/* ------------------------------------------------------------------ */
/* MLA decode steps for CUDA graphs: the position is a DEVICE tensor.  */
/*                                                                     */
/* mla_dev_prepare(cfg, st, dev) derives, once per step and device, the */
/* position-dependent index tensors every MLA layer of the step shares  */
/* (st: {pos (int64 [1]), ramp (int64 arange(T)), sparse, max_seq}).  */
/* mla_forward_dev(cfg, A, x, cl, st) then runs the layer: dense latent */
/* attention over the first index_topk cache rows (masked by position)  */
/* or the k-pool sparse selection over the whole pool array — the same  */
/* maths as mla_forward / mla_sparse, with no host-side positions.      */

export function mla_dev_prepare(cfg, st, dev) {
	let dv = 'cuda:' + dev;
	let P = cfg.index_kpool;
	let T = st.ramp.shape()[0];

	st.pos_idx = st.ramp.add(st.pos);                       // (T) int64
	st.qpos = st.pos_idx.to('float32');
	st.grp = st.qpos.mul(1.0 / P).to('int64');              // pool of each token
	st.members = st.grp.to('float32').view([T, 1]).mul(P * 1.0)
		.add(st.arP.view([1, P])).to('int64').reshape([-1]);

	if (!st.sparse) {
		let Lc = (cfg.index_topk < st.max_rows) ? cfg.index_topk : st.max_rows;
		let cols = torch.arange(Lc, 'int64', dv).view([1, Lc]);

		st.mask = torch.where(cols.le(st.pos_idx.view([T, 1])),
			torch.zeros([1], 'float16', dv), torch.full([1], -65504.0, 'float16', dv));
	}
}

export function mla_forward_dev(cfg, A, x, cl, st) {
	let T = x.shape()[0];
	let H = cfg.num_q_heads;
	let nope = cfg.qk_nope_head_dim, dc = cfg.kv_lora_rank;
	let eps = cfg.rms_norm_eps;
	let P = cfg.index_kpool, D = cfg.index_head_dim;
	let dv = 'cuda:' + cl.dev;

	// indexer keys / pools (cache writes) beside the EXL3 products of the
	// main chain — never two EXL3 products at once: they share per-device
	// lock buffers. A pool is rebuilt from the plane after every token of
	// its group; only complete pools are ever read.
	let keep = side(cl.dev, 1, function () {
		let k = torch.layer_norm(torch.matmul(x, A.idx_wk_t).to('float32'),
			A.idx_norm_w, A.idx_norm_b, 1e-6).to('float16');
		let gate = torch.matmul(x, A.idx_gate_t);

		cl.plane.index_copy_(0, st.pos_idx, torch.cat([k, gate], 1));

		let mem = cl.plane.index(st.members).view([T, P, 2 * D]).to('float32');
		let probs = mem.narrow(2, D, D).add(A.idx_ape.unsqueeze(0)).softmax(1);

		cl.pools.index_copy_(0, st.grp, probs.mul(mem.narrow(2, 0, D)).sum(1).to('float16'));

		return [ k, gate, mem, probs ];          // alive until the join
	});

	let qr = torch.rms_norm(A.q_a.forward(x), A.q_a_norm, eps);
	let q = A.q_b.forward(qr).view([T, H, nope]).transpose(0, 1);
	let ckv = torch.rms_norm(A.kv_a.forward(x), A.kv_a_norm, eps);

	ckv_put(cl, st.pos_idx, ckv);

	join(cl.dev, 1);

	let q_lat = torch.matmul(q, A.w_uk);                           // (H, T, dc)
	let o_lat;

	if (!st.sparse) {
		let Lc = st.mask.shape()[1];
		let kv = ckv_rows(cl, 0, Lc);
		let sc = torch.matmul(q_lat.reshape([H * T, dc]), kv.t())
			.view([H, T, Lc]).mul(cfg.sm_scale);

		sc = sc.add(st.mask).softmax(-1);
		o_lat = torch.matmul(sc.view([H * T, Lc]), kv).view([H, T, dc]);
	}
	else {
		let Hi = cfg.index_n_heads;
		let kp = int(cfg.index_topk / P);
		let q_idx = A.idx_wq_b.forward(qr).view([T, Hi, D]);
		let w = torch.matmul(x, A.idx_w_t).mul(math.pow(Hi * 1.0, -0.5));
		// fused scoring over the pool capacity (the graph's fixed shape),
		// masked by the device-side position; pools no query can select
		// yet cost nothing (L134)
		let scr = torch.idx_score(q_idx.contiguous(), cl.pools, w.to('float32').contiguous(),
			math.pow(D * 1.0, -0.5), st.pos, P);

		// the kp best pools (ties to the lower index), their tokens in
		// ascending pool order and the query's incomplete tail pool, with
		// an additive pad mask: one kernel instead of the topk, the dead /
		// tail / pad composite (~40 launches per layer, L138)
		let sel = torch.dsa_select(scr, st.pos, P, kp);
		let K = sel[0].shape()[1];
		let g = ckv_gather(cl, sel[0].reshape([-1])).view([T, K, dc]);
		let ql = q_lat.transpose(0, 1).contiguous();
		let a = torch.matmul(ql, g.transpose(1, 2)).mul(cfg.sm_scale);

		a = a.add(sel[1].view([T, 1, K]));
		a = a.softmax(-1);
		o_lat = torch.matmul(a, g).transpose(0, 1);
	}

	let o = torch.matmul(o_lat, A.w_uv_t);

	return A.o.forward(o.transpose(0, 1).reshape([T, H * cfg.v_head_dim]));
}

/* ------------------------------------------------------------------ */
/* MoE (layers 3..44): sigmoid router + 288 EXL3 experts + shared expert */

export function build_moe(ws, cfg, i, dev) {
	let key = `${LP}${i}.mlp`;
	let experts = [];

	for (let e = 0; e < cfg.n_routed_experts; e++)
		push(experts, build_mlp(ws, `${key}.experts.${e}`, dev));

	return {
		type: 'moe', dev: dev, experts: experts, layer: i,
		gate_t: ws.gpu(key + '.gate.weight', dev).to('float32').t().contiguous(),
		bias: ws.gpu(key + '.gate.e_score_correction_bias', dev).to('float32')
			.reshape([1, -1]).contiguous(),
		shared: build_mlp(ws, key + '.shared_experts', dev)
	};
}

/* token ids at or above this are multimodal rows (image embeddings), see dsv4.uc */
export const MM_BASE = 16777216;

/* Router (fp32): scores = sigmoid(y W^T); the top-k experts are chosen on
 * scores + e_score_correction_bias, their weights are the UNbiased scores
 * normalised to sum 1 and scaled by routed_scaling_factor.
 * y: (T, hidden) fp16. Returns {idx (T, k) int64, w (T, k) fp32}. */
const NO_FUSED_ROUTE = getenv('AMP_NO_FUSED_ROUTE') != null;
const NO_FUSED_PREP = getenv('AMP_NO_FUSED_PREP') != null;
export let TRACE = null;
export function set_trace(f) { TRACE = f; }

/* top-k expert ids the sigmoid router of M would pick for y (experiments) */
export function route_peek(cfg, M, y, k) {
	return mm32(y.to('float32'), M.gate_t).sigmoid().add(M.bias).topk(k, -1).indices;
}

export function moe_route(cfg, M, y) {
	if (M.router == 'gemma4') {
		// Gemma 4: the router logits come from the caller (M.rl, computed on the residual stream); softmax over all
		// experts, top-k, renormalised over the selected, times the per-expert scale of each selected expert
		let r = torch.v41_route(M.rl.contiguous(), null, cfg.num_experts_per_tok, 1.0);

		return {idx: r[0], w: r[1].mul(M.pes.index(r[0].reshape([-1])).view(r[0].shape()))};
	}

	if (M.router == 'softmax') {
		// Qwen3.8-Flash-Next ("std" router): softmax over all expert
		// logits, top-k, weights renormalised over the selected k
		if (!NO_FUSED_ROUTE && !getenv('AMP_QWEN4_NO_FUSED_ROUTE')) {
			// one kernel: top-k on the logits, softmax of the selected (the same weights)
			let r = torch.v41_route(mm32(y.to('float32'), M.gate_t), null, cfg.num_experts_per_tok, 1.0);

			if (TRACE)
				TRACE(cfg, M, y, r[0]);

			return {idx: r[0], w: r[1]};
		}

		let p = mm32(y.to('float32'), M.gate_t).softmax(-1);
		let tk = p.topk(cfg.num_experts_per_tok, -1);
		let w = tk.values;

		if (TRACE)
			TRACE(cfg, M, y, tk.indices);

		return {idx: tk.indices, w: w.div(w.sum(1, true).add(1e-20))};
	}

	if (M.router == 'sqrtsp') {
		// DeepSeek-V4: affinity sqrt(softplus(logit)); the bias steers the
		// selection only, hash layers (M.tid2eid) select by token id (M.ids,
		// int64 (T,) on this device); weights = affinities of the selected
		// experts normalised to sum 1, times routed_scaling_factor
		if (!M.tid2eid && !(M.vlon && M.bias_vl) && !NO_FUSED_ROUTE) {
			// one kernel: affinities, top-k on affinity + bias, normalized weights
			let r = torch.v41_route(mm32(y.to('float32'), M.gate_t), M.bias, cfg.num_experts_per_tok,
				cfg.routed_scaling_factor);

			return {idx: r[0], w: r[1]};
		}

		let sc = mm32(y.to('float32'), M.gate_t).softplus().sqrt();
		let vl = (M.vlon && M.bias_vl) ? M.ids.le(MM_BASE - 1) : null;      // text rows (chunk with image rows)
		let tids = vl ? torch.where(vl, M.ids, torch.zeros([1], 'int64', 'cuda:' + M.dev)) : M.ids;
		let sel = M.tid2eid ? M.tid2eid.index(tids) :
			sc.add(M.bias).topk(cfg.num_experts_per_tok, -1).indices;

		// image rows select on the vision bias (all layers, hash layers included)
		if (vl)
			sel = torch.where(vl.view([-1, 1]), sel,
				sc.add(M.bias_vl).topk(cfg.num_experts_per_tok, -1).indices);

		let sw = sc.gather(1, sel);

		return {idx: sel, w: sw.div(sw.sum(1, true).add(1e-20)).mul(cfg.routed_scaling_factor)};
	}

	let scores = mm32(y.to('float32'), M.gate_t).sigmoid();
	let idx = scores.add(M.bias).topk(cfg.num_experts_per_tok, -1).indices;
	let w = scores.gather(1, idx);

	w = w.div(w.sum(1, true).add(1e-20)).mul(cfg.routed_scaling_factor);

	if (TRACE)
		TRACE(cfg, M, y, idx);

	return {idx: idx, w: w};
}

/* Shared expert output (T, hidden) fp32; with a shared gate (Qwen) scaled by
 * sigmoid(y . gate) per token. */
function shared_out(cfg, M, y) {
	let s = mlp_forward(cfg, M.shared, y).to('float32');

	if (M.shared_gate)
		s = s.mul(mm32(y.to('float32'), M.shared_gate).sigmoid());

	return s;
}

/* Fused decode path state (torch.moe_coop): int64 pointer tables of the
 * experts' trellis / suh / svh data and the scratch for up to COOP_ROWS
 * tokens. Built lazily; null when the experts do not share one bitrate /
 * codebook (the composite path then serves every size). */
const COOP_ROWS = 8;

function build_coop(cfg, M) {
	let ex = M.experts;
	let g0 = ex[0].gate, u0 = ex[0].up, d0 = ex[0].down;
	let dv = 'cuda:' + M.dev;

	for (let e in ex)
		if (e.gate.K != g0.K || e.up.K != u0.K || e.down.K != d0.K ||
		    e.gate.mul1 != g0.mul1 || e.gate.mcg != g0.mcg ||
		    e.up.mul1 != g0.mul1 || e.down.mul1 != g0.mul1 ||
		    e.up.mcg != g0.mcg || e.down.mcg != g0.mcg || g0.K != u0.K)
			return null;

	function table(name, field) {
		let a = [];

		for (let e in ex)
			push(a, e[name][field].data_ptr());

		return torch.from_array(a, 'int64', dv);
	}

	let Hi = g0.in_features, I = g0.out_features, Ho = d0.out_features;
	let slots = COOP_ROWS * cfg.num_experts_per_tok;
	let ctr_len = slots * int(I / 128) + COOP_ROWS * int(Ho / 128) + 2 + (slots + 1) + slots;

	return {
		g_trellis: table('gate', 'trellis'), g_suh: table('gate', 'suh'), g_svh: table('gate', 'svh'),
		u_trellis: table('up', 'trellis'), u_suh: table('up', 'suh'), u_svh: table('up', 'svh'),
		d_trellis: table('down', 'trellis'), d_suh: table('down', 'suh'), d_svh: table('down', 'svh'),
		had_g: torch.zeros([slots, Hi], 'float16', dv),
		had_u: torch.zeros([slots, Hi], 'float16', dv),
		gu_g: torch.zeros([slots, I], 'float16', dv),
		gu_u: torch.zeros([slots, I], 'float16', dv),
		act_out: torch.zeros([slots, I], 'float16', dv),
		d_out: torch.zeros([slots, Ho], 'float32', dv),
		ctr: torch.zeros([ctr_len], 'int32', dv),
		out: torch.zeros([COOP_ROWS, cfg.hidden_size], 'float32', dv),
		Hi: Hi, Kg: g0.K, Ku: u0.K, Kd: d0.K, mcg: g0.mcg ? true : false,
		mul1: g0.mul1 ? true : false, act: (cfg.act == 'gelu') ? 1 : 0, act_limit: cfg.swiglu_limit + 0.0,
		// expert-parallel shards hold experts [e0, e1) of the router's range
		min_expert: (M.e0 != null) ? M.e0 : -1, max_expert: (M.e1 != null) ? M.e1 : -1
	};
}

/* y: (T, hidden) fp16, T <= COOP_ROWS: routing on the device, two fused
 * launches for the routed experts. Returns (T, hidden) fp32. */
function moe_forward_fused(cfg, M, y) {
	let T = y.shape()[0];
	// (M.shared == null: expert-parallel shards whose rank does not own the shared expert)
	let sh = M.shared ? side(M.dev, 1, () => shared_out(cfg, M, y).contiguous()) : null;
	let r = moe_route(cfg, M, y);

	if (sh)
		join(M.dev, 1);

	torch.moe_coop(y.contiguous(), r.idx.contiguous(), r.w.to('float16').contiguous(),
		M.coop, sh);

	return M.coop.out.narrow(0, 0, T);
}

/* Composite path: host grouping by expert, one gated MLP per active expert. */
export function moe_forward_ref(cfg, M, y) {
	let T = y.shape()[0];
	let H = cfg.hidden_size;
	let K = cfg.num_experts_per_tok;
	let dv = 'cuda:' + M.dev;
	let r = moe_route(cfg, M, y);
	let idx = r.idx.reshape([-1]).tolist();
	let wl = r.w.reshape([-1]).tolist();

	// group the (token, weight) assignments by expert
	let rows = [], wts = [], active = [];

	for (let e = 0; e < cfg.n_routed_experts; e++) {
		push(rows, null);
		push(wts, null);
	}

	for (let a = 0; a < length(idx); a++) {
		let e = idx[a];

		if (rows[e] == null) {
			rows[e] = [];
			wts[e] = [];
			push(active, e);
		}

		push(rows[e], int(a / K));
		push(wts[e], wl[a]);
	}

	let acc = torch.zeros([T, H], 'float32', dv);

	for (let e in active) {
		let rt = torch.from_array(rows[e], 'int64', dv);
		let ye = y.index(rt);
		let d = mlp_forward(cfg, M.experts[e], ye).to('float32');
		let wt = torch.from_array(wts[e], 'float32', dv).view([-1, 1]);

		acc.index_add_(0, rt, d.mul(wt));
	}

	return M.shared ? acc.add(shared_out(cfg, M, y)) : acc;
}

/* Token-sorted fused path (torch.moe_fused) for prefill chunks: experts
 * with up to FUSED_ROWS routed rows run gate/up/act/down in the kernel
 * (three row-tile launches), heavier ones through the composite path.
 * Temps are shared by all layers of a device. */
let FUSED_ROWS = 256;

/* Capacity (rows per expert) of the fused prefill kernel; a model that sizes it to its
 * chunk (a token picks an expert at most once) can use the readback-free path
 * (M.async_moe). Call before the first prefill: the per-device temps are sized by it. */
export function set_fused_rows(n) {
	FUSED_ROWS = n;
}
export let HPG = getenv('AMP_DSV4_HOSTPROF') ? {wait: 0, after: 0, n: 0, heavy: 0} : null;

function hp_now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}
let FUSED_TEMPS = {};
let FLAT_TOKEN = {};

function fused_temps(cfg, M) {
	let key = '' + M.dev;

	if (FUSED_TEMPS[key] == null) {
		let dv = 'cuda:' + M.dev;
		let C = torch.moe_concurrency(M.dev);
		let H = cfg.hidden_size, I = M.coop.gu_u.shape()[1];

		FUSED_TEMPS[key] = {
			ts_g: torch.zeros([C, FUSED_ROWS, H], 'float16', dv),
			ts_u: torch.zeros([C, FUSED_ROWS, H], 'float16', dv),
			ti_g: torch.zeros([C, FUSED_ROWS, I], 'float16', dv),
			ti_u: torch.zeros([C, FUSED_ROWS, I], 'float16', dv)
		};
	}

	let t = FUSED_TEMPS[key];

	for (let k in t)
		M.coop[k] = t[k];
}

/* Phase A (async, no host reads): routing, sort by expert, per-expert counts. */
function moe_sorted_begin(cfg, M, y) {
	let T = y.shape()[0];
	let K = cfg.num_experts_per_tok;
	let E = (M.e0 != null) ? length(M.experts) : cfg.n_routed_experts;
	let dv = 'cuda:' + M.dev;
	let A = T * K;
	let r = moe_route(cfg, M, y);

	if (M.fused_prep && !NO_FUSED_PREP) {
		// the whole dispatch (remap to local ids, stable sort by expert, counts / starts / inverse / kinds) in one kernel
		if (M.coop.ts_g == null)
			fused_temps(cfg, M);

		let pr = torch.moe_prep(r.idx.contiguous(), r.w.contiguous(), E, (M.e0 != null) ? M.e0 : 0);

		return {y: y, T: T, A: A, fe: pr[0], order: pr[1], tok: pr[2], wts: pr[3], count: pr[4], start: pr[5], inv: pr[6],
			kind: pr[7], prep: true};
	}

	let fe = r.idx.reshape([-1]);

	if (M.e0 != null) {
		// expert-parallel shard: global ids -> local ones, foreign picks
		// into the sentinel bucket E (never launched)
		let loc = fe.sub(torch.full([1], M.e0, 'int64', dv));
		let lo_ok = loc.neg().le(0).to('int64');
		let hi_ok = loc.le(E - 1).to('int64');
		let bad = lo_ok.mul(hi_ok).le(0);

		fe = torch.where(bad, torch.full([1], E, 'int64', dv), loc);
	}

	if (M.coop.ts_g == null)
		fused_temps(cfg, M);

	let fkey = dv + ':' + T;

	if (FLAT_TOKEN[fkey] == null) {
		let a = [], ones = [];

		for (let i = 0; i < A; i++) {
			push(a, int(i / K));
			push(ones, 1);
		}

		FLAT_TOKEN[fkey] = {
			tok: torch.from_array(a, 'int64', dv),
			ones: torch.from_array(ones, 'int64', dv)
		};
	}

	let ft = FLAT_TOKEN[fkey];
	let order = fe.to('float32').neg().topk(A, 0).indices;

	return {
		y: y, T: T, A: A, fe: fe, order: order,
		tok: ft.tok.gather(0, order),
		wts: r.w.reshape([-1]).to('float16').gather(0, order),
		count: torch.zeros([E + 1], 'int64', dv).index_add_(0, fe, ft.ones)
	};
}

/* The routed experts' weighted outputs land in per-assignment slots (a scratch of fp32 rows) and
 * one gather sums every token's slots in a fixed order, instead of atomic adds into the token row
 * in arrival order: bit-reproducible run to run, at ~1% of the prefill speed (129k tokens:
 * 1531 -> 1514 tok/s). An atomic sum of up to 6 terms rounds differently per order; fp16
 * narrowing and the depth of the network amplify that to differences of ~20 at the
 * massive-activation rows and flip near-tie tokens. AMP_MOE_ATOMIC=1 restores the atomic sum. */
const MOE_DET = getenv('AMP_MOE_ATOMIC') == null;

/* Phase B: read the counts (syncs this device only), launch the row-tile
 * groups, heavy experts through the composite path, shared expert. */
function moe_sorted_finish(cfg, M, p) {
	let E = (M.e0 != null) ? length(M.experts) : cfg.n_routed_experts;
	let H = cfg.hidden_size;
	let y = p.y, T = p.T;
	let dv = 'cuda:' + M.dev;
	let tok = p.tok, wts = p.wts, count = p.count;

	// No host readback: every expert fits the fused kernel (rows per expert <= T <= FUSED_ROWS),
	// the three row tiers pick their experts from the device-side counts
	if (M.async_moe && T <= FUSED_ROWS) {
		let out = torch.zeros([T, H], 'float32', dv);
		let yc = y.contiguous();

		if (MOE_DET) {
			// every assignment (the sentinel bin included, never written) owns slot = its sorted position
			let start = p.prep ? p.start : count.cumsum(0).sub(count);
			let scratch = torch.empty([max(p.A, 1), H], 'float32', dv);
			let inv = p.prep ? p.inv : p.order.to('float32').neg().topk(p.A, 0).indices;      // argsort of the permutation
			let kind = p.prep ? p.kind : torch.ones([E + 1], 'int64', dv).sub(count.le(0).to('int64'));

			if (M.coop.mul1) {
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 33, FUSED_ROWS, 64, scratch, start);
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 17, 32, 32, scratch, start);
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 1, 16, 16, scratch, start);
			}
			else {
				// (the 32 / 64 row tiles exist for the mul1 codebook only: one tier of 16 row tiles)
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 1, FUSED_ROWS, 16, scratch, start);
			}
			torch.moe_gather(out, scratch, p.fe, inv, start.narrow(0, 0, E), start.narrow(0, 0, E),
				kind.narrow(0, 0, E), wts);
		}
		else {
			if (M.coop.mul1) {
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 33, FUSED_ROWS, 64);
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 17, 32, 32);
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 1, 16, 16);
			}
			else
				torch.moe_fused(yc, out, count, tok, wts, M.coop, -1, 1, FUSED_ROWS, 16);
		}

		return M.shared ? out.add(shared_out(cfg, M, y)) : out;
	}

	let tw0 = HPG ? hp_now() : 0;
	let cl = count.tolist();

	if (HPG) {
		HPG.wait += hp_now() - tw0;
		HPG.n++;
	}

	let th0 = HPG ? hp_now() : 0;
	let out = torch.zeros([T, H], 'float32', dv);
	let c = M.coop;
	let n0 = 0, n1 = 0, n2 = 0, heavy = [];
	let starts = [], acc = 0;

	for (let e = 0; e < E; e++) {
		push(starts, acc);
		acc += cl[e];

		if (cl[e] == 0)
			continue;
		else if (!c.mul1 && cl[e] <= FUSED_ROWS)
			n0++;                                  // mcg: 16-row tiles for every expert the kernel holds
		else if (cl[e] <= 16)
			n0++;
		else if (cl[e] <= 32)
			n1++;
		else if (cl[e] <= FUSED_ROWS)
			n2++;
		else
			push(heavy, e);
	}

	let yc = y.contiguous();

	if (c.mul1) {
		if (n2) torch.moe_fused(yc, out, count, tok, wts, c, n2, 33, FUSED_ROWS, 64);
		if (n1) torch.moe_fused(yc, out, count, tok, wts, c, n1, 17, 32, 32);
		if (n0) torch.moe_fused(yc, out, count, tok, wts, c, n0, 1, 16, 16);
	}
	else if (n0)
		torch.moe_fused(yc, out, count, tok, wts, c, n0, 1, FUSED_ROWS, 16);

	for (let e in heavy) {
		let rt = tok.narrow(0, starts[e], cl[e]);
		let d = mlp_forward(cfg, (M.stage && M.stage.on) ? expert_host.mlp(M, e) : M.experts[e], y.index(rt)).to('float32');
		let wt = wts.narrow(0, starts[e], cl[e]).to('float32').view([-1, 1]);

		out.index_add_(0, rt, d.mul(wt));
	}

	let res = M.shared ? out.add(shared_out(cfg, M, y)) : out;

	if (HPG) {
		HPG.after += hp_now() - th0;
		HPG.heavy += length(heavy);
	}

	return res;
}

/* Start the MoE of y: returns a pending object for moe_finish. The
 * decode-sized and composite paths complete immediately. */
export function moe_begin(cfg, M, y) {
	let T = y.shape()[0];

	if (!getenv('AMP_GLM5_NO_COOP')) {
		if (M.coop == null)
			M.coop = build_coop(cfg, M) || false;

		if (M.coop && T <= COOP_ROWS)
			return {done: moe_forward_fused(cfg, M, y)};

		// host-resident experts, staged for this chunk (expert_host.begin): the prefill tables point at the staging buffer
		if (M.coop && M.stage && M.stage.on) {
			if (M.stage.pf == null)
				M.stage.pf = expert_host.tables(M);

			M.stage.dec = M.coop;
			M.coop = M.stage.pf;
		}

		if (M.coop && (M.coop.mul1 || M.coop.mcg) && !getenv('AMP_GLM5_NO_SORTED'))
			return {sorted: moe_sorted_begin(cfg, M, y)};
	}

	return {done: moe_forward_ref(cfg, M, y)};
}

export function moe_finish(cfg, M, p) {
	let r = p.sorted ? moe_sorted_finish(cfg, M, p.sorted) : p.done;

	if (M.stage && M.stage.on) {
		M.coop = M.stage.dec;
		M.stage.on = false;
	}

	return r;
}

/* y: (T, hidden) fp16. Returns the MoE output (T, hidden) fp32. */
export function moe_forward(cfg, M, y) {
	return moe_finish(cfg, M, moe_begin(cfg, M, y));
}

/* ------------------------------------------------------------------ */
/* model                                                               */

/* Bytes of the per-token cache state of one MLA layer: the latent (fp16,
 * or Q8 codes + group scales), the indexer plane [key || gate] and the
 * pooled keys. */
export function mla_cache_bytes_per_token(cfg, kv_bits) {
	return ((kv_bits == 8) ? cfg.kv_lora_rank + cfg.kv_lora_rank / 16 : 2 * cfg.kv_lora_rank) +
		4 * cfg.index_head_dim + 2 * cfg.index_head_dim / cfg.index_kpool;
}

/* Place the layers on `ngpu` devices: contiguous runs, minimizing the
 * LARGEST device footprint. A device's footprint is its layers' weights
 * plus, for MLA layers, the cache at opts.max_seq tokens (opts.kv_bits
 * picks its format), plus the fixed parts: the embedding (counted twice —
 * it is cast on the device while the raw upload is still resident) on the
 * first device, the final head + norm on the last, and — with
 * opts.mtp_bytes (MTP head weights) — the draft block with its own MLA
 * cache on whichever device it fits best (the hidden / logits hand-off is
 * a few KB, so any device will do). Returns {devs (per layer), mtp (the
 * draft block's device, null without one), load (per-device bytes)}. */
export function plan_devices(ws, cfg, ngpu, opts) {
	opts = opts || {};

	let nl = cfg.num_hidden_layers;
	let bytes = [];

	for (let i = 0; i < nl; i++)
		push(bytes, 0);

	let embed = 0, head = 0;

	for (let key in ws.list()) {
		if (index(key, LP) == 0) {
			let rest = substr(key, length(LP));
			let idx = +substr(rest, 0, index(rest, '.'));

			if (idx >= 0 && idx < nl)
				bytes[idx] += ws.info(key).nbytes;
		}
		else if (index(key, 'model.language_model.embed_tokens') == 0) {
			embed += ws.info(key).nbytes;
		}
		else if (index(key, 'lm_head') == 0 || index(key, 'model.language_model.norm') == 0) {
			head += ws.info(key).nbytes;
		}
	}

	// opts.shard: every "device" is a group of that many GPUs sharing each
	// layer (glm5_tp stages): weights, head and the MTP block divide by it,
	// while the embedding (one GPU) and the latent cache (replicated in the
	// group) stay whole per GPU
	let shard = opts.shard || 1;
	let kv_tok = mla_cache_bytes_per_token(cfg, opts.kv_bits) * (opts.max_seq || 0);

	for (let i = 0; i < nl; i++) {
		bytes[i] = bytes[i] / shard;

		if (cfg.layer_types[i] != 'linear_attention')
			bytes[i] += kv_tok;
	}

	// fixed parts per device
	let base = [];

	for (let d = 0; d < ngpu; d++)
		push(base, 0);

	// (a sharded stage gives its rank 0, which holds the embedding, a
	// correspondingly smaller weight share — glm5_tp stage_weights — so the
	// embedding adds nothing to the group's per-GPU footprint)
	base[0] += (shard > 1) ? 0 : 2 * embed;
	bytes[nl - 1] += head / shard;        // the head lives with the last layer

	let mtp_total = opts.mtp_bytes ? opts.mtp_bytes / shard + kv_tok : 0;

	// Exact contiguous partition: dp[d][i] = the best way to put layers
	// [0, i) on devices [0, d] (every device gets at least one layer; layer
	// 0 is on device 0, where the embedding lives), "best" = the smallest
	// largest-device footprint, then the smallest sum of squared footprints
	// (an even spread, not just a low maximum). `m` is the device that also
	// holds the draft block (-1: none). Footprints are in GiB so the
	// squares stay well inside a double's exact range.
	const GIB = 1073741824.0;

	let prefix = [ 0 ];

	for (let i = 0; i < nl; i++)
		push(prefix, prefix[i] + bytes[i]);

	function plan_for(m) {
		let dp = [];            // dp[d][i] = {mx, sq, from}

		for (let d = 0; d < ngpu; d++) {
			let row = [];

			for (let i = 0; i <= nl; i++)
				push(row, null);

			push(dp, row);
		}

		for (let d = 0; d < ngpu; d++) {
			let fixed = base[d] + ((m == d) ? mtp_total : 0);

			for (let i = d + 1; i <= nl - (ngpu - 1 - d); i++) {
				let b = null;

				for (let j = d; j < i; j++) {
					let prev = (d == 0) ? ((j == 0) ? {mx: 0, sq: 0} : null) : dp[d - 1][j];

					if (prev == null)
						continue;

					let ld = (fixed + prefix[i] - prefix[j]) / GIB;
					let mx = (ld > prev.mx) ? ld : prev.mx;
					let sq = prev.sq + ld * ld;

					if (b == null || mx < b.mx - 1e-9 || (mx < b.mx + 1e-9 && sq < b.sq))
						b = {mx: mx, sq: sq, from: j};
				}

				dp[d][i] = b;
			}
		}

		let end = dp[ngpu - 1][nl];

		if (end == null)
			return null;

		let devs = [];

		for (let i = 0; i < nl; i++)
			push(devs, 0);

		let at = nl;

		for (let d = ngpu - 1; d >= 0; d--) {
			let j = dp[d][at].from;

			for (let k = j; k < at; k++)
				devs[k] = d;

			at = j;
		}

		return {mx: end.mx, sq: end.sq, devs: devs, mtp: (m >= 0) ? m : null};
	}

	let best = null;

	for (let m = (mtp_total > 0) ? ngpu - 1 : -1; m >= (mtp_total > 0 ? 0 : -1); m--) {
		// opts.mtp_at pins the draft block to one device
		if (mtp_total > 0 && opts.mtp_at != null && m != opts.mtp_at)
			continue;

		let p = plan_for(m);

		// (ties keep the later device: the one next to the head)
		if (p && (best == null || p.mx < best.mx - 1e-9 ||
		    (p.mx < best.mx + 1e-9 && p.sq < best.sq - 1e-9)))
			best = p;

		if (mtp_total <= 0)
			break;
	}

	if (best == null)
		die('plan_devices: the model does not fit on ' + ngpu + ' devices');

	let load = [];

	for (let d = 0; d < ngpu; d++)
		push(load, base[d] + ((best.mtp == d) ? mtp_total : 0));

	for (let i = 0; i < nl; i++)
		load[best.devs[i]] += bytes[i];

	return {devs: best.devs, mtp: best.mtp, load: load};
}

/* RMSNorm of an fp32 tensor (T, D) with an fp16 weight, fp16 result: the
 * mean of the streams can exceed the fp16 range, so it is never rounded
 * to fp16 before the normalisation (as in the reference). */
export function rms_norm_f32(x, w, eps) {
	let r = x.mul(x).mean(-1, true).add(eps).rsqrt();

	return x.mul(r).to('float16').mul(w);
}

/* Build one layer's weights on its device. */
export function build_layer(ws, cfg, i, dev) {
	let L = {dev: dev, kind: cfg.layer_types[i]};

	L.hc_attn = build_hc(ws, i, 'attn', dev);
	L.hc_ffn = build_hc(ws, i, 'ffn', dev);
	L.input_layernorm = ws.gpu16(`${LP}${i}.input_layernorm.weight`, dev);
	L.post_attention_layernorm = ws.gpu16(`${LP}${i}.post_attention_layernorm.weight`, dev);

	L.attn = (L.kind == 'linear_attention') ?
		build_kda(ws, cfg, i, dev) : build_mla(ws, cfg, i, dev);

	L.mlp = (cfg.mlp_layer_types[i] == 'dense') ?
		build_mlp(ws, `${LP}${i}.mlp`, dev) : build_moe(ws, cfg, i, dev);
	L.moe = (cfg.mlp_layer_types[i] != 'dense');

	return L;
}

/* One transformer block on the streams (T, 4, hidden) fp32 (in place).
 * cl: this layer's cache entry; pos: tokens already cached. */
let PROF = null;

function prof_mark(dev) {
	torch.synchronize();

	let c = clock(true);

	return c[0] + c[1] / 1000000000.0;
}

/* Layer phase A: attention half incl. residual update, the ffn site's hc
 * mix + norm, and (MoE layers) the routing — everything that can be
 * enqueued without reading the device. */
export function block_a(cfg, L, streams, cl, pos) {
	let eps = cfg.rms_norm_eps;
	let r = hc_mix(cfg, L.hc_attn, streams);
	let xn = torch.rms_norm(r[2], L.input_layernorm, eps);
	let y = (L.kind == 'linear_attention') ?
		kda_forward(cfg, L.attn, xn, cl) :
		mla_forward(cfg, L.attn, xn, cl, pos);

	hc_apply(streams, y, r[0], r[1]);

	r = hc_mix(cfg, L.hc_ffn, streams);
	xn = torch.rms_norm(r[2], L.post_attention_layernorm, eps);

	return {
		streams: streams, r: r,
		pend: L.moe ? moe_begin(cfg, L.mlp, xn) : {done: mlp_forward(cfg, L.mlp, xn)}
	};
}

/* Layer phase B: finish the ffn (may read counts from the device) and
 * apply it to the streams. */
export function block_b(cfg, L, ctx) {
	let y = L.moe ? moe_finish(cfg, L.mlp, ctx.pend) : ctx.pend.done;

	hc_apply(ctx.streams, y, ctx.r[0], ctx.r[1]);

	return ctx.streams;
}

export function block_forward(cfg, L, streams, cl, pos) {
	if (!PROF)
		return block_b(cfg, L, block_a(cfg, L, streams, cl, pos));

	let t0 = prof_mark();
	let ctx = block_a(cfg, L, streams, cl, pos);
	let t1 = prof_mark();
	let out = block_b(cfg, L, ctx);
	let t2 = prof_mark();
	let k = (L.kind == 'linear_attention') ? 'kda' : 'mla';
	let m = L.moe ? 'moe' : 'mlp';

	// phase A mixes attention + routing; reported as one line
	PROF[k + '+a'] = (PROF[k + '+a'] || 0) + (t1 - t0);
	PROF[m + '+b'] = (PROF[m + '+b'] || 0) + (t2 - t1);

	return out;
}

/* ------------------------------------------------------------------ */
/* CUDA-graph decode (S = 1)                                           */
/*                                                                     */
/* Runs of consecutive non-MLA layers (KDA + dense/MoE ffn) on one     */
/* device are captured as one graph each: everything inside is         */
/* position-independent (KDA states update in place, the MoE routes    */
/* on the device, no host reads). The MLA layers run eagerly between   */
/* the graphs (their cache length is a host value).                    */

function build_graphs(model, cache, sizes) {
	let cfg = model.cfg;
	let nl = cfg.num_hidden_layers;
	let H = cfg.hidden_size;
	let by_size = {};

	// the item list of S-row steps (S = 1: plain decode; S = W: the MTP
	// verify, whose KDA layers keep the per-position states)
	function make_items(S) {
		let items = [];

		for (let i = 0; i < nl; i++) {
			let L = model.layers[i];
			let last = length(items) ? items[length(items) - 1] : null;

			if (L.kind != 'linear_attention')
				push(items, {eager: true, layer: i, dev: L.dev});
			else if (last && !last.eager && last.dev == L.dev)
				last.end = i + 1;
			else
				push(items, {eager: false, first: i, end: i + 1, dev: L.dev});
		}

		function body(it) {
			let h = it.h_in;

			for (let i = it.first; i < it.end; i++)
				h = block_forward(cfg, model.layers[i], h, cache.layers[i], 0);

			it.h_out = h;
		}

		for (let it in items) {
			if (it.eager)
				continue;

			it.h_in = torch.zeros([S, 4, H], 'float32', 'cuda:' + it.dev);
			body(it);                               // eager warm-up
		}

		torch.synchronize();

		for (let it in items) {
			if (it.eager)
				continue;

			it.graph = torch.graph();
			it.graph.capture_begin(it.dev);
			body(it);
			it.graph.capture_end();
		}

		torch.synchronize();

		return items;
	}

	for (let S in (sizes || [ 1 ]))
		by_size[S] = make_items(S);

	// tids: (S) int64 token ids on the first device
	function run_ids(tids, S, pos) {
		let items = by_size[S];
		let h = model.embed_tokens.index(tids).to('float32')
			.reshape([S, 1, H]).expand([S, 4, H]).contiguous();
		let dev = model.devs[0];

		for (let it in items) {
			if (it.eager) {
				if (it.dev != dev) {
					h = h.cuda(it.dev);
					dev = it.dev;
				}

				h = block_forward(cfg, model.layers[it.layer], h,
					cache.layers[it.layer], pos);
			}
			else {
				it.h_in.copy_(h);
				it.graph.replay();
				h = it.h_out;
				dev = it.dev;
			}
		}

		cache.length = pos + S;

		let x = rms_norm_f32(h.mean(1), model.norm, cfg.rms_norm_eps);

		// the post-norm hidden of every row is the MTP head's input
		cache.hidden = x.view([1, S, H]);

		return {logits: model.lm_head.forward(x), hidden: cache.hidden};
	}

	return {
		has: function (S) {return by_size[S] != null;},

		run: function (ids, pos) {
			return run_ids(torch.from_array(ids, 'int64', 'cuda:' + model.devs[0]),
				length(ids), pos);
		},

		// ids_dev: (1, S) int64 on the first device
		run_dev: function (ids_dev, pos) {
			let S = ids_dev.shape()[1];

			return run_ids(ids_dev.view([S]), S, pos);
		}
	};
}

/* The MTP draft head (mtp.safetensors, layer index = num_hidden_layers) on
 * ONE device: enorm / hnorm / eh_proj in front of a plain residual block
 * (no mHC) — MLA + MoE like a trunk layer — and shared_head.norm before
 * the model's lm_head. glm5_tp.load_mtp is the sharded twin. */
function load_mtp(model_dir, cfg, hdev, dev) {
	let ws = weights.open(model_dir, 'mtp.safetensors');
	let nl = cfg.num_hidden_layers;
	let pfx = `${LP}${nl}.`;

	ws.upload(function (key) {
		return (index(key, pfx) == 0) ? dev : null;
	});

	return {
		dev: hdev, cdev: dev,
		attn: build_mla(ws, cfg, nl, dev),
		mlp: build_moe(ws, cfg, nl, dev),
		input_layernorm: ws.gpu16(pfx + 'input_layernorm.weight', dev),
		post_attention_layernorm: ws.gpu16(pfx + 'post_attention_layernorm.weight', dev),
		enorm: ws.gpu16(pfx + 'enorm.weight', dev),
		hnorm: ws.gpu16(pfx + 'hnorm.weight', dev),
		fnorm: ws.gpu16(pfx + 'shared_head.norm.weight', dev),
		eh: models.linear(ws, pfx + 'eh_proj', false, dev)
	};
}

/* The draft block: e / h are (n, hidden) fp16 on the head's device (token
 * embeddings and the target's post-norm hidden of the PREVIOUS positions);
 * dc the draft cache, pos its length. Returns the post-norm draft hidden
 * (n, hidden) fp16 (same rounding points as the sharded twin). */
function mtp_core(cfg, M, e, h, dc, pos, st) {
	let eps = cfg.rms_norm_eps;
	let x = M.eh.forward(torch.cat([
		torch.rms_norm(e, M.enorm, eps), torch.rms_norm(h, M.hnorm, eps) ], -1))
		.to('float32');
	let r = rms_norm_f32(x, M.input_layernorm, eps);
	// st: device-position state of a graph capture, else the host position
	let a = st ? mla_forward_dev(cfg, M.attn, r, dc.layers[0], st) :
		mla_forward(cfg, M.attn, r, dc.layers[0], pos);

	x = x.add(a.to('float32'));

	let y = moe_forward(cfg, M.mlp, rms_norm_f32(x, M.post_attention_layernorm, eps))
		.to('float16');

	return rms_norm_f32(x.add(y.to('float32')), M.fnorm, eps);
}

/* CUDA-graph draft step (one token): the embedding and the target hidden
 * come in through persistent buffers on the draft device, the output stays
 * in a persistent buffer. Dense and sparse (pool-selected) MLA variants,
 * the position is a device value (glm5_tp.mtp_graph is the sharded twin). */
function build_mtp_graph(model, cache) {
	let cfg = model.cfg, M = model.mtp, dc = cache.mtp;
	let H = cfg.hidden_size, P = cfg.index_kpool;
	let dv = 'cuda:' + M.cdev;
	let ent = {
		emb: torch.zeros([1, H], 'float16', dv),
		hid: torch.zeros([1, H], 'float16', dv)
	};

	for (let variant = 0; variant < ((cache.max_seq > cfg.index_topk) ? 2 : 1); variant++) {
		let o = {
			sparse: (variant == 1), max_rows: dc.max_seq,
			pos: torch.zeros([1], 'int64', dv),
			ramp: torch.arange(1, 'int64', dv),
			arP: torch.arange(P, 'float32', dv),
			arT: torch.arange(P - 1, 'float32', dv)
		};

		let body = function () {
			mla_dev_prepare(cfg, o, M.cdev);
			ent['out' + variant] = mtp_core(cfg, M, ent.emb, ent.hid, dc, 0, o);
		};

		o.pos.fill_(0);
		body();
		torch.synchronize();

		let gr = torch.graph();

		gr.capture_begin(M.cdev);
		body();
		gr.capture_end();
		torch.synchronize();

		ent['graph' + variant] = gr;
		ent['st' + variant] = o;
	}

	return ent;
}

/* Load the text model. opts: devices (array of per-layer device indices,
 * or a GPU count for an automatic byte-balanced split; default 1). */
export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = cfg.num_hidden_layers;
	let devs = opts.devices;

	if (devs == null)
		devs = 1;

	let has_mtp = (opts.mtp != false && stat(model_dir + '/mtp.safetensors'));
	let mtp_dev = null;

	if (type(devs) != 'array') {
		let ngpu = devs;
		let mtp_bytes = 0;

		if (has_mtp) {
			let mw = weights.open(model_dir, 'mtp.safetensors');

			for (let key in mw.list())
				mtp_bytes += mw.info(key).nbytes;
		}

		// opts.max_seq (the cache length the caller will allocate) and
		// opts.kv_bits make the per-device cache bytes part of the balance
		let plan = plan_devices(ws, cfg, ngpu, {max_seq: opts.max_seq, kv_bits: opts.kv_bits,
			mtp_bytes: mtp_bytes});

		devs = plan.devs;
		mtp_dev = plan.mtp;

		let line = '';

		for (let d = 0; d < ngpu; d++)
			line += sprintf(' %.1f', plan.load[d] / 1073741824.0);

		warn('glm5: planned footprint per GPU (GiB, weights' +
			(opts.max_seq ? ' + cache' : '') + (has_mtp ? ' + MTP block on GPU ' + mtp_dev : '') +
			'):' + line + '\n');
	}

	let first = devs[0], last = devs[nl - 1];

	ws.upload(function (key) {
		if (index(key, LP) == 0) {
			let rest = substr(key, length(LP));
			let idx = +substr(rest, 0, index(rest, '.'));

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (index(key, 'model.language_model.embed_tokens') == 0)
			return first;

		if (index(key, 'model.language_model.norm') == 0 || index(key, 'lm_head') == 0)
			return last;

		return null;   // mtp / vision: not part of the text model
	});

	let layers = [];

	for (let i = 0; i < nl; i++)
		push(layers, build_layer(ws, cfg, i, devs[i]));

	let embed_t = ws.gpu16('model.language_model.embed_tokens.weight', first);

	ws.release('model.language_model.embed_tokens.weight', first);

	// the draft block runs where the planner found room (explicit device
	// lists: the device before the final one); the hidden / logits
	// interface stays on the last device (mtp.dev)
	if (has_mtp && mtp_dev == null)
		mtp_dev = (last > 0) ? last - 1 : last;

	let mtp = has_mtp ? load_mtp(model_dir, cfg, last, mtp_dev) : null;

	let model = {
		cfg: cfg, devs: devs, ws: ws, layers: layers,
		embed_tokens: embed_t,
		norm: ws.gpu16('model.language_model.norm.weight', last),
		lm_head: models.linear(ws, 'lm_head', false, last),
		mtp: mtp, kv_bits: opts.kv_bits,

		/* Draft step(s) for the engine: token ids (array) at draft
		 * positions pos.. (or ids_dev, (1, n) int64 on the first device),
		 * hidden (1, n, hidden) the target's post-norm hidden of the
		 * PREVIOUS positions. head: truthy -> logits. Returns {logits
		 * (1, n, V) | null, hidden (1, n, hidden)} on the head's device. */
		mtp_chunk: function (token_ids, hidden, dc, pos, head, ids_dev) {
			let H = cfg.hidden_size;
			let n = ids_dev ? ids_dev.shape()[1] : length(token_ids);
			let e = ids_dev ? this.embed_tokens.index(ids_dev.view([n])) :
				qwen35.embed_rows(this, dc.mm, token_ids, pos).view([n, H]);
			let mg = this.mtp_g;

			if (n == 1 && mg && head) {
				let v = (pos + 1 > cfg.index_topk) ? 1 : 0;

				mg.emb.copy_(e.view([1, H]));
				mg.hid.copy_(hidden.view([1, H]));
				mg['st' + v].pos.fill_(pos);
				mg['graph' + v].replay();

				// (a copy: the graph's output buffer is reused by the next step)
				let Dg = mg['out' + v].cuda(this.mtp.dev);

				if (this.mtp.cdev == this.mtp.dev)
					Dg = Dg.add(0.0);

				return {logits: this.lm_head.forward(Dg).view([1, 1, -1]),
					hidden: Dg.view([1, 1, H])};
			}

			let D = mtp_core(this.cfg, this.mtp, e.cuda(this.mtp.cdev),
				hidden.view([n, H]).cuda(this.mtp.cdev), dc, pos).cuda(this.mtp.dev);

			return {logits: head ? this.lm_head.forward(D).view([1, n, -1]) : null,
				hidden: D.view([1, n, H])};
		},

		mtp_chunk_ids: function (ids_dev, hidden, dc, pos, head) {
			return this.mtp_chunk(null, hidden, dc, pos, head, ids_dev);
		},

		/* Per-layer caches for max_seq positions. */
		mm_build: mm_build,

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];

				push(cache.layers, (L.kind == 'linear_attention') ?
					new_kda_state(this.cfg, 0, L.dev) :
					new_mla_cache(this.cfg, max_seq, L.dev, this.kv_bits));
			}

			if (this.mtp)
				cache.mtp = {length: 0, max_seq: max_seq,
					layers: [ new_mla_cache(this.cfg, max_seq, this.mtp.cdev, this.kv_bits) ]};

			return cache;
		},

		/* Run token ids (array) appended to the cache. Returns logits
		 * (rows, vocab) fp16 for the last `rows` positions (default 1)
		 * on the head's device. */
		forward_chunk: function (token_ids, cache, rows, trace, trace_rows) {
			let cfg = this.cfg;
			let T = length(token_ids);
			let pos = cache.length;
			let H = cfg.hidden_size;

			if (rows == null)
				rows = 1;

			// token embeddings, image rows replaced by the vision embeddings
			let h = qwen35.embed_rows(this, cache.mm, token_ids, pos).to('float32')
				.reshape([T, 1, H]).expand([T, 4, H]).contiguous();

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];

				if (i > 0 && L.dev != this.layers[i - 1].dev)
					h = h.cuda(L.dev);

				h = block_forward(cfg, L, h, cache.layers[i], pos);

				if (trace)
					push(trace, trace_rows ? h.narrow(0, T - trace_rows, trace_rows).cpu() : h.cpu());
			}

			cache.length = pos + T;

			if (rows == 0)
				return null;                 // intermediate prefill chunk: no head

			// final: mean over the streams, fp32 norm, head
			let x = h.mean(1);

			if (rows < T)
				x = x.narrow(0, T - rows, rows).contiguous();

			return this.lm_head.forward(rms_norm_f32(x, this.norm, cfg.rms_norm_eps));
		},

		graphs: function (cache, sizes) {
			if (this.mtp && cache.mtp && !getenv('AMP_GLM5_NO_MTP_GRAPH'))
				this.mtp_g = build_mtp_graph(this, cache);

			return build_graphs(this, cache, sizes);
		},

		/* Engine interface (as qwen35.prefill_pipeline): chunked prefill of
		 * ids at cache.length, one chunk per step round. opts: chunk, cuts
		 * (extra chunk boundaries), on_layer(chunk_end, li) — called for
		 * every layer after each chunk (the engine snapshots the recurrent
		 * states of a boundary chunk there). logits() are the last row. */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let cfg = this.cfg;
			let chunk = opts.chunk || 512;
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let p = start + chunk; p < start + total; p += chunk)
				push(bounds, p);

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			// stages: contiguous runs of layers on one device
			let stages = [];

			for (let i = 0; i < nl; i++) {
				if (!length(stages) || stages[length(stages) - 1].dev != this.layers[i].dev)
					push(stages, {first: i, end: i + 1, dev: this.layers[i].dev});
				else
					stages[length(stages) - 1].end = i + 1;
			}

			let ns = length(stages);
			let active = [];
			let next = 0, prev_end = start, done = 0, result = null, parts = [];

			for (let s = 0; s < ns; s++)
				push(active, null);

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				let end = bounds[next++];
				let pos = prev_end;
				let T = end - pos;
				let H = cfg.hidden_size;
				prev_end = end;
				active[0] = {pos: pos, T: T, end: end, layer: stages[0].first, ctx: null,
					h: qwen35.embed_rows(model, cache.mm, slice(ids, pos - start, end - start), pos)
						.to('float32').reshape([T, 1, H]).expand([T, 4, H]).contiguous()};
			}

			/* One round: every stage advances its chunk by one layer — all
			 * phase-A halves are enqueued first, so the GPUs work while the
			 * host reads the expert counts of the earlier stages in phase B. */
			function round() {
				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null)
						c.ctx = block_a(cfg, model.layers[c.layer], c.h,
							cache.layers[c.layer], c.pos);
				}

				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.ctx) {
						c.h = block_b(cfg, model.layers[c.layer], c.ctx);
						c.ctx = null;

						if (opts.on_layer)
							opts.on_layer(c.end, c.layer);

						c.layer++;
					}
				}

				for (let s = ns - 1; s >= 0; s--) {
					let c = active[s];

					if (!c || c.layer < stages[s].end)
						continue;

					if (s == ns - 1) {
						if (opts.hidden) {
							// post-norm hidden of every row (the MTP head's input)
							let xn = rms_norm_f32(c.h.mean(1), model.norm, cfg.rms_norm_eps);

							if (opts.on_hidden)
								opts.on_hidden(c.pos, xn.view([1, c.T, cfg.hidden_size]));
							else
								push(parts, xn);

							if (c.end == start + total)
								result = model.lm_head.forward(
									xn.narrow(0, c.T - 1, 1).contiguous());
						}
						else if (c.end == start + total) {
							let x = c.h.mean(1);

							result = model.lm_head.forward(rms_norm_f32(
								x.narrow(0, c.T - 1, 1).contiguous(), model.norm,
								cfg.rms_norm_eps));
						}

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;
					}
					else if (active[s + 1] == null) {
						c.h = c.h.cuda(stages[s + 1].dev);
						c.layer = stages[s + 1].first;
						active[s + 1] = c;
						active[s] = null;
					}
				}

				admit();
			}

			admit();

			return {
				step: function (rounds) {
					for (let r = 0; r < (rounds || 1); r++) {
						let busy = (next < length(bounds));

						for (let c in active)
							if (c) busy = true;

						if (!busy)
							break;

						round();
					}

					let more = (next < length(bounds));

					for (let c in active)
						if (c) more = true;

					return more;
				},

				done_tokens: function () {return done;},
				logits: function () {return result;},
				// (1, n, hidden) post-norm hidden of every new row (opts.hidden)
				hidden: function () {
					if (!length(parts))
						return null;

					let all = (length(parts) == 1) ? parts[0] : torch.cat(parts, 0);

					return all.view([1, all.shape()[0], cfg.hidden_size]);
				}
			};
		},

		/* Prefill ids in chunks of `chunk` tokens (default 512); returns the
		 * logits of the last `rows` positions (default 1). */
		prefill: function (token_ids, cache, chunk, rows, trace, trace_rows) {
			let n = length(token_ids);

			chunk = chunk || 512;

			for (let p = 0; p < n; p += chunk) {
				let e = (p + chunk < n) ? p + chunk : n;
				let last = (e == n);
				let r = this.forward_chunk(slice(token_ids, p, e), cache,
					last ? (rows == null ? 1 : rows) : 0, last ? trace : null, trace_rows);

				if (last)
					return r;
			}

			return null;
		}
	};

	return model;
}
