// gemma4.uc — Gemma 4 (Gemma4ForConditionalGeneration), text path: dense + MoE blocks, sliding-window and global attention.
//
// Per layer (reference: transformers modeling_gemma4, exllamav3 architecture/gemma4.py):
//   x += post_attention_norm(o_proj(attn(input_norm(x))))
//   m  = mlp(pre_feedforward_norm(x))                                  (dense gated MLP, gelu_tanh)
//   moe: h = post_feedforward_norm(post_feedforward_norm_1(m) + post_feedforward_norm_2(experts(pre_feedforward_norm_2(x))))
//        the router works on the residual x: rms(x) * scale / sqrt(hidden) -> 128 logits -> softmax -> top 8 ->
//        renormalised -> times per_expert_scale[expert]
//   x += h;  x *= layer_scalar
// Attention: q / k / v norms (v without weight), NEOX rope, NO 1/sqrt(d) scaling; 25 of 30 layers slide over the last
// 1024 tokens (8 kv heads of 256, rope theta 1e4), 5 layers are global (2 kv heads of 512, K = V before the norms, rope
// theta 1e6 on a quarter of the dims: 'proportional', the other angles are zero). Embeddings are scaled by 53 (the
// bf16 rounding of sqrt(2816)), logits are soft-capped: 30 * tanh(l / 30).
//
// The residual stream is fp32. Layers are split over the devices in contiguous runs (like dsv4.uc), prefill runs chunk
// by chunk through all of them, decode steps are eager. Sliding layers keep a ring of RING = 1024 + chunk rows and attend
// with qsa_attn over the 1024 last token slots; global layers attend in fp32 on the host-sized slabs of queries.

import * as models from './models.uc';
import * as weights from './weights.uc';
import * as glm5 from './glm5.uc';
import * as math from 'math';
import {readfile, stat} from 'fs';

const LP = 'model.language_model.layers.';
const EMBED_SCALE = 53.0;        // bf16(sqrt(2816))

export function load_config(model_dir) {
	let c = json(readfile(model_dir + '/config.json'));
	let t = c.text_config || c;
	let cfg = {
		model_dir: model_dir,
		hidden_size: t.hidden_size,
		num_hidden_layers: t.num_hidden_layers,
		layer_types: t.layer_types,
		num_q_heads: t.num_attention_heads,
		num_kv_heads: t.num_key_value_heads,
		num_global_kv_heads: t.num_global_key_value_heads || t.num_key_value_heads,
		head_dim: t.head_dim,
		global_head_dim: t.global_head_dim || t.head_dim,
		k_eq_v: t.attention_k_eq_v ? true : false,
		sliding_window: t.sliding_window,
		rms_norm_eps: t.rms_norm_eps,
		intermediate_size: t.intermediate_size,
		moe: t.enable_moe_block ? true : false,
		moe_intermediate_size: t.moe_intermediate_size || 0,
		n_routed_experts: t.num_experts || 0,
		num_experts_per_tok: t.top_k_experts || 0,
		vocab_size: t.vocab_size,
		softcap: t.final_logit_softcapping || 0.0,
		rope_local: t.rope_parameters.sliding_attention,
		rope_global: t.rope_parameters.full_attention,
		max_position: t.max_position_embeddings,
		swiglu_limit: 0.0,
		act: 'gelu',
		eos_token_id: c.eos_token_id,
		bos_token_id: t.bos_token_id,
		image_token_id: c.image_token_id,
		has_vision: (c.vision_config != null && c.image_token_id != null && stat(model_dir + '/processor_config.json') != null)
	};

	if (t.hidden_size_per_layer_input)
		die('gemma4: per-layer inputs are not implemented');

	return cfg;
}

/* RMS norm in fp32: fp32 in, fp32 out, w fp32 (1, H) */
function rms32(x, w, eps) {
	let r = x.mul(x).mean(-1, true).add(eps).rsqrt();

	return x.mul(r).mul(w);
}

/* inverse frequencies (fp32, n) for rotate-half over `hd`: the first `rot` pairs rotate with theta^(-2i / hd), the
 * rest stay (angle 0: 'proportional' rope of the global layers; sliding layers rotate all of them) */
function inv_freq(theta, hd, rot, dev) {
	let a = [];

	for (let i = 0; i < hd / 2; i++)
		push(a, (i < rot) ? 1.0 / math.pow(theta, 2.0 * i / hd) : 0.0);

	return torch.from_array(a, 'float32', 'cuda:' + dev).view([1, -1]);
}

/* ------------------------------------------------------------------ */
/* weights                                                             */

function norm32(ws, key, dev) {
	return ws.gpu(key, dev).to('float32').reshape([1, -1]).contiguous();
}

function build_layer(ws, cfg, i, dev) {
	let k = LP + i;
	let full = cfg.layer_types[i] == 'full_attention';
	let hd = full ? cfg.global_head_dim : cfg.head_dim;
	let H = cfg.hidden_size;
	let L = {
		dev: dev, full: full, hd: hd, kvh: full ? cfg.num_global_kv_heads : cfg.num_kv_heads, nh: cfg.num_q_heads,
		q: models.linear(ws, k + '.self_attn.q_proj', false, dev),
		k: models.linear(ws, k + '.self_attn.k_proj', false, dev),
		v: (full && cfg.k_eq_v) ? null : models.linear(ws, k + '.self_attn.v_proj', false, dev),
		o: models.linear(ws, k + '.self_attn.o_proj', false, dev),
		q_norm: ws.gpu16(k + '.self_attn.q_norm.weight', dev),
		k_norm: ws.gpu16(k + '.self_attn.k_norm.weight', dev),
		v_norm: torch.ones([hd], 'float16', 'cuda:' + dev),
		in_ln: ws.gpu16(k + '.input_layernorm.weight', dev),
		pre_ff: ws.gpu16(k + '.pre_feedforward_layernorm.weight', dev),
		post_attn: norm32(ws, k + '.post_attention_layernorm.weight', dev),
		post_ff: norm32(ws, k + '.post_feedforward_layernorm.weight', dev),
		mlp: glm5.build_mlp(ws, k + '.mlp', dev),
		scalar: ws.gpu(k + '.layer_scalar', dev).to('float32').reshape([1, 1]).contiguous()
	};

	if (cfg.moe) {
		let E = cfg.n_routed_experts;
		let experts = [];

		for (let e = 0; e < E; e++)
			push(experts, glm5.build_mlp(ws, `${k}.experts.${e}`, dev));

		L.post_ff1 = norm32(ws, k + '.post_feedforward_layernorm_1.weight', dev);
		L.post_ff2 = norm32(ws, k + '.post_feedforward_layernorm_2.weight', dev);
		L.pre_ff2 = ws.gpu16(k + '.pre_feedforward_layernorm_2.weight', dev);
		L.rscale = ws.gpu(k + '.router.scale', dev).to('float32').mul(1.0 / math.sqrt(H * 1.0)).reshape([1, -1]).contiguous();
		L.moe = {
			type: 'moe', dev: dev, experts: experts, router: 'gemma4', shared: null,
			gate_t: ws.gpu(k + '.router.proj.weight', dev).to('float32').t().contiguous(),          // (H, E)
			pes: ws.gpu(k + '.router.per_expert_scale', dev).to('float32').reshape([E, 1]).contiguous(),
			rl: null
		};
	}

	return L;
}

/* contiguous runs of layers over `n` devices, balanced by weight bytes (the embedding counts on the first, the head on
 * the last) */
function plan_devices(ws, cfg, n) {
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

	let total = embed + head;

	for (let b in bytes)
		total += b;

	let target = total / n;
	let devs = [], d = 0, acc = embed;

	for (let i = 0; i < nl; i++) {
		let need = bytes[i] + ((i == nl - 1) ? head : 0);

		if (d < n - 1 && acc > 0 && acc + need / 2 > target && (nl - i) > (n - 1 - d)) {
			d++;
			acc = 0;
		}

		acc += need;
		push(devs, d);
	}

	return devs;
}

/* ------------------------------------------------------------------ */
/* attention                                                           */

/* (cos, sin) (T, hd / 2) fp32 of the positions pos0 .. pos0 + T - 1 for the inverse frequencies inv (1, hd / 2) */
export function rope_cs(inv, pos0, T, dev) {
	let pos = torch.arange(T, 'float32', 'cuda:' + dev).add(pos0 * 1.0).view([T, 1]);
	let fr = pos.mul(inv);

	return {cos: fr.cos().contiguous(), sin: fr.sin().contiguous()};
}

/* ring slots of the token positions of the sliding layers: for every query row r (position pos0 + r) the cache slots of
 * its last `win` tokens, -1 before the first token: (T, win) int32 */
export function window_idx(pos0, T, win, ring, dev) {
	let dv = 'cuda:' + dev;
	let base = torch.arange(T, 'float32', dv).add(pos0 * 1.0 - (win - 1.0)).view([T, 1]);
	let tok = base.add(torch.arange(win, 'float32', dv).view([1, win]));
	let valid = tok.neg().le(0.0);
	let q = tok.mul(1.0 / ring).to('int64').to('float32');
	let slot = tok.sub(q.mul(ring * 1.0));

	return torch.where(valid, slot.to('int32'), torch.full([1], -1, 'int32', dv)).contiguous();
}

/* the one-row decode context of the position held in the device tensor posf (1) fp32: rope tables, the ring slot of
 * the new row and the window's slots, all computed on the device (CUDA-graph capture) */
export function dyn_ctx(cfg, inv, posf, ring, dev, glen) {
	let dv = 'cuda:' + dev, win = cfg.sliding_window;
	let fr = posf.view([1, 1]).mul(inv.l);
	let tok = torch.arange(win, 'float32', dv).view([1, win]).add(posf.sub(win - 1.0).view([1, 1]));
	let valid = tok.neg().le(0.0);
	let q = tok.mul(1.0 / ring).to('int64').to('float32');
	let slot = tok.sub(q.mul(ring * 1.0));
	let pq = posf.mul(1.0 / ring).to('int64').to('float32');
	let fg = posf.view([1, 1]).mul(inv.g);

	return {
		ring: ring,
		cs_l: {cos: fr.cos().contiguous(), sin: fr.sin().contiguous()},
		cs_g: {cos: fg.cos().contiguous(), sin: fg.sin().contiguous()},
		glen: glen, posf: posf,
		gslot: posf.to('int64').view([1]),
		widx: torch.where(valid, slot.to('int32'), torch.full([1], -1, 'int32', dv)).contiguous(),
		dyn: {slot: posf.sub(pq.mul(ring * 1.0)).to('int64').view([1])}
	};
}

/* global attention in fp32 for T query rows at pos0 over the cache rows [0, pos0 + T): K / V (len, kvh, hd) fp16,
 * q (T, nh, hd) fp16; scale 1. Query slabs bound the score matrix. Returns (T, nh * hd) fp16 */
function global_attn(L, q, K, V, pos0, T) {
	let nh = L.nh, kvh = L.kvh, hd = L.hd;
	let rep = int(nh / kvh);
	let len = pos0 + T;
	let dv = 'cuda:' + L.dev;
	let QB = max(8, min(T, int(100000000 / (rep * len))));
	let outs = [];

	for (let q0 = 0; q0 < T; q0 += QB) {
		let b = min(QB, T - q0);
		let qpos = torch.arange(b, 'float32', dv).add((pos0 + q0) * 1.0).view([b, 1]);
		let kpos = torch.arange(len, 'float32', dv).view([1, len]);
		let causal = qpos.sub(kpos).neg().le(0.0);               // kpos <= qpos
		let o = [];

		for (let j = 0; j < kvh; j++) {
			// the rep query heads of kv head j, as rows (rep * b, hd), head-major
			let qj = q.narrow(0, q0, b).narrow(1, j * rep, rep).transpose(0, 1).reshape([rep * b, hd]).to('float32');
			let kj = K.narrow(1, j, 1).reshape([len, hd]).to('float32');
			let vj = V.narrow(1, j, 1).reshape([len, hd]).to('float32');
			let s = torch.matmul(qj, kj.t()).view([rep, b, len]);

			s = torch.where(causal.view([1, b, len]), s, torch.full([1], -1e30, 'float32', dv)).view([rep * b, len]);
			push(o, torch.matmul(s.softmax(-1), vj).view([rep, b, hd]));
		}

		// (kvh, rep, b, hd) -> (b, nh * hd)
		push(outs, torch.cat(o, 0).view([nh, b, hd]).transpose(0, 1).reshape([b, nh * hd]).to('float16'));
	}

	return (length(outs) == 1) ? outs[0] : torch.cat(outs, 0);
}

/* the one-row global attention over the first glen cache rows with the position in the device tensor posf */
function global_attn_dyn(L, q, K, V, posf, glen) {
	let nh = L.nh, kvh = L.kvh, hd = L.hd;
	let rep = int(nh / kvh);
	let dv = 'cuda:' + L.dev;
	let kpos = torch.arange(glen, 'float32', dv).view([1, glen]);
	let late = kpos.sub(posf.view([1, 1])).neg().le(-0.5);           // kpos > pos
	let o = [];

	for (let j = 0; j < kvh; j++) {
		let qj = q.narrow(1, j * rep, rep).reshape([rep, hd]).to('float32');
		let kj = K.narrow(1, j, 1).reshape([glen, hd]).to('float32');
		let vj = V.narrow(1, j, 1).reshape([glen, hd]).to('float32');
		let sc = torch.matmul(qj, kj.t());

		sc = torch.where(late, torch.full([1], -1e30, 'float32', dv), sc);
		push(o, torch.matmul(sc.softmax(-1), vj));
	}

	return torch.cat(o, 0).view([1, nh * hd]).to('float16');
}

/* rows [a, b) of a ring of `ring` rows (position p lives in slot p mod ring) as one tensor */
function ring_rows(t, a, b, ring) {
	let s0 = a % ring, n = b - a;

	if (s0 + n <= ring)
		return t.narrow(0, s0, n);

	return torch.cat([ t.narrow(0, s0, ring - s0), t.narrow(0, 0, n - (ring - s0)) ], 0);
}

/* sliding-window attention of a prefill chunk (T rows) as fused sdpa with an explicit window mask over the keys
 * [pos0 - (win - 1), pos0 + T): K / V heads expanded to the q heads, scale 1 */
function window_attn(cfg, L, q, cl, pos0, T, ring, vspans) {
	let nh = L.nh, kvh = L.kvh, hd = L.hd, win = cfg.sliding_window;
	let rep = int(nh / kvh);
	let dv = 'cuda:' + L.dev;
	let kst = max(0, pos0 - (win - 1));
	let len = pos0 + T - kst;

	function heads(t) {
		return ring_rows(t, kst, pos0 + T, ring).unsqueeze(2).expand([len, kvh, rep, hd]).reshape([len, nh, hd])
			.transpose(0, 1).unsqueeze(0);
	}

	let qpos = torch.arange(T, 'float32', dv).add(pos0 * 1.0).view([T, 1]);
	let kpos = torch.arange(len, 'float32', dv).add(kst * 1.0).view([1, len]);
	let d = qpos.sub(kpos);
	let acc = d.neg().le(0.0).to('float32').mul(d.le(win - 1.0).to('float32'));

	// the rows of one image attend to each other in both directions (sliding layers only)
	for (let sp in (vspans || [])) {
		let inq = qpos.sub(sp[0] * 1.0).neg().le(0.0).to('float32').mul(qpos.sub(sp[1] - 1.0).le(0.0).to('float32'));
		let ink = kpos.sub(sp[0] * 1.0).neg().le(0.0).to('float32').mul(kpos.sub(sp[1] - 1.0).le(0.0).to('float32'));

		acc = acc.add(inq.mul(ink));
	}

	let ok = acc.neg().le(-0.5);
	let o = torch.sdpa(q.transpose(0, 1).unsqueeze(0), heads(cl.k), heads(cl.v), false, ok, false, 1.0);

	return o.transpose(1, 2).reshape([T, nh * hd]);
}

export function attn_forward(cfg, L, a, cl, pos0, ctx) {
	let T = a.shape()[0];
	let nh = L.nh, kvh = L.kvh, hd = L.hd, eps = cfg.rms_norm_eps;
	let cs = L.full ? ctx.cs_g : ctx.cs_l;
	let q = torch.qk_norm_rope(L.q.forward(a).view([1, T, nh, hd]), L.q_norm, cs.cos, cs.sin, hd, eps);
	let kraw = L.k.forward(a);
	let vraw = L.v ? L.v.forward(a) : kraw;                       // global layers: V = the K projection before its norm
	let kk = torch.qk_norm_rope(kraw.view([1, T, kvh, hd]), L.k_norm, cs.cos, cs.sin, hd, eps);
	let vv = torch.rms_norm(vraw.contiguous().view([T * kvh, hd]), L.v_norm, eps);

	if (L.full && ctx.dyn) {
		cl.k.index_copy_(0, ctx.gslot, kk.view([1, kvh, hd]));
		cl.v.index_copy_(0, ctx.gslot, vv.view([1, kvh, hd]));

		return L.o.forward(global_attn_dyn(L, q.view([1, nh, hd]), cl.k.narrow(0, 0, ctx.glen),
			cl.v.narrow(0, 0, ctx.glen), ctx.posf, ctx.glen));
	}

	if (L.full) {
		cl.k.narrow(0, pos0, T).copy_(kk.view([T, kvh, hd]));
		cl.v.narrow(0, pos0, T).copy_(vv.view([T, kvh, hd]));

		let len = pos0 + T;
		let o = global_attn(L, q.view([T, nh, hd]), cl.k.narrow(0, 0, len), cl.v.narrow(0, 0, len), pos0, T);

		return L.o.forward(o);
	}

	// graph-captured decode step: the position is a device tensor (ctx.dyn)
	if (ctx.dyn) {
		cl.k.index_copy_(0, ctx.dyn.slot, kk.view([1, kvh, hd]));
		cl.v.index_copy_(0, ctx.dyn.slot, vv.view([1, kvh, hd]));

		let o = torch.qsa_attn(q.view([1, nh, hd]).contiguous(), cl.k, cl.v, ctx.widx, 1.0);

		return L.o.forward(o.view([1, nh * hd]));
	}

	// sliding layer: the chunk's rows go to their ring slots (the chunk never exceeds RING - window rows), then qsa_attn
	// over the window's slots; scale 1
	let ring = ctx.ring;
	let s0 = pos0 % ring;

	if (s0 + T <= ring) {
		cl.k.narrow(0, s0, T).copy_(kk.view([T, kvh, hd]));
		cl.v.narrow(0, s0, T).copy_(vv.view([T, kvh, hd]));
	}
	else {
		let n1 = ring - s0;

		cl.k.narrow(0, s0, n1).copy_(kk.view([T, kvh, hd]).narrow(0, 0, n1));
		cl.v.narrow(0, s0, n1).copy_(vv.view([T, kvh, hd]).narrow(0, 0, n1));
		cl.k.narrow(0, 0, T - n1).copy_(kk.view([T, kvh, hd]).narrow(0, n1, T - n1));
		cl.v.narrow(0, 0, T - n1).copy_(vv.view([T, kvh, hd]).narrow(0, n1, T - n1));
	}

	// prefill chunks: fused sdpa with a window mask; decode / verify rows: the gathered kernel over the window's slots
	if ((T >= 64 || ctx.vspans) && !getenv('AMP_GEMMA4_NO_SDPA'))
		return L.o.forward(window_attn(cfg, L, q.view([T, nh, hd]), cl, pos0, T, ring, ctx.vspans));

	let o = torch.qsa_attn(q.view([T, nh, hd]).contiguous(), cl.k, cl.v, ctx.widx, 1.0);

	return L.o.forward(o.view([T, nh * hd]));
}

/* ------------------------------------------------------------------ */
/* block                                                               */

/* AMP_GEMMA4_HP=1: host seconds per phase of the layers (export HP) */
export let HP = getenv('AMP_GEMMA4_HP') ? {} : null;
let hp_t = 0;
let hp_off = false;
let HP_SYNC = getenv('AMP_GEMMA4_HPSYNC');

function hp(name) {
	if (!HP)
		return;

	if (HP_SYNC && !hp_off)
		torch.synchronize();

	let c = clock(true), t = c[0] + c[1] / 1e9;

	if (name != null)
		HP[name] = (HP[name] || 0) + t - hp_t;

	hp_t = t;
}

/* x (T, H) fp32 -> (T, H) fp32 */
function block_forward(cfg, L, x, cl, pos0, ctx) {
	let eps = cfg.rms_norm_eps;
	let T = x.shape()[0];

	hp(null);

	// attention
	let a = glm5.rms_norm_f32(x, L.in_ln, eps);
	let ao = attn_forward(cfg, L, a, cl, pos0, ctx);

	hp('attn');

	x = torch.norm32(ao, L.post_attn, eps, x);

	// feed-forward: the dense MLP and (every layer of this model) the routed experts side by side
	let f = glm5.rms_norm_f32(x, L.pre_ff, eps);
	let m = glm5.mlp_forward(cfg, L.mlp, f);
	let h;

	hp('mlp');

	if (L.moe) {
		// the router sees the residual: rms-normalised, scaled by the router scale / sqrt(hidden)
		let rin = torch.norm32(x, L.rscale, eps);

		hp('r.norm');

		L.moe.rl = torch.matmul(rin, L.moe.gate_t);

		hp('r.mm');

		let y2 = glm5.rms_norm_f32(x, L.pre_ff2, eps);

		hp('r.y2');

		let mo = glm5.moe_forward(cfg, L.moe, y2);

		hp('moe');

		L.moe.rl = null;
		h = torch.norm32(mo, L.post_ff2, eps, torch.norm32(m, L.post_ff1, eps));
	}
	else
		h = m;

	return torch.norm32(h, L.post_ff, eps, x, L.scalar);
}

/* ------------------------------------------------------------------ */
/* model                                                               */

/* ------------------------------------------------------------------ */
/* cache state: the window aux state, the host-RAM store, hot reload   */
/*                                                                     */
/* The sliding layers keep a ring: a state at length n is fully described by the last min(n, window) rows (position   */
/* order), which is the aux state (a snapshot of the live ring: the rows survive the generation that follows, a       */
/* rewind to the snapshot writes them back). The global layers keep rows [0, n) of their cache.                       */

/* the per-rank cache entries of a layer: one object, or an array under tensor parallelism */
function cl_list(c) {
	return (type(c) == 'array') ? c : [ c ];
}

function is_sliding(cfg, i) {
	return cfg.layer_types[i] != 'full_attention';
}

/* the rows (w, kvh, hd) of the positions [a, a + w) from / into a ring of `ring` rows */
function ring_read(t, a, w, ring) {
	return ring_rows(t, a, a + w, ring).add(0.0);
}

function ring_write(t, a, rows, ring) {
	let w = rows.shape()[0], s0 = a % ring;

	if (s0 + w <= ring)
		t.narrow(0, s0, w).copy_(rows);
	else {
		t.narrow(0, s0, ring - s0).copy_(rows.narrow(0, 0, ring - s0));
		t.narrow(0, 0, w - (ring - s0)).copy_(rows.narrow(0, ring - s0, w - (ring - s0)));
	}
}

/* the aux state: {n, layers: [per layer: null (global) | per rank {k, v}: the last min(n, window) rows]} (device copies) */
export function window_aux(cfg, cache, ring) {
	let n = cache.length, w = min(n, cfg.sliding_window);
	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let a = null;

		if (is_sliding(cfg, i)) {
			a = map(cl_list(cache.layers[i]), (cl) => ({k: ring_read(cl.k, n - w, w, ring), v: ring_read(cl.v, n - w, w, ring)}));

			if (type(cache.layers[i]) != 'array')
				a = a[0];
		}

		push(layers, a);
	}

	return {n: n, layers: layers};
}

/* write the window rows of aux back into the rings (the cache length is the caller's) */
export function window_restore(cfg, cache, aux, ring) {
	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		if (!is_sliding(cfg, i))
			continue;

		let cls = cl_list(cache.layers[i]), ax = cl_list(aux.layers[i]);

		for (let r = 0; r < length(cls); r++) {
			let w = ax[r].k.shape()[0];

			ring_write(cls[r].k, aux.n - w, ax[r].k.cuda(cls[r].dev), ring);
			ring_write(cls[r].v, aux.n - w, ax[r].v.cuda(cls[r].dev), ring);
		}
	}
}

/* bytes of a stored state of n tokens */
export function state_bytes(cfg, cache, n) {
	let b = 0;

	for (let i = 0; i < cfg.num_hidden_layers; i++)
		for (let cl in cl_list(cache.layers[i])) {
			let sh = cl.k.shape();

			b += 4 * sh[1] * sh[2] * (is_sliding(cfg, i) ? min(n, cfg.sliding_window) : n);
		}

	return b;
}

/* a value of an aux tree (tensors, arrays, objects) with every tensor passed through f */
function aux_map(v, f) {
	if (type(v) == 'resource')
		return f(v);

	if (type(v) == 'array')
		return map(v, (x) => aux_map(x, f));

	if (type(v) == 'object') {
		let r = {};

		for (let k in v)
			r[k] = aux_map(v[k], f);

		return r;
	}

	return v;
}

/* the store entry of the live state of n tokens: put(t) turns a device tensor into the entry's copy */
export function state_export(cfg, cache, n, put, snap_aux, ring) {
	let aux = window_aux(cfg, cache, ring);
	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let cls = cl_list(cache.layers[i]);

		push(layers, map(cls, (cl) => is_sliding(cfg, i) ? null : {k: put(cl.k.narrow(0, 0, n)), v: put(cl.v.narrow(0, 0, n))}));
	}

	return {n: n, layers: layers, win: aux_map(aux.layers, put),
		snap: snap_aux ? {n: snap_aux.n, layers: aux_map(snap_aux.layers, put)} : null};
}

/* an aux tree of host tensors on the devices of the cache's layers */
export function aux_to_device(cfg, cache, aux) {
	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let a = aux.layers[i];

		if (a == null) {
			push(layers, null);
			continue;
		}

		let cls = cl_list(cache.layers[i]), ax = cl_list(a);
		let m = map(ax, (x, r) => aux_map(x, (t) => t.cuda(cls[r].dev)));

		push(layers, (type(a) == 'array') ? m : m[0]);
	}

	return {n: aux.n, layers: layers};
}

/* back into the live buffers up to `start` tokens: start == em.n restores the live state, else the snapshot's */
export function state_import(cfg, cache, em, start, live, ring) {
	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		if (is_sliding(cfg, i))
			continue;

		let cls = cl_list(cache.layers[i]);

		for (let r = 0; r < length(cls); r++) {
			cls[r].k.narrow(0, 0, start).copy_(em.layers[i][r].k.narrow(0, 0, start));
			cls[r].v.narrow(0, 0, start).copy_(em.layers[i][r].v.narrow(0, 0, start));
		}
	}

	let src = live ? {n: em.n, layers: em.win} : em.snap;

	window_restore(cfg, cache, src, ring);
	cache.length = start;

	return {snap: em.snap};
}

/* ------------------------------------------------------------------ */
/* multimodal rows                                                     */

/* the image spans of a prompt: images = [{start, emb (n, hidden) fp16}] -> {spans: [{start, n, emb}]} */
export function mm_build(cfg, images) {
	if (!images || !length(images))
		return null;

	let spans = [], prev_end = 0;

	for (let im in images) {
		let n = im.emb ? im.emb.shape()[0] : im.n;

		if (im.start < prev_end)
			die('gemma4: overlapping or unordered image spans');

		push(spans, {start: im.start, n: n, emb: im.emb});
		prev_end = im.start + n;
	}

	return {spans: spans};
}

/* [start, end) pairs of the image spans that touch the rows [pos0, pos0 + n) (null: none) */
export function mm_spans(mm, pos0, n) {
	if (!mm)
		return null;

	let out = [];

	for (let sp in mm.spans)
		if (sp.start < pos0 + n && sp.start + sp.n > pos0)
			push(out, [ sp.start, sp.start + sp.n ]);

	return length(out) ? out : null;
}

/* embeddings (n, hidden) fp32 of the token ids at pos0 on device dev: the text rows scaled, the image rows (marker ids
 * >= 2^30) replaced by the tower's embeddings, which are not scaled */
export function embed_rows(embed_t, token_ids, pos0, mm, dev) {
	let n = length(token_ids);
	let ids = mm ? map(token_ids, (t) => (t >= 1073741824) ? 0 : t) : token_ids;
	let x = embed_t.index(torch.from_array(ids, 'int64', 'cuda:' + dev)).to('float32').mul(EMBED_SCALE);

	for (let sp in (mm ? mm.spans : [])) {
		let a = max(sp.start, pos0), b = min(sp.start + sp.n, pos0 + n);

		if (b > a)
			x.narrow(0, a - pos0, b - a).copy_(sp.emb.narrow(0, a - sp.start, b - a).to('float32').cuda(dev));
	}

	return x;
}

/* The chunk boundaries (sorted, the last one the end) of a prefill of `total` rows at `start`: every `chunk` rows plus the
 * extra cuts, with no boundary inside an image span (its rows attend to each other, so one chunk must hold them all)
 * unless the chunk would outgrow `cap` rows. */
export function plan_bounds(start, total, chunk, cuts, mm, cap) {
	let end = start + total;
	let bounds = [];

	for (let p = start + chunk; p < end; p += chunk)
		push(bounds, p);

	for (let c in (cuts || []))
		if (c > start && c < end && index(bounds, c) < 0)
			push(bounds, c);

	for (let sp in (mm ? mm.spans : []))
		for (let c in [ sp.start, sp.start + sp.n ])
			if (c > start && c < end && index(bounds, c) < 0)
				push(bounds, c);

	bounds = sort(bounds, (a, b) => a - b);

	if (mm) {
		let keep = [], prev = start;

		for (let b in bounds) {
			let inside = false;

			for (let sp in mm.spans)
				if (sp.start < b && b < sp.start + sp.n)
					inside = true;

			if (inside) {
				// move on to the end of the span unless that makes the chunk too long
				let e = b;

				for (let sp in mm.spans)
					if (sp.start < b && b < sp.start + sp.n)
						e = sp.start + sp.n;

				if (e - prev > cap)
					inside = false;
			}

			if (!inside) {
				push(keep, b);
				prev = b;
			}
		}

		bounds = keep;
	}

	push(bounds, end);

	return bounds;
}

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = cfg.num_hidden_layers;
	let ndev = opts.devices || 1;
	let devs = (type(ndev) == 'array') ? ndev : plan_devices(ws, cfg, ndev);
	let first = devs[0], last = devs[nl - 1];
	let chunk_cap = max(opts.prefill_rows || 1024, 1024);

	glm5.set_fused_rows(max(256, chunk_cap));

	ws.upload(function (key) {
		if (index(key, LP) == 0) {
			let rest = substr(key, length(LP));
			let idx = +substr(rest, 0, index(rest, '.'));

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (index(key, 'model.language_model.embed_tokens') == 0)
			return first;

		if (index(key, 'lm_head') == 0 || index(key, 'model.language_model.norm') == 0)
			return last;

		return null;       // vision tower / projector: not part of the text model
	});

	let layers = [];

	for (let i = 0; i < nl; i++)
		push(layers, build_layer(ws, cfg, i, devs[i]));

	// readback-free MoE (every expert holds at most one row per token of a chunk): no host stall, so the stages overlap
	if (!getenv('AMP_GEMMA4_SYNC_MOE'))
		for (let L in layers)
			if (L.moe)
				L.moe.async_moe = true;

	let embed_t = ws.gpu16('model.language_model.embed_tokens.weight', first);

	ws.release('model.language_model.embed_tokens.weight', first);

	let ring = cfg.sliding_window + chunk_cap;
	let invs = {};          // per device: {l: local inverse frequencies, g: global}

	for (let d in devs)
		if (!invs[d])
			invs[d] = {
				l: inv_freq(cfg.rope_local.rope_theta, cfg.head_dim, cfg.head_dim / 2, d),
				g: inv_freq(cfg.rope_global.rope_theta, cfg.global_head_dim,
					int((cfg.rope_global.partial_rotary_factor || 1.0) * cfg.global_head_dim / 2), d)
			};

	let model = {
		cfg: cfg, devs: devs, ws: ws, layers: layers, embed_tokens: embed_t, ring: ring, invs: invs, chunk_cap: chunk_cap, mm_build: mm_build,
		norm: ws.gpu16('model.language_model.norm.weight', last),
		lm_head: models.linear(ws, 'lm_head', false, last),

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i = 0; i < nl; i++) {
				let L = layers[i];
				let dv = 'cuda:' + L.dev;
				let rows = L.full ? max_seq : ring;

				push(cache.layers, {
					dev: L.dev, type: cfg.layer_types[i],
					k: torch.zeros([rows, L.kvh, L.hd], 'float16', dv),
					v: torch.zeros([rows, L.kvh, L.hd], 'float16', dv)
				});
			}

			return cache;
		},

		/* Engine hooks (snapshot of the sliding rings, host-RAM store, hot reload): see window_aux */
		aux_get: function (cache) {
			return window_aux(cfg, cache, ring);
		},

		aux_set: function (cache, aux) {
			window_restore(cfg, cache, aux, ring);
		},

		cache_bytes: function (cache, n) {
			return state_bytes(cfg, cache, n);
		},

		cache_export: function (cache, n, put, snap_aux, dst) {
			return state_export(cfg, cache, n, put, snap_aux, ring);
		},

		cache_import: function (cache, em, start, live) {
			let r = state_import(cfg, cache, em, start, live, ring);

			return {snap: r.snap ? aux_to_device(cfg, cache, r.snap) : null};
		},

		/* The token ids appended to the cache; returns the logits (rows, vocab) fp16 of the last `rows` positions
		 * (default 1; 0: none). A chunk longer than the ring's slack is cut. */
		forward_chunk: function (token_ids, cache, rows) {
			if (rows == null)
				rows = 1;

			let n = length(token_ids);

			if (n > chunk_cap) {
				let r = null;

				for (let p = 0; p < n; p += chunk_cap)
					r = this.forward_chunk(slice(token_ids, p, min(p + chunk_cap, n)), cache,
						(p + chunk_cap >= n) ? rows : 0);

				return r;
			}

			let pos0 = cache.length;
			let x = embed_rows(embed_t, token_ids, pos0, cache.mm, first);
			let vsp = mm_spans(cache.mm, pos0, n);
			let dev = first;
			let ctx = null;

			for (let i = 0; i < nl; i++) {
				let L = layers[i];

				if (L.dev != dev || ctx == null) {
					if (L.dev != dev)
						x = x.cuda(L.dev);

					dev = L.dev;
					ctx = {
						ring: ring,
						cs_l: rope_cs(invs[dev].l, pos0, n, dev),
						cs_g: rope_cs(invs[dev].g, pos0, n, dev),
						vspans: vsp,
						widx: ((n < 64 && !vsp) || getenv('AMP_GEMMA4_NO_SDPA')) ? window_idx(pos0, n, cfg.sliding_window, ring, dev) : null
					};
				}

				x = block_forward(cfg, L, x, cache.layers[i], pos0, ctx);
			}

			cache.length = pos0 + n;

			if (rows == 0)
				return null;

			x = (rows < n) ? x.narrow(0, n - rows, rows).contiguous() : x;

			let lg = this.lm_head.forward(glm5.rms_norm_f32(x, this.norm, cfg.rms_norm_eps));

			if (cfg.softcap > 0) {
				// soft cap: c * tanh(l / c)
				let e = lg.to('float32').mul(2.0 / cfg.softcap).exp();

				lg = e.sub(1.0).div(e.add(1.0)).mul(cfg.softcap).to('float16');
			}

			return lg;
		},

		/* CUDA-graph decode (one row): runs of sliding layers on one device are captured as one graph each (the position is
		 * a device tensor); the global layers run eagerly between them (their attention length is a host value). */
		graphs: function (cache, sizes) {
			if (getenv('AMP_GEMMA4_NO_GRAPHS'))
				return null;

			let model = this;
			let H = cfg.hidden_size;
			let runs = [];
			let posf = {};
			let gmax = cache.max_seq;

			for (let i = 0; i < nl; i++) {
				if (!length(runs) || runs[length(runs) - 1].dev != layers[i].dev)
					push(runs, {dev: layers[i].dev, first: i, end: i + 1, gr: {}});
				else
					runs[length(runs) - 1].end = i + 1;

				if (posf[layers[i].dev] == null)
					posf[layers[i].dev] = torch.zeros([1], 'float32', 'cuda:' + layers[i].dev);
			}

			for (let r in runs)
				r.h_in = torch.zeros([1, H], 'float32', 'cuda:' + r.dev);

			function body(r, glen) {
				let ctx = dyn_ctx(cfg, invs[r.dev], posf[r.dev], ring, r.dev, glen);
				let h = r.h_in;

				for (let i = r.first; i < r.end; i++)
					h = block_forward(cfg, layers[i], h, cache.layers[i], 0, ctx);

				return h;
			}

			// the attention length of the global layers is the graph's fixed shape: the smallest power of two >= pos + 1
			function bucket(pos) {
				let b = 256;

				while (b < pos + 1)
					b *= 2;

				return min(b, gmax);
			}

			function graph_of(r, glen) {
				let e = r.gr[glen];

				if (e)
					return e;

				// the warm-up runs write the rows of this very position (the replay rewrites them)
				hp_off = true;
				body(r, glen);
				torch.synchronize();

				e = {graph: torch.graph()};
				e.graph.capture_begin(r.dev);
				e.h_out = body(r, glen);
				e.graph.capture_end();
				torch.synchronize();
				hp_off = false;
				r.gr[glen] = e;

				return e;
			}

			function run_ids(tids, pos) {
				for (let d in posf)
					posf[d].copy_(torch.from_array([ pos * 1.0 ], 'float32', 'cuda:' + d));

				hp(null);

				let h = embed_t.index(tids).to('float32').mul(EMBED_SCALE);
				let glen = bucket(pos);

				hp('d.embed');

				for (let r in runs) {
					if (r.dev != first || r != runs[0])
						h = h.cuda(r.dev);

					let e = graph_of(r, glen);

					r.h_in.copy_(h);
					e.graph.replay();
					h = e.h_out;
					hp('d.graph');
				}

				cache.length = pos + 1;

				let lg = model.lm_head.forward(glm5.rms_norm_f32(h, model.norm, cfg.rms_norm_eps));

				if (cfg.softcap > 0) {
					let e = lg.to('float32').mul(2.0 / cfg.softcap).exp();

					lg = e.sub(1.0).div(e.add(1.0)).mul(cfg.softcap).to('float16');
				}

				hp('d.head');

				return {logits: lg, hidden: null};
			}

			// capture the first bucket now (a real position is not needed: nothing is read from the cache before a write)
			for (let r in runs)
				graph_of(r, bucket(0));

			return {
				has: function (S) {return S == 1;},
				run: function (ids, pos) {return run_ids(torch.from_array(ids, 'int64', 'cuda:' + first), pos);},
				run_dev: function (ids_dev, pos) {return run_ids(ids_dev.view([1]), pos);}
			};
		},

		/* The first stage's work on a chunk: embedding + its layers; stage_run() continues on the next device. */
		stage_begin: function (token_ids, pos0, mm) {
			return {pos0: pos0, n: length(token_ids), x: embed_rows(embed_t, token_ids, pos0, mm, first), dev: first, ctx: null,
				vspans: mm_spans(mm, pos0, length(token_ids))};
		},

		/* chunk c arrives at the stage that starts at layer i0: moves to its device, builds the per-device tables */
		stage_enter: function (c, i0) {
			let dev = layers[i0].dev;

			if (c.dev != dev) {
				hp(null);
				c.x = c.x.cuda(dev);
				c.dev = dev;
				c.ctx = null;
				hp('xfer');
			}

			if (!c.ctx)
				c.ctx = {
					ring: ring,
					cs_l: rope_cs(invs[dev].l, c.pos0, c.n, dev),
					cs_g: rope_cs(invs[dev].g, c.pos0, c.n, dev),
					vspans: c.vspans,
					widx: ((c.n < 64 && !c.vspans) || getenv('AMP_GEMMA4_NO_SDPA')) ? window_idx(c.pos0, c.n, cfg.sliding_window, ring, dev) : null
				};
		},

		/* the layers [i0, i1) (all on one device) of chunk c */
		stage_run: function (c, cache, i0, i1) {
			this.stage_enter(c, i0);

			for (let i = i0; i < i1; i++)
				c.x = block_forward(cfg, layers[i], c.x, cache.layers[i], c.pos0, c.ctx);
		},

		/* the logits (1, vocab) of the last row of a finished chunk */
		stage_head: function (c) {
			let x = c.x.narrow(0, c.n - 1, 1).contiguous();
			let lg = this.lm_head.forward(glm5.rms_norm_f32(x, this.norm, cfg.rms_norm_eps));

			if (cfg.softcap > 0) {
				let e = lg.to('float32').mul(2.0 / cfg.softcap).exp();

				lg = e.sub(1.0).div(e.add(1.0)).mul(cfg.softcap).to('float16');
			}

			return lg;
		},

		/* Engine interface: chunked prefill of ids at cache.length as a wavefront over the device runs: every round each
		 * stage advances its chunk, so the GPUs work on consecutive chunks at the same time (the MoE needs no host read). */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let chunk = min(opts.chunk || 1024, chunk_cap);
			let start = cache.length, total = length(ids);
			let bounds = plan_bounds(start, total, chunk, opts.cuts, cache.mm, chunk_cap);

			// contiguous layer runs per device
			let stages = [];

			for (let i = 0; i < nl; i++)
				if (!length(stages) || stages[length(stages) - 1].dev != layers[i].dev)
					push(stages, {i0: i, i1: i + 1, dev: layers[i].dev});
				else
					stages[length(stages) - 1].i1 = i + 1;

			let ns = length(stages);
			let active = map(stages, () => null);
			let next = 0, prev = start, done = 0, result = null;
			let snap_at = (opts.cuts && length(opts.cuts)) ? opts.cuts[0] : 0;

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				// a snapshot boundary is taken at the chunk end: nothing may be in flight behind it
				if (snap_at && prev == snap_at)
					for (let c in active)
						if (c)
							return;

				let end = bounds[next++];
				let c = model.stage_begin(slice(ids, prev - start, end - start), prev, cache.mm);

				c.end = end;
				prev = end;
				active[0] = c;
			}

			function round() {
				// the stages' layers are issued round-robin: a stage issued whole blocks the host on its full launch
				// queue and the other stages' GPUs run dry meanwhile
				let n = 0;

				for (let s = ns - 1; s >= 0; s--)
					if (active[s]) {
						model.stage_enter(active[s], stages[s].i0);
						n = max(n, stages[s].i1 - stages[s].i0);
					}

				for (let j = 0; j < n; j++)
					for (let s = ns - 1; s >= 0; s--) {
						let c = active[s], i = stages[s].i0 + j;

						if (c && i < stages[s].i1)
							c.x = block_forward(cfg, layers[i], c.x, cache.layers[i], c.pos0, c.ctx);
					}

				for (let s = ns - 1; s >= 0; s--) {
					let c = active[s];

					if (!c)
						continue;

					if (s == ns - 1) {
						if (c.end == start + total)
							result = model.stage_head(c);

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;

						if (opts.on_aux)
							opts.on_aux(c.end);
					}
					else {
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
							if (c)
								busy = true;

						if (!busy)
							break;

						round();
					}

					let more = (next < length(bounds));

					for (let c in active)
						if (c)
							more = true;

					return more;
				},

				done_tokens: function () {return done;},
				logits: function () {return result;},
				hidden: function () {return null;}
			};
		}
	};

	return model;
}
