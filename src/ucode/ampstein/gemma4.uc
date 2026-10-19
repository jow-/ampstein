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
import {readfile} from 'fs';

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
		image_token_id: c.image_token_id
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
function rope_cs(inv, pos0, T, dev) {
	let pos = torch.arange(T, 'float32', 'cuda:' + dev).add(pos0 * 1.0).view([T, 1]);
	let fr = pos.mul(inv);

	return {cos: fr.cos().contiguous(), sin: fr.sin().contiguous()};
}

/* ring slots of the token positions of the sliding layers: for every query row r (position pos0 + r) the cache slots of
 * its last `win` tokens, -1 before the first token: (T, win) int32 */
function window_idx(pos0, T, win, ring, dev) {
	let dv = 'cuda:' + dev;
	let base = torch.arange(T, 'float32', dv).add(pos0 * 1.0 - (win - 1.0)).view([T, 1]);
	let tok = base.add(torch.arange(win, 'float32', dv).view([1, win]));
	let valid = tok.neg().le(0.0);
	let q = tok.mul(1.0 / ring).to('int64').to('float32');
	let slot = tok.sub(q.mul(ring * 1.0));

	return torch.where(valid, slot.to('int32'), torch.full([1], -1, 'int32', dv)).contiguous();
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

function attn_forward(cfg, L, a, cl, pos0, ctx) {
	let T = a.shape()[0];
	let nh = L.nh, kvh = L.kvh, hd = L.hd, eps = cfg.rms_norm_eps;
	let cs = L.full ? ctx.cs_g : ctx.cs_l;
	let q = torch.qk_norm_rope(L.q.forward(a).view([1, T, nh, hd]), L.q_norm, cs.cos, cs.sin, hd, eps);
	let kraw = L.k.forward(a);
	let vraw = L.v ? L.v.forward(a) : kraw;                       // global layers: V = the K projection before its norm
	let kk = torch.qk_norm_rope(kraw.view([1, T, kvh, hd]), L.k_norm, cs.cos, cs.sin, hd, eps);
	let vv = torch.rms_norm(vraw.contiguous().view([T * kvh, hd]), L.v_norm, eps);

	if (L.full) {
		cl.k.narrow(0, pos0, T).copy_(kk.view([T, kvh, hd]));
		cl.v.narrow(0, pos0, T).copy_(vv.view([T, kvh, hd]));

		let len = pos0 + T;
		let o = global_attn(L, q.view([T, nh, hd]), cl.k.narrow(0, 0, len), cl.v.narrow(0, 0, len), pos0, T);

		return L.o.forward(o);
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

	let o = torch.qsa_attn(q.view([T, nh, hd]).contiguous(), cl.k, cl.v, ctx.widx, 1.0);

	return L.o.forward(o.view([T, nh * hd]));
}

/* ------------------------------------------------------------------ */
/* block                                                               */

/* x (T, H) fp32 -> (T, H) fp32 */
function block_forward(cfg, L, x, cl, pos0, ctx) {
	let eps = cfg.rms_norm_eps;
	let T = x.shape()[0];

	// attention
	let a = glm5.rms_norm_f32(x, L.in_ln, eps);
	let ao = attn_forward(cfg, L, a, cl, pos0, ctx);

	x = x.add(rms32(ao.to('float32'), L.post_attn, eps));

	// feed-forward: the dense MLP and (every layer of this model) the routed experts side by side
	let f = glm5.rms_norm_f32(x, L.pre_ff, eps);
	let m = glm5.mlp_forward(cfg, L.mlp, f).to('float32');
	let h;

	if (L.moe) {
		// the router sees the residual: rms-normalised, scaled by the router scale / sqrt(hidden)
		let rin = rms32(x, L.rscale, eps);

		L.moe.rl = torch.matmul(rin, L.moe.gate_t);

		let y2 = glm5.rms_norm_f32(x, L.pre_ff2, eps);
		let mo = glm5.moe_forward(cfg, L.moe, y2);

		L.moe.rl = null;
		h = rms32(m, L.post_ff1, eps).add(rms32(mo, L.post_ff2, eps));
	}
	else
		h = m;

	x = x.add(rms32(h, L.post_ff, eps));

	return x.mul(L.scalar);
}

/* ------------------------------------------------------------------ */
/* model                                                               */

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
		cfg: cfg, devs: devs, ws: ws, layers: layers, embed_tokens: embed_t, ring: ring,
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
			let ids = torch.from_array(token_ids, 'int64', 'cuda:' + first);
			let x = embed_t.index(ids).to('float32').mul(EMBED_SCALE);
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
						widx: window_idx(pos0, n, cfg.sliding_window, ring, dev)
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

		/* Engine interface: chunked prefill of ids at cache.length. */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let chunk = min(opts.chunk || 1024, chunk_cap);
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let p = start + chunk; p < start + total; p += chunk)
				push(bounds, p);

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			let next = 0, prev = start, done = 0, result = null;

			return {
				step: function (rounds) {
					for (let r = 0; r < (rounds || 1); r++) {
						if (next >= length(bounds))
							break;

						let end = bounds[next++];
						let last_c = (end == start + total);
						let lg = model.forward_chunk(slice(ids, prev - start, end - start), cache, last_c ? 1 : 0);

						if (last_c)
							result = lg;

						prev = end;
						done = end - start;

						if (opts.on_aux)
							opts.on_aux(end);
					}

					return next < length(bounds);
				},

				done_tokens: function () {return done;},
				logits: function () {return result;},
				hidden: function () {return null;}
			};
		}
	};

	return model;
}
