// ucode/dsv4.uc — DeepSeek-V4-Flash (DeepseekV4ForCausalLM), text.
//
// See docs/DSV4.md. The residual is a stack of 4 fp32 streams (mHC, the
// same hc_mix / hc_apply kernels as GLM-5.3); every sublayer site is
//
//     (post, comb, c) = hc_mix(streams)        c: fp16 collapsed input
//     y               = sublayer(norm(c))
//     streams         = post (x) y + comb^T streams
//
// Attention is MQA with ONE shared key/value head of 512 dims (V = K), 64
// query heads, an attention sink per head, a 128-token sliding window and,
// per layer type, a compressed pool: 'sliding' (none), 'csa' (windows of 4
// with the overlapping Ca/Cb scheme, a lightning indexer picks the top 512
// entries once the pool is larger) or 'hca' (windows of 128, dense).
// This file is the composite-op reference path (ATen matmuls; MQA is a
// plain GEMM over the keys) — correct first, fused kernels later.

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import {readfile} from 'fs';

const NEG = -1.0e30;
const DEC_NEG = -1.0e30;
const NO_FUSED = getenv('AMP_DSV4_NO_FUSED') ? true : false;
const OGRAPH = getenv('AMP_DSV4_OGRAPH') ? true : false;
let SIDE_HO = getenv('AMP_DSV4_NO_SIDE_HO') ? false : true;        // stage hand-off copies on a side stream

export function set_side_ho(b) {
	SIDE_HO = b ? true : false;
}
const FAST_PF = getenv('AMP_DSV4_PF32') ? false : true;      // fp16 tensor-core prefill attention
export let HP = getenv('AMP_DSV4_HOSTPROF') ? {a: 0, b: 0, attn: 0, s_proj: 0, s_comp: 0, s_core: 0, s_oproj: 0} : null;

function hp_tnow() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}       // host seconds in phase A / B of the prefill runner
const NEG16 = -60000.0;

/* ------------------------------------------------------------------ */
/* config                                                              */

export function load_config(model_dir) {
	let t = json(readfile(model_dir + '/config.json'));
	let nl = t.num_hidden_layers;
	let ratios = t.compress_ratios;
	let types = [];

	for (let i = 0; i < nl; i++) {
		let r = ratios ? ratios[i] : 0;

		push(types, r == 0 ? 'sliding' : r == 4 ? 'csa' : r == 128 ? 'hca' : null);

		if (types[i] == null)
			die(`dsv4: unsupported compress ratio ${r} at layer ${i}`);
	}

	let cfg = {
		model_dir: model_dir,
		hidden_size: t.hidden_size,
		num_hidden_layers: nl,
		vocab_size: t.vocab_size,
		rms_norm_eps: t.rms_norm_eps,
		layer_types: types,
		eos_token_id: t.eos_token_id,
		bos_token_id: t.bos_token_id,

		num_q_heads: t.num_attention_heads,
		head_dim: t.head_dim,
		rope_dim: t.qk_rope_head_dim,
		q_lora_rank: t.q_lora_rank,
		o_groups: t.o_groups,
		o_lora_rank: t.o_lora_rank,
		window: t.sliding_window,
		index_n_heads: t.index_n_heads,
		index_head_dim: t.index_head_dim,
		index_topk: t.index_topk,
		rope_theta: t.rope_theta,
		compress_rope_theta: t.compress_rope_theta,
		rope_scaling: t.rope_scaling,

		hc_mult: t.hc_mult,
		hc_eps: t.hc_eps,
		hc_sinkhorn_iters: t.hc_sinkhorn_iters,

		n_routed_experts: t.n_routed_experts,
		n_shared_experts: t.n_shared_experts,
		num_experts_per_tok: t.num_experts_per_tok,
		num_hash_layers: t.num_hash_layers,
		moe_intermediate_size: t.moe_intermediate_size,
		routed_scaling_factor: t.routed_scaling_factor,
		swiglu_limit: t.swiglu_limit,

		has_vision: t.vision_n_layers ? true : false,
		image_token_id: 129262,                      // <｜rl_image_pad｜>: the placeholder the chat renderer emits

		// DSpark block drafter (mtp.*)
		dspark_block: t.dspark_block_size || 0,
		dspark_noise_id: t.dspark_noise_token_id,
		dspark_taps: t.dspark_target_layer_ids || [],
		dspark_markov_rank: t.dspark_markov_rank || 256,
		num_mtp_layers: ratios ? length(ratios) - nl : 0
	};

	if (cfg.hc_mult != 4)
		die('dsv4: only hc_mult == 4 is supported');

	if (t.scoring_func != 'sqrtsoftplus')
		die(`dsv4: unsupported scoring_func '${t.scoring_func}'`);

	if (t.num_key_value_heads != 1)
		die('dsv4: expected shared-KV MQA (num_key_value_heads == 1)');

	cfg.sm_scale = math.pow(cfg.head_dim * 1.0, -0.5);

	return cfg;
}

export function compress_rate(kind) {
	return kind == 'csa' ? 4 : kind == 'hca' ? 128 : 0;
}

/* ------------------------------------------------------------------ */
/* rope (GPT-J interleaved pairs on the trailing rope_dim channels)    */

function floor_(x) {
	let f = int(x);

	return (f > x) ? f - 1 : f;
}

function ceil_(x) {
	let f = int(x);

	return (f < x) ? f + 1 : f;
}

/* exllamav3 util.rope.yarn_inv_freq: plain 1/base^(2i/dim) without
 * scaling, else the yarn interpolation ramp. Returns an fp32 device
 * tensor of dim/2 frequencies. */
export function yarn_inv_freq(dim, base, scaling, dev) {
	let n = int(dim / 2);
	let ext = [];

	for (let i = 0; i < n; i++)
		push(ext, 1.0 / math.pow(base * 1.0, (2 * i) / (dim * 1.0)));

	let out = ext;

	if (scaling && (scaling.type == 'yarn' || scaling.rope_type == 'yarn')) {
		let factor = scaling.factor * 1.0;
		let omp = scaling.original_max_position_embeddings * 1.0;
		let bf = (scaling.beta_fast != null) ? scaling.beta_fast * 1.0 : 32.0;
		let bs = (scaling.beta_slow != null) ? scaling.beta_slow * 1.0 : 1.0;

		function corr_dim(rot) {
			return (dim * math.log(omp / (rot * 2 * 3.14159265358979323846))) / (2 * math.log(base * 1.0));
		}

		let low = floor_(corr_dim(bf)), high = ceil_(corr_dim(bs));

		low = max(low, 0);
		high = min(high, dim - 1);

		if (low == high)
			high += 0.001;

		out = [];

		for (let i = 0; i < n; i++) {
			let lin = (i - low) / (high - low * 1.0);
			let efac = 1 - min(max(lin, 0.0), 1.0);
			let interp = ext[i] / factor;

			push(out, interp * (1 - efac) + ext[i] * efac);
		}
	}

	return torch.from_array(out, 'float32', 'cuda:' + dev);
}

/* cos / sin (T, 1, rd/2, 1) fp32 of the positions p0, p0 + step, ... */
function rope_cs(inv_freq, p0, T, step, dev) {
	let h = inv_freq.shape()[0];
	let ang = torch.arange(T, 'float32', 'cuda:' + dev).mul(step * 1.0).add(p0 * 1.0)
		.view([T, 1]).mul(inv_freq.view([1, h]));

	return [ang.cos().view([T, 1, h, 1]), ang.sin().view([T, 1, h, 1])];
}

/* Rotate the trailing rd channels of x (T, H, D) fp32 pairwise by (cos,
 * sin) (from rope_cs; pass sin.neg() to rotate back). Returns fp32. */
function rope_il(x, cs, rd) {
	let s = x.shape();
	let T = s[0], H = s[1], D = s[2];
	let tail = x.narrow(2, D - rd, rd).reshape([T, H, int(rd / 2), 2]);
	let e = tail.narrow(3, 0, 1), o = tail.narrow(3, 1, 1);
	let c = cs[0], sn = cs[1];
	let r = torch.cat([e.mul(c).sub(o.mul(sn)), e.mul(sn).add(o.mul(c))], 3)
		.reshape([T, H, rd]);

	return torch.cat([x.narrow(2, 0, D - rd), r], 2);
}

/* ------------------------------------------------------------------ */
/* compressor (HCA / CSA / indexer)                                    */

/* Static part of one compressor: projections, bias, norm, rope table. */
export function build_comp(ws, key, hd, m, overlapping, inv_freq, dev) {
	return {
		hd: hd, m: m, overlapping: overlapping,
		W: overlapping ? 2 * hd : hd,
		wkv: models.linear(ws, key + '.wkv', false, dev),
		wgate: models.linear(ws, key + '.wgate', false, dev),
		ape: ws.gpu(key + '.ape', dev).to('float32').contiguous(),
		norm_w: ws.gpu16(key + '.norm.weight', dev),
		inv_freq: inv_freq,
		dev: dev
	};
}

/* Per-sequence state of one compressor: the pool of emitted entries, the
 * raw rows of the incomplete window and the Ca overlap of the last one. */
function new_comp_state(C, cap) {
	return {
		pool: torch.zeros([cap, C.hd], 'float16', 'cuda:' + C.dev),
		n: 0, buf_kv: null, buf_gate: null, o_kv: null, o_gate: null
	};
}

/* Feed the chunk x (T, hidden) fp16 at absolute position pos0; emits
 * every completed window into S.pool (windows are positioned at their
 * first token: entry j is roped at j * m). */
function comp_forward(cfg, C, S, x, pos0) {
	let m = C.m, hd = C.hd, W = C.W;
	let dv = 'cuda:' + C.dev;
	let kv = C.wkv.forward(x).to('float32');
	let gate = C.wgate.forward(x).to('float32');

	if (S.buf_kv) {
		kv = torch.cat([S.buf_kv, kv], 0);
		gate = torch.cat([S.buf_gate, gate], 0);
	}

	let total = kv.shape()[0];
	let usable = int(total / m) * m;

	if (usable < total) {
		S.buf_kv = kv.narrow(0, usable, total - usable).contiguous();
		S.buf_gate = gate.narrow(0, usable, total - usable).contiguous();
	}
	else {
		S.buf_kv = null;
		S.buf_gate = null;
	}

	if (usable == 0)
		return 0;

	let nw = int(usable / m);
	let kvw = kv.narrow(0, 0, usable).reshape([nw, m, W]);
	let gw = gate.narrow(0, 0, usable).reshape([nw, m, W]).add(C.ape);
	let comp;

	if (C.overlapping) {
		// window w mixes its own second-half channels with the first-half
		// channels of window w-1 (the carried overlap for w == 0)
		let kb = kvw.narrow(2, hd, hd), gb = gw.narrow(2, hd, hd);
		let first_kv = S.o_kv ? S.o_kv.view([1, m, hd]) : torch.zeros([1, m, hd], 'float32', dv);
		let first_g = S.o_gate ? S.o_gate.view([1, m, hd]) : torch.full([1, m, hd], NEG, 'float32', dv);
		let pk = first_kv, pg = first_g;

		if (nw > 1) {
			pk = torch.cat([first_kv, kvw.narrow(0, 0, nw - 1).narrow(2, 0, hd)], 0);
			pg = torch.cat([first_g, gw.narrow(0, 0, nw - 1).narrow(2, 0, hd)], 0);
		}

		let ck = torch.cat([pk, kb], 1), cg = torch.cat([pg, gb], 1);

		S.o_kv = kvw.narrow(0, nw - 1, 1).narrow(2, 0, hd).reshape([m, hd]).contiguous();
		S.o_gate = gw.narrow(0, nw - 1, 1).narrow(2, 0, hd).reshape([m, hd]).contiguous();
		comp = ck.mul(cg.softmax(1)).sum(1);
	}
	else {
		comp = kvw.mul(gw.softmax(1)).sum(1);
	}

	let c16 = torch.rms_norm(comp.to('float16').contiguous(), C.norm_w, cfg.rms_norm_eps);
	let rd = cfg.rope_dim;
	let cs = rope_cs(C.inv_freq, S.n * m, nw, m, C.dev);
	let rot = rope_il(c16.to('float32').view([nw, 1, hd]), cs, rd).to('float16').view([nw, hd]);

	S.pool.narrow(0, S.n, nw).copy_(rot);
	S.n += nw;

	return nw;
}

/* ------------------------------------------------------------------ */
/* attention                                                           */

export function build_attn(ws, cfg, i, dev, pfx, kind0) {
	let key = (pfx || `layers.${i}`) + '.attn';
	let kind = kind0 || cfg.layer_types[i];
	let m = compress_rate(kind);
	let inv_main = yarn_inv_freq(cfg.rope_dim, cfg.rope_theta, null, dev);
	let inv_comp = yarn_inv_freq(cfg.rope_dim, cfg.compress_rope_theta, cfg.rope_scaling, dev);
	let A = {
		dev: dev, kind: kind, m: m,
		inv_freq: (kind == 'sliding') ? inv_main : inv_comp,
		inv_comp: inv_comp,
		q_a: models.linear(ws, key + '.wq_a', false, dev),
		q_norm: ws.gpu16(key + '.q_norm.weight', dev),
		q_b: models.linear(ws, key + '.wq_b', false, dev),
		wkv: models.linear(ws, key + '.wkv', false, dev),
		kv_norm: ws.gpu16(key + '.kv_norm.weight', dev),
		wo_a: [],
		wo_b: models.linear(ws, key + '.wo_b', false, dev),
		sinks: ws.gpu(key + '.attn_sink', dev).to('float32').contiguous(),
		comp: null, idx: null
	};

	for (let g = 0; g < cfg.o_groups; g++)
		push(A.wo_a, models.linear(ws, `${key}.wo_a.slice.${g}`, false, dev));

	if (kind != 'sliding')
		A.comp = build_comp(ws, key + '.compressor', cfg.head_dim, m, kind == 'csa', inv_comp, dev);

	if (kind == 'csa') {
		A.idx = build_comp(ws, key + '.indexer.compressor', cfg.index_head_dim, m, true, inv_comp, dev);
		A.idx_wq_b = models.linear(ws, key + '.indexer.wq_b', false, dev);
		A.idx_w = ws.gpu(key + '.indexer.weights_proj.weight', dev).to('float32').t().contiguous();
	}

	return A;
}

/* Device-resident compressor state (mirrors comp state S for S = 1 steps):
 * the m rows of the current window (slot = pos % m), the carried overlap
 * half, the entry count. */
function new_dec_comp(C, S) {
	let dv = 'cuda:' + C.dev;
	let cap = S.pool.shape()[0];

	return {
		kb: torch.zeros([C.m, C.W], 'float32', dv),
		kg: torch.zeros([C.m, C.W], 'float32', dv),
		// the overlap half of window w sits in ring slot w % 3 (see amp_dsv4.cu)
		ok: torch.zeros([3, C.m, C.hd], 'float32', dv),
		og: torch.full([3, C.m, C.hd], DEC_NEG, 'float32', dv),
		// pool entry j is visible once its window is complete: (j+1)*m <= pos+1
		pend: torch.arange(cap - 1, 'float32', dv).add(1.0).mul(C.m * 1.0).view([1, cap - 1])
	};
}

/* The raw key rows (post norm / rope) are only read through the sliding window (128 rows +
 * the chunk), so they live in a RING of KV_RING rows (position p in slot p % rows): 8192 rows
 * = 8 MB per layer instead of max_seq * 1 KB (268 MB per layer at 262144 tokens). The
 * kernels index modulo the tensor's row count; a rolled-back or restored position needs its
 * window rows intact, so snapshots save them (aux_get). AMP_DSV4_KV_RING=0 keeps full length. */
const KV_RING = +(getenv('AMP_DSV4_KV_RING') != null ? getenv('AMP_DSV4_KV_RING') : 8192);

export function ring_rows(max_seq) {
	return (KV_RING > 0 && KV_RING < max_seq) ? KV_RING : max_seq;
}

/* rows of absolute positions base .. base + n - 1 (base >= 0, n <= ring) as one (n, D) tensor */
export function kv_rows(cl, base, n) {
	let R = cl.kv.shape()[0];
	let a = base % R;

	if (a + n <= R)
		return cl.kv.narrow(0, a, n);

	return torch.cat([ cl.kv.narrow(0, a, R - a), cl.kv.narrow(0, 0, n - (R - a)) ], 0);
}

/* write rows (n, D) at absolute positions pos .. pos + n - 1 */
export function kv_put(cl, pos, rows) {
	let R = cl.kv.shape()[0];
	let n = rows.shape()[0];
	let a = pos % R;

	if (a + n <= R) {
		cl.kv.narrow(0, a, n).copy_(rows);
	}
	else {
		cl.kv.narrow(0, a, R - a).copy_(rows.narrow(0, 0, R - a));
		cl.kv.narrow(0, 0, n - (R - a)).copy_(rows.narrow(0, R - a, n - (R - a)));
	}
}

/* the window rows below position len (copy; the ring slots are reused by later positions) */
export function kv_window_get(cl, len, W) {
	let n = min(len, W - 1);

	if (n <= 0)
		return null;

	let out = torch.zeros([n, cl.kv.shape()[1]], 'float16', 'cuda:' + cl.dev);

	out.copy_(kv_rows(cl, len - n, n));

	return out;
}

export function kv_window_set(cl, len, rows) {
	if (rows)
		kv_put(cl, len - rows.shape()[0], rows);
}

/* --- host-RAM store of a layer's cache (the engine's prompt-cache store) ---------------
 * What defines a DeepSeek-V4 cache of `n` tokens: the compressed pool rows below the entry
 * count (written once, never rewritten by later tokens), and the small recurrent aux state
 * of aux_get (incomplete compressor windows, overlap halves, entry counts, the raw-KV window
 * rows). put(t) copies a device tensor to page-locked host memory (and accounts its bytes). */
export function pools_to_host(cl, put, dcl) {
	return map([ cl.comp, cl.idx ], function (S, k) {
		if (!S || S.n <= 0)
			return null;

		if (dcl) {
			// device-resident slot: the rows below the entry count, device to device
			let d = (k ? dcl.idx : dcl.comp).pool.narrow(0, 0, S.n);

			d.copy_(S.pool.narrow(0, 0, S.n));

			return d;
		}

		return put(S.pool.narrow(0, 0, S.n));
	});
}

/* one layer of a device slot: only the pools of a layer cache, same shapes */
export function new_slot_layer(A, max_seq) {
	let cap = A.m ? int(max_seq / A.m) + 2 : 0;

	return {
		comp: A.comp ? {pool: torch.zeros([cap, A.comp.hd], 'float16', 'cuda:' + A.comp.dev)} : null,
		idx: A.idx ? {pool: torch.zeros([cap, A.idx.hd], 'float16', 'cuda:' + A.idx.dev)} : null
	};
}

/* rows: [ comp entries, idx entries ] to write back (the aux entry counts of the state restored) */
export function pools_from_host(cl, h, rows) {
	let which = [ cl.comp, cl.idx ];

	for (let k = 0; k < 2; k++)
		if (which[k] && h[k] && rows[k] > 0)
			which[k].pool.narrow(0, 0, rows[k]).copy_(h[k].narrow(0, 0, rows[k]));
}

/* one layer's aux_get entry <-> host (tensors only; counts stay numbers) */
export function auxe_to_host(e, put) {
	return map(e, function (x) {
		if (x == null)
			return null;

		let o = {};

		for (let k in x)
			o[k] = (k == 'n' || x[k] == null) ? x[k] : put(x[k]);

		return o;
	});
}

export function auxe_to_dev(e, dev) {
	return map(e, function (x) {
		if (x == null)
			return null;

		let o = {};

		for (let k in x)
			o[k] = (k == 'n' || x[k] == null) ? x[k] : x[k].cuda(dev);

		return o;
	});
}

/* [ comp entries, idx entries ] of an aux_get entry */
export function aux_rows(e) {
	return [ e[0] ? e[0].n : 0, e[1] ? e[1].n : 0 ];
}

/* bytes of the store entry of one layer cache for n tokens (an upper bound) */
export function store_layer_bytes(cfg, A, n) {
	let b = min(n, cfg.window) * cfg.head_dim * 2;

	for (let C in [ A.comp, A.idx ])
		if (C)
			b += int(n / C.m) * C.hd * 2 + (2 * C.m * C.W + 2 * C.m * C.hd) * 4;

	return b;
}

/* Per-layer cache: the raw (post norm / rope) key rows for the window (a ring, see above)
 * and the compressor states. */
export function new_attn_cache(cfg, A, max_seq) {
	let dv = 'cuda:' + A.dev;
	let cap = A.m ? int(max_seq / A.m) + 2 : 0;
	let cl = {
		dev: A.dev,
		kv: torch.zeros([ring_rows(max_seq), cfg.head_dim], 'float16', dv),
		max_pos: max_seq,
		comp: A.comp ? new_comp_state(A.comp, cap) : null,
		idx: A.idx ? new_comp_state(A.idx, cap) : null
	};

	// device-resident twins for the graph decode path (see new_dec_comp)
	if (cl.comp)
		cl.comp.d = new_dec_comp(A.comp, cl.comp);

	if (cl.idx)
		cl.idx.d = new_dec_comp(A.idx, cl.idx);

	return cl;
}

/* Additive fp32 mask (R, n) from two conditions on float32 positions:
 * keep where lo <= key and key <= hi (lo / hi: (R, 1), key: (1, n)). */
function range_mask(key, lo, hi, dv) {
	let z = torch.zeros([1], 'float32', dv);
	let ng = torch.full([1], NEG, 'float32', dv);

	return torch.where(key.le(hi), z, ng).add(torch.where(lo.le(key), z, ng));
}

/* Indexer selection for the rows [r0, r0 + R) of the chunk: scores over
 * the indexer pool, top-k, -1 for masked picks. Returns int64 (R, K). */
function indexer_select(cfg, A, cl, q_idx, wts, pos0, r0, R, ne) {
	let dv = 'cuda:' + A.dev;
	let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, m = A.m;
	let scale = math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5);
	let sc;

	if (Di == 128 && Hi % 32 == 0 && !getenv('AMP_DSV4_IDX_COMPOSITE')) {
		// fused scoring (relu, head weights and visibility in one kernel, heads summed in fp32):
		// the composite below materializes (R, Hi, ne) fp32 = 2 GiB at 256k context (and
		// clamp() copies it). The visibility rule is the same: entry j needs (j + 1) * m <= qpos + 1.
		sc = torch.idx_score(q_idx.narrow(0, r0, R).to('float16').contiguous(), cl.idx.pool.narrow(0, 0, ne),
			wts.narrow(0, r0, R).contiguous(), scale, pos0 + r0, m).to('float32');
	}
	else {
		let pk = cl.idx.pool.narrow(0, 0, ne).to('float32');
		let qi = q_idx.narrow(0, r0, R).reshape([R * Hi, Di]);
		let s = torch.matmul(qi, pk.t()).view([R, Hi, ne]).clamp(0, null);

		sc = torch.matmul(wts.narrow(0, r0, R).view([R, 1, Hi]), s).view([R, ne]).mul(scale);
	}

	let qp = torch.arange(R, 'float32', dv).add(pos0 + r0 + 1.0).view([R, 1]);
	let pend = torch.arange(ne, 'float32', dv).add(1.0).mul(m * 1.0).view([1, ne]);
	let ok = pend.le(qp);

	sc = torch.where(ok, sc, torch.full([1], NEG, 'float32', dv));

	let k = min(cfg.index_topk, ne);
	let tk = sc.topk(k, -1);
	let dead = tk.values.le(NEG * 0.5);

	return torch.where(dead, torch.full([1], -1, 'int64', dv), tk.indices);
}

/* T rows (positions pos .. pos + T - 1, pi = pos as a device int64 (1)) into a compressor
 * through the fused kernel (device-twin state). */
function comp_rows(cfg, C, S, x, pi) {
	torch.dsv4_comp(C.wkv.forward(x), C.wgate.forward(x), C.ape, S.d.kb, S.d.kg,
		C.overlapping ? S.d.ok : null, C.overlapping ? S.d.og : null, S.pool,
		C.norm_w, C.inv_freq, pi, C.m, cfg.rope_dim, cfg.rms_norm_eps);
}

/* x: (T, hidden) fp16 (normed), cl: layer cache, pos0: tokens cached.
 * Returns (T, hidden) fp16. */
/* Grouped output projection (8 x wo_a, concat, wo_b) of attention output out (T, H, D) fp16.
 * The projection has fixed shapes and static weights, so for a full prefill chunk it is captured
 * once per (layer, T) as a CUDA graph (AMP_DSV4_OGRAPH=1) and replayed as ONE submission instead of
 * ~55 launches; the output buffer is persistent (the caller consumes it before the next replay
 * on the same stream). */
function oproj_run(cfg, A, og, T) {
	let G = cfg.o_groups, gw = int(cfg.num_q_heads / G) * cfg.head_dim;
	let ys = [];

	for (let g = 0; g < G; g++)
		push(ys, A.wo_a[g].forward(og.narrow(1, g, 1).reshape([T, gw]).contiguous()));

	return A.wo_b.forward(torch.cat(ys, 1));
}

function oproj(cfg, A, out, T) {
	let G = cfg.o_groups, gw = int(cfg.num_q_heads / G) * cfg.head_dim;
	let og = out.reshape([T, G, gw]);

	if (!OGRAPH || T < 512)
		return oproj_run(cfg, A, og, T);

	A.og = A.og || {};

	let e = A.og['' + T];

	if (!e) {
		e = A.og['' + T] = {n: 0, inp: torch.zeros([T, G, gw], 'float16', 'cuda:' + A.dev), g: null, out: null};
	}

	if (e.n < 2) {                     // warm-up (autotune, cuBLAS handles, allocator)
		e.n++;

		return oproj_run(cfg, A, og, T);
	}

	e.inp.copy_(og);

	if (!e.g) {
		e.g = torch.graph();
		e.g.capture_begin(A.dev);
		e.out = oproj_run(cfg, A, e.inp, T);
		e.g.capture_end();
	}

	e.g.replay();

	return e.out;
}

export function attn_forward(cfg, A, x, cl, pos0) {
	let T = x.shape()[0];
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let dv = 'cuda:' + A.dev;
	let eps = cfg.rms_norm_eps;
	let m = A.m;

	if (cl.kv.shape()[0] < cl.max_pos && T + cfg.window - 1 > cl.kv.shape()[0])
		die(`dsv4: a chunk of ${T} rows does not fit the ${cl.kv.shape()[0]}-row KV ring (DSV4_KV_RING)`);

	let ncs = (cl.ncs != null && cl.ncs >= 0) ? cl.ncs : -1;      // image chunk: start position

	// the fused path keeps the compressor state in the device twins (see amp_dsv4.cu)
	let fast = FAST_PF && !cl.dbg && (!A.comp || cl.comp.dmode);
	let pf = null, pi = null;
	let q16 = null, q16s = null, kv16 = null;

	let tp0 = HP ? hp_tnow() : 0;

	// q path
	let q_res = torch.rms_norm(A.q_a.forward(x), A.q_norm, eps);

	if (fast) {
		pf = torch.from_array([ pos0 * 1.0 ], 'float32', dv);
		pi = torch.from_array([ pos0 ], 'int64', dv);

		// norm + rope of q (scale folded in) and of the shared K=V row, written to the cache
		q16s = torch.dsv4_qkv(A.q_b.forward(q_res).view([T, H, D]), A.wkv.forward(x).view([T, D]),
			A.kv_norm, A.inv_freq, pf, pi, cl.kv, rd, eps, cfg.sm_scale);
	}
	else {
		let qf = A.q_b.forward(q_res).view([T, H, D]).to('float32');

		qf = qf.mul(qf.mul(qf).mean(-1, true).add(eps).rsqrt());

		// kv: weighted norm, then rope on both
		let kvn = torch.rms_norm(A.wkv.forward(x), A.kv_norm, eps);
		let cs = rope_cs(A.inv_freq, pos0, T, 1, A.dev);
		let qr = rope_il(qf, cs, rd);

		q16 = qr.to('float16').contiguous();
		q16s = FAST_PF ? qr.mul(cfg.sm_scale).to('float16').contiguous() : null;     // scale folded in
		kv16 = rope_il(kvn.to('float32').view([T, 1, D]), cs, rd).to('float16').view([T, D]);

		kv_put(cl, pos0, kv16);
	}

	if (HP) {let t = hp_tnow(); HP.s_proj += t - tp0; tp0 = t;}

	// compressed pools
	let ne = 0;

	if (A.comp) {
		if (fast) {
			comp_rows(cfg, A.comp, cl.comp, x, pi);

			if (A.idx)
				comp_rows(cfg, A.idx, cl.idx, x, pi);

			ne = int((pos0 + T) / m);
		}
		else {
			comp_forward(cfg, A.comp, cl.comp, x, pos0);

			if (A.idx)
				comp_forward(cfg, A.idx, cl.idx, x, pos0);

			ne = cl.comp.n;
		}
	}

	if (HP) {let t = hp_tnow(); HP.s_comp += t - tp0; tp0 = t;}

	let topk = (A.idx && ne > cfg.index_topk);
	let q_idx = null, wts = null;

	if (topk) {
		let Hi = cfg.index_n_heads, Di = cfg.index_head_dim;
		let qi = A.idx_wq_b.forward(q_res).view([T, Hi, Di]).to('float32');
		let cs2 = rope_cs(A.inv_comp, pos0, T, 1, A.dev);

		q_idx = rope_il(qi, cs2, rd).contiguous();
		wts = torch.matmul(x.to('float32'), A.idx_w);
	}

	// window keys: the previous (window - 1) rows and the chunk
	let W = cfg.window;
	let n_prior = min(W - 1, pos0);
	let base = pos0 - n_prior;
	let nk = n_prior + T;
	let scale = cfg.sm_scale;
	let outs = [];

	if (FAST_PF) {
		// fp16 tensor-core path: scores / probabilities in fp16 (softmax
		// accumulates in fp32 internally), the scale folded into q
		let Kw16 = kv_rows(cl, base, nk);
		let pool16 = (ne > 0 && !topk) ? cl.comp.pool.narrow(0, 0, ne) : null;
		let Kall = pool16 ? torch.cat([Kw16, pool16], 0) : Kw16;
		let ntot = nk + (pool16 ? ne : 0);
		let sink16 = A.sinks.to('float16').view([1, H, 1]);
		let QC16 = topk ? 128 : 512;
		let z = torch.zeros([1], 'float32', dv), ng = torch.full([1], NEG16, 'float32', dv);

		for (let r0 = 0; r0 < T; r0 += QC16) {
			let R = min(QC16, T - r0);
			let qs = q16s.narrow(0, r0, R);
			let sc, o, g16 = null, K = 0;

			if (topk) {
				let idx = indexer_select(cfg, A, cl, q_idx, wts, pos0, r0, R, ne);
				let valid = idx.le(-1);
				let safe = torch.where(valid, torch.zeros([1], 'int64', dv), idx);

				K = idx.shape()[1];
				g16 = cl.comp.pool.index(safe.reshape([-1])).view([R, K, D]);

				let wm = torch.dsv4_mask(A.dev, R, nk, 0, pos0 + r0, base, W, 1, ncs);
				let sw = torch.matmul(qs.view([R * H, D]), Kw16.t()).view([R, H, nk]).add(wm.view([R, 1, nk]));
				let sp = torch.matmul(qs, g16.transpose(1, 2)).add(
					torch.where(valid, ng, z).to('float16').view([R, 1, K]));

				sc = torch.cat([sw, sp, sink16.expand([R, H, 1])], 2);
			}
			else {
				let mask = torch.dsv4_mask(A.dev, R, nk, pool16 ? ne : 0, pos0 + r0, base, W, m ? m : 1, ncs);

				sc = torch.matmul(qs.view([R * H, D]), Kall.t()).view([R, H, ntot]).add(mask.view([R, 1, ntot]));
				sc = torch.cat([sc, sink16.expand([R, H, 1])], 2);
			}

			let p = sc.softmax(2);

			if (topk) {
				o = torch.matmul(p.narrow(2, 0, nk).reshape([R * H, nk]), Kw16).view([R, H, D])
					.add(torch.matmul(p.narrow(2, nk, K), g16));
			}
			else {
				o = torch.matmul(p.narrow(2, 0, ntot).reshape([R * H, ntot]), Kall).view([R, H, D]);
			}

			push(outs, torch.dsv4_derot(o.contiguous(), A.inv_freq,
				torch.from_array([ (pos0 + r0) * 1.0 ], 'float32', dv), rd));
		}
	}

	let Kw = FAST_PF ? null : kv_rows(cl, base, n_prior + T).to('float32');
	let poolf = (!FAST_PF && ne > 0 && !topk) ? cl.comp.pool.narrow(0, 0, ne).to('float32') : null;
	let sinkv = A.sinks.view([1, H, 1]);
	let QC = topk ? 128 : 256;

	for (let r0 = 0; !FAST_PF && r0 < T; r0 += QC) {
		let R = min(QC, T - r0);
		let qs = q16.narrow(0, r0, R).to('float32');
		let qp = torch.arange(R, 'float32', dv).add(pos0 + r0 * 1.0).view([R, 1]);

		// window part
		let sw = torch.matmul(qs.view([R * H, D]), Kw.t()).view([R, H, nk]).mul(scale);
		let kpos = torch.arange(nk, 'float32', dv).add(base * 1.0).view([1, nk]);

		sw = sw.add(range_mask(kpos, qp.sub(W - 1.0), qp, dv).view([R, 1, nk]));

		let parts = [sw], pk = null, nkp = 0, g = null;

		if (topk) {
			let idx = indexer_select(cfg, A, cl, q_idx, wts, pos0, r0, R, ne);
			let K = idx.shape()[1];
			let valid = idx.le(-1);
			let safe = torch.where(valid, torch.zeros([1], 'int64', dv), idx);

			g = cl.comp.pool.index(safe.reshape([-1])).view([R, K, D]).to('float32');

			let sp = torch.matmul(qs, g.transpose(1, 2)).mul(scale);
			let z = torch.zeros([1], 'float32', dv);

			sp = sp.add(torch.where(valid, torch.full([1], NEG, 'float32', dv), z).view([R, 1, K]));
			push(parts, sp);
			nkp = K;
		}
		else if (ne > 0) {
			let sp = torch.matmul(qs.view([R * H, D]), poolf.t()).view([R, H, ne]).mul(scale);
			let pend = torch.arange(ne, 'float32', dv).add(1.0).mul(m * 1.0).view([1, ne]);
			let z = torch.zeros([1], 'float32', dv);
			let ok = pend.le(qp.add(1.0));

			sp = sp.add(torch.where(ok, z, torch.full([1], NEG, 'float32', dv)).view([R, 1, ne]));
			push(parts, sp);
			nkp = ne;
		}

		push(parts, sinkv.expand([R, H, 1]));

		let p = torch.cat(parts, 2).softmax(2);
		let o = torch.matmul(p.narrow(2, 0, nk).reshape([R * H, nk]), Kw).view([R, H, D]);

		if (topk)
			o = o.add(torch.matmul(p.narrow(2, nk, nkp), g));
		else if (ne > 0)
			o = o.add(torch.matmul(p.narrow(2, nk, nkp).reshape([R * H, nkp]), poolf).view([R, H, D]));

		// de-rotate the output's rope slice by the query position
		let dcs = rope_cs(A.inv_freq, pos0 + r0, R, 1, A.dev);

		push(outs, rope_il(o, [dcs[0], dcs[1].neg()], rd).to('float16'));
	}

	if (HP) {let t = hp_tnow(); HP.s_core += t - tp0; tp0 = t;}

	let out = (length(outs) == 1) ? outs[0] : torch.cat(outs, 0);        // (T, H, D)

	if (cl.dbg) {                                 // tests: the attention internals
		cl.dbg.q = q16;
		cl.dbg.kv = kv16;
		cl.dbg.out = out;
		cl.dbg.pool = ne ? cl.comp.pool.narrow(0, 0, ne) : null;
	}

	let ob = oproj(cfg, A, out, T);

	if (HP) HP.s_oproj += hp_tnow() - tp0;

	return ob;
}

/* ------------------------------------------------------------------ */
/* device-side decode (S = 1): fixed shapes, the position and every state */
/* counter live on the device, so a whole run of layers is one CUDA graph  */


/* Per-device decode constants: position (int64 + float32), the window
 * offsets and zeros. */
export function new_dec_dev(cfg, dev) {
	let dv = 'cuda:' + dev;
	let W = cfg.window, ar = [], arf = [];

	for (let j = 0; j < W; j++) {
		push(ar, j - (W - 1));
		push(arf, (j - (W - 1)) * 1.0);
	}

	return {
		dev: dev,
		pos_i: torch.zeros([1], 'int64', dv),
		pos_f: torch.zeros([1, 1], 'float32', dv),
		ar_i: torch.from_array(ar, 'int64', dv),
		ar_f: torch.from_array(arf, 'float32', dv).view([1, W]),
		zero_f: torch.zeros([1, 1], 'float32', dv),
		zero_i: torch.zeros([1], 'int64', dv),
		neg_f: torch.full([1], DEC_NEG, 'float32', dv),
		m1_i: torch.full([1, 1], -1, 'int64', dv)
	};
}

export function dec_set_pos(st, pos) {
	st.pos_i.fill_(pos);
	st.pos_f.fill_(pos * 1.0);
}

/* (S, 1) fp32 row offsets 0 .. S-1 (cached per device state) */
export function dec_rows(st, S) {
	if (!st.rows)
		st.rows = {};

	if (!st.rows[S]) {
		let a = [];

		for (let i = 0; i < S; i++)
			push(a, i * 1.0);

		st.rows[S] = torch.from_array(a, 'float32', 'cuda:' + st.dev).view([S, 1]);
	}

	return st.rows[S];
}

/* cos / sin (S, 1, rd/2, 1) from device positions (S, 1) fp32 */
function rope_cs_dev(inv_freq, pos_f) {
	let h = inv_freq.shape()[0];
	let n = pos_f.shape()[0];
	let ang = pos_f.mul(inv_freq.view([1, h]));

	return [ang.cos().view([n, 1, h, 1]), ang.sin().view([n, 1, h, 1])];
}

/* eager -> device state (call at len = cache.length) */
export function comp_to_dev(C, S, len) {
	let d = S.d, m = C.m;

	S.dmode = true;
	let n = int(len / m), r = len - n * m;

	d.kb.fill_(0);
	d.kg.fill_(0);

	if (r > 0) {
		d.kb.narrow(0, 0, r).copy_(S.buf_kv);
		d.kg.narrow(0, 0, r).copy_(S.buf_gate);
	}

	if (C.overlapping) {
		d.ok.fill_(0);
		d.og.fill_(DEC_NEG);

		// the last completed window n - 1 lives in ring slot (n - 1) % 3
		if (S.o_kv) {
			let sl = (n - 1) % 3;

			d.ok.narrow(0, sl, 1).copy_(S.o_kv.view([1, m, C.hd]));
			d.og.narrow(0, sl, 1).copy_(S.o_gate.view([1, m, C.hd]));
		}
	}
}

/* device -> eager state */
export function comp_to_eager(C, S, len) {
	let d = S.d, m = C.m;

	S.dmode = false;
	let n = int(len / m), r = len - n * m;

	S.n = n;
	S.buf_kv = (r > 0) ? d.kb.narrow(0, 0, r).add(0.0) : null;
	S.buf_gate = (r > 0) ? d.kg.narrow(0, 0, r).add(0.0) : null;
	S.o_kv = (C.overlapping && n > 0) ? d.ok.narrow(0, (n - 1) % 3, 1).reshape([m, C.hd]).add(0.0) : null;
	S.o_gate = (C.overlapping && n > 0) ? d.og.narrow(0, (n - 1) % 3, 1).reshape([m, C.hd]).add(0.0) : null;
}

/* One token (x: (1, hidden) fp16) into the compressor; a completed window
 * is written to pool entry n, else to the dump row. */
function comp_step(cfg, C, S, x, st, pkv, pg) {
	torch.dsv4_comp(pkv ? pkv : C.wkv.forward(x), pg ? pg : C.wgate.forward(x), C.ape,
		S.d.kb, S.d.kg, C.overlapping ? S.d.ok : null, C.overlapping ? S.d.og : null, S.pool,
		C.norm_w, C.inv_freq, st.pos_i, C.m, cfg.rope_dim, cfg.rms_norm_eps);
}

/* Batched projections for the S = 1 decode path (one cooperative mgemm launch
 * instead of one GEMV per linear; needs one bitrate / codebook across the
 * bundle, else the plain linears stay): A.fan — q_a, wkv and the compressor /
 * indexer projections (same input x); A.fan_o — the o_group wo_a slices (one
 * input slice each). Also the shared expert's gate + up: build_gu(mlp). */
export function build_fans(A) {
	if (NO_FUSED || getenv('AMP_DSV4_NO_FAN'))
		return;

	let dev = A.dev, dv = 'cuda:' + dev;
	let lins = [ A.q_a, A.wkv ];

	if (A.comp)
		push(lins, A.comp.wkv, A.comp.wgate);

	if (A.idx)
		push(lins, A.idx.wkv, A.idx.wgate);

	try {
		let outs = map(lins, (l) => torch.empty([1, 1, l.out_features], 'float16', dv));

		A.fan = models.sliced(lins, outs, dev);
		A.fan.outs = outs;
	}
	catch (e) {
		A.fan = null;
	}

	try {
		let o = torch.empty([length(A.wo_a), 1, A.wo_a[0].out_features], 'float16', dv);

		A.fan_o = models.multi(A.wo_a, o, dev);
		A.fan_o.out = o;
	}
	catch (e) {
		A.fan_o = null;
	}
}

export function build_gu(M, dev) {
	if (NO_FUSED || getenv('AMP_DSV4_NO_FAN'))
		return;

	try {
		let o = torch.empty([2, 1, M.gate.out_features], 'float16', 'cuda:' + dev);

		M.gu = models.multi([ M.gate, M.up ], o, dev);
		M.gu.out = o;
	}
	catch (e) {
		M.gu = null;
	}
}

/* x: (1, hidden) fp16 (normed). Returns (1, hidden) fp16. */
export function attn_decode(cfg, A, x, cl, st) {
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let W = cfg.window;
	let dv = 'cuda:' + A.dev;
	let eps = cfg.rms_norm_eps;
	let scale = cfg.sm_scale;

	let S = x.shape()[0];
	let fan = (!NO_FUSED && S == 1 && A.fan) ? A.fan : null;

	if (fan)
		fan.run(x.view([1, 1, x.shape()[1]]));

	let q_res = torch.rms_norm(fan ? fan.outs[0].view([1, -1]) : A.q_a.forward(x), A.q_norm, eps);

	// fused path: qkv prep, compressor, attention core in three kernels (the
	// dense pool must fit the kernel's score buffer, else the composite path);
	// S rows = S consecutive tokens from the device position (verify steps)
	let cap = A.comp ? cl.comp.pool.shape()[0] - 1 : 0;
	let topk_f = (A.comp && A.idx && cap > cfg.index_topk);

	if (!NO_FUSED && (topk_f || W + cap <= 11000)) {
		let q16 = torch.dsv4_qkv(A.q_b.forward(q_res).view([S, H, D]),
			(fan ? fan.outs[1] : A.wkv.forward(x)).view([S, D]),
			A.kv_norm, A.inv_freq, st.pos_f, st.pos_i, cl.kv, rd, eps, 1.0, A.nc_block ? (getenv('AMP_DSPARK_NO_FP8KV') ? 0 : 1) : 0);
		let pool = null, idx = null, npm = 0;

		if (A.comp) {
			comp_step(cfg, A.comp, cl.comp, x, st, fan ? fan.outs[2] : null, fan ? fan.outs[3] : null);

			if (A.idx)
				comp_step(cfg, A.idx, cl.idx, x, st, fan ? fan.outs[4] : null, fan ? fan.outs[5] : null);

			pool = cl.comp.pool;
			npm = cap;
		}

		if (topk_f) {
			let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, K = cfg.index_topk;
			let pf = st.pos_f.add(dec_rows(st, S));                                           // (S, 1)
			let qi = A.idx_wq_b.forward(q_res).view([S, Hi, Di]).to('float32');
			let qi16 = rope_il(qi, rope_cs_dev(A.inv_comp, pf), rd).to('float16').view([S * Hi, Di]);
			let wts = torch.matmul(x.to('float32'), A.idx_w).to('float16').view([S, 1, Hi]);
			let Pi = cl.idx.pool.narrow(0, 0, cap);
			let s = torch.matmul(qi16, Pi.t()).clamp(0, null).view([S, Hi, cap]);
			let sc = torch.matmul(wts, s).view([S, cap]).to('float32')
				.mul(math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5));
			let ok = cl.comp.d.pend.le(pf.add(1.0));

			sc = torch.where(ok, sc, st.neg_f);

			let tk = sc.topk(K, -1);

			idx = torch.where(tk.values.le(DEC_NEG * 0.5), st.m1_i, tk.indices).contiguous();
			npm = K;
		}

		let out = torch.dsv4_attn(q16, cl.kv, st.pos_i, st.pos_f, pool, idx, A.sinks,
			A.inv_freq, W, A.m ? A.m : 1, rd, npm, scale, A.nc_block ? A.nc_block : 0);
		let G = cfg.o_groups, gw = int(H / G) * D;

		if (A.fan_o && S == 1) {
			A.fan_o.run(out.view([G, 1, gw]));

			return A.wo_b.forward(A.fan_o.out.view([1, -1]));
		}

		let og = out.reshape([S, G, gw]);
		let ys = [];

		for (let g = 0; g < G; g++)
			push(ys, A.wo_a[g].forward(og.narrow(1, g, 1).reshape([S, gw]).contiguous()));

		return A.wo_b.forward(torch.cat(ys, 1));
	}
	let qf = A.q_b.forward(q_res).view([1, H, D]).to('float32');

	qf = qf.mul(qf.mul(qf).mean(-1, true).add(eps).rsqrt());

	let kvn = torch.rms_norm(A.wkv.forward(x), A.kv_norm, eps);
	let cs = rope_cs_dev(A.inv_freq, st.pos_f);
	let q2 = rope_il(qf, cs, rd).view([H, D]);                  // fp32
	let kv16 = rope_il(kvn.to('float32').view([1, 1, D]), cs, rd).to('float16').view([1, D]);

	if (cl.kv.shape()[0] < cl.max_pos)
		die('dsv4: the composite decode path needs the full-length KV (AMP_DSV4_KV_RING=0)');

	cl.kv.index_copy_(0, st.pos_i, kv16);

	if (A.comp) {
		comp_step(cfg, A.comp, cl.comp, x, st);

		if (A.idx)
			comp_step(cfg, A.idx, cl.idx, x, st);
	}

	// window: the last W rows (rows before position 0 masked)
	let raw = st.pos_f.add(st.ar_f);                             // (1, W)
	let vwin = st.zero_f.le(raw);
	let wrow = torch.where(vwin.view([W]), st.pos_i.add(st.ar_i), st.zero_i);
	let Kw = cl.kv.index(wrow).to('float32');                    // (W, D)
	let sw = torch.matmul(q2, Kw.t()).mul(scale).add(torch.where(vwin, st.zero_f, st.neg_f));
	let parts = [sw], pv = null, np = 0, pool_is_g = false;

	if (A.comp) {
		let cap = cl.comp.pool.shape()[0] - 1;
		let topk = (A.idx && cap > cfg.index_topk);

		if (topk) {
			let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, K = cfg.index_topk;
			let qi = A.idx_wq_b.forward(q_res).view([1, Hi, Di]).to('float32');
			let cs2 = rope_cs_dev(A.inv_comp, st.pos_f);
			let qi16 = rope_il(qi, cs2, rd).to('float16').view([Hi, Di]);
			let wts = torch.matmul(x.to('float32'), A.idx_w).to('float16');                 // (1, Hi)
			let Pi = cl.idx.pool.narrow(0, 0, cap);
			let s = torch.matmul(qi16, Pi.t()).clamp(0, null);                              // (Hi, cap)
			let sc = torch.matmul(wts, s).to('float32')
				.mul(math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5));                   // (1, cap)
			let ok = cl.comp.d.pend.le(st.pos_f.add(1.0));

			sc = torch.where(ok, sc, st.neg_f);

			let tk = sc.topk(K, -1);
			let dead = tk.values.le(DEC_NEG * 0.5);                                         // (1, K)
			let safe = torch.where(dead, st.zero_i.view([1, 1]), tk.indices).view([K]);

			pv = cl.comp.pool.index(safe).to('float32');                                    // (K, D)
			pool_is_g = true;
			np = K;
			push(parts, torch.matmul(q2, pv.t()).mul(scale)
				.add(torch.where(dead, st.neg_f.view([1, 1]), st.zero_f)));
		}
		else {
			pv = cl.comp.pool.narrow(0, 0, cap).to('float32');
			np = cap;

			let ok = cl.comp.d.pend.le(st.pos_f.add(1.0));

			push(parts, torch.matmul(q2, pv.t()).mul(scale)
				.add(torch.where(ok, st.zero_f, st.neg_f)));
		}
	}

	push(parts, A.sinks.view([H, 1]));

	let p = torch.cat(parts, 1).softmax(1);
	let o = torch.matmul(p.narrow(1, 0, W), Kw);

	if (np)
		o = o.add(torch.matmul(p.narrow(1, W, np), pv));

	let ocs = rope_cs_dev(A.inv_freq, st.pos_f);
	let out = rope_il(o.view([1, H, D]), [ocs[0], ocs[1].neg()], rd).to('float16');
	let G = cfg.o_groups, gw = int(H / G) * D;
	let og = out.reshape([1, G, gw]);
	let ys = [];

	for (let g = 0; g < G; g++)
		push(ys, A.wo_a[g].forward(og.narrow(1, g, 1).reshape([1, gw]).contiguous()));

	return A.wo_b.forward(torch.cat(ys, 1));
}

/* ------------------------------------------------------------------ */
/* MoE + layer                                                         */

/* A gated expert / shared MLP: w1 = gate, w3 = up, w2 = down. */
function build_mlp(ws, key, dev) {
	return {
		gate: models.linear(ws, key + '.w1', false, dev),
		up: models.linear(ws, key + '.w3', false, dev),
		down: models.linear(ws, key + '.w2', false, dev)
	};
}

export function build_moe(ws, cfg, i, dev, pfx) {
	let key = (pfx || `layers.${i}`) + '.ffn';
	let experts = [];

	for (let e = 0; e < cfg.n_routed_experts; e++)
		push(experts, build_mlp(ws, `${key}.experts.${e}`, dev));

	return {
		type: 'moe', dev: dev, experts: experts, router: 'sqrtsp',
		gate_t: ws.gpu(key + '.gate.weight', dev).to('float32').t().contiguous(),
		bias: ws.gpu(key + '.gate.bias', dev).to('float32').reshape([1, -1]).contiguous(),
		async_moe: !getenv('AMP_DSV4_SYNC_MOE'),
		bias_vl: (!pfx && ws.has(key + '.gate.bias_vl')) ?
			ws.gpu(key + '.gate.bias_vl', dev).to('float32').reshape([1, -1]).contiguous() : null,
		tid2eid: (!pfx && i < cfg.num_hash_layers && ws.has(key + '.gate.tid2eid')) ?
			ws.gpu(key + '.gate.tid2eid', dev).to('int64').contiguous() : null,
		ids: null,
		shared: build_mlp(ws, key + '.shared_experts', dev)
	};
}

export function build_hc(ws, i, tag, dev, pfx) {
	let k = (pfx || `layers.${i}`) + `.hc_${tag}`;

	return {
		fn: ws.gpu(k + '_fn', dev).to('float32').contiguous(),
		base: ws.gpu(k + '_base', dev).to('float32').contiguous(),
		scale: ws.gpu(k + '_scale', dev).to('float32').contiguous()
	};
}

export function build_layer(ws, cfg, i, dev, pfx, kind) {
	let p = pfx || `layers.${i}`;

	return {
		dev: dev, kind: kind || cfg.layer_types[i], idx: i,
		hc_attn: build_hc(ws, i, 'attn', dev, pfx),
		hc_ffn: build_hc(ws, i, 'ffn', dev, pfx),
		attn_norm: ws.gpu16(`${p}.attn_norm.weight`, dev),
		ffn_norm: ws.gpu16(`${p}.ffn_norm.weight`, dev),
		attn: build_attn(ws, cfg, i, dev, pfx, kind),
		moe: build_moe(ws, cfg, i, dev, pfx)
	};
}

function finish_layer(L) {
	build_fans(L.attn);
	build_gu(L.moe.shared, L.dev);

	return L;
}

/* hc_mix with the collapsed input saturated to the fp16 range. The BOS
 * and first-text tokens carry massive activations (stream values up to ~9e4
 * from layer ~28; the fp32 streams are fine), so the pre-weighted sum can
 * exceed 65504, and the vendored kernel's half conversion then yields inf
 * (-> NaN after the norm). The reference saturates (oracle: mlp_in of such a
 * row matches a clamped collapse to 2e-4, an unclamped one to 6e-2). */
/* DSV4_PROBE: device-side peak trackers (graph safe: device ops only; the
 * accumulators are created lazily during the eager warm-up) of the fp32 stream
 * magnitude and of the half collapse BEFORE the saturation. probe_report()
 * returns [stream peak, collapse peak, non-finite flag] over every site since
 * the last report and resets them. */
const PROBE = getenv('AMP_DSV4_PROBE') != null;
const probe_accs = [];

function probe_note(hc, streams, c) {
	if (!hc._pk) {
		hc._pk = streams.abs().max().view([1]).to('float32').mul(0.0);
		hc._pc = streams.abs().max().view([1]).to('float32').mul(0.0);
		push(probe_accs, hc);
	}

	let sp = streams.abs().max().view([1]).to('float32');
	let cp = c.abs().max().view([1]).to('float32');

	hc._pk.copy_(torch.where(sp.le(hc._pk), hc._pk, sp));
	hc._pc.copy_(torch.where(cp.le(hc._pc), hc._pc, cp));
}

export function probe_report() {
	let sp = 0.0, cp = 0.0;

	for (let hc in probe_accs) {
		let a = hc._pk.item(), b = hc._pc.item();

		if (a > sp || a != a) sp = a;
		if (b > cp || b != b) cp = b;
		hc._pk.fill_(0.0);
		hc._pc.fill_(0.0);
	}

	return [ sp, cp, (cp > 65503.0 || cp != cp) ];
}

export function hc_collapse(cfg, hc, streams) {
	let r = glm5.hc_mix(cfg, hc, streams);

	if (PROBE)
		probe_note(hc, streams, r[2]);

	return [ r[0], r[1], r[2].clamp(-65504.0, 65504.0) ];
}

/* ------------------------------------------------------------------ */
/* multimodal rows: cache.mm = {spans: [ {lo, nc0, hi, emb} ]} in absolute  */
/* positions. [lo, hi) are the rows of an image block that carry embeddings     */
/* (their token ids are >= MM_BASE), [nc0, hi) is the image proper (START ..    */
/* END) that is prefilled as ONE chunk with bidirectional window attention;     */
/* emb is (hi - lo, hidden) fp16 on the first device.                           */

export const MM_BASE = glm5.MM_BASE;

/* images -> cache.mm spans; images: [{start (first row incl. lead pads), nc0 (IMAGE_START
 * position), emb (rows, hidden) fp16}] */
export function mm_build(cfg, images) {
	if (!images || !length(images))
		return null;

	let spans = [];

	for (let im in images)
		push(spans, {lo: im.start, nc0: im.nc0, hi: im.start + im.emb.shape()[0], emb: im.emb});

	return {spans: spans};
}

/* the chunk [pos, pos + T): {vlon: has image rows, nc: it is exactly an image} */
export function chunk_flags(mm, pos, T) {
	let r = {vlon: false, ncs: -1};

	if (!mm)
		return r;

	for (let sp in mm.spans) {
		if (sp.lo < pos + T && sp.hi > pos)
			r.vlon = true;

		if (sp.nc0 == pos && sp.hi == pos + T)
			r.ncs = pos;
	}

	return r;
}

/* embeddings (T, hidden) of the ids at pos.. with the image rows overlaid
 * (ids >= MM_BASE are looked up as token 0 first). On `dev` (the embedding's). */
export function embed_rows(embed_t, mm, ids, pos, dev) {
	let T = length(ids);
	let any = false;

	if (mm)
		for (let sp in mm.spans)
			if (sp.lo < pos + T && sp.hi > pos)
				any = true;

	let safe = ids;

	if (any) {
		safe = [];

		for (let t in ids)
			push(safe, t >= MM_BASE ? 0 : t);
	}

	let e = embed_t.index(torch.from_array(safe, 'int64', 'cuda:' + dev));

	if (any) {
		e = e.contiguous();

		for (let sp in mm.spans) {
			let a = max(sp.lo, pos), b = min(sp.hi, pos + T);

			if (b > a)
				e.narrow(0, a - pos, b - a).copy_(sp.emb.narrow(0, a - sp.lo, b - a).cuda(dev));
		}
	}

	return e;
}

/* One block on the streams (T, 4, hidden) fp32 (in place). ids: int64
 * (T,) token ids on the layer's device (hash routing). trace: optional
 * object receiving the intermediates (tests). */
/* Phase A: attention half incl. its residual update, the ffn site's mix +
 * norm and the MoE routing / launches that need no host read. Phase B
 * (block_b) finishes the MoE (it may read expert counts) and applies it. */
function flag_layer(cl, c) {
	cl.vlon = c.vlon;
	cl.ncs = c.ncs;

	return cl;
}

export function block_a(cfg, L, streams, cl, pos, ids, trace) {
	let eps = cfg.rms_norm_eps;
	let r = hc_collapse(cfg, L.hc_attn, streams);
	let xn = torch.rms_norm(r[2], L.attn_norm, eps);
	let ta0 = HP ? hp_tnow() : 0;
	let y = attn_forward(cfg, L.attn, xn, cl, pos);

	if (HP) HP.attn += hp_tnow() - ta0;

	if (trace) {
		trace.attn_in = xn;
		trace.attn_out = y;
		trace.hc_attn_post = r[0];
		trace.hc_attn_comb = r[1];
	}

	glm5.hc_apply(streams, y, r[0], r[1]);

	r = hc_collapse(cfg, L.hc_ffn, streams);
	xn = torch.rms_norm(r[2], L.ffn_norm, eps);
	L.moe.ids = ids;
	L.moe.vlon = cl.vlon ? true : false;

	if (trace) {
		trace.mlp_in = xn;
		trace.hc_ffn_post = r[0];
		trace.hc_ffn_comb = r[1];
	}

	return {streams: streams, r: r, trace: trace, pend: glm5.moe_begin(cfg, L.moe, xn)};
}

export function block_b(cfg, L, ctx) {
	let y = glm5.moe_finish(cfg, L.moe, ctx.pend);

	if (ctx.trace)
		ctx.trace.mlp_out = y;

	glm5.hc_apply(ctx.streams, y, ctx.r[0], ctx.r[1]);

	return ctx.streams;
}

export function block_forward(cfg, L, streams, cl, pos, ids, trace) {
	return block_b(cfg, L, block_a(cfg, L, streams, cl, pos, ids, trace));
}

/* One decode block (S = 1) on streams (1, 4, hidden); ids: (1) int64. */
export function block_decode(cfg, L, streams, cl, st, ids) {
	let eps = cfg.rms_norm_eps;
	let r = hc_collapse(cfg, L.hc_attn, streams);
	let xn = torch.rms_norm(r[2], L.attn_norm, eps);
	let y = attn_decode(cfg, L.attn, xn, cl, st);

	glm5.hc_apply(streams, y, r[0], r[1]);

	r = hc_collapse(cfg, L.hc_ffn, streams);
	xn = torch.rms_norm(r[2], L.ffn_norm, eps);
	L.moe.ids = ids;
	L.moe.vlon = false;
	y = glm5.moe_forward(cfg, L.moe, xn);

	glm5.hc_apply(streams, y, r[0], r[1]);

	return streams;
}

/* Final stream collapse (HyperHead): pre = sigmoid(rms(flat) . fn^T *
 * scale + base) + eps, x = sum_h pre_h * stream_h. (T, D) fp32. */
export function hc_head(cfg, HH, streams) {
	let T = streams.shape()[0];
	let flat = streams.reshape([T, cfg.hc_mult * cfg.hidden_size]);
	let r = flat.mul(flat).mean(-1, true).add(cfg.rms_norm_eps).rsqrt();
	let mixes = torch.matmul(flat.mul(r), HH.fn.t());
	let pre = mixes.mul(HH.scale).add(HH.base).sigmoid().add(cfg.hc_eps);

	return streams.mul(pre.view([T, cfg.hc_mult, 1])).sum(1);
}

/* ------------------------------------------------------------------ */
/* planner + model                                                     */

/* Contiguous layer ranges balanced by checkpoint bytes (embedding on the
 * first device, head on the last). Returns the per-layer device array. */
export function plan_devices(ws, cfg, ngpu, extra_last) {
	let nl = cfg.num_hidden_layers;
	let bytes = [];

	for (let i = 0; i < nl; i++)
		push(bytes, 0);

	let embed = 0, head = 0;

	for (let key in ws.list()) {
		if (index(key, 'layers.') == 0) {
			let idx = +substr(key, 7, index(key, '.', 7) - 7);

			if (idx >= 0 && idx < nl)
				bytes[idx] += ws.info(key).nbytes;
		}
		else if (index(key, 'embed.') == 0) {
			embed += ws.info(key).nbytes;
		}
		else if (index(key, 'head.') == 0 || index(key, 'norm.') == 0 ||
		         index(key, 'hc_head') == 0) {
			head += ws.info(key).nbytes;
		}
	}

	head += extra_last || 0;            // e.g. the drafter that lives with the last device
	let total = embed + head;

	for (let b in bytes)
		total += b;

	// greedy: close a device when adding the next layer overshoots the
	// target by more than leaving it short would
	let target = total / ngpu;
	let devs = [], d = 0, acc = embed;

	for (let i = 0; i < nl; i++) {
		let need = bytes[i] + ((i == nl - 1) ? head : 0);

		if (d < ngpu - 1 && acc > 0 && acc + need / 2 > target && (nl - i) > (ngpu - 1 - d)) {
			d++;
			acc = 0;
		}

		push(devs, d);
		acc += need;
	}

	return devs;
}

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = cfg.num_hidden_layers;
	let devs = opts.devices;

	if (devs == null)
		devs = 1;

	if (opts.layers && opts.layers < nl) {           // partial model (tests)
		nl = opts.layers;
		cfg.num_hidden_layers = nl;
	}

	if (type(devs) != 'array')
		devs = plan_devices(ws, cfg, devs);

	let first = devs[0], last = devs[nl - 1];

	glm5.set_fused_rows(max(256, opts.prefill_rows || 1024));

	ws.upload(function (key) {
		if (index(key, 'layers.') == 0) {
			let idx = +substr(key, 7, index(key, '.', 7) - 7);

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (key == 'embed.weight')
			return first;

		if (index(key, 'head.') == 0 || index(key, 'norm.') == 0 || index(key, 'hc_head') == 0)
			return last;

		return null;   // mtp / vision / aligner: not part of the text model
	});

	let layers = [];

	for (let i = 0; i < nl; i++)
		push(layers, finish_layer(build_layer(ws, cfg, i, devs[i])));

	let embed_t = ws.gpu16('embed.weight', first);

	ws.release('embed.weight', first);

	let model = {
		cfg: cfg, devs: devs, ws: ws, layers: layers, embed_tokens: embed_t, mm_build: mm_build,
		hc_head: {
			fn: ws.gpu('hc_head_fn', last).to('float32').contiguous(),
			base: ws.gpu('hc_head_base', last).to('float32').contiguous(),
			scale: ws.gpu('hc_head_scale', last).to('float32').contiguous()
		},
		norm: ws.gpu16('norm.weight', last),
		lm_head: models.linear(ws, 'head', false, last),

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i = 0; i < nl; i++)
				push(cache.layers, new_attn_cache(this.cfg, this.layers[i].attn, max_seq));

			return cache;
		},

		/* Run token ids (array) appended to the cache. Returns logits
		 * (rows, vocab) fp16 for the last `rows` positions (default 1).
		 * trace: optional array receiving {x: streams, ...} per layer
		 * (CPU copies, the last trace_rows rows). */
		forward_chunk: function (token_ids, cache, rows, trace, trace_rows) {
			let cfg = this.cfg;

			if (FAST_PF && !trace)
				this.to_dev(cache);
			else
				this.to_eager(cache);
			let T = length(token_ids);
			let pos = cache.length;
			let H = cfg.hidden_size;

			if (rows == null)
				rows = 1;

			let ids = torch.from_array(token_ids, 'int64', 'cuda:' + first);
			let h = embed_rows(this.embed_tokens, cache.mm, token_ids, pos, first).to('float32').reshape([T, 1, H])
				.expand([T, 4, H]).contiguous();
			let ids_l = ids;
			let fl = chunk_flags(cache.mm, pos, T);

			for (let i = 0; i < nl; i++) {
				cache.layers[i].vlon = fl.vlon;
				cache.layers[i].ncs = fl.ncs;
			}

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];

				if (i > 0 && L.dev != this.layers[i - 1].dev) {
					h = h.cuda(L.dev);
					ids_l = ids.cuda(L.dev);
				}

				let tr = trace ? {} : null;

				h = block_forward(cfg, L, h, cache.layers[i], pos, ids_l, tr);

				// DSpark taps: stream mean after the target layers, fp16 (saturated)
				if (cache.tap_cb && index(cfg.dspark_taps, i) >= 0) {
					if (!cache.tap_acc)
						cache.tap_acc = [];

					push(cache.tap_acc, h.mean(1).clamp(-65504.0, 65504.0).to('float16'));
				}

				if (trace) {
					let keep = trace_rows ? min(trace_rows, T) : T;

					for (let k in keys(tr))
						tr[k] = tr[k].narrow(0, T - keep, keep).cpu();

					tr.x = h.narrow(0, T - keep, keep).cpu();
					tr.xh = h.narrow(0, 0, min(8, T)).cpu();           // head rows (sink tokens)
					push(trace, tr);
				}
			}

			cache.length = pos + T;

			if (cache.tap_acc) {
				let parts = map(cache.tap_acc, (t) => t.cuda(cache.tap_dev));

				cache.tap_acc = null;
				cache.tap_cb(torch.cat(parts, 1), pos);
			}

			if (rows == 0)
				return null;                 // intermediate prefill chunk: no head

			let x = glm5.rms_norm_f32(hc_head(cfg, this.hc_head, h), this.norm, cfg.rms_norm_eps);

			if (rows < T)
				x = x.narrow(0, T - rows, rows).contiguous();

			return this.lm_head.forward(x);
		},

		/* The decode graphs keep the compressor state in device twins
		 * (S.d); these switch the live representation. */
		to_eager: function (cache) {
			if (cache.dmode != 'dev')
				return;

			for (let i = 0; i < nl; i++) {
				let A = this.layers[i].attn, cl = cache.layers[i];

				if (A.comp)
					comp_to_eager(A.comp, cl.comp, cache.length);

				if (A.idx)
					comp_to_eager(A.idx, cl.idx, cache.length);
			}

			cache.dmode = 'eager';
		},

		to_dev: function (cache) {
			if (cache.dmode == 'dev')
				return;

			for (let i = 0; i < nl; i++) {
				let A = this.layers[i].attn, cl = cache.layers[i];

				if (A.comp)
					comp_to_dev(A.comp, cl.comp, cache.length);

				if (A.idx)
					comp_to_dev(A.idx, cl.idx, cache.length);
			}

			cache.dmode = 'dev';
		},

		/* CUDA-graph decode (engine interface, as glm5.graphs): one graph
		 * per run of layers on a device, position and counters on the
		 * device. sizes: only S = 1. */
		graphs: function (cache, sizes) {
			let model = this;
			let cfg = this.cfg;
			let H = cfg.hidden_size;
			let sts = {};
			let items = [];

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];
				let last = length(items) ? items[length(items) - 1] : null;

				if (!sts[L.dev])
					sts[L.dev] = new_dec_dev(cfg, L.dev);

				if (last && last.dev == L.dev)
					last.end = i + 1;
				else
					push(items, {first: i, end: i + 1, dev: L.dev});
			}

			function body(it) {
				let h = it.h_in;

				for (let i = it.first; i < it.end; i++)
					h = block_decode(cfg, model.layers[i], h, cache.layers[i], sts[it.dev], it.ids_in);

				it.h_out = h;
			}

			// the device twins of a fresh cache start from its (empty) eager state
			cache.dmode = 'eager';
			model.to_dev(cache);

			for (let it in items) {
				it.h_in = torch.zeros([1, 4, H], 'float32', 'cuda:' + it.dev);
				it.ids_in = torch.zeros([1], 'int64', 'cuda:' + it.dev);
				body(it);                                   // eager warm-up
			}

			torch.synchronize();

			for (let it in items) {
				it.graph = torch.graph();
				it.graph.capture_begin(it.dev);
				body(it);
				it.graph.capture_end();
			}

			torch.synchronize();
			cache.dmode = 'eager';                           // the warm-up touched the twins

			function run_ids(tids, pos) {
				model.to_dev(cache);

				for (let d in keys(sts))
					dec_set_pos(sts[d], pos);

				let h = model.embed_tokens.index(tids).to('float32').reshape([1, 1, H])
					.expand([1, 4, H]).contiguous();

				for (let it in items) {
					it.h_in.copy_(h);
					it.ids_in.copy_(tids);
					it.graph.replay();
					h = it.h_out;
				}

				cache.length = pos + 1;

				let x = glm5.rms_norm_f32(hc_head(cfg, model.hc_head, h), model.norm, cfg.rms_norm_eps);

				return {logits: model.lm_head.forward(x), hidden: null};
			}

			return {
				has: function (S) {return S == 1;},

				run: function (ids, pos) {
					return run_ids(torch.from_array(ids, 'int64', 'cuda:' + model.devs[0]), pos);
				},

				run_dev: function (ids_dev, pos) {
					return run_ids(ids_dev.view([1]), pos);
				}
			};
		},

		/* Recurrent state outside the KV rows, for the engine's prompt
		 * snapshots: the compressors' incomplete windows, their overlap
		 * halves and entry counts. The state objects are replaced, never
		 * mutated in place, so holding the references is a snapshot; pool
		 * entries below the snapshot length are never rewritten by later tokens;
		 * the KV window rows live in a ring and are copied (kw). */
		aux_get: function (cache) {
			let out = [];

			this.to_eager(cache);

			for (let cl in cache.layers) {
				let e = [];

				for (let S in [ cl.comp, cl.idx ])
					push(e, S ? {n: S.n, bk: S.buf_kv, bg: S.buf_gate, ok: S.o_kv, og: S.o_gate} : null);

				push(e, {kw: kv_window_get(cl, cache.length, this.cfg.window)});
				push(out, e);
			}

			return out;
		},

		aux_set: function (cache, snap) {
			cache.dmode = 'eager';

			for (let cl in cache.layers) {
				if (cl.comp) cl.comp.dmode = false;
				if (cl.idx) cl.idx.dmode = false;
			}

			for (let i = 0; i < length(cache.layers); i++) {
				let cl = cache.layers[i];
				let which = [ cl.comp, cl.idx ];

				kv_window_set(cl, cache.length, snap[i][2].kw);

				for (let k = 0; k < 2; k++) {
					let S = which[k], v = snap[i][k];

					if (!S)
						continue;

					S.n = v.n;
					S.buf_kv = v.bk;
					S.buf_gate = v.bg;
					S.o_kv = v.ok;
					S.o_gate = v.og;
				}
			}
		},

		/* Host-RAM prompt-cache store (engine hooks, see engine.uc store_put / store_restore):
		 * the whole cache of n tokens as pinned host tensors, and back. snap_aux: the engine's
		 * snapshot aux state (device refs) when it is a prefix of the live state. */
		cache_bytes: function (cache, n) {
			let b = 0;

			for (let i = 0; i < length(cache.layers); i++)
				b += store_layer_bytes(this.cfg, this.layers[i].attn, n);

			if (this.dspark && this.dspark.ring_bytes)
				b += this.dspark.ring_bytes(n);

			return b;
		},

		new_slot: function (max_seq) {
			return {layers: map(this.layers, (L) => new_slot_layer(L.attn, max_seq))};
		},

		cache_export: function (cache, n, put, snap_aux, dst) {
			let aux = this.aux_get(cache);
			let layers = [];

			for (let i = 0; i < length(cache.layers); i++)
				push(layers, {pools: pools_to_host(cache.layers[i], put, dst ? dst.layers[i] : null),
					aux: auxe_to_host(aux[i], put)});

			return {
				layers: layers,
				snap: snap_aux ? map(snap_aux, (e) => auxe_to_host(e, put)) : null,
				drafter: (this.dspark && this.dspark.ring_export) ? this.dspark.ring_export(n, put) : null
			};
		},

		cache_import: function (cache, em, start, live) {
			let dev_aux = [];

			for (let i = 0; i < length(cache.layers); i++) {
				let cl = cache.layers[i];
				let ae = live ? em.layers[i].aux : em.snap[i];

				pools_from_host(cl, em.layers[i].pools, aux_rows(ae));
				push(dev_aux, auxe_to_dev(ae, cl.dev));
			}

			cache.length = start;
			this.aux_set(cache, dev_aux);

			if (em.drafter && this.dspark && this.dspark.ring_import)
				this.dspark.ring_import(em.drafter);

			return {snap: em.snap ? map(em.snap, (e, i) => auxe_to_dev(e, cache.layers[i].dev)) : null};
		},

		aux_reset: function (cache) {
			cache.dmode = 'eager';

			for (let cl in cache.layers) {
				if (cl.comp) cl.comp.dmode = false;
				if (cl.idx) cl.idx.dmode = false;
			}

			for (let cl in cache.layers)
				for (let S in [ cl.comp, cl.idx ])
					if (S) {
						S.n = 0;
						S.buf_kv = S.buf_gate = S.o_kv = S.o_gate = null;
					}
		},

		/* Engine interface (as glm5.prefill_pipeline): chunked prefill of
		 * ids at cache.length as a wavefront over the device stages — one
		 * layer per stage per round, so every GPU works on a different
		 * chunk (a layer sees its chunks in order, which is all the
		 * compressor / window state needs). opts: chunk, cuts (extra chunk
		 * boundaries), on_aux(chunk_end) when a chunk left the last layer
		 * (the engine snapshots there: the boundary chunk is the last one
		 * in flight past that point only if cuts isolate it, so the engine
		 * asks for a drain via cuts + sequential rounds). */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let cfg = this.cfg;
			let H = cfg.hidden_size;
			let chunk = opts.chunk || 1024;

			if (FAST_PF)
				this.to_dev(cache);
			else
				this.to_eager(cache);
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let p = start + chunk; p < start + total; p += chunk)
				push(bounds, p);

			// an image (START .. END) is prefilled as exactly one chunk
			if (cache.mm) {
				let kept = [];

				for (let b in bounds) {
					let inside = false;

					for (let sp in cache.mm.spans)
						if (b > sp.nc0 && b < sp.hi)
							inside = true;

					if (!inside)
						push(kept, b);
				}

				bounds = kept;

				for (let sp in cache.mm.spans)
					for (let c in [ sp.nc0, sp.hi ])
						if (c > start && c < start + total && index(bounds, c) < 0)
							push(bounds, c);
			}

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			let stages = [];

			for (let i = 0; i < nl; i++) {
				if (!length(stages) || stages[length(stages) - 1].dev != this.layers[i].dev)
					push(stages, {first: i, end: i + 1, dev: this.layers[i].dev});
				else
					stages[length(stages) - 1].end = i + 1;
			}

			let ns = length(stages);
			let active = [];
			let next = 0, prev_end = start, done = 0, result = null;
			let snap_at = (opts.cuts && length(opts.cuts)) ? opts.cuts[0] : 0;

			for (let s = 0; s < ns; s++)
				push(active, null);

			function admit() {
				// a snapshot boundary must see the layer state exactly at that
				// chunk's end: hold the next chunk back until it drained
				if (active[0] != null || next >= length(bounds))
					return;

				if (snap_at && prev_end == snap_at) {
					for (let c in active)
						if (c)
							return;
				}

				let end = bounds[next++];
				let pos = prev_end;
				let T = end - pos;
				let cids = slice(ids, pos - start, end - start);
				let dev0 = stages[0].dev;
				let idt = torch.from_array(cids, 'int64', 'cuda:' + dev0);

				prev_end = end;
				let fl = chunk_flags(cache.mm, pos, T);

				active[0] = {pos: pos, T: T, end: end, layer: stages[0].first, ctx: null, ids: idt,
					vlon: fl.vlon, ncs: fl.ncs,
					h: embed_rows(model.embed_tokens, cache.mm, cids, pos, dev0).to('float32').reshape([T, 1, H])
						.expand([T, 4, H]).contiguous()};
			}

			function tnow() {
				let c = clock(true);

				return c[0] + c[1] / 1e9;
			}

			function round() {
				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null) {
						let t0 = HP ? tnow() : 0;

						c.ctx = block_a(cfg, model.layers[c.layer], c.h, flag_layer(cache.layers[c.layer], c), c.pos, c.ids, null);

						if (HP) HP.a += tnow() - t0;
					}
				}

				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.ctx) {
						let t0 = HP ? tnow() : 0;

						c.h = block_b(cfg, model.layers[c.layer], c.ctx);
						c.ctx = null;
						c.layer++;

						let t1 = HP ? tnow() : 0;

						if (HP) HP.b += t1 - t0;

						// queue this stage's next layer BEFORE the host blocks on another
						// stage's expert counts
						if (c.layer < stages[s].end)
							c.ctx = block_a(cfg, model.layers[c.layer], c.h, flag_layer(cache.layers[c.layer], c), c.pos, c.ids, null);

						if (HP) HP.a += tnow() - t1;
					}
				}

				for (let s = ns - 1; s >= 0; s--) {
					let c = active[s];

					if (!c || c.layer < stages[s].end)
						continue;

					if (s == ns - 1) {
						if (c.end == start + total) {
							let x = glm5.rms_norm_f32(hc_head(cfg, model.hc_head, c.h)
								.narrow(0, c.T - 1, 1).contiguous(), model.norm, cfg.rms_norm_eps);

							result = model.lm_head.forward(x);
						}

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;

						if (opts.on_aux)
							opts.on_aux(c.end);
					}
					else if (active[s + 1] == null) {
						// the copy runs on a side stream of the SOURCE device (ATen copies cross-device
						// on the source's current stream: 67 MB = ~24 ms during which that GPU could
						// compute the next chunk); the old tensors stay referenced until the chunk
						// is done (the allocator knows nothing about the side stream)
						if (SIDE_HO) {
							torch.stream_side(stages[s].dev, 1);

							let nh = c.h.cuda(stages[s + 1].dev);
							let ni = c.ids.cuda(stages[s + 1].dev);

							torch.stream_main(stages[s].dev, 1);

							c.keep = c.keep || [];
							push(c.keep, c.h, c.ids);
							c.h = nh;
							c.ids = ni;
						}
						else {
							c.h = c.h.cuda(stages[s + 1].dev);
							c.ids = c.ids.cuda(stages[s + 1].dev);
						}

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
				hidden: function () {return null;}
			};
		},

		/* Prefill ids in chunks of `chunk` tokens (default 512). */
		prefill: function (token_ids, cache, chunk, rows, trace, trace_rows) {
			let n = length(token_ids);

			chunk = chunk || 512;

			for (let p = 0; p < n; p += chunk) {
				let e = min(p + chunk, n);
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
