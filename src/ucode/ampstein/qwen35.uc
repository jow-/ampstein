// ucode/qwen35.uc — Qwen3.5 / Qwen3.8 hybrid architecture.
//
// The model mixes two layer types (config text_config.layer_types):
//
//   - "linear_attention": GatedDeltaNet — a gated delta-rule recurrent
//     layer (Qwen3-Next lineage): fused q/k/v + output-gate + beta/decay
//     projections, a short causal conv1d with silu, and a per-head
//     fp32 recurrent state S (num_v_heads, k_head_dim, v_head_dim)
//     updated by the gated delta rule.
//
//   - "full_attention": standard GQA attention with per-head Q/K
//     RMSNorm (weight+1), partial NEOX RoPE (rotary_dim =
//     head_dim * partial_rotary_factor) and a per-head sigmoid output
//     gate packed into q_proj (out_features = 2 * num_q_heads *
//     head_dim; per head the first head_dim channels are q, the next
//     head_dim the gate).
//
// The language model only: the vision tower lives in qwen35_vision.uc and
// feeds this file through cache.mm (image spans -> MRoPE positions and
// embedding rows, see the multimodal section below); the MTP draft head is
// built when the checkpoint has one.
//
// Numerics contract (matches exllamav3's reference graph):
//   - GDN projections come out fp16; beta/decay gates and the delta
//     rule run in fp32; the recurrent state is fp32.
//   - The delta rule is the SEQUENTIAL recurrence (the exact math of
//     exllamav3's torch_recurrent_gated_delta_rule reference), not the
//     chunked parallel form: bit-comparable per token, slower prefill.
//     A chunked/fused prefill kernel is future work.
//
// Multi-GPU: opts.devices assigns one device index per layer (array of
// num_hidden_layers entries, or a single index for everything).
// Activations cross device boundaries with explicit copies at layer
// hand-off; every cache/state tensor lives on its layer's device.

import * as models from './models.uc';
import * as weights from './weights.uc';
import * as sampler from './sampler.uc';
import * as math from 'math';
import {readfile} from 'fs';

function dev_of_dev(d) {
	return (d == null) ? 0 : d;
}

/* ------------------------------------------------------------------ */
/* config                                                              */

export function load_config(model_dir) {
	let raw = json(readfile(model_dir + '/config.json'));
	let t = raw.text_config || raw;

	let rp = t.rope_parameters || {};
	let partial = rp.partial_rotary_factor != null ?
		rp.partial_rotary_factor : (t.partial_rotary_factor || 1.0);

	let cfg = {
		hidden_size: t.hidden_size,
		num_hidden_layers: t.num_hidden_layers,
		layer_types: t.layer_types,
		full_attention_interval: t.full_attention_interval,
		head_dim: t.head_dim,
		num_q_heads: t.num_attention_heads,
		num_kv_heads: t.num_key_value_heads,
		ln_num_k_heads: t.linear_num_key_heads,
		ln_k_head_dim: t.linear_key_head_dim,
		ln_num_v_heads: t.linear_num_value_heads,
		ln_v_head_dim: t.linear_value_head_dim,
		conv_kernel: t.linear_conv_kernel_dim,
		intermediate_size: t.intermediate_size,
		rms_norm_eps: t.rms_norm_eps,
		vocab_size: t.vocab_size,
		rope_theta: rp.rope_theta != null ? rp.rope_theta : t.rope_theta,
		rotary_dim: math.floor(t.head_dim * partial),
		bos_token_id: t.bos_token_id,
		eos_token_id: t.eos_token_id,
		// multimodal (vision) token ids + MRoPE layout; absent on
		// text-only checkpoints
		image_token_id: raw.image_token_id,
		vision_start_id: raw.vision_start_token_id,
		vision_end_id: raw.vision_end_token_id,
		vision_merge: raw.vision_config ?
			raw.vision_config.spatial_merge_size : null,
		mrope_section: rp.mrope_section
	};

	// derived GDN geometry
	cfg.ln_k_dim = cfg.ln_num_k_heads * cfg.ln_k_head_dim;
	cfg.ln_v_dim = cfg.ln_num_v_heads * cfg.ln_v_head_dim;
	cfg.fdim_qkv = 2 * cfg.ln_k_dim + cfg.ln_v_dim;
	cfg.ln_v_groups = cfg.ln_num_v_heads / cfg.ln_num_k_heads;

	for (let i = 0; i < length(cfg.layer_types); i++)
		if (cfg.layer_types[i] != 'linear_attention' &&
		    cfg.layer_types[i] != 'full_attention')
			die(`qwen35: unknown layer type '${cfg.layer_types[i]}'`);

	return cfg;
}

/* ------------------------------------------------------------------ */
/* RoPE (partial, NEOX)                                                */

/* Inverse frequencies for the FIRST rotary_dim channels of each head:
 * (rotary_dim/2,) float32 CPU tensor. */
export function rope_inv_freq(cfg) {
	let dim = cfg.rotary_dim;
	let theta = cfg.rope_theta;
	let freqs = [];

	for (let i = 0; i < dim; i += 2)
		push(freqs, 1.0 / math.pow(theta, (i * 1.0) / dim));

	return torch.from_array(freqs);
}

/* cos/sin tables for positions 0..max_seq-1: (max_seq, rotary_dim/2)
 * float32, on `dev`. Only the rotary channels need entries — the
 * pass-through channels never see them. */
export function rope_cache(cfg, max_seq, dev) {
	let inv = rope_inv_freq(cfg);
	let half = cfg.rotary_dim / 2;

	let positions = [];
	for (let i = 0; i < max_seq; i++)
		push(positions, i * 1.0);

	let t = torch.from_array(positions).view([max_seq, 1]);
	let freqs = torch.matmul(t, inv.unsqueeze(0));   // (max_seq, half)

	let cos = freqs.cos().cuda(dev);
	let sin = freqs.sin().cuda(dev);

	return {cos: cos, sin: sin};
}

/* ------------------------------------------------------------------ */
/* Multimodal: image spans, MRoPE positions, embedding overlay          */
/*                                                                     */
/* An image occupies a run of n = (gh/2)*(gw/2) consecutive prompt     */
/* positions (the vision tower's merged grid). Those rows of the       */
/* embedding stream are the tower's output instead of the token        */
/* embedding. The text layers use 3-axis MRoPE: text tokens advance    */
/* all three axes together, image tokens carry (t, row, col) offsets   */
/* from the image's base position, and the text after an image resumes */
/* at base + max(gh, gw) — so KV slot and rope position diverge by a   */
/* constant offset (<= 0) after each image.                            */
/*                                                                     */
/* Identity: the engine's token history stores a MARKER id               */
/* (>= MARK_BASE, derived from the image bytes) at every image         */
/* position, so prompt-prefix reuse can never confuse two images; the  */
/* gather below is fed the plain image-pad id instead.                 */

export const MARK_BASE = 1073741824;     // 1 << 30, above any vocabulary

/* Marker id for an image identified by a hex digest string. */
export function mm_marker(digest) {
	return MARK_BASE + hex(substr(digest, 0, 7));
}

/* Image spans for a prompt. images: array of {start (absolute prompt
 * position of the first image row), grid [1, gh, gw] (patch grid),
 * emb ((n, hidden) fp16 tensor, n = gh*gw/merge^2)}. Returns null for
 * no images, else the object the rope/embedding helpers take. */
export function mm_build(cfg, images) {
	if (!images || !length(images))
		return null;

	let m = cfg.vision_merge;

	if (!m)
		die('qwen35: checkpoint has no vision_config (text-only model)');

	let spans = [];
	let off = 0;
	let prev_end = 0;

	for (let im in images) {
		let gh = im.grid[1] / m, gw = im.grid[2] / m;
		let n = gh * gw;

		if (im.emb && im.emb.shape()[0] != n)
			die(`qwen35: image embedding has ${im.emb.shape()[0]} rows, grid needs ${n}`);

		if (im.start < prev_end)
			die('qwen35: overlapping or unordered image spans');

		let base = im.start + off;
		let ext = (gh > gw) ? gh : gw;

		if (ext < 1)
			ext = 1;

		push(spans, {
			start: im.start, n: n, gh: gh, gw: gw, emb: im.emb,
			base: base, off_before: off,
			off_after: base + ext - (im.start + n)
		});

		off = base + ext - (im.start + n);
		prev_end = im.start + n;
	}

	return {spans: spans, memo: {}, inv: {}, masks: {}, end_off: off};
}

/* RoPE offset for a TEXT token at absolute position `pos`: the rope
 * position is pos + mm_off(mm, pos). */
export function mm_off(mm, pos) {
	if (!mm)
		return 0;

	let off = 0;

	for (let sp in mm.spans) {
		if (pos < sp.start + sp.n)
			break;

		off = sp.off_after;
	}

	return off;
}

/* Spans overlapping [pos, pos + S): array of [span, lo, hi]. */
function mm_overlap(mm, pos, S) {
	let out = [];

	if (!mm)
		return out;

	for (let sp in mm.spans) {
		let lo = (sp.start > pos) ? sp.start : pos;
		let hi = (sp.start + sp.n < pos + S) ? sp.start + sp.n : pos + S;

		if (lo < hi)
			push(out, [ sp, lo, hi ]);
	}

	return out;
}

/* Token ids of a chunk made safe for the embedding gather: image
 * positions (markers) become the image-pad id. */
export function mm_safe_ids(cfg, mm, token_ids, pos) {
	let S = length(token_ids);
	let ov = mm_overlap(mm, pos, S);

	if (!length(ov))
		return token_ids;

	let ids = slice(token_ids);

	for (let o in ov)
		for (let i = o[1]; i < o[2]; i++)
			ids[i - pos] = cfg.image_token_id;

	return ids;
}

/* Overwrite the image rows of an embedding chunk h (1, S, hidden) for
 * absolute positions [pos, pos + S) with the vision embeddings. dev is
 * h's device index. */
function mm_overlay(h, mm, pos, S, dev) {
	for (let o in mm_overlap(mm, pos, S)) {
		let sp = o[0];

		h.narrow(1, o[1] - pos, o[2] - o[1]).copy_(
			sp.emb.narrow(0, o[1] - sp.start, o[2] - o[1])
				.unsqueeze(0).cuda(dev));
	}
}

/* Embedding rows of a token chunk at absolute position pos: the token
 * embedding, image rows replaced by the vision embeddings. */
export function embed_rows(model, mm, token_ids, pos) {
	let S = length(token_ids);
	let dev = model.devs[0];

	if (!mm)
		return model.embed_tokens.index(torch.from_array(token_ids, 'int64',
			'cuda:' + dev).view([1, S]));

	let t = torch.from_array(mm_safe_ids(model.cfg, mm, token_ids, pos),
		'int64', 'cuda:' + dev).view([1, S]);
	let h = model.embed_tokens.index(t);

	mm_overlay(h, mm, pos, S, dev);

	return h;
}

/* Section masks of the interleaved MRoPE: channel i of the rotary
 * half follows the H axis for i % 3 == 1 (i < 3*sec[1]), the W axis
 * for i % 3 == 2 (i < 3*sec[2]), else T. Three (1, half) fp32 device
 * tensors, cached per device. */
function mm_masks(cfg, mm, dev) {
	if (mm.masks[dev])
		return mm.masks[dev];

	let sec = cfg.mrope_section || [ cfg.rotary_dim / 2, 0, 0 ];
	let half = cfg.rotary_dim / 2;
	let mt = [], mh = [], mw = [];

	for (let i = 0; i < half; i++) {
		let h = (i % 3 == 1 && i < 3 * sec[1]);
		let w = (i % 3 == 2 && i < 3 * sec[2]);

		push(mh, h ? 1.0 : 0.0);
		push(mw, w ? 1.0 : 0.0);
		push(mt, (h || w) ? 0.0 : 1.0);
	}

	let dv = 'cuda:' + dev;
	let r = [
		torch.from_array(mt, 'float32', dv).view([1, half]),
		torch.from_array(mh, 'float32', dv).view([1, half]),
		torch.from_array(mw, 'float32', dv).view([1, half])
	];

	mm.masks[dev] = r;

	return r;
}

/* cos/sin rows (S, rotary_dim/2) fp32 for the chunk [pos, pos + S) of a
 * layer whose RoPE tables live in cl (cl.cos / cl.sin). Without image
 * tokens in range this is a narrow of the table at the shifted
 * position; with them the 3-axis positions are built explicitly. */
export function rope_rows(cfg, mm, cl, pos, S) {
	if (!mm) {
		return {cos: cl.cos.narrow(0, pos, S), sin: cl.sin.narrow(0, pos, S)};
	}

	let ov = mm_overlap(mm, pos, S);

	if (!length(ov)) {
		let o = pos + mm_off(mm, pos);

		return {cos: cl.cos.narrow(0, o, S), sin: cl.sin.narrow(0, o, S)};
	}

	let dev = cl.dev;
	let key = `${pos}:${S}:${dev}`;

	if (mm.memo[key])
		return mm.memo[key];

	// 3-axis positions per row
	let pt = [], ph = [], pw = [];
	let k = 0;

	for (let i = pos; i < pos + S; i++) {
		let sp = null;

		for (let o in ov) {
			if (i >= o[0].start && i < o[0].start + o[0].n) {
				sp = o[0];
				break;
			}
		}

		if (sp) {
			let r = i - sp.start;

			push(pt, sp.base);
			push(ph, sp.base + int(r / sp.gw));
			push(pw, sp.base + r % sp.gw);
		}
		else {
			let p = i + mm_off(mm, i);

			push(pt, p);
			push(ph, p);
			push(pw, p);
		}
	}

	if (!mm.inv[dev])
		mm.inv[dev] = rope_inv_freq(cfg).cuda(dev).view([1, 1, -1]);

	let half = cfg.rotary_dim / 2;
	let all = [];

	for (let a in [ pt, ph, pw ])
		for (let v in a)
			push(all, v * 1.0);

	let P = torch.from_array(all, 'float32', 'cuda:' + dev).view([3, S, 1]);
	let ang3 = P.mul(mm.inv[dev]);                       // (3, S, half)
	let m = mm_masks(cfg, mm, dev);
	let ang = ang3.narrow(0, 0, 1).squeeze(0).mul(m[0])
		.add(ang3.narrow(0, 1, 1).squeeze(0).mul(m[1]))
		.add(ang3.narrow(0, 2, 1).squeeze(0).mul(m[2]));

	let r = {cos: ang.cos(), sin: ang.sin()};

	mm.memo[key] = r;

	return r;
}

/* ------------------------------------------------------------------ */
/* norms                                                               */

/* RMSNorm with constant_bias 1.0 (Qwen3-style: effective weight is
 * weight + 1). w1 is the precomputed (weight + 1) fp16 tensor. */
export function rmsnorm_bias1(x, w1, eps) {
	// fused kernel (fp32 variance, fp16 cast before the weight multiply
	// — the same numerics as the composite below) for contiguous input;
	// the per-head q/k norms see strided views and take the composite
	if (x.is_contiguous())
		return torch.rms_norm(x, w1, eps);

	let f = x.to('float32');
	let ms = f.mul(f).mean(-1, true).add(eps).rsqrt();

	return f.mul(ms).to('float16').mul(w1);
}

/* l2-normalize the last dim (eps 1e-6) — the GDN q/k normalization. */
export function l2norm(x) {
	let f = x.mul(x).sum(-1, true).add(1e-6).rsqrt();

	return x.mul(f);
}

/* ------------------------------------------------------------------ */
/* GatedDeltaNet                                                       */

/* Repeat each head `rep` times CONSECUTIVELY (GQA grouping: output
 * head j reads source head floor(j / rep)): (S, H, d) -> (S, H*rep, d). */
export function repeat_heads(x, rep) {
	let s = x.shape();

	return x.unsqueeze(2)
		.expand([s[0], s[1], rep, s[2]])
		.reshape([s[0], s[1] * rep, s[2]]);
}

/* The gated delta rule, sequential form (fp32). Exact per-token math
 * of exllamav3's torch_recurrent_gated_delta_rule reference:
 *
 *   kv_mem = S^T k                    (per v head)
 *   v_new  = v - g_t * kv_mem
 *   S      = g_t * S + beta_t * k (x) v_new
 *   out    = S^T q * scale
 *
 * q, k: (S, Hv, dk) float32 (k already head-repeated to Hv); v:
 * (S, Hv, dv) float32; beta, gexp: (S, Hv) float32 (gexp = exp(g));
 * state: (Hv, dk, dv) float32 — UPDATED IN PLACE (mul_/add_ keep the
 * tensor identity, so the cache's tensor stays live); returns
 * (S, Hv, dv) float32. Batch size 1 only. */
export function delta_rule_scan(q, k, v, beta, gexp, state, scale, dev) {
	let S = q.shape()[0];
	let Hv = v.shape()[1];
	let dk = k.shape()[2];
	let dv = v.shape()[2];

	let out = torch.zeros([S, Hv, dv], 'float32', 'cuda:' + dev_of_dev(dev));

	for (let t = 0; t < S; t++) {
		let kt = k.narrow(0, t, 1).squeeze(0);        // (Hv, dk)
		let qt = q.narrow(0, t, 1).squeeze(0);        // (Hv, dk)
		let vt = v.narrow(0, t, 1).squeeze(0);        // (Hv, dv)
		let gt = gexp.narrow(0, t, 1).squeeze(0)      // (Hv, 1)
			.unsqueeze(-1);
		let bt = beta.narrow(0, t, 1).squeeze(0)      // (Hv, 1)
			.unsqueeze(-1);
		let gt3 = gt.unsqueeze(-1);                   // (Hv, 1, 1)

		// kv_mem = S^T k : (Hv, dv)
		let kv_mem = torch.matmul(state.transpose(1, 2),
			kt.unsqueeze(-1)).squeeze(-1);

		let v_new = vt.sub(kv_mem.mul(gt));

		// upd = k (x) (beta * v_new) : (Hv, dk, dv)
		let upd = kt.unsqueeze(-1).mul(v_new.mul(bt).unsqueeze(-2));

		state.mul_(gt3).add_(upd);

		let o = torch.matmul(state.transpose(1, 2),
			qt.unsqueeze(-1)).squeeze(-1).mul(scale);

		out.narrow(0, t, 1).copy_(o.unsqueeze(0));
	}

	return out;
}

/* One GDN ("linear_attention") layer. x: (1, S, hidden) fp16 on the
 * layer's device. L carries the linears, norms and state tensors. */
/* block-parallel GDN scan: opt-in (AMP_GDN_CHUNK=1), from this chunk
 * length on */
const GDN_CHUNK = (getenv('AMP_GDN_CHUNK') == '1');
const GDN_CHUNK_MIN = 64;

/* opt-out: separate a / b projections + gates for decode rows */
const GDN_NO_ABG = getenv('AMP_GDN_NO_ABG') != null;

export function gdn_forward(cfg, L, x, cl) {
	let S = x.shape()[1];
	let Hv = cfg.ln_num_v_heads;

	// projections (fp16)
	let qkv = L.in_proj_qkv.forward(x);     // (1,S,fdim_qkv)
	let z = L.in_proj_z.forward(x);         // (1,S,ln_v_dim)
	// fused: beta = sigmoid(b), gexp = exp(-exp(A_log) * softplus(a + dt_bias)).
	// Decode and verify rows (S <= 8) with fp16 a / b weights compute the
	// two projections in the same launch: at 4 rows the matmul falls on a
	// 28 us cutlass kernel each (two per GDN layer)
	let gates;

	if (S <= 8 && !GDN_NO_ABG && L.in_proj_a.weight != null &&
	    L.in_proj_a.bias == null && L.in_proj_b.bias == null &&
	    L.in_proj_b.weight != null) {
		gates = torch.gdn_ab_gates(x.contiguous(), L.in_proj_a.weight,
			L.in_proj_b.weight, L.dt_bias, L.neg_exp_a_log);
	}
	else {
		let b = L.in_proj_b.forward(x);         // (1,S,Hv)
		let a = L.in_proj_a.forward(x);         // (1,S,Hv)

		gates = torch.gdn_gates(a.contiguous(), b.contiguous(), L.dt_bias,
			L.neg_exp_a_log);
	}

	// speculative verify (the engine set cl.spec_w = W and this is the
	// W-token verify step): keep the state after each of the first W-1
	// tokens — the conv state per candidate here, from the window
	// [conv_state | inputs] before gdn_prep advances it, the recurrent
	// state in the scan — so a rejection after token a restores them
	// (engine: copy_many) instead of re-running the accepted tokens
	let spec = cl.spec_w && S == cl.spec_w;

	// fused: causal conv1d + silu + q/k/v split + fp32 + q/k l2norm,
	// conv state updated in place — and, for a verify window, the
	// per-token conv snapshots (cl.spec_conv, at most 4) written by the
	// same state kernel instead of a cat + one copy each
	let snaps = null;

	if (spec) {
		snaps = (S - 1 <= 4) ? slice(cl.spec_conv, 0, S - 1) : null;

		if (!snaps) {
			let km1 = cl.conv_state.shape()[2];
			let win = torch.cat([ cl.conv_state, qkv.transpose(1, 2) ], 2);

			for (let t = 0; t < S - 1; t++)
				cl.spec_conv[t].copy_(win.narrow(2, t + 1, km1));
		}
	}

	let qkv3 = torch.gdn_prep(qkv.contiguous(), cl.conv_state, L.conv_w,
		cfg.ln_num_k_heads, Hv, cfg.ln_k_head_dim, snaps);

	// delta rule (fp32 state): the sequential kernel; the 64-token-block
	// WY/UT kernel (torch.gdn_chunk, same math, tests/gdn_scan.uc) is
	// opt-in via AMP_GDN_CHUNK=1 — its fp32 CUDA-core block matmuls are
	// still slower than the sequential scan (4.3 vs 1.9 ms per 1024
	// tokens, docs/LESSONS.md L63); it needs register tiling / tensor cores
	let scan = (GDN_CHUNK && S >= GDN_CHUNK_MIN && cfg.ln_k_head_dim == 128 &&
		cfg.ln_v_head_dim == 128) ? torch.gdn_chunk : torch.gdn_scan;
	let core = spec ?
		torch.gdn_scan(qkv3[0], qkv3[1], qkv3[2], gates[0], gates[1],
			cl.rec_state, 1.0 / math.sqrt(cfg.ln_k_head_dim), cl.spec_states) :
		scan(qkv3[0], qkv3[1], qkv3[2], gates[0], gates[1],
			cl.rec_state, 1.0 / math.sqrt(cfg.ln_k_head_dim));

	// fused: gated RMSNorm over v_head_dim, * silu(z)
	let gated = torch.gdn_out_norm(core, z.contiguous(), L.norm_w,
		cfg.rms_norm_eps);

	return L.out_proj.forward(gated);
}

/* Reference GDN layer: the composite ATen form of every fused step above
 * (gates, conv1d + silu + split + l2norm, gated norm). Kept for the
 * fused-vs-composite test (tests/gdn_fused.uc); the k/v head dims must
 * be equal for the fused path (gdn_prep takes one head dim). */
export function gdn_forward_ref(cfg, L, x, cl) {
	let B = 1, S = x.shape()[1];
	let eps = cfg.rms_norm_eps;

	// projections (fp16)
	let qkv = L.in_proj_qkv.forward(x);     // (1,S,fdim_qkv)
	let z = L.in_proj_z.forward(x);         // (1,S,ln_v_dim)
	let b = L.in_proj_b.forward(x);         // (1,S,Hv)
	let a = L.in_proj_a.forward(x);         // (1,S,Hv)

	// gates (fp32): beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
	let beta = b.to('float32').sigmoid();
	let g = a.to('float32').add(L.dt_bias).softplus().mul(L.neg_exp_a_log);

	// causal conv1d (kernel taps) + silu, channel-major
	let xch = qkv.transpose(1, 2);                       // (1, D, S)
	let padded = torch.cat([cl.conv_state, xch], 2);     // (1, D, k-1+S)
	let D = cfg.fdim_qkv;
	let k = cfg.conv_kernel;

	let acc = L.conv_w.narrow(1, 0, 1).mul(padded.narrow(2, 0, S));

	for (let j = 1; j < k; j++)
		acc = acc.add(L.conv_w.narrow(1, j, 1)
			.mul(padded.narrow(2, j, S)));

	let conv_out = acc.silu();                           // (1, D, S)

	// state update: keep the last k-1 inputs
	cl.conv_state.copy_(padded.narrow(2, S, k - 1));

	// split q/k/v (token-major), fp32, l2-normalize q/k
	let qkv_t = conv_out.transpose(1, 2);                // (1, S, D)
	let q = qkv_t.narrow(-1, 0, cfg.ln_k_dim)
		.view([S, cfg.ln_num_k_heads, cfg.ln_k_head_dim])
		.to('float32');
	let kk = qkv_t.narrow(-1, cfg.ln_k_dim, cfg.ln_k_dim)
		.view([S, cfg.ln_num_k_heads, cfg.ln_k_head_dim])
		.to('float32');
	let v = qkv_t.narrow(-1, 2 * cfg.ln_k_dim, cfg.ln_v_dim)
		.view([S, cfg.ln_num_v_heads, cfg.ln_v_head_dim])
		.to('float32');

	q = l2norm(q);
	kk = l2norm(kk);

	// delta rule (fp32 state): one fused launch for the whole chunk
	// (torch.gdn_scan; the kernel maps v head j to q/k head j // groups
	// itself). delta_rule_scan above is the per-token reference it is
	// tested against (tests/gdn_scan.uc).
	let core = torch.gdn_scan(q.contiguous(), kk.contiguous(), v.contiguous(),
		beta.view([S, cfg.ln_num_v_heads]).contiguous(),
		g.exp().view([S, cfg.ln_num_v_heads]).contiguous(), cl.rec_state,
		1.0 / math.sqrt(cfg.ln_k_head_dim));

	// gated RMSNorm over v_head_dim: norm(core) * w, then * silu(z)
	let f = core;
	let ms = f.mul(f).mean(-1, true).add(eps).rsqrt();
	let n = f.mul(ms).to('float16').mul(L.norm_w);
	let gated = n.mul(z.view([1, S, cfg.ln_num_v_heads, cfg.ln_v_head_dim])
		.silu());

	return L.out_proj.forward(gated.view([1, S, cfg.ln_v_dim]));
}

/* One full-attention layer. */
export function attn_forward(cfg, L, x, cos, sin, cl, pos) {
	let S = x.shape()[1];
	let eps = cfg.rms_norm_eps;
	let nh = cfg.num_q_heads;
	let kvh = cfg.num_kv_heads;
	let hd = cfg.head_dim;

	// q_proj emits q and the output gate interleaved per head:
	// (1, S, nh, 2*hd) -> q = first hd channels, g = next hd
	let qg = L.q_proj.forward(x).view([1, S, nh, 2 * hd]);
	let q = qg.narrow(-1, 0, hd);
	let g = qg.narrow(-1, hd, hd);

	let kk = L.k_proj.forward(x).view([1, S, kvh, hd]);
	let v = L.v_proj.forward(x).view([1, S, kvh, hd]);

	// per-head Q/K RMSNorm (weight + 1) + partial RoPE, one fused kernel
	// each (the q half of the interleaved projection is read strided);
	// the composite form is rmsnorm_bias1 + rope_apply
	q = torch.qk_norm_rope(q, L.q_norm_w, cos, sin, cfg.rotary_dim, eps);
	kk = torch.qk_norm_rope(kk, L.k_norm_w, cos, sin, cfg.rotary_dim, eps);

	// to (B, H, S, hd)
	q = q.transpose(1, 2);
	kk = kk.transpose(1, 2);
	v = v.transpose(1, 2);

	let K, V;

	if (cl) {
		let len = pos + S;

		cl.k_cache.narrow(2, pos, S).copy_(kk);
		cl.v_cache.narrow(2, pos, S).copy_(v);

		K = cl.k_cache.narrow(2, 0, len);
		V = cl.v_cache.narrow(2, 0, len);
	} else {
		K = kk;
		V = v;
	}

	// FlashAttention-2 (GQA native): its causal mask is aligned
	// bottom-right, so a chunk of S queries at position pos sees keys
	// 0 .. pos + i — no mask tensor, no materialized S x (pos+S) score
	// matrix (the masked sdpa fell back to the fp32 math path, ~12% of
	// prefill time at 16k context). chunk_mask above remains for
	// reference/tests.
	let attn = torch.flash_attn(q, K, V, true);      // (1, nh, S, hd)

	// sigmoid output gate, fused with the (1, nh, S, hd) -> (1, S, nh*hd)
	// relayout
	return L.o_proj.forward(torch.attn_gate(attn, g));
}

/* Full-attention layer with DEVICE-side positions (CUDA-graph capture):
 * st = {pos_idx (S) int64, seqused (1) int32, cu_q, cu_k (int32),
 * cos, sin (S, rotary_dim/2) fp32 rows} — all persistent device
 * tensors whose VALUES change between graph replays. The KV write uses
 * index_copy_ at pos_idx, attention reads keys [0, seqused) of the full
 * cache through flash_attn_kvcache; every shape is fixed. */
export function attn_forward_dev(cfg, L, x, cl, st) {
	let S = x.shape()[1];
	let eps = cfg.rms_norm_eps;
	let nh = cfg.num_q_heads;
	let kvh = cfg.num_kv_heads;
	let hd = cfg.head_dim;
	let max_seq = cl.k_cache.shape()[2];

	let qg = L.q_proj.forward(x).view([1, S, nh, 2 * hd]);
	let q = qg.narrow(-1, 0, hd);
	let g = qg.narrow(-1, hd, hd);

	let kk = L.k_proj.forward(x).view([1, S, kvh, hd]);
	let v = L.v_proj.forward(x).view([1, S, kvh, hd]);

	q = torch.qk_norm_rope(q, L.q_norm_w, st.cos, st.sin, cfg.rotary_dim, eps);
	kk = torch.qk_norm_rope(kk, L.k_norm_w, st.cos, st.sin, cfg.rotary_dim, eps);

	cl.k_cache.index_copy_(2, st.pos_idx, kk.transpose(1, 2));
	cl.v_cache.index_copy_(2, st.pos_idx, v.transpose(1, 2));

	// split-KV decode attention sized by the device-side key count
	// (torch.attn_decode, S <= 5). FA2's varlen entry with seqused_k
	// sizes its grid by the cache CAPACITY and does not split the keys
	// for S > 1: with a 262144-slot cache 6 ms/token of attention at 4k
	// keys, 217 ms per 3-row verify at 180k (docs/LESSONS.md L71). The
	// tensor-core mma variant (2.35x faster at 180k keys, S = 3,
	// E = 256) takes the shapes this branch admits: head_dim
	// 64/128/256 and (nh/kvh) * S <= 32 — Qwen3.8 is rep 6, so S <= 5
	// gives rows <= 30; anything else keeps the shuffle kernel.
	let o;

	if (S <= 5)
		o = torch.attn_decode(q.contiguous().view([S, nh, hd]),
			cl.k_cache.view([kvh, max_seq, hd]), cl.v_cache.view([kvh, max_seq, hd]),
			st.seqused, 1.0 / math.sqrt(hd), null,
			(hd == 64 || hd == 128 || hd == 256) && (nh / kvh) * S <= 32 ? 1 : 0);
				// (S, nh, hd)
	else
		o = torch.flash_attn_kvcache(q.view([S, nh, hd]),
			cl.k_cache.view([kvh, max_seq, hd]).transpose(0, 1),
			cl.v_cache.view([kvh, max_seq, hd]).transpose(0, 1),
			st.cu_q, st.cu_k, st.seqused, max_seq, true);

	return L.o_proj.forward(torch.attn_gate(o.view([1, S, nh, hd]).transpose(1, 2), g));
}

/* SwiGLU MLP. */
/* gate and up as ONE cooperative launch for decode / verify rows (S <= 8):
 * two ~50 us gemms at four rows become one. The bundle needs both
 * products to share K and the codebook (mixed-bitrate checkpoints assign
 * K per tensor), and writes into persistent buffers, so it is built once
 * per row count and kept on the layer (the CUDA graphs bake the buffer
 * addresses). Returns null when the pair cannot be fused. */
const MLP_BUNDLE = getenv('AMP_NO_MLP_BUNDLE') == null;

function mlp_bundle(L, S) {
	let g = L.gate_proj, u = L.up_proj;

	if (!MLP_BUNDLE || g.trellis == null || u.trellis == null || g.bias || u.bias ||
	    g.K != u.K || g.mul1 != u.mul1 || g.mcg != u.mcg ||
	    g.in_features != u.in_features || g.out_features != u.out_features)
		return null;

	if (L.gu_bundle == null)
		L.gu_bundle = {};

	let key = '' + S;

	if (L.gu_bundle[key] == null) {
		let dev = g.dev;
		let out = torch.zeros([2, S, g.out_features], 'float16', 'cuda:' + dev);

		L.gu_bundle[key] = models.multi([ g, u ], out, dev);
	}

	return L.gu_bundle[key];
}

export function mlp_forward(L, x) {
	let S = x.shape()[1];
	let gu = (S <= 8 && x.shape()[0] == 1) ? mlp_bundle(L, S) : null;

	if (gu) {
		gu.run(x.contiguous());

		return L.down_proj.forward(torch.silu_mul(gu.out.narrow(0, 0, 1),
			gu.out.narrow(0, 1, 1)));
	}

	let gate = L.gate_proj.forward(x);
	let up = L.up_proj.forward(x);

	// fused silu(gate) * up (one kernel instead of two)
	return L.down_proj.forward(torch.silu_mul(gate, up));
}

/* One transformer layer (either type). */
export function layer_forward(cfg, L, cl, h, cos, sin, pos, st) {
	let eps = cfg.rms_norm_eps;

	let r = rmsnorm_bias1(h, L.input_layernorm, eps);
	let attn_out = (L.type == 'linear_attention') ?
		gdn_forward(cfg, L, r, cl) :
		(st ? attn_forward_dev(cfg, L, r, cl, st) :
			attn_forward(cfg, L, r, cos, sin, cl, pos));

	// fused residual add + RMSNorm: h += attn_out, r = norm(h) — the
	// weights are pre-biased (+1), so the plain fused kernel applies
	let hr = torch.rms_norm_res(h, attn_out.contiguous(),
		L.post_attention_layernorm, eps);

	h = hr[0];
	h = h.add(mlp_forward(L, hr[1]));

	return h;
}

/* ------------------------------------------------------------------ */
/* Pipelined prefill over a layer split                                */
/*                                                                     */
/* With layers split over several GPUs, a plain chunk-by-chunk forward */
/* keeps one GPU busy at a time. Worse, ucode issues every kernel from */
/* ONE host thread, and a device's launch queue holds far fewer        */
/* launches than one stage of one chunk needs (~110 per layer): the    */
/* host blocks on the busy device's full queue and cannot feed the     */
/* idle one. The fix is a wavefront schedule: chunks move through the  */
/* device stages, and every ROUND enqueues ONE layer per busy stage,   */
/* alternating between devices, so all queues stay fed.                */
/*                                                                     */
/* Correctness: a chunk enters a stage only after the previous chunk   */
/* has left it, so every layer still sees the chunks in order (the GDN */
/* recurrent state and the KV cache are written in sequence).          */

/* Device stages: maximal runs of consecutive layers on one device. */
function stages_of(model) {
	let out = [];
	let n = model.cfg.num_hidden_layers;

	for (let i = 0; i < n; i++) {
		let d = model.layers[i].dev;

		if (!length(out) || out[length(out) - 1].dev != d)
			push(out, {dev: d, first: i, end: i + 1});
		else
			out[length(out) - 1].end = i + 1;
	}

	return out;
}

/* Last-position logits from a final-layer hidden state (1, S, hidden):
 * final norm + lm_head on ONE row. */
export function last_logits(model, h) {
	let S = h.shape()[1];
	let last = model.cfg.num_hidden_layers - 1;

	if (model.layers[last].dev != model.devs[last])
		h = h.cuda(model.devs[last]);

	if (S > 1)
		h = h.narrow(1, S - 1, 1).contiguous();

	h = rmsnorm_bias1(h, model.norm, model.cfg.rms_norm_eps);

	return model.lm_head.forward(h);
}

/* Prefill `ids` into `cache` from position cache.length, pipelined.
 *
 * opts:
 *   chunk      chunk size (default 512)
 *   cuts       extra chunk boundaries (absolute positions), e.g. a
 *              snapshot point
 *   on_layer   function (chunk_end, layer_index) called right after a
 *              chunk's layer is ENQUEUED (stream order: anything it
 *              enqueues on that layer's device runs before the next
 *              chunk touches the layer — the snapshot hook)
 *   logits     compute last-position logits of the final chunk
 *              (default true)
 *   hidden     also keep the post-final-norm hidden state of every row
 *              (runner.hidden(); the MTP draft input)
 *
 * Returns a runner: step(rounds) advances up to `rounds` rounds and
 * returns true while work remains; done_tokens() counts tokens through
 * the whole model; logits() the result once finished. cache.length is
 * advanced when everything is enqueued. */
export function prefill_pipeline(model, cache, ids, opts) {
	opts = opts || {};

	let cfg = model.cfg;
	let chunk = opts.chunk || 512;
	let start = cache.length;
	let total = length(ids);
	let stages = stages_of(model);
	let ns = length(stages);

	// chunk boundaries (absolute positions)
	let bounds = [];

	for (let p = start + chunk; p < start + total; p += chunk)
		push(bounds, p);

	for (let i = 0; i < length(opts.cuts || []); i++) {
		let c = opts.cuts[i];

		if (c > start && c < start + total && index(bounds, c) < 0)
			push(bounds, c);
	}

	bounds = sort(bounds, (a, b) => a - b);
	push(bounds, start + total);

	let next_chunk = 0;
	let prev_end = start;
	let active = [];         // per stage: the chunk in it, or null
	let finished = 0;        // tokens through the whole model
	let result = null;
	let hidden_parts = [];   // opts.hidden: per-chunk post-norm hidden

	for (let s = 0; s < ns; s++)
		push(active, null);

	function admit() {
		if (active[0] != null || next_chunk >= length(bounds))
			return;

		let end = bounds[next_chunk++];
		let pos = prev_end;
		let S = end - pos;
		prev_end = end;
		active[0] = {pos: pos, S: S, end: end, layer: stages[0].first,
			h: embed_rows(model, cache.mm, slice(ids, pos - start, end - start),
				pos)};
	}

	function advance(s) {
		let c = active[s];
		let st = stages[s];

		if (!c || c.layer >= st.end)
			return;

		let i = c.layer;
		let L = model.layers[i];
		let cl = cache.layers[i];
		let cos = null, sin = null;

		if (L.type == 'full_attention') {
			let rr = rope_rows(cfg, cache.mm, cl, c.pos, c.S);

			cos = rr.cos;
			sin = rr.sin;
		}

		c.h = layer_forward(cfg, L, cl, c.h, cos, sin, c.pos);
		c.layer++;

		if (opts.on_layer)
			opts.on_layer(c.end, i);
	}

	/* move chunks that completed their stage on; last stage: finish */
	function hand_off(s) {
		let c = active[s];

		if (!c || c.layer < stages[s].end)
			return;

		if (s == ns - 1) {
			if (opts.hidden) {
				// post-final-norm hidden of every row (the MTP draft
				// input); the last row also yields the logits
				let hn = c.h;
				let last = model.cfg.num_hidden_layers - 1;

				if (model.layers[last].dev != model.devs[last])
					hn = hn.cuda(model.devs[last]);

				hn = rmsnorm_bias1(hn, model.norm, model.cfg.rms_norm_eps);
				if (opts.on_hidden)
					opts.on_hidden(c.end - c.S, hn);       // streamed: nothing is kept
				else
					push(hidden_parts, hn);

				if (c.end == start + total && opts.logits != false)
					result = model.lm_head.forward(
						hn.narrow(1, c.S - 1, 1).contiguous());
			}
			else if (c.end == start + total && opts.logits != false)
				result = last_logits(model, c.h);

			finished = c.end - start;
			active[s] = null;
			return;
		}

		if (active[s + 1] != null)
			return;          // next stage still busy: wait

		c.h = c.h.cuda(stages[s + 1].dev);
		active[s + 1] = c;
		active[s] = null;
	}

	function busy() {
		for (let s = 0; s < ns; s++)
			if (active[s])
				return true;

		return next_chunk < length(bounds);
	}

	return {
		step: function (rounds) {
			rounds = rounds || 1;

			for (let r = 0; r < rounds && busy(); r++) {
				// downstream first, so a stage frees up before its
				// predecessor tries to hand over
				for (let s = ns - 1; s >= 0; s--) {
					hand_off(s);
					advance(s);
				}

				admit();
			}

			if (!busy())
				cache.length = start + total;

			return busy();
		},

		done_tokens: function () {
			return finished;
		},

		logits: function () {
			return result;
		},

		/* opts.hidden: post-final-norm hidden states of all prefilled
		 * rows, (1, total, hidden) on the last device */
		hidden: function () {
			if (!length(hidden_parts))
				return null;

			return (length(hidden_parts) == 1) ? hidden_parts[0] :
				torch.cat(hidden_parts, 1);
		}
	};
}

/* ------------------------------------------------------------------ */
/* MTP draft head (EAGLE / DeepSeek-style)                             */
/*                                                                     */
/* The checkpoint carries one MTP block (config mtp_num_hidden_layers): */
/* pre-fc norms + fc (2H -> H), one full-attention transformer block,   */
/* final norm; the lm_head is borrowed from the target model. The draft */
/* input at position i is (embed(t_i), target_hidden[i-1]) — the target */
/* POST-FINAL-NORM hidden state one position back — concatenated and    */
/* projected; the output predicts t_(i+1). Chained draft steps feed the */
/* draft's own output hidden in place of the target hidden (EAGLE).     */

/* Build the MTP head. Lives on `dev` — the model's LAST device, next to
 * norm/lm_head (it consumes the post-norm hidden and shares the head). */
export function build_mtp(ws, cfg, dev) {
	dev = dev_of_dev(dev);
	let M = {dev: dev, type: 'full_attention'};
	let eps = cfg.rms_norm_eps;

	M.pre_fc_norm_hidden =
		ws.gpu16('mtp.pre_fc_norm_hidden.weight', dev).add(1.0);
	M.pre_fc_norm_embedding =
		ws.gpu16('mtp.pre_fc_norm_embedding.weight', dev).add(1.0);
	M.fc = models.linear(ws, 'mtp.fc', true, dev);

	if (!M.fc)
		die('qwen35.build_mtp: no mtp.fc weight');

	// one full-attention transformer block (same shape as a model layer)
	let key = 'mtp.layers.0';

	M.input_layernorm = ws.gpu16(key + '.input_layernorm.weight', dev)
		.add(1.0);
	M.post_attention_layernorm =
		ws.gpu16(key + '.post_attention_layernorm.weight', dev)
			.add(1.0);

	M.q_proj = models.linear(ws, key + '.self_attn.q_proj', true, dev);
	M.k_proj = models.linear(ws, key + '.self_attn.k_proj', true, dev);
	M.v_proj = models.linear(ws, key + '.self_attn.v_proj', true, dev);
	M.o_proj = models.linear(ws, key + '.self_attn.o_proj', true, dev);

	M.q_norm_w = ws.gpu16(key + '.self_attn.q_norm.weight', dev).add(1.0);
	M.k_norm_w = ws.gpu16(key + '.self_attn.k_norm.weight', dev).add(1.0);

	M.gate_proj = models.linear(ws, key + '.mlp.gate_proj', true, dev);
	M.up_proj = models.linear(ws, key + '.mlp.up_proj', true, dev);
	M.down_proj = models.linear(ws, key + '.mlp.down_proj', true, dev);

	M.norm = ws.gpu16('mtp.norm.weight', dev).add(1.0);

	return M;
}

/* Run the MTP head over a chunk of S tokens paired with target hidden
 * states. token_ids: array (S); hidden: (1, S, hidden) fp16 tensor on
 * the MTP device — the target post-final-norm state ONE POSITION BACK
 * for each token (hidden[0] may be zeros for the very first token);
 * cl: draft cache ({k_cache, v_cache, cos, sin} as built by
 * new_mtp_cache); pos: draft position of the first token; head: the
 * model's lm_head, or null to skip the logits. Returns {logits: (1, S, V)
 * or null, hidden: (1, S, hidden)} — hidden[i] pairs with token i+1 for
 * the next chained draft step. */
/* Core draft step over DEVICE token ids: ids is an int64 (1, S) tensor
 * on embed_dev. The device-side draft chain (engine.uc) feeds the
 * previous step's argmax result straight back without a host
 * round-trip; the array variant below uploads the ids. */
export function mtp_chunk_ids(cfg, M, embed_tokens, embed_dev, ids,
	hidden, cl, pos, head) {
	let eps = cfg.rms_norm_eps;
	let S = ids.shape()[1];

	let emb = embed_tokens.index(ids).cuda(M.dev);

	if (cl.mm)
		mm_overlay(emb, cl.mm, pos, S, M.dev);

	let e = rmsnorm_bias1(emb, M.pre_fc_norm_embedding, eps);
	let y = rmsnorm_bias1(hidden, M.pre_fc_norm_hidden, eps);

	let x = M.fc.forward(torch.cat([e, y], -1));

	// transformer block (identical structure to a model layer)
	let rr = rope_rows(cfg, cl.mm, cl, pos, S);

	let r = rmsnorm_bias1(x, M.input_layernorm, eps);
	x = x.add(attn_forward(cfg, M, r, rr.cos, rr.sin, cl, pos));

	r = rmsnorm_bias1(x, M.post_attention_layernorm, eps);
	x = x.add(mlp_forward(M, r));

	// draft final norm + borrowed head (head == null: skip it — seeding
	// the draft cache over a prompt needs no logits)
	let D = rmsnorm_bias1(x, M.norm, eps);

	return {logits: head ? head.forward(D) : null, hidden: D};
}

/* Array-ids variant: token_ids is a host array of S token ids (one
 * blocking H2D upload per call — the synchronous paths only). */
export function mtp_chunk(cfg, M, embed_tokens, embed_dev, token_ids,
	hidden, cl, pos, head) {
	let S = length(token_ids);
	let ids = torch.from_array(mm_safe_ids(cfg, cl.mm, token_ids, pos),
		'int64').view([1, S]).cuda(embed_dev);

	return mtp_chunk_ids(cfg, M, embed_tokens, embed_dev, ids,
		hidden, cl, pos, head);
}

/* Draft KV cache + rope tables for the MTP head (same layout as a
 * full-attention layer cache). */
export function new_mtp_cache(cfg, max_seq, dev) {
	return {
		k_cache: torch.zeros([1, cfg.num_kv_heads, max_seq, cfg.head_dim],
			'float16', 'cuda:' + dev),
		v_cache: torch.zeros([1, cfg.num_kv_heads, max_seq, cfg.head_dim],
			'float16', 'cuda:' + dev),
		cos: null,
		sin: null,
		dev: dev,
		length: 0
	};
}

/* ------------------------------------------------------------------ */
/* model                                                               */

function dev_list(cfg, devices) {
	if (devices == null)
		devices = 0;

	if (type(devices) == 'array') {
		if (length(devices) != cfg.num_hidden_layers)
			die(`qwen35: devices array must have ${cfg.num_hidden_layers} entries`);

		return devices;
	}

	let out = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++)
		push(out, devices);

	return out;
}

/* Build one transformer layer's weights/norms (no caches). `dev` is
 * the layer's device index. Exported for tests that assemble partial
 * models. */
export function build_layer(ws, cfg, i, dev) {
	let key = `model.language_model.layers.${i}`;
	let L = {type: cfg.layer_types[i], dev: dev_of_dev(dev)};

	L.input_layernorm = ws.gpu16(key + '.input_layernorm.weight', dev)
		.add(1.0);
	L.post_attention_layernorm =
		ws.gpu16(key + '.post_attention_layernorm.weight', dev)
			.add(1.0);

	L.gate_proj = models.linear(ws, key + '.mlp.gate_proj', true, dev);
	L.up_proj = models.linear(ws, key + '.mlp.up_proj', true, dev);
	L.down_proj = models.linear(ws, key + '.mlp.down_proj', true, dev);

	if (L.type == 'linear_attention') {
		L.in_proj_qkv = models.linear(ws, key + '.linear_attn.in_proj_qkv', true, dev);
		L.in_proj_z = models.linear(ws, key + '.linear_attn.in_proj_z', true, dev);
		L.in_proj_b = models.linear(ws, key + '.linear_attn.in_proj_b', true, dev);
		L.in_proj_a = models.linear(ws, key + '.linear_attn.in_proj_a', true, dev);
		L.out_proj = models.linear(ws, key + '.linear_attn.out_proj', true, dev);

		// fp32 gate constants
		L.dt_bias = ws.gpu(key + '.linear_attn.dt_bias', dev)
			.to('float32');
		L.neg_exp_a_log = ws.gpu(key + '.linear_attn.A_log', dev)
			.to('float32').exp().neg();

		// conv weight: (D, 1, k) bf16 -> (D, k) fp16
		L.conv_w = ws.gpu(key + '.linear_attn.conv1d.weight', dev)
			.squeeze(1).to('float16');
		L.norm_w = ws.gpu(key + '.linear_attn.norm.weight', dev)
			.to('float16');
	} else {
		L.q_proj = models.linear(ws, key + '.self_attn.q_proj', true, dev);
		L.k_proj = models.linear(ws, key + '.self_attn.k_proj', true, dev);
		L.v_proj = models.linear(ws, key + '.self_attn.v_proj', true, dev);
		L.o_proj = models.linear(ws, key + '.self_attn.o_proj', true, dev);

		L.q_norm_w = ws.gpu(key + '.self_attn.q_norm.weight', dev)
			.to('float16').add(1.0);
		L.k_norm_w = ws.gpu(key + '.self_attn.k_norm.weight', dev)
			.to('float16').add(1.0);
	}

	return L;
}

/* Build the model. opts:
 *   devices  layer -> device map (number, or array of num_hidden_layers
 *            indices). Default: all on 0.
 *
 * Returns the model object; call model.new_cache(max_seq) before
 * forward_chunk/generate. */
export function load(model_dir, opts) {
	opts = opts || {};
	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let devs = dev_list(cfg, opts.devices);

	let first_dev = devs[0];
	let last_dev = devs[cfg.num_hidden_layers - 1];

	// upload everything, routed per layer device (vision/MTP tensors
	// of the checkpoint stay on the host — untouched)
	let lprefix = 'model.language_model.layers.';

	ws.upload(function (key) {
		if (index(key, lprefix) == 0) {
			let rest = substr(key, length(lprefix));
			let dot = index(rest, '.');
			let idx = +substr(rest, 0, dot);

			return (idx >= 0 && idx < cfg.num_hidden_layers) ?
				devs[idx] : null;
		}

		if (index(key, 'model.language_model.embed_tokens') == 0)
			return first_dev;

		if (index(key, 'model.language_model.norm') == 0 ||
		    index(key, 'lm_head') == 0)
			return last_dev;

		return null;   // not a language-model tensor: skip
	});

	// embed on the first layer's device
	let embed_tokens = ws.gpu16('model.language_model.embed_tokens.weight',
		first_dev);

	// optional MTP draft head: route its tensors to the last device
	let mtp = null;

	if (opts.mtp != false && ws.has('mtp.fc.trellis')) {
		let mprefix = 'mtp.';

		ws.upload(function (key) {
			if (index(key, mprefix) == 0)
				return last_dev;

			return null;   // everything else: already routed / skip
		});

		mtp = build_mtp(ws, cfg, last_dev);
	}

	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++)
		push(layers, build_layer(ws, cfg, i, devs[i]));

	// final norm + head on the last device
	let final_norm = ws.gpu16('model.language_model.norm.weight', last_dev)
		.add(1.0);

	let head;
	if (ws.has('lm_head.trellis'))
		head = models.LinearEXL3Seq(ws, 'lm_head', last_dev);
	else if (ws.has('lm_head.weight'))
		head = models.LinearFP16(ws, 'lm_head', last_dev);
	else
		die('qwen35.load: no lm_head weight and embeddings are not tied');

	let model = {
		cfg: cfg,
		layers: layers,
		devs: devs,
		embed_tokens: embed_tokens,
		norm: final_norm,
		lm_head: head,
		mtp: mtp,

		/* Allocate per-layer caches for max_seq positions: GDN conv +
		 * recurrent state, full-attention KV. */
		new_cache: function (max_seq) {
			let cfg = this.cfg;
			let cache = {max_seq: max_seq, length: 0, layers: []};

			// rope tables per device that hosts a full-attention layer
			let rope_by_dev = {};

			for (let i = 0; i < cfg.num_hidden_layers; i++) {
				let L = this.layers[i];
				let cl = {type: L.type, dev: L.dev};

				if (L.type == 'linear_attention') {
					cl.conv_state = torch.zeros(
						[1, cfg.fdim_qkv, cfg.conv_kernel - 1],
						'float16', 'cuda:' + L.dev);
					cl.rec_state = torch.zeros(
						[cfg.ln_num_v_heads, cfg.ln_k_head_dim,
						 cfg.ln_v_head_dim],
						'float32', 'cuda:' + L.dev);
				} else {
					cl.k_cache = torch.zeros(
						[1, cfg.num_kv_heads, max_seq, cfg.head_dim],
						'float16', 'cuda:' + L.dev);
					cl.v_cache = torch.zeros(
						[1, cfg.num_kv_heads, max_seq, cfg.head_dim],
						'float16', 'cuda:' + L.dev);

					if (!rope_by_dev[L.dev]) {
						let rc = rope_cache(cfg, max_seq, L.dev);

						rope_by_dev[L.dev] = rc;
					}

					let rc = rope_by_dev[L.dev];

					cl.cos = rc.cos;
					cl.sin = rc.sin;
				}

				push(cache.layers, cl);
			}

			// draft cache for the MTP head (last device)
			if (this.mtp) {
				let dc = new_mtp_cache(cfg, max_seq, last_dev);
				let rc = rope_cache(cfg, max_seq, last_dev);

				dc.cos = rc.cos;
				dc.sin = rc.sin;
				cache.mtp = dc;

				// GDN recurrent-state snapshots for speculative rollback:
				// the verify forward advances every GDN layer's state
				// through ALL window tokens; rejected candidates pollute
				// it (KV caches roll back with a length counter, the
				// delta-rule state cannot). One buffer per GDN layer,
				// allocated once, swapped in/out with copy_.
				if (this.mtp) {
					cache.gdn_snap = {};

					for (let i = 0; i < cfg.num_hidden_layers; i++) {
						let cl = cache.layers[i];

						if (cl.type != 'linear_attention')
							continue;

						cache.gdn_snap[i] = [
							torch.zeros(cl.rec_state.shape(),
								'float32', 'cuda:' + cl.dev),
							torch.zeros(cl.conv_state.shape(),
								'float16', 'cuda:' + cl.dev)
						];
					}
				}
			}

			return cache;
		},

		/* Run a chunk of token ids (array) appending to the cache.
		 * Returns logits for the whole chunk: (1, S, vocab) fp16 on
		 * the head device. */
		forward_chunk: function (token_ids, cache) {
			let cfg = this.cfg;
			let S = length(token_ids);
			let pos = cache.length;

			// non-blocking upload: a blocking H2D copy would make the
			// host wait for the first device's queue to drain
			// (image rows come from the vision tower, see embed_rows)
			let h = embed_rows(this, cache.mm, token_ids, pos);   // (1, S, hidden)

			for (let i = 0; i < cfg.num_hidden_layers; i++) {
				let L = this.layers[i];
				let cl = cache.layers[i];

				// cross-device hand-off
				if (i > 0 && L.dev != this.layers[i - 1].dev)
					h = h.cuda(L.dev);

				let cos = null, sin = null;

				if (L.type == 'full_attention') {
					let rr = rope_rows(cfg, cache.mm, cl, pos, S);

					cos = rr.cos;
					sin = rr.sin;
				}

				h = layer_forward(cfg, L, cl, h, cos, sin, pos);
			}

			cache.length = pos + S;

			// final norm + head (last device)
			let last = this.cfg.num_hidden_layers - 1;

			if (this.layers[last].dev != this.devs[last])
				h = h.cuda(this.devs[last]);

			h = rmsnorm_bias1(h, this.norm, this.cfg.rms_norm_eps);

			// stash the post-final-norm hidden state (MTP draft input)
			cache.hidden = h;

			return this.lm_head.forward(h);
		},

		/* Stateless prefill: fresh cache, discarded afterwards. */
		forward: function (token_ids) {
			let cache = this.new_cache(length(token_ids));
			let logits = this.forward_chunk(token_ids, cache);

			return logits;
		},

		/* Generation: prompt ids -> prompt ++ generated ids.
		 * opts: temperature (0 = greedy), top_k, seed. */
		generate: function (prompt, max_new, opts) {
			opts = opts || {};

			if (opts.seed != null)
				math.srand(opts.seed);

			let cache = this.new_cache(length(prompt) + max_new);
			let out = [];

			for (let i = 0; i < length(prompt); i++)
				push(out, prompt[i]);

			let logits = this.forward_chunk(prompt, cache);
			let last = logits.narrow(1, length(prompt) - 1, 1);

			for (let n = 0; n < max_new; n++) {
				let tok = sampler.sample(last, opts);

				push(out, tok);

				if (tok == this.cfg.eos_token_id)
					break;

				if (n + 1 >= max_new)
					break;

				last = this.forward_chunk([tok], cache);
			}

			return out;
		},

		/* Speculative generation with the MTP draft head (greedy only —
		 * temperature sampling needs rejection-sampling rules, future
		 * work). opts: window (draft chain length W, default 4), seed.
		 *
		 * Round structure (EAGLE-1):
		 *   entering: target processed positions 0..p; t_next = accepted
		 *     token for p+1 (unprocessed); H = target post-norm hidden at
		 *     p; draft cache filled through position p.
		 *   chain: W draft steps. Step j feeds (c_(j-1), D_(j-1)) at draft
		 *     position p+j, where c_0 = t_next, D_0 = H; output c_j is the
		 *     candidate for target position p+j+1.
		 *   verify: ONE target forward of [t_next, c_1 .. c_(W-1)] (W
		 *     tokens at positions p+1..p+W). Row j predicts the token at
		 *     p+j+2, i.e. checks candidate c_(j+1).
		 *   accept: longest prefix a of matching rows; tokens c_1..c_a are
		 *     confirmed, plus the bonus b = argmax(row a). Roll both
		 *     caches back to p+a+2 entries (rejected verify/draft writes
		 *     are overwritten by the next round).
		 *
		 * Returns prompt ++ generated ids. */
		generate_mtp: function (prompt, max_new, opts) {
			opts = opts || {};

			if (!this.mtp)
				return this.generate(prompt, max_new, opts);

			if (opts.seed != null)
				math.srand(opts.seed);

			let W = (opts.window != null) ? opts.window : 4;
			let cfg = this.cfg;
			let eps = cfg.rms_norm_eps;
			let M = this.mtp;
			let mdev = M.dev;
			let edev = this.devs[0];

			let cache = this.new_cache(length(prompt) + max_new + W + 2);
			let dc = cache.mtp;
			let out = [];

			for (let i = 0; i < length(prompt); i++)
				push(out, prompt[i]);

			// prefill: logits + post-norm hidden for every prompt position
			let logits = this.forward_chunk(prompt, cache);
			let H_all = cache.hidden;          // (1, S, hidden)
			let S = length(prompt);

			let t_next = sampler.argmax_last(logits);
			push(out, t_next);

			// seed the draft cache: input i = (t_i, H_(i-1)) at draft
			// position i, H_-1 = zeros (exllamav3's carry initialization)
			let zeros = torch.zeros([1, 1, cfg.hidden_size],
				'float16', 'cuda:' + mdev);
			let seed_h = torch.cat(
				[zeros, H_all.narrow(1, 0, S - 1).cuda(mdev)], 1);

			mtp_chunk(cfg, M, this.embed_tokens, edev, prompt, seed_h,
				dc, 0, this.lm_head);

			dc.length = S;

			// round state: H = hidden at position p = S-1
			let H = H_all.narrow(1, S - 1, 1);
			let p = S - 1;

			while (length(out) < max_new + S) {
				// draft chain: W steps
				let c = [t_next];       // c[0] = t_next (seed token)
				let D = H;              // D[0] = target hidden

				for (let j = 1; j <= W; j++) {
					let r = mtp_chunk(cfg, M, this.embed_tokens, edev,
						[c[j - 1]], D, dc, p + j, this.lm_head);

					push(c, sampler.argmax_last(r.logits));
					D = r.hidden;
				}

				dc.length = p + W + 1;

				// verify: one target forward of W tokens. Snapshot every
				// GDN recurrent + conv state first — the forward advances
				// them through all W tokens, and rejected candidates would
				// pollute them for everything that follows.
				for (let k in cache.gdn_snap) {
					cache.gdn_snap[k][0].copy_(cache.layers[k].rec_state);
					cache.gdn_snap[k][1].copy_(cache.layers[k].conv_state);
				}

				let window = [c[0]];

				for (let j = 1; j < W; j++)
					push(window, c[j]);

				let vlogits = this.forward_chunk(window, cache);
				let vrows = vlogits.view([W, -1]);
				let vpred = vrows.argmax(-1).tolist();
				let vhidden = cache.hidden;    // (1, W, hidden)

				// accept the longest matching prefix. a == W (everything
				// matched): no bonus — c_W itself is the confirmed token for
				// position p+W+1 and becomes the next round's seed; the
				// target has only processed up to p+W. a < W: bonus b = the
				// target's own prediction at row a (position p+a+2).
				let a = 0;
				let done = false;

				while (a < W && vpred[a] == c[a + 1])
					a++;

				if (opts.debug)
					printf('round: p=%d c=%J vpred=%J a=%d\n',
						p, c, vpred, a);

				for (let j = 1; j <= a; j++) {
					push(out, c[j]);

					if (c[j] == cfg.eos_token_id)
						done = true;
				}

				if (a < W) {
					let b = vpred[a];

					push(out, b);
					t_next = b;

					if (b == cfg.eos_token_id)
						done = true;
				} else {
					t_next = c[W];
				}

				// roll both caches back past the rejected writes; the next
				// round's verify re-feeds from there
				let p_new = p + ((a < W) ? a + 1 : W);

				if (a < W - 1) {
					// rejected candidates polluted the GDN recurrent states
					// (the verify advanced them through all W tokens):
					// restore the pre-verify snapshot, rewind the target
					// cache to the pre-verify length and replay the
					// accepted prefix (rows 0..a — same tokens, same
					// positions, so the KV rewrites are idempotent). The
					// replay also yields the correct hidden at the last
					// accepted position.
					for (let k in cache.gdn_snap) {
						cache.layers[k].rec_state.copy_(cache.gdn_snap[k][0]);
						cache.layers[k].conv_state.copy_(cache.gdn_snap[k][1]);
					}

					cache.length = p + 1;   // pre-verify length

					this.forward_chunk(slice(c, 0, a + 1), cache);

					H = cache.hidden.narrow(1, a, 1);
				} else {
					// a >= W-1: every fed token was confirmed — the caches
					// are already consistent (length p+W+1)
					H = vhidden.narrow(1, (a < W) ? a : W - 1, 1);
				}

				p = p_new;
				dc.length = p_new + 1;

				if (length(out) >= max_new + S)
					done = true;

				if (done)
					break;
			}

			// a round may overshoot max_new — truncate
			return (length(out) > max_new + S) ?
				slice(out, 0, max_new + S) : out;
		}
	};

	return model;
}