// ucode/dsv41.uc — DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM), text.
//
// See docs/DSV41.md. Reference: DeepSeek's inference/model.py + engram.py (read, not run: the
// FP4 release checkpoint does not fit). Differences from V4 (dsv4.uc):
//
//   - hidden 5120, 40 layers, 384 routed experts (top 6) + 1 shared, no hash-routed layers
//   - the hyper-connection collapse weights ("pre") are one site late: the attention input is
//     collapsed with the pre of the previous FFN site, the FFN input with the pre of this layer's
//     attention site; the final collapse uses the last FFN's pre (no HyperHead)
//   - attention is MQA (one 512-dim K = V head, 64 query heads, sink, 128-token window) plus
//     compressed keys: layers 2-19 pool pairs of tokens (ratio 2, softmax gate), layers 20-39 keep
//     one key per token (ratio 1). Only the "kv source" layers (2, 8, 14, 20) compute a pool;
//     the layers after them read it. The "index source" layers (2, 8, 14, 20, 24, 28, 32, 36) pick
//     the top 512 pool entries per query with a 32-head lightning indexer; the others reuse the
//     last pick. Layer 20 additionally chooses candidate blocks (2048 blocks of 8) that restrict
//     the picks of layers 24-36.
//   - Engram: hashed 2..4-gram embeddings (fp8 tables of 384M rows, streamed from disk) are added
//     into the residual streams before layers 1 and 14
//   - the dense weights are block-scaled FP8 (32 x 32 blocks, e8m0 scales), only the routed
//     experts are EXL3 (mixed 1..6 bits per matrix)
//
// Composite-op path: correct first, fused kernels where measured.

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';
import {readfile, stat} from 'fs';

const NEGH = -60000.0;
const RING = +(getenv('AMP_DSV41_RING') || 8192);          // raw window rows kept per layer (rollback span)
const SCRATCH_ELEMS = 32 * 1024 * 1024;                    // fp8 -> fp16 dequant slice, elements
const PROF = getenv('AMP_DSV41_PROF') != null;
const NO_ENGRAM = getenv('AMP_DSV41_NO_ENGRAM') != null;

/* ------------------------------------------------------------------ */
/* config                                                              */

export function load_config(model_dir) {
	let root = json(readfile(model_dir + '/config.json'));
	let t = root.text_config || root;
	let nl = t.num_hidden_layers;
	let ratios = t.compress_ratios;

	let cfg = {
		model_dir: model_dir,
		hidden_size: t.hidden_size,
		num_hidden_layers: nl,
		vocab_size: t.vocab_size,
		rms_norm_eps: t.rms_norm_eps,
		eos_token_id: root.eos_token_id,
		bos_token_id: root.bos_token_id,

		num_q_heads: t.num_attention_heads,
		head_dim: t.head_dim,
		rope_dim: t.qk_rope_head_dim,
		q_lora_rank: t.q_lora_rank,
		o_groups: t.o_groups,
		o_lora_rank: t.o_lora_rank,
		window: t.sliding_window,
		rope_theta: t.rope_theta,
		compress_rope_theta: t.compress_rope_theta,
		rope_scaling: t.rope_scaling,

		ratios: slice(ratios, 0, nl),
		kv_sources: t.kv_source_layer_ids,
		index_sources: t.index_source_layer_ids,
		candidate_source: (t.candidate_source_layer_id != null) ? t.candidate_source_layer_id : -1,
		candidate_blocks: t.candidate_topk_blocks,
		candidate_block_size: t.candidate_block_size,
		index_n_heads: t.index_n_heads,
		index_head_dim: t.index_head_dim,
		index_topk: t.index_topk,

		hc_mult: t.hc_mult,
		hc_eps: t.hc_eps,
		hc_sinkhorn_iters: t.hc_sinkhorn_iters,

		n_routed_experts: t.n_routed_experts,
		n_shared_experts: t.n_shared_experts,
		num_experts_per_tok: t.num_experts_per_tok,
		moe_intermediate_size: t.moe_intermediate_size,
		routed_scaling_factor: t.routed_scaling_factor,
		swiglu_limit: t.swiglu_limit,

		engram_layers: t.engram_layer_ids || [],
		engram_ngram: t.engram_max_ngram_size,
		engram_heads: t.engram_n_heads,
		engram_dim: t.engram_head_dim,
		engram_pad_id: t.engram_pad_token_id,

		has_vision: false,                       // the vision tower is not implemented yet
		image_token_id: root.image_token_id,

		dspark_block: t.dspark_block_size || 0,
		dspark_noise_id: t.dspark_noise_token_id,
		dspark_taps: t.dspark_target_layer_ids || [],
		num_mtp_layers: t.num_nextn_predict_layers || 0,

		is_v41: true
	};

	if (cfg.hc_mult != 4)
		die('dsv41: only hc_mult == 4 is supported');

	if (t.scoring_func != 'sqrtsoftplus')
		die(`dsv41: unsupported scoring_func '${t.scoring_func}'`);

	if (t.num_key_value_heads != 1)
		die('dsv41: expected shared-KV MQA (num_key_value_heads == 1)');

	// the pool each layer reads (last kv source at or before it) and the index-key owner
	cfg.src = [];

	for (let i = 0; i < nl; i++) {
		let s = -1;

		for (let k in cfg.kv_sources)
			if (k <= i && k > s)
				s = k;

		push(cfg.src, (cfg.ratios[i] > 0) ? s : -1);
	}

	cfg.sm_scale = math.pow(cfg.head_dim * 1.0, -0.5);

	return cfg;
}

/* ------------------------------------------------------------------ */
/* linears                                                             */

let SCRATCH = {};

function scratch(dev) {
	if (SCRATCH[dev] == null)
		SCRATCH[dev] = torch.empty([SCRATCH_ELEMS], 'float16', 'cuda:' + dev);

	return SCRATCH[dev];
}

/* x @ dequant(w[r0 .. r0 + n))^T for a block-scaled fp8 matrix, in row slices of at most SCRATCH_ELEMS */
function fp8_matmul(w, s, dev, x, r0, n) {
	let K = w.shape()[1];
	let per = int(SCRATCH_ELEMS / K / 32) * 32;
	let parts = [];

	for (let a = 0; a < n; a += per) {
		let m = min(per, n - a);
		let wd = torch.fp8_dequant(w.narrow(0, r0 + a, m), s.narrow(0, int((r0 + a) / 32), int(m / 32)), scratch(dev));

		push(parts, torch.matmul(x, wd.t()));
	}

	return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
}

/* Block-scaled FP8 linear (checkpoint key + '.weight' / '.scale'). x: (M, K) fp16 -> (M, N) fp16. */
export function lin8(ws, key, dev) {
	let w = ws.gpu(key + '.weight', dev), s = ws.gpu(key + '.scale', dev);
	let sh = w.shape();

	return {
		w: w, s: s, dev: dev, out_features: sh[0], in_features: sh[1],

		forward: function (x) {
			if (x.shape()[0] <= 8)
				return torch.fp8_gemv(x.contiguous(), this.w, this.s);

			return fp8_matmul(this.w, this.s, this.dev, x, 0, this.out_features);
		}
	};
}

/* Grouped FP8 linear (the attention's wo_a): G blocks of R rows, block g sees columns [g * K, (g + 1) * K)
 * of x only. x: (M, G * K) fp16 -> (M, G * R). */
export function lin8g(ws, key, G, dev) {
	let w = ws.gpu(key + '.weight', dev), s = ws.gpu(key + '.scale', dev);
	let sh = w.shape();

	return {
		w: w, s: s, dev: dev, groups: G, rows: int(sh[0] / G), in_features: sh[1],

		forward: function (x) {
			let M = x.shape()[0], K = this.in_features, R = this.rows;
			let parts = [];

			for (let g = 0; g < G; g++) {
				let xg = x.narrow(1, g * K, K).contiguous();

				if (M <= 8)
					push(parts, torch.fp8_gemv(xg, this.w.narrow(0, g * R, R), this.s.narrow(0, int(g * R / 32), int(R / 32))));
				else
					push(parts, fp8_matmul(this.w, this.s, this.dev, xg, g * R, R));
			}

			return torch.cat(parts, 1);
		}
	};
}

/* bf16 / fp32 matrix used in fp32: x (M, K) -> (M, N) fp32 */
function mat32(ws, key, dev) {
	return ws.gpu(key + '.weight', dev).to('float32').t().contiguous();
}

/* ------------------------------------------------------------------ */
/* rope                                                                */

function rope_cs(inv_freq, p0, T, step, dev) {
	let h = inv_freq.shape()[0];
	let ang = torch.arange(T, 'float32', 'cuda:' + dev).mul(step * 1.0).add(p0 * 1.0)
		.view([T, 1]).mul(inv_freq.view([1, h]));

	return [ang.cos().view([T, 1, h, 1]), ang.sin().view([T, 1, h, 1])];
}

/* x (T, H, D) fp32: rotate the trailing rd channels pairwise (GPT-J) */
function rope_il(x, cs, rd) {
	let s = x.shape();
	let T = s[0], H = s[1], D = s[2];
	let tail = x.narrow(2, D - rd, rd).reshape([T, H, int(rd / 2), 2]);
	let e = tail.narrow(3, 0, 1), o = tail.narrow(3, 1, 1);
	let c = cs[0], sn = cs[1];
	let r = torch.cat([e.mul(c).sub(o.mul(sn)), e.mul(sn).add(o.mul(c))], 3).reshape([T, H, rd]);

	return torch.cat([x.narrow(2, 0, D - rd), r], 2);
}

/* ------------------------------------------------------------------ */
/* hyper-connections                                                   */

export function build_hc(ws, p, tag, dev) {
	let k = `${p}.hc_${tag}`;
	let fn = ws.gpu(k + '_fn', dev).to('float32').contiguous();

	return {
		fn: fn,
		fn_pre_t: fn.narrow(0, 0, 4).t().contiguous(),
		base: ws.gpu(k + '_base', dev).to('float32').contiguous(),
		base_pre: ws.gpu(k + '_base', dev).to('float32').narrow(0, 0, 4).contiguous(),
		scale: ws.gpu(k + '_scale', dev).to('float32').contiguous(),
		scale_pre: ws.gpu(k + '_scale', dev).to('float32').narrow(0, 0, 1).contiguous()
	};
}

/* streams (T, 4, D) fp32 -> {pre (T, 4), post (T, 4), comb (T, 4, 4)} */
export function hc_mixes(cfg, hc, streams) {
	let T = streams.shape()[0];
	let r = glm5.hc_mix(cfg, hc, streams);
	let flat = streams.reshape([T, 4 * cfg.hidden_size]);
	let rs = flat.mul(flat).mean(-1, true).add(cfg.rms_norm_eps).rsqrt();
	let pre = torch.matmul(flat, hc.fn_pre_t).mul(rs).mul(hc.scale_pre).add(hc.base_pre).sigmoid().add(cfg.hc_eps);

	return {pre: pre, post: r[0], comb: r[1]};
}

/* (T, D) fp16: streams collapsed with pre, saturated to the fp16 range (massive activations) */
export function hc_collapse(streams, pre) {
	let T = streams.shape()[0];

	return streams.mul(pre.view([T, 4, 1])).sum(1).clamp(-65504.0, 65504.0).to('float16');
}

/* ------------------------------------------------------------------ */
/* Engram                                                              */

export function build_engram_meta(ws_dir, cfg) {
	let path = ws_dir + '/engram_meta.safetensors';

	if (!stat(path))
		die(`dsv41: ${path} is missing; generate it once with: python3 tests/oracle/make_engram_meta.py ${ws_dir}`);

	let st = torch.st(path);
	let M = {
		token_map: st.tensor('token_map').to('int64').view([-1, 1]),
		pad: st.tensor('pad').to('int64').tolist()[0],
		keep: st,                           // the tensors below may view its mapping
		mult: [], offs: [], sizes: []
	};
	let mult = st.tensor('multipliers').to('int64'), offs = st.tensor('offsets').to('int64');
	let primes = st.tensor('primes').to('int64');

	for (let l = 0; l < length(cfg.engram_layers); l++) {
		push(M.mult, mult.narrow(0, l, 1).view([-1]).contiguous());
		push(M.offs, offs.narrow(0, l, 1).view([-1]).contiguous());
		push(M.sizes, primes.narrow(0, l, 1).view([-1]).contiguous());
	}

	return M;
}

export function build_engram(ws, cfg, i, dev) {
	let key = `layers.${i}.engram`;

	return {
		layer: i, dev: dev, key: key,
		wkv: lin8(ws, key + '.wkv', dev),
		weight: ws.gpu(key + '.q_weight', dev).to('float32').mul(ws.gpu(key + '.k_weight', dev).to('float32')).contiguous(),
		file: ws.has(key + '.embed.weight') ? ws.file(key + '.embed.weight') : null,
		emb_buf: torch.zeros([1, (cfg.engram_ngram - 1) * cfg.engram_heads * cfg.engram_dim], 'float16', 'cuda:' + dev),
		wname: key + '.embed.weight',
		sname: key + '.embed.scale'
	};
}

/* Hash ids of the T tokens of a chunk for engram layer slot `li`, gathered from the table and
 * dequantized: (T, heads * 3 * dim) fp16 on E.dev. cids: compressed ids of [3 history tokens, chunk]. */
function engram_rows(cfg, meta, E, li, hist_t, T) {
	let H = (cfg.engram_ngram - 1) * cfg.engram_heads;

	if (NO_ENGRAM)                                 // tests without the 200 GB tables
		return torch.zeros([T, H * cfg.engram_dim], 'float16', 'cuda:' + E.dev);

	let n = T * H;
	let uids = torch.zeros([n], 'int64'), inverse = torch.zeros([n], 'int64'), heads = torch.zeros([n], 'int32');
	let U = torch.ngram_hash(hist_t, T, meta.mult[li], meta.offs[li], meta.sizes[li], cfg.engram_heads,
		meta.pad, uids, inverse, heads);
	let u = uids.narrow(0, 0, U).contiguous();
	let dv = 'cuda:' + E.dev;
	let rw = torch.pinned_empty([U, cfg.engram_dim], 'float8_e4m3fn');
	let rsc = torch.pinned_empty([U, cfg.engram_dim / 32], 'float8_e8m0fnu');

	E.file.gather(E.wname, u, rw);
	E.file.gather(E.sname, u, rsc);

	let ew = rw.cuda(E.dev).to('float32').view([U, cfg.engram_dim / 32, 32]);
	let es = rsc.cuda(E.dev).to('float32').view([U, cfg.engram_dim / 32, 1]);
	let rows = ew.mul(es).view([U, cfg.engram_dim]).to('float16');

	return rows.index(inverse.cuda(E.dev)).view([T, H * cfg.engram_dim]);
}

/* streams (T, 4, D) fp32 updated in place */
export function engram_apply(cfg, E, streams, emb) {
	let T = streams.shape()[0], D = cfg.hidden_size;
	let kv = E.wkv.forward(emb);
	let key = kv.narrow(1, 0, 4 * D).to('float32').view([T, 4, D]);
	let value = kv.narrow(1, 4 * D, D).to('float32').view([T, 1, D]);
	let eps = cfg.rms_norm_eps;
	let rstd = streams.mul(streams).mean(-1).add(eps).rsqrt().mul(key.mul(key).mean(-1).add(eps).rsqrt());
	let dot = streams.mul(E.weight).mul(key).sum(-1).mul(rstd).mul(math.pow(D * 1.0, -0.5));
	let a = dot.abs().clamp(1.0e-6, null).sqrt();
	let gate = torch.where(dot.le(0.0), a.neg(), a).sigmoid();

	streams.add_(gate.view([T, 4, 1]).mul(value));
}

/* ------------------------------------------------------------------ */
/* attention                                                           */

export function build_attn(ws, cfg, i, dev) {
	let key = `layers.${i}.attn`;
	let ratio = cfg.ratios[i];
	let A = {
		dev: dev, idx: i, ratio: ratio,
		inv_freq: dsv4.yarn_inv_freq(cfg.rope_dim, ratio ? cfg.compress_rope_theta : cfg.rope_theta,
			ratio ? cfg.rope_scaling : null, dev),
		q_a: lin8(ws, key + '.wq_a', dev),
		q_norm: ws.gpu16(key + '.q_norm.weight', dev),
		q_b: lin8(ws, key + '.wq_b', dev),
		wkv: lin8(ws, key + '.wkv', dev),
		kv_norm: ws.gpu16(key + '.kv_norm.weight', dev),
		wo_a: lin8g(ws, key + '.wo_a', cfg.o_groups, dev),
		wo_b: lin8(ws, key + '.wo_b', dev),
		sinks: ws.gpu(key + '.attn_sink', dev).to('float32').contiguous(),
		comp: null, indexer: null,
		kv_source: index(cfg.kv_sources, i) >= 0,
		idx_source: index(cfg.index_sources, i) >= 0
	};

	if (A.kv_source) {
		A.comp = {
			norm: ws.gpu16(key + '.compressor.norm.weight', dev),
			wkv: null, wgate: null, wkv16: null
		};

		if (ratio > 1) {
			A.comp.wkv = mat32(ws, key + '.compressor.wkv', dev);
			A.comp.wgate = mat32(ws, key + '.compressor.wgate', dev);
		}
		else {
			A.comp.wkv16 = ws.gpu16(key + '.compressor.wkv.weight', dev).t().contiguous();
		}
	}

	if (A.idx_source) {
		A.indexer = {
			wq_b: lin8(ws, key + '.indexer.wq_b', dev),
			w_t: mat32(ws, key + '.indexer.weights_proj', dev),
			owns_k: A.kv_source
		};

		if (A.kv_source) {
			A.indexer.wk_t = ws.gpu16(key + '.indexer.wk.weight', dev).t().contiguous();
			A.indexer.k_norm = ws.gpu16(key + '.indexer.k_norm.weight', dev);
		}
	}

	return A;
}

/* ------------------------------------------------------------------ */
/* cache                                                               */

/* per-layer window ring + (for kv sources) the pool, index keys and the half-built pair */
export function new_layer_cache(cfg, A, max_seq) {
	let dv = 'cuda:' + A.dev;
	let cl = {
		dev: A.dev,
		ring: torch.zeros([RING, cfg.head_dim], 'float16', dv)
	};

	if (A.kv_source) {
		let cap = int(max_seq / A.ratio) + 2;

		cl.pool = torch.zeros([cap, cfg.head_dim], 'float16', dv);
		cl.kidx = torch.zeros([cap, cfg.index_head_dim], 'float16', dv);
		cl.n = 0;                       // pool entries written
		// ratio 2: the kv / gate rows of the first token of a half-built pair (valid while the next
		// position is odd); persistent so that graph steps update them in place
		cl.pk = torch.zeros([1, cfg.head_dim], 'float32', dv);
		cl.pg = torch.zeros([1, cfg.head_dim], 'float32', dv);
	}

	return cl;
}

/* ring rows of absolute positions [a, b) (b - a <= RING) */
function ring_rows(cl, a, b) {
	let n = b - a;

	if (n <= 0)
		return torch.zeros([0, 512], 'float16', 'cuda:' + cl.dev);

	let s = a % RING;

	if (s + n <= RING)
		return cl.ring.narrow(0, s, n);

	return torch.cat([cl.ring.narrow(0, s, RING - s), cl.ring.narrow(0, 0, n - (RING - s))], 0);
}

function ring_put(cl, pos0, rows) {
	let T = rows.shape()[0];

	if (T > RING) {
		rows = rows.narrow(0, T - RING, RING);
		pos0 += T - RING;
		T = RING;
	}

	let s = pos0 % RING;

	if (s + T <= RING) {
		cl.ring.narrow(0, s, T).copy_(rows);
	}
	else {
		cl.ring.narrow(0, s, RING - s).copy_(rows.narrow(0, 0, RING - s));
		cl.ring.narrow(0, 0, T - (RING - s)).copy_(rows.narrow(0, RING - s, T - (RING - s)));
	}
}

/* ------------------------------------------------------------------ */
/* compressor (kv source layers)                                       */

/* Feed the chunk (xn (T, D) fp16 at absolute pos0) to the layer's compressor: appends the completed
 * pool entries (rope'd) and index keys. Returns the number of new entries. */
function compress_chunk(cfg, A, cl, xn, pos0) {
	let C = A.comp, ratio = A.ratio;
	let dv = 'cuda:' + A.dev;
	let rd = cfg.rope_dim, eps = cfg.rms_norm_eps;
	let T = xn.shape()[0];
	let lat;                                          // (n, 512) fp16 pre-rope, normed
	let base;                                         // absolute position of the first entry's first token

	if (ratio == 1) {
		lat = torch.rms_norm(torch.matmul(xn, C.wkv16), C.norm, eps);
		base = pos0;
	}
	else {
		let x32 = xn.to('float32');
		let kv = torch.matmul(x32, C.wkv), g = torch.matmul(x32, C.wgate);

		if (pos0 % 2 == 1) {                          // the first token of the pair came with the last chunk
			kv = torch.cat([cl.pk, kv], 0);
			g = torch.cat([cl.pg, g], 0);
			base = pos0 - 1;
		}
		else {
			base = pos0;
		}

		let total = kv.shape()[0];
		let usable = total - total % 2;

		if (usable < total) {
			cl.pk.copy_(kv.narrow(0, usable, 1));
			cl.pg.copy_(g.narrow(0, usable, 1));
		}

		if (usable == 0)
			return 0;

		let nw = int(usable / 2);
		let kvw = kv.narrow(0, 0, usable).reshape([nw, 2, cfg.head_dim]);
		let gw = g.narrow(0, 0, usable).reshape([nw, 2, cfg.head_dim]);

		lat = torch.rms_norm(kvw.mul(gw.softmax(1)).sum(1).to('float16').contiguous(), C.norm, eps);
	}

	let n = lat.shape()[0];
	let e0 = int(base / ratio);                       // index of the first new entry

	if (e0 != cl.n)
		die(`dsv41: compressor of layer ${A.idx} out of step (${e0} != ${cl.n})`);

	// index key: k_norm(wk(latent)), rope at the entry's position
	let I = A.indexer;
	let cs = rope_cs(A.inv_freq, base, n, ratio, A.dev);

	if (I && I.owns_k) {
		let k = torch.rms_norm(torch.matmul(lat, I.wk_t), I.k_norm, eps);

		cl.kidx.narrow(0, e0, n).copy_(rope_il(k.to('float32').view([n, 1, cfg.index_head_dim]), cs, rd).to('float16').view([n, cfg.index_head_dim]));
	}

	cl.pool.narrow(0, e0, n).copy_(rope_il(lat.to('float32').view([n, 1, cfg.head_dim]), cs, rd).to('float16').view([n, cfg.head_dim]));
	cl.n = e0 + n;

	return n;
}

/* the pool / index keys of source layer `src` as seen from device dev (a mirror on other devices,
 * brought up to date on every call) */
function pool_for(cache, src, dev) {
	let P = cache.pools[src];
	let cl = P.owner;

	if (cl.dev == dev)
		return cl;

	let m = null;

	for (let c in P.mirrors)
		if (c.dev == dev)
			m = c;

	if (!m) {
		let cap = cl.pool.shape()[0];
		let dv = 'cuda:' + dev;

		m = {
			dev: dev, n: 0,
			pool: torch.zeros([cap, cl.pool.shape()[1]], 'float16', dv),
			kidx: torch.zeros([cap, cl.kidx.shape()[1]], 'float16', dv)
		};

		push(P.mirrors, m);
	}

	if (m.n > cl.n)
		m.n = cl.n;

	if (m.n < cl.n) {
		let k = cl.n - m.n;

		m.pool.narrow(0, m.n, k).copy_(cl.pool.narrow(0, m.n, k));
		m.kidx.narrow(0, m.n, k).copy_(cl.kidx.narrow(0, m.n, k));
		m.n = cl.n;
	}

	return m;
}

/* ------------------------------------------------------------------ */
/* indexer                                                             */

/* Candidate source layer: per query keep the candidate_blocks highest-scoring blocks of
 * candidate_block_size pool rows (a block scores by its best row; the block of the newest visible
 * row is always kept). sc (R, S) fp16 for the rows pos0 .. pos0 + R - 1. Returns (R, k) int64 block
 * ids, -1 where fewer blocks are reachable. */
function candidate_blocks(cfg, sc, pos0, ratio, dv) {
	let bs = cfg.candidate_block_size;
	let R = sc.shape()[0], S = sc.shape()[1];
	let nb = int((S + bs - 1) / bs);
	let pad = nb * bs - S;
	let s2 = (pad > 0) ? torch.cat([sc, torch.full([R, pad], NEGH, 'float16', dv)], 1) : sc;
	let bsc = s2.view([R, nb, bs]).max(2).to('float32');

	// the block holding the newest visible row of each query (ratio 1: visible rows = position + 1)
	let lbl = [];

	for (let r = 0; r < R; r++)
		push(lbl, int((pos0 + r) / bs));

	let lb = torch.from_array(lbl, 'int64', dv).view([R, 1]);
	let ar = torch.arange(nb, 'int64', dv).view([1, nb]);
	let hit = ar.le(lb).to('int32').mul(lb.le(ar).to('int32'));

	bsc = torch.where(hit.le(0), bsc, torch.full([1], 1.0e30, 'float32', dv));

	let top = bsc.topk(min(cfg.candidate_blocks, nb), -1);

	return torch.where(top.values.le(NEGH), torch.full([1], -1, 'int64', dv), top.indices);
}

/* sc (R, S) fp16: rows outside the candidate blocks (R, k) int64 (-1 = none) become masked */
function apply_candidates(cfg, sc, blocks, dv) {
	let bs = cfg.candidate_block_size;
	let R = sc.shape()[0], S = sc.shape()[1];
	let nb = int((S + bs - 1) / bs), k = blocks.shape()[1];
	let ro = [];

	for (let r = 0; r < R; r++)
		push(ro, r * nb);

	let rowoff = torch.from_array(ro, 'int64', dv).view([R, 1]);
	let dummy = torch.full([1], R * nb, 'int64', dv);
	let flat = torch.where(blocks.le(-1), dummy, blocks.add(rowoff)).reshape([-1]);
	let m = torch.zeros([R * nb + 1], 'float16', dv);

	m.index_copy_(0, flat, torch.ones([R * k], 'float16', dv));

	let mm = m.narrow(0, 0, R * nb).view([R, nb, 1]).expand([R, nb, bs]).reshape([R, nb * bs]).narrow(1, 0, S);

	return torch.where(mm.le(0.0), torch.full([1], NEGH, 'float16', dv), sc);
}

/* top-k picks of the layer's indexer: (T, k) int32 (negative = none) or null when the pool is empty.
 * xn (T, D) fp16, qr (T, 1280) fp16 (normed q latent). ctx.cand holds the candidate blocks of layer 20. */
function index_picks(cfg, A, ctx, cache, xn, qr, pos0, src) {
	let I = A.indexer, ratio = A.ratio;
	let T = xn.shape()[0];
	let dv = 'cuda:' + A.dev;
	let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, rd = cfg.rope_dim;
	let S = int((pos0 + T) / ratio);

	if (S <= 0)
		return null;

	let pc = pool_for(cache, src, A.dev);
	let qi = I.wq_b.forward(qr).view([T, Hi, Di]);
	let cs = rope_cs(A.inv_freq, pos0, T, 1, A.dev);
	let q16 = rope_il(qi.to('float32'), cs, rd).to('float16').contiguous();
	let w = torch.matmul(xn.to('float32'), I.w_t).contiguous();
	let kk = pc.kidx.narrow(0, 0, S);
	let scale = math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5);
	let K = min(cfg.index_topk, S);
	let SLAB = max(1, int(96 * 1024 * 1024 / S));
	let outs = [], cand_new = [];

	for (let r0 = 0; r0 < T; r0 += SLAB) {
		let R = min(SLAB, T - r0);
		let sc = torch.idx_score(q16.narrow(0, r0, R).contiguous(), kk, w.narrow(0, r0, R).contiguous(), scale,
			pos0 + r0, ratio);

		if (S > cfg.candidate_blocks * cfg.candidate_block_size && cfg.candidate_source >= 0) {
			if (A.idx == cfg.candidate_source) {
				push(cand_new, candidate_blocks(cfg, sc, pos0 + r0, ratio, dv));
			}
			else if (A.idx > cfg.candidate_source && ctx.cand) {
				let ck = 'd' + A.dev;
				let cb = ctx.cand[ck];

				if (cb == null)
					cb = ctx.cand[ck] = (ctx.cand.dev == A.dev) ? ctx.cand.src : ctx.cand.src.cuda(A.dev);

				sc = apply_candidates(cfg, sc, cb.narrow(0, r0, R), dv);
			}
		}

		let tk = sc.topk(K, -1);

		push(outs, torch.where(tk.values.le(NEGH), torch.full([1], -1, 'int64', dv), tk.indices).to('int32'));
	}

	if (length(cand_new))
		ctx.cand = {dev: A.dev, src: (length(cand_new) == 1) ? cand_new[0] : torch.cat(cand_new, 0)};

	return (length(outs) == 1) ? outs[0] : torch.cat(outs, 0);
}

/* ------------------------------------------------------------------ */
/* attention forward                                                   */

export function attn_forward(cfg, A, ctx, cache, cl, xn, pos0) {
	let T = xn.shape()[0];
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let eps = cfg.rms_norm_eps;
	let dev = A.dev;
	let ratio = A.ratio;

	let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
	let q = A.q_b.forward(qr).view([T, H, D]);
	let cs = rope_cs(A.inv_freq, pos0, T, 1, dev);
	let q16 = rope_il(q.to('float32'), cs, rd).to('float16').contiguous();
	let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
	let kv16 = rope_il(kv.to('float32').view([T, 1, D]), cs, rd).to('float16').view([T, D]).contiguous();

	// window keys: the previous (window - 1) rows from the ring plus this chunk
	let h = min(pos0, cfg.window - 1);
	let kvw = (h > 0) ? torch.cat([ring_rows(cl, pos0 - h, pos0), kv16], 0) : kv16;

	ring_put(cl, pos0, kv16);

	let pool = null, idx = null;

	if (ratio > 0) {
		let src = cfg.src[A.idx];

		if (A.kv_source)
			compress_chunk(cfg, A, cl, xn, pos0);

		if (A.idx_source) {
			idx = index_picks(cfg, A, ctx, cache, xn, qr, pos0, src);
			ctx.topk = idx ? {dev: dev, t: idx} : null;
		}
		else if (ctx.topk) {
			let ck = 'd' + dev;

			if (ctx.topk[ck] == null)
				ctx.topk[ck] = (ctx.topk.dev == dev) ? ctx.topk.t : ctx.topk.t.cuda(dev);

			idx = ctx.topk[ck];
		}

		if (idx) {
			let pc = pool_for(cache, src, dev);
			let S = int((pos0 + T) / ratio);

			pool = pc.pool.narrow(0, 0, S);
		}
	}

	let o = torch.v41_attn(q16, kvw, pos0 - h, pool, idx, A.sinks, A.inv_freq, pos0, cfg.sm_scale, cfg.window);
	let y = A.wo_a.forward(o.view([T, H * D]));

	return A.wo_b.forward(y);
}

/* ------------------------------------------------------------------ */
/* decode with device-side positions (CUDA graphs)                     */
/*
 * Everything that depends on the position is read from the device (st.meta), the shapes depend only on
 * the capacities of the graph's bucket: the pool of a ratio-1 source is read up to caps[1] entries (a
 * ratio-2 one up to caps[2]); entries past the position are masked by the indexer, the half-built pair of
 * the ratio-2 compressors is handled without branches (the pooled row is written every step, to the slot
 * of the pair: on even positions it is not visible yet, on odd ones it is right).
 */

export function new_dec_dev(cfg, dev) {
	return {dev: dev, meta: torch.zeros([8], 'int64', 'cuda:' + dev)};
}

/* meta: [pos, pos % ring, pos / 2, 2 * (pos / 2), pos / candidate block, 0, 0, 0] */
export function dec_set_pos(cfg, st, pos) {
	st.meta.copy_(torch.from_array([pos, pos % RING, int(pos / 2), 2 * int(pos / 2),
		int(pos / cfg.candidate_block_size), 0, 0, 0], 'int64'));
}

function meta_i(st, k) {
	return st.meta.narrow(0, k, 1);
}

function meta_f(st, k) {
	return st.meta.narrow(0, k, 1).to('float32');
}

/* cos / sin (1, 1, rd/2, 1) of the position pos_f (1) fp32 */
function rope_cs_dev(inv_freq, pos_f) {
	let h = inv_freq.shape()[0];
	let ang = pos_f.view([1, 1]).mul(inv_freq.view([1, h]));

	return [ang.cos().view([1, 1, h, 1]), ang.sin().view([1, 1, h, 1])];
}

/* the pool of source layer src on dev for graph bodies (a mirror is created by the warm-up run) */
function pool_dec(cache, src, dev) {
	let P = cache.pools[src];

	if (P.owner.dev == dev)
		return P.owner;

	for (let m in P.mirrors)
		if (m.dev == dev)
			return m;

	return pool_for(cache, src, dev);
}

/* host side, between graph replays: bring the entry j of the owner's pool and index keys to the mirrors */
export function mirror_entry(cache, src, j) {
	let P = cache.pools[src], cl = P.owner;

	for (let m in P.mirrors) {
		m.pool.narrow(0, j, 1).copy_(cl.pool.narrow(0, j, 1));
		m.kidx.narrow(0, j, 1).copy_(cl.kidx.narrow(0, j, 1));
	}
}

/* one token into the compressor of a kv source layer; writes the pool and index key of its slot */
function comp_decode(cfg, A, cl, xn, st) {
	let C = A.comp, ratio = A.ratio;
	let rd = cfg.rope_dim, eps = cfg.rms_norm_eps;
	let lat, posf, slot;

	if (ratio == 1) {
		lat = torch.rms_norm(torch.matmul(xn, C.wkv16), C.norm, eps);
		posf = meta_f(st, 0);
		slot = meta_i(st, 0);
	}
	else {
		let x32 = xn.to('float32');
		let kv = torch.matmul(x32, C.wkv), g = torch.matmul(x32, C.wgate);
		let m = torch.cat([cl.pg, g], 0).max(0, true);
		let e0 = cl.pg.sub(m).exp(), e1 = g.sub(m).exp();
		let pooled = cl.pk.mul(e0).add(kv.mul(e1)).div(e0.add(e1));

		cl.pk.copy_(kv);
		cl.pg.copy_(g);
		lat = torch.rms_norm(pooled.to('float16').contiguous(), C.norm, eps);
		posf = meta_f(st, 3);
		slot = meta_i(st, 2);
	}

	let cs = rope_cs_dev(A.inv_freq, posf);
	let I = A.indexer;

	if (I && I.owns_k) {
		let k = torch.rms_norm(torch.matmul(lat, I.wk_t), I.k_norm, eps);

		cl.kidx.index_copy_(0, slot, rope_il(k.to('float32').view([1, 1, cfg.index_head_dim]), cs, rd).to('float16')
			.view([1, cfg.index_head_dim]));
	}

	cl.pool.index_copy_(0, slot, rope_il(lat.to('float32').view([1, 1, cfg.head_dim]), cs, rd).to('float16')
		.view([1, cfg.head_dim]));
}

/* A tensor a graph body shares with a later one: the producer's tensor on its own device, and on another
 * device a persistent copy that the host refreshes between replays (ctx.links). h: {dev, t | key}. */
function remote(ctx, h, dev, key) {
	key = key || 't';

	if (h.dev == dev)
		return h[key];

	if (ctx.links == null)
		ctx.links = [];

	let ck = 'd' + dev + key;

	if (h[ck] == null) {
		h[ck] = h[key].cuda(dev);
		push(ctx.links, {src: h[key], dst: h[ck]});
	}

	return h[ck];
}

/* the picks of the indexer for the one token: (1, k) int32 over the first Scap pool entries */
function index_picks_dec(cfg, A, ctx, cache, xn, qr, st, Scap, src) {
	let I = A.indexer, ratio = A.ratio;
	let dv = 'cuda:' + A.dev;
	let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, rd = cfg.rope_dim;
	let pc = pool_dec(cache, src, A.dev);
	let qi = I.wq_b.forward(qr).view([1, Hi, Di]);
	let cs = rope_cs_dev(A.inv_freq, meta_f(st, 0));
	let q16 = rope_il(qi.to('float32'), cs, rd).to('float16').contiguous();
	let w = torch.matmul(xn.to('float32'), I.w_t).contiguous();
	let sc = torch.idx_score(q16, pc.kidx.narrow(0, 0, Scap), w, math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5),
		meta_i(st, 0), ratio);
	let bs = cfg.candidate_block_size;

	if (Scap > cfg.candidate_blocks * bs && cfg.candidate_source >= 0) {
		let nb = int((Scap + bs - 1) / bs);

		if (A.idx == cfg.candidate_source) {
			let pad = nb * bs - Scap;
			let s2 = (pad > 0) ? torch.cat([sc, torch.full([1, pad], NEGH, 'float16', dv)], 1) : sc;
			let bsc = s2.view([1, nb, bs]).max(2).to('float32');
			let hit = torch.arange(nb, 'int64', dv).view([1, nb]).le(meta_i(st, 4).view([1, 1])).to('int32')
				.mul(meta_i(st, 4).view([1, 1]).le(torch.arange(nb, 'int64', dv).view([1, nb])).to('int32'));

			bsc = torch.where(hit.le(0), bsc, torch.full([1], 1.0e30, 'float32', dv));

			let top = bsc.topk(min(cfg.candidate_blocks, nb), -1);

			ctx.cand = {dev: A.dev,
				src: torch.where(top.values.le(NEGH), torch.full([1], -1, 'int64', dv), top.indices)};
		}
		else if (A.idx > cfg.candidate_source && ctx.cand) {
			let cb = remote(ctx, ctx.cand, A.dev, 'src');
			let k = cb.shape()[1];
			let flat = torch.where(cb.le(-1), torch.full([1], nb, 'int64', dv), cb).reshape([-1]);
			let m = torch.zeros([nb + 1], 'float16', dv);

			m.index_copy_(0, flat, torch.ones([k], 'float16', dv));

			let mm = m.narrow(0, 0, nb).view([nb, 1]).expand([nb, bs]).reshape([1, nb * bs]).narrow(1, 0, Scap);

			sc = torch.where(mm.le(0.0), torch.full([1], NEGH, 'float16', dv), sc);
		}
	}

	let tk = sc.topk(min(cfg.index_topk, Scap), -1);

	return torch.where(tk.values.le(NEGH), torch.full([1], -1, 'int64', dv), tk.indices).to('int32');
}

/* attention of one token at the device position; xn (1, D) fp16 */
export function attn_decode(cfg, A, ctx, cache, cl, xn, st, caps) {
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let eps = cfg.rms_norm_eps;
	let dev = A.dev;
	let ratio = A.ratio;

	let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
	let q = A.q_b.forward(qr).view([1, H, D]);
	let cs = rope_cs_dev(A.inv_freq, meta_f(st, 0));
	let q16 = rope_il(q.to('float32'), cs, rd).to('float16').contiguous();
	let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
	let kv16 = rope_il(kv.to('float32').view([1, 1, D]), cs, rd).to('float16').view([1, D]).contiguous();

	cl.ring.index_copy_(0, meta_i(st, 1), kv16);

	let pool = null, idx = null;

	if (ratio > 0) {
		let src = cfg.src[A.idx];
		let Scap = caps[ratio];

		if (A.kv_source)
			comp_decode(cfg, A, cl, xn, st);

		if (A.idx_source) {
			idx = index_picks_dec(cfg, A, ctx, cache, xn, qr, st, Scap, src);
			ctx.topk = {dev: dev, t: idx};
		}
		else if (ctx.topk) {
			idx = remote(ctx, ctx.topk, dev);
		}

		if (idx)
			pool = pool_dec(cache, src, dev).pool.narrow(0, 0, Scap);
	}

	let o = torch.v41_attn(q16, cl.ring, 0, pool, idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
		meta_i(st, 0), RING);

	return A.wo_b.forward(A.wo_a.forward(o.view([1, H * D])));
}

/* ------------------------------------------------------------------ */
/* MoE                                                                 */

function build_mlp8(ws, key, dev) {
	return {
		gate: lin8(ws, key + '.w1', dev),
		up: lin8(ws, key + '.w3', dev),
		down: lin8(ws, key + '.w2', dev)
	};
}

function build_expert(ws, key, dev) {
	return {
		gate: models.linear(ws, key + '.w1', false, dev),
		up: models.linear(ws, key + '.w3', false, dev),
		down: models.linear(ws, key + '.w2', false, dev)
	};
}

/* Fused decode state (torch.moe_coop) for experts of mixed bit widths: pointer tables of the experts'
 * trellis / suh / svh data, a uint8 table of every matrix's width, and the scratch for COOP_ROWS tokens. */
const COOP_ROWS = 8;

export function build_coop(cfg, M) {
	let ex = M.experts;
	let dv = 'cuda:' + M.dev;
	let g0 = ex[0].gate, d0 = ex[0].down;

	for (let e in ex)
		if (!e.gate.mul1 || !e.up.mul1 || !e.down.mul1 || e.gate.K > 6 || e.up.K > 6 || e.down.K > 6 ||
		    e.gate.K != int(e.gate.K) || e.up.K != int(e.up.K) || e.down.K != int(e.down.K))
			return null;

	function table(name, field) {
		let a = [];

		for (let e in ex)
			push(a, e[name][field].data_ptr());

		return torch.from_array(a, 'int64', dv);
	}

	function bits(name) {
		let a = [];

		for (let e in ex)
			push(a, int(e[name].K));

		return torch.from_array(a, 'uint8', dv);
	}

	let Hi = g0.in_features, I = g0.out_features, Ho = d0.out_features;
	let slots = COOP_ROWS * cfg.num_experts_per_tok;
	let ctr_len = slots * int(I / 128) + COOP_ROWS * int(Ho / 128) + 2 + (slots + 1) + slots;

	return {
		g_trellis: table('gate', 'trellis'), g_suh: table('gate', 'suh'), g_svh: table('gate', 'svh'),
		u_trellis: table('up', 'trellis'), u_suh: table('up', 'suh'), u_svh: table('up', 'svh'),
		d_trellis: table('down', 'trellis'), d_suh: table('down', 'suh'), d_svh: table('down', 'svh'),
		g_bits: bits('gate'), u_bits: bits('up'), d_bits: bits('down'),
		had_g: torch.zeros([slots, Hi], 'float16', dv),
		had_u: torch.zeros([slots, Hi], 'float16', dv),
		gu_g: torch.zeros([slots, I], 'float16', dv),
		gu_u: torch.zeros([slots, I], 'float16', dv),
		act_out: torch.zeros([slots, I], 'float16', dv),
		d_out: torch.zeros([slots, Ho], 'float32', dv),
		ctr: torch.zeros([ctr_len], 'int32', dv),
		out: torch.zeros([COOP_ROWS, cfg.hidden_size], 'float32', dv),
		Hi: Hi, Kg: 1, Ku: 1, Kd: 1, mcg: false, mul1: true, act: 0, act_limit: cfg.swiglu_limit + 0.0
	};
}

export function build_moe(ws, cfg, i, dev) {
	let key = `layers.${i}.ffn`;
	let experts = [];

	for (let e = 0; e < cfg.n_routed_experts; e++)
		push(experts, build_expert(ws, `${key}.experts.${e}`, dev));

	let M = {
		type: 'moe', dev: dev, experts: experts, router: 'sqrtsp',
		gate_t: ws.gpu(key + '.gate.weight', dev).to('float32').t().contiguous(),
		bias: ws.gpu(key + '.gate.bias', dev).to('float32').reshape([1, -1]).contiguous(),
		bias_vl: null, tid2eid: null, ids: null, vlon: false, async_moe: true,
		shared: build_mlp8(ws, key + '.shared_experts', dev)
	};

	// glm5.moe_forward then runs the fused decode kernel (<= 8 rows) and the token-sorted fused kernel
	// (prefill chunks) with the per-expert bit widths
	M.coop = (getenv('AMP_DSV41_NO_COOP') ? null : build_coop(cfg, M)) || false;

	return M;
}

export function build_layer(ws, cfg, i, dev) {
	let p = `layers.${i}`;

	return {
		dev: dev, idx: i,
		hc_attn: build_hc(ws, p, 'attn', dev),
		hc_ffn: build_hc(ws, p, 'ffn', dev),
		attn_norm: ws.gpu16(`${p}.attn_norm.weight`, dev),
		ffn_norm: ws.gpu16(`${p}.ffn_norm.weight`, dev),
		attn: build_attn(ws, cfg, i, dev),
		moe: build_moe(ws, cfg, i, dev)
	};
}

/* One block on streams (T, 4, D) fp32 (updated in place); pre (T, 4): the collapse weights of this
 * block's attention input. Returns the pre of the next block's attention input. */
function block_forward(cfg, L, ctx, cache, cl, streams, pre, pos0) {
	let eps = cfg.rms_norm_eps;
	let am = hc_mixes(cfg, L.hc_attn, streams);
	let xn = torch.rms_norm(hc_collapse(streams, pre), L.attn_norm, eps);
	let y = attn_forward(cfg, L.attn, ctx, cache, cl, xn, pos0);

	glm5.hc_apply(streams, y, am.post, am.comb);

	let fm = hc_mixes(cfg, L.hc_ffn, streams);

	xn = torch.rms_norm(hc_collapse(streams, am.pre), L.ffn_norm, eps);
	y = glm5.moe_forward(cfg, L.moe, xn);
	glm5.hc_apply(streams, y, fm.post, fm.comb);

	return fm.pre;
}

/* One block on the one token (device position); streams (1, 4, D) fp32 updated in place. */
export function block_decode(cfg, L, ctx, cache, cl, streams, pre, st, caps) {
	let eps = cfg.rms_norm_eps;
	let am = hc_mixes(cfg, L.hc_attn, streams);
	let xn = torch.rms_norm(hc_collapse(streams, pre), L.attn_norm, eps);
	let y = attn_decode(cfg, L.attn, ctx, cache, cl, xn, st, caps);

	glm5.hc_apply(streams, y, am.post, am.comb);

	let fm = hc_mixes(cfg, L.hc_ffn, streams);

	xn = torch.rms_norm(hc_collapse(streams, am.pre), L.ffn_norm, eps);
	y = glm5.moe_forward(cfg, L.moe, xn);
	glm5.hc_apply(streams, y, fm.post, fm.comb);

	return fm.pre;
}

/* ------------------------------------------------------------------ */
/* planner + model                                                     */

/* Bytes of the pool (compressed rows + index keys) of kv source layer src for max_seq positions */
export function pool_bytes(cfg, src, max_seq) {
	return (int(max_seq / cfg.ratios[src]) + 2) * (cfg.head_dim + cfg.index_head_dim) * 2;
}

/* Contiguous layer ranges for ngpu devices that minimize the fullest device: checkpoint bytes of the
 * layers (the embedding on the first device, the head on the last) plus the pools it must hold (its
 * own source layers' and a mirror of every pool its layers read from another device). Returns the
 * per-layer device array. */
export function plan_devices(ws, cfg, ngpu, max_seq, extra_first, extra_last) {
	let nl = cfg.num_hidden_layers;
	let bytes = [];

	for (let i = 0; i < nl; i++)
		push(bytes, 0);

	for (let key in ws.list()) {
		if (index(key, 'layers.') == 0 && index(key, '.engram.embed') < 0) {
			let idx = +substr(key, 7, index(key, '.', 7) - 7);

			if (idx >= 0 && idx < nl)
				bytes[idx] += ws.info(key).nbytes;
		}
	}

	function cost(a, b) {                            // layers [a, b) on one device
		let c = 0, own = {}, mir = {};

		for (let i = a; i < b; i++) {
			c += bytes[i];

			if (index(cfg.kv_sources, i) >= 0)
				own[i] = true;
		}

		for (let i = a; i < b; i++)
			if (cfg.src[i] >= 0 && !own[cfg.src[i]])
				mir[cfg.src[i]] = true;

		for (let k in keys(own))
			c += pool_bytes(cfg, +k, max_seq);

		for (let k in keys(mir))
			c += pool_bytes(cfg, +k, max_seq);

		return c + (a == 0 ? (extra_first || 0) : 0) + (b == nl ? (extra_last || 0) : 0);
	}

	// best[d][i]: smallest possible maximum over d devices holding the first i layers
	let best = [], from = [];

	for (let d = 0; d <= ngpu; d++) {
		push(best, []);
		push(from, []);

		for (let i = 0; i <= nl; i++) {
			push(best[d], (d == 0 && i == 0) ? 0 : 1e30);
			push(from[d], -1);
		}
	}

	for (let d = 1; d <= ngpu; d++)
		for (let i = 1; i <= nl; i++)
			for (let a = d - 1; a < i; a++) {
				if (best[d - 1][a] >= 1e30)
					continue;

				let m = max(best[d - 1][a], cost(a, i));

				if (m < best[d][i]) {
					best[d][i] = m;
					from[d][i] = a;
				}
			}

	let devs = [];

	for (let i = 0; i < nl; i++)
		push(devs, 0);

	for (let d = ngpu, i = nl; d >= 1; d--) {
		let a = from[d][i];

		for (let j = a; j < i; j++)
			devs[j] = d - 1;

		i = a;
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
		devs = plan_devices(ws, cfg, devs, opts.max_seq || 262144, 1400000000, 1400000000);

	let first = devs[0], last = devs[nl - 1];

	glm5.set_fused_rows(max(256, opts.prefill_rows || 1024));

	let meta = build_engram_meta(model_dir, cfg);

	ws.upload(function (key) {
		if (index(key, 'layers.') == 0) {
			if (index(key, '.engram.embed') > 0)
				return null;                         // streamed from disk

			let idx = +substr(key, 7, index(key, '.', 7) - 7);

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (key == 'embed.weight')
			return first;

		if (key == 'head.weight' || key == 'norm.weight')
			return last;

		return null;      // mtp / vision: not part of the text model yet
	});

	let layers = [];

	for (let i = 0; i < nl; i++)
		push(layers, build_layer(ws, cfg, i, devs[i]));

	let engrams = {};

	for (let l in cfg.engram_layers)
		if (l < nl)
			engrams[l] = build_engram(ws, cfg, l, devs[l]);

	let embed_t = ws.gpu16('embed.weight', first);

	ws.release('embed.weight', first);

	let model = {
		cfg: cfg, devs: devs, ws: ws, layers: layers, engrams: engrams, meta: meta, embed_tokens: embed_t,
		norm: ws.gpu16('norm.weight', last),
		lm_head: models.linear(ws, 'head', false, last),

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: [], pools: {}, cids: [], toks: []};

			for (let i = 0; i < nl; i++) {
				let cl = new_layer_cache(this.cfg, this.layers[i].attn, max_seq);

				push(cache.layers, cl);

				if (this.layers[i].attn.kv_source)
					cache.pools[i] = {owner: cl, mirrors: []};
			}

			return cache;
		},

		/* Run token ids (array) appended to the cache. Returns logits (rows, vocab) fp16 for the last
		 * `rows` positions (default 1), or null for rows == 0. */
		forward_chunk: function (token_ids, cache, rows) {
			let cfg = this.cfg;
			let T = length(token_ids);
			let pos0 = cache.length;
			let H = cfg.hidden_size;

			if (rows == null)
				rows = 1;

			// token history for the engram n-grams (a rollback truncates it)
			if (length(cache.cids) > pos0)
				splice(cache.cids, pos0, length(cache.cids) - pos0);

			let idt = torch.from_array(token_ids, 'int64');
			let cn = this.meta.token_map.index(idt).view([-1]).tolist();
			let hist = [];

			for (let k = 3; k >= 1; k--)
				push(hist, (pos0 - k >= 0) ? cache.cids[pos0 - k] : this.meta.pad);

			for (let c in cn)
				push(hist, c);

			let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

			for (let c in cn)
				push(cache.cids, c);

			let h = this.embed_tokens.index(torch.from_array(token_ids, 'int64', 'cuda:' + first)).to('float32')
				.reshape([T, 1, H]).expand([T, 4, H]).contiguous();
			let pre = torch.zeros([T, 4], 'float32', 'cuda:' + first);

			pre.narrow(1, 0, 1).fill_(1.0);

			let ctx = {topk: null, cand: null};

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];

				if (i > 0 && L.dev != this.layers[i - 1].dev) {
					h = h.cuda(L.dev);
					pre = pre.cuda(L.dev);
				}

				let li = index(cfg.engram_layers, i);

				if (li >= 0) {
					let E = this.engrams[i];

					engram_apply(cfg, E, h, engram_rows(cfg, this.meta, E, li, hist_t, T));
				}

				pre = block_forward(cfg, L, ctx, cache, cache.layers[i], h, pre, pos0);
			}

			cache.length = pos0 + T;

			if (rows == 0)
				return null;

			let x = hc_collapse(h, pre);

			x = torch.rms_norm(x, this.norm, cfg.rms_norm_eps);

			if (rows < T)
				x = x.narrow(0, T - rows, rows).contiguous();

			return this.lm_head.forward(x);
		},

		/* CUDA-graph decode (engine interface, as glm5.graphs): one graph per run of layers on a device and
		 * per context bucket (the capacity of the pools the graph reads), position and slots on the device.
		 * sizes: only S = 1. */
		graphs: function (cache, sizes) {
			let model = this;
			let cfg = this.cfg;
			let H = cfg.hidden_size;
			let segs = [];
			let max_seq = cache.max_seq;

			for (let i = 0; i < nl; i++) {
				let L = this.layers[i];

				if (length(segs) && segs[length(segs) - 1].dev == L.dev)
					segs[length(segs) - 1].end = i + 1;
				else
					push(segs, {first: i, end: i + 1, dev: L.dev});
			}

			let sts = {};

			for (let sg in segs)
				if (!sts[sg.dev])
					sts[sg.dev] = new_dec_dev(cfg, sg.dev);

			// buckets: capacity (positions) of the ratio-1 pools a graph reads
			let caps_list = [];

			for (let c = 4096; c < max_seq; c *= 4)
				push(caps_list, c);

			push(caps_list, max_seq);

			let gpool = {};
			let buckets = [];

			for (let c1 in caps_list) {
				let caps = [ 0, c1, int((c1 + 1) / 2) ];
				let ctx = {topk: null, cand: null, links: []};
				let items = [];

				for (let sg in segs) {
					let dv = 'cuda:' + sg.dev;
					let it = {first: sg.first, end: sg.end, dev: sg.dev,
						h_in: torch.zeros([1, 4, H], 'float32', dv), pre_in: torch.zeros([1, 4], 'float32', dv),
						logits: null, last: (sg.end == nl)};

					it.pre_in.narrow(1, 0, 1).fill_(1.0);
					push(items, it);
				}

				function body(it) {
					let h = it.h_in, pre = it.pre_in;

					for (let i = it.first; i < it.end; i++) {
						let li = index(cfg.engram_layers, i);

						if (li >= 0) {
							let E = model.engrams[i];

							engram_apply(cfg, E, h, E.emb_buf);
						}

						pre = block_decode(cfg, model.layers[i], ctx, cache, cache.layers[i], h, pre, sts[it.dev], caps);
					}

					it.pre_out = pre;

					if (it.last) {
						let x = torch.rms_norm(hc_collapse(h, pre), model.norm, cfg.rms_norm_eps);

						it.logits = model.lm_head.forward(x);
					}

					it.h_out = h;
				}

				for (let sg in segs)
					dec_set_pos(cfg, sts[sg.dev], 0);

				// eager warm-up (also creates the mirrors and the remote copies), then capture
				for (let it in items) {
					body(it);
					torch.synchronize();
				}

				for (let it in items) {
					it.graph = torch.graph();
					it.graph.capture_begin(it.dev, null, gpool[it.dev]);
					body(it);
					it.graph.capture_end();

					if (!gpool[it.dev])
						gpool[it.dev] = it.graph.pool();
				}

				torch.synchronize();
				push(buckets, {c1: c1, items: items, ctx: ctx});
			}

			// the warm-up and the captures wrote position 0 of every cache: start from empty again
			model.aux_reset(cache);

			function run_ids(tid, pos) {
				let b = null;

				for (let x in buckets)
					if (x.c1 > pos && !b)
						b = x;

				if (!b)
					die(`dsv41: position ${pos} is past the context of ${max_seq}`);

				if (length(cache.cids) > pos)
					splice(cache.cids, pos, length(cache.cids) - pos);

				for (let sg in segs)
					dec_set_pos(cfg, sts[sg.dev], pos);

				// Engram rows of the token: hashed on the host, gathered from the table, into the layers' buffers
				let cn = model.meta.token_map.index(torch.from_array([ tid ], 'int64')).view([-1]).tolist()[0];
				let hist = [];

				for (let k = 3; k >= 1; k--)
					push(hist, (pos - k >= 0) ? cache.cids[pos - k] : model.meta.pad);

				push(hist, cn);

				let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

				for (let li = 0; li < length(cfg.engram_layers); li++) {
					let l = cfg.engram_layers[li];

					if (l < nl) {
						let E = model.engrams[l];

						E.emb_buf.copy_(engram_rows(cfg, model.meta, E, li, hist_t, 1));
					}
				}

				push(cache.cids, cn);

				let h = model.embed_tokens.index(torch.from_array([ tid ], 'int64', 'cuda:' + first)).to('float32')
					.reshape([1, 1, H]).expand([1, 4, H]).contiguous();
				let pre = null, logits = null;

				for (let it in b.items) {
					it.h_in.copy_(h);

					if (pre)
						it.pre_in.copy_(pre);

					it.graph.replay();
					h = it.h_out;
					pre = it.pre_out;

					for (let lk in b.ctx.links)
						lk.dst.copy_(lk.src);

					for (let i = it.first; i < it.end; i++) {
						let A = model.layers[i].attn;

						if (A.kv_source)
							mirror_entry(cache, i, int(pos / A.ratio));
					}

					if (it.last)
						logits = it.logits;
				}

				for (let i = 0; i < nl; i++) {
					let A = model.layers[i].attn;

					if (A.kv_source)
						cache.layers[i].n = int((pos + 1) / A.ratio);
				}

				cache.length = pos + 1;

				return {logits: logits, hidden: null};
			}

			return {
				has: function (S) {return S == 1;},

				run: function (ids, pos) {
					return run_ids(ids[0], pos);
				},

				run_dev: function (ids_dev, pos) {
					return run_ids(ids_dev.view([1]).tolist()[0], pos);
				}
			};
		},

		/* Recurrent state outside the KV rows, for the engine's prompt snapshots: per layer the pool entry
		 * count, the half-built pair of the ratio-2 compressors (the state objects are replaced, never
		 * mutated in place, so holding the references is a snapshot) and a copy of the window rows (the
		 * ring is overwritten by later tokens). */
		aux_get: function (cache) {
			let out = [];
			let W = this.cfg.window - 1;

			for (let cl in cache.layers) {
				let n = min(cache.length, W);

				push(out, {n: cl.n, pk: (cl.pk && cache.length % 2 == 1) ? cl.pk.add(0.0) : null,
					pg: (cl.pk && cache.length % 2 == 1) ? cl.pg.add(0.0) : null,
					kw: n > 0 ? ring_rows(cl, cache.length - n, cache.length).add(0.0) : null});
			}

			return out;
		},

		aux_set: function (cache, snap) {
			for (let i = 0; i < length(cache.layers); i++) {
				let cl = cache.layers[i], v = snap[i];

				if (this.layers[i].attn.kv_source) {
					cl.n = v.n;

					if (v.pk) {
						cl.pk.copy_(v.pk);
						cl.pg.copy_(v.pg);
					}
				}

				if (v.kw)
					ring_put(cl, cache.length - v.kw.shape()[0], v.kw);
			}
		},

		aux_reset: function (cache) {
			for (let cl in cache.layers) {
				if (cl.pool)
					cl.n = 0;
			}

			cache.cids = [];
		},

		/* Engine interface (as dsv4.prefill_pipeline): chunked prefill of ids at cache.length, one chunk per
		 * round. opts: chunk, cuts (extra chunk boundaries), on_aux(chunk_end) after every chunk. */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
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

			let next = 0, done = 0, result = null;

			return {
				step: function (rounds) {
					for (let r = 0; r < (rounds || 1) && next < length(bounds); r++) {
						let end = bounds[next++];
						let pos = cache.length;
						let last = (end == start + total);
						let lg = model.forward_chunk(slice(ids, pos - start, end - start), cache, last ? 1 : 0);

						if (last)
							result = lg;

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
		},

		/* Prefill ids in chunks of `chunk` tokens (default 512). */
		prefill: function (token_ids, cache, chunk, rows) {
			let n = length(token_ids);

			chunk = chunk || 512;

			for (let p = 0; p < n; p += chunk) {
				let e = min(p + chunk, n);
				let lastc = (e == n);
				let r = this.forward_chunk(slice(token_ids, p, e), cache, lastc ? (rows == null ? 1 : rows) : 0);

				if (lastc)
					return r;
			}

			return null;
		}
	};

	return model;
}
