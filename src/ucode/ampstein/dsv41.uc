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
import * as ggufq from './ggufq.uc';
import {readfile, stat} from 'fs';

const NEGH = -60000.0;
const MAX_ROWS = 8;                       // rows of a multi-row decode step
const RING = +(getenv('AMP_DSV41_RING') || 8192);          // raw window rows kept per layer (rollback span)
const SCRATCH_ELEMS = 32 * 1024 * 1024;                    // fp8 -> fp16 dequant slice, elements
const HPROF = getenv('AMP_DSV41_HPROF') != null;
const PROF = getenv('AMP_DSV41_PROF') != null || HPROF;
const NO_ENGRAM = getenv('AMP_DSV41_NO_ENGRAM') != null;
const NO_GROUPED = getenv('AMP_DSV41_NO_GROUPED') != null;        // wo_a: one GEMV per group instead of the grouped kernel
const NO_HC2 = getenv('AMP_DSV41_NO_HC2') != null;            // composite mHC mixes + collapse instead of the fused hc_mix2
let ROWOFF = {};
const SKIP = getenv('AMP_DSV41_SKIP');                      // profiling: 'attn' | 'moe' skipped in multi-row steps
const ROWLOOP = getenv('AMP_DSV41_ROWLOOP') != null;          // multi-row decode: attention row by row (reference path)

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
		layer_types: map(slice(ratios, 0, nl), (r) => r ? 'compressed' : 'sliding'),
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
		dspark_experts: t.dspark_n_routed_experts || 0,
		dspark_topk: t.dspark_num_experts_per_tok || 0,
		dspark_rank: t.dspark_markov_rank || 0,

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

/* AMP_DSV41_PROF=1: accumulate the time of the phases of the eager path (synchronizes between them) */
export let PROFILE = {};
export let HOSTPROF = getenv('AMP_DSV41_HOSTPROF') ? [] : null;
let prof_t = 0;

function prof_now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

function prof(name) {
	if (!PROF)
		return;

	if (!HPROF)                     // AMP_DSV41_HPROF=1: host time only (no synchronize): where the launches go
		torch.synchronize();

	let t = prof_now();

	if (name != null)
		PROFILE[name] = (PROFILE[name] || 0) + (t - prof_t);

	prof_t = t;
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
/* the same from the tensors (w (N, K) e4m3, s (N / 32, K / 32) e8m0) */
export function lin8_from(w, s, dev) {
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

/* The linear under a checkpoint key, whatever the pack stores there: block-scaled FP8 (key + '.weight' / '.scale': the
 * SAGE packs), EXL3 (key + '.trellis': the 2.50bpw packs) or a plain fp16 / bf16 matrix. The three behave alike:
 * forward(x) with x (M, K) fp16 -> (M, N) fp16, in_features, out_features, and the tensor-parallel shards below. */
export function lin8(ws, key, dev) {
	if (ws.gguf && ws.qtype(key + '.weight'))            // a GGUF checkpoint: ggml block-quantized rows (ggufq.uc)
		return ggufq.linear(ws, key, dev);

	if (ws.has(key + '.trellis'))
		return models.LinearEXL3(ws, key, dev);

	if (!ws.has(key + '.scale'))
		return models.LinearFP16(ws, key, dev);

	return lin8_from(ws.gpu(key + '.weight', dev), ws.gpu(key + '.scale', dev), dev);
}

/* Grouped FP8 linear (the attention's wo_a): G blocks of R rows, block g sees columns [g * K, (g + 1) * K)
 * of x only. x: (M, G * K) fp16 -> (M, G * R). */
export function lin8g_from(w, s, G, dev) {
	let sh = w.shape();

	return {
		w: w, s: s, dev: dev, groups: G, rows: int(sh[0] / G), in_features: sh[1],

		forward: function (x) {
			let M = x.shape()[0], K = this.in_features, R = this.rows;
			let parts = [];

			if (M <= 8 && !NO_GROUPED)
				return torch.fp8_gemv_grouped(x.contiguous(), this.w, this.s, G);

			// a prefill chunk: dequantize the whole matrix once and run the G groups as one batched matmul
			if (M > 8 && !NO_GROUPED && this.w.numel() <= SCRATCH_ELEMS) {
				let wd = torch.fp8_dequant(this.w, this.s, scratch(this.dev)).view([G, R, K]).transpose(1, 2);
				let y = torch.matmul(x.contiguous().view([M, G, K]).transpose(0, 1), wd);          // (G, M, R)

				return y.transpose(0, 1).reshape([M, G * R]);
			}

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

/* Grouped EXL3 linear (wo_a of the 2.50bpw packs: one matrix per group, key + '.slice.g'). */
function lin3g_from(slices, dev) {
	return {
		slices: slices, dev: dev, groups: length(slices), rows: slices[0].out_features, in_features: slices[0].in_features,

		forward: function (x) {
			let K = this.in_features, parts = [];

			for (let g = 0; g < this.groups; g++)
				push(parts, this.slices[g].forward(x.narrow(1, g * K, K).contiguous()));

			return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
		}
	};
}

export function lin8g(ws, key, G, dev) {
	if (ws.gguf && ws.qtype(key + '.weight'))
		return ggufq.linear_grouped(ws, key, G, dev);

	if (ws.has(key + '.slice.0.trellis')) {
		let sl = [];

		for (let g = 0; g < G; g++)
			push(sl, models.LinearEXL3(ws, `${key}.slice.${g}`, dev));

		return lin3g_from(sl, dev);
	}

	return lin8g_from(ws.gpu(key + '.weight', dev), ws.gpu(key + '.scale', dev), G, dev);
}

/* tensor-parallel shards of FP8 linears (copies on dev) */
export function lin8_to(l, dev) {
	if (l.qtype)
		return ggufq.linear_to(l, dev);

	if (l.trellis)
		return models.cols(l, [ [ 0, l.out_features ] ], dev);

	if (l.weight)
		return models.fp16_cols(l, 0, l.out_features, dev);

	return lin8_from(l.w.cuda(dev), l.s.cuda(dev), dev);
}

/* output rows [r0, r0 + n) */
export function lin8_rows(l, r0, n, dev) {
	if (l.qtype)
		return ggufq.linear_rows(l, r0, n, dev);

	if (l.trellis)
		return models.cols(l, [ [ r0, n ] ], dev);

	if (l.weight)
		return models.fp16_cols(l, r0, n, dev);

	return lin8_from(l.w.narrow(0, r0, n).contiguous().cuda(dev), l.s.narrow(0, int(r0 / 32), int(n / 32)).contiguous().cuda(dev), dev);
}

/* input columns [c0, c0 + n): the partial sum of a row-parallel layer */
export function lin8_cols(l, c0, n, dev) {
	if (l.qtype)
		return ggufq.linear_cols(l, c0, n, dev);

	if (l.trellis)
		return models.rows(l, c0, n, dev);

	if (l.weight) {
		let w = l.weight.narrow(1, c0, n).contiguous().cuda(dev);

		return {weight: w, bias: null, in_features: n, out_features: w.shape()[0],
			forward: function (x) {return torch.matmul(x, this.weight.t());}};
	}

	return lin8_from(l.w.narrow(1, c0, n).contiguous().cuda(dev), l.s.narrow(1, int(c0 / 32), int(n / 32)).contiguous().cuda(dev), dev);
}

/* groups [g0, g0 + n) of a grouped linear */
export function lin8g_groups(l, g0, n, dev) {
	if (l.qtype)
		return ggufq.grouped_groups(l, g0, n, dev);

	if (l.slices)
		return lin3g_from(map(slice(l.slices, g0, g0 + n), (s) => models.cols(s, [ [ 0, s.out_features ] ], dev)), dev);

	let R = l.rows;

	return lin8g_from(l.w.narrow(0, g0 * R, n * R).contiguous().cuda(dev),
		l.s.narrow(0, int(g0 * R / 32), int(n * R / 32)).contiguous().cuda(dev), n, dev);
}

/* A compressor projection of xn (M, K) fp16 (x32: the same in fp32): the dense matrix (K, N) fp32 or an EXL3 linear */
function cmm32(xn, x32, W) {
	return (type(W) == 'object') ? W.forward(xn).to('float32') : torch.matmul(x32, W);
}

function cmm16(xn, W) {
	return (type(W) == 'object') ? W.forward(xn) : torch.matmul(xn, W);
}

/* a copy of such a projection (matrix or linear) on dev */
export function comp_to(W, dev) {
	return (type(W) == 'object') ? lin8_to(W, dev) : W.cuda(dev);
}

/* bf16 / fp32 matrix used in fp32: x (M, K) -> (M, N) fp32 */
export function mat32(ws, key, dev) {
	let m = ws.gpu(key + '.weight', dev).to('float32').t().contiguous();

	ws.release(key + '.weight', dev);            // the bf16 upload is not needed any more

	return m;
}

/* ------------------------------------------------------------------ */
/* rope                                                                */

export function rope_cs(inv_freq, p0, T, step, dev) {
	let h = inv_freq.shape()[0];
	let ang = torch.arange(T, 'float32', 'cuda:' + dev).mul(step * 1.0).add(p0 * 1.0)
		.view([T, 1]).mul(inv_freq.view([1, h]));

	return [ang.cos().view([T, 1, h, 1]), ang.sin().view([T, 1, h, 1])];
}

/* x (T, H, D) fp32: rotate the trailing rd channels pairwise (GPT-J) */
export function rope_il(x, cs, rd) {
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

/* One mHC site of the "one site late" form in a single fused pass: the mixes of the streams (post, comb, this
 * site's pre) and the sublayer input col = streams collapsed with pre_prev (the previous site's pre), (T, D) fp16
 * saturated to the fp16 range. Returns {pre, post, comb, col}. */
export function hc_site(cfg, hc, streams, pre_prev) {
	if (NO_HC2) {
		let m = hc_mixes(cfg, hc, streams);

		return {pre: m.pre, post: m.post, comb: m.comb, col: hc_collapse(streams, pre_prev)};
	}

	let r = torch.hc_mix2(streams, hc.fn, hc.base, hc.scale, cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters,
		pre_prev);

	return {post: r[0], comb: r[1], col: r[2], pre: r[3]};
}

/* (T, D) fp16: streams collapsed with pre, saturated to the fp16 range (massive activations) */
export function hc_collapse(streams, pre) {
	let T = streams.shape()[0];

	return streams.mul(pre.view([T, 4, 1])).sum(1).clamp(-65504.0, 65504.0).to('float16');
}

/* ------------------------------------------------------------------ */
/* Engram                                                              */

export function build_engram_meta(ws_dir, cfg, ws) {
	let path = ws_dir + '/engram_meta.safetensors';
	let st = null, T;

	if (ws && ws.gguf) {
		T = ws.engram_meta(cfg.engram_pad_id);          // the hash tables are tensors of the GGUF itself
	}
	else {
		if (!stat(path))
			die(`dsv41: ${path} is missing; generate it once with: python3 tests/oracle/make_engram_meta.py ${ws_dir}`);

		st = torch.st(path);
		T = {token_map: st.tensor('token_map'), pad: st.tensor('pad'), multipliers: st.tensor('multipliers'),
			offsets: st.tensor('offsets'), primes: st.tensor('primes')};
	}

	let M = {
		token_map: T.token_map.to('int64').view([-1, 1]),
		pad: T.pad.to('int64').tolist()[0],
		keep: st ?? T,                      // the tensors below may view its mapping
		mult: [], offs: [], sizes: []
	};
	let mult = T.multipliers.to('int64'), offs = T.offsets.to('int64');
	let primes = T.primes.to('int64');

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
		// (a GGUF checkpoint: IQ4_NL rows, no scale tensor)
		qt: (ws.gguf && ws.has(key + '.embed.weight')) ? ws.qtype(key + '.embed.weight') : null,
		row_bytes: (ws.gguf && ws.has(key + '.embed.weight')) ? int(ws.info(key + '.embed.weight').nbytes / ws.info(key + '.embed.weight').shape[0]) : 0,
		emb_buf: torch.zeros([1, (cfg.engram_ngram - 1) * cfg.engram_heads * cfg.engram_dim], 'float16', 'cuda:' + dev),
		emb_bufs: {},
		wname: key + '.embed.weight',
		sname: key + '.embed.scale'
	};
}

/* Hash ids of the T tokens of a chunk for engram layer slot `li`, gathered from the table and
 * dequantized: (T, heads * 3 * dim) fp16 on E.dev. cids: compressed ids of [3 history tokens, chunk]. */
/* The rows of the T tokens of a chunk for engram layer slot `li` (history hist_t): hashes on the host, then the table
 * rows are read from disk by a worker thread. engram_finish waits for them. */
export function engram_start(cfg, meta, E, li, hist_t, T) {
	let H = (cfg.engram_ngram - 1) * cfg.engram_heads;

	if (NO_ENGRAM)                                 // tests without the 200 GB tables
		return {zero: true, T: T};

	let n = T * H;
	let uids = torch.zeros([n], 'int64'), inverse = torch.zeros([n], 'int64'), heads = torch.zeros([n], 'int32');
	let U = torch.ngram_hash(hist_t, T, meta.mult[li], meta.offs[li], meta.sizes[li], cfg.engram_heads,
		meta.pad, uids, inverse, heads);
	let u = uids.narrow(0, 0, U).contiguous();
	if (E.qt) {
		let rq = torch.empty([U, E.row_bytes], 'uint8');

		return {T: T, U: U, inverse: inverse, rq: rq, u: u, t1: E.file.gather_async(E.wname, u, rq)};
	}

	let rw = torch.empty([U, cfg.engram_dim], 'float8_e4m3fn');
	let rsc = torch.empty([U, cfg.engram_dim / 32], 'float8_e8m0fnu');

	return {T: T, U: U, inverse: inverse, rw: rw, rsc: rsc, u: u,
		t1: E.file.gather_async(E.wname, u, rw), t2: E.file.gather_async(E.sname, u, rsc)};
}

/* (T, heads * 3 * dim) fp16 on E.dev from a started chunk */
export function engram_finish(cfg, E, st) {
	let H = (cfg.engram_ngram - 1) * cfg.engram_heads;

	if (st.zero)
		return torch.zeros([st.T, H * cfg.engram_dim], 'float16', 'cuda:' + E.dev);

	let U = st.U;

	torch.gather_wait(st.t1);

	if (st.rq) {
		let rows = torch.gguf_dequant(st.rq.cuda(E.dev, true), E.qt, 'float16');

		return rows.index(st.inverse.cuda(E.dev, true)).view([st.T, H * cfg.engram_dim]);
	}

	torch.gather_wait(st.t2);

	// non-blocking copies (staged through the caching pinned allocator): the host never waits for the device,
	// so the chunks of a prefill keep flowing through the pipeline of GPUs
	let ew = st.rw.cuda(E.dev, true).to('float32').view([U, cfg.engram_dim / 32, 32]);
	let es = st.rsc.cuda(E.dev, true).to('float32').view([U, cfg.engram_dim / 32, 1]);
	let rows = ew.mul(es).view([U, cfg.engram_dim]).to('float16');

	return rows.index(st.inverse.cuda(E.dev, true)).view([st.T, H * cfg.engram_dim]);
}

export function engram_rows(cfg, meta, E, li, hist_t, T) {
	return engram_finish(cfg, E, engram_start(cfg, meta, E, li, hist_t, T));
}

/* Start reading the table rows of the tokens ids (which follow the cache's history) into the page cache
 * without waiting: the hashes depend on the token ids only, so a whole prompt can be prefetched while the
 * first chunks compute. */
export function engram_prefetch(cfg, meta, engrams, cache, ids, pos0, lo, hi) {
	let H = (cfg.engram_ngram - 1) * cfg.engram_heads;
	let n = min(length(ids), hi == null ? length(ids) : hi);
	let SL = 4096;

	for (let a = lo || 0; a < n; a += SL) {
		let T = min(SL, n - a);
		let cn = meta.token_map.index(torch.from_array(slice(ids, a, a + T), 'int64')).view([-1]).tolist();
		let hist = [];

		for (let k = 3; k >= 1; k--) {
			let q = pos0 + a - k;

			if (q < 0)
				push(hist, meta.pad);
			else if (q < pos0)
				push(hist, (q < length(cache.cids)) ? cache.cids[q] : meta.pad);
			else
				push(hist, meta.token_map.index(torch.from_array([ ids[q - pos0] ], 'int64')).view([-1]).tolist()[0]);
		}

		for (let c in cn)
			push(hist, c);

		let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

		for (let li = 0; li < length(cfg.engram_layers); li++) {
			let E = engrams[cfg.engram_layers[li]];

			if (!E)
				continue;

			let nn = T * H;
			let uids = torch.zeros([nn], 'int64'), inverse = torch.zeros([nn], 'int64'), heads = torch.zeros([nn], 'int32');
			let U = torch.ngram_hash(hist_t, T, meta.mult[li], meta.offs[li], meta.sizes[li], cfg.engram_heads,
				meta.pad, uids, inverse, heads);
			let u = uids.narrow(0, 0, U).contiguous();

			E.file.prefetch(E.wname, u);

			if (!E.qt)
				E.file.prefetch(E.sname, u);
		}
	}
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

		// (a dense matrix as before, or an EXL3 linear in the 2.50bpw packs: cmm32 / cmm16 below)
		if (ratio > 1) {
			A.comp.wkv = ws.has(key + '.compressor.wkv.trellis') ? models.LinearEXL3(ws, key + '.compressor.wkv', dev) :
				mat32(ws, key + '.compressor.wkv', dev);
			A.comp.wgate = ws.has(key + '.compressor.wgate.trellis') ? models.LinearEXL3(ws, key + '.compressor.wgate', dev) :
				mat32(ws, key + '.compressor.wgate', dev);
		}
		else {
			A.comp.wkv16 = ws.has(key + '.compressor.wkv.trellis') ? models.LinearEXL3(ws, key + '.compressor.wkv', dev) :
				ws.gpu16(key + '.compressor.wkv.weight', dev).t().contiguous();
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

/* ---- pools of compressed rows: fp16, or Q8 (AMP_DSV41_KV=q8): int8 codes + one fp16 scale per 32 elements ---- */

const PROMPT_SKIP = getenv('AMP_DSV41_PROMPT_SKIP') != '0';     // the prompt skip (prefill_pipeline), docs/DSV41.md

// score elements of one tile of the prefill indexer (rows x pool entries; the topk behind it needs a few times that in workspace):
// AMP_DSV41_IDX_SLAB; a GGUF model, which has little VRAM left over, takes a smaller tile (set in load)
let IDX_SLAB = +(getenv('AMP_DSV41_IDX_SLAB') || 32 * 1024 * 1024);

const KV_MODE = getenv('AMP_DSV41_KV') || 'q8';                   // q8 (pool + index keys) | q8pool (pool only) | fp16
const KV_Q8 = (KV_MODE == 'q8' || KV_MODE == 'q8pool');

/* kind: 'pool' (the compressed rows) or 'kidx' (the indexer keys) */
function pool_new(kind, cap, dim, dv) {
	if (KV_MODE == 'q8' || (KV_MODE == 'q8pool' && kind == 'pool'))
		return {q: torch.zeros([cap, dim], 'int8', dv), s: torch.zeros([cap, int(dim / 32)], 'float16', dv)};

	return torch.zeros([cap, dim], 'float16', dv);
}

function pool_is_q8(P) {
	return type(P) == 'object';
}

function pool_dim(P) {
	return pool_is_q8(P) ? P.q.shape()[1] : P.shape()[1];
}

function pool_cap(P) {
	return pool_is_q8(P) ? P.q.shape()[0] : P.shape()[0];
}

/* src (n, dim) fp16 into rows `at` (an integer first row, or an int64 tensor of n destination rows) */
function pool_put(P, src, at) {
	let n = src.shape()[0];

	if (pool_is_q8(P))
		torch.kv_q8_put(src.contiguous(), P.q, P.s, at);
	else if (type(at) == 'int')
		P.narrow(0, at, n).copy_(src);
	else
		P.index_copy_(0, at, src);
}

/* what the kernels take for the first n rows: a tensor, or [codes, scales] */
function pool_view(P, n) {
	return pool_is_q8(P) ? [ P.q.narrow(0, 0, n), P.s.narrow(0, 0, n) ] : P.narrow(0, 0, n);
}

/* rows [a, a + k) of S into D (the other device's mirror) */
function pool_copy(D, S, a, k) {
	if (pool_is_q8(S)) {
		D.q.narrow(0, a, k).copy_(S.q.narrow(0, a, k));
		D.s.narrow(0, a, k).copy_(S.s.narrow(0, a, k));
	}
	else {
		D.narrow(0, a, k).copy_(S.narrow(0, a, k));
	}
}

/* The owner entry of a pool whose rows are written from elsewhere (a source layer on a worker host, docs/RPC.md): the
 * pool and index keys of cache.pools[src].owner on dev, filled by the host from what the worker sends. */
export function new_pool_stub(cfg, ratio, max_seq, dev) {
	let cap = int(max_seq / ratio) + 2, dv = 'cuda:' + dev;

	return {dev: dev, n: 0, pool: pool_new('pool', cap, cfg.head_dim, dv), kidx: pool_new('kidx', cap, cfg.index_head_dim, dv)};
}

/* per-layer window ring + (for kv sources) the pool, index keys and the half-built pair */
export function new_layer_cache(cfg, A, max_seq) {
	let dv = 'cuda:' + A.dev;
	let cl = {
		dev: A.dev,
		ring: torch.zeros([RING, cfg.head_dim], 'float16', dv)
	};

	if (A.kv_source) {
		let cap = int(max_seq / A.ratio) + 2;

		cl.pool = pool_new('pool', cap, cfg.head_dim, dv);
		cl.kidx = pool_new('kidx', cap, cfg.index_head_dim, dv);
		cl.n = 0;                       // pool entries written
		// ratio 2: the kv / gate rows of the first token of a half-built pair (valid while the next
		// position is odd); persistent so that graph steps update them in place
		cl.pk = torch.zeros([1, cfg.head_dim], 'float32', dv);
		cl.pg = torch.zeros([1, cfg.head_dim], 'float32', dv);

		// the half-built pair after each row of a multi-row step (speculative verify), for rollbacks
		if (A.ratio == 2) {
			cl.rk = torch.zeros([MAX_ROWS, cfg.head_dim], 'float32', dv);
			cl.rg = torch.zeros([MAX_ROWS, cfg.head_dim], 'float32', dv);
		}
	}

	return cl;
}

/* ring rows of absolute positions [a, b) (b - a <= RING) */
export function ring_rows(cl, a, b) {
	let n = b - a;

	if (n <= 0)
		return torch.zeros([0, 512], 'float16', 'cuda:' + cl.dev);

	let s = a % RING;

	if (s + n <= RING)
		return cl.ring.narrow(0, s, n);

	return torch.cat([cl.ring.narrow(0, s, RING - s), cl.ring.narrow(0, 0, n - (RING - s))], 0);
}

export function ring_put(cl, pos0, rows) {
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

/* The entry count of a source layer's pool moves (a reset, a rewind, new writes from entry n on): the mirrors
 * must forget what they hold from n on, they would otherwise keep the rows of the previous sequence. */
export function set_pool_n(cache, src, n) {
	let P = cache.pools[src];

	P.owner.n = n;

	for (let m in P.mirrors)
		if (m.n > n)
			m.n = n;
}

/* Feed the chunk (xn (T, D) fp16 at absolute pos0) to the layer's compressor: appends the completed
 * pool entries (rope'd) and index keys. Returns the number of new entries. */
function compress_chunk(cfg, A, cache, cl, xn, pos0) {
	let C = A.comp, ratio = A.ratio;
	let dv = 'cuda:' + A.dev;
	let rd = cfg.rope_dim, eps = cfg.rms_norm_eps;
	let T = xn.shape()[0];
	let lat;                                          // (n, 512) fp16 pre-rope, normed
	let base;                                         // absolute position of the first entry's first token

	if (ratio == 1) {
		lat = torch.rms_norm(cmm16(xn, C.wkv16), C.norm, eps);
		base = pos0;
	}
	else {
		let x32 = xn.to('float32');
		let kv = cmm32(xn, x32, C.wkv), g = cmm32(xn, x32, C.wgate);

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

	// a rewind (or the engine's warm-up, which jumps to the end of the context) leaves the count elsewhere:
	// the entries from e0 on are written from here
	set_pool_n(cache, A.idx, e0);

	// index key: k_norm(wk(latent)), rope at the entry's position
	let I = A.indexer;
	let posf = torch.arange(n, 'float32', dv).mul(ratio * 1.0).add(base * 1.0);

	if (I && I.owns_k) {
		let k = torch.rms_norm(torch.matmul(lat, I.wk_t), I.k_norm, eps);

		pool_put(cl.kidx, torch.v41_rope(k.view([n, 1, cfg.index_head_dim]), posf, A.inv_freq).view([n, cfg.index_head_dim]), e0);
	}

	pool_put(cl.pool, torch.v41_rope(lat.view([n, 1, cfg.head_dim]), posf, A.inv_freq).view([n, cfg.head_dim]), e0);
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
		let cap = pool_cap(cl.pool);
		let dv = 'cuda:' + dev;

		m = {
			dev: dev, n: 0,
			pool: pool_new('pool', cap, pool_dim(cl.pool), dv),
			kidx: pool_new('kidx', cap, pool_dim(cl.kidx), dv)
		};

		push(P.mirrors, m);
	}

	if (m.n < cl.n) {
		let k = cl.n - m.n;

		pool_copy(m.pool, cl.pool, m.n, k);
		pool_copy(m.kidx, cl.kidx, m.n, k);
		m.n = cl.n;
	}

	return m;
}

/* ------------------------------------------------------------------ */
/* indexer                                                             */

/* device index of a 'cuda:N' string */
function devidx(dv) {
	return +substr(dv, 5);
}

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

	let lb = torch.from_array(lbl, 'int64').cuda(devidx(dv), true).view([R, 1]);
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
	let rk = `${dv}:${R}:${nb}`;                // row offsets: built once (the eager warm-up of a graph), then reused

	if (ROWOFF[rk] == null) {
		let ro = [];

		for (let r = 0; r < R; r++)
			push(ro, r * nb);

		ROWOFF[rk] = torch.from_array(ro, 'int64').cuda(devidx(dv), true).view([R, 1]);
	}

	let rowoff = ROWOFF[rk];
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
	let q16 = torch.v41_rope(qi, torch.arange(T, 'float32', dv).add(pos0 * 1.0), A.inv_freq);
	let w = torch.matmul(xn.to('float32'), I.w_t).contiguous();
	let kk = pool_view(pc.kidx, S);
	let scale = math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5);
	let K = min(cfg.index_topk, S);
	let SLAB = max(1, int(IDX_SLAB / S));
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

/* The attention of a chunk as a list of steps: the ranks of a tensor-parallel stage run step k of rank 0, step k of
 * rank 1, ... so that their launches interleave (a rank that is issued whole before its peer's first launch makes the
 * peer's all-reduce wait for the host). Returns {steps: [fn ...], out: fn -> y}. */
export function attn_forward_steps(cfg, A, ctx, cache, cl, xn, pos0) {
	let T = xn.shape()[0];
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let eps = cfg.rms_norm_eps;
	let dev = A.dev;
	let ratio = A.ratio;
	let st = {};

	function qkv() {
		prof(null);

		st.qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);

		let q = A.q_b.forward(st.qr).view([T, H, D]);

		prof('a.qproj');

		st.posf = torch.arange(T, 'float32', 'cuda:' + dev).add(pos0 * 1.0);
		st.q16 = torch.v41_rope(q, st.posf, A.inv_freq);

		prof('b.qrope');
	}

	function kvrows() {
		let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
		let kv16 = torch.v41_rope(kv.view([T, 1, D]), st.posf, A.inv_freq).view([T, D]);

		prof('c.kv');

		// window keys: the previous (window - 1) rows from the ring plus this chunk
		st.h = min(pos0, cfg.window - 1);
		st.kvw = (st.h > 0) ? torch.cat([ring_rows(cl, pos0 - st.h, pos0), kv16], 0) : kv16;

		ring_put(cl, pos0, kv16);
	}

	function compress() {
		if (ratio > 0 && A.kv_source)
			compress_chunk(cfg, A, cache, cl, xn, pos0);

		prof('d.compress');
	}

	function select() {
		st.pool = null;
		st.idx = null;

		if (ratio > 0) {
			let src = cfg.src[A.idx];

			if (A.idx_source) {
				st.idx = index_picks(cfg, A, ctx, cache, xn, st.qr, pos0, src);

				prof('e.index');
				ctx.topk = st.idx ? {dev: dev, t: st.idx} : null;
			}
			else if (ctx.topk) {
				let ck = 'd' + dev;

				if (ctx.topk[ck] == null)
					ctx.topk[ck] = (ctx.topk.dev == dev) ? ctx.topk.t : ctx.topk.t.cuda(dev);

				st.idx = ctx.topk[ck];
			}

			if (st.idx) {
				let pc = pool_for(cache, src, dev);
				let S = int((pos0 + T) / ratio);

				st.pool = pool_view(pc.pool, S);
			}
		}
	}

	function attend() {
		st.o = torch.v41_attn(st.q16, st.kvw, pos0 - st.h, st.pool, st.idx, A.sinks, A.inv_freq, pos0, cfg.sm_scale, cfg.window);

		prof('f.attn');
	}

	function project() {
		let y = A.wo_a.forward(st.o.view([T, H * D]));

		prof('g.wo_a');

		st.y = A.wo_b.forward(y);

		prof('h.wo_b');
	}

	// (compress alone: what a later layer reads of a position the prompt skip does not run further, dsv41_tp)
	return {steps: [ qkv, kvrows, compress, select, attend, project ], out: function () {return st.y;}, compress: compress};
}

export function attn_forward(cfg, A, ctx, cache, cl, xn, pos0) {
	let a = attn_forward_steps(cfg, A, ctx, cache, cl, xn, pos0);

	for (let f in a.steps)
		f();

	return a.out();
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
/* The metadata rows go through a page-locked host tensor with non-blocking copies: the host does not wait for each device, and the six
 * segments of a step share one tensor (built once per position). The tensor is rewritten only when the position changes, after
 * the step that read it has been waited for (the host reads the results of every step before it starts the next). */
let POS_PIN = {pos: -1, i: 0, t: []};

export function dec_set_pos(cfg, st, pos) {
	if (POS_PIN.pos != pos || !length(POS_PIN.t)) {
		// (a ring of 8: a step whose copies have not run yet keeps its tensor while the next positions are prepared)
		if (!length(POS_PIN.t))
			for (let k = 0; k < 8; k++)
				push(POS_PIN.t, torch.pinned_empty([8], 'int64'));

		POS_PIN.i = (POS_PIN.i + 1) % 8;
		POS_PIN.t[POS_PIN.i].copy_(torch.from_array([pos, pos % RING, int(pos / 2), 2 * int(pos / 2), int(pos / cfg.candidate_block_size), 0, 0, 0], 'int64'));
		POS_PIN.pos = pos;
	}

	st.meta.copy_(POS_PIN.t[POS_PIN.i], true);
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
		pool_copy(m.pool, cl.pool, j, 1);
		pool_copy(m.kidx, cl.kidx, j, 1);
	}
}

/* ---- the host-RAM prompt-cache store (dsv41_tp cache_export / cache_import) ---- */

/* the first n rows of a pool (fp16: [rows], Q8: [codes, scales]) */
export function pool_rows(P, n) {
	return pool_is_q8(P) ? [ P.q.narrow(0, 0, n), P.s.narrow(0, 0, n) ] : [ P.narrow(0, 0, n) ];
}

/* the rows saved by pool_rows back into the first rows of P */
export function pool_load(P, rows) {
	let n = rows[0].shape()[0];

	if (pool_is_q8(P)) {
		P.q.narrow(0, 0, n).copy_(rows[0]);
		P.s.narrow(0, 0, n).copy_(rows[1]);
	}
	else
		P.narrow(0, 0, n).copy_(rows[0]);
}

/* bytes per pool row */
export function pool_row_bytes(P) {
	return pool_is_q8(P) ? P.q.shape()[1] + P.s.shape()[1] * 2 : P.shape()[1] * 2;
}

/* after the owner's rows were replaced: every mirror of source layer src copies the first n rows again */
export function mirror_reload(cache, src, n) {
	let P = cache.pools[src], cl = P.owner;

	for (let m in P.mirrors) {
		if (n > 0) {
			pool_copy(m.pool, cl.pool, 0, n);
			pool_copy(m.kidx, cl.kidx, 0, n);
		}

		m.n = n;
	}
}

/* one token into the compressor of a kv source layer; writes the pool and index key of its slot */
function comp_decode(cfg, A, cl, xn, st) {
	let C = A.comp, ratio = A.ratio;
	let rd = cfg.rope_dim, eps = cfg.rms_norm_eps;
	let lat, posf, slot;

	if (ratio == 1) {
		lat = torch.rms_norm(cmm16(xn, C.wkv16), C.norm, eps);
		posf = meta_i(st, 0);
		slot = meta_i(st, 0);
	}
	else {
		let x32 = xn.to('float32');
		let kv = cmm32(xn, x32, C.wkv), g = cmm32(xn, x32, C.wgate);
		let m = torch.cat([cl.pg, g], 0).max(0, true);
		let e0 = cl.pg.sub(m).exp(), e1 = g.sub(m).exp();
		let pooled = cl.pk.mul(e0).add(kv.mul(e1)).div(e0.add(e1));

		cl.pk.copy_(kv);
		cl.pg.copy_(g);
		lat = torch.rms_norm(pooled.to('float16').contiguous(), C.norm, eps);
		posf = meta_i(st, 3);
		slot = meta_i(st, 2);
	}

	let I = A.indexer;

	if (I && I.owns_k) {
		let k = torch.rms_norm(torch.matmul(lat, I.wk_t), I.k_norm, eps);

		pool_put(cl.kidx, torch.v41_rope(k.view([1, 1, cfg.index_head_dim]), posf, A.inv_freq)
			.view([1, cfg.index_head_dim]), slot);
	}

	pool_put(cl.pool, torch.v41_rope(lat.view([1, 1, cfg.head_dim]), posf, A.inv_freq)
		.view([1, cfg.head_dim]), slot);
}

/* A tensor a graph body shares with a later one: the producer's tensor on its own device, and on another
 * device a persistent copy that the host refreshes between replays (ctx.links). h: {dev, t | key}. */
function remote(ctx, h, dev, key) {
	key = key || 't';

	if (h.dev == dev)
		return h[key];

	// the destination is persistent (created by the eager warm-up); every run of a body, the capture
	// included, re-registers the producer's tensor of that run as the copy's source
	let k = h.id + ':' + key + ':' + dev;

	if (ctx.dst == null) {
		ctx.dst = {};
		ctx.lk = {};
	}

	if (ctx.dst[k] == null)
		ctx.dst[k] = h[key].cuda(dev);

	ctx.lk[k] = {src: h[key], dst: ctx.dst[k]};

	return ctx.dst[k];
}

/* the picks of the indexer for the one token: (1, k) int32 over the first Scap pool entries */
function index_picks_dec(cfg, A, ctx, cache, xn, qr, st, Scap, src) {
	let I = A.indexer, ratio = A.ratio;
	let dv = 'cuda:' + A.dev;
	let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, rd = cfg.rope_dim;
	let pc = pool_dec(cache, src, A.dev);
	let qi = I.wq_b.forward(qr).view([1, Hi, Di]);
	let q16 = torch.v41_rope(qi, meta_i(st, 0), A.inv_freq);
	let w = torch.matmul(xn.to('float32'), I.w_t).contiguous();
	let sc = torch.idx_score(q16, pool_view(pc.kidx, Scap), w, math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5),
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

			ctx.cand = {dev: A.dev, id: 'c',
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
	let q16 = torch.v41_rope(q, meta_i(st, 0), A.inv_freq);
	let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
	let kv16 = torch.v41_rope(kv.view([1, 1, D]), meta_i(st, 0), A.inv_freq).view([1, D]);

	cl.ring.index_copy_(0, meta_i(st, 1), kv16);

	let pool = null, idx = null;

	if (ratio > 0) {
		let src = cfg.src[A.idx];
		let Scap = caps[ratio];

		if (A.kv_source)
			comp_decode(cfg, A, cl, xn, st);

		if (A.idx_source) {
			idx = index_picks_dec(cfg, A, ctx, cache, xn, qr, st, Scap, src);
			ctx.topk = {dev: dev, t: idx, id: 'p' + A.idx};
		}
		else if (ctx.topk) {
			idx = remote(ctx, ctx.topk, dev);
		}

		if (idx)
			pool = pool_view(pool_dec(cache, src, dev).pool, Scap);
	}

	let o = torch.v41_attn(q16, cl.ring, 0, pool, idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
		meta_i(st, 0), RING);

	return A.wo_b.forward(A.wo_a.forward(o.view([1, H * D])));
}

/* ------------------------------------------------------------------ */
/* multi-row decode (speculative verify)                                */
/*
 * S consecutive tokens at the positions pos .. pos + S - 1 in one graph. The projections, the norms, the
 * hyper-connections and the MoE run batched (the weights are read once); the position-dependent middle of
 * the attention (rope, ring write, compressor, indexer, the attention kernel) runs per row with that row's
 * own device position, in order, so every row sees exactly what a single-token step at its position sees.
 */

/* S rows of device positions on dev: meta (S, 8), rows[s] = {dev, meta: view of row s} (a dec_set_pos state) */
export function new_dec_rows(cfg, dev, S) {
	let meta = torch.zeros([S, 8], 'int64', 'cuda:' + dev);
	let rows = [];

	for (let s = 0; s < S; s++)
		push(rows, {dev: dev, meta: meta.narrow(0, s, 1).view([8])});

	return {dev: dev, S: S, meta: meta, rows: rows};
}

let ROWS_PIN = {};

export function dec_set_rows(cfg, st, pos) {
	let pn = ROWS_PIN[st.S];

	if (!pn)
		pn = ROWS_PIN[st.S] = {pos: -1, i: 0, t: map(slice([0,1,2,3,4,5,6,7]), () => torch.pinned_empty([st.S, 8], 'int64'))};

	if (pn.pos != pos) {
		let a = [];

		for (let s = 0; s < st.S; s++) {
			let p = pos + s;

			push(a, p, p % RING, int(p / 2), 2 * int(p / 2), int(p / cfg.candidate_block_size), (p % 2 == 1) ? int(p / 2) : -1, 0, 0);
		}

		pn.i = (pn.i + 1) % 8;
		pn.t[pn.i].copy_(torch.from_array(a, 'int64').view([st.S, 8]));
		pn.pos = pos;
	}

	st.meta.copy_(pn.t[pn.i], true);
}

/* attention of the S rows of xn (S, D) fp16; sts: the rows' states, ctxs: one indexer context per row */
export function attn_decode_rows(cfg, A, ctxs, cache, cl, xn, sts, caps) {
	let S = xn.shape()[0];
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let eps = cfg.rms_norm_eps;
	let dev = A.dev;
	let ratio = A.ratio;
	let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
	let q = A.q_b.forward(qr).view([S, H, D]);
	let kvn = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
	let os = [];

	for (let s = 0; s < S; s++) {
		let st = sts[s], ctx = ctxs[s];
		let cs = rope_cs_dev(A.inv_freq, meta_f(st, 0));
		let q16 = rope_il(q.narrow(0, s, 1).to('float32'), cs, rd).to('float16').contiguous();
		let kv16 = rope_il(kvn.narrow(0, s, 1).to('float32').view([1, 1, D]), cs, rd).to('float16').view([1, D]).contiguous();
		let xs = xn.narrow(0, s, 1);

		cl.ring.index_copy_(0, meta_i(st, 1), kv16);

		let pool = null, idx = null;

		if (ratio > 0) {
			let src = cfg.src[A.idx];
			let Scap = caps[ratio];

			if (A.kv_source) {
				comp_decode(cfg, A, cl, xs, st);

				// the half-built pair after this row (a rollback to s + 1 rows restarts from it)
				if (ratio == 2) {
					cl.rk.narrow(0, s, 1).copy_(cl.pk);
					cl.rg.narrow(0, s, 1).copy_(cl.pg);
				}
			}

			if (A.idx_source) {
				idx = index_picks_dec(cfg, A, ctx, cache, xs, qr.narrow(0, s, 1), st, Scap, src);
				ctx.topk = {dev: dev, t: idx, id: 'p' + A.idx};
			}
			else if (ctx.topk) {
				idx = remote(ctx, ctx.topk, dev);
			}

			if (idx)
				pool = pool_view(pool_dec(cache, src, dev).pool, Scap);
		}

		push(os, torch.v41_attn(q16, cl.ring, 0, pool, idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
			meta_i(st, 0), RING));
	}

	return A.wo_b.forward(A.wo_a.forward(torch.cat(os, 0).view([S, H * D])));
}

/* cos / sin (S, 1, rd/2, 1) of S positions posf (S, 1) fp32 */
export function rope_cs_rows(inv_freq, posf) {
	let h = inv_freq.shape()[0];
	let S = posf.shape()[0];
	let ang = posf.mul(inv_freq.view([1, h]));

	return [ang.cos().view([S, 1, h, 1]), ang.sin().view([S, 1, h, 1])];
}

function col_i(st, k) {
	return st.meta.narrow(1, k, 1);                  // (S, 1) int64
}

/* the S rows into the compressor of a kv source layer (positions in st.meta), in order */
function comp_decode_rows(cfg, A, cl, xn, st) {
	let C = A.comp, ratio = A.ratio;
	let rd = cfg.rope_dim, eps = cfg.rms_norm_eps;
	let S = xn.shape()[0];
	let dv = 'cuda:' + A.dev;
	let lat, posf, slot;

	if (ratio == 1) {
		lat = torch.rms_norm(cmm16(xn, C.wkv16), C.norm, eps);
		posf = col_i(st, 0);
		slot = col_i(st, 0).reshape([-1]).contiguous();       // a strided view of the meta column otherwise
	}
	else {
		let x32 = xn.to('float32');
		let kv = cmm32(xn, x32, C.wkv), g = cmm32(xn, x32, C.wgate);

		// row s completes the pair (row s - 1, row s) when its position is odd; the partner of row 0 is the
		// half-built pair of the cache
		let pk = torch.cat([cl.pk, kv.narrow(0, 0, S - 1)], 0), pg = torch.cat([cl.pg, g.narrow(0, 0, S - 1)], 0);
		let m = torch.cat([pg.view([S, 1, -1]), g.view([S, 1, -1])], 1).max(1);
		let e0 = pg.sub(m).exp(), e1 = g.sub(m).exp();
		let pooled = pk.mul(e0).add(kv.mul(e1)).div(e0.add(e1));

		cl.rk.narrow(0, 0, S).copy_(kv);
		cl.rg.narrow(0, 0, S).copy_(g);
		cl.pk.copy_(kv.narrow(0, S - 1, 1));
		cl.pg.copy_(g.narrow(0, S - 1, 1));
		lat = torch.rms_norm(pooled.to('float16').contiguous(), C.norm, eps);
		posf = col_i(st, 3);

		// rows that start a pair write their (unfinished) pooled value to the spare slot at the end
		let cs_ = col_i(st, 5).reshape([-1]);

		slot = torch.where(cs_.le(-1), torch.full([1], pool_cap(cl.pool) - 1, 'int64', dv), cs_);
	}

	let I = A.indexer;

	if (I && I.owns_k) {
		let k = torch.rms_norm(torch.matmul(lat, I.wk_t), I.k_norm, eps);

		pool_put(cl.kidx, torch.v41_rope(k.view([S, 1, cfg.index_head_dim]), posf, A.inv_freq)
			.view([S, cfg.index_head_dim]), slot);
	}

	pool_put(cl.pool, torch.v41_rope(lat.view([S, 1, cfg.head_dim]), posf, A.inv_freq)
		.view([S, cfg.head_dim]), slot);
}

/* the picks of the indexer for the S rows: (S, k) int32 over the first Scap pool entries */
function index_picks_dec_rows(cfg, A, ctx, cache, xn, qr, st, Scap, src) {
	let I = A.indexer, ratio = A.ratio;
	let dv = 'cuda:' + A.dev;
	let S = xn.shape()[0];
	let Hi = cfg.index_n_heads, Di = cfg.index_head_dim, rd = cfg.rope_dim;
	let pc = pool_dec(cache, src, A.dev);
	let qi = I.wq_b.forward(qr).view([S, Hi, Di]);
	let q16 = torch.v41_rope(qi, col_i(st, 0), A.inv_freq);
	let w = torch.matmul(xn.to('float32'), I.w_t).contiguous();
	let sc = torch.idx_score(q16, pool_view(pc.kidx, Scap), w, math.pow(Di * 1.0, -0.5) * math.pow(Hi * 1.0, -0.5),
		st.meta.narrow(0, 0, 1).narrow(1, 0, 1), ratio);
	let bs = cfg.candidate_block_size;

	if (Scap > cfg.candidate_blocks * bs && cfg.candidate_source >= 0) {
		let nb = int((Scap + bs - 1) / bs);

		if (A.idx == cfg.candidate_source) {
			let pad = nb * bs - Scap;
			let s2 = (pad > 0) ? torch.cat([sc, torch.full([S, pad], NEGH, 'float16', dv)], 1) : sc;
			let bsc = s2.view([S, nb, bs]).max(2).to('float32');
			let lb = col_i(st, 4);
			let ar = torch.arange(nb, 'int64', dv).view([1, nb]);
			let hit = ar.le(lb).to('int32').mul(lb.le(ar).to('int32'));

			bsc = torch.where(hit.le(0), bsc, torch.full([1], 1.0e30, 'float32', dv));

			let top = bsc.topk(min(cfg.candidate_blocks, nb), -1);

			ctx.cand = {dev: A.dev, id: 'c',
				src: torch.where(top.values.le(NEGH), torch.full([1], -1, 'int64', dv), top.indices)};
		}
		else if (A.idx > cfg.candidate_source && ctx.cand) {
			sc = apply_candidates(cfg, sc, remote(ctx, ctx.cand, A.dev, 'src'), dv);
		}
	}

	let tk = sc.topk(min(cfg.index_topk, Scap), -1);

	return torch.where(tk.values.le(NEGH), torch.full([1], -1, 'int64', dv), tk.indices).to('int32');
}

/* attention of the S rows of xn (S, D) fp16 at the positions of st.meta, all rows at once */
export function attn_decode_batch(cfg, A, ctx, cache, cl, xn, st, caps) {
	let S = xn.shape()[0];
	let H = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
	let eps = cfg.rms_norm_eps;
	let dev = A.dev;
	let ratio = A.ratio;
	let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
	let q = A.q_b.forward(qr).view([S, H, D]);
	let q16 = torch.v41_rope(q, col_i(st, 0), A.inv_freq);
	let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
	let kv16 = torch.v41_rope(kv.view([S, 1, D]), col_i(st, 0), A.inv_freq).view([S, D]);

	cl.ring.index_copy_(0, col_i(st, 1).reshape([-1]), kv16);

	let pool = null, idx = null;

	if (ratio > 0) {
		let src = cfg.src[A.idx];
		let Scap = caps[ratio];

		if (A.kv_source)
			comp_decode_rows(cfg, A, cl, xn, st);

		if (A.idx_source) {
			idx = index_picks_dec_rows(cfg, A, ctx, cache, xn, qr, st, Scap, src);
			ctx.topk = {dev: dev, t: idx, id: 'p' + A.idx};
		}
		else if (ctx.topk) {
			idx = remote(ctx, ctx.topk, dev);
		}

		if (idx)
			pool = pool_view(pool_dec(cache, src, dev).pool, Scap);
	}

	let o = torch.v41_attn(q16, cl.ring, 0, pool, idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
		st.meta.narrow(0, 0, 1).narrow(1, 0, 1), RING);

	return A.wo_b.forward(A.wo_a.forward(o.view([S, H * D])));
}

/* One block on S rows; streams (S, 4, D) fp32 updated in place, pre (S, 4). Returns the next pre. */
export function block_decode_rows(cfg, L, ctxs, cache, cl, streams, pre, st, caps) {
	let eps = cfg.rms_norm_eps;
	let am = hc_site(cfg, L.hc_attn, streams, pre);
	let xn = torch.rms_norm(am.col, L.attn_norm, eps);
	let y = SKIP == 'attn' ? torch.zeros([xn.shape()[0], cfg.hidden_size], 'float16', 'cuda:' + L.dev) :
		ROWLOOP ? attn_decode_rows(cfg, L.attn, ctxs, cache, cl, xn, st.rows, caps) :
		attn_decode_batch(cfg, L.attn, ctxs[0], cache, cl, xn, st, caps);

	glm5.hc_apply(streams, y, am.post, am.comb);

	let fm = hc_site(cfg, L.hc_ffn, streams, am.pre);

	xn = torch.rms_norm(fm.col, L.ffn_norm, eps);
	y = SKIP == 'moe' ? torch.zeros([xn.shape()[0], cfg.hidden_size], 'float32', 'cuda:' + L.dev) : glm5.moe_forward(cfg, L.moe, xn);
	glm5.hc_apply(streams, y, fm.post, fm.comb);

	return fm.pre;
}

/* ------------------------------------------------------------------ */
/* MoE                                                                 */

export function build_mlp8(ws, key, dev) {
	// GGUF: gate and up of the same type run as one launch with the SwiGLU (ggufq.mlp); mlp_forward calls M.fused when it is set
	if (ws.gguf && ws.qtype(key + '.w1.weight') && ws.qtype(key + '.w1.weight') == ws.qtype(key + '.w3.weight') && ws.qtype(key + '.w1.weight') != 'IQ4_NL')
		return ggufq.mlp(ws, key, dev);

	return {
		gate: lin8(ws, key + '.w1', dev),
		up: lin8(ws, key + '.w3', dev),
		down: lin8(ws, key + '.w2', dev)
	};
}

export function build_expert(ws, key, dev) {
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
		Hi: Hi, Kg: 1, Ku: 1, Kd: 1, mcg: false, mul1: true, act: 0, act_limit: cfg.swiglu_limit + 0.0,
		// expert-parallel shards hold experts [e0, e1) of the router's range
		min_expert: (M.e0 != null) ? M.e0 : -1, max_expert: (M.e1 != null) ? M.e1 : -1
	};
}

export function build_moe(ws, cfg, i, dev) {
	let key = `layers.${i}.ffn`;
	let experts = [];

	if (ws.gguf) {
		// GGUF checkpoint: the routed experts are three stacked quantized matrices, run by ggufq.moe_forward
		// (glm5.moe_begin calls M.gq_forward)
		return {
			type: 'moe', dev: dev, experts: experts, router: 'sqrtsp',
			gate_t: mat32(ws, key + '.gate', dev),
			bias: ws.gpu(key + '.gate.bias', dev).to('float32').reshape([1, -1]).contiguous(),
			bias_vl: null, tid2eid: null, ids: null, vlon: false,
			shared: build_mlp8(ws, key + '.shared_experts', dev),
			gq: ggufq.experts(ws, key, dev),
			gq_forward: (cfg, M, y) => ggufq.moe_forward(cfg, M, y, {route: glm5.moe_route, act: glm5.act_mul, mlp: glm5.mlp_forward, side: glm5.par_side, join: glm5.par_join})
		};
	}

	for (let e = 0; e < cfg.n_routed_experts; e++)
		push(experts, build_expert(ws, `${key}.experts.${e}`, dev));

	let M = {
		type: 'moe', dev: dev, experts: experts, router: 'sqrtsp',
		gate_t: mat32(ws, key + '.gate', dev),
		bias: ws.gpu(key + '.gate.bias', dev).to('float32').reshape([1, -1]).contiguous(),
		bias_vl: null, tid2eid: null, ids: null, vlon: false, async_moe: true, fused_prep: true,
		shared: build_mlp8(ws, key + '.shared_experts', dev)
	};

	// glm5.moe_forward then runs the fused decode kernel (<= 8 rows) and the token-sorted fused kernel
	// (prefill chunks) with the per-expert bit widths
	// (mixed integer widths: the per-expert tables; one half-integer width, the 2.50bpw packs: null, glm5.moe_begin
	// builds the uniform-width tables on first use)
	M.coop = getenv('AMP_DSV41_NO_COOP') ? false : (build_coop(cfg, M) || null);

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

	prof(null);

	let am = hc_site(cfg, L.hc_attn, streams, pre);
	let xn = torch.rms_norm(am.col, L.attn_norm, eps);

	prof('0.hc_attn');

	let y = attn_forward(cfg, L.attn, ctx, cache, cl, xn, pos0);

	glm5.hc_apply(streams, y, am.post, am.comb);

	prof('i.hc_apply');

	let fm = hc_site(cfg, L.hc_ffn, streams, am.pre);

	xn = torch.rms_norm(fm.col, L.ffn_norm, eps);

	prof('j.hc_ffn');

	y = glm5.moe_forward(cfg, L.moe, xn);

	prof('k.moe');

	glm5.hc_apply(streams, y, fm.post, fm.comb);

	prof('l.hc_apply2');

	return fm.pre;
}

/* One block on the one token (device position); streams (1, 4, D) fp32 updated in place. */
export function block_decode(cfg, L, ctx, cache, cl, streams, pre, st, caps) {
	let eps = cfg.rms_norm_eps;
	let am = hc_site(cfg, L.hc_attn, streams, pre);
	let xn = torch.rms_norm(am.col, L.attn_norm, eps);
	let y = attn_decode(cfg, L.attn, ctx, cache, cl, xn, st, caps);

	glm5.hc_apply(streams, y, am.post, am.comb);

	let fm = hc_site(cfg, L.hc_ffn, streams, am.pre);

	xn = torch.rms_norm(fm.col, L.ffn_norm, eps);
	y = glm5.moe_forward(cfg, L.moe, xn);
	glm5.hc_apply(streams, y, fm.post, fm.comb);

	return fm.pre;
}

/* ------------------------------------------------------------------ */
/* planner + model                                                     */

/* Bytes of the pool (compressed rows + index keys) of kv source layer src for max_seq positions */
export function pool_bytes(cfg, src, max_seq) {
	let rp = KV_Q8 ? cfg.head_dim + cfg.head_dim / 16 : cfg.head_dim * 2;               // codes + fp16 scale per 32
	let rk = (KV_MODE == 'q8') ? cfg.index_head_dim + cfg.index_head_dim / 16 : cfg.index_head_dim * 2;
	let row = rp + rk;

	return (int(max_seq / cfg.ratios[src]) + 2) * row;
}

/* checkpoint bytes of every layer (the Engram tables excluded) */
export function layer_bytes(ws, cfg) {
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

	return bytes;
}

/* bytes of layers [a, b) on one device: weights, the pools of its source layers and a mirror of every pool
 * its layers read from another device */
export function seg_cost(cfg, bytes, a, b, max_seq) {
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

	return c;
}

/* checkpoint bytes of the drafter's stages (mtp.<b>.*), plus a working set */
export function spark_stage_bytes(ws, cfg) {
	let out = [];

	// (+ a working set per stage: 400 MB for the EXL3 packs, measured at ~100 MB with a GGUF trunk: stage tensors 1.1 - 1.4 GiB in all)
	for (let b = 0; b < cfg.num_mtp_layers; b++)
		push(out, ws.gguf ? 100000000 : 400000000);

	for (let key in ws.list()) {
		if (index(key, 'mtp.') == 0) {
			let b = +substr(key, 4, index(key, '.', 4) - 4);

			if (b >= 0 && b < length(out))
				out[b] += ws.info(key).nbytes;
		}
	}

	return out;
}

/* The devices the drafter's stages were planned on (set by plan_devices) */
export let SPARK_DEVS = null;

/* Contiguous layer ranges for ngpu devices that minimize the fullest device: checkpoint bytes of the
 * layers (the head on the last device, extra_last for its working set) plus the pools it must hold. The
 * embedding table goes to the least loaded device afterwards (embed_device). With `spark` (bytes of each
 * drafter stage) every stage also takes a device of its own choosing: all assignments are tried and the
 * best one is left in SPARK_DEVS. Returns the per-layer device array. */
export function plan_devices(ws, cfg, ngpu, max_seq, extra_first, extra_last, spark, shard, first) {
	let nl = cfg.num_hidden_layers;
	let a0 = first || 0;          // layers below this one live on another host: -1 in the result
	let bytes = layer_bytes(ws, cfg);

	// tensor-parallel stages: every GPU of a stage holds 1 / shard of the layers' weights
	if (shard > 1)
		bytes = map(bytes, (b) => b / shard);
	let ct = [];

	for (let a = 0; a <= nl; a++) {
		push(ct, []);

		for (let b = 0; b <= nl; b++)
			push(ct[a], (b > a) ? seg_cost(cfg, bytes, a, b, max_seq) + (a == a0 ? (extra_first || 0) : 0) +
				(b == nl ? (extra_last || 0) : 0) : 0);
	}

	// the best split for the per-device extra bytes: [fullest device, per-layer device array]
	function dp(extra) {
		// best[d][i]: smallest possible maximum over d devices holding the first i layers
		let best = [], from = [];

		for (let d = 0; d <= ngpu; d++) {
			push(best, []);
			push(from, []);

			for (let i = 0; i <= nl; i++) {
				push(best[d], (d == 0 && i == a0) ? 0 : 1e30);
				push(from[d], -1);
			}
		}

		for (let d = 1; d <= ngpu; d++)
			for (let i = a0 + 1; i <= nl; i++)
				for (let a = a0 + d - 1; a < i; a++) {
					if (best[d - 1][a] >= 1e30)
						continue;

					let m = max(best[d - 1][a], ct[a][i] + extra[d - 1]);

					if (m < best[d][i]) {
						best[d][i] = m;
						from[d][i] = a;
					}
				}

		let devs = [];

		for (let i = 0; i < nl; i++)
			push(devs, i < a0 ? -1 : 0);

		for (let d = ngpu, i = nl; d >= 1; d--) {
			let a = from[d][i];

			for (let j = a; j < i; j++)
				devs[j] = d - 1;

			i = a;
		}

		return [ best[ngpu][nl], devs ];
	}

	let zero = [];

	for (let d = 0; d < ngpu; d++)
		push(zero, 0);

	SPARK_DEVS = null;

	if (!spark || !length(spark))
		return dp(zero)[1];

	let ns = length(spark);
	let bestm = 1e30, bestd = null, bests = null;

	function assign(stage, used) {
		if (stage == ns) {
			let extra = map(zero, () => 0);

			for (let b = 0; b < ns; b++)
				extra[used[b]] += spark[b];

			let r = dp(extra);

			if (r[0] < bestm) {
				bestm = r[0];
				bestd = r[1];
				bests = slice(used);
			}

			return;
		}

		for (let d = 0; d < ngpu; d++) {
			if (index(used, d) >= 0)
				continue;

			push(used, d);
			assign(stage + 1, used);
			pop(used);
		}
	}

	assign(0, []);
	SPARK_DEVS = bests;

	return bestd;
}

/* GGUF checkpoints fill the GPUs to the brim (IQ2_XXS: 128 GiB of weights on 6 x 23.5 GiB) and a layer is 3 - 3.8 GiB, so the contiguous plan
 * of plan_devices leaves one device with a few hundred MiB for graphs, caches and the working set of a prefill chunk. This moves layers (and swaps
 * pairs of layers) off the devices that would keep less than `reserve` bytes free, choosing the change that adds the fewest hops of the
 * activations between devices (80 KB per token and hop). Counted per device: the layers, the pools of the kv sources it owns and a mirror of
 * every pool its layers read from another device (max_seq positions), the head (with the last layer), the drafter's stages (sb bytes, on the
 * devices sd, moved too) and the embedding (on the least loaded device). Returns {devs, embed, load, cap}. */
export function rebalance(cfg, devs, bytes, ngpu, embed_bytes, head_bytes, reserve, max_seq, sb, sd) {
	let nl = length(devs);
	let cap = [];

	for (let d = 0; d < ngpu; d++)
		push(cap, torch.mem_info(d)[3] - reserve);

	devs = slice(devs);
	sd = sd ? slice(sd) : [];
	sb = sb ?? [];

	function loads(dv, ed) {
		let L = [];

		for (let d = 0; d < ngpu; d++)
			push(L, 0);

		L[dv[nl - 1]] += head_bytes;                    // the head lives with the last layer, wherever it is

		if (ed != null)
			L[ed] += embed_bytes;

		for (let b = 0; b < length(sb); b++)           // the drafter's stages (they may share a device)
			L[sd[b]] += sb[b];

		for (let i = 0; i < nl; i++)
			L[dv[i]] += bytes[i];

		let seen = {};

		for (let i = 0; i < nl; i++) {
			let src = cfg.src[i];

			if (index(cfg.kv_sources, i) >= 0 && !seen[`${i}.${dv[i]}`]) {
				seen[`${i}.${dv[i]}`] = true;
				L[dv[i]] += pool_bytes(cfg, i, max_seq);
			}

			if (src >= 0 && src < nl && dv[src] != dv[i] && !seen[`${src}.${dv[i]}`]) {
				seen[`${src}.${dv[i]}`] = true;
				L[dv[i]] += pool_bytes(cfg, src, max_seq);
			}
		}

		return L;
	}

	function over(L) {
		let o = 0;

		for (let d = 0; d < ngpu; d++)
			o = max(o, L[d] - cap[d]);

		return o;
	}

	function hops(dv) {
		let h = 0;

		for (let i = 1; i < nl; i++)
			if (dv[i] != dv[i - 1])
				h++;

		return h;
	}

	// the embedding goes where there is the most room
	let ed = 0, base = loads(devs, null);

	for (let d = 1; d < ngpu; d++)
		if (base[d] - cap[d] < base[ed] - cap[ed])
			ed = d;

	for (let iter = 0; iter < 300; iter++) {
		let L = loads(devs, ed), o0 = over(L);

		if (o0 <= 0)
			break;

		let h0 = hops(devs), best = null, bscore = 1e30;

		// a move or a swap that lowers the worst overload; fewest added hops first, then the lowest overload
		for (let i = 0; i < nl; i++) {
			let di = devs[i];

			if (L[di] - cap[di] < o0 * 0.999)       // only layers of the device(s) at the worst overload
				continue;

			for (let r = 0; r < ngpu; r++) {
				if (r == di)
					continue;

				devs[i] = r;

				let o1 = over(loads(devs, ed)), sc = (hops(devs) - h0) * 1e12 + o1;

				if (o1 < o0 && sc < bscore) {
					bscore = sc;
					best = [ i, -1 ];
				}

				for (let j = 0; j < nl; j++) {
					if (devs[j] != r || j == i)
						continue;

					devs[j] = di;

					let o2 = over(loads(devs, ed)), sc2 = (hops(devs) - h0) * 1e12 + o2;

					if (o2 < o0 && sc2 < bscore) {
						bscore = sc2;
						best = [ i, j ];
					}

					devs[j] = r;
				}

				devs[i] = di;
			}
		}

		// the drafter's stages move freely (no hops): a stage on an overloaded device to the device that leaves the lowest overload
		let sbest = null, sscore = bscore;

		for (let b = 0; b < length(sb); b++) {
			let db = sd[b];

			if (L[db] - cap[db] < o0 * 0.999)
				continue;

			for (let r = 0; r < ngpu; r++) {
				if (r == db)
					continue;

				sd[b] = r;

				let o1 = over(loads(devs, ed));

				sd[b] = db;

				if (o1 < o0 && o1 < sscore) {
					sscore = o1;
					sbest = [ b, r ];
				}
			}
		}

		if (sbest) {
			sd[sbest[0]] = sbest[1];
			continue;
		}

		if (!best)
			break;

		// re-derive the target device of the winning change
		let i = best[0], j = best[1], di = devs[i];

		if (j < 0) {
			let tb = null, tbs = 1e30;

			for (let r = 0; r < ngpu; r++) {
				if (r == di)
					continue;

				devs[i] = r;

				let sc = (hops(devs) - h0) * 1e12 + over(loads(devs, ed));

				if (sc < tbs) {
					tbs = sc;
					tb = r;
				}

				devs[i] = di;
			}

			devs[i] = tb;
		}
		else {
			let r = null, tbs = 1e30;

			// (the swap partner j sits on the device r)
			r = devs[j];
			devs[i] = r;
			devs[j] = di;
		}
	}

	// feasible: now as few device changes as the budgets allow (each is a copy and a graph launch per decode step, ~1 ms): move a layer
	// into the device of a neighbour layer, swap two layers, while it stays within every budget
	if (over(loads(devs, ed)) <= 0) {
		for (let iter = 0; iter < 200; iter++) {
			let h0 = hops(devs), hbest = h0, change = null;

			for (let i = 0; i < nl; i++) {
				let di = devs[i];

				for (let r in [ devs[i > 0 ? i - 1 : 0], devs[i < nl - 1 ? i + 1 : nl - 1] ]) {
					if (r == di)
						continue;

					devs[i] = r;

					let hh = hops(devs);

					if (hh < hbest && over(loads(devs, ed)) <= 0) {
						hbest = hh;
						change = [ i, r, -1 ];
					}

					devs[i] = di;
				}

				for (let j = i + 1; j < nl; j++) {
					let dj = devs[j];

					if (dj == di)
						continue;

					devs[i] = dj;
					devs[j] = di;

					let hh = hops(devs);

					if (hh < hbest && over(loads(devs, ed)) <= 0) {
						hbest = hh;
						change = [ i, dj, j ];
					}

					devs[i] = di;
					devs[j] = dj;
				}
			}

			if (!change)
				break;

			if (change[2] >= 0)
				devs[change[2]] = devs[change[0]];

			devs[change[0]] = change[1];
		}
	}

	let L = loads(devs, ed);

	for (let d = 0; d < ngpu; d++)
		if (L[d] > cap[d])
			warn(sprintf('dsv41: cuda:%d is planned %.2f GiB over its budget (%.2f GiB wanted free after the weights, the pools and the drafter)\n', d,
				(L[d] - cap[d]) / 1073741824.0, reserve / 1073741824.0));

	return {devs: devs, embed: ed, load: L, cap: cap, spark: sd};
}

/* the device with the fewest bytes planned (weights + pools; the head's device counts its head): where the
 * embedding table lives. The lookup is one row per token, wherever it is. */
export function embed_device(ws, cfg, devs, max_seq, head_bytes, extra) {
	let bytes = layer_bytes(ws, cfg);
	let nl = cfg.num_hidden_layers;
	let load = {}, best = devs[0], bl = 1e30;

	for (let i = 0; i < nl; i++) {
		let a = i;

		while (a > 0 && devs[a - 1] == devs[i])
			a--;

		if (a == i)
			load[devs[i]] = seg_cost(cfg, bytes, i, (function () { let e = i; while (e < nl && devs[e] == devs[i]) e++; return e; })(), max_seq);
	}

	load[devs[nl - 1]] += head_bytes || 0;

	for (let d in keys(extra || {}))
		if (load[d] != null)
			load[d] += extra[d];

	for (let d in keys(load))
		if (load[d] < bl) {
			bl = load[d];
			best = +d;
		}

	return best;
}

/* free / total GiB of every device in use */
export function mem_report(devs, tag) {
	let seen = {}, parts = [];

	for (let d in devs) {
		if (seen[d])
			continue;

		seen[d] = true;

		let m = torch.mem_info(d);

		push(parts, sprintf('cuda:%d %.1f/%.1f (cached %.1f)', d, m[2] / 1073741824.0, m[3] / 1073741824.0,
			(m[1] - m[0]) / 1073741824.0));
	}

	warn(`dsv41: ${tag}: free/total GiB ${join(', ', parts)}\n`);
}

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = cfg.num_hidden_layers;
	let devs = opts.devices;

	if (ws.gguf && !getenv('AMP_DSV41_IDX_SLAB'))
		IDX_SLAB = 4 * 1024 * 1024;

	if (devs == null)
		devs = 1;

	if (opts.layers && opts.layers < nl) {           // partial model (tests)
		nl = opts.layers;
		cfg.num_hidden_layers = nl;
	}

	let spark_devs = null, spark_load = {}, spark_bytes = null;

	if (type(devs) != 'array') {
		let sb = (opts.spark && cfg.dspark_block && (ws.has('mtp.0.main_proj.weight') || ws.has('mtp.0.main_proj.trellis'))) ? spark_stage_bytes(ws, cfg) : null;

		devs = plan_devices(ws, cfg, devs, opts.max_seq || 262144, 1000000000, 2000000000, sb);   // first device: Engram + chunk working set
		spark_devs = SPARK_DEVS;
		spark_bytes = sb;

		for (let b = 0; spark_devs && b < length(spark_devs); b++)
			spark_load[spark_devs[b]] = (spark_load[spark_devs[b]] || 0) + sb[b];
	}

	// GGUF checkpoints: balance the devices by moving layers where a contiguous run would leave too little free memory
	let ed_planned = null;

	if (ws.gguf && opts.devices != null && type(opts.devices) != 'array' && !getenv('AMP_DSV41_NO_REBALANCE')) {
		let ngpu = opts.devices;

		// the layers, the drafter's stages (placed freely, spark_devs is replaced by the result) and the embedding together
		let eb = (ws.qtype('embed.weight') && !getenv('AMP_GGUF_EMBED_FP16')) ? ws.info('embed.weight').nbytes : cfg.vocab_size * cfg.hidden_size * 2;
		let rb = rebalance(cfg, devs, layer_bytes(ws, cfg), ngpu, eb, ws.info('head.weight').nbytes, +(getenv('AMP_GGUF_RESERVE_MIB') || 1200) * 1048576.0,
			opts.max_seq || 262144, spark_bytes, spark_devs);

		devs = rb.devs;
		ed_planned = rb.embed;

		if (spark_bytes) {
			spark_devs = rb.spark;
			spark_load = {};

			for (let b = 0; b < length(spark_devs); b++)
				spark_load[spark_devs[b]] = (spark_load[spark_devs[b]] || 0) + spark_bytes[b];
		}
	}

	let first = devs[0], last = devs[nl - 1];
	let ed = (opts.embed_dev != null) ? opts.embed_dev : (ed_planned != null) ? ed_planned :
		embed_device(ws, cfg, devs, opts.max_seq || 262144, 1300000000, spark_load);

	glm5.set_fused_rows(max(256, opts.prefill_rows || 1024));

	let meta = build_engram_meta(model_dir, cfg, ws);

	ws.upload(function (key) {
		if (index(key, 'layers.') == 0) {
			if (index(key, '.engram.embed') > 0)
				return null;                         // streamed from disk

			let idx = +substr(key, 7, index(key, '.', 7) - 7);

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (key == 'embed.weight')
			return ed;

		if (substr(key, 0, 5) == 'head.' || key == 'norm.weight')
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

	let cnt = {};

	for (let d in devs)
		cnt[d] = (cnt[d] || 0) + 1;

	warn(`dsv41: layers per device: ${join(' ', map(keys(cnt), (k) => k + ':' + cnt[k]))}\n`);

	{
		let hops = 0;

		for (let i = 1; i < nl; i++)
			if (devs[i] != devs[i - 1])
				hops++;

		warn(`dsv41: layer plan ${join('', devs)} (${hops} device changes; each is a copy and a graph launch per decode step)\n`);
	}
	mem_report(devs, `layers loaded (embedding on cuda:${ed}, head on cuda:${last}, drafter on ${spark_devs ? 'cuda:' + join(',cuda:', spark_devs) : '-'})`);

	let embed_t;

	if (ws.gguf && ws.qtype('embed.weight') && !getenv('AMP_GGUF_EMBED_FP16')) {
		// a GGUF checkpoint: the table stays quantized on the GPU (0.42 GiB instead of 1.3), the rows of the tokens are dequantized by id
		// (stateless: also inside the decode graphs)
		embed_t = {raw: ws.gpu('embed.weight', ed), qt: ws.qtype('embed.weight'),
			index: function (idt) {return torch.gguf_embed(this.raw, this.qt, idt);}};
	}
	else {
		embed_t = ws.gpu16('embed.weight', ed);

		ws.release('embed.weight', ed);
	}

	let lm_head = ws.gguf ? ggufq.linear(ws, 'head', last) : models.linear(ws, 'head', false, last);

	for (let sfx in [ 'weight', 'trellis', 'suh', 'svh', 'mul1' ])      // lm_head holds its own copy
		ws.release('head.' + sfx, last);

	let model = {
		cfg: cfg, devs: devs, spark_devs: spark_devs, ws: ws, layers: layers, engrams: engrams, meta: meta, embed_tokens: embed_t, embed_dev: ed,
		norm: ws.gpu16('norm.weight', last),
		lm_head: lm_head,

		/* logits (rows, vocab) fp16 of a normed state xn (rows, hidden) fp16 on any device */
		head_full: function (xn) {return this.lm_head.forward(xn.cuda(last));},

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

		/* A chunk in flight: its token ids embedded, the Engram history, the mHC state. */
		chunk_begin: function (token_ids, cache) {
			let cfg = this.cfg;
			let T = length(token_ids);
			let pos0 = cache.length;
			let H = cfg.hidden_size;

			// token history for the engram n-grams (a rollback truncates it)
			if (length(cache.cids) > pos0)
				splice(cache.cids, pos0, length(cache.cids) - pos0);

			while (length(cache.cids) < pos0)             // history that was never fed (warm-up): neutral ids
				push(cache.cids, this.meta.pad);

			let idt = torch.from_array(token_ids, 'int64');
			let cn = this.meta.token_map.index(idt).view([-1]).tolist();
			let hist = [];

			for (let k = 3; k >= 1; k--)
				push(hist, (pos0 - k >= 0) ? cache.cids[pos0 - k] : this.meta.pad);

			for (let c in cn)
				push(hist, c);

			for (let c in cn)
				push(cache.cids, c);

			let h = this.embed_tokens.index(idt.cuda(ed, true));

			if (ed != first)
				h = h.cuda(first);

			h = h.to('float32').reshape([T, 1, H]).expand([T, 4, H]).contiguous();
			let pre = torch.zeros([T, 4], 'float32', 'cuda:' + first);

			pre.narrow(1, 0, 1).fill_(1.0);
			cache.length = pos0 + T;

			let hist_t = torch.from_array(hist, 'int64').view([1, -1]);
			let eg = {};

			// the Engram rows are read from disk by a worker thread while the other chunks are issued
			for (let li = 0; li < length(cfg.engram_layers); li++) {
				let l = cfg.engram_layers[li];

				if (l < nl && this.engrams[l])
					eg[l] = engram_start(cfg, this.meta, this.engrams[l], li, hist_t, T);
			}

			return {T: T, pos0: pos0, end: pos0 + T, dev: first, hist_t: hist_t, eg: eg,
				h: h, pre: pre, ctx: {topk: null, cand: null}};
		},

		/* layers [first, end) of the chunk (all on one device) */
		chunk_layers: function (c, cache, l0, l1) {
			let cfg = this.cfg;

			for (let i = l0; i < l1; i++) {
				let L = this.layers[i];

				// the prompt skip: past the last kv source this chunk is not needed (prefill_pipeline)
				if (c.skip_after != null && i > c.skip_after)
					break;

				if (c.dev != L.dev) {
					c.h = c.h.cuda(L.dev);
					c.pre = c.pre.cuda(L.dev);
					c.dev = L.dev;
				}

				let li = index(cfg.engram_layers, i);

				if (li >= 0) {
					let E = this.engrams[i];

					prof(null);

					let emb = c.eg[i] ? engram_finish(cfg, E, c.eg[i]) : engram_rows(cfg, this.meta, E, li, c.hist_t, c.T);

					c.eg[i] = null;

					prof('m.engram_rows');
					engram_apply(cfg, E, c.h, emb);
					prof('n.engram_apply');
				}

				if (c.taps && index(cfg.dspark_taps, i) >= 0)
					push(c.taps, {dev: c.dev, t: c.h.mean(1)});     // the stream mean at the input of a tapped layer (T, D) fp32

				// the last kv source of a skipped chunk: its compressor alone (the pool rows and index keys later layers read)
				if (c.skip_after != null && i == c.skip_after) {
					let a = hc_site(cfg, L.hc_attn, c.h, c.pre);

					attn_forward_steps(cfg, L.attn, c.ctx, cache, cache.layers[i], torch.rms_norm(a.col, L.attn_norm, cfg.rms_norm_eps),
						c.pos0).compress();
					break;
				}

				let th0 = HOSTPROF ? prof_now() : 0;

				c.pre = block_forward(cfg, L, c.ctx, cache, cache.layers[i], c.h, c.pre, c.pos0);

				if (HOSTPROF)
					push(HOSTPROF, [i, prof_now() - th0]);
			}
		},

		/* the DSpark taps of a finished chunk: (T, 3 * D) fp32 on the first tapped layer's device */
		chunk_taps: function (c) {
			let dev = c.taps[0].dev;

			return torch.cat(map(c.taps, (t) => (t.dev == dev) ? t.t : t.t.cuda(dev, true)), 1);
		},

		/* the head over the last `rows` rows of a finished chunk */
		chunk_head: function (c, rows) {
			let cfg = this.cfg;
			let x = torch.rms_norm(hc_collapse(c.h, c.pre), this.norm, cfg.rms_norm_eps);

			if (rows < c.T)
				x = x.narrow(0, c.T - rows, rows).contiguous();

			return this.lm_head.forward(x);
		},

		/* Run token ids (array) appended to the cache. Returns logits (rows, vocab) fp16 for the last
		 * `rows` positions (default 1), or null for rows == 0. */
		forward_chunk: function (token_ids, cache, rows, on_taps) {
			if (rows == null)
				rows = 1;

			let c = this.chunk_begin(token_ids, cache);

			if (on_taps)
				c.taps = [];

			this.chunk_layers(c, cache, 0, nl);

			if (on_taps)
				on_taps(this.chunk_taps(c), c.pos0);

			return (rows == 0) ? null : this.chunk_head(c, rows);
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

			// AMP_DSV41_GRAPH_BUCKETS=n keeps only the n largest buckets (a context above a bucket's capacity uses the next one)
			// (each bucket keeps ~0.5 GiB of graph memory in total, 0.9 GiB with the 5-row set; a single bucket of max_seq decodes as fast
			// at short and at long contexts on a GGUF model, so that is the default there)
			let nbk = getenv('AMP_DSV41_GRAPH_BUCKETS') ?? (model.ws.gguf ? 1 : null);

			if (nbk != null)
				caps_list = slice(caps_list, -(+nbk));

			let gpool = {};
			let buckets = [];

			for (let c1 in caps_list) {
				let caps = [ 0, c1, int((c1 + 1) / 2) ];
				let ctx = {topk: null, cand: null, dst: {}, lk: {}};
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

					it.taps = [];

					for (let i = it.first; i < it.end; i++) {
						let li = index(cfg.engram_layers, i);

						if (li >= 0) {
							let E = model.engrams[i];

							engram_apply(cfg, E, h, E.emb_buf);
						}

						if (index(cfg.dspark_taps, i) >= 0)
							push(it.taps, {dev: it.dev, t: h.mean(1)});

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
					glm5.set_par(!getenv('AMP_DSV41_NO_PAR'));         // the shared expert on a forked stream
					body(it);
					glm5.set_par(false);
					it.graph.capture_end();

					if (!gpool[it.dev])
						gpool[it.dev] = it.graph.pool();
				}

				torch.synchronize();
				push(buckets, {c1: c1, items: items, ctx: ctx});
			}

			mem_report(model.devs, 'graphs captured (1 row)');

			// multi-row steps (speculative verify): S rows per replay, per S its own set of graphs
			let rowsets = {};
			let tap_layers = cfg.dspark_taps;

			// Which verify sizes get a graph (each set costs ~0.3 GiB over six GPUs): all that the engine asks for, or with
			// AMP_DSV41_ROW_SIZES=a,b,.. only those (a GGUF model: 3 and the largest); a size without a graph runs the next larger one and
			// takes the extra rows back with rollback() - they are rejected rows
			let want = getenv('AMP_DSV41_ROW_SIZES') ? map(split(getenv('AMP_DSV41_ROW_SIZES'), ','), (x) => +x) :
				model.ws.gguf ? [ 3, 5 ] : null;
			let sizes_all = sizes;

			if (want) {
				let mx = max(...sizes);

				sizes = filter(sizes, (S) => S < 2 || index(want, S) >= 0 || S == mx);
			}

			for (let S in sizes) {
				if (S < 2)
					continue;

				if (S > MAX_ROWS)
					die(`dsv41: at most ${MAX_ROWS} rows per step`);

				let stsS = {};

				for (let sg in segs)
					stsS[sg.dev] = new_dec_rows(cfg, sg.dev, S);

				let bks = [];

				for (let c1 in caps_list) {
					let caps = [ 0, c1, int((c1 + 1) / 2) ];
					let ctxs = [];
					let items = [];

					for (let r = 0; r < S; r++)
						push(ctxs, {topk: null, cand: null, dst: {}, lk: {}});

					for (let sg in segs) {
						let dv = 'cuda:' + sg.dev;
						let it = {first: sg.first, end: sg.end, dev: sg.dev,
							h_in: torch.zeros([S, 4, H], 'float32', dv), pre_in: torch.zeros([S, 4], 'float32', dv),
							logits: null, last: (sg.end == nl), taps: []};

						it.pre_in.narrow(1, 0, 1).fill_(1.0);
						push(items, it);
					}

					function body_rows(it) {
						let h = it.h_in, pre = it.pre_in;

						it.taps = [];

						for (let i = it.first; i < it.end; i++) {
							let li = index(cfg.engram_layers, i);

							if (li >= 0) {
								let E = model.engrams[i];

								engram_apply(cfg, E, h, E.emb_bufs[S]);
							}

							if (index(tap_layers, i) >= 0)
								push(it.taps, {dev: it.dev, t: h.mean(1)});

							pre = block_decode_rows(cfg, model.layers[i], ctxs, cache, cache.layers[i], h, pre,
								stsS[it.dev], caps);
						}

						it.pre_out = pre;

						if (it.last) {
							let x = torch.rms_norm(hc_collapse(h, pre), model.norm, cfg.rms_norm_eps);

							it.logits = model.lm_head.forward(x);
							it.am = it.logits.argmax(-1);
						}

						it.h_out = h;
					}

					for (let E in values(model.engrams))
						if (E.emb_bufs[S] == null)
							E.emb_bufs[S] = torch.zeros([S, E.emb_buf.shape()[1]], 'float16', 'cuda:' + E.dev);

					for (let sg in segs)
						dec_set_rows(cfg, stsS[sg.dev], 0);

					for (let it in items) {
						body_rows(it);
						torch.synchronize();
					}

					for (let it in items) {
						it.graph = torch.graph();
						it.graph.capture_begin(it.dev, null, gpool[it.dev]);
						glm5.set_par(!getenv('AMP_DSV41_NO_PAR'));
						body_rows(it);
						glm5.set_par(false);
						it.graph.capture_end();

						if (!gpool[it.dev])
							gpool[it.dev] = it.graph.pool();
					}

					torch.synchronize();
					push(bks, {c1: c1, items: items, ctxs: ctxs});
				}

				rowsets[S] = {buckets: bks, sts: stsS};
				mem_report(model.devs, `graphs captured (${S} rows)`);
			}

			// the warm-up and the captures wrote position 0 of every cache: start from empty again
			model.aux_reset(cache);
			mem_report(model.devs, 'graphs captured');

			function run_ids(tid, pos) {
				let b = null;

				for (let x in buckets)
					if (x.c1 > pos && !b)
						b = x;

				if (!b)
					die(`dsv41: position ${pos} is past the context of ${max_seq}`);

				if (length(cache.cids) > pos)
					splice(cache.cids, pos, length(cache.cids) - pos);

				while (length(cache.cids) < pos)
					push(cache.cids, model.meta.pad);

				for (let sg in segs)
					dec_set_pos(cfg, sts[sg.dev], pos);

				// Engram rows of the token: hashed on the host, gathered from the table, into the layers' buffers
				let tpp = PROF ? prof_now() : 0;
				let cn = model.meta.token_map.index(torch.from_array([ tid ], 'int64')).view([-1]).tolist()[0];
				let hist = [];

				for (let k = 3; k >= 1; k--)
					push(hist, (pos - k >= 0) ? cache.cids[pos - k] : model.meta.pad);

				push(hist, cn);

				let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

				// every layer's gathers are started now; a layer's rows are waited for just before the segment that holds it is
				// replayed, so the later layers' reads overlap the earlier segments' compute
				let ests = [];

				for (let li = 0; li < length(cfg.engram_layers); li++) {
					let l = cfg.engram_layers[li];

					if (l < nl)
						push(ests, [ l, model.engrams[l], engram_start(cfg, model.meta, model.engrams[l], li, hist_t, 1) ]);
				}

				push(cache.cids, cn);

				if (PROF)
					PROFILE['dec.engram_host'] = (PROFILE['dec.engram_host'] || 0) + prof_now() - tpp;

				let h = model.embed_tokens.index(torch.from_array([ tid ], 'int64').cuda(ed, true));

				if (ed != first)
					h = h.cuda(first);

				h = h.to('float32').reshape([1, 1, H]).expand([1, 4, H]).contiguous();
				let pre = null, logits = null, taps = [];

				let tp0 = PROF ? prof_now() : 0;

				if (PROF) {
					torch.synchronize();
					PROFILE['dec.host_pre'] = (PROFILE['dec.host_pre'] || 0) + prof_now() - tp0;
				}

				for (let it in b.items) {
					let ts0 = PROF ? prof_now() : 0;

					for (let es in ests) {
						if (es[2] && es[0] >= it.first && es[0] < it.end) {
							es[1].emb_buf.copy_(engram_finish(cfg, es[1], es[2]));
							es[2] = null;
						}
					}

					it.h_in.copy_(h);

					if (pre)
						it.pre_in.copy_(pre);

					it.graph.replay();

					if (PROF) {
						torch.synchronize();
						PROFILE[`dec.seg${it.dev}`] = (PROFILE[`dec.seg${it.dev}`] || 0) + prof_now() - ts0;
					}

					h = it.h_out;
					pre = it.pre_out;

					for (let k in keys(b.ctx.lk))
						b.ctx.lk[k].dst.copy_(b.ctx.lk[k].src);

					for (let i = it.first; i < it.end; i++) {
						let A = model.layers[i].attn;

						if (A.kv_source)
							mirror_entry(cache, i, int(pos / A.ratio));
					}

					push(taps, ...it.taps);

					if (it.last)
						logits = it.logits;
				}

				for (let i = 0; i < nl; i++) {
					let A = model.layers[i].attn;

					if (A.kv_source)
						set_pool_n(cache, i, int((pos + 1) / A.ratio));        // mirror_entry copied the new entry
				}

				cache.length = pos + 1;
				model.last_rows = {pos: pos, S: 1};

				let dt = length(taps) ? taps[0].dev : 0;

				return {logits: logits, hidden: null,
					taps: length(taps) ? torch.cat(map(taps, (t) => (t.dev == dt) ? t.t : t.t.cuda(dt)), 1) : null};
			}

			/* S = length(ids) rows at pos .. pos + S - 1 through one replay chain: {logits (S, vocab), am (the
			 * argmax of every row), taps (S, 3 * hidden) fp32}. cache.length becomes pos + S; rollback()
			 * takes back the rows a speculation did not keep. */
			const ROWPROF = getenv('AMP_DSV41_ROWPROF') ? {n: 0, t: {}} : null;

			function rowprof(name, t0) {
				ROWPROF.t[name] = (ROWPROF.t[name] || 0) + prof_now() - t0;

				if (name == 'embed' && ++ROWPROF.n % 100 == 0)
					warn('run_rows prelude (cumulative ms from entry, per call): ' + join(', ', map(keys(ROWPROF.t), (k) => sprintf('%s %.2f', k, ROWPROF.t[k] / ROWPROF.n))) + '\n');
			}

			function run_rows(ids, pos) {
				let S = length(ids);
				let rs = rowsets[S];

				if (!rs)
					die(`dsv41: no graph for ${S} rows`);

				let b = null;

				for (let x in rs.buckets)
					if (x.c1 > pos + S - 1 && !b)
						b = x;

				if (!b)
					die(`dsv41: position ${pos + S - 1} is past the context of ${max_seq}`);

				if (length(cache.cids) > pos)
					splice(cache.cids, pos, length(cache.cids) - pos);

				while (length(cache.cids) < pos)
					push(cache.cids, model.meta.pad);

				let rp0 = ROWPROF ? prof_now() : 0;

				for (let sg in segs)
					dec_set_rows(cfg, rs.sts[sg.dev], pos);

				if (ROWPROF)
					rowprof('set', rp0);

				let idt = torch.from_array(ids, 'int64');
				let cn = model.meta.token_map.index(idt).view([-1]).tolist();
				let hist = [];

				for (let k = 3; k >= 1; k--)
					push(hist, (pos - k >= 0) ? cache.cids[pos - k] : model.meta.pad);

				for (let c in cn) {
					push(hist, c);
					push(cache.cids, c);
				}

				let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

				if (ROWPROF)
					rowprof('tok', rp0);

				// every Engram layer's gathers start now; a layer's rows are waited for just before the segment that holds it is replayed
				let ests = [];

				for (let li = 0; li < length(cfg.engram_layers); li++) {
					let l = cfg.engram_layers[li];

					if (l < nl)
						push(ests, [ l, model.engrams[l], engram_start(cfg, model.meta, model.engrams[l], li, hist_t, S) ]);
				}

				if (ROWPROF)
					rowprof('engram start', rp0);

				let h = model.embed_tokens.index(idt.cuda(ed, true));

				if (ed != first)
					h = h.cuda(first);

				h = h.to('float32').reshape([S, 1, H]).expand([S, 4, H]).contiguous();

				if (ROWPROF)
					rowprof('embed', rp0);

				let pre = null, logits = null, am = null, taps = [];

				for (let it in b.items) {
					for (let es in ests) {
						if (es[2] && es[0] >= it.first && es[0] < it.end) {
							es[1].emb_bufs[S].copy_(engram_finish(cfg, es[1], es[2]));
							es[2] = null;
						}
					}

					if (ROWPROF && !it.first)
						rowprof('engram wait (seg 0)', rp0);

					it.h_in.copy_(h);

					if (pre)
						it.pre_in.copy_(pre);

					it.graph.replay();
					h = it.h_out;
					pre = it.pre_out;

					for (let ctx in b.ctxs)
						for (let k in keys(ctx.lk))
							ctx.lk[k].dst.copy_(ctx.lk[k].src);

					for (let i = it.first; i < it.end; i++) {
						let A = model.layers[i].attn;

						if (A.kv_source)
							for (let j = int(pos / A.ratio); j <= int((pos + S - 1) / A.ratio); j++)
								mirror_entry(cache, i, j);
					}

					push(taps, ...it.taps);

					if (it.last) {
						logits = it.logits;
						am = it.am;
					}
				}

				for (let i = 0; i < nl; i++) {
					let A = model.layers[i].attn;

					if (A.kv_source)
						set_pool_n(cache, i, int((pos + S) / A.ratio));
				}

				cache.length = pos + S;
				model.last_rows = {pos: pos, S: S};

				let dt = length(taps) ? taps[0].dev : 0;

				return {logits: logits, am: am,
					taps: length(taps) ? torch.cat(map(taps, (t) => (t.dev == dt) ? t.t : t.t.cuda(dt)), 1) : null};
			}

			let captured = sort(filter(map(keys(rowsets), (k) => +k), (k) => k > 1));

			/* S rows through the graph of S rows, or of the next larger captured size with the surplus rows taken back */
			function run_rows_padded(ids, pos) {
				let S = length(ids);

				if (rowsets[S])
					return run_rows(ids, pos);

				let S2 = null;

				for (let k in captured)
					if (k > S && S2 == null)
						S2 = k;

				if (S2 == null)
					die(`dsv41: no graph for ${S} rows`);

				let ids2 = slice(ids);

				while (length(ids2) < S2)
					push(ids2, ids[S - 1]);

				let r = run_rows(ids2, pos);

				model.rollback(cache, pos + S);

				return {logits: r.logits.narrow(0, 0, S), am: r.am.narrow(0, 0, S), taps: r.taps ? r.taps.narrow(0, 0, S) : null};
			}

			return {
				has: function (S) {return S == 1 || (length(captured) > 0 && S >= 2 && S <= captured[length(captured) - 1]);},

				rows: run_rows_padded,

				run: function (ids, pos) {
					return run_ids(ids[0], pos);
				},

				run_dev: function (ids_dev, pos) {
					return run_ids(ids_dev.view([1]).tolist()[0], pos);
				}
			};
		},

		/* Take back the rows of the last multi-row step beyond the length L (p0 < L <= p0 + S): the pool entry
		 * counts and the half-built pair of the ratio-2 compressors follow the kept rows. */
		rollback: function (cache, L) {
			let lr = this.last_rows;

			if (!lr || L < lr.pos + 1 || L > lr.pos + lr.S)
				die(`dsv41: rollback to ${L} is outside the last step`);

			for (let i = 0; i < nl; i++) {
				let A = this.layers[i].attn;

				if (!A.kv_source)
					continue;

				set_pool_n(cache, i, int(L / A.ratio));

				if (A.ratio == 2) {
					let cl = cache.layers[i];

					cl.pk.copy_(cl.rk.narrow(0, L - 1 - lr.pos, 1));
					cl.pg.copy_(cl.rg.narrow(0, L - 1 - lr.pos, 1));
				}
			}

			cache.length = L;
			this.last_rows = {pos: lr.pos, S: L - lr.pos};
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
					set_pool_n(cache, i, v.n);

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
			for (let i = 0; i < length(cache.layers); i++)
				if (cache.layers[i].pool)
					set_pool_n(cache, i, 0);

			cache.cids = [];
		},

		/* ---- the host-RAM prompt-cache store (engine.uc store_*, --cache-ram): the pool and index-key rows of the kv source
		 * layers, the aux state (pool counts, half-built pairs, window rows: the window ring needs nothing beyond those), the
		 * Engram id history and the drafter's ring. The mirrors of the pools on the other GPUs are reloaded whole after an
		 * import (the decode graphs read them as they are). ---- */
		cache_bytes: function (cache, n) {
			let cfg = this.cfg, W = cfg.window - 1;
			let b = 8 * n;

			for (let i = 0; i < nl; i++) {
				let cl = cache.layers[i];

				b += min(n, W) * cfg.head_dim * 2 + 2 * cfg.head_dim * 4;

				if (cl.pool)
					b += (int(n / this.layers[i].attn.ratio) + 1) * (pool_row_bytes(cl.pool) + pool_row_bytes(cl.kidx));
			}

			if (this.dspark && this.dspark.ring_bytes)
				b += this.dspark.ring_bytes(n);

			return b;
		},

		cache_export: function (cache, n, put, snap_aux, dst) {
			if (dst)
				die('dsv41: cache_export: no device slots');

			if (cache.length != n)
				die(`dsv41: cache_export: ${n} tokens asked, the cache holds ${cache.length}`);

			let aux_map = (aux) => map(aux, (e) => e ? {n: e.n, pk: e.pk ? put(e.pk) : null, pg: e.pg ? put(e.pg) : null,
				kw: e.kw ? put(e.kw) : null} : null);

			return {
				pools: map(cache.layers, (cl) => cl.pool ? {pool: map(pool_rows(cl.pool, cl.n), put), kidx: map(pool_rows(cl.kidx, cl.n), put)} : null),
				aux: aux_map(this.aux_get(cache)), snap: snap_aux ? aux_map(snap_aux) : null,
				cids: slice(cache.cids, 0, n),
				drafter: (this.dspark && this.dspark.ring_export) ? this.dspark.ring_export(n, put) : null
			};
		},

		// start == the entry's length (live) or its snapshot's (the snapshot's aux state; the pools hold its rows too)
		cache_import: function (cache, em, start, live) {
			for (let i = 0; i < nl; i++) {
				let P = em.pools[i], cl = cache.layers[i];

				if (P) {
					pool_load(cl.pool, P.pool);
					pool_load(cl.kidx, P.kidx);
				}
			}

			cache.length = start;

			let dev_aux = (aux) => map(aux, (e, i) => e ? {n: e.n, pk: e.pk ? e.pk.cuda(cache.layers[i].dev) : null,
				pg: e.pg ? e.pg.cuda(cache.layers[i].dev) : null, kw: e.kw ? e.kw.cuda(cache.layers[i].dev) : null} : null);

			this.aux_set(cache, dev_aux(live ? em.aux : em.snap));

			for (let i = 0; i < nl; i++)
				if (cache.layers[i].pool)
					mirror_reload(cache, i, cache.layers[i].n);

			cache.cids = slice(em.cids, 0, start);
			this.last_rows = null;

			if (em.drafter && this.dspark && this.dspark.ring_import)
				this.dspark.ring_import(em.drafter, start);

			return {snap: em.snap ? dev_aux(em.snap) : null};
		},

		/* Engine interface (as dsv4.prefill_pipeline): chunked prefill of ids at cache.length as a wavefront over
		 * the device stages: one stage per chunk per round, so every GPU works on a different chunk (a layer
		 * sees its chunks in order, which is all the window ring, the pools and the pairs need). The host issues
		 * the work of all stages in a round before it can block on a full launch queue. opts: chunk, cuts (extra
		 * chunk boundaries), on_aux(chunk_end) when a chunk left the last stage. A snapshot boundary
		 * (cuts[0]) holds the next chunk back until the pipeline has drained, so the layer state of that
		 * moment is the state at the boundary. */
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

			// The prompt skip (docs/DSV41.md): the layers after the last kv source need a position only within
			// (layers after it + 1) windows of the prompt's end, a chunk that ends before runs up to the source's compressor.
			let S_last = max(...filter(model.cfg.kv_sources, (k) => k < nl));
			let cone_from = start + total - (nl - S_last + 1) * model.cfg.window;
			let skip_on = (!opts.warmup && cone_from > start && (opts.prompt_skip ?? PROMPT_SKIP)) ? true : false;

			for (let s = 0; s < ns; s++)
				push(active, null);

			if (!NO_ENGRAM)
				engram_prefetch(model.cfg, model.meta, model.engrams, cache, ids, start);

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				if (snap_at && prev_end == snap_at) {
					for (let c in active)
						if (c)
							return;
				}

				let end = bounds[next++];
				let last = (end == start + total);
				let c = model.chunk_begin(slice(ids, prev_end - start, end - start), cache);

				if (opts.on_taps)
					c.taps = [];

				c.stage = 0;
				c.last = last;
				c.skip_after = (skip_on && end <= cone_from) ? S_last : null;
				prev_end = end;
				active[0] = c;
			}

			function round() {
				for (let s = ns - 1; s >= 0; s--) {
					let c = active[s];

					if (!c)
						continue;

					model.chunk_layers(c, cache, stages[s].first, stages[s].end);

					if (s == ns - 1 || (c.skip_after != null && stages[s].end > c.skip_after)) {
						// (a skipped chunk ends in the stage of the last kv source and has no taps)
						if (opts.on_taps && c.skip_after == null)
							opts.on_taps(model.chunk_taps(c), c.pos0);

						if (c.last)
							result = model.chunk_head(c, 1);

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
