// ucode/qwen4.uc — Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration).
//
// Plan and architecture notes: docs/QWEN4.md. A Qwen3.5-lineage hybrid
// (GatedDeltaNet + gated GQA attention, 48 layers, hidden 2560) with
//
//   - 512-expert MoE on every layer (softmax router, shared expert with a
//     sigmoid gate; the expert kernels and loaders are glm5.uc's),
//   - "gated-residual" hyper-connections: the residual is 4 fp32 streams;
//     each sublayer reads an elementwise gated MEAN of the streams and its
//     output is injected back per stream (no input/post layernorms),
//   - a sigmoid-gated GDN output norm,
//   - a PLE layer before block 1 that adds hashed n-gram embeddings (rows of
//     a 39 GB table that stays on the host, gathered from the mmap'd file)
//     to every stream,
//   - QSA sparse attention on the full-attention layers: exact (= dense)
//     while the context fits 4 * block_topk + 3 tokens.
//
// Numerics follow exllamav3's qwen4_exp.py; the composite gated-residual mix
// below is the cuBLAS-path reference (GatedResidual._mix_ref).

import * as models from './models.uc';
import * as weights from './weights.uc';
import * as qwen35 from './qwen35.uc';
import * as glm5 from './glm5.uc';
import * as ggufq from './ggufq.uc';
import * as qprof from './qwen4_prof.uc';
import * as math from 'math';
import {readfile} from 'fs';

export const LP = 'model.language_model.layers.';
const NO_FUSED_MIX = (getenv('AMP_QWEN4_NO_FUSED_MIX') == '1');
const QSA_REF = (getenv('AMP_QWEN4_QSA_REF') == '1');       // composite sparse attention instead of the kernel
const MTP_TAP = (getenv('AMP_QWEN4_MTP_TAP') != '0');     // MTP input: per-stream (reference default) or averaged

/* GGUF expert placement, set by the server from its options (configure(); the AMP_QWEN4_* variables are the fallback for
 * tests and experiments): {resident: fraction in VRAM, mtp_resident, profile: file ranking the experts, count: count the
 * router's picks, save: file the counts are written to (null: not written)} */
let CONF = {};
let PROF = null;                 // the profile read once: false when there is none
let PROFILER = null;             // the live routing counters (qwen4_prof.uc)

export function configure(o) {
	CONF = o || {};
	PROF = null;
	PROFILER = null;
}

let LAYER_DEV = [];              // device of every layer (set by load), for per-GPU resident fractions

/* fraction of the routed experts kept in VRAM; the rest stay pinned on the host (GGUF only). CONF.resident is one
 * fraction or one per GPU (the last GPU also carries lm_head and the draft head, so it usually takes fewer) */
function expert_resident_fraction(layer) {
	let v = (layer == null) ? (CONF.mtp_resident ?? getenv('AMP_QWEN4_MTP_EXPERT_RESIDENT')) :
		(CONF.resident ?? getenv('AMP_QWEN4_EXPERT_RESIDENT'));

	// the draft head defaults to the backbone's fraction
	if (v == null && layer == null)
		v = CONF.resident ?? getenv('AMP_QWEN4_EXPERT_RESIDENT');

	if (type(v) == 'array') {
		let d = (layer == null) ? length(v) - 1 : LAYER_DEV[layer];

		v = v[min(d ?? length(v) - 1, length(v) - 1)];
	}

	return (v == null) ? 1.0 : +v;
}

/* Placement numbers of a loaded GGUF model for the fit survey (fit.uc): per GPU the layer count, the bytes of all routed
 * experts of its layers and the bytes of those resident now (the backbone's experts, not the draft head's). */
export function fit_info(model) {
	let out = [];

	for (let i = 0; i < model.nl; i++) {
		let L = model.layers[i], M = L.moe;

		if (!M || !M.gq)
			continue;

		let E = M.n_experts_total, R = M.gq.w1.shape()[0];
		let per = M.gq.w1.shape()[1] * M.gq.w1.shape()[2] + M.gq.w3.shape()[1] * M.gq.w3.shape()[2] +
			M.gq.w2.shape()[1] * M.gq.w2.shape()[2];
		let o = out[L.dev];

		if (!o)
			o = out[L.dev] = {layers: 0, all: 0, resident: 0};

		o.layers++;
		o.all += E * per;
		o.resident += R * per;
	}

	return out;
}

/* true when some layer keeps routed experts on the host */
function any_cold() {
	let v = CONF.resident ?? getenv('AMP_QWEN4_EXPERT_RESIDENT');

	if (type(v) == 'array')
		return length(filter(v, (x) => +x < 1.0)) > 0;

	return v != null && +v < 1.0;
}

/* the profile (qwen4_prof.uc): the experts of `layer`, most picked first, or null */
function profile_order(layer) {
	if (PROF == null) {
		let path = CONF.profile ?? getenv('AMP_QWEN4_EXPERT_PROFILE');

		PROF = (path ? qprof.read(path) : null) ?? false;

		if (path)
			warn(PROF ? `qwen4: expert profile ${path} (${PROF.tokens ? PROF.tokens + ' tokens' : 'ranking only'})\n` :
				`qwen4: expert profile ${path} not usable, experts placed by index\n`);
	}

	return PROF ? PROF.order[layer] : null;
}

/* ------------------------------------------------------------------ */
/* config                                                              */

export function load_config(model_dir) {
	let cfg = qwen35.load_config(model_dir);
	let raw = json(readfile(model_dir + '/config.json'));
	let t = raw.text_config;

	cfg.model_dir = model_dir;
	cfg.hc_mult = t.hc_count;
	cfg.hc_rank = t.hc_lowrank;
	cfg.n_routed_experts = t.num_experts;
	cfg.num_experts_per_tok = t.num_experts_per_tok;
	cfg.swiglu_limit = 0.0;
	cfg.indexer_budget = t.indexer_budget;
	cfg.indexer_compress_ratio = t.indexer_compress_ratio;
	cfg.indexer_heads = t.indexer_n_heads;
	cfg.indexer_head_dim = t.indexer_head_dim;
	cfg.output_gate_type = t.output_gate_type;
	cfg.ple_layer_ids = t.ple_layer_ids;
	cfg.ple_embed_dim = t.ple_embed_dim;
	cfg.ple_kernel = t.ple_conv_kernel_size;
	cfg.ngram_size = t.ngram_size;
	cfg.heads_per_ngram = t.heads_per_ngram;
	cfg.ngram_parts = t.split_ngram_parts;
	cfg.sparse_threshold = 4 * (t.indexer_budget / t.indexer_compress_ratio) + 3;

	if (cfg.output_gate_type != 'sigmoid')
		die('qwen4: only the sigmoid GDN output gate is implemented');

	return cfg;
}

/* ------------------------------------------------------------------ */
/* gated-residual hyper-connection sites                               */

/* One site (key = layer.attn_hyper_connection / mlp_hyper_connection /
 * hyper_connection_mixer). inject: the site form with the block_inject
 * projection (post gates); the final mixer has none. */
export function build_gr(ws, cfg, key, dev, inject) {
	let H = cfg.hc_mult, D = cfg.hidden_size;
	let w = ws.gpu(key + '.hc_norm.weight', dev).to('float32');

	// the GGUF conversion stores the hc norm already with the +1 constant
	if (!ws.gguf)
		w = w.add(1.0);

	w = w.view([H, D]).contiguous();
	let down = ws.gpu16(key + '.input_mix_weight_down.weight', dev);   // (rank, H*D)
	let up = ws.gpu16(key + '.input_mix_weight_up.weight', dev);       // (H*D, rank)
	let proj = inject ?
		torch.cat([ down, ws.gpu16(key + '.block_inject_weight.weight', dev) ], 0) : down;

	// fused decode kernel tables (torch.gr_mix): cat(down, inject) * w folded,
	// the up projection repacked (H, D/4, rank, 4)
	let wf = w.view([H * D]);

	return {
		dev: dev, inject: inject, rank: cfg.hc_rank,
		w: w,
		proj_t: proj.t().contiguous(),        // (H*D, rank [+ H])
		up_t: up.t().contiguous(),            // (rank, H*D)
		fn: proj.to('float32').mul(wf).to('float16').contiguous(),
		upx: up.view([H, int(D / 4), 4, cfg.hc_rank]).transpose(2, 3).contiguous(),
		w_h: wf.to('float16').contiguous()
	};
}

/* streams (R, 4, D) fp32 contiguous -> [post (R, 4) fp32 | null, mixed (R, D)
 * fp16]. */
export function gr_mix(cfg, G, streams) {
	let R = streams.shape()[0];
	let H = cfg.hc_mult, D = cfg.hidden_size;

	if (R <= 8 && !NO_FUSED_MIX)
		return torch.gr_mix(streams, G.fn, G.upx, G.w_h, cfg.rms_norm_eps, G.inject);

	let ms = streams.mul(streams).mean(-1, true).add(cfg.rms_norm_eps).rsqrt();
	let normed = streams.mul(ms).mul(G.w).to('float16');            // (R, H, D)
	let dm = torch.matmul(normed.view([R, H * D]), G.proj_t);       // (R, rank [+ H]) fp16
	let t = dm.narrow(1, 0, G.rank).div(H * 1.0).silu();
	let g = torch.matmul(t.contiguous(), G.up_t);                   // (R, H*D) fp16
	let mixed = g.to('float32').sigmoid().view([R, H, D])
		.mul(normed.to('float32')).mean(1).to('float16').contiguous();
	let post = null;

	if (G.inject)
		post = dm.narrow(1, G.rank, H).to('float32').div(H * 1.0).sigmoid().mul(2.0)
			.contiguous();

	return [ post, mixed ];
}

/* ------------------------------------------------------------------ */
/* PLE layer + n-gram embedding                                        */

/* Weights + host-side table handles of the PLE layer at block index idx.
 * The table stays in the mmap'd safetensors file (page cache = RAM, else
 * the SSD): a forward gathers only the unique rows its tokens hash to. */
export function build_ple(ws, cfg, idx, dev) {
	let key = `${LP}${idx}.ple`;
	let H = cfg.hc_mult, D = cfg.hidden_size;

	if (ws.gguf) {
		let tk = 'ple_table.weight';
		let raw = ws.tensor(tk);
		let meta = ws.meta;

		return {
			dev: dev, idx: idx,
			key_w: ws.gpu16(key + '.key_proj.weight', dev).t().contiguous(),     // (E, H*D)
			value_w: ws.gpu16(key + '.value_proj.weight', dev).t().contiguous(), // (E, D)
			norm_key: ws.gpu(key + '.norm_key.weight', dev).to('float32').sub(1.0).to('bfloat16'),
			norm_query: ws.gpu(key + '.norm_query.weight', dev).to('float32').sub(1.0).to('bfloat16'),
			norm_conv: ws.gpu(key + '.norm_conv.weight', dev).to('float32').sub(1.0).to('bfloat16'),
			conv_w: ws.gpu(key + '.conv1d.weight', dev).contiguous().view([H * D, 1, cfg.ple_kernel]),
			state_len: (cfg.ple_kernel - 1) * cfg.ngram_size,
			trellis: raw,
			file: ws.file(tk),
			wname: ws.gguf_name(tk),
			total_rows: null,
			K: 0,
			words: raw.shape()[1],
			offsets: torch.from_array(meta['qwen4exp.ple.head_offsets'], 'int64'),
			sizes: torch.from_array(meta['qwen4exp.ple.head_vocab_sizes'], 'int64'),
			mult: torch.from_array(meta['qwen4exp.ple.layer_multipliers'], 'int64'),
			bias: null,
			heads: (cfg.ngram_size - 1) * cfg.heads_per_ngram,
			gguf: true
		};
	}

	let tab = weights.open(cfg.model_dir, 'ngram_embedding.safetensors');
	let tk = `${key}.ple_embedding.ngram_embedding`;
	// the n-gram table is one tensor (4 / 6 bpw checkpoints) or consecutive equal shards (ng4: 128 x (2.5 M, 41)),
	// which lie back to back in the file: gathers address them as one table
	let sharded = !tab.has(tk + '.trellis');
	let tname = sharded ? tk + '.shard_0.trellis' : tk + '.trellis';
	let trellis = tab.tensor(tname);

	return {
		dev: dev, idx: idx,
		key_w: ws.gpu16(key + '.key_proj.weight', dev).t().contiguous(),     // (E, H*D)
		value_w: ws.gpu16(key + '.value_proj.weight', dev).t().contiguous(), // (E, D)
		norm_key: ws.gpu(key + '.norm_key.weight', dev),
		norm_query: ws.gpu(key + '.norm_query.weight', dev),
		norm_conv: ws.gpu(key + '.norm_conv.weight', dev),
		conv_w: ws.gpu(key + '.conv1d.weight', dev).contiguous(),
		state_len: (cfg.ple_kernel - 1) * cfg.ngram_size,
		trellis: trellis,                                                    // (rows, 1 + 10 K) int16, mmap
		file: tab.file(tname),                                               // the same table for row gathers
		wname: tname,
		total_rows: sharded ? trellis.shape()[0] * cfg.ngram_parts : null,
		K: int((trellis.shape()[1] - 1) / 10),                               // 6 (the 4/6 bpw checkpoints), 4 (ng4)
		words: trellis.shape()[1],
		offsets: tab.tensor(tk + '.head_offsets'),
		sizes: tab.tensor(tk + '.head_vocab_sizes'),
		mult: tab.tensor(tk + '.layer_multipliers'),
		bias: tab.tensor(tk + '.head_bias').cuda(dev),                       // (16, 160) fp16
		heads: (cfg.ngram_size - 1) * cfg.heads_per_ngram
	};
}

/* PLE recurrent state: the conv tail (persistent device tensor, updated in
 * place so CUDA graphs can capture the layer) and the last ngram-1 token ids
 * (host). */
export function new_ple_state(cfg, P) {
	let ids = [];

	for (let i = 0; i < cfg.ngram_size - 1; i++)
		push(ids, cfg.eos_token_id);

	return {
		dev: P.dev,
		conv: torch.zeros([1, cfg.hc_mult * cfg.hidden_size, P.state_len], 'float16',
			'cuda:' + P.dev),
		ids: ids
	};
}

/* Host half, in two steps. ple_start hashes the chunk's n-grams (ctx_ids: the ngram-1 ids before the chunk) and
 * has a worker thread read the unique table rows from disk (many reads in flight, exactly the rows: no page
 * faults on the launching thread, no readahead of 100 KB around every 122-byte row); ple_finish waits for the
 * rows and decodes them on the GPU. The rows depend on the token ids only, so a prefill starts them chunks ahead.
 * Returns the n-gram embedding (1, S, 2560) fp16 on the PLE device; job.next is the id context after the chunk. */
export function ple_start(cfg, P, ctx_ids, token_ids) {
	let S = length(token_ids);
	let ctx = cfg.ngram_size - 1;
	let hist = [];

	for (let i in ctx_ids)
		push(hist, i);

	for (let i in token_ids)
		push(hist, (i >= qwen35.MARK_BASE) ? cfg.image_token_id : i);   // image rows hash the placeholder

	let n = S * P.heads;
	let hist_t = torch.from_array(hist, 'int64').view([1, -1]);
	let uids = torch.zeros([n], 'int64');
	let inverse = torch.zeros([n], 'int64');
	let heads = torch.zeros([n], 'int32');
	let U = torch.ngram_hash(hist_t, S, P.mult, P.offsets, P.sizes,
		cfg.heads_per_ngram, cfg.eos_token_id, uids, inverse, heads);
	let u = uids.narrow(0, 0, U).contiguous();
	let packed = torch.empty([U, P.words], P.gguf ? 'uint8' : 'int16');

	return {S: S, U: U, inverse: inverse, heads: heads, packed: packed,
		ticket: P.total_rows ? P.file.gather_async(P.wname, u, packed, P.total_rows) : P.file.gather_async(P.wname, u, packed),
		next: slice(hist, length(hist) - ctx)};
}

export function ple_finish(cfg, P, job) {
	let dv = 'cuda:' + P.dev;

	torch.gather_wait(job.ticket);

	// non-blocking copies: the host never waits for the device, so the stages keep flowing
	let packed = job.packed.cuda(P.dev, true);

	if (P.gguf) {
		let rows = torch.gguf_dequant(packed, 'IQ4_NL', 'float16');

		return rows.index(job.inverse.cuda(P.dev, true)).view([1, job.S, P.heads * 160]);
	}

	let rows = torch.empty([job.U, 160], 'float16', dv);

	torch.ngram_dequant(packed, P.K, job.heads.narrow(0, 0, job.U).cuda(P.dev, true), P.bias, rows);

	return rows.index(job.inverse.cuda(P.dev, true)).view([1, job.S, P.heads * 160]);
}

/* both steps at once; advances the id context st.ids */
export function ple_prepare(cfg, P, token_ids, st) {
	let job = ple_start(cfg, P, st.ids, token_ids);

	st.ids = job.next;

	return ple_finish(cfg, P, job);
}

/* Device half (capturable): add the PLE delta to the streams (S, 4, D) fp32
 * in place and advance the conv tail. */
export function ple_apply(cfg, P, streams, emb, st) {
	let S = streams.shape()[0];
	let H = cfg.hc_mult, D = cfg.hidden_size;
	let dv = 'cuda:' + P.dev;
	let delta = torch.empty([1, S, H, D], 'float32', dv);
	let cstream = torch.empty([1, H * D, P.state_len + S], 'float16', dv);

	torch.ple_forward(streams.view([1, S, H, D]), emb, P.key_w, P.value_w,
		P.norm_key, P.norm_query, P.norm_conv, P.conv_w, st.conv,
		cfg.rms_norm_eps, 1.0 / math.sqrt(D), cfg.ngram_size, delta, cstream);

	streams.add_(delta.view([S, H, D]));

	// speculative verify (the engine set st.spec_w = W): keep the conv tail after each
	// of the first W-1 rows so a rejection restores it (engine: copy_many)
	if (st.spec_w && S >= 2 && S <= st.spec_w)
		for (let i = 0; i < S - 1; i++)
			st.spec_conv[i].copy_(cstream.narrow(2, i + 1, P.state_len));

	st.conv.copy_(cstream.narrow(2, S, P.state_len));

	return streams;
}

export function ple_forward(cfg, P, streams, token_ids, st) {
	return ple_apply(cfg, P, streams, ple_prepare(cfg, P, token_ids, st), st);
}

/* ------------------------------------------------------------------ */
/* GatedDeltaNet with the sigmoid-gated output norm                     */

const GDN_TRACE = getenv('AMP_GDN_TRACE') != null;
let gdn_trace_n = 0, gdn_trace_calls = 0;

function nf(t) {
	let x = t.to('float32');

	return x.sub(x).abs().max().item() != 0.0;
}

const GDN_NO_ABG = getenv('AMP_QWEN4_NO_ABG') ? true : false;

export function gdn_forward(cfg, L, x, cl) {
	let S = x.shape()[1];
	let Hv = cfg.ln_num_v_heads;
	let qkv = L.in_proj_qkv.forward(x);
	let z = L.in_proj_z.forward(x);
	let a = null, b = null, gates;

	// decode and verify rows (S <= 8) with fp16 a / b weights: both products and the gate math in one launch (at 3 rows
	// the two matmuls fall on a 15 us cuBLAS kernel each, per GDN layer)
	if (S <= 8 && !GDN_NO_ABG && L.in_proj_a.weight != null && L.in_proj_a.bias == null && L.in_proj_b.bias == null &&
	    L.in_proj_b.weight != null) {
		gates = torch.gdn_ab_gates(x.contiguous(), L.in_proj_a.weight, L.in_proj_b.weight, L.dt_bias, L.neg_exp_a_log);
	}
	else {
		b = L.in_proj_b.forward(x);
		a = L.in_proj_a.forward(x);
		gates = torch.gdn_gates(a.contiguous(), b.contiguous(), L.dt_bias, L.neg_exp_a_log);
	}

	// speculative verify (the engine set cl.spec_w = W and this is the W-token verify
	// step): keep the state after each of the first W-1 tokens — the conv window per
	// candidate here (before gdn_prep advances it), the recurrent state in the scan
	let spec = cl.spec_w && S >= 2 && S <= cl.spec_w;       // (any verify size up to the window: adaptive windows)

	if (spec) {
		let km1 = cl.conv_state.shape()[2];
		let win = torch.cat([ cl.conv_state, qkv.transpose(1, 2) ], 2);

		for (let t = 0; t < S - 1; t++)
			cl.spec_conv[t].copy_(win.narrow(2, t + 1, km1));
	}

	let qkv3 = torch.gdn_prep(qkv.contiguous(), cl.conv_state, L.conv_w,
		cfg.ln_num_k_heads, Hv, cfg.ln_k_head_dim);
	let core = spec ?
		torch.gdn_scan(qkv3[0], qkv3[1], qkv3[2], gates[0], gates[1], cl.rec_state,
			1.0 / math.sqrt(cfg.ln_k_head_dim), cl.spec_states.narrow(0, 0, S - 1)) :
		torch.gdn_scan(qkv3[0], qkv3[1], qkv3[2], gates[0], gates[1], cl.rec_state,
			1.0 / math.sqrt(cfg.ln_k_head_dim));

	if (GDN_TRACE && S >= 8) {  // prefill-sized only: graph-captured step sizes (<= W) stay untouched
		let bad = [];

		gdn_trace_calls++;

		for (let kv in [ [ 'x', x ], [ 'qkv', qkv ], [ 'z', z ], [ 'a', a ], [ 'b', b ], [ 'g0', gates[0] ], [ 'g1', gates[1] ],
		                 [ 'q3', qkv3[0] ], [ 'k3', qkv3[1] ], [ 'v3', qkv3[2] ], [ 'core', core ],
		                 [ 'rec', cl.rec_state ], [ 'conv', cl.conv_state ] ])
			if (nf(kv[1]))
				push(bad, kv[0]);

		if (length(bad) && gdn_trace_n < 40) {
			gdn_trace_n++;
			warn(sprintf('gdn-trace: call %d S=%d dev=%s spec=%s non-finite: %s\n', gdn_trace_calls, S,
				x.device ? x.device() : '?', spec ? 'y' : 'n', join(',', bad)));
		}
	}

	// o = rms_norm(core) * w * sigmoid(z)   (fp32; the KDA out norm)
	let o = torch.kda_out_norm(core, z.view([S, cfg.ln_v_dim]).to('float32').contiguous(), L.norm_w32,
		cfg.rms_norm_eps);

	let ov = o.view([1, S, cfg.ln_v_dim]);

	if (L.out_group) {
		let hd = cfg.ln_v_dim / cfg.ln_num_v_heads;

		ov = ov.view([1, S, 16, 3, hd]).transpose(2, 3).contiguous().view([1, S, cfg.ln_v_dim]);
	}

	return L.out_proj.forward(ov);
}

/* ------------------------------------------------------------------ */
/* the K / V cache of a full-attention layer: fp16 rows (max_seq, kvh, hd), or (set_kv_q8, --cache-mode Q8) int8      */
/* codes + a fp16 scale per 32 values (kv_q8_put at write time, dequantized in the qsa_attn kernel's tile staging)    */

let KV_Q8 = (getenv('AMP_QWEN4_KV') == 'q8');

export function set_kv_q8(b) {
	KV_Q8 = b ? true : false;
}

export function kv_alloc(cfg, max_seq, dv) {
	let kvh = cfg.num_kv_heads, hd = cfg.head_dim;

	if (!KV_Q8)
		return {k_cache: torch.zeros([max_seq, kvh, hd], 'float16', dv),
			v_cache: torch.zeros([max_seq, kvh, hd], 'float16', dv)};

	let kq = torch.zeros([max_seq, kvh * hd], 'int8', dv), ks = torch.zeros([max_seq, kvh * hd / 32], 'float16', dv);
	let vq = torch.zeros([max_seq, kvh * hd], 'int8', dv), vs = torch.zeros([max_seq, kvh * hd / 32], 'float16', dv);

	return {kq: kq, ks: ks, vq: vq, vs: vs, kq3: kq.view([max_seq, kvh, hd]), ks3: ks.view([max_seq, kvh, hd / 32]),
		vq3: vq.view([max_seq, kvh, hd]), vs3: vs.view([max_seq, kvh, hd / 32])};
}

/* rows kk, v (S, kvh, hd) fp16 into the cache at `pos`: a start row (integer) or an int64 tensor of S rows */
function kv_put(cfg, cl, kk, v, pos, S) {
	let w = cfg.num_kv_heads * cfg.head_dim;

	if (cl.kq) {
		torch.kv_q8_put(kk.reshape([S, w]).contiguous(), cl.kq, cl.ks, pos);
		torch.kv_q8_put(v.reshape([S, w]).contiguous(), cl.vq, cl.vs, pos);
	}
	else if (type(pos) == 'int') {
		cl.k_cache.narrow(0, pos, S).copy_(kk.view([S, cfg.num_kv_heads, cfg.head_dim]));
		cl.v_cache.narrow(0, pos, S).copy_(v.view([S, cfg.num_kv_heads, cfg.head_dim]));
	}
	else {
		cl.k_cache.index_copy_(0, pos, kk.view([S, cfg.num_kv_heads, cfg.head_dim]));
		cl.v_cache.index_copy_(0, pos, v.view([S, cfg.num_kv_heads, cfg.head_dim]));
	}
}

/* the cache arguments of torch.qsa_attn */
function kv_k(cl) {
	return cl.kq ? [ cl.kq3, cl.ks3 ] : cl.k_cache;
}

function kv_v(cl) {
	return cl.kq ? [ cl.vq3, cl.vs3 ] : cl.v_cache;
}

/* rows [0, len) as fp16 (len, kvh, hd) */
function kv_rows(cfg, cl, len) {
	if (!cl.kq)
		return [ cl.k_cache.narrow(0, 0, len), cl.v_cache.narrow(0, 0, len) ];

	let sh = [ len, cfg.num_kv_heads, cfg.head_dim ];

	return [ torch.kv_q8_get(cl.kq, cl.ks, 0, len).view(sh), torch.kv_q8_get(cl.vq, cl.vs, 0, len).view(sh) ];
}

const QSA_SCORE_ELEMS = 50331648;       // fp32 score elements per selection slab (192 MiB)

/* ------------------------------------------------------------------ */
/* full attention with the QSA indexer                                  */
/*                                                                     */
/* The KV cache of a full-attention layer is (max_seq, kvh, head_dim) so */
/* a row gather (tensor.index) fetches the selected tokens' K/V of all  */
/* kv heads. Next to it the indexer keeps raw keys (max_seq, 128) and   */
/* the mean-pooled, normed, roped block keys (max_seq / 4, 128): a block */
/* is pooled the moment its 4th key arrives.                            */

/* Indexer step for a chunk: projects queries (normed + roped), stores the
 * raw keys, pools the blocks the chunk completed. Returns the roped index
 * queries (S, heads, 128) fp16. */
export function indexer_update(cfg, L, x, cl, cos, sin, pos) {
	let S = x.shape()[1];
	let nh = cfg.indexer_heads, hd = cfg.indexer_head_dim;
	let cr = cfg.indexer_compress_ratio;
	let qk = L.idx_qk.forward(x);                                  // (1, S, (nh + 1) * hd)
	let q = qk.narrow(-1, 0, nh * hd).view([1, S, nh, hd]);
	let raw = qk.narrow(-1, nh * hd, hd).view([S, hd]);

	q = torch.qk_norm_rope(q, L.idx_qn, cos, sin, cfg.rotary_dim, cfg.rms_norm_eps);

	cl.raw_k.narrow(0, pos, S).copy_(raw);

	let nb0 = int(pos / cr), nb1 = int((pos + S) / cr);

	if (nb1 > nb0) {
		let nb = nb1 - nb0;
		let pooled = cl.raw_k.narrow(0, nb0 * cr, nb * cr).view([nb, cr, hd])
			.to('float32').mean(1).to('float16').contiguous();
		let starts = [];

		for (let j = nb0; j < nb1; j++)
			push(starts, j * cr);

		let st = torch.from_array(starts, 'int64', 'cuda:' + cl.dev);
		let pk = torch.qk_norm_rope(pooled.view([1, nb, 1, hd]), L.idx_kn,
			cl.cos.index(st), cl.sin.index(st), cfg.rotary_dim, cfg.rms_norm_eps);

		cl.pooled.narrow(0, nb0, nb).copy_(pk.view([nb, hd]));
	}

	return q.view([S, nh, hd]);
}

/* The selection core on device tensors (so a CUDA graph can replay it):
 * qpos_t, nbq_t (R, 1) and tail_t (R, cr) int64, nb the number of pooled
 * blocks scanned (masked per query by nbq_t). */
export function qsa_select_t(cfg, cl, qi, qpos_t, nbq_t, tail_t, nb, R) {
	let cr = cfg.indexer_compress_ratio;
	let dv = 'cuda:' + cl.dev;
	let ktop = min(int(cfg.indexer_budget / cr), nb);
	let ones = torch.ones([R, cr], 'float32', dv);

	// block scores: sum over the index heads of relu(q . pooled) / sqrt(dk)
	let s = torch.matmul(qi.to('float32').view([R * cfg.indexer_heads, cfg.indexer_head_dim]),
		cl.pooled.narrow(0, 0, nb).to('float32').t());               // (R * heads, nb)
	let sc = s.clamp(0.0, null).view([R, cfg.indexer_heads, nb]).sum(1)
		.mul(1.0 / math.sqrt(cfg.indexer_head_dim * 1.0));            // (R, nb)
	let cols = torch.arange(nb, 'int64', dv).view([1, nb]);
	let hidden = nbq_t.le(cols);                                      // block not yet complete for this query

	sc = torch.where(hidden, torch.full([1], -1e30, 'float32', dv), sc);

	let top = sc.topk(ktop, -1);
	let bad = top.values.le(-1e29).to('float32');                      // (R, ktop) padding picks
	let four = torch.full([1], cr, 'int64', dv);
	let sel = top.indices.mul(four).view([R, ktop, 1])
		.add(torch.arange(cr, 'int64', dv).view([1, 1, cr])).view([R, ktop * cr]);
	let selok = torch.ones([R, ktop], 'float32', dv).sub(bad)
		.view([R, ktop, 1]).mul(torch.ones([1, 1, cr], 'float32', dv)).view([R, ktop * cr]);

	let tok = torch.cat([ sel, tail_t ], 1);
	let ok = torch.cat([ selok, ones ], 1).mul(tok.le(qpos_t).to('float32'));

	return {
		tok: torch.where(ok.le(0.5), torch.zeros([1], 'int64', dv), tok).contiguous(),
		ok: ok
	};
}

/* Token lists of the sparse attention for R consecutive queries starting at
 * absolute position p0: per query the tokens of its top block_topk complete
 * blocks plus the tail of the incomplete one. Returns {tok (R, Kt) int64
 * (invalid entries 0), ok (R, Kt) fp32 (1 valid / 0 not)}. */
function qsa_select(cfg, cl, qi, p0, R, nb) {
	let cr = cfg.indexer_compress_ratio;
	let dv = 'cuda:' + cl.dev;

	// query positions, complete blocks per query ((p + 1) / cr, exact in fp32 for
	// p < 2^24) and the tail tokens, all on the device (a host loop costs ~1 us per
	// element: 6 ms per 1024-row slab and layer)
	let qpos_t = torch.arange(R, 'int64', dv).add(torch.full([1], p0, 'int64', dv)).view([R, 1]);
	let nbq_t = qpos_t.add(torch.full([1], 1, 'int64', dv)).to('float32')
		.mul(1.0 / cr).to('int64');
	let tail_t = nbq_t.mul(torch.full([1], cr, 'int64', dv))
		.add(torch.arange(cr, 'int64', dv).view([1, cr]));

	return qsa_select_t(cfg, cl, qi, qpos_t, nbq_t, tail_t, nb, R);
}

/* Composite reference of qsa_gather_attend (tests, AMP_QWEN4_QSA_REF=1): gathers the
 * K/V rows with ATen, softmax in fp32. q: (R, nh, hd) fp16. */
export function qsa_gather_attend_ref(cfg, cl, q, sel, R) {
	let nh = cfg.num_q_heads, kvh = cfg.num_kv_heads, hd = cfg.head_dim;
	let rep = int(nh / kvh);
	let dv = 'cuda:' + cl.dev;
	let scale = 1.0 / math.sqrt(hd * 1.0);
	let Kt = sel.tok.shape()[1];
	let flat = sel.tok.view([R * Kt]);
	let ms = cl.k_cache.shape()[0];
	let Kg = cl.k_cache.view([ms, kvh * hd]).index(flat).view([R, Kt, kvh, hd])
		.transpose(1, 2).contiguous();
	let Vg = cl.v_cache.view([ms, kvh * hd]).index(flat).view([R, Kt, kvh, hd])
		.transpose(1, 2).contiguous();
	let qs = q.view([R, kvh, rep, hd]);
	let sco = torch.matmul(qs, Kg.transpose(2, 3)).to('float32').mul(scale);   // (R, kvh, rep, Kt)
	let mask = sel.ok.view([R, 1, 1, Kt]).le(0.5);

	sco = torch.where(mask, torch.full([1], -1e30, 'float32', dv), sco).softmax(-1);

	return torch.matmul(sco.to('float16'), Vg).view([R, nh, hd]);
}

/* Attention of R queries over the token lists of `sel` (tok (R, Kt) int64 with
 * padding entries 0, ok (R, Kt) fp32 validity): the fused gathered-attention
 * kernel (torch.qsa_attn) reads the selected K/V rows straight from the cache.
 * q: (R, nh, hd) fp16. */
function qsa_gather_attend(cfg, cl, q, sel, R) {
	if (QSA_REF)
		return qsa_gather_attend_ref(cfg, cl, q, sel, R);

	let dv = 'cuda:' + cl.dev;
	let idx = torch.where(sel.ok.le(0.5), torch.full([1], -1, 'int64', dv), sel.tok)
		.to('int32').contiguous();

	return torch.qsa_attn(q.contiguous(), kv_k(cl), kv_v(cl), idx,
		1.0 / math.sqrt(cfg.head_dim * 1.0));
}

/* Gathered GQA attention for the queries of a chunk (q: (S, nh, hd) fp16,
 * qi: roped index queries): returns (S, nh, hd) fp16. */
function qsa_attend(cfg, cl, q, qi, pos, S) {
	let cr = cfg.indexer_compress_ratio;
	let nb_all = int((pos + S) / cr);
	// slab rows: the fp32 score matrix (rows * heads * blocks) stays within ~192 MiB
	let slab = (S == 1) ? 1 : max(32, int(QSA_SCORE_ELEMS / (cfg.indexer_heads * max(nb_all, 1)) / 32) * 32);
	let outs = [];

	for (let r0 = 0; r0 < S; r0 += slab) {
		let R = min(slab, S - r0);
		let p0 = pos + r0;
		let nb = int((p0 + R) / cr);            // complete blocks at the slab's last query
		let sel = qsa_select(cfg, cl, qi.narrow(0, r0, R).contiguous(), p0, R, nb);

		let ot = qsa_gather_attend(cfg, cl, q.narrow(0, r0, R), sel, R);

		if (GDN_TRACE && S >= 8 && nf(ot)) {
			let n = pos + S;

			warn(sprintf('qsa-trace: dev %s pos %d S %d slab r0 %d R %d nb %d slab-size %d | out bad | in: q %s qi %s k_cache %s v_cache %s pooled %s | valid-avg %g sel-max %d\n',
				cl.dev, pos, S, r0, R, nb, slab,
				nf(q.narrow(0, r0, R)) ? 'BAD' : 'ok', nf(qi.narrow(0, r0, R)) ? 'BAD' : 'ok',
				nf(cl.k_cache.narrow(0, 0, n)) ? 'BAD' : 'ok', nf(cl.v_cache.narrow(0, 0, n)) ? 'BAD' : 'ok',
				nf(cl.pooled.narrow(0, 0, nb)) ? 'BAD' : 'ok',
				sel.ok.sum().item() / R, sel.tok.max().item()));
		}

		push(outs, ot);
	}

	return (length(outs) == 1) ? outs[0] : torch.cat(outs, 0);
}

/* One full-attention layer: gated GQA with the QSA indexer. x (1, S, D) fp16;
 * cos/sin the chunk's rope rows. Dense (FlashAttention) while the context
 * fits the selection budget (the selection then keeps everything), gathered
 * sparse attention beyond it. */
export function attn_forward(cfg, L, x, cos, sin, cl, pos, mr) {
	let S = x.shape()[1];
	let eps = cfg.rms_norm_eps;
	let nh = cfg.num_q_heads, kvh = cfg.num_kv_heads, hd = cfg.head_dim;

	let qi = indexer_update(cfg, L, x, cl, cos, sin, pos);

	let qg = L.q_proj.forward(x).view([1, S, nh, 2 * hd]);
	let q = qg.narrow(-1, 0, hd);
	let g = qg.narrow(-1, hd, hd);
	let kk = L.k_proj.forward(x).view([1, S, kvh, hd]);
	let v = L.v_proj.forward(x).view([1, S, kvh, hd]);

	// main attention: MRoPE rows (mr) when the chunk has image tokens / follows an
	// image; the indexer keeps the plain cache positions (exllamav3: cache lengths)
	let mcos = mr ? mr.cos : cos, msin = mr ? mr.sin : sin;

	q = torch.qk_norm_rope(q, L.q_norm_w, mcos, msin, cfg.rotary_dim, eps);       // (1, S, nh, hd)
	kk = torch.qk_norm_rope(kk, L.k_norm_w, mcos, msin, cfg.rotary_dim, eps);

	kv_put(cfg, cl, kk, v, pos, S);

	let len = pos + S;
	let attn;

	if (len <= cfg.sparse_threshold) {
		let kvr = kv_rows(cfg, cl, len);
		let K = kvr[0].transpose(0, 1).unsqueeze(0).contiguous();
		let V = kvr[1].transpose(0, 1).unsqueeze(0).contiguous();

		attn = torch.attn_gate(torch.flash_attn(q.transpose(1, 2), K, V, true), g);
	} else {
		let o = qsa_attend(cfg, cl, q.view([S, nh, hd]), qi, pos, S);

		attn = torch.attn_gate(o.view([1, S, nh, hd]).transpose(1, 2), g);
	}

	return L.o_proj.forward(attn);
}


/* Per-step index state of a decode graph (S rows), one packed int64 vector
 * the host rewrites before every replay (positions are host values, the
 * graph only sees tensors): [pos S][nbq S][tail cr*S][blk B][blk_rows cr*B]
 * [blk_start B] with B the blocks a window of S tokens can touch. */
export function dev_state(cfg, S, max_seq, dev) {
	let cr = cfg.indexer_compress_ratio;
	let B = int((S + cr - 2) / cr) + 1;
	let n = (3 + cr) * S + (2 + cr) * B;
	let pack = torch.zeros([n], 'int64', 'cuda:' + dev);
	let o = 0;
	let st = {pack: pack, S: S, B: B, n: n, nb: int(max_seq / cr), cr: cr};

	st.pos = pack.narrow(0, o, S); o += S;
	st.rpos = pack.narrow(0, o, S); o += S;           // rope positions of the main attention
	st.nbq = pack.narrow(0, o, S).view([S, 1]); o += S;
	st.tail = pack.narrow(0, o, cr * S).view([S, cr]); o += cr * S;
	st.blk = pack.narrow(0, o, B); o += B;
	st.blk_rows = pack.narrow(0, o, cr * B); o += cr * B;
	st.blk_start = pack.narrow(0, o, B);
	st.qpos = st.pos.view([S, 1]);

	return st;
}

/* The values of the packed state for a step at absolute position pos. */
export function dev_values(st, pos, mm) {
	let S = st.S, B = st.B, cr = st.cr;
	let v = [];

	for (let i = 0; i < S; i++)
		push(v, pos + i);

	for (let i = 0; i < S; i++)
		push(v, pos + i + qwen35.mm_off(mm, pos + i));

	for (let i = 0; i < S; i++)
		push(v, int((pos + i + 1) / cr));

	for (let i = 0; i < S; i++)
		for (let t = 0; t < cr; t++)
			push(v, int((pos + i + 1) / cr) * cr + t);

	let b0 = int(pos / cr), blks = [];

	for (let j = 0; j < B; j++)
		push(blks, min(b0 + j, st.nb - 1));

	for (let b in blks)
		push(v, b);

	for (let b in blks)
		for (let t = 0; t < cr; t++)
			push(v, b * cr + t);

	for (let b in blks)
		push(v, b * cr);

	return v;
}

/* attn_forward for a CUDA-graph step: every position-dependent quantity comes
 * from the device state st; the selection scans all pooled-block capacity (masked
 * per query) and gathers a fixed 2052 tokens, so the shapes never change. While
 * the context fits the budget the selection keeps every token: the same
 * result as dense attention. */
const NO_DSEL = getenv('AMP_QWEN4_NO_DSEL') ? true : false;
let idx_w = {};

/* The decode step's token lists (S, K) int32, -1 invalid, with the DSA kernels instead of the composite
 * chain: idx_score scores the live pools only (the indexer heads padded to its 32, zero queries and weights),
 * dsa_select takes the top pools, their tokens and the incomplete tail from the device-side position. The
 * chain scanned the whole pool capacity (65536 blocks at ctx 262144) for every attention layer of every step. */
export function qsa_idx_dev(cfg, cl, qi, st, S) {
	let dv = 'cuda:' + cl.dev;
	let ih = cfg.indexer_heads, ihd = cfg.indexer_head_dim, cr = cfg.indexer_compress_ratio;
	let kp = int(cfg.indexer_budget / cr);
	let key = cl.dev + ':' + S;

	if (!idx_w[key]) {
		let w = torch.zeros([S, 32], 'float32', dv);

		w.narrow(1, 0, ih).fill_(1.0);
		idx_w[key] = w;
	}

	let qp = torch.cat([ qi.view([S, ih, ihd]), torch.zeros([S, 32 - ih, ihd], 'float16', dv) ], 1).contiguous();
	let p0 = st.pos.narrow(0, 0, 1);
	let sc = torch.idx_score(qp, cl.pooled, idx_w[key], 1.0 / math.sqrt(ihd * 1.0), p0, cr);
	let r = torch.dsa_select(sc, p0, cr, kp);

	return torch.where(r[1].le(-1.0), torch.full([1], -1, 'int64', dv), r[0]).to('int32').contiguous();
}

export function attn_forward_dev(cfg, L, x, cl, st) {
	let S = x.shape()[1];
	let eps = cfg.rms_norm_eps;
	let nh = cfg.num_q_heads, kvh = cfg.num_kv_heads, hd = cfg.head_dim;
	let ih = cfg.indexer_heads, ihd = cfg.indexer_head_dim, B = st.B;
	let cos = cl.cos.index(st.pos), sin = cl.sin.index(st.pos);          // (S, rd/2), indexer
	let mcos = cl.cos.index(st.rpos), msin = cl.sin.index(st.rpos);      // main attention

	// indexer: queries, raw keys, the pooled blocks the new tokens touch
	let qk = L.idx_qk.forward(x);
	let qi = torch.qk_norm_rope(qk.narrow(-1, 0, ih * ihd).view([1, S, ih, ihd]), L.idx_qn,
		cos, sin, cfg.rotary_dim, eps).view([S, ih, ihd]);

	cl.raw_k.index_copy_(0, st.pos, qk.narrow(-1, ih * ihd, ihd).view([S, ihd]));

	let pooled = cl.raw_k.index(st.blk_rows).view([B, cfg.indexer_compress_ratio, ihd])
		.to('float32').mean(1).to('float16').contiguous();
	let pk = torch.qk_norm_rope(pooled.view([1, B, 1, ihd]), L.idx_kn,
		cl.cos.index(st.blk_start), cl.sin.index(st.blk_start), cfg.rotary_dim, eps);

	cl.pooled.index_copy_(0, st.blk, pk.view([B, ihd]));

	// attention projections, KV write
	let qg = L.q_proj.forward(x).view([1, S, nh, 2 * hd]);
	let q = torch.qk_norm_rope(qg.narrow(-1, 0, hd), L.q_norm_w, mcos, msin, cfg.rotary_dim, eps);
	let g = qg.narrow(-1, hd, hd);
	let kk = torch.qk_norm_rope(L.k_proj.forward(x).view([1, S, kvh, hd]), L.k_norm_w,
		mcos, msin, cfg.rotary_dim, eps);
	let v = L.v_proj.forward(x).view([1, S, kvh, hd]);

	kv_put(cfg, cl, kk, v, st.pos, S);

	let o;

	if (NO_DSEL) {
		let sel = qsa_select_t(cfg, cl, qi, st.qpos, st.nbq, st.tail, st.nb, S);

		o = qsa_gather_attend(cfg, cl, q.view([S, nh, hd]), sel, S);
	}
	else {
		o = torch.qsa_attn(q.view([S, nh, hd]).contiguous(), kv_k(cl), kv_v(cl), qsa_idx_dev(cfg, cl, qi, st, S),
			1.0 / math.sqrt(cfg.head_dim * 1.0));
	}

	return L.o_proj.forward(torch.attn_gate(o.view([1, S, nh, hd]).transpose(1, 2), g));
}

/* ------------------------------------------------------------------ */
/* layers                                                              */

function build_moe(ws, cfg, key, dev, layer) {
	if (ws.gguf) {
		let E = cfg.n_routed_experts;
		let f = expert_resident_fraction(layer);
		let R = (f >= 1.0) ? E : max(1, min(E, int(E * f)));
		let ids = [], cold_ids = [];

		if (R < E && layer != null) {
			let p = profile_order(layer);

			if (p && length(p)) {
				ids = slice(p, 0, min(R, length(p)));

				let in_res = {};
				for (let e in ids)
					in_res[e] = true;

				// pad to R if the profile covers fewer experts than the resident budget
				for (let e = 0; e < E && length(ids) < R; e++)
					if (!in_res[e]) {
						push(ids, e);
						in_res[e] = true;
					}

				for (let e = 0; e < E; e++)
					if (!in_res[e])
						push(cold_ids, e);
			}
		}

		if (!length(ids)) {
			for (let e = 0; e < R; e++)
				push(ids, e);

			for (let e = R; e < E; e++)
				push(cold_ids, e);
		}

		let res_of = [], cold_of = [];

		for (let e = 0; e < E; e++) {
			res_of[e] = -1;
			cold_of[e] = -1;
		}

		for (let s = 0; s < length(ids); s++)
			res_of[ids[s]] = s;

		for (let s = 0; s < length(cold_ids); s++)
			cold_of[cold_ids[s]] = s;

		let gq = ggufq.experts_pick(ws, key, dev, ids);
		let gq_cold = length(cold_ids) ? ggufq.experts_host_mapped(ws, key, cold_ids, dev) : null;

		let pt1 = [], pt3 = [], pt2 = [];
		let rb1 = gq.w1.shape()[2], rb3 = gq.w3.shape()[2], rb2 = gq.w2.shape()[2];
		let n1 = gq.w1.shape()[1], n3 = gq.w3.shape()[1], n2 = gq.w2.shape()[1];

		for (let e = 0; e < E; e++) {
			let rs = res_of[e], cs = cold_of[e];
			let src = (rs >= 0) ? gq : gq_cold;
			let slot = (rs >= 0) ? rs : cs;

			push(pt1, src.w1.data_ptr() + slot * n1 * rb1);
			push(pt3, src.w3.data_ptr() + slot * n3 * rb3);
			push(pt2, src.w2.data_ptr() + slot * n2 * rb2);
		}

		// device-side row mapping for the pointer-table decode path (T <= 8): slot s reads x row s / Kt.
		// Precomputed so a CUDA graph capture does not stage a host array through pinned memory.
		let slots = [];

		for (let s = 0; s < 8 * cfg.num_experts_per_tok; s++)
			push(slots, int(s / cfg.num_experts_per_tok));

		return {
			type: 'moe', dev: dev, router: 'softmax', layer: layer,
			gq: gq, gq_cold: gq_cold,
			pt_host: {pt1: pt1, pt3: pt3, pt2: pt2},
			pt1: torch.from_array(pt1, 'int64', 'cuda:' + dev),
			pt3: torch.from_array(pt3, 'int64', 'cuda:' + dev),
			pt2: torch.from_array(pt2, 'int64', 'cuda:' + dev),
			slot_rows: torch.from_array(slots, 'int64', 'cuda:' + dev),
			res_of: res_of, cold_of: cold_of, n_experts_total: E,
			gate_t: ws.gpu(key + '.gate.weight', dev).to('float32').t().contiguous(),
			shared: ggufq.mlp(ws, key + '.shared_expert', dev),
			shared_gate: ws.gpu(key + '.shared_expert_gate.weight', dev).to('float32')
				.t().contiguous(),
			gq_forward: function (cfg, M, y) {
				return ggufq.moe_forward(cfg, M, y, {route: glm5.moe_route, act: glm5.act_mul, mlp: glm5.mlp_forward, side: glm5.par_side, join: glm5.par_join});
			}
		};
	}

	let experts = [];

	for (let e = 0; e < cfg.n_routed_experts; e++)
		push(experts, glm5.build_mlp(ws, `${key}.experts.${e}`, dev));

	return {
		type: 'moe', dev: dev, router: 'softmax', experts: experts,
		gate_t: ws.gpu(key + '.gate.weight', dev).to('float32').t().contiguous(),
		shared: glm5.build_mlp(ws, key + '.shared_expert', dev),
		shared_gate: ws.gpu(key + '.shared_expert_gate.weight', dev).to('float32')
			.t().contiguous()
	};
}

/* The GDN / full-attention weights of a layer (no hyper-connection sites, no MoE),
 * all on `dev`. Shared by the layer-split loader and the tensor-parallel one. */
/* GGUF qwen4exp stores the linear-attention value heads group-major: (3 groups, 16 heads), while the HF / EXL3
 * layout is head-major (16 heads, 3 groups). The q/k part is unchanged. These helpers put the GGUF tensors into
 * the HF order the kernels expect. */
function perm_all_value_rows(t) {
	let rb = t.shape()[1];

	return t.view([3, 16, 128, rb]).transpose(0, 1).contiguous().view([6144, rb]);
}

function perm_value_rows(t) {
	let rb = t.shape()[1];
	let qk = t.narrow(0, 0, 4096);
	let v = perm_all_value_rows(t.narrow(0, 4096, 6144));

	return torch.cat([ qk, v ], 0);
}

function perm_heads_rows(t) {
	return t.view([3, 16, t.shape()[1]]).transpose(0, 1).contiguous().view([48, t.shape()[1]]);
}

function perm_heads_vec(t) {
	return t.view([3, 16]).t().contiguous().view([48]);
}

/* a plain linear over an fp16 tensor (same forward as models.LinearFP16) */
function tensor_linear(w) {
	return {
		weight: w, bias: null, in_features: w.shape()[1], out_features: w.shape()[0],

		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x : x.view([-1, this.in_features]);
			let y = torch.matmul(flat, this.weight.t());

			return (length(s) == 2) ? y : y.view([s[0], s[1], this.out_features]);
		}
	};
}

export function build_sublayer_at(ws, cfg, key, type, dev) {
	let L = {type: type, dev: dev};

	let lin = function (ws, key, seq, dev, perm) {
		if (!ws.gguf)
			return models.linear(ws, key, seq, dev);

		let qt = ws.qtype(key + '.weight');

		if (qt) {
			let raw = ws.gpu(key + '.weight', dev);

			if (perm) {
				let p = perm(raw);

				ws.drop(key + '.weight', dev);
				raw = p;
			}

			return ggufq.linear_from(raw, qt, dev);
		}

		let w = ws.gpu16(key + '.weight', dev);

		if (perm) {
			let p = perm(w);

			ws.drop(key + '.weight', dev);
			w = p;
		}

		return tensor_linear(w);
	};

	if (L.type == 'linear_attention') {
		let k = key + '.linear_attn';
		let g = ws.gguf;

		L.in_proj_qkv = lin(ws, k + '.in_proj_qkv', true, dev, g ? perm_value_rows : null);
		L.in_proj_z = lin(ws, k + '.in_proj_z', true, dev, g ? perm_all_value_rows : null);
		L.in_proj_b = lin(ws, k + '.in_proj_b', true, dev, g ? perm_heads_rows : null);
		L.in_proj_a = lin(ws, k + '.in_proj_a', true, dev, g ? perm_heads_rows : null);
		L.out_proj = lin(ws, k + '.out_proj', true, dev);
		L.out_group = g;        // out_proj's input columns are group-major in the GGUF
		L.dt_bias = ws.gpu(k + '.dt_bias', dev).to('float32');
		L.neg_exp_a_log = ws.gpu(k + '.A_log', dev).to('float32');

		if (g) {
			L.dt_bias = perm_heads_vec(L.dt_bias);
			L.neg_exp_a_log = perm_heads_vec(L.neg_exp_a_log);   // ssm_a is already -exp(A_log)
		} else {
			L.neg_exp_a_log = L.neg_exp_a_log.exp().neg();
		}

		L.conv_w = ws.gpu(k + '.conv1d.weight', dev).squeeze(1).to('float16');

		if (g)
			L.conv_w = perm_value_rows(L.conv_w);

		L.norm_w32 = ws.gpu(k + '.norm.weight', dev).to('float32').contiguous();
	} else {
		let k = key + '.self_attn';

		L.q_proj = lin(ws, k + '.q_proj', true, dev);
		L.k_proj = lin(ws, k + '.k_proj', true, dev);
		L.v_proj = lin(ws, k + '.v_proj', true, dev);
		L.o_proj = lin(ws, k + '.o_proj', true, dev);
		L.q_norm_w = ws.gpu(k + '.q_norm.weight', dev).to('float16');
		L.k_norm_w = ws.gpu(k + '.k_norm.weight', dev).to('float16');

		if (!ws.gguf) {
			L.q_norm_w = L.q_norm_w.add(1.0);
			L.k_norm_w = L.k_norm_w.add(1.0);
		}

		if (ws.gguf) {
			L.idx_q = lin(ws, k + '.indexer.q_proj', true, dev);
			L.idx_k = lin(ws, k + '.indexer.k_proj', true, dev);
			L.idx_qn = ws.gpu(k + '.indexer.q_norm.weight', dev).to('float16');
			L.idx_kn = ws.gpu(k + '.indexer.k_norm.weight', dev).to('float16');

			if (!ws.gguf) {
				L.idx_qn = L.idx_qn.add(1.0);
				L.idx_kn = L.idx_kn.add(1.0);
			}
			L.idx_qk = {
				forward: function (x) {
					return torch.cat([ L.idx_q.forward(x), L.idx_k.forward(x) ], -1);
				}
			};
		} else {
			L.idx_qk = models.linear(ws, k + '.indexer.index_qk_proj', true, dev);
			L.idx_qn = ws.gpu(k + '.indexer.q_layernorm.weight', dev).to('float16').add(1.0);
			L.idx_kn = ws.gpu(k + '.indexer.k_layernorm.weight', dev).to('float16').add(1.0);
		}
	}

	return L;
}

export function build_layer_at(ws, cfg, key, type, dev, layer) {
	let L = build_sublayer_at(ws, cfg, key, type, dev);

	L.attn_hc = build_gr(ws, cfg, key + '.attn_hyper_connection', dev, true);
	L.mlp_hc = build_gr(ws, cfg, key + '.mlp_hyper_connection', dev, true);
	L.moe = build_moe(ws, cfg, key + '.mlp', dev, layer);

	// count the router's picks of the layers with host-resident experts (the profile of the next start)
	if (CONF.count && layer != null && L.moe.gq_cold) {
		PROFILER ??= qprof.create(cfg.n_routed_experts, cfg.num_experts_per_tok, PROF ? PROF.counts : null, PROF ? PROF.tokens : 0);
		L.moe.prof = PROFILER.hook(layer, dev);
	}

	return L;
}

export function build_layer(ws, cfg, i, dev) {
	let L = build_layer_at(ws, cfg, LP + i, cfg.layer_types[i], dev, i);
	L.moe.layer = i;

	return L;
}

/* Phase A of a layer on streams (S, 4, D) fp32 (updated in place): the
 * attention site and the MoE routing (async, no host read). Returns the
 * pending MoE for layer_b. */
export function layer_a(cfg, L, cl, streams, cos, sin, pos, st, mr) {
	let S = streams.shape()[0];
	let D = cfg.hidden_size;

	let m = gr_mix(cfg, L.attn_hc, streams);
	let x = m[1].view([1, S, D]);
	let y = (L.type == 'linear_attention') ?
		gdn_forward(cfg, L, x, cl) :
		(st ? attn_forward_dev(cfg, L, x, cl, st) : attn_forward(cfg, L, x, cos, sin, cl, pos, mr));

	torch.hc_apply(streams, y.view([S, D]).contiguous(), m[0], null);

	m = gr_mix(cfg, L.mlp_hc, streams);

	return {post: m[0], moe: glm5.moe_begin(cfg, L.moe, m[1])};
}

/* Phase B: the experts (reads the per-expert counts: syncs this device) and
 * the residual injection. */
export function layer_b(cfg, L, streams, ctx) {
	torch.hc_apply(streams, glm5.moe_finish(cfg, L.moe, ctx.moe), ctx.post, null);

	return streams;
}

export function layer_forward(cfg, L, cl, streams, cos, sin, pos, st, mr) {
	return layer_b(cfg, L, streams, layer_a(cfg, L, cl, streams, cos, sin, pos, st, mr));
}

/* ------------------------------------------------------------------ */
/* model                                                               */

function dev_list(cfg, devices) {
	let n = cfg.num_hidden_layers;

	if (type(devices) == 'array') {
		if (length(devices) != n)
			die(`qwen4: devices array must have ${n} entries`);

		return devices;
	}

	let out = [];
	let ng = devices != null ? devices : torch.device_count();

	for (let i = 0; i < n; i++)
		push(out, int(i * ng / n));

	return out;
}

/* Embed a chunk and run it through the layers (the stream stack lives on the
 * layer's device). Returns the streams (S, 4, D) fp32 on the last layer's
 * device and advances cache.length. on_layer(li) runs after each layer. */
function run_layers(model, token_ids, cache, on_layer) {
	let cfg = model.cfg;
	let S = length(token_ids);
	let pos = cache.length;
	let H = cfg.hc_mult, D = cfg.hidden_size;
	let h = qwen35.embed_rows(model, cache.mm, token_ids, pos);      // (1, S, D) fp16
	let streams = h.view([S, 1, D]).to('float32').expand([S, H, D]).contiguous();

	for (let i = 0; i < model.nl; i++) {
		let L = model.layers[i];

		if (i > 0 && L.dev != model.layers[i - 1].dev)
			streams = streams.cuda(L.dev);

		if (model.ples[i]) {
			ple_forward(cfg, model.ples[i], streams, token_ids, cache.ple);

			if (cache.trace)
				cache.trace.ple = streams.cpu();
		}

		let cl = cache.layers[i];

		layer_forward(cfg, L, cl, streams,
			cl.cos ? cl.cos.narrow(0, pos, S) : null,
			cl.sin ? cl.sin.narrow(0, pos, S) : null, pos, null,
			cl.cos ? qwen35.rope_rows(cfg, cache.mm, cl, pos, S) : null);

		if (cache.trace)
			push(cache.trace.blocks, streams.cpu());

		if (on_layer)
			on_layer(i);
	}

	cache.length = pos + S;

	return streams;
}

/* (S, 4, D) fp32 stack -> (1, S, 4D) fp16 */
export function flat_stack_of(model, stack) {
	return stack.to('float16').view([1, stack.shape()[0], model.cfg.hc_mult * model.cfg.hidden_size]);
}

/* The draft head's view of a stream stack: (1, S, 4 * D) fp16 on the head device. */
function flat_stack(model, streams) {
	let S = streams.shape()[0];

	if (model.layers[model.nl - 1].dev != model.mixer.dev)
		streams = streams.cuda(model.mixer.dev);

	return streams.to('float16').view([1, S, model.cfg.hc_mult * model.cfg.hidden_size]);
}

/* Collapse the stream stack with the final mixer (no final norm): (S, D) fp16
 * on the head device. */
function head_hidden(model, streams) {
	if (model.layers[model.nl - 1].dev != model.mixer.dev)
		streams = streams.cuda(model.mixer.dev);

	return gr_mix(model.cfg, model.mixer, streams)[1];
}

/* ------------------------------------------------------------------ */
/* CUDA-graph decode                                                   */
/*                                                                     */
/* One graph per device stage (a contiguous run of layers) and step size.*/
/* Everything position-dependent comes from tensors the host rewrites     */
/* before a replay (dev_state / dev_values: KV and indexer write rows,   */
/* the per-query block counts and tail tokens); the QSA selection scans    */
/* the whole pooled-block capacity masked per query and gathers a fixed    */
/* number of tokens, so no shape depends on the context length. The PLE   */
/* layer reads its n-gram embedding from a persistent buffer filled by    */
/* the host (hashing + table gather stay on the CPU).                     */

function build_graphs(model, cache, sizes) {
	let cfg = model.cfg;
	let nl = model.nl;
	let H = cfg.hidden_size;
	let by_size = {};

	function make_items(S) {
		let items = [];

		for (let i = 0; i < nl; i++) {
			let L = model.layers[i];
			let last = length(items) ? items[length(items) - 1] : null;

			if (last && last.dev == L.dev)
				last.end = i + 1;
			else
				push(items, {first: i, end: i + 1, dev: L.dev});
		}

		function body(it) {
			let h = it.h_in;

			for (let i = it.first; i < it.end; i++) {
				if (model.ples[i])
					ple_apply(cfg, model.ples[i], h, it.emb, cache.ple);

				layer_forward(cfg, model.layers[i], cache.layers[i], h, null, null, 0, it.st);
			}

			it.h_out = h;
		}

		for (let it in items) {
			let dv = 'cuda:' + it.dev;

			it.h_in = torch.zeros([S, cfg.hc_mult, H], 'float32', dv);
			it.st = dev_state(cfg, S, cache.max_seq, it.dev);
			it.st.pack.copy_(torch.from_array(dev_values(it.st, 0, null), 'int64', dv));

			for (let i = it.first; i < it.end; i++)
				if (model.ples[i])
					it.emb = torch.zeros([1, S, model.ples[i].heads * 160], 'float16', dv);

			body(it);                               // eager warm-up
		}

		torch.synchronize();

		for (let it in items) {
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

	function run_ids(ids, pos) {
		let S = length(ids);
		let items = by_size[S];
		let streams = qwen35.embed_rows(model, cache.mm, ids, pos).view([S, 1, H]).to('float32')
			.expand([S, cfg.hc_mult, H]).contiguous();
		let dev = model.devs[0];
		let vals = dev_values(items[0].st, pos, cache.mm);

		for (let it in items) {
			if (it.dev != dev) {
				streams = streams.cuda(it.dev);
				dev = it.dev;
			}

			it.st.pack.copy_(torch.from_array(vals, 'int64', 'cuda:' + it.dev));

			if (it.emb)
				for (let i = it.first; i < it.end; i++)
					if (model.ples[i])
						it.emb.copy_(ple_prepare(cfg, model.ples[i], ids, cache.ple));

			it.h_in.copy_(streams);
			it.graph.replay();
			streams = it.h_out;
		}

		cache.length = pos + S;

		let hid = head_hidden(model, streams);

		cache.hidden = flat_stack(model, streams);

		return {logits: model.lm_head.forward(hid.view([1, S, H])), hidden: cache.hidden};
	}

	return {
		has: function (S) {return by_size[S] != null;},
		run: function (ids, pos) {return run_ids(ids, pos);}
	};
}

/* ------------------------------------------------------------------ */
/* MTP draft head                                                      */
/*                                                                     */
/* (exllamav3 qwen4_exp_mtp.py, stream-tap form.) Input per draft token: the  */
/* target's PRE-mixer stream stack (4 x hidden, rounded to fp16 like the     */
/* reference's export) and the token embedding:                              */
/*   streams_i = fc_hidden(rms(stack_i) * (1 + w_h)) + fc_embedding(          */
/*               rms(embed(tok)) * (1 + w_e))                                */
/* then ONE trunk-style full-attention block (own KV + indexer cache) and the  */
/* head's own combine-less mixer; the shared lm_head gives the logits. The   */
/* block's output stack feeds the next draft step.                           */

export function build_mtp(ws, ws_patch, cfg, dev) {
	let H = cfg.hc_mult, D = cfg.hidden_size;

	if (ws.gguf) {
		return {
			dev: dev,
			norm_h: ws.gpu('mtp.pre_fc_norm_hidden.weight', dev).to('float32').view([H, D]).contiguous(),
			norm_e: ws.gpu('mtp.pre_fc_norm_embedding.weight', dev).to('float16'),
			eh_proj: ggufq.linear(ws, 'mtp.eh_proj', dev),
			layer: build_layer_at(ws, cfg, 'mtp.layers.0', 'full_attention', dev, null),
			mixer: build_gr(ws, cfg, 'mtp.hyper_connection_mixer', dev, false)
		};
	}

	return {
		dev: dev,
		norm_h: ws.gpu('mtp.pre_fc_norm_hidden.weight', dev).to('float32').add(1.0).view([H, D]).contiguous(),
		norm_e: ws.gpu('mtp.pre_fc_norm_embedding.weight', dev).to('float16').add(1.0),
		fc_hidden: models.linear(ws, 'mtp.fc_hidden', true, dev),
		fc_embedding: models.linear(ws, 'mtp.fc_embedding', true, dev),
		layer: build_layer_at(ws, cfg, 'mtp.layers.0', 'full_attention', dev, null),
		mixer: build_gr(ws_patch, cfg, 'mtp.hyper_connection_mixer', dev, false)
	};
}

/* Draft cache of the MTP block (a full-attention layer cache entry). */
export function new_mtp_cache(cfg, max_seq, dev) {
	let dv = 'cuda:' + dev;
	let rc = qwen35.rope_cache(cfg, max_seq, dev);

	return {
		type: 'full_attention', dev: dev, length: 0,
		...kv_alloc(cfg, max_seq, dv),
		raw_k: torch.zeros([max_seq, cfg.indexer_head_dim], 'float16', dv),
		pooled: torch.zeros([int(max_seq / cfg.indexer_compress_ratio) + 1,
			cfg.indexer_head_dim], 'float16', dv),
		cos: rc.cos, sin: rc.sin
	};
}

/* One draft step for S rows at draft positions pos..pos+S-1: token ids (array),
 * `stack` (S, 4, D) fp32 (the target's pre-mixer stack, or the previous draft
 * step's output). Returns {stack: (S, 4, D) fp32 (next step's input, rounded to
 * fp16 like the reference), hidden: (S, D) fp16 (mixer output), logits (1, S, V)
 * when `head` is given}. */
/* The draft head's input combine: embedding rows emb (1, S, D) fp16 and the previous
 * stack (S, 4, D) fp32 (both on M.dev) -> the block's input streams (S, 4, D) fp32. */
export function mtp_input(cfg, M, emb, S, stack) {
	let H = cfg.hc_mult, D = cfg.hidden_size;
	let eps = cfg.rms_norm_eps;

	stack = stack.cuda(M.dev);

	let ms = stack.mul(stack).mean(-1, true).add(eps).rsqrt();
	let normed = stack.mul(ms).mul(M.norm_h).to('float16');                       // (S, H, D)

	if (M.eh_proj) {
		// GGUF MTP fuses the embedding and hidden projections into one 2D -> D linear per stream
		let e = qwen35.rmsnorm_bias1(emb, M.norm_e, eps);                       // (1, S, D)
		let ee = e.view([S, 1, D]).expand([S, H, D]);                          // (S, H, D)
		let inp = torch.cat([ ee, normed ], -1);                               // (S, H, 2D)
		let h = M.eh_proj.forward(inp.view([1, S * H, 2 * D])).view([S, H, D]);

		return h.to('float32').contiguous();
	}

	let e = M.fc_embedding.forward(qwen35.rmsnorm_bias1(emb, M.norm_e, eps));          // (1, S, D)
	let h = M.fc_hidden.forward(normed.view([1, S * H, D])).view([S, H, D]);

	if (MTP_TAP)
		return h.to('float32').add(e.to('float32').view([S, 1, D])).contiguous();

	return h.to('float32').mean(1).add(e.to('float32').view([S, D])).view([S, 1, D])
		.expand([S, H, D]).contiguous();
}

/* The draft step on the embedding rows emb (1, S, D) fp16 (on the draft device). */
export function mtp_core(model, M, emb, S, stack, dc, pos, head) {
	let cfg = model.cfg;
	let D = cfg.hidden_size;
	let streams = mtp_input(cfg, M, emb, S, stack);

	layer_forward(cfg, M.layer, dc, streams, dc.cos.narrow(0, pos, S), dc.sin.narrow(0, pos, S), pos, null,
		qwen35.rope_rows(cfg, dc.mm, dc, pos, S));

	let hid = gr_mix(cfg, M.mixer, streams)[1];
	let r = {stack: streams.to('float16').to('float32'), hidden: hid};

	if (head)
		r.logits = head.forward(hid.view([1, S, D]));

	return r;
}

export function mtp_step(model, M, ids, stack, dc, pos, head) {
	let S = length(ids);
	let emb = qwen35.embed_rows(model, dc.mm, ids, pos);

	return mtp_core(model, M, (model.devs[0] != M.dev) ? emb.cuda(M.dev) : emb, S, stack, dc, pos, head);
}

/* opts: devices (number of GPUs to spread the layers over, or one index per
 * layer; default all). */
export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let ws = weights.open(model_dir);
	let devs = dev_list(cfg, opts.devices);

	LAYER_DEV = devs;

	let nl = opts.layers || cfg.num_hidden_layers;    // partial models for tests
	let partial = nl < cfg.num_hidden_layers;
	let first_dev = devs[0], last_dev = devs[cfg.num_hidden_layers - 1];

	ws.upload(function (key) {
		if (index(key, LP) == 0) {
			let rest = substr(key, length(LP));
			let idx = +substr(rest, 0, index(rest, '.'));

			if (any_cold() && index(rest, '.mlp.exps.') >= 0)
				return null;

			return (idx >= 0 && idx < nl) ? devs[idx] : null;
		}

		if (index(key, 'model.language_model.embed_tokens') == 0)
			return first_dev;

		if (!partial && (index(key, 'model.language_model.hyper_connection_mixer') == 0 ||
		    index(key, 'lm_head') == 0))
			return last_dev;

		return null;
	});

	let embed_tokens = ws.gpu16('model.language_model.embed_tokens.weight', first_dev);
	ws.release('model.language_model.embed_tokens.weight', first_dev);
	let layers = [], ples = {};
	let mtp = null;

	// the MTP head is opt-in (opts.mtp = true). EXL3 packs carry it in the main
	// checkpoint or a mixer patch; GGUF packs carry it in a separate mtp-*.gguf
	// shard (blk.48.nextn.*), mapped to the mtp.* keys by ggufw.
	if (!partial && opts.mtp == true && (ws.has('mtp.fc_hidden.trellis') || ws.has('mtp.eh_proj.weight'))) {
		let in_main = ws.has('mtp.hyper_connection_mixer.hc_norm.weight');
		let pw = in_main ? ws : weights.open(model_dir, 'mtp_hyper_connection_mixer_patch.safetensors');

		ws.upload(function (key) {
			if (index(key, 'mtp.') != 0)
				return null;

			// the MTP experts are placed by build_moe (resident subset + mapped host cold)
			if (expert_resident_fraction(null) < 1.0 && index(key, '.mlp.exps.') >= 0)
				return null;

			return last_dev;
		});

		if (!in_main)
			pw.upload((key) => last_dev);

		mtp = build_mtp(ws, pw, cfg, last_dev);
	}

	for (let i = 0; i < nl; i++) {
		if (index(cfg.ple_layer_ids, i + 1) >= 0)
			ples[i] = build_ple(ws, cfg, i, devs[i]);

		push(layers, build_layer(ws, cfg, i, devs[i]));
	}

	let model = {
		cfg: cfg, layers: layers, devs: devs, ples: ples, nl: nl, mtp: mtp,
		hidden_width: cfg.hc_mult * cfg.hidden_size,      // the draft head's hidden: the flattened stack
		embed_tokens: embed_tokens,
		mixer: partial ? null :
			build_gr(ws, cfg, 'model.language_model.hyper_connection_mixer', last_dev, false),
		lm_head: partial ? null : ((ws.gguf && ws.qtype('lm_head.weight')) ? ggufq.linear(ws, 'lm_head', last_dev) : models.linear(ws, 'lm_head', true, last_dev)),

		new_cache: function (max_seq) {
			let cfg = this.cfg;
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i in this.ples)
				cache.ple = new_ple_state(cfg, this.ples[i]);

			if (this.mtp)
				cache.mtp = new_mtp_cache(cfg, max_seq, this.mtp.dev);
			let rope_by_dev = {};

			for (let i = 0; i < this.nl; i++) {
				let L = this.layers[i];
				let cl = {type: L.type, dev: L.dev};
				let dv = 'cuda:' + L.dev;

				if (L.type == 'linear_attention') {
					cl.conv_state = torch.zeros([1, cfg.fdim_qkv, cfg.conv_kernel - 1], 'float16', dv);
					cl.rec_state = torch.zeros([cfg.ln_num_v_heads, cfg.ln_k_head_dim,
						cfg.ln_v_head_dim], 'float32', dv);
				} else {
					for (let k, x in kv_alloc(cfg, max_seq, dv))
						cl[k] = x;

					cl.raw_k = torch.zeros([max_seq, cfg.indexer_head_dim], 'float16', dv);
					cl.pooled = torch.zeros([int(max_seq / cfg.indexer_compress_ratio) + 1,
						cfg.indexer_head_dim], 'float16', dv);

					if (!rope_by_dev[L.dev])
						rope_by_dev[L.dev] = qwen35.rope_cache(cfg, max_seq, L.dev);

					cl.cos = rope_by_dev[L.dev].cos;
					cl.sin = rope_by_dev[L.dev].sin;
				}

				push(cache.layers, cl);
			}

			return cache;
		},

		/* Run a chunk of token ids appending to the cache. Returns logits
		 * (1, S, vocab) fp16 on the head device; cache.hidden = the collapsed
		 * stream stack (S, hidden) fp16. */
		forward_chunk: function (token_ids, cache) {
			let S = length(token_ids);
			let streams = run_layers(this, token_ids, cache, null);

			cache.stack = streams;                 // pre-mixer stack (MTP draft input)

			if (!this.mixer)
				return null;

			let hid = head_hidden(this, streams);

			cache.hidden = flat_stack(this, streams);   // the draft head's input

			return this.lm_head.forward(hid.view([1, S, this.cfg.hidden_size]));
		},

		/* Engine interface (as glm5/qwen35 prefill_pipeline): wavefront prefill of
		 * ids at cache.length over the device stages — every ROUND advances the
		 * chunk of each busy stage by one layer, so all GPUs work and the host
		 * (one thread) keeps every queue fed. opts: chunk, cuts (extra chunk
		 * boundaries), on_layer(chunk_end, li) after every layer of a chunk (the
		 * engine snapshots the GDN states of a boundary chunk there), on_aux(
		 * chunk_end) right after the chunk passed the PLE layer, hidden /
		 * on_hidden(pos, H) (post-mixer rows, MTP draft input). logits() are the
		 * last row. */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let cfg = this.cfg;
			let nl = this.nl;
			let H = cfg.hidden_size;
			let chunk = opts.chunk || 1024;
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
				let cids = slice(ids, pos - start, end - start);

				prev_end = end;
				active[0] = {pos: pos, T: T, end: end, ids: cids, layer: stages[0].first, ctx: null,
					h: qwen35.embed_rows(model, cache.mm, cids, pos).to('float32')
						.reshape([T, 1, H]).expand([T, cfg.hc_mult, H]).contiguous()};
			}

			function round() {
				// phase A of every busy stage first: the GPUs work while the host
				// reads the expert counts of the earlier stages in phase B
				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null) {
						let i = c.layer, cl = cache.layers[i];

						if (model.ples[i]) {
							ple_forward(cfg, model.ples[i], c.h, c.ids, cache.ple);

							if (opts.on_aux)
								opts.on_aux(c.end);
						}

						c.ctx = layer_a(cfg, model.layers[i], cl, c.h,
							cl.cos ? cl.cos.narrow(0, c.pos, c.T) : null,
							cl.sin ? cl.sin.narrow(0, c.pos, c.T) : null, c.pos, null,
							cl.cos ? qwen35.rope_rows(cfg, cache.mm, cl, c.pos, c.T) : null);
					}
				}

				for (let s = 0; s < ns; s++) {
					let c = active[s];

					if (c && c.ctx) {
						layer_b(cfg, model.layers[c.layer], c.h, c.ctx);
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
						if (opts.hidden || opts.on_hidden) {
							let hid = flat_stack(model, c.h);

							if (opts.on_hidden)
								opts.on_hidden(c.pos, hid);
							else
								push(parts, hid);
						}

						if (c.end == start + total && opts.logits != false)
							result = model.lm_head.forward(head_hidden(model,
								c.h.narrow(0, c.T - 1, 1).contiguous()).view([1, 1, H]));

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

				// (1, n, 4 * hidden) pre-mixer stacks of every new row (opts.hidden)
				hidden: function () {
					if (!length(parts))
						return null;

					return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
				}
			};
		},

		graphs: function (cache, sizes) {
			return build_graphs(this, cache, sizes);
		},

		/* The PLE layer is recurrent state outside the per-layer cache entries:
		 * the engine snapshots / restores / resets it with the GDN states. The
		 * conv tail is a persistent tensor (CUDA graphs capture the layer), so a
		 * snapshot clones it. */
		aux_get: function (cache) {
			let c = torch.zeros(cache.ple.conv.shape(), 'float16', 'cuda:' + cache.ple.dev);

			c.copy_(cache.ple.conv);

			return {conv: c, ids: slice(cache.ple.ids)};
		},

		aux_set: function (cache, o) {
			cache.ple.conv.copy_(o.conv);
			cache.ple.ids = slice(o.ids);
		},

		/* MTP draft step (engine interface): token ids, the target's (or the previous
		 * draft step's) flattened stack (1, S, 4D) fp16, draft cache, draft position. */
		mtp_chunk: function (ids, hidden, dc, pos, head) {
			let S = length(ids);
			let r = mtp_step(this, this.mtp, ids, hidden.reshape([S, this.cfg.hc_mult,
				this.cfg.hidden_size]).to('float32'), dc, pos, head);

			return {logits: r.logits, hidden: flat_stack_of(this, r.stack)};
		},

		/* Speculative verify rollback of the PLE conv tail: per accepted-prefix length a,
		 * the destination (live state) and source (state after a + 1 rows) tensors the
		 * engine restores with copy_many next to the GDN states. Allocated before the
		 * graphs are captured. */
		spec_bufs: function (cache, W) {
			let out = [];

			if (!cache.ple)
				return out;

			cache.ple.spec_w = W;
			cache.ple.spec_conv = [];

			for (let a = 0; a < W - 1; a++) {
				let c = torch.zeros(cache.ple.conv.shape(), 'float16', 'cuda:' + cache.ple.dev);

				push(cache.ple.spec_conv, c);
				push(out, {d: [ cache.ple.conv ], s: [ c ]});
			}

			return out;
		},

		/* The id context of the PLE hashing follows the accepted token history. */
		aux_ids: function (cache, ids) {
			let n = length(ids), ctx = this.cfg.ngram_size - 1, out = [];

			for (let i = n - ctx; i < n; i++)
				push(out, (i >= 0) ? ids[i] : this.cfg.eos_token_id);

			cache.ple.ids = out;
		},

		aux_reset: function (cache) {
			cache.ple.conv.fill_(0);

			for (let i = 0; i < length(cache.ple.ids); i++)
				cache.ple.ids[i] = this.cfg.eos_token_id;
		}
	};

	// Optional: after loading, drop the model files' page cache (a third copy
	// of bytes already in VRAM/host RAM). Keep the PLE table's file: it is still
	// streamed row by row.
	if (getenv('AMP_QWEN4_DROP_FILE_CACHE') == '1' && ws.gguf) {
		let ple_file = ws.file('ple_table.weight');

		for (let key in ws.list()) {
			let f = ws.file(key);

			if (f !== ple_file)
				f.drop_cache();
		}
	}

	model.prof = PROFILER;

	// move the experts the router picks most into VRAM (qwen4_prof.uc: rebalance), between two requests
	model.rebalance = !PROFILER ? null : function (opts) {
		let Ms = [];

		for (let i = 0; i < nl; i++)
			Ms[i] = model.layers[i].moe;

		return PROFILER.rebalance(Ms, ggufq.swap_experts, ggufq.sync_tables, opts);
	};

	return model;
}
