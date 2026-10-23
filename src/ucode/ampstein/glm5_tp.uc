// ucode/glm5_tp.uc — tensor/expert-parallel GLM-5.3-Flash over N GPUs,
// optionally as a PIPELINE OF TENSOR-PARALLEL STAGES.
//
// One stage = a group of ranks that all run every layer of the stage's
// contiguous layer range, each on its share, and the ranks' partial sublayer
// outputs are summed by an all-reduce; the residual streams (4 x fp32 per
// token) are replicated bit-identically inside the group. A decode token
// then reads 1/Gs of a stage's weights per GPU, the group's GPUs at once —
// the layer split (glm5.uc) reads all of them, one GPU at a time.
//
//   opts.stages = 1  (default)   one group over all devices: plain TP
//   opts.stages = S              S groups of N/S devices, layers split over
//                                the groups (the hand-off between stages is
//                                the replicated stream, a few KB per token)
//
// Why stages: the MLA latent cache is replicated inside a group, so with
// S stages every GPU holds the cache of ~1/S of the MLA layers instead of
// all of them (what makes a 256k context fit), and a stage's all-reduces
// span only its Gs GPUs. Decode latency is the sum of the stages' times;
// prefill is a wavefront over the stages (prefill_pipeline).
//
// Sharding inside a group (rank r's share of each split dimension is cut by
// largest remainder to the alignment the EXL3 slicing needs):
//   KDA        q|k|v by head pairs (conv taps, dt_bias, decay, the f/g/b
//              gate projections by head), o_proj by ROWS (partial sum)
//   MLA        q_b / W_UK / W_UV by head, o_proj by rows; q_a, kv_a, the
//              indexer and the latent cache are replicated
//   dense MLP  gate/up columns, down rows
//   MoE        expert parallel: experts [e0, e1) per rank (the fused MoE
//              kernel filters the router's picks to its range), the
//              shared expert sharded like a dense MLP; the router is
//              replicated
//   lm_head    by vocabulary columns over the LAST stage's ranks, logits
//              gathered on its rank 0
//   norms, mHC parameters   replicated; embedding: first stage, rank 0
//
// Usage:
//   let m = glm5_tp.load(dir, {devices: [0, 1, 2, 3, 4, 5]});         // TP-6
//   let m = glm5_tp.load(dir, {devices: 6, stages: 3, max_seq: 262144}); // 3 x TP-2
//   let cache = m.new_cache(max_seq);
//   let logits = m.forward_chunk(ids, cache, rows);

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as qwen35 from './qwen35.uc';
import {stat} from 'fs';

const LP = glm5.LP;

/* Cut `total` (a multiple of `unit`) into per-rank [start, length]
 * ranges proportional to weights w (largest remainder). */
function split_n(total, unit, w) {
	let n = total / unit, G = length(w), sum = 0;

	for (let x in w)
		sum += x;

	let cnt = [], rem = [], used = 0;

	for (let g = 0; g < G; g++) {
		let exact = n * w[g] / sum;
		let c = int(exact);

		push(cnt, c);
		push(rem, [ exact - c, g ]);
		used += c;
	}

	rem = sort(rem, (a, b) => (b[0] - a[0]) || (a[1] - b[1]));

	for (let k = 0; used < n; k++, used++)
		cnt[rem[k % G][1]]++;

	let out = [], at = 0;

	for (let g = 0; g < G; g++) {
		push(out, [ at * unit, cnt[g] * unit ]);
		at += cnt[g];
	}

	return out;
}

function plan_of(cfg, w) {
	return {
		weights: w,
		kda: split_n(cfg.kda.num_heads, 2, w),               // heads [first, count]
		mla: split_n(cfg.num_q_heads, 1, w),
		mlp: split_n(cfg.intermediate_size, 256, w),
		shared: split_n(cfg.moe_intermediate_size * cfg.n_shared_experts, 256, w),
		experts: split_n(cfg.n_routed_experts, 1, w),         // experts [first, count]
		vocab: split_n(cfg.vocab_size, 256, w)
	};
}

/* Full-width copy of an EXL3 linear on another device. */
function copy_lin(lin, dev) {
	return models.cols(lin, [ [ 0, lin.out_features ] ], dev);
}

/* gate/up by columns, down by rows over the intermediate range ri. */
function shard_mlp(M, ri, dev) {
	return {
		gate: models.cols(M.gate, [ ri ], dev),
		up: models.cols(M.up, [ ri ], dev),
		down: models.rows(M.down, ri[0], ri[1], dev)
	};
}

/* head ranges of a per-head or per-channel vector */
function head_slice(t, H, hd, h0, n) {
	let per = t.numel() / H;       // 1 (per head) or hd (per channel)

	return t.narrow(0, h0 * per, n * per).contiguous();
}

function shard_kda(K, plan, g, dev) {
	let hd = K.head_dim, H = K.heads;
	let h0 = plan.kda[g][0], n = plan.kda[g][1];
	let D = H * hd;
	let a = h0 * hd, len = n * hd;

	if (n == 0)
		die('glm5_tp: a rank without KDA heads (too many ranks)');

	return {
		type: 'kda', dev: dev, heads: n, head_dim: hd, kernel: K.kernel,
		lower_bound: K.lower_bound,
		qkv: models.cols(K.qkv, [ [ a, len ], [ D + a, len ], [ 2 * D + a, len ] ], dev),
		o: models.rows(K.o, a, len, dev),
		conv_w: torch.cat([ K.conv_w.narrow(0, a, len), K.conv_w.narrow(0, D + a, len),
			K.conv_w.narrow(0, 2 * D + a, len) ], 0).contiguous().cuda(dev),
		b_t: K.b_t.narrow(1, h0, n).contiguous().cuda(dev),
		fa_t: K.fa_t.cuda(dev),
		fb_t: K.fb_t.narrow(1, a, len).contiguous().cuda(dev),
		ga_t: K.ga_t.cuda(dev),
		gb_t: K.gb_t.narrow(1, a, len).contiguous().cuda(dev),
		decay: head_slice(K.decay, H, hd, h0, n).cuda(dev),
		dt_bias: head_slice(K.dt_bias, H, hd, h0, n).cuda(dev),
		o_norm: K.o_norm.cuda(dev)
	};
}

function shard_mla(cfg, A, plan, g, dev) {
	let h0 = plan.mla[g][0], n = plan.mla[g][1];
	let nope = cfg.qk_nope_head_dim, vd = cfg.v_head_dim;

	if (n == 0)
		die('glm5_tp: a rank without MLA heads (too many ranks)');

	return {
		type: 'mla', dev: dev,
		idx_wq_b: copy_lin(A.idx_wq_b, dev),
		idx_wk_t: A.idx_wk_t.cuda(dev),
		idx_w_t: A.idx_w_t.cuda(dev),
		idx_norm_w: A.idx_norm_w.cuda(dev),
		idx_norm_b: A.idx_norm_b.cuda(dev),
		idx_gate_t: A.idx_gate_t.cuda(dev),
		idx_ape: A.idx_ape.cuda(dev),
		q_a: copy_lin(A.q_a, dev),
		q_a_norm: A.q_a_norm.cuda(dev),
		q_b: models.cols(A.q_b, [ [ h0 * nope, n * nope ] ], dev),
		kv_a: copy_lin(A.kv_a, dev),
		kv_a_norm: A.kv_a_norm.cuda(dev),
		o: models.rows(A.o, h0 * vd, n * vd, dev),
		w_uk: A.w_uk.narrow(0, h0, n).contiguous().cuda(dev),
		w_uv_t: A.w_uv_t.narrow(0, h0, n).contiguous().cuda(dev)
	};
}

function hc_to(hc, dev) {
	return {fn: hc.fn.cuda(dev), base: hc.base.cuda(dev), scale: hc.scale.cuda(dev)};
}

/* All-reduce of one fp16 partial per rank of stage S -> per-rank sums
 * (bit-identical). Small payloads through the stage's P2P communicator
 * (decode), larger ones by peer copies summed in fp32 in rank order. */
function allreduce(S, p) {
	let G = length(p);

	// timing experiment only (WRONG numerics): skip the reduction entirely
	if (getenv('AMP_GLM5_TP_FAKE_AR'))
		return p;

	if (p[0].numel() <= S.comm_cap)
		return map(p, (x, g) => S.comm.allreduce(x.contiguous(), g));

	if (S.comm_pf && p[0].numel() <= S.comm_pf_cap)
		return map(p, (x, g) => S.comm_pf.allreduce(x.contiguous(), g));

	let out = [];

	for (let g = 0; g < G; g++) {
		let dev = S.devs[g];
		let acc = p[0].cuda(dev).to('float32');

		for (let r = 1; r < G; r++)
			acc = acc.add(p[r].cuda(dev).to('float32'));

		push(out, acc.to('float16'));
	}

	return out;
}

/* Asynchronous form of the prefill all-reduce, for the double-buffered
 * pipeline: the P2P kernel of micro-chunk m runs on a SIDE stream of every
 * rank's device (fork slot 2 + m, its own communicator S.comm_ov[m] so the
 * two micro-chunks' reductions never share staging buffers) while the compute
 * stream goes on with the other micro-chunk. The handle keeps the partials
 * and the outputs alive; ar_end makes the compute stream wait for the
 * reduction and returns the per-rank sums. Payloads that do not fit the
 * micro-chunk communicator reduce synchronously (allreduce). */
function ar_begin(S, p, m, sync) {
	let h = {S: S, m: m, p: p, outs: null, async: false};

	if (S.comm_ov && p[0].numel() <= S.comm_ov_cap && !sync) {
		let outs = [];

		for (let g = 0; g < length(p); g++) {
			torch.stream_side(S.devs[g], 2 + m);
			push(outs, S.comm_ov[m].allreduce(p[g].contiguous(), g));
			torch.stream_main(S.devs[g], 2 + m);
		}

		h.outs = outs;
		h.async = true;
	}
	else {
		h.outs = allreduce(S, p);
	}

	return h;
}

function ar_end(h) {
	if (h.async)
		for (let g = 0; g < length(h.outs); g++)
			torch.stream_wait(h.S.devs[g], 2 + h.m);

	h.p = null;

	return h.outs;
}

/* The MTP draft head (mtp.safetensors, layer index = num_hidden_layers):
 * enorm / hnorm / eh_proj in front of ONE plain residual block (no mHC) —
 * MLA + MoE exactly like a trunk layer, sharded the same way — and
 * shared_head.norm before the (shared, sharded) lm_head. eh_proj and the
 * norms are replicated. Experts go straight to their owners. devs / plan /
 * expert_dev are the owning stage's. */
function load_mtp(model_dir, cfg, plan, devs, expert_dev) {
	let G = length(devs);
	let ws = weights.open(model_dir, 'mtp.safetensors');
	let st = devs[0];
	let nl = cfg.num_hidden_layers;
	let pfx = `${LP}${nl}.`;

	ws.upload(function (key) {
		if (index(key, pfx) != 0)
			return null;

		let at = index(key, '.experts.');

		if (at >= 0) {
			let rest = substr(key, at + 9);

			return expert_dev[+substr(rest, 0, index(rest, '.'))];
		}

		return st;
	});

	let attn = glm5.build_mla(ws, cfg, nl, st);
	let mkey = `${pfx}mlp`;
	let router = {
		gate_t: ws.gpu(mkey + '.gate.weight', st).to('float32').t().contiguous(),
		bias: ws.gpu(mkey + '.gate.e_score_correction_bias', st).to('float32')
			.reshape([1, -1]).contiguous()
	};
	let shared = glm5.build_mlp(ws, mkey + '.shared_experts', st);
	let eh = models.linear(ws, pfx + 'eh_proj', false, st);
	let ranks = [];

	for (let g = 0; g < G; g++) {
		let dev = devs[g];
		let e0 = plan.experts[g][0], en = plan.experts[g][1];
		let ex = [];

		for (let e = e0; e < e0 + en; e++)
			push(ex, glm5.build_mlp(ws, `${mkey}.experts.${e}`, dev));

		push(ranks, {
			dev: dev,
			attn: shard_mla(cfg, attn, plan, g, dev),
			input_layernorm: ws.gpu16(pfx + 'input_layernorm.weight', st).cuda(dev),
			post_attention_layernorm: ws.gpu16(pfx + 'post_attention_layernorm.weight', st).cuda(dev),
			enorm: ws.gpu16(pfx + 'enorm.weight', st).cuda(dev),
			hnorm: ws.gpu16(pfx + 'hnorm.weight', st).cuda(dev),
			fnorm: ws.gpu16(pfx + 'shared_head.norm.weight', st).cuda(dev),
			eh: copy_lin(eh, dev),
			mlp: {
				type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en,
				gate_t: router.gate_t.cuda(dev), bias: router.bias.cuda(dev),
				shared: shard_mlp(shared, plan.shared[g], dev)
			}
		});
	}

	return {dev: devs[0], ranks: ranks};
}

/* largest all-reduce payload in rows through the P2P kernel */
const P2P_ROWS = 16;

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = glm5.load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = cfg.num_hidden_layers;
	let devs = opts.devices;

	if (devs == null || type(devs) != 'array') {
		devs = [];

		for (let d = 0; d < (opts.devices || torch.device_count()); d++)
			push(devs, d);
	}

	let G = length(devs);
	let NS = opts.stages || 1;

	// opts.own = [first, end): the stages of the (global) pipeline this process builds. The
	// others live on another host (docs/RPC.md): their layers are not loaded, and `devices`
	// lists only the local GPUs, Gs per owned stage. Every host plans the whole pipeline
	// (stage_of) the same way; the coordinator hands its stage_end to the workers.
	let own = opts.own || [ 0, NS ];
	let n_own = own[1] - own[0];
	let remote = opts.remote;              // rpc_stage.client: the worker that owns the stages before `own` (stage 0 remote)

	if (own[0] < 0 || n_own < 1 || own[1] > NS)
		die(`glm5_tp: stages ${own[0]}..${own[1]} out of ${NS}`);

	if (G % n_own)
		die(`glm5_tp: ${G} devices do not split into ${n_own} stages`);

	let Gs = G / n_own;                    // ranks per stage
	let has_mtp = (opts.mtp != false && own[1] == NS && stat(model_dir + '/mtp.safetensors'));

	// per-rank shares inside a stage. Rank 0 of the first stage also holds
	// the embedding table (1.2 GiB) and the gathered logits / staging: shed
	// that much of its shard so every rank ends up with the same footprint
	// (measured: rank 0 sat 1.8 GiB above the others with equal shares,
	// which made it the first to run out when the replicated latent cache
	// was allocated)
	function stage_weights(s) {
		if (opts.weights)
			return opts.weights;

		let w = [];

		for (let r = 0; r < Gs; r++)
			push(w, 1);

		// (with several stages the first one is the lightest in memory and
		// the shed share would only make rank 0 wait for its partner at
		// every all-reduce: measured 14% less compute on rank 0)
		if (s == 0 && Gs > 1 && NS == 1) {
			let total = 0;

			for (let key in ws.list())
				if (index(key, 'model.visual') != 0)
					total += ws.info(key).nbytes;

			let per_rank = total / (G * 1073741824.0);
			let w0 = 1.0 - 1.8 / per_rank;

			w[0] = (w0 > 0.5) ? w0 : 0.5;
		}

		return w;
	}

	// layers -> stages: contiguous runs, balanced by per-GPU footprint
	// (sharded weights + the replicated cache at opts.max_seq + the MTP
	// block, which lives with the last stage)
	let stage_of = [];

	for (let i = 0; i < nl; i++)
		push(stage_of, 0);

	if (NS > 1) {
		let mtp_bytes = 0;

		if (has_mtp) {
			let mw = weights.open(model_dir, 'mtp.safetensors');

			for (let key in mw.list())
				mtp_bytes += mw.info(key).nbytes;
		}

		let pl = glm5.plan_devices(ws, cfg, NS, {max_seq: opts.max_seq, kv_bits: opts.kv_bits,
			mtp_bytes: mtp_bytes, mtp_at: NS - 1, shard: Gs});
		let line = '';

		stage_of = pl.devs;

		// opts.stage_end: explicit stage boundaries (the layer after each
		// stage's last one, e.g. [ 14, 30 ] for 3 stages) — the memory-
		// balanced plan above ignores that MLA layers and KDA layers cost
		// different amounts of time per chunk
		if (opts.stage_end) {
			let at = 0;

			for (let i = 0; i < nl; i++) {
				while (at < length(opts.stage_end) && i >= opts.stage_end[at])
					at++;

				stage_of[i] = at;
			}
		}

		for (let s = 0; s < NS; s++)
			line += sprintf(' %.1f', pl.load[s] / 1073741824.0);

		let bounds = '';

		for (let i = 1; i < nl; i++)
			if (stage_of[i] != stage_of[i - 1])
				bounds += ' ' + i;

		warn(`glm5_tp: ${NS} stages x ${Gs} ranks, planned GiB per GPU:${line}; stages start at layers 0${bounds}\n`);
	}

	// opts.plan_only: just the plan, before anything is uploaded. A coordinator sends it as
	// `stage_end` to the workers, which then plan nothing themselves (docs/RPC.md)
	if (opts.plan_only) {
		let ends = [];

		for (let i = 1; i < nl; i++)
			if (stage_of[i] != stage_of[i - 1])
				push(ends, i);

		return {stage_of: stage_of, stage_end: ends, stages: NS};
	}

	let stages = [];
	let cap = P2P_ROWS * cfg.hidden_size;
	let cap_pf = (opts.prefill_rows || 1024) * cfg.hidden_size;

	for (let s = 0; s < NS; s++) {
		if (s < own[0] || s >= own[1]) {
			push(stages, {id: s, remote: true, G: Gs, first: -1, end: -1});
			continue;
		}

		let sd = slice(devs, (s - own[0]) * Gs, (s - own[0] + 1) * Gs);
		let plan = plan_of(cfg, stage_weights(s));
		let S = {
			id: s, devs: sd, G: Gs, plan: plan, first: -1, end: -1, expert_dev: {},
			cg: map(sd, (d, r) => ({...cfg, num_q_heads: plan.mla[r][1]})),
			comm_cap: cap, comm_pf_cap: cap_pf
		};

		for (let r = 0; r < Gs; r++)
			for (let e = plan.experts[r][0]; e < plan.experts[r][0] + plan.experts[r][1]; e++)
				S.expert_dev[e] = sd[r];

		push(stages, S);
	}

	for (let i = 0; i < nl; i++) {
		let S = stages[stage_of[i]];

		if (S.first < 0)
			S.first = i;

		S.end = i + 1;
	}

	for (let S in stages)
		if (S.first < 0)
			die('glm5_tp: a stage without layers');

	let layers = [];
	let t_load = time();

	for (let i = 0; i < nl; i++) {
		let S = stages[stage_of[i]];

		if (S.remote) {
			push(layers, null);
			continue;
		}

		let plan = S.plan;
		let st = S.devs[i % Gs];
		let pfx = `${LP}${i}.`;

		// experts straight to their owner, everything else to the
		// layer's staging device (sliced into the shards below)
		ws.upload(function (key) {
			if (index(key, pfx) != 0)
				return null;

			let at = index(key, '.experts.');

			if (at >= 0) {
				let rest = substr(key, at + 9);

				return S.expert_dev[+substr(rest, 0, index(rest, '.'))];
			}

			return st;
		});

		let kind = cfg.layer_types[i];
		let moe = (cfg.mlp_layer_types[i] != 'dense');
		let attn = (kind == 'linear_attention') ? glm5.build_kda(ws, cfg, i, st) :
			glm5.build_mla(ws, cfg, i, st);
		let hc_a = glm5.build_hc(ws, i, 'attn', st);
		let hc_f = glm5.build_hc(ws, i, 'ffn', st);
		let in_ln = ws.gpu16(`${pfx}input_layernorm.weight`, st);
		let post_ln = ws.gpu16(`${pfx}post_attention_layernorm.weight`, st);
		let mkey = `${pfx}mlp`;
		let dense = moe ? null : glm5.build_mlp(ws, mkey, st);
		let router = null, shared = null;

		if (moe) {
			router = {
				gate_t: ws.gpu(mkey + '.gate.weight', st).to('float32').t().contiguous(),
				bias: ws.gpu(mkey + '.gate.e_score_correction_bias', st).to('float32')
					.reshape([1, -1]).contiguous()
			};
			shared = glm5.build_mlp(ws, mkey + '.shared_experts', st);
		}

		let row = [];

		for (let g = 0; g < Gs; g++) {
			let dev = S.devs[g];
			let R = {
				dev: dev, kind: kind, moe: moe,
				hc_attn: hc_to(hc_a, dev), hc_ffn: hc_to(hc_f, dev),
				input_layernorm: in_ln.cuda(dev),
				post_attention_layernorm: post_ln.cuda(dev),
				attn: (kind == 'linear_attention') ? shard_kda(attn, plan, g, dev) :
					shard_mla(cfg, attn, plan, g, dev)
			};

			if (!moe) {
				R.mlp = shard_mlp(dense, plan.mlp[g], dev);
			}
			else {
				let e0 = plan.experts[g][0], en = plan.experts[g][1];
				let ex = [];

				for (let e = e0; e < e0 + en; e++)
					push(ex, glm5.build_mlp(ws, `${mkey}.experts.${e}`, dev));

				R.mlp = {
					type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en,
					gate_t: router.gate_t.cuda(dev), bias: router.bias.cuda(dev),
					shared: shard_mlp(shared, plan.shared[g], dev)
				};
			}

			push(row, R);
		}

		push(layers, row);

		// the staging copies are not needed any more
		for (let key in ws.list())
			if (index(key, pfx) == 0 && index(key, '.experts.') < 0)
				ws.release(key, st);

		attn = dense = shared = router = null;

		if (i % 4 == 3)
			gc();
	}

	gc();

	let first = stages[0], last = stages[NS - 1];
	let own_first = !first.remote, own_last = !last.remote;

	// embedding on the first stage's rank 0; lm_head sharded by vocabulary
	// over the last stage
	ws.upload(function (key) {
		if (own_first && index(key, 'model.language_model.embed_tokens') == 0)
			return devs[0];

		if (own_last && (index(key, 'model.language_model.norm') == 0 || index(key, 'lm_head') == 0))
			return last.devs[0];

		return null;
	});

	let embed_t = null, lm_heads = null, norm_w = null;

	if (own_first) {
		embed_t = ws.gpu16('model.language_model.embed_tokens.weight', devs[0]);

		ws.release('model.language_model.embed_tokens.weight', devs[0]);
	}

	if (own_last) {
		let head_full = models.linear(ws, 'lm_head', false, last.devs[0]);

		lm_heads = map(last.devs, (d, g) => models.cols(head_full, [ last.plan.vocab[g] ], d));
		norm_w = ws.gpu16('model.language_model.norm.weight', last.devs[0]);
		head_full = null;
	}

	gc();

	// the draft block lives with the last stage (its logits come from the
	// same sharded head, and one multi-GPU graph covers just that group)
	let mst = last;
	let mtp = null;

	if (has_mtp) {
		mtp = load_mtp(model_dir, cfg, mst.plan, mst.devs, mst.expert_dev);
		gc();
	}

	let own_layers = [ stages[own[0]].first, stages[own[1] - 1].end ];

	warn(sprintf('glm5_tp: %d ranks (%d stage%s%s) loaded in %.0fs%s\n', G, n_own, n_own > 1 ? 's' : '',
		n_own < NS ? sprintf(' of %d: stages %d..%d, layers %d..%d', NS, own[0], own[1] - 1, own_layers[0], own_layers[1] - 1) : '',
		time() - t_load, mtp ? ' (with the MTP head)' : ''));

	for (let S in stages) {
		if (S.remote)
			continue;

		S.comm = torch.p2p_comm(S.devs, cap);
		// prefill chunks: a second communicator sized for the largest chunk
		S.comm_pf = getenv('AMP_GLM5_TP_COPY_AR') ? null : torch.p2p_comm(S.devs, cap_pf);

		// double-buffered prefill (EXPERIMENTAL, opt-in: load opts.overlap or AMP_GLM5_TP_OVERLAP=1; measured
		// slower, see docs/LESSONS.md L94): one communicator per micro-chunk (half a chunk)
		S.comm_ov_cap = int(((opts.prefill_rows || 1024) + 1) / 2) * cfg.hidden_size;
		S.comm_ov = (getenv('AMP_GLM5_TP_COPY_AR') || !(opts.overlap || getenv('AMP_GLM5_TP_OVERLAP'))) ? null :
			[ torch.p2p_comm(S.devs, S.comm_ov_cap), torch.p2p_comm(S.devs, S.comm_ov_cap) ];
	}

	let eps = cfg.rms_norm_eps;
	let H = cfg.hidden_size;

	return {
		cfg: cfg, layers: layers, devs: devs, G: G, tp: true,
		stages: stages, stage_of: stage_of, last: last,
		own: own, own_layers: own_layers,   // the stages / layers [first, end) built here (all of them unless split over hosts)
		comm: first.comm, comm_cap: cap,
		embed_tokens: embed_t,
		norm: own_last ? map(last.devs, (d) => norm_w.cuda(d)) : null,
		lm_heads: lm_heads,
		lm_head: lm_heads ? lm_heads[0] : null,     // engine: "the model has a head" (logits come from the shards)
		mtp: mtp, kv_bits: opts.kv_bits,
		spec_any: true,           // the step graphs keep per-position KDA states for any window up to the engine's largest (glm5.kda_forward)

		mm_build: glm5.mm_build,

		/* KDA state holders of layer i: one per rank of its stage */
		states: function (cache, i) {
			return cache.layers[i];
		},

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};
			let kvb = this.kv_bits;

			for (let i = 0; i < nl; i++) {
				let S = stages[stage_of[i]];
				let row = [];

				if (S.remote) {
					push(cache.layers, null);
					continue;
				}

				for (let g = 0; g < Gs; g++) {
					let dev = S.devs[g];

					if (cfg.layer_types[i] == 'linear_attention') {
						let la = cfg.kda, n = S.plan.kda[g][1];

						push(row, {
							dev: dev,
							conv_state: torch.zeros([1, 3 * n * la.head_dim,
								la.short_conv_kernel_size - 1], 'float16', 'cuda:' + dev),
							rec_state: torch.zeros([n, la.head_dim, la.head_dim],
								'float32', 'cuda:' + dev)
						});
					}
					else
						push(row, glm5.new_mla_cache(cfg, max_seq, dev, kvb));
				}

				push(cache.layers, row);
			}

			if (this.mtp) {
				cache.mtp = {length: 0, max_seq: max_seq,
					layers: map(mst.devs, (d) => glm5.new_mla_cache(cfg, max_seq, d, kvb))};
			}

			// the layers of the remote first stages keep their caches on the worker
			if (first.remote && remote)
				cache.remote = remote.new_cache(max_seq);

			return cache;
		},

		/* zero the recurrent states of the layers built here (and of the remote stages): a fresh
		 * context, e.g. after graphs() clobbered the cache with its warm-up */
		reset_cache: function (cache) {
			for (let i = 0; i < nl; i++)
				for (let cl in (cache.layers[i] ?? []))
					if (cl.rec_state) {
						cl.rec_state.fill_(0);
						cl.conv_state.fill_(0);
					}

			cache.length = 0;

			if (cache.remote != null)
				remote.reset(cache.remote);
		},

		/* release the cache of the remote stages (the local ones go with the object) */
		free_cache: function (cache) {
			if (cache.remote != null) {
				remote.free(cache.remote);
				cache.remote = null;
			}
		},

		/* The replicated streams of stage `from`, copied to every rank of
		 * stage `to`: rank r receives its replica from rank r of the sender,
		 * so the copies use both senders' links (all from rank 0 would put
		 * 2 x 200 MB per 2048-token chunk through one x4 link). */
		handoff: function (h, from, to) {
			return map(stages[to].devs, (d, r) => h[r].cuda(d));
		},

		/* One layer on all ranks of its stage, in two halves so a pipelined
		 * prefill can enqueue the attention of every stage before any stage
		 * reads its MoE counts: layer_a runs the attention site and enqueues
		 * the ffn up to the routing; layer_b finishes the ffn (may read the
		 * expert counts from the device). h: the stage's per-rank replicated
		 * streams (S, 4, hidden) fp32 (updated in place); st: per-rank
		 * device-position states (graph capture) or null. */
		layer_a: function (i, h, cache, pos, st) {
			let S = stages[stage_of[i]];
			let cg = S.cg;
			let L = this.layers[i];
			let cls = cache.layers[i];
			let a = [], p = [];

			for (let g = 0; g < Gs; g++) {
				let r = glm5.hc_mix(cg[g], L[g].hc_attn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].input_layernorm, eps);

				push(a, r);
				push(p, (L[g].kind == 'linear_attention') ?
					glm5.kda_forward(cg[g], L[g].attn, xn, cls[g]) :
					st ? glm5.mla_forward_dev(cg[g], L[g].attn, xn, cls[g], st[g]) :
					glm5.mla_forward(cg[g], L[g].attn, xn, cls[g], pos));
			}

			let s = allreduce(S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(h[g], s[g], a[g][0], a[g][1]);

			// ffn site: phase A on every rank
			let f = [], pend = [];

			for (let g = 0; g < Gs; g++) {
				let r = glm5.hc_mix(cg[g], L[g].hc_ffn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].post_attention_layernorm, eps);

				push(f, r);
				push(pend, L[g].moe ? glm5.moe_begin(cg[g], L[g].mlp, xn) :
					{done: glm5.mlp_forward(cg[g], L[g].mlp, xn)});
			}

			return {S: S, L: L, cg: cg, h: h, f: f, pend: pend};
		},

		layer_b: function (c) {
			let L = c.L, p = [];

			for (let g = 0; g < Gs; g++) {
				let y = L[g].moe ? glm5.moe_finish(c.cg[g], L[g].mlp, c.pend[g]) : c.pend[g].done;

				push(p, y.to('float16'));
			}

			let s = allreduce(c.S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(c.h[g], s[g], c.f[g][0], c.f[g][1]);

			return c.h;
		},

		layer: function (i, h, cache, pos, st) {
			return this.layer_b(this.layer_a(i, h, cache, pos, st));
		},

		/* Double-buffered layer for a chunk split into micro-chunks
		 * (c.hs[m]: per-rank streams, c.pos[m]: cache position). The
		 * reduction of one micro-chunk's partials runs on a side stream
		 * while the compute stream works on the other micro-chunk:
		 *
		 *   layer_pair_a:  [apply prev ffn(0)] attn(0) ->AR  [apply prev ffn(1)] attn(1) ->AR
		 *                  <AR(0)> apply, ffn up to routing (0)   <AR(1)> same (1)
		 *   layer_pair_b:  ffn finish (0) ->AR, ffn finish (1) ->AR   (c.ar: applied by the
		 *                  next layer's pair_a, or pair_flush at the end of a stage)
		 *
		 * Micro-chunk 1 follows 0 in position, and every layer runs 0 before 1,
		 * so the KDA state and the MLA cache dependencies hold. */
		layer_pair_a: function (i, c, cache) {
			let S = stages[stage_of[i]];
			let cg = S.cg;
			let L = this.layers[i];
			let cls = cache.layers[i];
			let n = length(c.hs);
			let ha = [], a = [], ctx = [];

			for (let m = 0; m < n; m++) {
				if (c.ar)
					this.pair_apply(c.ar[m], c.hs[m]);

				let am = [], pm = [];

				for (let g = 0; g < Gs; g++) {
					let r = glm5.hc_mix(cg[g], L[g].hc_attn, c.hs[m][g]);
					let xn = torch.rms_norm(r[2], L[g].input_layernorm, eps);

					push(am, r);
					push(pm, (L[g].kind == 'linear_attention') ?
						glm5.kda_forward(cg[g], L[g].attn, xn, cls[g]) :
						glm5.mla_forward(cg[g], L[g].attn, xn, cls[g], c.pos[m]));
				}

				push(a, am);
				push(ha, ar_begin(S, pm, m, c.sync));
			}

			c.ar = null;

			for (let m = 0; m < n; m++) {
				let s = ar_end(ha[m]);

				for (let g = 0; g < Gs; g++)
					glm5.hc_apply(c.hs[m][g], s[g], a[m][g][0], a[m][g][1]);

				let f = [], pend = [];

				for (let g = 0; g < Gs; g++) {
					let r = glm5.hc_mix(cg[g], L[g].hc_ffn, c.hs[m][g]);
					let xn = torch.rms_norm(r[2], L[g].post_attention_layernorm, eps);

					push(f, r);
					push(pend, L[g].moe ? glm5.moe_begin(cg[g], L[g].mlp, xn) :
						{done: glm5.mlp_forward(cg[g], L[g].mlp, xn)});
				}

				push(ctx, {f: f, pend: pend});
			}

			return ctx;
		},

		layer_pair_b: function (i, c, ctx) {
			let S = stages[stage_of[i]];
			let cg = S.cg;
			let L = this.layers[i];
			let ar = [];

			for (let m = 0; m < length(ctx); m++) {
				let p = [];

				for (let g = 0; g < Gs; g++) {
					let y = L[g].moe ? glm5.moe_finish(cg[g], L[g].mlp, ctx[m].pend[g]) :
						ctx[m].pend[g].done;

					push(p, y.to('float16'));
				}

				let h = ar_begin(S, p, m, c.sync);

				h.f = ctx[m].f;
				push(ar, h);
			}

			c.ar = ar;
		},

		/* add a micro-chunk's pending ffn reduction into its streams */
		pair_apply: function (h, hm) {
			let s = ar_end(h);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(hm[g], s[g], h.f[g][0], h.f[g][1]);
		},

		/* the last layer of a stage: apply the pending ffn reductions */
		pair_flush: function (c) {
			if (!c.ar)
				return;

			for (let m = 0; m < length(c.hs); m++)
				this.pair_apply(c.ar[m], c.hs[m]);

			c.ar = null;
		},

		/* CUDA-graph decode steps (sizes <= P2P_ROWS tokens): per size TWO
		 * multi-GPU graphs captured on the first device with all the other
		 * devices as side devices — dense MLA (context <= index_topk) and
		 * sparse (the k-pool selection). Positions live in device buffers;
		 * the all-reduces need no cross-device graph edges (only the stage
		 * hand-offs and the start-of-step position broadcast do). Clobbers
		 * the cache (eager warm-up): call before any prefill, then zero the
		 * states. Returns {has(S), run(ids, pos) -> {logits (1, V) on the
		 * last stage's rank 0}}. */
		graphs: function (cache, sizes) {
			let tp = this;

			// Split over hosts (docs/RPC.md phase 5): the graph of a process covers its own stages.
			// rem_in: the first stages are remote, the graph starts from persistent input buffers
			// (hin, filled from the worker's streams before each replay); rem_out: the last stages
			// are remote (this is the worker), the graph ends with the streams of its last stage
			let rem_in = first.remote, rem_out = last.remote;
			let from = own_layers[0], to = own_layers[1];

			if (rem_in && !remote)
				die('glm5_tp: the first stages are remote: a client is needed for graphs');

			// (the worker captures its graphs first: their warm-up clobbers its cache like ours)
			if (rem_in)
				remote.graphs(cache.remote, sizes);

			let P = cfg.index_kpool;
			let by_size = {};
			let pool = null;     // the graphs replay one at a time: one shared private pool

			for (let n = 0; n < length(sizes); n++) {
				let S = sizes[n];

				if (S > P2P_ROWS)
					continue;

				let ent = {S: S, ids: rem_in ? null : torch.zeros([S], 'int64', 'cuda:' + devs[0]), bar: [],
					hin: rem_in ? map(stages[stage_of[from]].devs, (d) => torch.zeros([S, 4, H], 'float32', 'cuda:' + d)) : null};

				for (let g = 0; g < G; g++)
					push(ent.bar, torch.zeros([1], 'float32', 'cuda:' + devs[g]));

				for (let variant = 0; variant < ((cache.max_seq > cfg.index_topk) ? 2 : 1); variant++) {
					let sparse = (variant == 1);
					let st = [];

					for (let g = 0; g < G; g++) {
						let dv = 'cuda:' + devs[g];
						let sg = int(g / Gs), r = g % Gs;
						let o = {
							sparse: sparse, max_rows: cache.max_seq,
							pos: torch.zeros([1], 'int64', dv),
							ramp: torch.arange(S, 'int64', dv),
							arP: torch.arange(P, 'float32', dv),
							arT: torch.arange(P - 1, 'float32', dv)
						};


						push(st, o);
					}

					let body = function () {
						// the other ranks take the position from rank 0
						for (let g = 1; g < G; g++)
							st[g].pos.copy_(st[0].pos);

						for (let g = 0; g < G; g++)
							glm5.mla_dev_prepare(cfg, st[g], devs[g]);

						let h, cur = stage_of[from];

						if (rem_in)
							h = ent.hin;
						else {
							let h0 = tp.embed_tokens.index(ent.ids).to('float32')
								.reshape([S, 1, H]).expand([S, 4, H]).contiguous();

							h = map(first.devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));
						}

						for (let i = from; i < to; i++) {
							if (stage_of[i] != cur) {
								h = tp.handoff(h, cur, stage_of[i]);
								cur = stage_of[i];
							}

							h = tp.layer(i, h, cache, 0, slice(st, (cur - own[0]) * Gs, (cur - own[0] + 1) * Gs));
						}

						ent['out' + variant] = rem_out ? {h: h[0]} : tp.head(h, S, true);
					};

					st[0].pos.fill_(0);
					body();
					torch.synchronize();

					let gr = torch.graph();

					gr.capture_begin(devs[0], slice(devs, 1), pool);
					glm5.set_par(!getenv('AMP_GLM5_NO_PAR'));
					body();
					glm5.set_par(false);
					gr.capture_end();

					if (!pool && !getenv("AMP_GLM5_NO_SHARED_POOL"))
						pool = gr.pool();
					torch.synchronize();

					ent['graph' + variant] = gr;
					ent['st' + variant] = st;
				}

				by_size[S] = ent;
			}

			let barrier = function (ent) {
				for (let g = 1; g < G; g++)
					ent.bar[g].copy_(ent.bar[0]);
			};

			let replay = function (ent, pos) {
				let S = ent.S;
				let v = (pos + S > cfg.index_topk) ? 1 : 0;

				ent['st' + v][0].pos.fill_(pos);

				barrier(ent);     // rank 0 waits for prior work on the others
				ent['graph' + v].replay();
				barrier(ent);     // the others wait for the replay
				cache.length = pos + S;

				let o = ent['out' + v];

				if (rem_out)
					return {h: o.h};

				cache.hidden = o.hidden.view([1, S, H]);

				return {logits: o.logits, hidden: cache.hidden};
			};

			let mg = (tp.mtp && cache.mtp) ? tp.mtp_graph(cache) : null;

			tp.cache_mtp_graph = mg;

			return {
				has: function (S) {
					return by_size[S] != null;
				},

				run: function (ids, pos) {
					let S = length(ids);
					let ent = by_size[S];

					if (!ent)
						die(`glm5_tp: no graph captured for step size ${S}`);

					if (rem_in) {
						// the worker's graph computes the streams of its last stage; they land here
						// in the connection's pinned buffer and go to every rank of our first stage
						let r = remote.forward(cache.remote, pos, ids, true);

						for (let g = 0; g < Gs; g++)
							ent.hin[g].copy_(r.h);
					}
					else
						ent.ids.copy_(torch.from_array(ids, 'int64', 'cuda:' + devs[0]));

					return replay(ent, pos);
				},

				/* ids_dev: int64 (1, S) on the first device (the draft chain's
				 * device-side token ids) */
				run_dev: function (ids_dev, pos) {
					if (rem_in)
						die('glm5_tp: run_dev needs the token ids on this host (the first stages are remote)');

					let S = ids_dev.shape()[1];
					let ent = by_size[S];

					if (!ent)
						die(`glm5_tp: no graph captured for step size ${S}`);

					ent.ids.copy_(ids_dev.view([S]));

					return replay(ent, pos);
				},

				mtp: mg
			};
		},

		/* The MTP block on all ranks of its stage: embs / hids are per-rank
		 * (T, hidden) fp16 (token embeddings and the target's post-norm
		 * hidden of the previous positions); dc the draft cache; st the
		 * device-position state of a graph capture (else host pos). Returns
		 * the per-rank post-norm draft hidden (T, hidden) fp16. */
		mtp_core: function (embs, hids, dc, pos, st) {
			let M = this.mtp.ranks;
			let cg = mst.cg;
			let x = [], p = [];

			for (let g = 0; g < Gs; g++) {
				let e = torch.rms_norm(embs[g], M[g].enorm, eps);
				let y = torch.rms_norm(hids[g], M[g].hnorm, eps);

				push(x, M[g].eh.forward(torch.cat([e, y], -1)).to('float32'));
			}

			for (let g = 0; g < Gs; g++) {
				let r = glm5.rms_norm_f32(x[g], M[g].input_layernorm, eps);

				push(p, st ? glm5.mla_forward_dev(cg[g], M[g].attn, r, dc.layers[g], st[g]) :
					glm5.mla_forward(cg[g], M[g].attn, r, dc.layers[g], pos));
			}

			let s = allreduce(mst, p);
			let pend = [];

			for (let g = 0; g < Gs; g++) {
				x[g] = x[g].add(s[g].to('float32'));

				push(pend, glm5.moe_begin(cg[g], M[g].mlp,
					glm5.rms_norm_f32(x[g], M[g].post_attention_layernorm, eps)));
			}

			for (let g = 0; g < Gs; g++)
				p[g] = glm5.moe_finish(cg[g], M[g].mlp, pend[g]).to('float16');

			s = allreduce(mst, p);

			return map(x, (v, g) => glm5.rms_norm_f32(v.add(s[g].to('float32')), M[g].fnorm, eps));
		},

		/* Draft logits of per-rank draft hiddens D: the sharded head,
		 * gathered on the stage's rank 0 -> (T, V) */
		mtp_logits: function (D) {
			let ls = [];

			for (let g = 0; g < Gs; g++) {
				let l = this.lm_heads[g].forward(D[g]);

				push(ls, (g == 0) ? l : l.cuda(last.devs[0]));
			}

			return torch.cat(ls, -1);
		},

		/* CUDA-graph draft step (one token): dense and sparse variants,
		 * persistent embedding / hidden input buffers (filled by mtp_chunk
		 * before the replay; the graph covers the draft stage only). */
		mtp_graph: function (cache) {
			let tp = this;
			let P = cfg.index_kpool;
			let dc = cache.mtp;
			let md = mst.devs;
			let pool = null;     // both variants share one private pool
			let ent = {
				emb: torch.zeros([1, H], 'float16', 'cuda:' + md[0]),
				hid: torch.zeros([1, H], 'float16', 'cuda:' + md[0]),
				bar: map(md, (d) => torch.zeros([1], 'float32', 'cuda:' + d))
			};

			for (let variant = 0; variant < ((cache.max_seq > cfg.index_topk) ? 2 : 1); variant++) {
				let sparse = (variant == 1);
				let st = [];

				for (let g = 0; g < Gs; g++) {
					let dv = 'cuda:' + md[g];
					let o = {
						sparse: sparse, max_rows: dc.max_seq,
						pos: torch.zeros([1], 'int64', dv),
						ramp: torch.arange(1, 'int64', dv),
						arP: torch.arange(P, 'float32', dv),
						arT: torch.arange(P - 1, 'float32', dv)
					};

					push(st, o);
				}

				let body = function () {
					for (let g = 1; g < Gs; g++)
						st[g].pos.copy_(st[0].pos);

					for (let g = 0; g < Gs; g++)
						glm5.mla_dev_prepare(cfg, st[g], md[g]);

					let embs = map(md, (d, g) => (g == 0) ? ent.emb : ent.emb.cuda(d));
					let hids = map(md, (d, g) => (g == 0) ? ent.hid : ent.hid.cuda(d));
					let D = tp.mtp_core(embs, hids, dc, 0, st);

					ent['out' + variant] = {logits: tp.mtp_logits(D), hidden: D[0]};
				};

				st[0].pos.fill_(0);
				body();
				torch.synchronize();

				let gr = torch.graph();

				gr.capture_begin(md[0], slice(md, 1), pool);
				glm5.set_par(!getenv('AMP_GLM5_NO_PAR'));
				body();
				glm5.set_par(false);
				gr.capture_end();

				if (!pool && !getenv("AMP_GLM5_NO_SHARED_POOL"))
					pool = gr.pool();
				torch.synchronize();

				ent['graph' + variant] = gr;
				ent['st' + variant] = st;
			}

			return ent;
		},

		/* Draft step(s) for the engine: token ids (array) at draft
		 * positions pos.., hidden (1, n, hidden) the target's post-norm
		 * hidden of the PREVIOUS positions. head: truthy -> logits.
		 * Returns {logits (1, n, V) | null, hidden (1, n, hidden)} on the
		 * draft stage's first device. */
		mtp_chunk: function (token_ids, hidden, dc, pos, head, ids_dev) {
			let n = ids_dev ? ids_dev.shape()[1] : length(token_ids);
			let mg = this.cache_mtp_graph;
			let md = mst.devs;
			let e0 = ids_dev ? this.embed_tokens.index(ids_dev.view([n])) :
				qwen35.embed_rows(this, dc.mm, token_ids, pos).view([n, H]);

			if (n == 1 && mg && head) {
				let v = (pos + 1 > cfg.index_topk) ? 1 : 0;

				mg.emb.copy_(e0.view([1, H]));
				mg.hid.copy_(hidden.view([1, H]));
				mg['st' + v][0].pos.fill_(pos);

				for (let g = 1; g < Gs; g++)
					mg.bar[g].copy_(mg.bar[0]);

				mg['graph' + v].replay();

				for (let g = 1; g < Gs; g++)
					mg.bar[g].copy_(mg.bar[0]);

				let o = mg['out' + v];

				return {logits: o.logits.view([1, 1, -1]),
					hidden: o.hidden.view([1, 1, H])};
			}

			let h0 = hidden.view([n, H]);
			let D = this.mtp_core(map(md, (d) => e0.cuda(d)), map(md, (d) => h0.cuda(d)),
				dc, pos, null);

			return {logits: head ? this.mtp_logits(D).view([1, n, -1]) : null,
				hidden: D[0].view([1, n, H])};
		},

		mtp_chunk_ids: function (ids_dev, hidden, dc, pos, head) {
			return this.mtp_chunk(null, hidden, dc, pos, head, ids_dev);
		},

		/* Logits of the last `rows` positions (default 1), gathered on the
		 * last stage's rank 0; with want_hidden: {logits (null for rows ==
		 * 0), hidden} where hidden is the post-norm hidden of ALL rows
		 * (S, hidden) there (the MTP head's input). h: the last stage's
		 * per-rank streams. */
		head: function (h, rows, want_hidden) {
			let S = h[0].shape()[0];
			let ls = [];
			let hidden = null;

			if (rows == null)
				rows = 1;

			for (let g = 0; g < last.G; g++) {
				let xn = glm5.rms_norm_f32(h[g].mean(1), this.norm[g], eps);

				if (g == 0)
					hidden = xn;

				if (rows == 0)
					continue;

				let x = (rows < S) ? xn.narrow(0, S - rows, rows).contiguous() : xn;
				let l = this.lm_heads[g].forward(x);

				push(ls, (g == 0) ? l : l.cuda(last.devs[0]));
			}

			let logits = length(ls) ? torch.cat(ls, -1) : null;

			return want_hidden ? {logits: logits, hidden: hidden} : logits;
		},

		embed: function (ids, mm, pos) {
			let T = length(ids);
			let h0 = qwen35.embed_rows(this, mm, ids, pos || 0).to('float32')
				.reshape([T, 1, H]).expand([T, 4, H]).contiguous();

			return map(first.devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));
		},

		/* Forward ids at cache.length (eager). Returns the logits of the
		 * last `rows` positions (default 1); rows == 0: none. */
		forward_chunk: function (ids, cache, rows, on_layer, want_hidden) {
			let pos = cache.length;
			let h, cur = 0, from = 0;

			if (first.remote) {
				// the worker runs the embedding and its layers; what comes back is one replica of
				// the streams of its last stage, copied to every rank of the first local stage
				if (!remote || cache.remote == null)
					die('glm5_tp: the first stages are remote: a client and a cache of it are needed');

				if (cache.mm)
					die('glm5_tp: images are not supported with remote stages');

				let r = remote.forward(cache.remote, pos, ids);

				from = own_layers[0];
				cur = stage_of[from];
				h = map(stages[cur].devs, (d) => r.h.cuda(d));
			}
			else
				h = this.embed(ids, cache.mm, pos);

			for (let i = from; i < nl; i++) {
				if (stage_of[i] != cur) {
					h = this.handoff(h, cur, stage_of[i]);
					cur = stage_of[i];
				}

				h = this.layer(i, h, cache, pos);

				if (on_layer)
					on_layer(i);
			}

			cache.length = pos + length(ids);

			if (want_hidden)
				return this.head(h, rows, true);

			return (rows == 0) ? null : this.head(h, rows);
		},

		/* engine interface: chunked prefill, a wavefront over the stages —
		 * every round each stage advances its chunk by one layer (all
		 * attention halves are enqueued before any stage reads its MoE
		 * counts), so with S stages S chunks are in flight, one per stage */
		prefill_pipeline: function (cache, ids, opts) {
			if (first.remote)
				die('glm5_tp: prefill_pipeline does not support remote stages yet (docs/RPC.md phase 6)');

			let tp = this;
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

			let active = [];
			let ov = (stages[0].comm_ov != null && opts.overlap != false);   // double-buffered micro-chunks

			for (let s = 0; s < NS; s++)
				push(active, null);

			let next = 0, prev_end = start, done = 0, logits = null, parts = [];

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				let end = bounds[next++];
				let pos = prev_end;
				let T = end - pos;

				prev_end = end;

				// double buffering: the chunk travels as two consecutive
				// micro-chunks, interleaved layer by layer (layer_pair_a/b)
				let T0 = (ov && T >= 512 && T <= 2 * int(stages[0].comm_ov_cap / H)) ? int(T / 2) : T;
				let poss = [ pos ], Ts = [ T0 ];

				if (T0 < T) {
					push(poss, pos + T0);
					push(Ts, T - T0);
				}

				active[0] = {pos: poss, Ts: Ts, end: end, layer: stages[0].first, ctx: null, ar: null, sync: opts.ov_sync,
					hs: map(poss, (p, m) => tp.embed(slice(ids, p - start, p - start + Ts[m]), cache.mm, p))};
			}

			function round() {
				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null)
						c.ctx = ov ? tp.layer_pair_a(c.layer, c, cache) :
							tp.layer_a(c.layer, c.hs[0], cache, c.pos[0]);
				}

				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.ctx) {
						if (ov)
							tp.layer_pair_b(c.layer, c, c.ctx);
						else
							c.hs[0] = tp.layer_b(c.ctx);

						c.ctx = null;

						if (opts.on_layer)
							opts.on_layer(c.end, c.layer);

						c.layer++;
					}
				}

				for (let s = NS - 1; s >= 0; s--) {
					let c = active[s];

					if (!c || c.layer < stages[s].end)
						continue;

					if (ov)
						tp.pair_flush(c);

					if (s == NS - 1) {
						let fin = (c.end == start + total);

						for (let m = 0; m < length(c.hs); m++) {
							let lastm = fin && (m == length(c.hs) - 1);
							let r = tp.head(c.hs[m], lastm ? 1 : 0, opts.hidden);
							let lg = r;

							if (opts.hidden) {
								lg = r.logits;

								if (opts.on_hidden)
									opts.on_hidden(c.pos[m], r.hidden.view([1, c.Ts[m], H]));
								else
									push(parts, r.hidden);
							}

							if (lastm)
								logits = lg;
						}

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;
					}
					else if (active[s + 1] == null) {
						c.hs = map(c.hs, (h) => tp.handoff(h, s, s + 1));
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
				logits: function () {return logits;},

				/* (1, n, hidden) post-norm hidden of every new row (opts.hidden
				 * without opts.on_hidden) */
				hidden: function () {
					if (!length(parts))
						return null;

					let all = (length(parts) == 1) ? parts[0] : torch.cat(parts, 0);

					return all.view([1, all.shape()[0], H]);
				}
			};
		},

		/* Prefill ids in chunks (tests / benches). */
		prefill: function (ids, cache, chunk, rows) {
			let n = length(ids);

			chunk = chunk || 512;

			for (let p = 0; p < n; p += chunk) {
				let e = (p + chunk < n) ? p + chunk : n;
				let r = this.forward_chunk(slice(ids, p, e), cache,
					(e == n) ? (rows == null ? 1 : rows) : 0);

				if (e == n)
					return r;
			}
		}
	};
}
