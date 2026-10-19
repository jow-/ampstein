// ucode/qwen4_tp.uc — tensor / expert-parallel Qwen3.8-Flash-Next over N GPUs,
// optionally as a PIPELINE OF TENSOR-PARALLEL STAGES.
//
// One stage = a group of ranks that all run every layer of the stage's contiguous
// layer range, each on its share, and the ranks' partial sublayer outputs are
// summed by an all-reduce; the residual streams (4 x fp32 per token) are
// replicated bit-identically inside the group. A decode token then reads 1/Gs of
// a stage's weights per GPU, the group's GPUs at once — the layer split (qwen4.uc)
// reads all of them, one GPU at a time.
//
//   opts.stages = 1  (default)   one group over all devices: plain TP
//   opts.stages = S              S groups of N/S devices, layers split over the
//                                groups (the hand-off between stages is the
//                                replicated stream, 40 KB per token)
//
// Why stages: the host thread issues every kernel of every rank, and one ucode
// thread cannot feed six GPUs through a prefill (~57k launches per chunk, GPUs
// busy a quarter of the time). A stage launches for Gs ranks only, and the prefill
// is a wavefront over the stages (prefill_pipeline): with S stages S chunks are in
// flight, one per group. Decode is the sum of the stages' times, in ONE multi-GPU
// graph. Plan and architecture: docs/QWEN4.md.
//
// Sharding inside a group (rank r's share of each split dimension is cut by
// largest remainder to the alignment the EXL3 slicing needs):
//   GDN        q|k|v by k-head PAIRS (conv taps, in_proj_z/a/b, dt_bias, A_log by
//              head), out_proj by ROWS (partial sum) — as qwen35_tp
//   attention  the QSA layers run WHOLE on one rank of the group (exllamav3 does the
//              same: the indexer is headless and there are only 2 KV heads); owners
//              rotate through the ranks so weights and caches balance, the other
//              ranks contribute a zero partial
//   MoE        expert parallel: experts [e0, e1) per rank (the fused kernel
//              filters the router's picks to its range); the router is
//              replicated; the shared expert lives on one rank per layer
//   PLE        whole on device 0, which broadcasts the updated streams (stage 0)
//   lm_head    by vocabulary columns over the LAST stage's ranks; logits gathered
//              on its rank 0
//   gated-residual mixes, norms: replicated in the stage; embedding: device 0
// The all-reduce (torch.p2p_comm) sums in fp32 in rank order on every rank, so
// the streams stay bit-identical.
//
//   let m = qwen4_tp.load(dir, {devices: 6});                  // TP-6
//   let m = qwen4_tp.load(dir, {devices: 6, stages: 3});       // 3 stages x TP-2
//   let cache = m.new_cache(max_seq);
//   let logits = m.forward_chunk(ids, cache);

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as qwen35 from './qwen35.uc';
import * as qwen4 from './qwen4.uc';

/* GDN_TRACE: per-layer magnitude of the fp32 streams and of the mixed (fp16) sublayer
 * input of the first rank, for prefill-sized steps. Prints rows whose peak is large
 * or non-finite (first 400 lines). */
const TRACE = getenv('AMP_GDN_TRACE') != null;
let trace_layer = -1, trace_n = 0;

function trace_site(site, h, m, S) {
	let hx = h[0].abs().max().item();
	let x = m[0][1].to('float32');
	let xx = x.abs().max().item();
	let nfin = x.sub(x).abs().max().item() != 0.0;

	if ((nfin || hx != hx || hx > 20000.0 || xx > 20000.0) && trace_n++ < 400)
		warn(sprintf('trace: layer %d %s S=%d stream-max %g mixed-max %g%s\n', trace_layer, site, S, hx, xx,
			nfin ? ' MIXED-NONFINITE' : ''));
}


const LP = qwen4.LP;

/* largest all-reduce payload (rows) through the P2P kernel for decode / verify */
const P2P_ROWS = 8;

/* Cut `total` (a multiple of `unit`) into per-rank [start, length] ranges
 * proportional to weights w (largest remainder). */
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

/* per-rank GDN geometry: this rank's share of the k heads (and their v heads) */
function shard_cfg(cfg, plan, g) {
	let c = {...cfg};
	let hk = plan.gdn_k[g][1];
	let rep = cfg.ln_num_v_heads / cfg.ln_num_k_heads;

	c.ln_num_k_heads = hk;
	c.ln_num_v_heads = hk * rep;
	c.ln_k_dim = hk * cfg.ln_k_head_dim;
	c.ln_v_dim = c.ln_num_v_heads * cfg.ln_v_head_dim;
	c.fdim_qkv = 2 * c.ln_k_dim + c.ln_v_dim;

	return c;
}

/* A gated-residual site on another device. */
function gr_to(G, dev) {
	return {
		dev: dev, inject: G.inject, rank: G.rank,
		w: G.w.cuda(dev), proj_t: G.proj_t.cuda(dev), up_t: G.up_t.cuda(dev),
		fn: G.fn.cuda(dev), upx: G.upx.cuda(dev), w_h: G.w_h.cuda(dev)
	};
}

/* rank g's shard of a GDN layer's weights (L: the full layer on its staging device) */
function shard_gdn(cfg, plan, L, g, dev) {
	let rep = cfg.ln_num_v_heads / cfg.ln_num_k_heads;
	let kd = cfg.ln_num_k_heads * cfg.ln_k_head_dim;          // full q (= k) width
	let k0 = plan.gdn_k[g][0] * cfg.ln_k_head_dim;
	let kq = plan.gdn_k[g][1] * cfg.ln_k_head_dim;
	let v0 = plan.gdn_k[g][0] * rep * cfg.ln_v_head_dim;
	let vh = plan.gdn_k[g][1] * rep * cfg.ln_v_head_dim;
	let h0 = plan.gdn_k[g][0] * rep;
	let hv = plan.gdn_k[g][1] * rep;

	if (!kq)
		die('qwen4_tp: a rank without GDN heads (too many ranks per stage)');

	return {
		type: 'linear_attention', dev: dev,
		in_proj_qkv: models.cols(L.in_proj_qkv, [ [ k0, kq ], [ kd + k0, kq ], [ 2 * kd + v0, vh ] ], dev),
		in_proj_z: models.cols(L.in_proj_z, [ [ v0, vh ] ], dev),
		in_proj_b: L.in_proj_b.trellis ? models.cols(L.in_proj_b, [ [ h0, hv ] ], dev) :
			models.fp16_cols(L.in_proj_b, h0, hv, dev),
		in_proj_a: L.in_proj_a.trellis ? models.cols(L.in_proj_a, [ [ h0, hv ] ], dev) :
			models.fp16_cols(L.in_proj_a, h0, hv, dev),
		out_proj: models.rows(L.out_proj, v0, vh, dev),
		conv_w: torch.cat([ L.conv_w.narrow(0, k0, kq), L.conv_w.narrow(0, kd + k0, kq),
			L.conv_w.narrow(0, 2 * kd + v0, vh) ], 0).contiguous().cuda(dev),
		dt_bias: L.dt_bias.narrow(0, h0, hv).contiguous().cuda(dev),
		neg_exp_a_log: L.neg_exp_a_log.narrow(0, h0, hv).contiguous().cuda(dev),
		norm_w32: L.norm_w32.cuda(dev)
	};
}

/* All-reduce of one fp16 partial per rank of stage S -> per-rank sums, bit-identical.
 * Small payloads through the stage's P2P communicator (decode / verify), prefill chunks
 * through a second one sized for the chunk; anything else by peer copies summed in
 * fp32 in rank order (the same rounding as the kernel). */
function allreduce(S, p) {
	let G = length(p);

	if (p[0].numel() <= S.comm_cap)
		return map(p, (x, g) => S.comm.allreduce(x.contiguous(), g));

	if (S.comm_pf && p[0].numel() <= S.comm_pf_cap)
		return map(p, (x, g) => S.comm_pf.allreduce(x.contiguous(), g));

	let out = [];

	for (let g = 0; g < G; g++) {
		let acc = p[0].cuda(S.devs[g]).to('float32');

		for (let r = 1; r < G; r++)
			acc = acc.add(p[r].cuda(S.devs[g]).to('float32'));

		push(out, acc.to('float16'));
	}

	return out;
}

/* opts: devices (a count, or an array of CUDA devices; default all), stages (groups of
 * devices/stages ranks, default 1), stage_end (explicit stage boundaries: the layer after
 * each stage's last one), layers (first N layers only, no head: for per-block tests),
 * prefill_rows (largest chunk through the P2P all-reduce, default 1024), weights
 * (per-rank GDN shares inside a stage), mtp (true: load the MTP head) */
export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = qwen4.load_config(model_dir);
	let ws = weights.open(model_dir);
	let nl = opts.layers || cfg.num_hidden_layers;
	let partial = nl < cfg.num_hidden_layers;
	let devs = opts.devices;

	if (devs == null || type(devs) != 'array') {
		devs = [];

		for (let d = 0; d < (opts.devices || torch.device_count()); d++)
			push(devs, d);
	}

	let G = length(devs);
	let NS = opts.stages || 1;

	if (G % NS)
		die(`qwen4_tp: ${G} devices do not split into ${NS} stages`);

	let Gs = G / NS;                       // ranks per stage
	let D = cfg.hidden_size, H = cfg.hc_mult;
	let NL = cfg.num_hidden_layers;

	// layers -> stages: contiguous runs (even by default)
	let stage_of = [];

	for (let i = 0; i < NL; i++)
		push(stage_of, int(i * NS / NL));

	if (opts.stage_end) {
		let at = 0;

		for (let i = 0; i < NL; i++) {
			while (at < length(opts.stage_end) && i >= opts.stage_end[at])
				at++;

			stage_of[i] = at;
		}
	}

	let stages = [];

	for (let s = 0; s < NS; s++) {
		let sd = slice(devs, s * Gs, (s + 1) * Gs);
		let w = opts.weights, we = [];

		if (!w) {
			w = [];

			for (let r = 0; r < Gs; r++)
				push(w, 1);
		}

		// device 0 also holds the embedding table (1.3 GiB), the PLE layer and the gathered
		// logits: shed some of its experts
		for (let r = 0; r < Gs; r++)
			push(we, (s == 0 && r == 0 && Gs > 1) ? 0.9 : 1.0);

		let plan = {
			gdn_k: split_n(cfg.ln_num_k_heads, 2, w),                 // k heads [first, count]
			experts: split_n(cfg.n_routed_experts, 1, we),            // experts [first, count]
			vocab: split_n(cfg.vocab_size, 256, w)
		};
		let S = {
			id: s, devs: sd, G: Gs, plan: plan, first: -1, end: -1, expert_rank: [],
			cg: map(sd, (d, r) => shard_cfg(cfg, plan, r))
		};

		for (let r = 0; r < Gs; r++)
			for (let e = plan.experts[r][0]; e < plan.experts[r][0] + plan.experts[r][1]; e++)
				S.expert_rank[e] = r;

		push(stages, S);
	}

	for (let i = 0; i < NL; i++) {
		let S = stages[stage_of[i]];

		if (S.first < 0)
			S.first = i;

		S.end = i + 1;
	}

	for (let S in stages)
		if (S.first < 0)
			die('qwen4_tp: a stage without layers');

	let first = stages[0], last = stages[NS - 1];

	// owners (rank inside the layer's stage): attention layers rotate through the ranks
	// (counting from rank 1 in stage 0: rank 0 has the embedding / PLE), the shared
	// expert of layer i sits on rank (i + 2) % Gs
	let attn_owner = [], shared_owner = [], n_attn = map(stages, () => 0);

	for (let i = 0; i < nl; i++) {
		let s = stage_of[i];

		if (cfg.layer_types[i] == 'full_attention') {
			attn_owner[i] = (n_attn[s] + (s == 0 ? 1 : 0)) % Gs;
			n_attn[s]++;
		}

		shared_owner[i] = (i + 2) % Gs;
	}

	// ---- one batched upload of everything, routed to where it is used (the
	// staged ones — GDN, router, gated-residual tables — are sliced / copied below)
	let by_layer = [];
	let has_mtp = (!partial && opts.mtp == true && ws.has('mtp.fc_hidden.trellis'));
	let mtp_ao = 1 % Gs, mtp_so = 3 % Gs;      // owners of the draft block's attention / shared expert (last stage)
	let by_mtp = [];

	for (let i = 0; i < nl; i++)
		push(by_layer, []);

	ws.upload(function (key) {
		if (index(key, LP) == 0) {
			let rest = substr(key, length(LP));
			let dot = index(rest, '.');
			let i = +substr(rest, 0, dot);

			if (i < 0 || i >= nl)
				return null;

			let S = stages[stage_of[i]];
			let sub = substr(rest, dot + 1);

			if (index(sub, 'mlp.experts.') == 0) {
				let r2 = substr(sub, 12);

				return S.devs[S.expert_rank[+substr(r2, 0, index(r2, '.'))]];
			}

			if (index(sub, 'mlp.shared_expert.') == 0)
				return S.devs[shared_owner[i]];

			if (index(sub, 'self_attn.') == 0)
				return S.devs[attn_owner[i]];

			if (index(sub, 'ple.') == 0)
				return devs[0];

			push(by_layer[i], key);

			return S.devs[i % Gs];
		}

		if (has_mtp && index(key, 'mtp.') == 0) {
			if (index(key, 'mtp.layers.0.mlp.experts.') == 0) {
				let r2 = substr(key, 25);

				return last.devs[last.expert_rank[+substr(r2, 0, index(r2, '.'))]];
			}

			if (index(key, 'mtp.layers.0.mlp.shared_expert.') == 0)
				return last.devs[mtp_so];

			if (index(key, 'mtp.layers.0.self_attn.') == 0)
				return last.devs[mtp_ao];

			push(by_mtp, key);

			return last.devs[0];
		}

		if (index(key, 'model.language_model.embed_tokens') == 0)
			return devs[0];

		if (!partial && (index(key, 'model.language_model.hyper_connection_mixer') == 0 ||
		    index(key, 'lm_head') == 0))
			return last.devs[0];

		return null;
	});

	let t_load = time();
	let layers = [], ples = {};

	/* The per-rank row of one block (trunk layer or the MTP block) of stage S from the
	 * staged weights: sublayer shards (GDN) or the owner's whole attention, replicated
	 * gated-residual sites and router, this rank's experts, the shared expert on its
	 * owner. */
	function make_row(S, key, type_i, ao, so, st, staged) {
		let sd = S.devs;
		let hc_a = qwen4.build_gr(ws, cfg, key + '.attn_hyper_connection', st, true);
		let hc_f = qwen4.build_gr(ws, cfg, key + '.mlp_hyper_connection', st, true);
		let mkey = key + '.mlp';
		let gate_t = ws.gpu(mkey + '.gate.weight', st).to('float32').t().contiguous();
		let sgate_t = ws.gpu(mkey + '.shared_expert_gate.weight', st).to('float32').t().contiguous();
		let shared = glm5.build_mlp(ws, mkey + '.shared_expert', sd[so]);
		let full = null, own = null;

		if (type_i == 'linear_attention')
			full = qwen4.build_sublayer_at(ws, cfg, key, type_i, st);
		else
			own = qwen4.build_sublayer_at(ws, cfg, key, type_i, sd[ao]);

		let row = [];

		for (let g = 0; g < Gs; g++) {
			let dev = sd[g];
			let R = (type_i == 'linear_attention') ? shard_gdn(cfg, S.plan, full, g, dev) :
				(g == ao) ? own : {type: type_i, dev: dev};
			let e0 = S.plan.experts[g][0], en = S.plan.experts[g][1];
			let ex = [];

			for (let e = e0; e < e0 + en; e++)
				push(ex, glm5.build_mlp(ws, `${mkey}.experts.${e}`, dev));

			R.attn_hc = gr_to(hc_a, dev);
			R.mlp_hc = gr_to(hc_f, dev);
			R.moe = {
				type: 'moe', dev: dev, router: 'softmax', experts: ex, e0: e0, e1: e0 + en,
				gate_t: gate_t.cuda(dev),
				shared: (g == so) ? shared : null,
				shared_gate: (g == so) ? sgate_t.cuda(dev) : null
			};

			push(row, R);
		}

		// the staging copies are not needed any more
		for (let k in staged)
			ws.drop(k, st);

		return row;
	}

	for (let i = 0; i < nl; i++) {
		let S = stages[stage_of[i]];

		if (index(cfg.ple_layer_ids, i + 1) >= 0)
			ples[i] = qwen4.build_ple(ws, cfg, i, devs[0]);

		push(layers, make_row(S, LP + i, cfg.layer_types[i], attn_owner[i], shared_owner[i],
			S.devs[i % Gs], by_layer[i]));

		if (i % 4 == 3)
			gc();
	}

	gc();

	// the MTP draft block: a full-attention block like a trunk layer (own cache) on the
	// last stage, its input combine on that stage's rank 0, its own mixer from the
	// separate patch file
	let mtp = null;

	if (has_mtp) {
		// the MTP block's own mixer: in the main checkpoint (4.05 bpw: already uploaded with the other mtp.*
		// tensors) or in a separate patch file (6.05 bpw)
		let in_main = ws.has('mtp.hyper_connection_mixer.hc_norm.weight');
		let pw = in_main ? ws : weights.open(model_dir, 'mtp_hyper_connection_mixer_patch.safetensors');
		let d0 = last.devs[0];

		if (!in_main)
			pw.upload((key) => d0);

		let mx = qwen4.build_gr(pw, cfg, 'mtp.hyper_connection_mixer', d0, false);

		mtp = {
			dev: d0, owner: mtp_ao, S: last,
			norm_h: ws.gpu('mtp.pre_fc_norm_hidden.weight', d0).to('float32').add(1.0).view([H, D]).contiguous(),
			norm_e: ws.gpu('mtp.pre_fc_norm_embedding.weight', d0).to('float16').add(1.0),
			fc_hidden: models.linear(ws, 'mtp.fc_hidden', true, d0),
			fc_embedding: models.linear(ws, 'mtp.fc_embedding', true, d0),
			mixer: map(last.devs, (d) => gr_to(mx, d)),
			row: make_row(last, 'mtp.layers.0', 'full_attention', mtp_ao, mtp_so, d0, by_mtp)
		};

		gc();
	}

	let embed_t = ws.gpu16('model.language_model.embed_tokens.weight', devs[0]);

	ws.drop('model.language_model.embed_tokens.weight', devs[0]);

	let mixer = null, lm_heads = null;

	if (!partial) {
		let ld = last.devs[0];
		let mx = qwen4.build_gr(ws, cfg, 'model.language_model.hyper_connection_mixer', ld, false);

		mixer = map(last.devs, (d) => gr_to(mx, d));

		let head_full = models.linear(ws, 'lm_head', false, ld);

		lm_heads = map(last.devs, (d, g) => models.cols(head_full, [ last.plan.vocab[g] ], d));
		head_full = null;

		for (let sfx in [ 'trellis', 'suh', 'svh', 'mul1' ])
			ws.drop('lm_head.' + sfx, ld);
	}

	gc();

	warn(sprintf('qwen4_tp: %d ranks (%d stage%s) loaded in %.0fs\n', G, NS, NS > 1 ? 's' : '', time() - t_load));

	let cap = P2P_ROWS * D;
	let cap_pf = (opts.prefill_rows || 1024) * D;

	for (let S in stages) {
		S.comm = torch.p2p_comm(S.devs, cap);
		S.comm_cap = cap;
		S.comm_pf = opts.copy_ar ? null : torch.p2p_comm(S.devs, cap_pf);
		S.comm_pf_cap = cap_pf;
	}

	let tp = {
		cfg: cfg, layers: layers, devs: devs, G: G, NS: NS, Gs: Gs, nl: nl, tp: true,
		stages: stages, stage_of: stage_of, last: last,
		ples: ples, mixer: mixer, lm_heads: lm_heads, mtp: mtp,
		lm_head: lm_heads ? lm_heads[0] : null,     // engine: "the model has a head"
		embed_tokens: embed_t,
		attn_owner: attn_owner,
		hidden_width: H * D,

		/* GDN state holders of layer i: one per rank of its stage */
		states: function (cache, i) {
			return cache.layers[i];
		},

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i in this.ples)
				cache.ple = qwen4.new_ple_state(cfg, this.ples[i]);

			function attn_cache(dev) {
				let dv = 'cuda:' + dev;
				let rc = qwen35.rope_cache(cfg, max_seq, dev);

				return {
					k_cache: torch.zeros([max_seq, cfg.num_kv_heads, cfg.head_dim], 'float16', dv),
					v_cache: torch.zeros([max_seq, cfg.num_kv_heads, cfg.head_dim], 'float16', dv),
					raw_k: torch.zeros([max_seq, cfg.indexer_head_dim], 'float16', dv),
					pooled: torch.zeros([int(max_seq / cfg.indexer_compress_ratio) + 1,
						cfg.indexer_head_dim], 'float16', dv),
					cos: rc.cos, sin: rc.sin
				};
			}

			for (let i = 0; i < nl; i++) {
				let S = stages[stage_of[i]];
				let row = [];

				for (let g = 0; g < Gs; g++) {
					let dev = 'cuda:' + S.devs[g];
					let cl = {type: cfg.layer_types[i], dev: S.devs[g]};

					if (cl.type == 'linear_attention') {
						cl.conv_state = torch.zeros([1, S.cg[g].fdim_qkv, cfg.conv_kernel - 1],
							'float16', dev);
						cl.rec_state = torch.zeros([S.cg[g].ln_num_v_heads, cfg.ln_k_head_dim,
							cfg.ln_v_head_dim], 'float32', dev);
					}
					else if (g == attn_owner[i]) {
						let a = attn_cache(S.devs[g]);

						for (let k in a)
							cl[k] = a[k];
					}

					push(row, cl);
				}

				push(cache.layers, row);
			}

			if (this.mtp) {
				let mrow = [];

				for (let g = 0; g < Gs; g++) {
					let cl = {type: 'full_attention', dev: last.devs[g]};

					if (g == this.mtp.owner) {
						let a = attn_cache(last.devs[g]);

						for (let k in a)
							cl[k] = a[k];
					}

					push(mrow, cl);
				}

				cache.mtp = {max_seq: max_seq, length: 0, layers: [ mrow ]};
			}

			return cache;
		},

		/* Phase A of layer i on the ranks of its stage (h: the per-rank replicated
		 * streams (S, 4, D) fp32, updated in place): the attention site with its
		 * all-reduce, then the mlp site up to the routing. st: per-rank device
		 * index state (graph capture) or null. */
		layer_a: function (i, h, cache, pos, st) {
			trace_layer = i;

			return this.row_a(stages[stage_of[i]], this.layers[i], cache.layers[i], attn_owner[i],
				h, pos, st, cache.mm);
		},

		row_a: function (St, L, cls, own, h, pos, st, mm) {
			let S = h[0].shape()[0];
			let cg = St.cg;
			let m = [], p = [];

			for (let g = 0; g < Gs; g++)
				push(m, qwen4.gr_mix(cfg, L[g].attn_hc, h[g]));

			if (TRACE && S >= 8)
				trace_site('attn', h, m, S);

			for (let g = 0; g < Gs; g++) {
				let x = m[g][1].view([1, S, D]);

				if (L[g].type == 'linear_attention')
					push(p, qwen4.gdn_forward(cg[g], L[g], x, cls[g]).view([S, D]));
				else if (g == own) {
					let cl = cls[g];

					push(p, (st ? qwen4.attn_forward_dev(cfg, L[g], x, cl, st[g]) :
						qwen4.attn_forward(cfg, L[g], x, cl.cos.narrow(0, pos, S), cl.sin.narrow(0, pos, S),
							cl, pos, qwen35.rope_rows(cfg, mm, cl, pos, S))).view([S, D]));
				}
				else
					push(p, torch.zeros([S, D], 'float16', 'cuda:' + St.devs[g]));
			}

			let s = allreduce(St, p);

			for (let g = 0; g < Gs; g++)
				torch.hc_apply(h[g], s[g], m[g][0], null);

			let f = [], pend = [];

			for (let g = 0; g < Gs; g++) {
				let fm = qwen4.gr_mix(cfg, L[g].mlp_hc, h[g]);

				push(f, fm);

				if (TRACE && S >= 8 && g == 0)
					trace_site('mlp', h, [ fm ], S);
				push(pend, glm5.moe_begin(cfg, L[g].moe, fm[1]));
			}

			return {S: St, L: L, h: h, f: f, pend: pend};
		},

		layer_b: function (c) {
			let p = [];

			for (let g = 0; g < Gs; g++)
				push(p, glm5.moe_finish(cfg, c.L[g].moe, c.pend[g]).to('float16'));

			let s = allreduce(c.S, p);

			for (let g = 0; g < Gs; g++)
				torch.hc_apply(c.h[g], s[g], c.f[g][0], null);

			return c.h;
		},

		layer: function (i, h, cache, pos, st) {
			return this.layer_b(this.layer_a(i, h, cache, pos, st));
		},

		/* The replicated streams of stage `from`, copied to the ranks of stage `to`:
		 * rank r receives its replica from rank r of the sender */
		handoff: function (h, from, to) {
			return map(stages[to].devs, (d, r) => h[r].cuda(d));
		},

		/* The PLE layer in front of block i (device 0, stage 0), then the updated streams
		 * to the other ranks of the stage. emb: its n-gram embedding when already prepared
		 * (graphs). */
		ple_step: function (i, h, ids, cache, emb) {
			let P = this.ples[i];

			if (emb)
				qwen4.ple_apply(cfg, P, h[0], emb, cache.ple);
			else
				qwen4.ple_forward(cfg, P, h[0], ids, cache.ple);

			for (let g = 1; g < Gs; g++)
				h[g].copy_(h[0]);
		},

		/* the embedding of a chunk as replicated streams of stage 0 */
		embed: function (ids, cache, pos) {
			let S = length(ids);
			let h0 = qwen35.embed_rows(this, cache.mm, ids, pos).view([S, 1, D]).to('float32')
				.expand([S, H, D]).contiguous();

			return map(first.devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));
		},

		/* The final mixer + the sharded head on the last stage's replicated streams. rows:
		 * the number of trailing positions to score (null: all); returns logits gathered
		 * on the stage's rank 0 and the collapsed hidden of that rank. */
		head: function (h, rows) {
			let S = h[0].shape()[0];
			let ls = [];
			let hid0 = null;

			for (let g = 0; g < Gs; g++) {
				let x = (rows != null && rows < S) ? h[g].narrow(0, S - rows, rows).contiguous() : h[g];
				let hid = qwen4.gr_mix(cfg, this.mixer[g], x)[1];
				let n = (rows != null && rows < S) ? rows : S;
				let l = this.lm_heads[g].forward(hid.view([1, n, D]));

				push(ls, (g == 0) ? l : l.cuda(last.devs[0]));

				if (g == 0)
					hid0 = hid;
			}

			return {logits: torch.cat(ls, -1), hidden: hid0};
		},

		/* Forward ids at cache.length (eager). Returns {logits, h} — the logits of
		 * the last `rows` positions (null: all; 0: none). on_layer(li) runs after
		 * each layer. */
		forward: function (ids, cache, rows, on_layer) {
			let S = length(ids);
			let pos = cache.length;
			let h = this.embed(ids, cache, pos);
			let cur = 0;

			for (let i = 0; i < nl; i++) {
				if (stage_of[i] != cur) {
					h = this.handoff(h, cur, stage_of[i]);
					cur = stage_of[i];
				}

				if (this.ples[i])
					this.ple_step(i, h, ids, cache, null);

				h = this.layer(i, h, cache, pos, null);

				if (cache.trace)
					push(cache.trace.blocks, h[0].cpu());

				if (on_layer)
					on_layer(i);
			}

			cache.length = pos + S;
			cache.stack = h[0];

			if (partial || rows === 0)
				return {logits: null, h: h};

			let r = this.head(h, rows);

			cache.hidden = qwen4.flat_stack_of(this, h[0]);

			return {logits: r.logits, h: h};
		},

		forward_chunk: function (ids, cache) {
			return this.forward(ids, cache, null).logits;
		},

		/* Zero the recurrent states and rewind (KV need no clearing). */
		reset: function (cache) {
			for (let i = 0; i < nl; i++)
				for (let g = 0; g < Gs; g++) {
					let cl = cache.layers[i][g];

					if (cl.rec_state) {
						cl.rec_state.fill_(0);
						cl.conv_state.fill_(0);
					}
				}

			this.aux_reset(cache);
			cache.length = 0;
		},

		/* CUDA-graph steps of the given sizes: ONE multi-GPU graph per size covering all
		 * the stages, captured on device 0 with the other ranks as side devices; the
		 * all-reduces need no cross-device graph edges, only the embedding, the PLE
		 * broadcast and the stage hand-offs do. Every position-dependent index comes
		 * from a per-rank packed buffer rewritten before each replay (qwen4.dev_state /
		 * dev_values); the PLE n-gram embedding is hashed on the host into a persistent
		 * buffer. Clobbers the cache (eager warm-up): call before any prefill, then
		 * reset(cache). Returns {has(S), run(ids, pos) -> {logits (1, S, V) on the last
		 * stage's rank 0, hidden}}. */
		graphs: function (cache, sizes) {
			let tp = this;
			let by_size = {};
			let ple_i = -1;

			for (let i in this.ples)
				ple_i = +i;

			for (let S in sizes) {
				if (S > P2P_ROWS)
					continue;

				let st = {
					S: S, ids: torch.zeros([S], 'int64', 'cuda:' + devs[0]),
					dev: map(devs, (d) => qwen4.dev_state(cfg, S, cache.max_seq, d)),
					bar: map(devs, (d) => torch.zeros([1], 'float32', 'cuda:' + d)),
					emb: (ple_i >= 0) ? torch.zeros([1, S, this.ples[ple_i].heads * 160], 'float16',
						'cuda:' + devs[0]) : null
				};

				let put = function (pos) {
					let vals = qwen4.dev_values(st.dev[0], pos, cache.mm);

					for (let g = 0; g < G; g++)
						st.dev[g].pack.copy_(torch.from_array(vals, 'int64', 'cuda:' + devs[g]));
				};

				let body = function () {
					let h0 = tp.embed_tokens.index(st.ids).view([S, 1, D]).to('float32')
						.expand([S, H, D]).contiguous();
					let h = map(first.devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));
					let cur = 0;

					for (let i = 0; i < nl; i++) {
						if (stage_of[i] != cur) {
							h = tp.handoff(h, cur, stage_of[i]);
							cur = stage_of[i];
						}

						if (tp.ples[i])
							tp.ple_step(i, h, null, cache, st.emb);

						h = tp.layer(i, h, cache, 0, slice(st.dev, cur * Gs, (cur + 1) * Gs));
					}

					st.h = h;
					st.out = tp.head(h, null);
				};

				put(0);
				body();
				torch.synchronize();

				st.graph = torch.graph();
				st.graph.capture_begin(devs[0], slice(devs, 1));
				glm5.set_par(!getenv('AMP_QWEN4_NO_PAR'));       // the shared expert beside the router chain
				body();
				glm5.set_par(false);
				st.graph.capture_end();
				torch.synchronize();

				st.put = put;
				by_size[S] = st;
			}

			let barrier = function (st) {
				for (let g = 1; g < G; g++)
					st.bar[g].copy_(st.bar[0]);
			};

			if (tp.mtp && cache.mtp && !getenv('AMP_QWEN4_NO_MTP_GRAPH'))
				tp.mtp_g = tp.mtp_graph(cache);

			return {
				has: function (S) {
					return by_size[S] != null;
				},

				run: function (ids, pos) {
					let S = length(ids);
					let st = by_size[S];

					if (!st)
						die(`qwen4_tp: no graph captured for step size ${S}`);

					st.ids.copy_(torch.from_array(ids, 'int64', 'cuda:' + devs[0]));
					st.put(pos);

					if (st.emb)
						st.emb.copy_(qwen4.ple_prepare(cfg, tp.ples[ple_i], ids, cache.ple));

					barrier(st);     // rank 0 waits for prior work on the others
					st.graph.replay();
					barrier(st);     // the others wait for the replay
					cache.length = pos + S;
					cache.stack = st.h[0];
					cache.hidden = qwen4.flat_stack_of(tp, st.h[0]);

					return {logits: st.out.logits, hidden: cache.hidden};
				}
			};
		},

		/* engine interface: chunked prefill of ids at cache.length, a WAVEFRONT over the
		 * stages — every round each stage advances its chunk by one layer (all attention
		 * halves are enqueued before any stage reads its MoE counts), so with S stages S
		 * chunks are in flight, one per group. opts: chunk, cuts, on_layer(chunk_end, li),
		 * on_aux(chunk_end) after the PLE layer, hidden / on_hidden(pos, H) (the
		 * flattened pre-mixer stacks, the MTP head's input), logits. */
		prefill_pipeline: function (cache, ids, opts) {
			let tp = this;
			let chunk = opts.chunk || 1024;
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let q = start + chunk; q < start + total; q += chunk)
				push(bounds, q);

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			let active = map(stages, () => null);
			let next = 0, prev_end = start, done = 0, result = null, parts = [];

			// the PLE rows of the chunks are read from disk by a worker thread, a few chunks ahead of the wavefront
			let ple_i = -1;

			for (let i in tp.ples)
				ple_i = +i;

			let pjobs = [], tail = (ple_i >= 0 && cache.ple) ? cache.ple.ids : null;
			let ahead = +(getenv('AMP_QWEN4_PLE_AHEAD') || 3);

			function ple_ahead(upto) {
				while (tail && length(pjobs) < length(bounds) && length(pjobs) <= upto) {
					let k = length(pjobs);
					let job = qwen4.ple_start(cfg, tp.ples[ple_i], tail, slice(ids, (k ? bounds[k - 1] : start) - start,
						bounds[k] - start));

					tail = job.next;
					push(pjobs, job);
				}
			}

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				let k = next;
				let end = bounds[next++];
				let pos = prev_end;
				let cids = slice(ids, pos - start, end - start);

				prev_end = end;
				ple_ahead(k + ahead);
				active[0] = {pos: pos, T: end - pos, end: end, ids: cids, layer: stages[0].first, ctx: null,
					h: tp.embed(cids, cache, pos), pj: pjobs[k]};
			}

			function round() {
				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null) {
						if (tp.ples[c.layer]) {
							if (c.pj) {
								tp.ple_step(c.layer, c.h, null, cache, qwen4.ple_finish(cfg, tp.ples[c.layer], c.pj));
								cache.ple.ids = c.pj.next;
								c.pj = null;
							}
							else {
								tp.ple_step(c.layer, c.h, c.ids, cache, null);
							}

							if (opts.on_aux)
								opts.on_aux(c.end);
						}

						c.ctx = tp.layer_a(c.layer, c.h, cache, c.pos, null);
					}
				}

				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.ctx) {
						tp.layer_b(c.ctx);
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

					if (s == NS - 1) {
						if (opts.hidden || opts.on_hidden) {
							let hid = qwen4.flat_stack_of(tp, c.h[0]);

							if (opts.on_hidden)
								opts.on_hidden(c.pos, hid);
							else
								push(parts, hid);
						}

						if (c.end == start + total && opts.logits != false)
							result = tp.head(c.h, 1).logits;

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;
					}
					else if (active[s + 1] == null) {
						c.h = tp.handoff(c.h, s, s + 1);
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
				hidden: function () {
					if (!length(parts))
						return null;

					return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
				}
			};
		},

		/* Per-position rollback of the PLE conv tail for the speculative verify. */
		spec_bufs: function (cache, W) {
			let out = [];

			cache.ple.spec_w = W;
			cache.ple.spec_conv = [];

			for (let a = 0; a < W - 1; a++) {
				let c = torch.zeros(cache.ple.conv.shape(), 'float16', 'cuda:' + cache.ple.dev);

				push(cache.ple.spec_conv, c);
				push(out, {d: [ cache.ple.conv ], s: [ c ]});
			}

			return out;
		},

		/* One draft step (eager) for S rows at draft positions pos..: the combined input
		 * streams on the draft stage's rank 0, replicated, the block on the ranks of that
		 * stage, then rank 0's mixer. stack: (S, 4, D) fp32 on M.dev. Returns {stack
		 * (S,4,D) fp32 rounded to fp16, hidden (S, D) fp16 collapsed} on M.dev. */
		mtp_run_rows: function (emb, S, stack, dc, pos, st) {
			let M = this.mtp;
			let s0 = qwen4.mtp_input(cfg, M, emb, S, stack);
			let h = map(M.S.devs, (d, g) => (g == 0) ? s0 : s0.cuda(d));
			let c = this.row_a(M.S, M.row, dc.layers[0], M.owner, h, pos, st, dc.mm);

			h = this.layer_b(c);

			return {stack: h[0].to('float16').to('float32'), hidden: qwen4.gr_mix(cfg, M.mixer[0], h[0])[1]};
		},

		/* CUDA-graph draft step (one token): the whole block on the draft stage's ranks as
		 * ONE multi-GPU graph, with the sharded greedy argmax inside. Inputs are persistent
		 * buffers on the draft rank 0 (the embedding row, looked up eagerly from the table
		 * on device 0; the previous stack) plus the per-rank packed index state; run()
		 * fills them and replays. */
		mtp_graph: function (cache) {
			let tp = this;
			let M = this.mtp;
			let dc = cache.mtp;
			let md = M.S.devs;
			let ent = {
				emb: torch.zeros([1, 1, D], 'float16', 'cuda:' + md[0]),
				stack: torch.zeros([1, H, D], 'float32', 'cuda:' + md[0]),
				dev: map(md, (d) => qwen4.dev_state(cfg, 1, cache.max_seq, d)),
				bar: map(md, (d) => torch.zeros([1], 'float32', 'cuda:' + d)),
				voff: null
			};

			let offs = [];

			for (let g = 0; g < Gs; g++)
				push(offs, [ last.plan.vocab[g][0] ]);

			ent.voff = torch.from_array(offs, 'int64', 'cuda:' + md[0]);

			let put = function (pos) {
				let vals = qwen4.dev_values(ent.dev[0], pos, dc.mm);

				for (let g = 0; g < Gs; g++)
					ent.dev[g].pack.copy_(torch.from_array(vals, 'int64', 'cuda:' + md[g]));
			};

			let body = function () {
				let s0 = qwen4.mtp_input(cfg, M, ent.emb, 1, ent.stack);
				let h = map(md, (d, g) => (g == 0) ? s0 : s0.cuda(d));
				let c = tp.row_a(M.S, M.row, dc.layers[0], M.owner, h, 0, ent.dev, null);

				h = tp.layer_b(c);

				ent.out_stack = h[0].to('float16').to('float32');
				ent.out_hid = qwen4.gr_mix(cfg, M.mixer[0], h[0])[1];

				let vals = [], idxs = [];

				for (let g = 0; g < Gs; g++) {
					let l = tp.lm_heads[g].forward(ent.out_hid.cuda(md[g]).view([1, 1, D])).view([-1]);

					push(vals, l.max().view([1]).to('float32').cuda(md[0]));
					push(idxs, l.argmax(-1).view([1]).cuda(md[0]));
				}

				ent.vals = torch.cat(vals, 0);
				ent.idx = torch.cat(idxs, 0);

				// the greedy token on the device too (the chain feeds it to the next step without a host read)
				let best = ent.vals.argmax(-1).view([1]);

				ent.tok = ent.voff.index(best).add(ent.idx.view([Gs, 1]).index(best)).view([1]);
			};

			put(0);
			body();
			torch.synchronize();

			ent.graph = torch.graph();
			ent.graph.capture_begin(md[0], slice(md, 1));
			glm5.set_par(!getenv('AMP_QWEN4_NO_PAR'));
			body();
			glm5.set_par(false);
			ent.graph.capture_end();
			torch.synchronize();

			ent.run = function (id, stack, dc2, pos) {
				// the embedding row (the table lives on device 0)
				ent.emb.copy_(tp.embed_tokens.index(torch.from_array([ id ], 'int64', 'cuda:' + devs[0]))
					.view([1, 1, D]).cuda(md[0]));
				ent.stack.copy_(stack);
				put(pos);

				for (let g = 1; g < Gs; g++)
					ent.bar[g].copy_(ent.bar[0]);

				ent.graph.replay();

				for (let g = 1; g < Gs; g++)
					ent.bar[g].copy_(ent.bar[0]);

				let v = ent.vals.tolist(), ix = ent.idx.tolist();
				let best = 0;

				for (let g = 1; g < Gs; g++)
					if (v[g] > v[best])
						best = g;

				return {token: last.plan.vocab[best][0] + ix[best],
					hidden: qwen4.flat_stack_of(tp, ent.out_stack), collapsed: ent.out_hid};
			};

			/* W greedy draft steps chained on the device: step k embeds the token step k-1 picked, nothing is read back
			 * until the W tokens are done. hidden: the target's flattened stack (1, 1, 4D) fp16; seed: the round's
			 * first token (host); p: the target position of the seed. Returns the W token ids. */
			ent.chain = function (seed, hidden, p, W) {
				let acc = torch.zeros([W], 'int64', 'cuda:' + md[0]);
				let stack = hidden.reshape([1, H, D]).to('float32').cuda(M.dev);
				let row = tp.embed_tokens.index(torch.from_array([ seed ], 'int64', 'cuda:' + devs[0]));

				for (let k = 1; k <= W; k++) {
					ent.emb.copy_(row.view([1, 1, D]).cuda(md[0]));
					ent.stack.copy_(stack);
					put(p + k);

					for (let g = 1; g < Gs; g++)
						ent.bar[g].copy_(ent.bar[0]);

					ent.graph.replay();

					for (let g = 1; g < Gs; g++)
						ent.bar[g].copy_(ent.bar[0]);

					acc.narrow(0, k - 1, 1).copy_(ent.tok);
					stack = ent.out_stack;

					if (k < W)
						row = tp.embed_tokens.index(ent.tok.cuda(devs[0]));
				}

				return acc.tolist();
			};

			return ent;
		},

		/* Engine interface: draft step on token ids (array) with the target's / the
		 * previous draft step's flattened stack (1, S, 4D) fp16. head: truthy -> logits
		 * (1, S, V) from the sharded head. With the draft graph (S == 1, no head) the
		 * greedy token comes from the in-graph sharded argmax: r.token. */
		mtp_chunk: function (ids, hidden, dc, pos, head) {
			let S = length(ids);
			let M = this.mtp;
			let stack = hidden.reshape([S, H, D]).to('float32').cuda(M.dev);

			if (S == 1 && !head && this.mtp_g)
				return this.mtp_g.run(ids[0], stack, dc, pos);

			let emb = qwen35.embed_rows(this, dc.mm, ids, pos).cuda(M.dev);
			let r = this.mtp_run_rows(emb, S, stack, dc, pos, null);
			let o = {hidden: qwen4.flat_stack_of(this, r.stack), collapsed: r.hidden};

			if (head) {
				let ls = [];

				for (let g = 0; g < Gs; g++) {
					let l = this.lm_heads[g].forward(r.hidden.cuda(last.devs[g]).view([1, S, D]));

					push(ls, (g == 0) ? l : l.cuda(last.devs[0]));
				}

				o.logits = torch.cat(ls, -1);
			}

			return o;
		},

		/* the greedy draft chain of a round without host reads between the steps (engine: replaces the
		 * step-by-step loop when present) */
		mtp_chain: function (seed, hidden, p, W) {
			return this.mtp_g.chain(seed, hidden, p, W);
		},

		/* greedy draft token from a collapsed hidden (1 row) */
		draft_argmax: function (D1) {
			let vals = [], idxs = [];

			for (let g = 0; g < Gs; g++) {
				let l = this.lm_heads[g].forward(D1.cuda(last.devs[g]).view([1, 1, D])).view([-1]);

				push(vals, l.max().view([1]).to('float32').cuda(last.devs[0]));
				push(idxs, l.argmax(-1).view([1]).cuda(last.devs[0]));
			}

			let v = torch.cat(vals, 0).tolist(), ix = torch.cat(idxs, 0).tolist();
			let best = 0;

			for (let g = 1; g < Gs; g++)
				if (v[g] > v[best])
					best = g;

			return last.plan.vocab[best][0] + ix[best];
		},

		/* The PLE state is recurrent state outside the per-layer entries (device 0). */
		aux_get: function (cache) {
			let c = torch.zeros(cache.ple.conv.shape(), 'float16', 'cuda:' + cache.ple.dev);

			c.copy_(cache.ple.conv);

			return {conv: c, ids: slice(cache.ple.ids)};
		},

		aux_set: function (cache, o) {
			cache.ple.conv.copy_(o.conv);
			cache.ple.ids = slice(o.ids);
		},

		aux_reset: function (cache) {
			cache.ple.conv.fill_(0);

			for (let i = 0; i < length(cache.ple.ids); i++)
				cache.ple.ids[i] = cfg.eos_token_id;
		},

		aux_ids: function (cache, ids) {
			let n = length(ids), ctx = cfg.ngram_size - 1, out = [];

			for (let i = n - ctx; i < n; i++)
				push(out, (i >= 0) ? ids[i] : cfg.eos_token_id);

			cache.ple.ids = out;
		}
	};

	return tp;
}
