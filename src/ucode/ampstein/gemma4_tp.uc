// ucode/gemma4_tp.uc — tensor-parallel Gemma 4 (26B-A4B) over N GPUs.
//
// Every layer runs on ALL ranks, each on its share of it; the residual stream x (fp32) is replicated
// (bit-identical on every rank) and the ranks' partial sublayer outputs are summed by an all-reduce:
//   attention   whole KV-head groups per rank (sliding: 8 kv heads, global: 2), o_proj by rows
//   dense MLP   gate / up by columns, down by rows over the 128-aligned (padded) intermediate
//   experts     expert parallel: experts [e0, e1) per rank, routing replicated
//   lm_head     by vocabulary ranges in rank order, logits gathered on rank 0
//   norms, router, layer scalars: replicated; the embedding stays on rank 0
// Two all-reduces per layer: the attention output, then the dense MLP and the experts' partial sums stacked
// (the two are normalised separately afterwards, so they cannot be summed first).
//
//   let m = gemma4.load(dir, {devices: n, prefill_rows: chunk});
//   let tp = gemma4_tp.shard(m, {devices: [0, 1]});          // consumes m's layers

import * as gemma4 from './gemma4.uc';
import * as glm5 from './glm5.uc';
import * as models from './models.uc';
import { split_n } from './qwen35_tp.uc';

const EMBED_SCALE = 53.0;

/* largest all-reduce payload in rows through the P2P kernel: decode; prefill chunks use a second communicator */
const P2P_ROWS = 16;
const PF_ROWS = 2048;

function allreduce(tp, p) {
	let G = length(p);

	if (p[0].numel() <= tp.comm_cap)
		return map(p, (x, g) => tp.comm.allreduce(x, g));

	if (tp.comm_pf && p[0].numel() <= tp.comm_pf_cap)
		return map(p, (x, g) => tp.comm_pf.allreduce(x.contiguous(), g));

	let out = [];

	for (let g = 0; g < G; g++) {
		let dev = tp.devs[g];
		let acc = p[0].cuda(dev).to('float32');

		for (let r = 1; r < G; r++)
			acc = acc.add(p[r].cuda(dev).to('float32'));

		push(out, acc.to('float16'));
	}

	return out;
}

function copy_lin(lin, dev) {
	return models.cols(lin, [ [ 0, lin.out_features ] ], dev);
}

/* rank g's shard of layer L on device dev */
function shard_layer(cfg, plan, L, g, dev) {
	let hd = L.hd;
	let kv = plan.kv[L.full ? 1 : 0][g];             // [first, count] kv heads
	let qpk = int(L.nh / L.kvh);
	let mi = plan.mlp[g];
	let to = (t) => t.cuda(dev);
	let S = {
		dev: dev, full: L.full, hd: hd, kvh: kv[1], nh: kv[1] * qpk,
		q: models.cols(L.q, [ [ kv[0] * qpk * hd, kv[1] * qpk * hd ] ], dev),
		k: models.cols(L.k, [ [ kv[0] * hd, kv[1] * hd ] ], dev),
		v: L.v ? models.cols(L.v, [ [ kv[0] * hd, kv[1] * hd ] ], dev) : null,
		o: models.rows(L.o, kv[0] * qpk * hd, kv[1] * qpk * hd, dev),
		q_norm: to(L.q_norm), k_norm: to(L.k_norm), v_norm: torch.ones([hd], 'float16', 'cuda:' + dev),
		in_ln: to(L.in_ln), pre_ff: to(L.pre_ff), post_attn: to(L.post_attn), post_ff: to(L.post_ff),
		scalar: to(L.scalar),
		mlp: {
			gate: models.cols(L.mlp.gate, [ mi ], dev),
			up: models.cols(L.mlp.up, [ mi ], dev),
			down: models.rows(L.mlp.down, mi[0], mi[1], dev)
		}
	};

	if (L.moe) {
		let e0 = plan.experts[g][0], en = plan.experts[g][1];
		let ex = [];

		for (let e = e0; e < e0 + en; e++) {
			let x = L.moe.experts[e];

			push(ex, (dev == L.dev) ? x : {gate: copy_lin(x.gate, dev), up: copy_lin(x.up, dev), down: copy_lin(x.down, dev)});
		}

		S.post_ff1 = to(L.post_ff1);
		S.post_ff2 = to(L.post_ff2);
		S.pre_ff2 = to(L.pre_ff2);
		S.rscale = to(L.rscale);
		S.moe = {
			type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en, router: 'gemma4', shared: null,
			gate_t: to(L.moe.gate_t), pes: to(L.moe.pes), rl: null
		};
	}

	return S;
}

/* opts.devices: CUDA devices, one rank each (default [0, 1]); opts.weights: per-rank share weights (default equal);
 * opts.prefill_rows: the prefill chunk (sizes the P2P communicator) */
export function shard(model, opts) {
	opts = opts || {};

	let cfg = model.cfg;
	let devs = opts.devices || [ 0, 1 ];
	let G = length(devs);
	let w = opts.weights;
	let nl = cfg.num_hidden_layers;
	let V = model.lm_head.out_features;
	let ring = model.ring, chunk_cap = model.chunk_cap;

	if (!w) {
		w = [];

		for (let g = 0; g < G; g++)
			push(w, 1);
	}

	let L0 = model.layers[0];
	let plan = {
		kv: [ split_n(cfg.num_kv_heads, 1, w), split_n(cfg.num_global_kv_heads, 1, w) ],
		mlp: split_n(L0.mlp.gate.out_features, 128, w),
		experts: split_n(cfg.n_routed_experts, 1, w),
		vocab: split_n(V, 256, w)
	};

	for (let t in plan.kv)
		for (let r in t)
			if (!r[1])
				die('gemma4_tp: a rank without attention heads is not supported (too many ranks)');

	let layers = [];

	for (let i = 0; i < nl; i++) {
		let L = model.layers[i];

		push(layers, map(devs, (d, g) => shard_layer(cfg, plan, L, g, d)));
		model.layers[i] = null;

		if (i % 4 == 3)
			gc();
	}

	gc();

	let heads = map(devs, (d, g) => models.cols(model.lm_head, [ plan.vocab[g] ], d));
	let cap = P2P_ROWS * cfg.hidden_size;
	let pf_rows = (getenv('AMP_TP_PF_ROWS') != null) ? int(getenv('AMP_TP_PF_ROWS')) : (opts.prefill_rows ? 2 * opts.prefill_rows : PF_ROWS);
	let invs = model.invs;
	let embed_t = model.embed_tokens;
	let norms = map(devs, (d) => model.norm.cuda(d));
	let first = devs[0];

	let tp = {
		cfg: cfg, plan: plan, layers: layers, devs: devs, G: G, tp: true,
		comm: torch.p2p_comm(devs, cap), comm_cap: cap,
		comm_pf: (pf_rows > P2P_ROWS) ? torch.p2p_comm(devs, pf_rows * cfg.hidden_size) : null,
		comm_pf_cap: pf_rows * cfg.hidden_size,
		ring: ring, chunk_cap: chunk_cap,
		embed_tokens: embed_t, norm: norms, lm_heads: heads, lm_head: model.lm_head,

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i = 0; i < nl; i++) {
				let row = [];

				for (let g = 0; g < G; g++) {
					let S = layers[i][g];
					let dv = 'cuda:' + devs[g];
					let rows = S.full ? max_seq : ring;

					push(row, {
						dev: devs[g], type: cfg.layer_types[i],
						k: torch.zeros([rows, S.kvh, S.hd], 'float16', dv),
						v: torch.zeros([rows, S.kvh, S.hd], 'float16', dv)
					});
				}

				push(cache.layers, row);
			}

			return cache;
		},

		/* the per-rank attention contexts of a chunk of n rows at pos0 */
		contexts: function (pos0, n) {
			return map(devs, (d) => ({
				ring: ring,
				cs_l: gemma4.rope_cs(invs[d].l, pos0, n, d),
				cs_g: gemma4.rope_cs(invs[d].g, pos0, n, d),
				widx: (n < 64 || getenv('AMP_GEMMA4_NO_SDPA')) ? gemma4.window_idx(pos0, n, cfg.sliding_window, ring, d) : null
			}));
		},

		/* one layer on all ranks: x (per rank (T, H) fp32, replicated) -> the new list */
		layer: function (i, x, cache, pos0, ctxs) {
			let eps = cfg.rms_norm_eps;
			let Ls = layers[i], cl = cache.layers[i];
			let T = x[0].shape()[0];
			let p = [];

			for (let g = 0; g < G; g++)
				push(p, gemma4.attn_forward(cfg, Ls[g], glm5.rms_norm_f32(x[g], Ls[g].in_ln, eps), cl[g], pos0, ctxs[g]));

			let s = allreduce(this, p);

			for (let g = 0; g < G; g++)
				x[g] = torch.norm32(s[g], Ls[g].post_attn, eps, x[g]);

			p = [];

			for (let g = 0; g < G; g++) {
				let S = Ls[g];
				let m = glm5.mlp_forward(cfg, S.mlp, glm5.rms_norm_f32(x[g], S.pre_ff, eps));

				S.moe.rl = torch.matmul(torch.norm32(x[g], S.rscale, eps), S.moe.gate_t);

				let mo = glm5.moe_forward(cfg, S.moe, glm5.rms_norm_f32(x[g], S.pre_ff2, eps));

				S.moe.rl = null;
				push(p, torch.cat([ m, mo.to('float16') ], 0));
			}

			s = allreduce(this, p);

			for (let g = 0; g < G; g++) {
				let S = Ls[g];
				let h = torch.norm32(s[g].narrow(0, T, T), S.post_ff2, eps, torch.norm32(s[g].narrow(0, 0, T), S.post_ff1, eps));

				x[g] = torch.norm32(h, S.post_ff, eps, x[g], S.scalar);
			}

			return x;
		},

		/* logits (rows, V) fp16 of the last `rows` rows of x, on rank 0 */
		head: function (x, rows) {
			let ls = [];

			for (let g = 0; g < G; g++) {
				let xr = (rows < x[g].shape()[0]) ? x[g].narrow(0, x[g].shape()[0] - rows, rows).contiguous() : x[g];
				let l = heads[g].forward(glm5.rms_norm_f32(xr, norms[g], cfg.rms_norm_eps));

				push(ls, (g == 0) ? l : l.cuda(first));
			}

			let lg = torch.cat(ls, -1);

			if (cfg.softcap > 0) {
				let e = lg.to('float32').mul(2.0 / cfg.softcap).exp();

				lg = e.sub(1.0).div(e.add(1.0)).mul(cfg.softcap).to('float16');
			}

			return lg;
		},

		embed: function (ids_t) {
			let x0 = embed_t.index(ids_t).to('float32').mul(EMBED_SCALE);

			return map(devs, (d, g) => (g == 0) ? x0 : x0.cuda(d));
		},

		/* the token ids appended to the cache; logits (rows, V) of the last `rows` positions (default 1; 0: none) */
		forward_chunk: function (token_ids, cache, rows) {
			if (rows == null)
				rows = 1;

			let n = length(token_ids);

			if (n > chunk_cap) {
				let r = null;

				for (let q = 0; q < n; q += chunk_cap)
					r = this.forward_chunk(slice(token_ids, q, min(q + chunk_cap, n)), cache, (q + chunk_cap >= n) ? rows : 0);

				return r;
			}

			let pos0 = cache.length;
			let x = this.embed(torch.from_array(token_ids, 'int64', 'cuda:' + first));
			let ctxs = this.contexts(pos0, n);

			for (let i = 0; i < nl; i++)
				x = this.layer(i, x, cache, pos0, ctxs);

			cache.length = pos0 + n;

			return rows ? this.head(x, rows) : null;
		},

		/* engine interface: chunked prefill at cache.length, one chunk per step (all ranks work on every chunk) */
		prefill_pipeline: function (cache, ids, opts) {
			let tp = this;
			let chunk = min(opts.chunk || 1024, chunk_cap);
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let q = start + chunk; q < start + total; q += chunk)
				push(bounds, q);

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			let next = 0, done = 0, logits = null;

			return {
				step: function (rounds) {
					for (let r = 0; r < (rounds || 1) && next < length(bounds); r++) {
						let end = bounds[next++];
						let a = cache.length;
						let last = next == length(bounds);
						let lg = tp.forward_chunk(slice(ids, a - start, end - start), cache, last ? 1 : 0);

						if (last)
							logits = lg;

						done = end - start;

						if (opts.on_aux)
							opts.on_aux(end);
					}

					return next < length(bounds);
				},

				done_tokens: function () {return done;},
				logits: function () {return logits;},
				hidden: function () {return null;}
			};
		},

		/* CUDA-graph decode (one row): ONE multi-GPU graph per global-attention length bucket, captured lazily on rank
		 * 0's device with the other ranks as side devices (the all-reduces need no cross-device edges). */
		graphs: function (cache, sizes) {
			if (getenv('AMP_GEMMA4_NO_GRAPHS'))
				return null;

			let tp = this;
			let gmax = cache.max_seq;
			let ids = torch.zeros([1], 'int64', 'cuda:' + first);
			let posf = map(devs, (d) => torch.zeros([1], 'float32', 'cuda:' + d));
			let bar = map(devs, (d) => torch.zeros([1], 'float32', 'cuda:' + d));
			let gr = {};

			function body(glen) {
				for (let g = 1; g < G; g++)
					posf[g].copy_(posf[0]);

				let ctxs = map(devs, (d, g) => gemma4.dyn_ctx(cfg, invs[d], posf[g], ring, d, glen));
				let x = tp.embed(ids);

				for (let i = 0; i < nl; i++)
					x = tp.layer(i, x, cache, 0, ctxs);

				return tp.head(x, 1);
			}

			function bucket(pos) {
				let b = 256;

				while (b < pos + 1)
					b *= 2;

				return min(b, gmax);
			}

			function graph_of(glen) {
				if (gr[glen])
					return gr[glen];

				// the warm-up run writes the rows of the current position (the replay rewrites them)
				body(glen);
				torch.synchronize();

				let e = {graph: torch.graph()};

				e.graph.capture_begin(first, slice(devs, 1));
				e.out = body(glen);
				e.graph.capture_end();
				torch.synchronize();
				gr[glen] = e;

				return e;
			}

			// stream barriers rank 0 <-> the others: the graph launches on rank 0's stream only
			function barrier() {
				for (let g = 1; g < G; g++)
					bar[g].copy_(bar[0]);
			}

			function run_ids(ids_src, pos) {
				ids.copy_(ids_src);
				posf[0].copy_(torch.from_array([ pos * 1.0 ], 'float32', 'cuda:' + first));

				let e = graph_of(bucket(pos));

				barrier();
				e.graph.replay();
				barrier();
				cache.length = pos + 1;

				return {logits: e.out, hidden: null};
			}

			graph_of(bucket(0));

			return {
				has: function (S) {return S == 1;},
				run: function (tids, pos) {return run_ids(torch.from_array(tids, 'int64', 'cuda:' + first), pos);},
				run_dev: function (ids_dev, pos) {return run_ids(ids_dev.view([1]), pos);}
			};
		}
	};

	return tp;
}
