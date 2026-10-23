// ucode/dsv41_tp.uc — DeepSeek-V4.1-Flash as a PIPELINE OF TENSOR-PARALLEL STAGES (`--tp --stages 3` = three
// pairs of GPUs), the layout of dsv4_tp.uc / glm5_tp.uc.
//
// One stage = a group of Gs ranks that all run every layer of the stage's contiguous layer range, each on its
// share; the ranks' partial sublayer outputs are summed by an all-reduce and the residual streams (4 x fp32 per
// token) stay replicated, bit-identical, inside the group. A decode token then reads 1/Gs of a stage's weights
// per GPU.
//
// Sharding inside a group:
//   attention  query heads (wq_b rows, sinks), o_groups (wo_a groups) and wo_b by input columns (a partial sum);
//              wq_a, wkv, the compressors and the indexer are replicated (one shared K = V head, the compressor
//              and indexer are small), so dsv41.attn_forward / attn_decode run unchanged on a config copy with
//              the local head / group counts
//   MoE        expert parallel: experts [e0, e1) per rank (the fused kernels filter the router's picks to the
//              range), the shared expert sharded like a dense MLP (w1 / w3 rows, w2 columns); router replicated
//   Engram     replicated (each rank applies the same rows)
//   head       by vocabulary columns over the last stage's ranks, logits gathered on its rank 0
//
// A "lane" is rank r of every stage: it owns a complete cache (rings, pools, mirrors) over its own devices, exactly
// like the single-chain model; the lanes differ only in the head / expert shard they compute.
//
// let m = dsv41_tp.load(dir, {devices: 6, stages: 3, max_seq: 262144});

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as dsv4_tp from './dsv4_tp.uc';
import * as dsv41 from './dsv41.uc';

const P2P_ROWS = 8;
const MAX_ROWS = 8;
const NEGH = -60000.0;

/* the weights of layer i's attention on rank g of Gs (A lives on the staging device) */
function shard_attn(cfg, A, g, Gs, dev) {
	let nh = int(cfg.num_q_heads / Gs), ng = int(cfg.o_groups / Gs), D = cfg.head_dim;
	let R = {
		dev: dev, idx: A.idx, ratio: A.ratio, kv_source: A.kv_source, idx_source: A.idx_source,
		inv_freq: A.inv_freq.cuda(dev),
		q_a: dsv41.lin8_to(A.q_a, dev), q_norm: A.q_norm.cuda(dev),
		q_b: dsv41.lin8_rows(A.q_b, g * nh * D, nh * D, dev),
		wkv: dsv41.lin8_to(A.wkv, dev), kv_norm: A.kv_norm.cuda(dev),
		wo_a: dsv41.lin8g_groups(A.wo_a, g * ng, ng, dev),
		wo_b: dsv41.lin8_cols(A.wo_b, g * ng * cfg.o_lora_rank, ng * cfg.o_lora_rank, dev),
		sinks: A.sinks.narrow(0, g * nh, nh).contiguous().cuda(dev),
		comp: null, indexer: null
	};

	if (A.comp) {
		let C = A.comp;

		R.comp = {norm: C.norm.cuda(dev), wkv: C.wkv ? dsv41.comp_to(C.wkv, dev) : null, wgate: C.wgate ? dsv41.comp_to(C.wgate, dev) : null,
			wkv16: C.wkv16 ? dsv41.comp_to(C.wkv16, dev) : null};
	}

	if (A.indexer) {
		let I = A.indexer;

		R.indexer = {wq_b: dsv41.lin8_to(I.wq_b, dev), w_t: I.w_t.cuda(dev), owns_k: I.owns_k};

		if (I.owns_k) {
			R.indexer.wk_t = I.wk_t.cuda(dev);
			R.indexer.k_norm = I.k_norm.cuda(dev);
		}
	}

	return R;
}

/* a copy of the replicated mHC parameters of a site */
function hc_copy(hc, dev) {
	return {fn: hc.fn.cuda(dev), fn_pre_t: hc.fn_pre_t.cuda(dev), base: hc.base.cuda(dev), base_pre: hc.base_pre.cuda(dev),
		scale: hc.scale.cuda(dev), scale_pre: hc.scale_pre.cuda(dev)};
}

/* the shared expert on rank g: w1 / w3 rows, w2 columns of the intermediate range ri */
function shard_mlp8(M, ri, dev) {
	return {
		gate: dsv41.lin8_rows(M.gate, ri[0], ri[1], dev),
		up: dsv41.lin8_rows(M.up, ri[0], ri[1], dev),
		down: dsv41.lin8_cols(M.down, ri[0], ri[1], dev)
	};
}

function clamp16(x) {
	return x.clamp(-65504.0, 65504.0).to('float16');
}

/* The routed + shared expert partials reach beyond the fp16 range on the massive-activation rows (the full sum is
 * fp32 in the single-chain model): they travel through the fp16 all-reduce divided by FFN_SCALE, and hc_apply
 * multiplies its post weights by it again. */
const FFN_SCALE = 16.0;

function ffn_partial(y) {
	return y.mul(1.0 / FFN_SCALE).clamp(-65504.0, 65504.0).to('float16');
}

/* ---- recurrent state outside the KV rows (engine snapshots), rollback of speculative rows ---- */

function dsv41_tp_state(model, allreduce) {
	let cfg = model.cfg, layers = model.layers, Gs = model.Gs, nl = model.nl;
	let nr = model.nr;            // layers 0 .. nr-1 live on the worker (docs/RPC.md): their state is fetched from / sent to it
	let W = cfg.window - 1;

	/* per lane, per layer: the pool entry count, the half-built pair of the ratio-2 compressors and a copy of the
	 * window rows (the ring is overwritten by later tokens) */
	model.aux_get = function (cache) {
		let out = [];

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g], row = [];

			for (let cl in lane.layers) {
				let n = min(cache.length, W);

				if (!cl) {
					push(row, null);
					continue;
				}

				push(row, {n: cl.n, pk: (cl.pk && cache.length % 2 == 1) ? cl.pk.add(0.0) : null,
					pg: (cl.pk && cache.length % 2 == 1) ? cl.pg.add(0.0) : null,
					kw: n > 0 ? dsv41.ring_rows(cl, cache.length - n, cache.length).add(0.0) : null});
			}

			push(out, row);
		}

		return nr ? {lanes: out, remote: model.remote_client.aux_get(cache.rcache, cache.length)} : out;
	};

	model.aux_set = function (cache, snap) {
		if (nr) {
			model.remote_client.aux_put(cache.rcache, snap.remote, cache.length);
			snap = snap.lanes;
		}

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g];

			for (let i = nr; i < nl; i++) {
				let cl = lane.layers[i], v = snap[g][i];

				if (layers[i][g].attn.kv_source) {
					dsv41.set_pool_n(lane, i, v.n);

					if (v.pk) {
						cl.pk.copy_(v.pk);
						cl.pg.copy_(v.pg);
					}
				}

				if (v.kw)
					dsv41.ring_put(cl, cache.length - v.kw.shape()[0], v.kw);
			}
		}
	};

	model.aux_reset = function (cache) {
		for (let g = 0; g < Gs; g++)
			for (let i = nr; i < nl; i++)
				if (cache.lanes[g].layers[i].pool)
					dsv41.set_pool_n(cache.lanes[g], i, 0);

		cache.cids = [];

		if (nr)
			model.remote_client.reset(cache.rcache);
	};

	/* take back the rows of the last multi-row step beyond the length L (p0 < L <= p0 + S) */
	model.rollback = function (cache, L) {
		let lr = model.last_rows;

		if (!lr || L < lr.pos + 1 || L > lr.pos + lr.S)
			die(`dsv41_tp: rollback to ${L} is outside the last step`);

		if (nr)
			model.remote_client.rollback(cache.rcache, L);

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g];

			for (let i = nr; i < nl; i++) {
				let A = layers[i][g].attn;

				if (!A.kv_source)
					continue;

				dsv41.set_pool_n(lane, i, int(L / A.ratio));

				if (A.ratio == 2) {
					let cl = lane.layers[i];

					cl.pk.copy_(cl.rk.narrow(0, L - 1 - lr.pos, 1));
					cl.pg.copy_(cl.rg.narrow(0, L - 1 - lr.pos, 1));
				}
			}
		}

		cache.length = L;
		model.last_rows = {pos: lr.pos, S: L - lr.pos};
	};
}

/* ---- CUDA-graph decode: one multi-GPU graph per (stage, step size, context bucket) ---- */

function dsv41_tp_graphs(model, allreduce) {
	let cfg = model.cfg, cg = model.cg, layers = model.layers, devs = model.devs;
	let Gs = model.Gs, NS = model.NS, nl = model.nl, stages = model.stages, stage_of = model.stage_of;
	let engrams = model.engrams, meta = model.meta, last = model.last;
	let H = cfg.hidden_size, eps = cfg.rms_norm_eps;
	let ed = model.embed_dev;
	let nr = model.nr;                             // layers 0 .. nr-1 run on the worker host (docs/RPC.md)
	let no_head = model.no_head;                   // this is the worker: its graphs end with the streams
	let first_s = nr ? stage_of[nr] : 0;           // the first stage of this process

	// AMP_DSV41_TP_STEPPROF=1: seconds per phase of a decode step (synchronizes after the graph replays)
	let SPR = getenv('AMP_DSV41_TP_STEPPROF') ? model.stepprof = {} : null;
	let sp_t = 0;

	function sp_lap(name, sync) {
		if (!SPR)
			return;

		if (sync)
			torch.synchronize();

		let c = clock(true), t = c[0] + c[1] / 1e9;

		if (name != null)
			SPR[name] = (SPR[name] || 0) + t - sp_t;

		sp_t = t;
	}

	/* S rows (or one token) of layer i on the ranks of its stage; h / pre per rank, ctxs per lane, st per rank */
	function block_dec(cache, i, h, pre, ctxs, sts, caps, S) {
		let Sg = stages[stage_of[i]];
		let L = layers[i];
		let am = [], fm = [], p = [];

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g];

			am[g] = dsv41.hc_site(cg, L[g].hc_attn, h[g], pre[g]);

			let xn = torch.rms_norm(am[g].col, L[g].attn_norm, eps);

			p[g] = (S == 1) ? dsv41.attn_decode(cg, L[g].attn, ctxs[g], lane, lane.layers[i], xn, sts[g], caps) :
				dsv41.attn_decode_batch(cg, L[g].attn, ctxs[g], lane, lane.layers[i], xn, sts[g], caps);
		}

		let s = allreduce(Sg, p);

		for (let g = 0; g < Gs; g++)
			glm5.hc_apply(h[g], s[g], am[g].post, am[g].comb);

		for (let g = 0; g < Gs; g++) {
			fm[g] = dsv41.hc_site(cg, L[g].hc_ffn, h[g], am[g].pre);

			let xn = torch.rms_norm(fm[g].col, L[g].ffn_norm, eps);

			p[g] = ffn_partial(glm5.moe_forward(cg, L[g].moe, xn));
		}

		s = allreduce(Sg, p);

		let out = [];

		for (let g = 0; g < Gs; g++) {
			glm5.hc_apply(h[g], s[g], fm[g].post.mul(FFN_SCALE), fm[g].comb);
			push(out, fm[g].pre);
		}

		return out;
	}

	model.graphs = function (cache, sizes) {
		let max_seq = cache.max_seq;
		let caps_list = [];

		for (let c = 4096; c < max_seq; c *= 4)
			push(caps_list, c);

		push(caps_list, max_seq);

		// AMP_DSV41_TP_BUCKETS=8192,65536: pool capacities of the graphs (the last one is always max_seq);
		// AMP_DSV41_TP_SIZES=1,3,5: step sizes (a speculative round then verifies the nearest smaller one)
		// Defaults: two pool buckets (16384, max_seq) and the step sizes 1 / 3 / 5: every captured graph costs memory on
		// all six GPUs (the full set of 4 buckets x 5 sizes is ~0.4 GiB per GPU more, which is what the prefill chunk
		// would otherwise get); AMP_DSV41_TP_BUCKETS=full / AMP_DSV41_TP_SIZES=full restore the complete sets.
		let bk = getenv('AMP_DSV41_TP_BUCKETS') || '16384';

		if (bk != 'full') {
			caps_list = filter(map(split(bk, ','), (x) => +x), (c) => c < max_seq);
			push(caps_list, max_seq);
		}

		let sz = getenv('AMP_DSV41_TP_SIZES') || '1,3,5';

		if (sz != 'full')
			sizes = filter(map(split(sz, ','), (x) => +x), (x) => index(sizes || [ 1 ], x) >= 0 || x == 1);

		// (the worker captures first: its warm-up clobbers its caches like ours; both are reset afterwards)
		if (nr) {
			model.remote_sizes = sizes || [ 1 ];
			model.remote_client.graphs(cache.rcache, model.remote_sizes);
		}

		dsv41.mem_report(devs, 'before graphs (caches allocated)');

		let sets = {};
		let gpool = {};                              // one allocator pool per stage (rank 0's device) shared by its graphs
		let tap_layers = cfg.dspark_taps;

		for (let S in (sizes || [ 1 ])) {
			if (S > MAX_ROWS)
				die(`dsv41_tp: at most ${MAX_ROWS} rows per step`);

			let sts = {};

			for (let d in devs)
				sts[d] = (S == 1) ? dsv41.new_dec_dev(cfg, d) : dsv41.new_dec_rows(cfg, d, S);

			// engram buffers of S rows
			for (let l in keys(engrams))
				for (let E in engrams[l])
					if (S > 1 && E.emb_bufs[S] == null)
						E.emb_bufs[S] = torch.zeros([S, E.emb_buf.shape()[1]], 'float16', 'cuda:' + E.dev);

			let buckets = [];

			for (let c1 in caps_list) {
				let caps = [ 0, c1, int((c1 + 1) / 2) ];
				let ctxs = [];
				let items = [];

				for (let g = 0; g < Gs; g++)
					push(ctxs, {topk: null, cand: null, dst: {}, lk: {}});

				for (let s = 0; s < NS; s++) {
					let Sg = stages[s];
					let it = {s: s, first: Sg.first, end: Sg.end, h_in: [], pre_in: [], taps: [], bar: []};

					for (let g = 0; g < Gs; g++) {
						let dv = 'cuda:' + Sg.devs[g];
						let pre = torch.zeros([S, 4], 'float32', dv);

						pre.narrow(1, 0, 1).fill_(1.0);
						push(it.h_in, torch.zeros([S, 4, H], 'float32', dv));
						push(it.pre_in, pre);
						push(it.bar, torch.zeros([1], 'float32', dv));
					}

					push(items, it);
				}

				function body(it) {
					let Sg = stages[it.s];
					let h = it.h_in, pre = it.pre_in;
					let stg = map(Sg.devs, (d) => sts[d]);

					it.taps = [];

					for (let i = it.first; i < it.end; i++) {
						if (engrams[i])
							for (let g = 0; g < Gs; g++)
								dsv41.engram_apply(cfg, engrams[i][g], h[g], (S == 1) ? engrams[i][g].emb_buf : engrams[i][g].emb_bufs[S]);

						if (index(tap_layers, i) >= 0)
							push(it.taps, {dev: Sg.devs[0], t: h[0].mean(1)});

						pre = block_dec(cache, i, h, pre, ctxs, stg, caps, S);
					}

					it.h_out = h;
					it.pre_out = pre;

					if (it.s == NS - 1 && no_head) {
						// (nothing to do: the streams h / pre are the result)
					}
					else if (it.s == NS - 1) {
						let xs = [];

						for (let g = 0; g < last.G; g++)
							push(xs, torch.rms_norm(dsv41.hc_collapse(h[g], pre[g]), model.norm[g], eps));

						it.logits = model.head_state(xs);
						it.am = it.logits.argmax(-1);
					}
				}

				for (let d in devs)
					if (S == 1)
						dsv41.dec_set_pos(cfg, sts[d], 0);
					else
						dsv41.dec_set_rows(cfg, sts[d], 0);

				// eager warm-up (creates the mirrors and the remote copies), then capture
				for (let it in items) {
					body(it);
					torch.synchronize();
				}

				for (let it in items) {
					let Sg = stages[it.s];

					it.graph = torch.graph();
					it.graph.capture_begin(Sg.devs[0], slice(Sg.devs, 1), gpool[Sg.devs[0]]);
					glm5.set_par(!getenv('AMP_DSV41_NO_PAR'));         // the shared expert on a forked stream, beside the routed ones
					body(it);
					glm5.set_par(false);
					it.graph.capture_end();

					if (!gpool[Sg.devs[0]])
						gpool[Sg.devs[0]] = it.graph.pool();
				}

				torch.synchronize();
				push(buckets, {c1: c1, items: items, ctxs: ctxs});
			}

			sets[S] = {buckets: buckets, sts: sts};
			dsv41.mem_report(devs, `graphs captured (${S} rows)`);
		}

		model.aux_reset(cache);
		dsv41.mem_report(devs, 'graphs captured');
		torch.empty_cache();
		dsv41.mem_report(devs, 'graphs captured, cache emptied');

		/* S = length(ids) rows at pos .. pos + S - 1: {logits (S, vocab), am, taps (S, 3 * hidden) fp32} */
		function run_rows(ids, pos) {
			let S = length(ids);
			let gs = sets[S];

			sp_lap(null);

			if (!gs)
				die(`dsv41_tp: no graph for ${S} rows`);

			let b = null;

			for (let x in gs.buckets)
				if (x.c1 > pos + S - 1 && !b)
					b = x;

			if (!b)
				die(`dsv41_tp: position ${pos + S - 1} is past the context of ${max_seq}`);

			if (length(cache.cids) > pos)
				splice(cache.cids, pos, length(cache.cids) - pos);

			while (length(cache.cids) < pos)
				push(cache.cids, meta.pad);

			for (let d in devs)
				if (S == 1)
					dsv41.dec_set_pos(cfg, gs.sts[d], pos);
				else
					dsv41.dec_set_rows(cfg, gs.sts[d], pos);

			// (with remote layers the worker starts on its share at once: it embeds and runs them while this prepares)
			if (nr)
				model.remote_client.forward_begin(cache.rcache, pos, ids, true);

			let idt = torch.from_array(ids, 'int64');
			let cn = meta.token_map.index(idt).view([-1]).tolist();
			let hist = [];

			for (let k = 3; k >= 1; k--)
				push(hist, (pos - k >= 0) ? cache.cids[pos - k] : meta.pad);

			for (let c in cn) {
				push(hist, c);
				push(cache.cids, c);
			}

			let hist_t = torch.from_array(hist, 'int64').view([1, -1]);

			for (let li = 0; li < length(cfg.engram_layers); li++) {
				let l = cfg.engram_layers[li];

				if (l >= nr && l < nl) {
					let emb = dsv41.engram_rows(cfg, meta, engrams[l][0], li, hist_t, S);

					for (let g = 0; g < Gs; g++) {
						let E = engrams[l][g];

						(S == 1 ? E.emb_buf : E.emb_bufs[S]).copy_(g == 0 ? emb : emb.cuda(E.dev, true));
					}
				}
			}

			let h = nr ? null : model.embed_tokens.index(idt.cuda(ed, true)).to('float32').reshape([S, 1, H]).expand([S, 4, H]).contiguous();
			let taps = [], logits = null, am = null, prev = null, hidden = null;

			sp_lap('pre');

			if (nr) {
				let r = model.remote_client.forward_end(1800000);

				if (r.n != S || r.pos != pos)
					die(`dsv41_tp: the worker answered ${r.n} rows at ${r.pos}, expected ${S} at ${pos}`);

				// (stands in for the previous stage: its streams, on every rank of the first local stage)
				let s0 = stages[first_s];
				let W = 4 * H + 4, R = int(r.packed.shape()[1] / W);
				let pks = map(s0.devs, (d, g) => r.packed.narrow(1, (g % R) * W, W).contiguous().cuda(d));

				prev = {h_out: map(pks, (pk) => pk.narrow(1, 0, 4 * H).contiguous().view([S, 4, H])),
					pre_out: map(pks, (pk) => pk.narrow(1, 4 * H, 4).contiguous())};
				sp_lap('worker');
			}

			for (let it in b.items) {
				let Sg = stages[it.s];

				for (let g = 0; g < Gs; g++) {
					if (!prev) {
						it.h_in[g].copy_(h.cuda(Sg.devs[g], true));
					}
					else {
						it.h_in[g].copy_(prev.h_out[g]);
						it.pre_in[g].copy_(prev.pre_out[g]);
					}
				}

				// the ranks' streams meet before and after the multi-GPU graph (as dsv4_tp's replay)
				for (let g = 1; g < Gs; g++)
					it.bar[g].copy_(it.bar[0]);

				sp_lap('copy_in');
				it.graph.replay();

				for (let g = 1; g < Gs; g++)
					it.bar[g].copy_(it.bar[0]);

				sp_lap('graph' + it.s, true);

				for (let ctx in b.ctxs)
					for (let k in keys(ctx.lk))
						ctx.lk[k].dst.copy_(ctx.lk[k].src);

				for (let g = 0; g < Gs; g++)
					for (let i = it.first; i < it.end; i++) {
						let A = layers[i][g].attn;

						if (A.kv_source)
							for (let j = int(pos / A.ratio); j <= int((pos + S - 1) / A.ratio); j++)
								dsv41.mirror_entry(cache.lanes[g], i, j);
					}

				push(taps, ...it.taps);
				sp_lap('post');

				if (it.s == NS - 1) {
					logits = it.logits;
					am = it.am;

					if (no_head)
						hidden = {h: it.h_out, pre: it.pre_out};
				}

				prev = it;
			}

			for (let g = 0; g < Gs; g++)
				for (let i = nr; i < nl; i++) {
					let A = layers[i][g].attn;

					if (A.kv_source)
						dsv41.set_pool_n(cache.lanes[g], i, int((pos + S) / A.ratio));
				}

			cache.length = pos + S;
			model.last_rows = {pos: pos, S: S};

			let dt = length(taps) ? taps[0].dev : 0;

			return {logits: logits, am: am, hidden: hidden,
				taps: length(taps) ? torch.cat(map(taps, (t) => (t.dev == dt) ? t.t : t.t.cuda(dt)), 1) : null};
		}

		return {
			has: function (S) {return sets[S] != null;},
			rows: run_rows,
			run: function (ids, pos) {return run_rows(ids, pos);},
			run_dev: function (ids_dev, pos) {return run_rows(ids_dev.view([-1]).tolist(), pos);}
		};
	};
}

function dsv41_tp_methods(model) {
	let cfg = model.cfg, cg = model.cg, layers = model.layers, devs = model.devs;
	let Gs = model.Gs, NS = model.NS, nl = model.nl, stages = model.stages, stage_of = model.stage_of;
	let engrams = model.engrams, meta = model.meta, last = model.last;
	let H = cfg.hidden_size, eps = cfg.rms_norm_eps;
	let ed = model.embed_dev;
	let ENG0 = {};                                 // layer -> the rank-0 copy (the table reader)
	let nr = model.nr;                             // layers 0 .. nr-1 run on the worker host (docs/RPC.md)
	let first_s = nr ? stage_of[nr] : 0;           // the first stage of this process

	for (let l in keys(engrams))
		ENG0[l] = engrams[l][0];

	/* ---- caches: one per lane (rank r of every stage) ---- */

	model.new_cache = function (max_seq) {
		let lanes = [];

		for (let g = 0; g < Gs; g++) {
			let lane = {max_seq: max_seq, length: 0, layers: [], pools: {}, cids: []};

			for (let i = 0; i < nl; i++) {
				if (i < nr) {
					push(lane.layers, null);
					continue;
				}

				let cl = dsv41.new_layer_cache(cfg, layers[i][g].attn, max_seq);

				push(lane.layers, cl);

				if (layers[i][g].attn.kv_source)
					lane.pools[i] = {owner: cl, mirrors: []};
			}

			push(lanes, lane);
		}

		let cache = {max_seq: max_seq, length: 0, cids: [], toks: [], lanes: lanes, layers: lanes[0].layers, pools: lanes[0].pools};

		// the caches of layers 0 .. nr-1 are the worker's: this holds their id
		if (nr)
			cache.rcache = model.remote_client.new_cache(max_seq);

		return cache;
	};

	model.free_cache = function (cache) {
		if (nr && cache.rcache != null) {
			model.remote_client.free(cache.rcache);
			cache.rcache = null;
		}
	};

	/* sum of one fp16 partial per rank -> per-rank sums (saturated like every fp16 hand-off of the model) */
	function allreduce(S, p) {
		if (getenv('AMP_DSV41_TP_NOAR'))             // experiment: the cost of the communication (wrong results)
			return p;

		return dsv4_tp.allreduce(S, p);
	}

	/* ---- prefill (eager) ---- */

	/* A chunk in flight: embedded token ids on every rank of stage 0, the Engram history, the mHC state. */
	model.chunk_begin = function (token_ids, cache, issued) {
		let T = length(token_ids);
		let pos0 = cache.length;

		if (length(cache.cids) > pos0)
			splice(cache.cids, pos0, length(cache.cids) - pos0);

		while (length(cache.cids) < pos0)
			push(cache.cids, meta.pad);

		let idt = torch.from_array(token_ids, 'int64');
		let cn = meta.token_map.index(idt).view([-1]).tolist();
		let hist = [];

		for (let k = 3; k >= 1; k--)
			push(hist, (pos0 - k >= 0) ? cache.cids[pos0 - k] : meta.pad);

		for (let c in cn)
			push(hist, c);

		for (let c in cn)
			push(cache.cids, c);

		let h0 = null;

		if (nr) {
			// the worker embeds and runs layers 0 .. nr-1 (issued already by a pipelined prefill)
			if (!issued)
				model.remote_client.forward_begin(cache.rcache, pos0, token_ids);
		}
		else {
			let e = model.embed_tokens.index(idt.cuda(ed, true));

			h0 = e.to('float32').reshape([T, 1, H]).expand([T, 4, H]).contiguous();
		}

		let s0 = stages[first_s];
		let c = {T: T, pos0: pos0, end: pos0 + T, stage: nr ? -1 : stage_of[0], h: [], pre: [], ctx: [], taps: null,
			hist_t: torch.from_array(hist, 'int64').view([1, -1]), eg: {}};

		// the Engram rows are read from disk by a worker thread while the other chunks are issued
		for (let l in keys(engrams))
			c.eg[l] = dsv41.engram_start(cfg, meta, ENG0[l], index(cfg.engram_layers, +l), c.hist_t, T);

		for (let g = 0; g < Gs; g++) {
			push(c.ctx, {topk: null, cand: null});

			if (nr)
				continue;

			let dv = 'cuda:' + s0.devs[g];
			let pre = torch.zeros([T, 4], 'float32', dv);

			pre.narrow(1, 0, 1).fill_(1.0);
			push(c.h, (s0.devs[g] == ed) ? h0 : h0.cuda(s0.devs[g], true));
			push(c.pre, pre);
		}

		cache.length = pos0 + T;

		return c;
	};

	/* AMP_DSV41_TP_HOSTPROF=1: host seconds per phase of the prefill layers (dsv41_tp.HOSTPROF; the host only, no syncs) */
	let HP = getenv('AMP_DSV41_TP_HOSTPROF') ? model.hostprof = {} : null;
	let BAL = getenv('AMP_DSV41_TP_BALANCE') ? model.balance = {} : null;
	let hp_t = 0;

	function hp(name) {
		if (!HP)
			return;

		let c = clock(true), t = c[0] + c[1] / 1e9;

		if (name != null)
			HP[name] = (HP[name] || 0) + t - hp_t;

		hp_t = t;
	}

	/* the Engram rows of the chunk for layer i on every rank */
	function engram_chunk(i, c) {
		let li = index(cfg.engram_layers, i);
		let emb = c.eg[i] ? dsv41.engram_finish(cfg, ENG0[i], c.eg[i]) : dsv41.engram_rows(cfg, meta, ENG0[i], li, c.hist_t, c.T);

		c.eg[i] = null;

		return map(engrams[i], (E, g) => (g == 0) ? emb : emb.cuda(E.dev, true));
	}

	/* layer i on all ranks of its stage for the chunk (eager) */
	function layer_fwd(i, c, cache) {
		let S = stages[stage_of[i]];
		let L = layers[i];
		let am = [], fm = [], p = [];

		hp(null);

		if (engrams[i]) {
			let emb = engram_chunk(i, c);

			for (let g = 0; g < Gs; g++)
				dsv41.engram_apply(cfg, engrams[i][g], c.h[g], emb[g]);
		}

		if (c.taps && index(cfg.dspark_taps, i) >= 0)
			push(c.taps, {dev: S.devs[0], t: c.h[0].mean(1)});

		hp('engram');

		let at = [];

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g];

			am[g] = dsv41.hc_site(cg, L[g].hc_attn, c.h[g], c.pre[g]);

			let xn = torch.rms_norm(am[g].col, L[g].attn_norm, eps);

			push(at, dsv41.attn_forward_steps(cg, L[g].attn, c.ctx[g], lane, lane.layers[i], xn, c.pos0));
		}

		hp('hc_attn');

		// the ranks' launches alternate step by step (see attn_forward_steps)
		for (let k = 0; k < length(at[0].steps); k++)
			for (let g = 0; g < Gs; g++)
				at[g].steps[k]();

		for (let g = 0; g < Gs; g++)
			p[g] = at[g].out();

		hp('attn');

		let s = allreduce(S, p);

		hp('allreduce');

		for (let g = 0; g < Gs; g++)
			glm5.hc_apply(c.h[g], s[g], am[g].post, am[g].comb);

		hp('hc_apply');

		let pend = [];

		for (let g = 0; g < Gs; g++) {
			fm[g] = dsv41.hc_site(cg, L[g].hc_ffn, c.h[g], am[g].pre);

			let xn = torch.rms_norm(fm[g].col, L[g].ffn_norm, eps);

			push(pend, glm5.moe_begin(cg, L[g].moe, xn));
		}

		hp('hc_ffn');

		if (BAL && pend[0].sorted) {                 // experiment: how unevenly the ranks' experts are loaded
			let cnt = map(pend, (q, g) => q.sorted.count.narrow(0, 0, length(L[g].moe.experts)).sum().item());
			let key = '' + i;

			BAL[key] = BAL[key] || {n: 0, mx: 0, mean: 0};
			BAL[key].n++;
			BAL[key].mx += max(...cnt);
			BAL[key].mean += (cnt[0] + cnt[1]) / 2.0;
		}

		for (let g = 0; g < Gs; g++)
			p[g] = ffn_partial(glm5.moe_finish(cg, L[g].moe, pend[g]));

		hp('moe');

		s = allreduce(S, p);

		hp('allreduce');

		for (let g = 0; g < Gs; g++) {
			glm5.hc_apply(c.h[g], s[g], fm[g].post.mul(FFN_SCALE), fm[g].comb);
			c.pre[g] = fm[g].pre;
		}
	}

	/* ---- double-buffered prefill (opt-in, AMP_DSV41_TP_OV=1; measured slower, docs/LESSONS.md L165): the chunk travels as two micro-chunks whose all-reduces (PCIe x4: a quarter of
	 * the round) run on side streams, one communicator each, behind the other micro-chunk's compute ---- */

	let OV_COPY = getenv('AMP_DSV41_TP_OV_COPY') ? true : false;

	function ar_begin(S, p, m) {
		let xs = [], outs = [];

		if (OV_COPY) {
			// the copy engines move the partials (no spinning kernel competing with the compute for SMs and L2)
			let tmp = [];

			for (let g = 0; g < Gs; g++) {
				push(xs, p[g].contiguous());
				torch.stream_side(S.devs[g], 2 + m);
			}

			for (let g = 0; g < Gs; g++) {
				let acc = xs[g].to('float32');

				for (let r = 1; r < Gs; r++) {
					let q = xs[(g + r) % Gs].cuda(S.devs[g], true);

					push(tmp, q);
					acc = acc.add(q);
				}

				push(outs, acc.clamp(-65504.0, 65504.0).to('float16'));
			}

			for (let g = 0; g < Gs; g++)
				torch.stream_main(S.devs[g], 2 + m);

			return {S: S, m: m, xs: [ xs, tmp ], outs: outs};
		}

		for (let g = 0; g < Gs; g++) {
			push(xs, p[g].contiguous());
			torch.stream_side(S.devs[g], 2 + m);
			push(outs, S.comm_ov[m].allreduce(xs[g], g));
			torch.stream_main(S.devs[g], 2 + m);
		}

		return {S: S, m: m, xs: xs, outs: outs};
	}

	// the compute stream waits for the reduction; the partials stay referenced until then
	function ar_end(h) {
		for (let g = 0; g < Gs; g++)
			torch.stream_wait(h.S.devs[g], 2 + h.m);

		h.xs = null;

		return h.outs;
	}

	function sv(t, sb) {
		return t.narrow(0, sb.o, sb.T);
	}

	function attn_phase(i, c, m, cache) {
		let sb = c.sub[m], L = layers[i];
		let am = [], at = [];

		for (let g = 0; g < Gs; g++) {
			let lane = cache.lanes[g];

			am[g] = dsv41.hc_site(cg, L[g].hc_attn, sv(c.h[g], sb), sv(c.pre[g], sb));

			let xn = torch.rms_norm(am[g].col, L[g].attn_norm, eps);

			push(at, dsv41.attn_forward_steps(cg, L[g].attn, sb.ctx[g], lane, lane.layers[i], xn, sb.pos0));
		}

		for (let k = 0; k < length(at[0].steps); k++)
			for (let g = 0; g < Gs; g++)
				at[g].steps[k]();

		return {am: am, p: map(at, (a) => a.out())};
	}

	function ffn_phase(i, c, m, a, s) {
		let sb = c.sub[m], L = layers[i];
		let fm = [], pend = [], p = [];

		for (let g = 0; g < Gs; g++) {
			glm5.hc_apply(sv(c.h[g], sb), s[g], a.am[g].post, a.am[g].comb);
			fm[g] = dsv41.hc_site(cg, L[g].hc_ffn, sv(c.h[g], sb), a.am[g].pre);

			let xn = torch.rms_norm(fm[g].col, L[g].ffn_norm, eps);

			push(pend, glm5.moe_begin(cg, L[g].moe, xn));
		}

		for (let g = 0; g < Gs; g++)
			p[g] = ffn_partial(glm5.moe_finish(cg, L[g].moe, pend[g]));

		return {fm: fm, p: p};
	}

	function ffn_apply(c, m, f, s) {
		let sb = c.sub[m];

		for (let g = 0; g < Gs; g++) {
			glm5.hc_apply(sv(c.h[g], sb), s[g], f.fm[g].post.mul(FFN_SCALE), f.fm[g].comb);
			sv(c.pre[g], sb).copy_(f.fm[g].pre);
		}
	}

	// the last reduction of a layer is applied after the next layer's first micro-chunk is issued
	function flush_carry(c) {
		if (c.carry) {
			let f = c.carry;

			c.carry = null;
			f();
		}
	}

	function ov_ok(c) {
		return stages[0].comm_ov && c.T >= 256 && c.T <= 2 * stages[0].comm_ov_rows;
	}

	function layer_fwd_ov(i, c, cache) {
		let S = stages[stage_of[i]];

		if (!c.sub) {
			let T0 = 2 * int((c.T + 3) / 4);

			c.sub = [
				{o: 0, T: T0, pos0: c.pos0, ctx: map(c.ctx, () => ({topk: null, cand: null}))},
				{o: T0, T: c.T - T0, pos0: c.pos0 + T0, ctx: map(c.ctx, () => ({topk: null, cand: null}))}
			];
		}

		if (engrams[i] || (c.taps && index(cfg.dspark_taps, i) >= 0)) {
			flush_carry(c);

			if (engrams[i]) {
				let emb = engram_chunk(i, c);

				for (let g = 0; g < Gs; g++)
					dsv41.engram_apply(cfg, engrams[i][g], c.h[g], emb[g]);
			}

			if (c.taps && index(cfg.dspark_taps, i) >= 0)
				push(c.taps, {dev: S.devs[0], t: c.h[0].mean(1)});
		}

		let a0 = attn_phase(i, c, 0, cache);
		let r0 = ar_begin(S, a0.p, 0);

		flush_carry(c);

		let a1 = attn_phase(i, c, 1, cache);
		let r1 = ar_begin(S, a1.p, 1);

		let f0 = ffn_phase(i, c, 0, a0, ar_end(r0));
		let rf0 = ar_begin(S, f0.p, 0);

		let f1 = ffn_phase(i, c, 1, a1, ar_end(r1));
		let rf1 = ar_begin(S, f1.p, 1);

		ffn_apply(c, 0, f0, ar_end(rf0));
		c.carry = () => ffn_apply(c, 1, f1, ar_end(rf1));
	}

	function layer_run(i, c, cache) {
		if (ov_ok(c))
			return layer_fwd_ov(i, c, cache);

		flush_carry(c);
		layer_fwd(i, c, cache);
	}

	/* the chunk through stage s: copies the streams to the stage's ranks when it comes from another stage */
	model.chunk_stage_begin = function (c, s) {
		let S = stages[s];

		if (c.stage == -1) {
			// the streams come from the worker (hc * hidden floats and the hc pre-weights per row)
			let r = model.remote_client.forward_end(1800000);

			if (r.n != c.T || r.pos != c.pos0)
				die(`dsv41_tp: the worker answered ${r.n} rows at ${r.pos}, expected ${c.T} at ${c.pos0}`);

			let W = 4 * H + 4, R = int(r.packed.shape()[1] / W);

			for (let g = 0; g < Gs; g++) {
				let pk = r.packed.narrow(1, (g % R) * W, W).contiguous().cuda(S.devs[g]);

				push(c.h, pk.narrow(1, 0, 4 * H).contiguous().view([c.T, 4, H]));
				push(c.pre, pk.narrow(1, 4 * H, 4).contiguous());
			}

			c.stage = s;

			return;
		}

		if (c.stage != s) {
			for (let g = 0; g < Gs; g++) {
				c.h[g] = c.h[g].cuda(S.devs[g], true);
				c.pre[g] = c.pre[g].cuda(S.devs[g], true);
			}

			c.stage = s;
		}
	};

	/* AMP_DSV41_TP_EARLY_HANDOFF=1: the hand-off of a finished stage is issued at once (126 MB per rank over PCIe x4, ~50 ms;
	 * measured slower: it competes with the all-reduces for the link) */
	model.chunk_handoff = function (c, s) {
		if (s + 1 < NS && getenv('AMP_DSV41_TP_EARLY_HANDOFF'))
			model.chunk_stage_begin(c, s + 1);
	};

	model.chunk_stage = function (c, cache, s) {
		model.chunk_stage_begin(c, s);

		for (let i = stages[s].first; i < stages[s].end; i++)
			layer_run(i, c, cache);

		flush_carry(c);
	};

	/* the active chunks' stages together, layer by layer round-robin: a stage issued whole blocks the host on its
	 * full launch queue and the other stages' GPUs run dry meanwhile */
	model.chunk_stages = function (cs, cache) {
		let n = 0;

		for (let e in cs) {
			model.chunk_stage_begin(e.c, e.s);
			n = max(n, stages[e.s].end - stages[e.s].first);
		}

		for (let j = 0; j < n; j++)
			for (let e in cs)
				if (stages[e.s].first + j < stages[e.s].end)
					layer_run(stages[e.s].first + j, e.c, cache);

		for (let e in cs) {
			flush_carry(e.c);
			model.chunk_handoff(e.c, e.s);
		}
	};

	/* the DSpark taps of a finished chunk: (T, 3 * hidden) fp32 on the first tapped layer's device */
	model.chunk_taps = function (c) {
		let dev = c.taps[0].dev;

		return torch.cat(map(c.taps, (t) => (t.dev == dev) ? t.t : t.t.cuda(dev, true)), 1);
	};

	/* the head over the last `rows` rows of a finished chunk: logits (rows, vocab) fp16 on the last stage's rank 0 */
	model.chunk_head = function (c, rows) {
		let xs = [];

		for (let g = 0; g < last.G; g++) {
			let x = torch.rms_norm(dsv41.hc_collapse(c.h[g], c.pre[g]), model.norm[g], eps);

			push(xs, (rows < c.T) ? x.narrow(0, c.T - rows, rows).contiguous() : x);
		}

		return model.head_state(xs);
	};

	/* the sharded head on per-rank normed states xs[g] (R, hidden) fp16 */
	model.head_state = function (xs) {
		let ls = [];

		for (let g = 0; g < last.G; g++) {
			let l = model.lm_heads[g].forward(xs[g]);

			push(ls, (g == 0) ? l : l.cuda(last.devs[0]));
		}

		return torch.cat(ls, -1);
	};

	/* logits (rows, vocab) fp16 of a normed state xn on any device, on the last stage's rank 0 */
	model.head_full = function (xn) {
		return model.head_state(map(last.devs, (d) => xn.cuda(d)));
	};

	model.forward_chunk = function (token_ids, cache, rows, on_taps) {
		if (rows == null)
			rows = 1;

		let c = model.chunk_begin(token_ids, cache);

		if (on_taps)
			c.taps = [];

		for (let s = 0; s < NS; s++)
			model.chunk_stage(c, cache, s);

		if (on_taps)
			on_taps(model.chunk_taps(c), c.pos0);

		return (rows == 0) ? null : model.chunk_head(c, rows);
	};

	model.prefill = function (token_ids, cache, chunk, rows) {
		let n = length(token_ids);

		chunk = chunk || 512;

		for (let p = 0; p < n; p += chunk) {
			let e = min(p + chunk, n);
			let lastc = (e == n);
			let r = model.forward_chunk(slice(token_ids, p, e), cache, lastc ? (rows == null ? 1 : rows) : 0);

			if (lastc)
				return r;
		}

		return null;
	};

	/* Engine interface: chunked prefill as a wavefront over the stages (one stage per chunk per round). */
	model.prefill_pipeline = function (cache, ids, opts) {
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
		let next = 0, prev_end = start, done = 0, result = null;
		let snap_at = (opts.cuts && length(opts.cuts)) ? opts.cuts[0] : 0;

		for (let s = 0; s < NS; s++)
			push(active, null);

		/* the page cache is filled a few chunks ahead, in order: one big request queue would stall the first gathers */
		let ahead = +(getenv('AMP_DSV41_TP_AHEAD') || 3), pf = 0;

		function prefetch(upto) {
			if (!getenv('AMP_DSV41_NO_ENGRAM') && upto > pf) {
				dsv41.engram_prefetch(cfg, meta, ENG0, cache, ids, start, pf, upto);
				pf = max(pf, upto);
			}
		}

		prefetch(min(total, ahead * chunk));

		/* With remote layers the worker runs the chunks one command each, in order, and is kept two chunks ahead: it
		 * computes chunk n + 1 while this host is busy with chunk n. Not past the snapshot boundary though (the state
		 * the engine snapshots there is the worker's too: its answer must not include the chunks behind the cut). */
		let reqs = 0;

		function issue() {
			// (called by admit() after it took chunk next - 1: that one and the one behind it are requested)
			while (nr && reqs < length(bounds) && reqs < next + 1) {
				let pos = reqs ? bounds[reqs - 1] : start;

				if (reqs >= next && snap_at && pos >= snap_at)
					break;

				model.remote_client.forward_begin(cache.rcache, pos, slice(ids, pos - start, bounds[reqs] - start));
				reqs++;
			}
		}

		function admit() {
			if (active[0] != null || next >= length(bounds))
				return;

			if (snap_at && prev_end == snap_at) {
				for (let c in active)
					if (c)
						return;
			}

			let end = bounds[next++];

			let t0 = clock(true);

			prefetch(min(total, end - start + ahead * chunk));

			let t1 = clock(true);
			let last_c = (end == start + total);

			issue();

			let c = model.chunk_begin(slice(ids, prev_end - start, end - start), cache, nr ? true : false);

			issue();
			let t2 = clock(true);

			if (getenv('AMP_DSV41_TP_HOSTPROF'))
				warn(sprintf('admit: prefetch %.1f ms, chunk_begin %.1f ms\n', (t1[0] - t0[0]) * 1e3 + (t1[1] - t0[1]) / 1e6,
					(t2[0] - t1[0]) * 1e3 + (t2[1] - t1[1]) / 1e6));

			if (opts.on_taps)
				c.taps = [];

			c.last = last_c;
			prev_end = end;
			active[0] = c;
		}

		function round() {
			let cs = [];

			for (let s = NS - 1; s >= 0; s--)
				if (active[s])
					push(cs, {c: active[s], s: s});

			if (!getenv('AMP_DSV41_TP_WHOLE'))
				model.chunk_stages(cs, cache);

			for (let s = NS - 1; s >= 0; s--) {
				let c = active[s];

				if (!c)
					continue;

				if (getenv('AMP_DSV41_TP_WHOLE'))
					model.chunk_stage(c, cache, s);

				if (s == NS - 1) {
					if (opts.on_taps)
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
	};

	/* ---- split over hosts (docs/RPC.md) ---- */

	model.remote = nr > 0;
	model.remote_layer = (i) => i < nr;

	/* The worker is gone (its host, the process or the link)? Wait for it to dial in again and give it empty caches;
	 * the engine then discards everything it cached. Also drops the replies that an aborted prefill left unread
	 * (the ping). false: nothing happened. Raises when the worker does not come back in time. */
	model.remote_recover = function (cache) {
		if (!nr || !model.remote_client || model.remote_client.alive(3000))
			return false;

		if (!model.remote_rejoin)
			die('dsv41_tp: the worker is gone');

		warn('dsv41_tp: the worker is gone, waiting for it to dial in again\n');

		model.remote_client = model.remote_rejoin(model.remote_client);
		cache.rcache = model.remote_client.new_cache(cache.max_seq);

		if (model.remote_sizes)
			model.remote_client.graphs(cache.rcache, model.remote_sizes);

		warn('dsv41_tp: the worker is back (empty caches)\n');

		return true;
	};

	model.reset_cache = function (cache) {
		model.aux_reset(cache);
		cache.length = 0;
		model.last_rows = null;
	};

	/* The worker's side. rows `ids` at `pos` through this (partial) model, the way the coordinator's chunk_begin and
	 * run_rows continue from them: eagerly, or by replaying the captured graph set G. The reply is one host tensor,
	 * float32 [n, ranks * (4 * hidden + 4)]: the four residual streams and the four mHC pre-weights of every row, per rank. */
	model.rpc_forward = function (cache, ids, pos, G) {
		let n = length(ids), h = null, pre = null;

		if (G) {
			let r = G.run(ids, pos).hidden;

			h = r.h;
			pre = r.pre;
		}
		else {
			cache.length = pos;

			let c = model.chunk_begin(ids, cache);

			for (let s = 0; s < NS; s++)
				model.chunk_stage(c, cache, s);

			h = c.h;
			pre = c.pre;
		}

		// (the ranks' replicas of the streams differ in the last bits: each rank's copy is sent, so that the ranks of the
		// coordinator's first stage continue from their own, as in a model on one host)
		let parts = [];

		for (let g = 0; g < length(h); g++)
			push(parts, torch.cat([ h[g].reshape([n, 4 * H]), pre[g] ], 1).cpu());

		return torch.cat(parts, 1).contiguous();
	};

	/* The state outside the pools that a snapshot of the cache holds (aux_get), as bytes for the connection:
	 * {layout, bytes}; and back. */
	model.aux_pack = function (cache) {
		let snap = model.aux_get(cache), lay = [], parts = [];

		for (let row in snap)
			for (let v in row) {
				push(lay, [ v.n, v.pk ? 1 : 0, v.kw ? v.kw.shape()[0] : 0 ]);

				if (v.pk) {
					push(parts, v.pk.reshape([-1]).cpu().view_dtype('uint8'));
					push(parts, v.pg.reshape([-1]).cpu().view_dtype('uint8'));
				}

				if (v.kw)
					push(parts, v.kw.contiguous().reshape([-1]).cpu().view_dtype('uint8'));
			}

		return {layout: lay, bytes: length(parts) ? torch.cat(parts, 0) : torch.zeros([0], 'uint8')};
	};

	model.aux_unpack = function (cache, layout, bytes) {
		let D = cfg.head_dim, off = 0, k = 0, snap = [];

		function take(n, dtype, shape) {
			let t = bytes.narrow(0, off, n).view_dtype(dtype).view(shape);

			off += n;

			return t;
		}

		for (let g = 0; g < Gs; g++) {
			let row = [];

			for (let i = 0; i < nl; i++) {
				let q = layout[k++], dev = cache.lanes[g].layers[i].dev;
				let v = {n: q[0], pk: null, pg: null, kw: null};

				if (q[1]) {
					v.pk = take(4 * D, 'float32', [ 1, D ]).cuda(dev);
					v.pg = take(4 * D, 'float32', [ 1, D ]).cuda(dev);
				}

				if (q[2])
					v.kw = take(2 * D * q[2], 'float16', [ q[2], D ]).cuda(dev);

				push(row, v);
			}

			push(snap, row);
		}

		model.aux_set(cache, snap);
	};

	dsv41_tp_state(model, allreduce);
	dsv41_tp_graphs(model, allreduce);
}

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = dsv41.load_config(model_dir);
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

	if (opts.layers && opts.layers < nl) {           // partial model (tests)
		nl = opts.layers;
		cfg.num_hidden_layers = nl;
	}

	if (G % NS)
		die(`dsv41_tp: ${G} devices do not split into ${NS} stages`);

	let Gs = G / NS;

	if (cfg.num_q_heads % Gs || cfg.o_groups % Gs || cfg.n_routed_experts % Gs)
		die(`dsv41_tp: ${Gs} ranks per stage do not divide heads / groups / experts`);

	// Split over hosts (docs/RPC.md): the first `nr` layers run on a worker host (opts.remote: the rpc_stage client),
	// this process holds the rest. A worker is itself a partial model: opts.layers = nr, opts.stages = 1, opts.no_head.
	let nr = opts.remote ? (opts.remote_layers || 0) : 0;
	let remote = nr ? opts.remote : null;

	if (nr) {
		// the hop carries only the streams: a layer behind it must not read the compressed pool or the top-k picks of one before it
		let first_c = -1;

		for (let i = nr; i < nl && first_c < 0; i++)
			if (cfg.ratios[i] > 0)
				first_c = i;

		if (nr >= nl || (first_c >= 0 && (index(cfg.kv_sources, first_c) < 0 || index(cfg.index_sources, first_c) < 0)))
			die(`dsv41_tp: the worker cannot take ${nr} layers: layer ${first_c} would read the pool of a layer on the worker (a boundary at a layer in ${join(',', cfg.kv_sources)} works)`);

		if (opts.spark)
			die('dsv41_tp: the DSpark drafter is not available with remote layers yet');
	}

	let max_seq = opts.max_seq || 262144;
	// with the drafter (opts.spark) every pipeline stage also holds one of its 3 stages, sharded over the pair
	let sb = (opts.spark && cfg.dspark_block && ws.has('mtp.0.main_proj.weight') && cfg.num_mtp_layers == NS) ?
		map(dsv41.spark_stage_bytes(ws, cfg), (b) => b / Gs) : null;
	let stage_of = dsv41.plan_devices(ws, cfg, NS, max_seq, 1000000000, 1000000000, sb, Gs, nr);
	let spark_stages = sb ? dsv41.SPARK_DEVS : null;

	// AMP_DSV41_TP_SPLIT=13,13,14: layers per stage (and AMP_DSV41_TP_SPARK=0,2,1: the stage of each drafter stage),
	// for hand-tuning the memory balance; 'auto' keeps the planner's. 13,13,14 is the default for 40 layers on 3
	// stages: the last stage's ratio-1 layers cost the most prefill time, and it fits at 256k context, chunk 1536
	let spl = getenv('AMP_DSV41_TP_SPLIT') || ((NS == 3 && nl == 40 && !nr) ? '13,13,14' : 'auto');

	if (spl != 'auto') {
		let sz = map(split(spl, ','), (x) => +x), at = 0;

		stage_of = [];

		for (let k = 0; k < nr; k++)         // (layers per stage: of the layers on this host)
			push(stage_of, -1);

		for (let s = 0; s < length(sz); s++)
			for (let k = 0; k < sz[s]; k++)
				push(stage_of, s);

		if (length(stage_of) != nl)
			die(`AMP_DSV41_TP_SPLIT: ${length(stage_of)} layers, the model has ${nl}`);

		if (sb)
			spark_stages = getenv('AMP_DSV41_TP_SPARK') ? map(split(getenv('AMP_DSV41_TP_SPARK'), ','), (x) => +x) : [ 0, 1, 2 ];
	}
	let bounds = '';

	for (let i = 1; i < nl; i++)
		if (stage_of[i] != stage_of[i - 1] && stage_of[i] >= 0)
			bounds += ' ' + i;

	warn(`dsv41_tp: ${NS} stage(s) x ${Gs} ranks; stages start at layers ${nr}${bounds}${nr ? ` (layers 0..${nr - 1} on the worker)` : ''}\n`);

	let cg = {...cfg, num_q_heads: int(cfg.num_q_heads / Gs), o_groups: int(cfg.o_groups / Gs)};
	let H = cfg.hidden_size;
	let eps = cfg.rms_norm_eps;
	let cap = P2P_ROWS * H;
	let cap_pf = (opts.prefill_rows || 1024) * H;
	let w1 = [];

	for (let r = 0; r < Gs; r++)
		push(w1, 1);

	let plan = {
		shared: dsv4_tp.split_n(cfg.moe_intermediate_size * cfg.n_shared_experts, 256, w1),
		experts: dsv4_tp.split_n(cfg.n_routed_experts, 1, w1),
		vocab: dsv4_tp.split_n(cfg.vocab_size, 256, w1)
	};
	let stages = [];

	for (let s = 0; s < NS; s++) {
		let sd = slice(devs, s * Gs, (s + 1) * Gs);
		let S = {id: s, devs: sd, G: Gs, first: -1, end: -1, expert_dev: {}, comm_cap: cap, comm_pf_cap: cap_pf};

		for (let r = 0; r < Gs; r++)
			for (let e = plan.experts[r][0]; e < plan.experts[r][0] + plan.experts[r][1]; e++)
				S.expert_dev[e] = sd[r];

		push(stages, S);
	}

	for (let i = nr; i < nl; i++) {
		let S = stages[stage_of[i]];

		if (S.first < 0)
			S.first = i;

		S.end = i + 1;
	}

	glm5.set_fused_rows(max(256, opts.prefill_rows || 1024));

	let meta = dsv41.build_engram_meta(model_dir, cfg);
	let layers = [];
	let engrams = {};
	let t_load = time();

	// (the layers on the worker: stubs that say what the layer is, for the loops over all layers)
	for (let i = 0; i < nr; i++)
		push(layers, map(slice(devs, 0, Gs), () => ({remote: true, idx: i,
			attn: {kv_source: index(cfg.kv_sources, i) >= 0, ratio: cfg.ratios[i], remote: true}})));

	for (let i = nr; i < nl; i++) {
		let S = stages[stage_of[i]];
		let st = S.devs[i % Gs];                  // staging device of the layer's replicated / sliced tensors
		let pfx = `layers.${i}.`;

		// experts straight to their owner, everything else to the staging device (sliced into the shards below)
		ws.upload(function (key) {
			if (index(key, pfx) != 0)
				return null;

			if (index(key, '.engram.embed') > 0)
				return null;                         // streamed from disk

			let at = index(key, '.experts.');

			if (at >= 0) {
				let rest = substr(key, at + 9);

				return S.expert_dev[+substr(rest, 0, index(rest, '.'))];
			}

			return st;
		});

		let attn = dsv41.build_attn(ws, cfg, i, st);
		let hc_a = dsv41.build_hc(ws, `layers.${i}`, 'attn', st);
		let hc_f = dsv41.build_hc(ws, `layers.${i}`, 'ffn', st);
		let an = ws.gpu16(`${pfx}attn_norm.weight`, st);
		let fnw = ws.gpu16(`${pfx}ffn_norm.weight`, st);
		let mkey = `${pfx}ffn`;
		let gate_t = dsv41.mat32(ws, mkey + '.gate', st);
		let bias = ws.gpu(mkey + '.gate.bias', st).to('float32').reshape([1, -1]).contiguous();
		let shared = dsv41.build_mlp8(ws, mkey + '.shared_experts', st);
		let row = [];

		for (let g = 0; g < Gs; g++) {
			let dev = S.devs[g];
			let e0 = plan.experts[g][0], en = plan.experts[g][1];
			let ex = [];

			for (let e = e0; e < e0 + en; e++)
				push(ex, {
					gate: models.linear(ws, `${mkey}.experts.${e}.w1`, false, dev),
					up: models.linear(ws, `${mkey}.experts.${e}.w3`, false, dev),
					down: models.linear(ws, `${mkey}.experts.${e}.w2`, false, dev)
				});

			let M = {
				type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en, router: 'sqrtsp',
				gate_t: gate_t.cuda(dev), bias: bias.cuda(dev), bias_vl: null, tid2eid: null, ids: null, vlon: false,
				async_moe: true, fused_prep: true, shared: shard_mlp8(shared, plan.shared[g], dev)
			};

			// (mixed integer widths: the per-expert tables; one half-integer width, the 2.50bpw packs: null, glm5.moe_begin
			// builds the uniform-width tables on first use)
			M.coop = getenv('AMP_DSV41_NO_COOP') ? false : (dsv41.build_coop(cfg, M) || null);

			push(row, {
				dev: dev, idx: i,
				hc_attn: hc_copy(hc_a, dev), hc_ffn: hc_copy(hc_f, dev),
				attn_norm: an.cuda(dev), ffn_norm: fnw.cuda(dev),
				attn: shard_attn(cfg, attn, g, Gs, dev),
				moe: M
			});
		}

		push(layers, row);

		// the Engram of layers 1 / 14: a copy on every rank (the table rows are streamed from disk)
		if (index(cfg.engram_layers, i) >= 0) {
			let E = dsv41.build_engram(ws, cfg, i, st);
			let per = [];

			for (let g = 0; g < Gs; g++) {
				let dev = S.devs[g];
				let Eg = {layer: i, dev: dev, key: E.key, wkv: dsv41.lin8_to(E.wkv, dev), weight: E.weight.cuda(dev),
					file: E.file, wname: E.wname, sname: E.sname, emb_bufs: {},
					emb_buf: torch.zeros([1, E.emb_buf.shape()[1]], 'float16', 'cuda:' + dev)};

				push(per, Eg);
			}

			engrams[i] = per;
		}

		for (let key in ws.list())
			if (index(key, pfx) == 0 && index(key, '.experts.') < 0 && index(key, '.engram.embed') < 0)
				ws.release(key, st);

		attn = shared = null;

		if (i % 4 == 3)
			gc();
	}

	gc();

	let last = stages[stage_of[nl - 1]];

	let ed = devs[0];
	let no_head = opts.no_head ? true : false;       // a worker: the embedding, no head

	ws.upload(function (key) {
		if (key == 'embed.weight')
			return remote ? null : ed;

		if ((substr(key, 0, 5) == 'head.' || key == 'norm.weight') && !no_head)
			return last.devs[0];

		return null;
	});

	// (with remote layers the worker embeds: this process never sees the table)
	let embed_t = remote ? null : ws.gpu16('embed.weight', ed);

	if (!remote)
		ws.release('embed.weight', ed);

	let head_full = no_head ? null : models.linear(ws, 'head', false, last.devs[0]);
	// rank g multiplies its vocabulary slice: the head is a dense fp16 matrix (1.59bpw pack) or EXL3 (1017 packs)
	let lm_heads = no_head ? [] : map(last.devs, function (d, g) {
		if (head_full.trellis)
			return models.cols(head_full, [ [ plan.vocab[g][0], plan.vocab[g][1] ] ], d);

		let wg = head_full.weight.narrow(0, plan.vocab[g][0], plan.vocab[g][1]).contiguous().cuda(d);

		return {weight: wg, in_features: wg.shape()[1], out_features: wg.shape()[0],
			forward: function (x) {return torch.matmul(x, this.weight.t());}};
	});
	let norm_w = no_head ? null : ws.gpu16('norm.weight', last.devs[0]);

	if (!no_head)
		for (let sfx in [ 'weight', 'trellis', 'suh', 'svh', 'mul1' ])
			ws.release('head.' + sfx, last.devs[0]);

	head_full = null;
	gc();

	warn(sprintf('dsv41_tp: %d ranks (%d stage%s) loaded in %.0fs\n', G, NS, NS > 1 ? 's' : '', time() - t_load));
	dsv41.mem_report(devs, 'weights loaded');
	torch.empty_cache();
	dsv41.mem_report(devs, 'weights loaded, cache emptied');

	for (let S in stages) {
		S.comm = torch.p2p_comm(S.devs, cap);

		if (!getenv('AMP_DSV41_TP_OV')) {
			S.comm_pf = torch.p2p_comm(S.devs, cap_pf);
		}
		else {
			// two half-chunk communicators for the double-buffered prefill; the first also serves the small chunks
			S.comm_ov_rows = int(((opts.prefill_rows || 1024) + 1) / 2);
			S.comm_ov = [ torch.p2p_comm(S.devs, S.comm_ov_rows * H), torch.p2p_comm(S.devs, S.comm_ov_rows * H) ];
			S.comm_pf = S.comm_ov[0];
			S.comm_pf_cap = S.comm_ov_rows * H;
		}
	}

	let model = {
		cfg: cfg, cg: cg, layers: layers, devs: devs, G: G, tp: true, nl: nl, ws: ws, plan: plan, Gs: Gs, NS: NS,
		stages: stages, stage_of: stage_of, last: last, engrams: engrams, meta: meta, spark_stages: spark_stages,
		embed_tokens: embed_t, embed_dev: ed, lm_heads: lm_heads, lm_head: lm_heads[0],
		norm: no_head ? [] : map(last.devs, (d) => norm_w.cuda(d)),
		nr: nr, remote_client: remote, remote_rejoin: opts.recover, no_head: no_head
	};

	dsv41_tp_methods(model);

	return model;
}

/* How many layers a worker host takes (docs/RPC.md): the first `nr`, the largest count that fits the worker's GPUs
 * (weights split over its ranks, the pools of its source layers, the embedding on its first GPU, and a working set for
 * graphs, communicators and the prefill chunk), and ends where a new compressed-pool group starts (layer 2, 8, 14, ...:
 * the hop carries only the streams). o: {gpus: [total bytes of each worker GPU], max_seq, layers (a forced count)}.
 * Returns {nr, per_gpu: bytes of the busiest worker GPU}. */
export function remote_plan(model_dir, o) {
	let cfg = dsv41.load_config(model_dir);
	let ws = weights.open(model_dir);
	let Gw = length(o.gpus);
	let bytes = map(dsv41.layer_bytes(ws, cfg), (b) => b / Gw);
	let embed = ws.info('embed.weight').nbytes;
	let work = 5 * 1073741824;
	let budget = min(...o.gpus) * 0.93;
	let best = 0, best_cost = 0;

	for (let b = 1; b < cfg.num_hidden_layers; b++) {
		if (index(cfg.kv_sources, b) < 0 || index(cfg.index_sources, b) < 0)
			continue;

		let cost = dsv41.seg_cost(cfg, bytes, 0, b, o.max_seq) + embed + work;

		if (o.layers ? b == o.layers : cost <= budget) {
			best = b;
			best_cost = cost;
		}
	}

	if (!best)
		die(o.layers ? `dsv41_tp: ${o.layers} layers cannot be a worker's share (a boundary at ${join(',', cfg.kv_sources)})` :
			'dsv41_tp: the worker\'s GPUs cannot hold even the first layers');

	return {nr: best, per_gpu: best_cost, budget: budget};
}
