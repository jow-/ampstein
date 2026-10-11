// ucode/dsv4_tp.uc — DeepSeek-V4-Flash as a PIPELINE OF TENSOR-PARALLEL STAGES
// (the layout of glm5_tp --stages: `--tp --stages 3` = three pairs of GPUs).
//
// One stage = a group of Gs ranks that all run every layer of the stage's
// contiguous layer range, each on its share; the ranks' partial sublayer
// outputs are summed by an all-reduce and the residual streams (4 x fp32 per
// token) are replicated bit-identically inside the group. A decode token then
// reads 1/Gs of a stage's weights per GPU; prefill is a wavefront over the
// stages.
//
// Sharding inside a group:
//   attention  query heads (q_b columns, sinks) and o_groups (wo_a slices;
//              wo_b by ROWS = partial sum); q_a, wkv, the compressors, the
//              indexer and the window K=V cache are replicated (one shared
//              KV head: replicating it is cheap) — dsv4.attn_forward /
//              attn_decode run unchanged on a config copy with the local
//              head / group counts
//   MoE        expert parallel: experts [e0, e1) per rank (the fused MoE
//              kernel filters the router's picks to its range), the shared
//              expert sharded like a dense MLP (gate/up columns, down rows);
//              the router is replicated
//   head       by vocabulary columns over the LAST stage's ranks, logits
//              gathered on its rank 0; embedding on rank 0 of stage 0
//
// let m = dsv4_tp.load(dir, {devices: 6, stages: 3, max_seq: 65536});

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';

const P2P_ROWS = 8;
const FAST_PF = getenv('AMP_DSV4_PF32') ? false : true;      // fused prefill path (device-twin compressor state)
const SIDE_HO = getenv('AMP_DSV4_NO_SIDE_HO') ? false : true;        // stage hand-off copies on a side stream

/* [start, count] ranges of `total` units of `unit` split by weights w */
export function split_n(total, unit, w) {
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

export function copy_lin(lin, dev) {
	return models.cols(lin, [ [ 0, lin.out_features ] ], dev);
}

export function build_mlp(ws, key, dev) {
	return {
		gate: models.linear(ws, key + '.w1', false, dev),
		up: models.linear(ws, key + '.w3', false, dev),
		down: models.linear(ws, key + '.w2', false, dev)
	};
}

/* gate/up by columns, down by rows over the intermediate range ri */
export function shard_mlp(M, ri, dev) {
	return {
		gate: models.cols(M.gate, [ ri ], dev),
		up: models.cols(M.up, [ ri ], dev),
		down: models.rows(M.down, ri[0], ri[1], dev)
	};
}

export function hc_to(hc, dev) {
	return {fn: hc.fn.cuda(dev), base: hc.base.cuda(dev), scale: hc.scale.cuda(dev)};
}

function copy_comp(C, dev) {
	if (!C)
		return null;

	return {
		hd: C.hd, m: C.m, overlapping: C.overlapping, W: C.W, dev: dev,
		wkv: copy_lin(C.wkv, dev), wgate: copy_lin(C.wgate, dev),
		ape: C.ape.cuda(dev), norm_w: C.norm_w.cuda(dev), inv_freq: C.inv_freq.cuda(dev)
	};
}

/* Rank g's attention shard (heads and groups [g * n, (g + 1) * n)). */
export function shard_attn(cfg, A, g, Gs, dev) {
	let nh = int(cfg.num_q_heads / Gs), ng = int(cfg.o_groups / Gs), D = cfg.head_dim;
	let R = {
		dev: dev, kind: A.kind, m: A.m,
		inv_freq: A.inv_freq.cuda(dev), inv_comp: A.inv_comp.cuda(dev),
		q_a: copy_lin(A.q_a, dev), q_norm: A.q_norm.cuda(dev),
		q_b: models.cols(A.q_b, [ [ g * nh * D, nh * D ] ], dev),
		wkv: copy_lin(A.wkv, dev), kv_norm: A.kv_norm.cuda(dev),
		wo_a: [],
		wo_b: models.rows(A.wo_b, g * ng * cfg.o_lora_rank, ng * cfg.o_lora_rank, dev),
		sinks: A.sinks.narrow(0, g * nh, nh).contiguous().cuda(dev),
		comp: copy_comp(A.comp, dev), idx: copy_comp(A.idx, dev)
	};

	for (let j = 0; j < ng; j++)
		push(R.wo_a, copy_lin(A.wo_a[g * ng + j], dev));

	if (A.idx) {
		R.idx_wq_b = copy_lin(A.idx_wq_b, dev);
		R.idx_w = A.idx_w.cuda(dev);
	}

	return R;
}

/* Sum of one fp16 partial per rank of the stage -> per-rank sums (bit-
 * identical): small payloads through the stage's P2P communicator, larger
 * ones through the prefill communicator or by peer copies. */
export function allreduce(S, p) {
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

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = dsv4.load_config(model_dir);
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
		die(`dsv4_tp: ${G} devices do not split into ${NS} stages`);

	let Gs = G / NS;

	if (cfg.num_q_heads % Gs || cfg.o_groups % Gs || cfg.n_routed_experts % Gs)
		die(`dsv4_tp: ${Gs} ranks per stage do not divide heads / groups / experts`);

	let stage_of = [];

	if (NS > 1) {
		let extra = 0;

		if (opts.drafter)                 // the DSpark blocks (mtp.*) live with the last stage
			for (let key in ws.list())
				if (index(key, 'mtp.') == 0)
					extra += ws.info(key).nbytes;

		stage_of = dsv4.plan_devices(ws, cfg, NS, extra);
	}
	else {
		for (let i = 0; i < nl; i++)
			push(stage_of, 0);
	}
	let bounds = '';

	for (let i = 1; i < nl; i++)
		if (stage_of[i] != stage_of[i - 1])
			bounds += ' ' + i;

	warn(`dsv4_tp: ${NS} stage(s) x ${Gs} ranks; stages start at layers 0${bounds}\n`);

	let cg = {...cfg, num_q_heads: int(cfg.num_q_heads / Gs), o_groups: int(cfg.o_groups / Gs)};
	let cap = P2P_ROWS * cfg.hidden_size;
	let cap_pf = (opts.prefill_rows || 1024) * cfg.hidden_size;
	let w1 = [];

	for (let r = 0; r < Gs; r++)
		push(w1, 1);

	let stages = [];
	let plan = {
		shared: split_n(cfg.moe_intermediate_size * cfg.n_shared_experts, 256, w1),
		experts: split_n(cfg.n_routed_experts, 1, w1),
		vocab: split_n(cfg.vocab_size, 256, w1)
	};

	for (let s = 0; s < NS; s++) {
		let sd = slice(devs, s * Gs, (s + 1) * Gs);
		let S = {id: s, devs: sd, G: Gs, first: -1, end: -1, expert_dev: {}, comm_cap: cap, comm_pf_cap: cap_pf};

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

	let layers = [];
	let t_load = time();

	glm5.set_fused_rows(max(256, opts.prefill_rows || 1024));

	for (let i = 0; i < nl; i++) {
		let S = stages[stage_of[i]];
		let st = S.devs[i % Gs];
		let pfx = `layers.${i}.`;

		// experts straight to their owner, everything else to the layer's
		// staging device (sliced into the shards below)
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

		let attn = dsv4.build_attn(ws, cfg, i, st);
		let hc_a = dsv4.build_hc(ws, i, 'attn', st);
		let hc_f = dsv4.build_hc(ws, i, 'ffn', st);
		let an = ws.gpu16(`${pfx}attn_norm.weight`, st);
		let fn_ = ws.gpu16(`${pfx}ffn_norm.weight`, st);
		let mkey = `${pfx}ffn`;
		let gate_t = ws.gpu(mkey + '.gate.weight', st).to('float32').t().contiguous();
		let bias = ws.gpu(mkey + '.gate.bias', st).to('float32').reshape([1, -1]).contiguous();
		let bias_vl = ws.has(mkey + '.gate.bias_vl') ?
			ws.gpu(mkey + '.gate.bias_vl', st).to('float32').reshape([1, -1]).contiguous() : null;
		let tid = (i < cfg.num_hash_layers && ws.has(mkey + '.gate.tid2eid')) ?
			ws.gpu(mkey + '.gate.tid2eid', st).to('int64').contiguous() : null;
		let shared = build_mlp(ws, mkey + '.shared_experts', st);
		let row = [];

		for (let g = 0; g < Gs; g++) {
			let dev = S.devs[g];
			let e0 = plan.experts[g][0], en = plan.experts[g][1];
			let ex = [];

			for (let e = e0; e < e0 + en; e++)
				push(ex, build_mlp(ws, `${mkey}.experts.${e}`, dev));

			push(row, {
				dev: dev, kind: cfg.layer_types[i], idx: i,
				hc_attn: hc_to(hc_a, dev), hc_ffn: hc_to(hc_f, dev),
				attn_norm: an.cuda(dev), ffn_norm: fn_.cuda(dev),
				attn: shard_attn(cfg, attn, g, Gs, dev),
				moe: {
					type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en, router: 'sqrtsp',
					async_moe: !getenv('AMP_DSV4_SYNC_MOE'),
					gate_t: gate_t.cuda(dev), bias: bias.cuda(dev), bias_vl: bias_vl ? bias_vl.cuda(dev) : null,
					tid2eid: tid ? tid.cuda(dev) : null, ids: null,
					shared: shard_mlp(shared, plan.shared[g], dev)
				}
			});
		}

		for (let R in row) {
			dsv4.build_fans(R.attn);
			dsv4.build_gu(R.moe.shared, R.dev);
		}

		push(layers, row);

		for (let key in ws.list())
			if (index(key, pfx) == 0 && index(key, '.experts.') < 0)
				ws.release(key, st);

		attn = shared = null;

		if (i % 4 == 3)
			gc();
	}

	gc();

	let first = stages[0], last = stages[stage_of[nl - 1]];

	ws.upload(function (key) {
		if (key == 'embed.weight')
			return devs[0];

		if (index(key, 'head.') == 0 || index(key, 'norm.') == 0 || index(key, 'hc_head') == 0)
			return last.devs[0];

		return null;
	});

	let embed_t = ws.gpu16('embed.weight', devs[0]);

	ws.release('embed.weight', devs[0]);

	let head_full = models.linear(ws, 'head', false, last.devs[0]);
	let lm_heads = map(last.devs, (d, g) => models.cols(head_full, [ plan.vocab[g] ], d));
	let norm_w = ws.gpu16('norm.weight', last.devs[0]);
	let hh = {
		fn: ws.gpu('hc_head_fn', last.devs[0]).to('float32').contiguous(),
		base: ws.gpu('hc_head_base', last.devs[0]).to('float32').contiguous(),
		scale: ws.gpu('hc_head_scale', last.devs[0]).to('float32').contiguous()
	};

	head_full = null;
	gc();

	warn(sprintf('dsv4_tp: %d ranks (%d stage%s) loaded in %.0fs\n', G, NS, NS > 1 ? 's' : '', time() - t_load));

	for (let S in stages) {
		S.comm = torch.p2p_comm(S.devs, cap);
		S.comm_pf = torch.p2p_comm(S.devs, cap_pf);
	}

	let eps = cfg.rms_norm_eps;
	let H = cfg.hidden_size;

	return {
		cfg: cfg, cg: cg, layers: layers, devs: devs, G: G, tp: true, nl: nl, ws: ws, plan: plan, Gs: Gs, NS: NS,
		stages: stages, stage_of: stage_of, last: last,
		embed_tokens: embed_t, lm_heads: lm_heads, lm_head: lm_heads[0], mm_build: dsv4.mm_build,
		norm: map(last.devs, (d) => norm_w.cuda(d)),
		hc_heads: map(last.devs, (d) => hc_to(hh, d)),

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};

			for (let i = 0; i < nl; i++)
				push(cache.layers, map(layers[i], (R) => dsv4.new_attn_cache(cg, R.attn, max_seq)));

			return cache;
		},

		/* the replicated streams of one stage copied to the next: rank r
		 * receives from rank r of the sender */
		handoff: function (h, to, keep) {
			if (!SIDE_HO || !keep || to < 1)
				return map(stages[to].devs, (d, r) => h[r].cuda(d));

			// copies on a side stream of each source rank (see dsv4.uc), sources kept alive in `keep`
			let out = [];

			for (let r = 0; r < length(stages[to].devs); r++) {
				let sd = stages[to - 1].devs[r];

				torch.stream_side(sd, 1);
				push(out, h[r].cuda(stages[to].devs[r]));
				torch.stream_main(sd, 1);
				push(keep, h[r]);
			}

			return out;
		},

		/* token ids on every rank of a stage */
		ids_to: function (ids0, s) {
			return map(stages[s].devs, (d) => ids0.cuda(d));
		},

		/* Layer i on all ranks of its stage, in two halves (as glm5_tp):
		 * layer_a runs the attention site and enqueues the ffn up to the
		 * routing, layer_b finishes the ffn. h: per-rank streams (T, 4,
		 * hidden) fp32 (in place); ids: per-rank int64 (T) token ids. */
		layer_a: function (i, h, cache, pos, ids, fl) {
			let S = stages[stage_of[i]];
			let L = layers[i];
			let p = [], a = [];

			for (let g = 0; g < Gs; g++) {
				cache.layers[i][g].ncs = fl ? fl.ncs : -1;
				cache.layers[i][g].vlon = fl ? fl.vlon : false;

				let r = dsv4.hc_collapse(cg, L[g].hc_attn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].attn_norm, eps);

				push(a, r);
				push(p, dsv4.attn_forward(cg, L[g].attn, xn, cache.layers[i][g], pos));
			}

			let s = allreduce(S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(h[g], s[g], a[g][0], a[g][1]);

			let f = [], pend = [];

			for (let g = 0; g < Gs; g++) {
				let r = dsv4.hc_collapse(cg, L[g].hc_ffn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].ffn_norm, eps);

				L[g].moe.ids = ids[g];
				L[g].moe.vlon = fl ? fl.vlon : false;
				push(f, r);
				push(pend, glm5.moe_begin(cg, L[g].moe, xn));
			}

			return {S: S, L: L, h: h, f: f, pend: pend};
		},

		layer_b: function (c) {
			let p = [];

			for (let g = 0; g < Gs; g++)
				push(p, glm5.moe_finish(cg, c.L[g].moe, c.pend[g]).to('float16'));

			let s = allreduce(c.S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(c.h[g], s[g], c.f[g][0], c.f[g][1]);

			return c.h;
		},

		/* One decode step of layer i (S rows, device-side positions st[g]) */
		layer_dec: function (i, h, cache, st, ids) {
			return this.block_dec(stages[stage_of[i]], layers[i], cache.layers[i], h, st, ids);
		},

		/* A decode block on the ranks of stage S: L / cls = per-rank layer objects /
		 * cache objects (also used for the DSpark drafter blocks) */
		block_dec: function (S, L, cls, h, st, ids) {
			let p = [], a = [];

			for (let g = 0; g < Gs; g++) {
				let r = dsv4.hc_collapse(cg, L[g].hc_attn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].attn_norm, eps);

				push(a, r);
				push(p, dsv4.attn_decode(cg, L[g].attn, xn, cls[g], st[g]));
			}

			let s = allreduce(S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(h[g], s[g], a[g][0], a[g][1]);

			let f = [];

			p = [];

			for (let g = 0; g < Gs; g++) {
				let r = dsv4.hc_collapse(cg, L[g].hc_ffn, h[g]);
				let xn = torch.rms_norm(r[2], L[g].ffn_norm, eps);

				L[g].moe.ids = ids ? ids[g] : null;
				L[g].moe.vlon = false;
				push(f, r);
				push(p, glm5.moe_forward(cg, L[g].moe, xn).to('float16'));
			}

			s = allreduce(S, p);

			for (let g = 0; g < Gs; g++)
				glm5.hc_apply(h[g], s[g], f[g][0], f[g][1]);

			return h;
		},

		/* final collapse + norm + sharded head over the last stage's ranks;
		 * rows: how many trailing rows get logits. Returns logits on the
		 * last stage's rank 0. */
		head: function (h, rows) {
			let xs = [];

			for (let g = 0; g < last.G; g++)
				push(xs, glm5.rms_norm_f32(dsv4.hc_head(cg, this.hc_heads[g], h[g]), this.norm[g], eps));

			return this.head_state(xs, rows);
		},

		/* the sharded head on per-rank normed states xs[g] (T, hidden) fp16 */
		head_state: function (xs, rows) {
			let T = xs[0].shape()[0];
			let ls = [];

			for (let g = 0; g < last.G; g++) {
				let x = (rows < T) ? xs[g].narrow(0, T - rows, rows).contiguous() : xs[g];
				let l = this.lm_heads[g].forward(x);

				push(ls, (g == 0) ? l : l.cuda(last.devs[0]));
			}

			return torch.cat(ls, -1);
		},

		embed_streams: function (ids0, T, mm, ids, pos) {
			let e = mm ? dsv4.embed_rows(this.embed_tokens, mm, ids, pos, devs[0]) : this.embed_tokens.index(ids0);
			let h0 = e.to('float32').reshape([T, 1, H])
				.expand([T, 4, H]).contiguous();

			return map(stages[0].devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));
		},

		/* eager forward of token ids appended to the cache */
		forward_chunk: function (token_ids, cache, rows) {
			if (FAST_PF)
				this.to_dev(cache);
			else
				this.to_eager(cache);

			let T = length(token_ids);
			let pos = cache.length;
			let ids0 = torch.from_array(token_ids, 'int64', 'cuda:' + devs[0]);
			let h = this.embed_streams(ids0, T, cache.mm, token_ids, pos);
			let cur = 0, idr = this.ids_to(ids0, 0);
			let fl = dsv4.chunk_flags(cache.mm, pos, T);

			if (rows == null)
				rows = 1;

			for (let i = 0; i < nl; i++) {
				if (stage_of[i] != cur) {
					cur = stage_of[i];
					h = this.handoff(h, cur);
					idr = this.ids_to(ids0, cur);
				}

				this.layer_b(this.layer_a(i, h, cache, pos, idr, fl));
			}

			cache.length = pos + T;

			return (rows == 0) ? null : this.head(h, rows);
		},

		prefill: function (token_ids, cache, chunk, rows) {
			let n = length(token_ids);

			chunk = chunk || 512;

			for (let p = 0; p < n; p += chunk) {
				let e = min(p + chunk, n);
				let last_c = (e == n);
				let r = this.forward_chunk(slice(token_ids, p, e), cache, last_c ? (rows == null ? 1 : rows) : 0);

				if (last_c)
					return r;
			}

			return null;
		},

		/* wavefront over the stages: one layer per stage per round */
		prefill_pipeline: function (cache, ids, opts) {
			let model = this;
			let chunk = opts.chunk || 1024;
			let start = cache.length, total = length(ids);
			let bounds = [];

			if (FAST_PF)
				this.to_dev(cache);
			else
				this.to_eager(cache);

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

			let active = [];
			let next = 0, prev_end = start, done = 0, result = null;
			let snap_at = (opts.cuts && length(opts.cuts)) ? opts.cuts[0] : 0;

			for (let s = 0; s < NS; s++)
				push(active, null);

			function admit() {
				if (active[0] != null || next >= length(bounds))
					return;

				if (snap_at && prev_end == snap_at)
					for (let c in active)
						if (c)
							return;

				let end = bounds[next++];
				let pos = prev_end;
				let T = end - pos;
				let cids = slice(ids, pos - start, end - start);
				let ids0 = torch.from_array(cids, 'int64', 'cuda:' + devs[0]);

				prev_end = end;
				active[0] = {pos: pos, T: T, end: end, layer: stages[0].first, ctx: null,
					fl: dsv4.chunk_flags(cache.mm, pos, T),
					ids0: ids0, ids: model.ids_to(ids0, 0), h: model.embed_streams(ids0, T, cache.mm, cids, pos)};
			}

			function round() {
				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.layer < stages[s].end && c.ctx == null)
						c.ctx = model.layer_a(c.layer, c.h, cache, c.pos, c.ids, c.fl);
				}

				for (let s = 0; s < NS; s++) {
					let c = active[s];

					if (c && c.ctx) {
						c.h = model.layer_b(c.ctx);
						c.ctx = null;

						// DSpark taps: stream mean after the target layers (on rank 0 of the stage)
						if (opts.on_taps && index(cfg.dspark_taps, c.layer) >= 0) {
							if (!c.taps)
								c.taps = [];

							push(c.taps, c.h[0].mean(1).clamp(-65504.0, 65504.0).to('float16'));
						}

						c.layer++;

						// queue this stage's next layer before the host blocks on another
						// stage's expert counts
						if (c.layer < stages[s].end)
							c.ctx = model.layer_a(c.layer, c.h, cache, c.pos, c.ids, c.fl);
					}
				}

				for (let s = NS - 1; s >= 0; s--) {
					let c = active[s];

					if (!c || c.layer < stages[s].end)
						continue;

					if (s == NS - 1) {
						if (c.end == start + total)
							result = model.head(c.h, 1);

						if (opts.on_taps && c.taps)
							opts.on_taps(torch.cat(map(c.taps, (t) => t.cuda(last.devs[0])), 1), c.pos);

						cache.length = c.end;
						done = c.end - start;
						active[s] = null;

						if (opts.on_aux)
							opts.on_aux(c.end);
					}
					else if (active[s + 1] == null) {
						c.keep = c.keep || [];
						c.h = model.handoff(c.h, s + 1, c.keep);
						c.ids = model.ids_to(c.ids0, s + 1);
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

		/* eager <-> device-resident compressor state, every rank */
		to_eager: function (cache) {
			if (cache.dmode != 'dev')
				return;

			for (let i = 0; i < nl; i++)
				for (let g = 0; g < Gs; g++) {
					let A = layers[i][g].attn, cl = cache.layers[i][g];

					if (A.comp)
						dsv4.comp_to_eager(A.comp, cl.comp, cache.length);

					if (A.idx)
						dsv4.comp_to_eager(A.idx, cl.idx, cache.length);
				}

			cache.dmode = 'eager';
		},

		to_dev: function (cache) {
			if (cache.dmode == 'dev')
				return;

			for (let i = 0; i < nl; i++)
				for (let g = 0; g < Gs; g++) {
					let A = layers[i][g].attn, cl = cache.layers[i][g];

					if (A.comp)
						dsv4.comp_to_dev(A.comp, cl.comp, cache.length);

					if (A.idx)
						dsv4.comp_to_dev(A.idx, cl.idx, cache.length);
				}

			cache.dmode = 'dev';
		},

		aux_get: function (cache) {
			this.to_eager(cache);

			return map(cache.layers, (row) => map(row, function (cl) {
				let e = map([ cl.comp, cl.idx ], (S) => S ?
					{n: S.n, bk: S.buf_kv, bg: S.buf_gate, ok: S.o_kv, og: S.o_gate} : null);

				push(e, {kw: dsv4.kv_window_get(cl, cache.length, cfg.window)});

				return e;
			}));
		},

		aux_set: function (cache, snap) {
			cache.dmode = 'eager';

			for (let row in cache.layers)
				for (let cl in row) {
					if (cl.comp) cl.comp.dmode = false;
					if (cl.idx) cl.idx.dmode = false;
				}

			for (let i = 0; i < nl; i++)
				for (let g = 0; g < Gs; g++) {
					let cl = cache.layers[i][g];
					let which = [ cl.comp, cl.idx ];

					dsv4.kv_window_set(cl, cache.length, snap[i][g][2].kw);

					for (let k = 0; k < 2; k++) {
						let S = which[k], v = snap[i][g][k];

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

		/* Host-RAM prompt-cache store (engine hooks; see dsv4.uc). Per layer, per rank. */
		cache_bytes: function (cache, n) {
			let b = 0;

			for (let i = 0; i < nl; i++)
				for (let g = 0; g < Gs; g++)
					b += dsv4.store_layer_bytes(cfg, layers[i][g].attn, n);

			if (this.dspark && this.dspark.ring_bytes)
				b += this.dspark.ring_bytes(n);

			return b;
		},

		new_slot: function (max_seq) {
			return {layers: map(layers, (row) => map(row, (R) => dsv4.new_slot_layer(R.attn, max_seq)))};
		},

		cache_export: function (cache, n, put, snap_aux, dst) {
			let aux = this.aux_get(cache);
			let out = [];

			for (let i = 0; i < nl; i++)
				push(out, map(cache.layers[i], (cl, g) => ({
					pools: dsv4.pools_to_host(cl, put, dst ? dst.layers[i][g] : null),
					aux: dsv4.auxe_to_host(aux[i][g], put)})));

			return {
				layers: out,
				snap: snap_aux ? map(snap_aux, (row) => map(row, (e) => dsv4.auxe_to_host(e, put))) : null,
				drafter: (this.dspark && this.dspark.ring_export) ? this.dspark.ring_export(n, put) : null
			};
		},

		cache_import: function (cache, em, start, live) {
			let dev_aux = [];

			for (let i = 0; i < nl; i++) {
				let row = [];

				for (let g = 0; g < Gs; g++) {
					let cl = cache.layers[i][g];
					let ae = live ? em.layers[i][g].aux : em.snap[i][g];

					dsv4.pools_from_host(cl, em.layers[i][g].pools, dsv4.aux_rows(ae));
					push(row, dsv4.auxe_to_dev(ae, cl.dev));
				}

				push(dev_aux, row);
			}

			cache.length = start;
			this.aux_set(cache, dev_aux);

			if (em.drafter && this.dspark && this.dspark.ring_import)
				this.dspark.ring_import(em.drafter);

			return {snap: em.snap ? map(em.snap, (row, i) =>
				map(row, (e, g) => dsv4.auxe_to_dev(e, cache.layers[i][g].dev))) : null};
		},

		aux_reset: function (cache) {
			cache.dmode = 'eager';

			for (let row in cache.layers)
				for (let cl in row) {
					if (cl.comp) cl.comp.dmode = false;
					if (cl.idx) cl.idx.dmode = false;
				}

			for (let row in cache.layers)
				for (let cl in row)
					for (let S in [ cl.comp, cl.idx ])
						if (S) {
							S.n = 0;
							S.buf_kv = S.buf_gate = S.o_kv = S.o_gate = null;
						}
		},

		/* CUDA-graph decode: per step size S (1 = plain decode, W = a speculative
		 * verify window of W consecutive tokens) ONE multi-GPU graph captured on
		 * the first device with all other devices as side devices; positions are
		 * device buffers per rank, the all-reduces need no graph edges. Rolling
		 * back a verify step is just resetting the position (see amp_dsv4.cu).
		 * run(ids, pos) -> {logits (S, V) on the last stage's rank 0, taps
		 * (S, 3 * hidden) fp16 — the DSpark target taps of the S rows}. */
		graphs: function (cache, sizes) {
			let tp = this;
			let sts = map(devs, (d) => dsv4.new_dec_dev(cfg, d));
			let by_size = {};
			let tapdev = last.devs[0];

			tp.dec_states = sts;

			function build(S) {
				let ent = {S: S, ids: torch.zeros([S], 'int64', 'cuda:' + devs[0]), bar: [],
					taps: map(cfg.dspark_taps, () => torch.zeros([S, H], 'float16', 'cuda:' + tapdev))};

				for (let g = 0; g < G; g++)
					push(ent.bar, torch.zeros([1], 'float32', 'cuda:' + devs[g]));

				ent.body = function () {
					let h = tp.embed_streams(ent.ids, S);
					let cur = 0, idr = tp.ids_to(ent.ids, 0);

					for (let i = 0; i < nl; i++) {
						if (stage_of[i] != cur) {
							cur = stage_of[i];
							h = tp.handoff(h, cur);
							idr = tp.ids_to(ent.ids, cur);
						}

						h = tp.layer_dec(i, h, cache, slice(sts, cur * Gs, (cur + 1) * Gs), idr);

						let ti = index(cfg.dspark_taps, i);

						if (ti >= 0)
							ent.taps[ti].copy_(h[0].mean(1).clamp(-65504.0, 65504.0).to('float16'));
					}

					ent.out = tp.head(h, S);
					ent.am = ent.out.argmax(-1);
				};

				return ent;
			}

			cache.dmode = 'eager';
			tp.to_dev(cache);

			for (let sz in (sizes || [ 1 ])) {
				if (sz > P2P_ROWS)
					continue;

				let ent = build(sz);

				for (let st in sts)
					dsv4.dec_set_pos(st, 0);

				ent.body();                                    // eager warm-up
				torch.synchronize();

				ent.graph = torch.graph();
				ent.graph.capture_begin(devs[0], slice(devs, 1));
				glm5.set_par(!getenv('AMP_DSV4_NO_PAR'));          // forked side streams (shared expert)
				ent.body();
				glm5.set_par(false);
				ent.graph.capture_end();
				torch.synchronize();
				by_size[sz] = ent;
			}

			cache.dmode = 'eager';                            // the warm-up touched the twins

			function replay(ent, pos) {
				tp.to_dev(cache);

				for (let st in sts)
					dsv4.dec_set_pos(st, pos);

				for (let g = 1; g < G; g++)
					ent.bar[g].copy_(ent.bar[0]);

				ent.graph.replay();

				for (let g = 1; g < G; g++)
					ent.bar[g].copy_(ent.bar[0]);

				cache.length = pos + ent.S;

				return {logits: ent.out, am: ent.am, ent: ent,
					taps: (length(ent.taps) == 1) ? ent.taps[0] : torch.cat(ent.taps, 1)};
			}

			return {
				has: function (S) {return by_size[S] != null;},

				run: function (ids, pos) {
					let ent = by_size[length(ids)];

					if (!ent)
						die(`dsv4_tp: no graph captured for step size ${length(ids)}`);

					ent.ids.copy_(torch.from_array(ids, 'int64', 'cuda:' + devs[0]));

					return replay(ent, pos);
				},

				run_dev: function (ids_dev, pos) {
					let ent = by_size[ids_dev.shape()[1]];

					ent.ids.copy_(ids_dev.view([-1]));

					return replay(ent, pos);
				},

				/* replay with ent.ids already filled by the caller (device-side) */
				replay_size: function (S, pos) {
					return replay(by_size[S], pos);
				},

				ent: function (S) {return by_size[S];}
			};
		}
	};
}
