// ucode/dsv41_spark.uc — the DeepSeek-V4.1 DSpark block drafter (mtp.* in the checkpoint).
//
// One round makes ONE forward over a block of dspark_block (5) tokens [seed, noise x 4] through 3 stages
// (plain sliding-window blocks, mHC "one site late" like the trunk, 128 FP4 experts top 3 + a shared FP8
// expert) and emits 5 draft tokens:
//
//   taps        stream mean at the INPUT of target layers 37 / 38 / 39 (after the Engram), per position
//   main row    main_norm(main_proj(cat(taps)))                (5120)  per position
//   main kv     per stage b: rope(kv_norm_b(wkv_b(main row)), position)   (512), a ring of RING rows
//   stage b     queries = the 5 block tokens (positions p0 .. p0 + 4) attend NON-CAUSALLY over
//               [the last <= 128 main kv rows ++ the block's own kv] with per-head sinks
//   state       hc collapse of the last stage -> (norm -> the TARGET's head) -> logits (5, V)
//   draft       greedy chain with the markov bias:  out[0] = seed;
//               logits[i] += markov_head(markov_embed[out[i]]);  out[i + 1] = argmax
//   confidence  sigmoid(proj([collapsed state_i, markov_embed[out[i]]])) gates draft i + 1
//
//   let sp = dsv41_spark.load(model, {devs: [d0, d1, d2]});
//   sp.update_rows(taps, pos)     taps (n, 3 * hidden) fp32/fp16: the main rows of positions pos .. pos + n - 1
//   sp.draft(seed, p0)            -> {drafts: [d1 .. d5], conf: [c1 .. c5]}   (main rows < p0 must be in)

import * as math from 'math';
import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';
import * as dsv41 from './dsv41.uc';

const RING = 1024;                      // main kv rows kept per stage (>= window + the longest verify block; the slack
                                        // keeps the rows of a reused prefix from being overwritten by a long old tail)
const SKIP_POS = 1099511627776;         // wbase of the dummy window: its rows lie past every query

/* massive-activation rows (BOS, the first tokens: stream means up to ~1e5) overflow fp16. main_proj is
 * followed by an RMSNorm, so a positive per-row scale is exact: bring every row into a safe range. */
function safe_taps(taps) {
	let t = taps.to('float32');
	let a = t.abs().max(1, true);
	let sc = a.div(2048.0).clamp(1.0, null);

	return t.div(sc).to('float16');
}

function build_expert_set(ws, key, E, dev) {
	let t = {w1: [], w3: [], w2: [], s1: [], s3: [], s2: [], keep: []};

	for (let e = 0; e < E; e++) {
		let k = `${key}.experts.${e}`;

		for (let m in [ 'w1', 'w3', 'w2' ]) {
			let w = ws.gpu(`${k}.${m}.weight`, dev), s = ws.gpu(`${k}.${m}.scale`, dev);

			push(t[m], w.data_ptr());
			push(t['s' + substr(m, 1)], s.data_ptr());
			push(t.keep, w, s);
		}
	}

	let dv = 'cuda:' + dev;

	for (let m in [ 'w1', 'w3', 'w2' ]) {
		t[m] = torch.from_array(t[m], 'int64', dv);
		t['s' + substr(m, 1)] = torch.from_array(t['s' + substr(m, 1)], 'int64', dv);
	}

	return t;
}

function ring_get(ring, a, b) {
	let n = b - a;
	let s = a % RING;

	if (s + n <= RING)
		return ring.narrow(0, s, n);

	return torch.cat([ring.narrow(0, s, RING - s), ring.narrow(0, 0, n - (RING - s))], 0);
}

function ring_put(ring, pos0, rows) {
	let T = rows.shape()[0];

	if (T > RING) {
		rows = rows.narrow(0, T - RING, RING);
		pos0 += T - RING;
		T = RING;
	}

	let s = pos0 % RING;

	if (s + T <= RING) {
		ring.narrow(0, s, T).copy_(rows);
	}
	else {
		ring.narrow(0, s, RING - s).copy_(rows.narrow(0, 0, RING - s));
		ring.narrow(0, 0, T - (RING - s)).copy_(rows.narrow(0, RING - s, T - (RING - s)));
	}
}

export function load(model, opts) {
	opts = opts || {};

	let cfg = model.cfg, ws = model.ws;
	let B = cfg.dspark_block, nb = cfg.num_mtp_layers;
	let H = cfg.hidden_size, eps = cfg.rms_norm_eps;
	let devs = opts.devs;
	let E = cfg.dspark_experts, topk = cfg.dspark_topk;
	let I = cfg.moe_intermediate_size;

	if (!nb || !B || !ws.has('mtp.0.main_proj.weight'))
		die('dsv41_spark: the checkpoint has no drafter');

	if (length(devs) != nb)
		die(`dsv41_spark: ${nb} stage devices needed`);

	// the router and the experts are the drafter's own (128, top 3); everything else follows the trunk
	let dcfg = {};

	for (let k in keys(cfg))
		dcfg[k] = cfg[k];

	dcfg.num_experts_per_tok = topk;

	ws.upload(function (key) {
		if (index(key, 'mtp.') != 0)
			return null;

		let b = +substr(key, 4, index(key, '.', 4) - 4);

		if (index(key, '.markov_head.') > 0 || index(key, '.confidence_head.') > 0 || index(key, '.norm.weight') > 0)
			return devs[nb - 1];

		return devs[b];
	});

	let stages = [];

	for (let b = 0; b < nb; b++) {
		let dev = devs[b], dv = 'cuda:' + dev;
		let p = `mtp.${b}`;
		let key = `${p}.attn`;
		let A = {
			dev: dev, idx: b, ratio: 0,
			inv_freq: dsv4.yarn_inv_freq(cfg.rope_dim, cfg.rope_theta, null, dev),
			q_a: dsv41.lin8(ws, key + '.wq_a', dev),
			q_norm: ws.gpu16(key + '.q_norm.weight', dev),
			q_b: dsv41.lin8(ws, key + '.wq_b', dev),
			wkv: dsv41.lin8(ws, key + '.wkv', dev),
			kv_norm: ws.gpu16(key + '.kv_norm.weight', dev),
			wo_a: dsv41.lin8g(ws, key + '.wo_a', cfg.o_groups, dev),
			wo_b: dsv41.lin8(ws, key + '.wo_b', dev),
			sinks: ws.gpu(key + '.attn_sink', dev).to('float32').contiguous()
		};
		let mk = `${p}.ffn`;
		let ex = build_expert_set(ws, mk, E, dev);
		let M = {
			type: 'moe', dev: dev, router: 'sqrtsp', ex: ex,
			gate_t: dsv41.mat32(ws, mk + '.gate', dev),
			bias: ws.gpu(mk + '.gate.bias', dev).to('float32').reshape([1, -1]).contiguous(),
			bias_vl: null, tid2eid: null, ids: null, vlon: false,
			shared: dsv41.build_mlp8(ws, mk + '.shared_experts', dev)
		};

		push(stages, {
			dev: dev, A: A, M: M,
			hc_attn: dsv41.build_hc(ws, p, 'attn', dev),
			hc_ffn: dsv41.build_hc(ws, p, 'ffn', dev),
			attn_norm: ws.gpu16(`${p}.attn_norm.weight`, dev),
			ffn_norm: ws.gpu16(`${p}.ffn_norm.weight`, dev),
			ring: torch.zeros([RING, cfg.head_dim], 'float16', dv)
		});
	}

	let d0 = devs[0], dl = devs[nb - 1], dvl = 'cuda:' + dl;
	let last = `mtp.${nb - 1}`;
	let main_proj = dsv41.lin8(ws, 'mtp.0.main_proj', d0);
	let main_norm = ws.gpu16('mtp.0.main_norm.weight', d0);
	let dnorm = ws.gpu16(last + '.norm.weight', dl);
	let mk_embed = ws.gpu16(last + '.markov_head.embed.weight', dl);                   // (V, rank)
	let mk_head_t = ws.gpu16(last + '.markov_head.head.weight', dl).t().contiguous();  // (rank, V)
	let confw = ws.gpu(last + '.confidence_head.proj.weight', dl).to('float32').reshape([-1, 1]).contiguous();

	// ms of a verify step of 1 .. 6 rows (measured: examples/dsv41_verify_bench.uc) plus the drafter's own time
	let costs = getenv('AMP_DSPARK_COSTS') ? map(split(getenv('AMP_DSPARK_COSTS'), ','), (x) => +x) :
		[ 36.8, 45.2, 54.2, 61.3, 67.1, 73.0 ];

	let d = {stages: stages, B: B, devs: devs, dcfg: dcfg, 
		cost: map(costs, (c) => c + (+(getenv('AMP_DSPARK_DRAFT_MS') || 8))),
		threshold: opts.threshold != null ? opts.threshold : +(getenv('AMP_DSPARK_CONF') || 0.3)};

	/* main rows of the positions pos .. pos + n - 1 from taps (n, 3 * hidden), then per stage the kv rows */
	let stamp = [];

	for (let i = 0; i < RING; i++)
		push(stamp, -1);

	d.update_rows = function (taps, pos0) {
		let n = taps.shape()[0];

		for (let i = max(0, n - RING); i < n; i++)
			stamp[(pos0 + i) % RING] = pos0 + i;
		let mx = torch.rms_norm(main_proj.forward(safe_taps(taps.cuda(d0, true))), main_norm, eps);

		for (let S in stages) {
			let A = S.A;
			let x = (S.dev == d0) ? mx : mx.cuda(S.dev, true);
			let kv = torch.rms_norm(A.wkv.forward(x), A.kv_norm, eps);
			let cs = dsv41.rope_cs(A.inv_freq, pos0, n, 1, S.dev);
			let kv16 = dsv41.rope_il(kv.to('float32').view([n, 1, cfg.head_dim]), cs, cfg.rope_dim).to('float16')
				.view([n, cfg.head_dim]).contiguous();

			ring_put(S.ring, pos0, kv16);
		}
	};

	/* the FP4 experts of a stage: routing, gathered GEMVs, swiglu with the router weights; y (R, H) fp16 */
	function moe(S, y) {
		let M = S.M, ex = M.ex;
		let R = y.shape()[0];
		let lim = cfg.swiglu_limit + 0.0;
		let rt = glm5.moe_route(dcfg, M, y);
		let ids = rt.idx.reshape([-1]).contiguous();
		let g = torch.fp4_moe_gemv(y, ids, ex.w1, ex.s1, I, topk);
		let u = torch.fp4_moe_gemv(y, ids, ex.w3, ex.s3, I, topk);

		g = g.clamp(null, lim);
		u = u.clamp(-lim, lim);

		let act = g.silu().mul(u).mul(rt.w.reshape([-1, 1])).to('float16').contiguous();
		let o = torch.fp4_moe_gemv(act, ids, ex.w2, ex.s2, H, 1);
		let sh = glm5.mlp_forward(cfg, M.shared, y).to('float32');

		return o.view([R, topk, H]).sum(1).add(sh);
	}

	/* the block attention of stage S for the block tokens at p0 .. p0 + R - 1; xn (R, H) fp16 */
	function attn(S, xn, p0) {
		let A = S.A;
		let R = xn.shape()[0];
		let Hq = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
		let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
		let q = A.q_b.forward(qr).view([R, Hq, D]);
		let cs = dsv41.rope_cs(A.inv_freq, p0, R, 1, S.dev);
		let q16 = dsv41.rope_il(q.to('float32'), cs, rd).to('float16').contiguous();
		let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
		let kv16 = dsv41.rope_il(kv.to('float32').view([R, 1, D]), cs, rd).to('float16').view([R, D]).contiguous();
		let W = min(cfg.window, p0);
		let keys_ = (W > 0) ? torch.cat([ring_get(S.ring, p0 - W, p0), kv16], 0) : kv16;
		let Nc = W + R;
		let idx = torch.arange(Nc, 'int32', 'cuda:' + S.dev).view([1, Nc]).expand([R, Nc]).contiguous();

		// the pool path of the kernel attends to whatever idx lists; the window rows lie past every query
		let o = torch.v41_attn(q16, kv16, SKIP_POS, keys_, idx, A.sinks, A.inv_freq, p0, cfg.sm_scale, cfg.window);

		return A.wo_b.forward(A.wo_a.forward(o.view([R, Hq * D])));
	}

	/* one stage over the block; streams (R, 4, H) fp32 updated in place, pre (R, 4) the collapse weights of
	 * the attention input; returns the pre of the next stage */
	function stage(S, streams, pre, p0) {
		let am = dsv41.hc_site(cfg, S.hc_attn, streams, pre);
		let xn = torch.rms_norm(am.col, S.attn_norm, eps);
		let y = attn(S, xn, p0);

		glm5.hc_apply(streams, y, am.post, am.comb);

		let fm = dsv41.hc_site(cfg, S.hc_ffn, streams, am.pre);

		xn = torch.rms_norm(fm.col, S.ffn_norm, eps);
		y = moe(S, xn);
		glm5.hc_apply(streams, y, fm.post, fm.comb);

		return fm.pre;
	}

	/* ---- the graph form: positions, window slots and the seed live in small device buffers that the host
	 * refreshes before each replay (window = the last cfg.window main rows, so p0 >= cfg.window) ---- */

	let Wn = cfg.window;
	let gb = [];                                         // per stage device: {posf, slots, pos_t, idx}

	for (let S in stages) {
		let dv = 'cuda:' + S.dev;

		push(gb, {
			posf: torch.zeros([B, 1], 'float32', dv), slots: torch.zeros([Wn], 'int64', dv),
			pos_t: torch.zeros([1, 1], 'int64', dv),
			idx: torch.arange(Wn + B, 'int32', dv).view([1, Wn + B]).expand([B, Wn + B]).contiguous()
		});
	}

	function attn_g(S, P, xn) {
		let A = S.A;
		let R = xn.shape()[0];
		let Hq = cfg.num_q_heads, D = cfg.head_dim, rd = cfg.rope_dim;
		let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
		let q = A.q_b.forward(qr).view([R, Hq, D]);
		let q16 = torch.v41_rope(q, P.posf, A.inv_freq);
		let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
		let kv16 = torch.v41_rope(kv.view([R, 1, D]), P.posf, A.inv_freq).view([R, D]);
		let keys_ = torch.cat([S.ring.index(P.slots), kv16], 0);
		let o = torch.v41_attn(q16, kv16, SKIP_POS, keys_, P.idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
			P.pos_t, 0);

		return A.wo_b.forward(A.wo_a.forward(o.view([R, Hq * D])));
	}

	function stage_g(S, P, streams, pre) {
		let am = dsv41.hc_site(cfg, S.hc_attn, streams, pre);
		let xn = torch.rms_norm(am.col, S.attn_norm, eps);
		let y = attn_g(S, P, xn);

		glm5.hc_apply(streams, y, am.post, am.comb);

		let fm = dsv41.hc_site(cfg, S.hc_ffn, streams, am.pre);

		xn = torch.rms_norm(fm.col, S.ffn_norm, eps);
		y = moe(S, xn);
		glm5.hc_apply(streams, y, fm.post, fm.comb);

		return fm.pre;
	}

	// graph IO buffers
	let G = {stage: [], col: torch.zeros([B, H], 'float16', dvl), xn: torch.zeros([B, H], 'float16', dvl),
		logits: torch.zeros([B, cfg.vocab_size], 'float32', dvl), seed: torch.zeros([1], 'int64', dvl),
		res: torch.zeros([2 * B], 'float32', dvl)};

	for (let S in stages) {
		let dv = 'cuda:' + S.dev;
		let o = {h: torch.zeros([B, 4, H], 'float32', dv), pre: torch.zeros([B, 4], 'float32', dv),
			h_out: torch.zeros([B, 4, H], 'float32', dv), pre_out: torch.zeros([B, 4], 'float32', dv)};

		o.pre.narrow(1, 0, 1).fill_(1.0);
		push(G.stage, o);
	}

	function chain_g() {
		let prev = G.seed;
		let outs = [], embs = [];

		for (let i = 0; i < B; i++) {
			let emb = mk_embed.index(prev);
			let row = G.logits.narrow(0, i, 1).add(torch.matmul(emb, mk_head_t).to('float32'));

			push(embs, emb);
			prev = row.argmax(-1).view([1]);
			push(outs, prev);
		}

		let conf = torch.matmul(torch.cat([G.col.to('float32'), torch.cat(embs, 0).to('float32')], 1), confw)
			.sigmoid().view([B]);

		G.res.copy_(torch.cat([torch.cat(outs, 0).to('float32'), conf], 0));
	}

	function body(b) {
		let S = stages[b], o = G.stage[b];
		let streams = o.h.add(0.0);                         // the stage works on a copy (the input buffer stays)
		let pre = stage_g(S, gb[b], streams, o.pre);

		o.h_out.copy_(streams);
		o.pre_out.copy_(pre);

		if (b == nb - 1) {
			let col = dsv41.hc_collapse(streams, pre);

			G.col.copy_(col);
			G.xn.copy_(torch.rms_norm(col, dnorm, eps));
		}
	}

	d.gio = G;

	d.build_graphs = function () {
		let hd = model.devs[cfg.num_hidden_layers - 1];

		d.gs = [];

		for (let b = 0; b < nb; b++) {
			// eager warm-up (kernel selection, allocator), then capture
			body(b);
			torch.synchronize();

			let g = torch.graph();

			g.capture_begin(stages[b].dev);
			body(b);
			g.capture_end();
			push(d.gs, g);
		}

		chain_g();
		torch.synchronize();

		d.gchain = torch.graph();
		d.gchain.capture_begin(dl);
		chain_g();
		d.gchain.capture_end();
		torch.synchronize();
		d.ghd = hd;
	};

	/* one drafting round through the graphs (p0 >= window) */
	d.draft_graph = function (seed, p0) {
		let posv = [], sl = [];

		for (let j = 0; j < B; j++)
			push(posv, (p0 + j) * 1.0);

		for (let k = 0; k < Wn; k++)
			push(sl, (p0 - Wn + k) % RING);

		let pf = torch.from_array(posv, 'float32').view([B, 1]), st = torch.from_array(sl, 'int64');
		let pt = torch.from_array([ p0 ], 'int64').view([1, 1]);

		for (let b = 0; b < nb; b++) {
			gb[b].posf.copy_(pf.cuda(stages[b].dev, true));
			gb[b].slots.copy_(st.cuda(stages[b].dev, true));
			gb[b].pos_t.copy_(pt.cuda(stages[b].dev, true));
		}

		let ids = [ seed ];

		for (let i = 1; i < B; i++)
			push(ids, cfg.dspark_noise_id);

		let e = model.embed_tokens.index(torch.from_array(ids, 'int64').cuda(model.embed_dev, true)).to('float32');

		G.stage[0].h.copy_(e.reshape([B, 1, H]).expand([B, 4, H]).contiguous().cuda(d0));
		G.seed.fill_(seed);

		for (let b = 0; b < nb; b++) {
			if (b > 0) {
				G.stage[b].h.copy_(G.stage[b - 1].h_out);
				G.stage[b].pre.copy_(G.stage[b - 1].pre_out);
			}

			d.gs[b].replay();
		}

		let logits = model.lm_head.forward(G.xn.cuda(d.ghd));

		G.logits.copy_(logits.to('float32').cuda(dl));
		d.gchain.replay();

		let r = G.res.tolist();

		return {drafts: map(slice(r, 0, B), (x) => int(x + 0.5)), conf: slice(r, B)};
	};

	/* one drafting round for the seed token at position p0 (the main rows < p0 are in) */
	d.draft_eager = function (seed, p0) {
		let ids = [ seed ];

		for (let i = 1; i < B; i++)
			push(ids, cfg.dspark_noise_id);

		let e = model.embed_tokens.index(torch.from_array(ids, 'int64', 'cuda:' + model.embed_dev)).to('float32');
		let streams = e.reshape([B, 1, H]).expand([B, 4, H]).contiguous().cuda(d0);
		let pre = torch.zeros([B, 4], 'float32', 'cuda:' + d0);

		pre.narrow(1, 0, 1).fill_(1.0);

		let cur = d0;

		for (let S in stages) {
			if (S.dev != cur) {
				streams = streams.cuda(S.dev);
				pre = pre.cuda(S.dev);
				cur = S.dev;
			}

			pre = stage(S, streams, pre, p0);
		}

		let col = dsv41.hc_collapse(streams, pre);                                   // (B, H) fp16, on devs[nb - 1]
		let xn = torch.rms_norm(col, dnorm, eps);
		let hd = model.devs[cfg.num_hidden_layers - 1];
		let logits = model.lm_head.forward(xn.cuda(hd)).cuda(dl).to('float32');       // (B, V)
		let prev = torch.from_array([ seed ], 'int64', dvl);
		let outs = [], embs = [];

		for (let i = 0; i < B; i++) {
			let emb = mk_embed.index(prev);                                           // (1, rank)
			let row = logits.narrow(0, i, 1).add(torch.matmul(emb, mk_head_t).to('float32'));

			push(embs, emb);
			prev = row.argmax(-1).view([1]);
			push(outs, prev);
		}

		let conf = torch.matmul(torch.cat([col.to('float32'), torch.cat(embs, 0).to('float32')], 1), confw)
			.sigmoid().view([B]);
		let dr = torch.cat(outs, 0).tolist();

		return {drafts: dr, conf: conf.tolist()};
	};

	d.draft = function (seed, p0) {
		return (d.gs && p0 >= Wn) ? d.draft_graph(seed, p0) : d.draft_eager(seed, p0);
	};

	/* the rows a draft at p0 reads (the last window positions before it) are in the ring */
	d.ready = function (p0) {
		for (let q = max(0, p0 - cfg.window); q < p0; q++)
			if (stamp[q % RING] != q)
				return false;

		return true;
	};

	/* engine interface (see engine.dspark_round): the verify graphs come from model.graphs(cache, sizes) */
	d.build = function (graph, cache, windows) {
		d.graph = graph;
		d.cache = cache;

		if (!getenv('AMP_DSPARK_NO_GRAPH'))
			d.build_graphs();
	};

	d.prefill_rows = function (taps, pos) {
		d.update_rows(taps, pos);
	};

	/* one round from the seed token (not yet in the target cache) at p0: draft, verify the longest confident
	 * prefix of the drafts (at most Vmax) in one target step, feed the drafter the verified rows.
	 * {V, dr, am (the target's greedy token after each of the V + 1 rows), logits (V + 1, vocab), conf} */
	let PR = getenv('AMP_DSPARK_PROF') ? {n: 0, draft: 0, verify: 0, update: 0, tail: 0} : null;

	function now() {
		let c = clock(true);

		return c[0] * 1000.0 + c[1] / 1e6;
	}

	d.run = function (seed, p0, Vmax) {
		let t0 = PR ? now() : 0;
		let r = d.draft(seed, p0);

		if (PR) {
			torch.synchronize();
			PR.draft += now() - t0;
			t0 = now();
		}

		// verify the prefix of drafts that maximizes the expected tokens per millisecond: a step of V + 1 rows
		// costs cost[V]; draft k counts with the probability that all drafts up to it are right (the product
		// of the confidences)
		let V = 0, best = 1.0 / d.cost[0], prod = 1.0, acc = 0.0;

		for (let k = 0; k < min(Vmax, B); k++) {
			prod *= r.conf[k];
			acc += prod;

			let rate = (1.0 + acc) / d.cost[k + 1];

			if (rate > best) {
				best = rate;
				V = k + 1;
			}
		}

		let ids = [ seed, ...slice(r.drafts, 0, V) ];
		let ver = (V == 0) ? d.graph.run([ seed ], p0) : d.graph.rows(ids, p0);

		if (PR) {
			torch.synchronize();
			PR.verify += now() - t0;
			t0 = now();
		}

		// the rows of rejected drafts are overwritten before they are read (positions are consumed in order)
		d.update_rows(ver.taps, p0);

		if (PR) {
			torch.synchronize();
			PR.update += now() - t0;
			t0 = now();
		}

		let am = ver.logits.to('float32').argmax(-1).tolist();

		if (PR) {
			PR.tail += now() - t0;

			if (++PR.n % 50 == 0)
				warn(sprintf('dspark prof (ms/round, %d rounds): draft %.1f verify %.1f update %.1f tail %.1f\n', PR.n,
					PR.draft / PR.n, PR.verify / PR.n, PR.update / PR.n, PR.tail / PR.n));
		}

		return {V: V, dr: slice(r.drafts, 0, V), am: am, logits: ver.logits, conf: r.conf};
	};

	/* the engine keeps `len` rows of the last round (the seed and the accepted drafts) */
	d.rollback = function (len) {
		model.rollback(d.cache, len);
	};

	return d;
}
