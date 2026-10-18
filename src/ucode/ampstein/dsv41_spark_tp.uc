// ucode/dsv41_spark_tp.uc — the DSpark block drafter (see dsv41_spark.uc for the algorithm) inside the 3 x TP-2 pipeline
// of dsv41_tp.uc: drafter stage b runs on the GPU pair of pipeline stage b, sharded like a trunk layer (attention heads /
// o_groups / wo_b columns, expert parallel FP4 experts 64 + 64, the shared expert by rows / columns; the router and
// the main-kv rings are replicated). A round is one multi-GPU graph per stage, the head, and a small chain graph on
// the last stage's first rank.
//
//   let sp = dsv41_spark_tp.load(model, {});     model = dsv41_tp.load(...)
//   sp.update_rows(taps, pos) / sp.run(seed, p0, Vmax) / sp.ready(p0) / sp.rollback(len)   (engine interface)

import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';
import * as dsv4_tp from './dsv4_tp.uc';
import * as dsv41 from './dsv41.uc';

const RING = 1024;
const SKIP_POS = 1099511627776;
const FFN_SCALE = 16.0;                 // the MoE partials travel through the fp16 all-reduce divided by this (dsv41_tp.uc)

function safe_taps(taps) {
	let t = taps.to('float32');
	let a = t.abs().max(1, true);
	let sc = a.div(2048.0).clamp(1.0, null);

	return t.div(sc).to('float16');
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

/* pointer tables of the FP4 experts [e0, e0 + n) of a stage on dev */
function expert_set(ws, key, e0, n, dev) {
	let t = {w1: [], w3: [], w2: [], s1: [], s3: [], s2: [], keep: []};

	for (let e = e0; e < e0 + n; e++) {
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

function hc_copy(hc, dev) {
	return {fn: hc.fn.cuda(dev), fn_pre_t: hc.fn_pre_t.cuda(dev), base: hc.base.cuda(dev), base_pre: hc.base_pre.cuda(dev),
		scale: hc.scale.cuda(dev), scale_pre: hc.scale_pre.cuda(dev)};
}

export function load(model, opts) {
	opts = opts || {};

	let cfg = model.cfg, ws = model.ws;
	let B = cfg.dspark_block, nb = cfg.num_mtp_layers;
	let H = cfg.hidden_size, eps = cfg.rms_norm_eps;
	let Gs = model.Gs;
	let E = cfg.dspark_experts, topk = cfg.dspark_topk;
	let I = cfg.moe_intermediate_size;
	let sdev = opts.stages || [ 0, 1, 2 ];                     // drafter stage b -> pipeline stage (GPU pair)

	if (!nb || !B || !ws.has('mtp.0.main_proj.weight'))
		die('dsv41_spark_tp: the checkpoint has no drafter');

	if (length(sdev) != nb || E % Gs || I % (32 * Gs))
		die('dsv41_spark_tp: stage assignment / expert split');

	let Eloc = E / Gs;
	let nh = int(cfg.num_q_heads / Gs), ng = int(cfg.o_groups / Gs), D = cfg.head_dim;
	let ones = [];

	for (let g = 0; g < Gs; g++)
		push(ones, 1);

	let shared_split = dsv4_tp.split_n(I, 256, ones);
	let dcfg = {};

	for (let k in keys(cfg))
		dcfg[k] = cfg[k];

	dcfg.num_experts_per_tok = topk;

	let pairs = map(sdev, (s) => model.stages[s].devs);
	let dlast = pairs[nb - 1][0];

	ws.upload(function (key) {
		if (index(key, 'mtp.') != 0)
			return null;

		let b = +substr(key, 4, index(key, '.', 4) - 4);
		let at = index(key, '.experts.');

		if (at >= 0) {
			let rest = substr(key, at + 9);

			return pairs[b][int(+substr(rest, 0, index(rest, '.')) / Eloc)];
		}

		return pairs[b][0];
	});

	let stages = [];

	for (let b = 0; b < nb; b++) {
		let pr = pairs[b], st = pr[0];
		let p = `mtp.${b}`, key = `${p}.attn`, mk = `${p}.ffn`;
		let A = {
			inv_freq: dsv4.yarn_inv_freq(cfg.rope_dim, cfg.rope_theta, null, st),
			q_a: dsv41.lin8(ws, key + '.wq_a', st), q_norm: ws.gpu16(key + '.q_norm.weight', st),
			q_b: dsv41.lin8(ws, key + '.wq_b', st), wkv: dsv41.lin8(ws, key + '.wkv', st),
			kv_norm: ws.gpu16(key + '.kv_norm.weight', st),
			wo_a: dsv41.lin8g(ws, key + '.wo_a', cfg.o_groups, st), wo_b: dsv41.lin8(ws, key + '.wo_b', st),
			sinks: ws.gpu(key + '.attn_sink', st).to('float32').contiguous()
		};
		let hc_a = dsv41.build_hc(ws, p, 'attn', st), hc_f = dsv41.build_hc(ws, p, 'ffn', st);
		let an = ws.gpu16(`${p}.attn_norm.weight`, st), fnw = ws.gpu16(`${p}.ffn_norm.weight`, st);
		let gate_t = dsv41.mat32(ws, mk + '.gate', st);
		let bias = ws.gpu(mk + '.gate.bias', st).to('float32').reshape([1, -1]).contiguous();
		let shared = dsv41.build_mlp8(ws, mk + '.shared_experts', st);
		let ranks = [];

		for (let g = 0; g < Gs; g++) {
			let dev = pr[g], dv = 'cuda:' + dev;

			push(ranks, {
				dev: dev,
				A: {
					inv_freq: A.inv_freq.cuda(dev),
					q_a: dsv41.lin8_to(A.q_a, dev), q_norm: A.q_norm.cuda(dev),
					q_b: dsv41.lin8_rows(A.q_b, g * nh * D, nh * D, dev),
					wkv: dsv41.lin8_to(A.wkv, dev), kv_norm: A.kv_norm.cuda(dev),
					wo_a: dsv41.lin8g_groups(A.wo_a, g * ng, ng, dev),
					wo_b: dsv41.lin8_cols(A.wo_b, g * ng * cfg.o_lora_rank, ng * cfg.o_lora_rank, dev),
					sinks: A.sinks.narrow(0, g * nh, nh).contiguous().cuda(dev)
				},
				M: {
					dev: dev, router: 'sqrtsp', ex: expert_set(ws, mk, g * Eloc, Eloc, dev), e0: g * Eloc,
					gate_t: gate_t.cuda(dev), bias: bias.cuda(dev), bias_vl: null, tid2eid: null, ids: null, vlon: false,
					shared: {
						gate: dsv41.lin8_rows(shared.gate, shared_split[g][0], shared_split[g][1], dev),
						up: dsv41.lin8_rows(shared.up, shared_split[g][0], shared_split[g][1], dev),
						down: dsv41.lin8_cols(shared.down, shared_split[g][0], shared_split[g][1], dev)
					}
				},
				hc_attn: hc_copy(hc_a, dev), hc_ffn: hc_copy(hc_f, dev),
				attn_norm: an.cuda(dev), ffn_norm: fnw.cuda(dev),
				ring: torch.zeros([RING, D], 'float16', dv)
			});
		}

		push(stages, {b: b, S: model.stages[sdev[b]], ranks: ranks});

		for (let k in ws.list())
			if (index(k, p + '.') == 0 && index(k, '.experts.') < 0 && index(k, '.markov_head.') < 0 &&
			    index(k, '.confidence_head.') < 0 && index(k, 'main_proj') < 0 && index(k, 'main_norm') < 0 &&
			    index(k, p + '.norm.weight') < 0)
				ws.release(k, st);

		A = shared = null;
		gc();
	}

	torch.empty_cache();                    // the staged copies of the shards

	let d0 = pairs[0][0], dl = dlast, dvl = 'cuda:' + dl;
	let last = `mtp.${nb - 1}`;
	let main_proj = dsv41.lin8(ws, 'mtp.0.main_proj', d0);
	let main_norm = ws.gpu16('mtp.0.main_norm.weight', d0);
	let dnorm = ws.gpu16(last + '.norm.weight', dl);
	let mk_embed = ws.gpu16(last + '.markov_head.embed.weight', dl);
	let mk_head_t = ws.gpu16(last + '.markov_head.head.weight', dl).t().contiguous();
	let confw = ws.gpu(last + '.confidence_head.proj.weight', dl).to('float32').reshape([-1, 1]).contiguous();

	let costs = getenv('AMP_DSPARK_COSTS') ? map(split(getenv('AMP_DSPARK_COSTS'), ','), (x) => +x) :
		[ 24.5, 27.9, 34.6, 39.6, 44.5, 49.0 ];
	let d = {stages: stages, B: B, tp: true, cost: map(costs, (c) => c + (+(getenv('AMP_DSPARK_DRAFT_MS') || 8)))};
	let stamp = [];

	for (let i = 0; i < RING; i++)
		push(stamp, -1);

	/* main rows of the positions pos0 .. pos0 + n - 1 from taps (n, 3 * hidden): kv rows into every rank's ring */
	d.update_rows = function (taps, pos0) {
		let n = taps.shape()[0];

		for (let i = max(0, n - RING); i < n; i++)
			stamp[(pos0 + i) % RING] = pos0 + i;

		let mx = torch.rms_norm(main_proj.forward(safe_taps(taps.cuda(d0, true))), main_norm, eps);

		for (let S in stages)
			for (let R in S.ranks) {
				let A = R.A;
				let x = (R.dev == d0) ? mx : mx.cuda(R.dev, true);
				let kv = torch.rms_norm(A.wkv.forward(x), A.kv_norm, eps);
				let cs = dsv41.rope_cs(A.inv_freq, pos0, n, 1, R.dev);
				let kv16 = dsv41.rope_il(kv.to('float32').view([n, 1, D]), cs, cfg.rope_dim).to('float16')
					.view([n, D]).contiguous();

				ring_put(R.ring, pos0, kv16);
			}
	};

	/* ---- the graph form (window = the last cfg.window main rows, so p0 >= cfg.window) ---- */

	let Wn = cfg.window;

	for (let S in stages)
		for (let R in S.ranks) {
			let dv = 'cuda:' + R.dev;

			R.P = {posf: torch.zeros([B, 1], 'float32', dv), slots: torch.zeros([Wn], 'int64', dv),
				pos_t: torch.zeros([1, 1], 'int64', dv),
				idx: torch.arange(Wn + B, 'int32', dv).view([1, Wn + B]).expand([B, Wn + B]).contiguous()};
		}

	/* the block attention of rank R: partial (B, H) fp16 */
	function attn_g(R, xn) {
		let A = R.A, P = R.P;
		let n = xn.shape()[0];
		let rd = cfg.rope_dim;
		let qr = torch.rms_norm(A.q_a.forward(xn), A.q_norm, eps);
		let q = A.q_b.forward(qr).view([n, nh, D]);
		let q16 = torch.v41_rope(q, P.posf, A.inv_freq);
		let kv = torch.rms_norm(A.wkv.forward(xn), A.kv_norm, eps);
		let kv16 = torch.v41_rope(kv.view([n, 1, D]), P.posf, A.inv_freq).view([n, D]);
		let keys_ = torch.cat([R.ring.index(P.slots), kv16], 0);
		let o = torch.v41_attn(q16, kv16, SKIP_POS, keys_, P.idx, A.sinks, A.inv_freq, 0, cfg.sm_scale, cfg.window, 0,
			P.pos_t, 0);

		return A.wo_b.forward(A.wo_a.forward(o.view([n, nh * D])));
	}

	/* the FP4 experts of rank R (local ids; foreign picks contribute nothing) + the shared expert shard: fp32 partial */
	function moe_g(R, y) {
		let M = R.M, ex = M.ex;
		let n = y.shape()[0];
		let lim = cfg.swiglu_limit + 0.0;
		let rt = glm5.moe_route(dcfg, M, y);
		let dv = 'cuda:' + R.dev;
		let loc = rt.idx.sub(torch.full([1], M.e0, 'int64', dv));
		let lo_ok = loc.neg().le(0).to('int64');
		let hi_ok = loc.le(Eloc - 1).to('int64');
		let bad = lo_ok.mul(hi_ok).le(0);
		let ids = torch.where(bad, torch.full([1], -1, 'int64', dv), loc).reshape([-1]).contiguous();
		let g = torch.fp4_moe_gemv(y, ids, ex.w1, ex.s1, I, topk);
		let u = torch.fp4_moe_gemv(y, ids, ex.w3, ex.s3, I, topk);

		g = g.clamp(null, lim);
		u = u.clamp(-lim, lim);

		let act = g.silu().mul(u).mul(rt.w.reshape([-1, 1])).to('float16').contiguous();
		let o = torch.fp4_moe_gemv(act, ids, ex.w2, ex.s2, H, 1);
		let sh = glm5.mlp_forward(cfg, M.shared, y).to('float32');

		return o.view([n, topk, H]).sum(1).add(sh);
	}

	/* one drafter stage on its pair: h / pre per rank (updated); returns the per-rank pre */
	function stage_g(S, h, pre) {
		let ranks = S.ranks, am = [], fm = [], p = [];

		for (let g = 0; g < Gs; g++) {
			let R = ranks[g];

			am[g] = dsv41.hc_site(cfg, R.hc_attn, h[g], pre[g]);
			p[g] = attn_g(R, torch.rms_norm(am[g].col, R.attn_norm, eps));
		}

		let s = dsv4_tp.allreduce(S.S, p);

		for (let g = 0; g < Gs; g++)
			glm5.hc_apply(h[g], s[g], am[g].post, am[g].comb);

		for (let g = 0; g < Gs; g++) {
			let R = ranks[g];

			fm[g] = dsv41.hc_site(cfg, R.hc_ffn, h[g], am[g].pre);
			p[g] = moe_g(R, torch.rms_norm(fm[g].col, R.ffn_norm, eps)).mul(1.0 / FFN_SCALE).clamp(-65504.0, 65504.0)
				.to('float16');
		}

		s = dsv4_tp.allreduce(S.S, p);

		let out = [];

		for (let g = 0; g < Gs; g++) {
			glm5.hc_apply(h[g], s[g], fm[g].post.mul(FFN_SCALE), fm[g].comb);
			push(out, fm[g].pre);
		}

		return out;
	}

	// graph IO
	let G = {col: torch.zeros([B, H], 'float16', dvl), xn: torch.zeros([B, H], 'float16', dvl),
		logits: torch.zeros([B, cfg.vocab_size], 'float32', dvl), seed: torch.zeros([1], 'int64', dvl),
		res: torch.zeros([2 * B], 'float32', dvl)};

	for (let S in stages) {
		S.h_in = [];
		S.pre_in = [];
		S.h_out = [];
		S.pre_out = [];
		S.bar = [];

		for (let R in S.ranks) {
			let dv = 'cuda:' + R.dev;
			let pre = torch.zeros([B, 4], 'float32', dv);

			pre.narrow(1, 0, 1).fill_(1.0);
			push(S.h_in, torch.zeros([B, 4, H], 'float32', dv));
			push(S.pre_in, pre);
			push(S.h_out, torch.zeros([B, 4, H], 'float32', dv));
			push(S.pre_out, torch.zeros([B, 4], 'float32', dv));
			push(S.bar, torch.zeros([1], 'float32', dv));
		}
	}

	function body(b) {
		let S = stages[b];
		let h = map(S.h_in, (t) => t.add(0.0));                  // work on copies (the inputs stay)
		let pre = stage_g(S, h, S.pre_in);

		for (let g = 0; g < Gs; g++) {
			S.h_out[g].copy_(h[g]);
			S.pre_out[g].copy_(pre[g]);
		}

		if (b == nb - 1) {
			let col = dsv41.hc_collapse(h[0], pre[0]);

			G.col.copy_(col);
			G.xn.copy_(torch.rms_norm(col, dnorm, eps));
		}
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

	d.build_graphs = function () {
		d.gs = [];

		for (let b = 0; b < nb; b++) {
			body(b);
			torch.synchronize();

			let g = torch.graph();
			let pr = pairs[b];

			g.capture_begin(pr[0], slice(pr, 1));
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
	};

	d.draft = function (seed, p0) {
		let posv = [], sl = [];

		for (let j = 0; j < B; j++)
			push(posv, (p0 + j) * 1.0);

		for (let k = 0; k < Wn; k++)
			push(sl, (p0 - Wn + k) % RING);

		let pf = torch.from_array(posv, 'float32').view([B, 1]), st = torch.from_array(sl, 'int64');
		let pt = torch.from_array([ p0 ], 'int64').view([1, 1]);

		for (let S in stages)
			for (let R in S.ranks) {
				R.P.posf.copy_(pf.cuda(R.dev, true));
				R.P.slots.copy_(st.cuda(R.dev, true));
				R.P.pos_t.copy_(pt.cuda(R.dev, true));
			}

		let ids = [ seed ];

		for (let i = 1; i < B; i++)
			push(ids, cfg.dspark_noise_id);

		let e = model.embed_tokens.index(torch.from_array(ids, 'int64').cuda(model.embed_dev, true)).to('float32');
		let h0 = e.reshape([B, 1, H]).expand([B, 4, H]).contiguous();

		for (let g = 0; g < Gs; g++)
			stages[0].h_in[g].copy_(h0.cuda(pairs[0][g], true));

		G.seed.fill_(seed);

		for (let b = 0; b < nb; b++) {
			let S = stages[b];

			if (b > 0)
				for (let g = 0; g < Gs; g++) {
					S.h_in[g].copy_(stages[b - 1].h_out[g]);
					S.pre_in[g].copy_(stages[b - 1].pre_out[g]);
				}

			for (let g = 1; g < Gs; g++)
				S.bar[g].copy_(S.bar[0]);

			d.gs[b].replay();

			for (let g = 1; g < Gs; g++)
				S.bar[g].copy_(S.bar[0]);
		}

		let logits = model.head_full(G.xn);

		G.logits.copy_(logits.to('float32').cuda(dl));
		d.gchain.replay();

		let r = G.res.tolist();

		return {drafts: map(slice(r, 0, B), (x) => int(x + 0.5)), conf: slice(r, B)};
	};

	/* the last window of main rows is in the ring (p0 >= window: the graph form needs the full window) */
	d.ready = function (p0) {
		if (p0 < Wn)
			return false;

		for (let q = p0 - Wn; q < p0; q++)
			if (stamp[q % RING] != q)
				return false;

		return true;
	};

	d.build = function (graph, cache, windows) {
		d.graph = graph;
		d.cache = cache;
		d.build_graphs();
	};

	d.prefill_rows = function (taps, pos) {
		d.update_rows(taps, pos);
	};

	let PR = getenv('AMP_DSPARK_PROF') ? {n: 0, draft: 0, verify: 0, update: 0} : null;

	function now() {
		let c = clock(true);

		return c[0] * 1000.0 + c[1] / 1e6;
	}

	/* one round from the seed token (not yet in the target cache) at p0 */
	d.run = function (seed, p0, Vmax) {
		let t0 = PR ? now() : 0;
		let r = d.draft(seed, p0);

		if (PR) {
			PR.draft += now() - t0;
			t0 = now();
		}

		// the draft length that maximizes the expected tokens per millisecond (see dsv41_spark.uc)
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

		while (V > 0 && !d.graph.has(V + 1))          // only some step sizes may have been captured
			V--;

		let ids = [ seed, ...slice(r.drafts, 0, V) ];
		let ver = (V == 0) ? d.graph.run([ seed ], p0) : d.graph.rows(ids, p0);

		if (PR) {
			torch.synchronize();
			PR.verify += now() - t0;
			t0 = now();
		}

		d.update_rows(ver.taps, p0);

		let am = ver.logits.to('float32').argmax(-1).tolist();

		if (PR) {
			PR.update += now() - t0;

			if (++PR.n % 50 == 0)
				warn(sprintf('dspark prof (ms/round, %d rounds): draft %.1f verify %.1f update %.1f\n', PR.n,
					PR.draft / PR.n, PR.verify / PR.n, PR.update / PR.n));
		}

		return {V: V, dr: slice(r.drafts, 0, V), am: am, logits: ver.logits, conf: r.conf};
	};

	d.rollback = function (len) {
		model.rollback(d.cache, len);
	};

	return d;
}
