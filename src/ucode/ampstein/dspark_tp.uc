// ucode/dspark_tp.uc — the DSpark block drafter inside the 3 x TP-2 pipeline
// (see dspark.uc for the algorithm, dsv4_tp.uc for the layout).
//
// The three drafter blocks are sharded over the LAST stage's two ranks exactly like
// trunk layers (attention heads / o_groups, expert-parallel MoE, shared expert by
// columns); their attention runs the decode kernels in block mode (all B queries see
// the last 128 main-kv rows plus the block). A whole speculative round is three
// CUDA graphs and ONE host read:
//
//   draft   [seed, noise x 4] -> blocks -> state -> sharded head -> markov chain -> d1..d5
//   verify  the target, B + 1 rows [seed, d1 .. d5], greedy argmax per row (+ taps)
//   update  main rows of the verify block's positions from the taps (fixed shape; rows
//           past the accepted prefix are overwritten before they are read)
//
//   let sp = dspark_tp.load(tp, graph, {max_seq});
//   sp.prefill_rows(taps, pos)        main rows of prompt positions (eager)
//   let r = sp.round(seed, p0)        -> {tokens: [..accepted.., bonus], n_acc}

import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';
import * as dsv4_tp from './dsv4_tp.uc';

/* The massive-activation rows (BOS, the first tokens: stream means up to ~1e5)
 * overflow the fp16 EXL3 projection (NaN). main_proj is followed by an RMSNorm, so
 * a positive per-row scale is exact: bring every row's magnitude into a safe range. */
function safe_taps(taps) {
	let a = taps.to('float32').abs().topk(1, 1).values;              // (n, 1) row maxima
	let sc = a.div(2048.0).clamp(1.0, null);

	return taps.to('float32').div(sc).to('float16');
}

const FP8KV = getenv('AMP_DSPARK_NO_FP8KV') ? 0 : 1;      // e4m3 fake-quant of the drafter's kv rows (as trained)

export function load(tp, opts) {
	opts = opts || {};

	let cfg = tp.cfg, cg = tp.cg, ws = tp.ws, Gs = tp.Gs;
	let S = tp.last;                         // the last stage
	let B = cfg.dspark_block, nb = cfg.num_mtp_layers;
	let max_seq = opts.max_seq || 8192;
	let H = cfg.hidden_size;
	let eps = cfg.rms_norm_eps;
	let plan = tp.plan;

	if (!nb || !B || !ws.has('mtp.0.main_proj.trellis'))
		die('dspark_tp: the checkpoint has no drafter');

	// experts straight to their owner, the rest to the stage's first rank
	ws.upload(function (key) {
		if (index(key, 'mtp.') != 0)
			return null;

		let at = index(key, '.experts.');

		if (at >= 0) {
			let rest = substr(key, at + 9);

			return S.expert_dev[+substr(rest, 0, index(rest, '.'))];
		}

		return S.devs[0];
	});

	let st0 = S.devs[0];
	let blocks = [];

	for (let b = 0; b < nb; b++) {
		let pfx = `mtp.${b}`;
		let attn = dsv4.build_attn(ws, cfg, b, st0, pfx, 'sliding');
		let hc_a = dsv4.build_hc(ws, b, 'attn', st0, pfx);
		let hc_f = dsv4.build_hc(ws, b, 'ffn', st0, pfx);
		let an = ws.gpu16(`${pfx}.attn_norm.weight`, st0);
		let fnw = ws.gpu16(`${pfx}.ffn_norm.weight`, st0);
		let mkey = `${pfx}.ffn`;
		let gate_t = ws.gpu(mkey + '.gate.weight', st0).to('float32').t().contiguous();
		let bias = ws.gpu(mkey + '.gate.bias', st0).to('float32').reshape([1, -1]).contiguous();
		let shared = dsv4_tp.build_mlp(ws, mkey + '.shared_experts', st0);
		let row = [];

		for (let g = 0; g < Gs; g++) {
			let dev = S.devs[g];
			let e0 = plan.experts[g][0], en = plan.experts[g][1];
			let ex = [];

			for (let e = e0; e < e0 + en; e++)
				push(ex, dsv4_tp.build_mlp(ws, `${mkey}.experts.${e}`, dev));

			let R = {
				dev: dev, kind: 'sliding', idx: b,
				hc_attn: dsv4_tp.hc_to(hc_a, dev), hc_ffn: dsv4_tp.hc_to(hc_f, dev),
				attn_norm: an.cuda(dev), ffn_norm: fnw.cuda(dev),
				attn: dsv4_tp.shard_attn(cfg, attn, g, Gs, dev),
				moe: {
					type: 'moe', dev: dev, experts: ex, e0: e0, e1: e0 + en, router: 'sqrtsp', async_moe: false,
					gate_t: gate_t.cuda(dev), bias: bias.cuda(dev), tid2eid: null, ids: null,
					shared: dsv4_tp.shard_mlp(shared, plan.shared[g], dev)
				}
			};

			R.attn.nc_block = B;
			push(row, R);
		}

		push(blocks, row);

		for (let key in ws.list())
			if (index(key, pfx + '.') == 0 && index(key, '.experts.') < 0)
				ws.release(key, st0);

		attn = shared = null;
		gc();
	}

	// per rank: the main-kv rows of every block (cache object with kv only), the tap projection
	let last = `mtp.${nb - 1}`;
	let main_proj = models.linear(ws, 'mtp.0.main_proj', false, st0);
	let main_norm = ws.gpu16('mtp.0.main_norm.weight', st0);
	let hh = {
		fn: ws.gpu(last + '.hc_head_fn', st0).to('float32').contiguous(),
		base: ws.gpu(last + '.hc_head_base', st0).to('float32').contiguous(),
		scale: ws.gpu(last + '.hc_head_scale', st0).to('float32').contiguous()
	};
	let dnorm = ws.gpu16(last + '.norm.weight', st0);
	let mk1 = ws.gpu16(last + '.markov_head.markov_w1.weight', st0);
	let mk2 = ws.gpu16(last + '.markov_head.markov_w2.weight', st0);
	let confw = ws.gpu16(last + '.confidence_head.proj.weight', st0);              // (1, hidden + rank)

	let rk = [];

	for (let g = 0; g < Gs; g++) {
		let dev = S.devs[g], dv = 'cuda:' + dev;

		push(rk, {
			dev: dev,
			cls: map(blocks, () => ({dev: dev, kv: torch.zeros([dsv4.ring_rows(max_seq), cfg.head_dim], 'float16', dv)})),
			main_proj: dsv4_tp.copy_lin(main_proj, dev),
			main_norm: main_norm.cuda(dev),
			hc_head: dsv4_tp.hc_to(hh, dev),
			norm: dnorm.cuda(dev),
			dummy_q: torch.zeros([1, 1, cfg.head_dim], 'float16', dv)
		});
	}

	// block-chain state shared by the graphs
	let dev0 = tp.devs[0];
	let ent_d = {
		ids: torch.from_array((function () {
			let a = [ 0 ];

			for (let i = 1; i < B; i++)
				push(a, cfg.dspark_noise_id);

			return a;
		})(), 'int64', 'cuda:' + dev0),
		drafts: null
	};

	let d = {blocks: blocks, rk: rk, B: B, ent: ent_d,
		threshold: opts.threshold != null ? opts.threshold : +(getenv('AMP_DSPARK_CONF') || 0.3)};

	/* main rows for the positions pos .. pos + n - 1 from taps (n, 3 * hidden) fp16
	 * (device of any rank): main_norm(main_proj(taps)), then per block the weighted
	 * kv norm + rope written into that block's main-kv cache. Shape-generic and
	 * graph-safe when the position state st[g] holds pos. */
	d.update_rows = function (taps, st) {
		let n = taps.shape()[0];

		if (n > rk[0].cls[0].kv.shape()[0] - cfg.window)
			die(`dspark: ${n} main rows in one update do not fit the KV ring`);

		for (let g = 0; g < Gs; g++) {
			let R = rk[g];
			let mx = torch.rms_norm(R.main_proj.forward(safe_taps(taps.cuda(R.dev))), R.main_norm, eps);
			let dq = (n == 1) ? R.dummy_q : torch.zeros([n, 1, cfg.head_dim], 'float16', 'cuda:' + R.dev);

			for (let b = 0; b < nb; b++) {
				let A = blocks[b][g].attn;

				torch.dsv4_qkv(dq, A.wkv.forward(mx).view([n, cfg.head_dim]), A.kv_norm, A.inv_freq,
					st[g].pos_f, st[g].pos_i, R.cls[b].kv, cfg.rope_dim, eps, 1.0, FP8KV);
			}
		}
	};

	/* the draft chain: ids buffer [seed, noise ...] on dev0 -> d1 .. dB (device int64 (B)) */
	d.draft_body = function (sts) {
		let h0 = tp.embed_tokens.index(ent_d.ids).to('float32').reshape([B, 1, H])
			.expand([B, 4, H]).contiguous();
		let h = map(S.devs, (dv, g) => h0.cuda(dv));
		let dst = [];

		for (let g = 0; g < Gs; g++)
			push(dst, sts[g]);

		for (let b = 0; b < nb; b++) {
			let cls = map(rk, (R) => R.cls[b]);

			h = tp.block_dec(S, blocks[b], cls, h, dst, null);
		}

		let xs = [], pre0 = null;

		for (let g = 0; g < Gs; g++) {
			let pre = dsv4.hc_head(cg, rk[g].hc_head, h[g]);

			if (g == 0)
				pre0 = pre;

			push(xs, glm5.rms_norm_f32(pre, rk[g].norm, eps));
		}

		let logits = tp.head_state(xs, B).to('float32');                       // (B, V) on S.devs[0]
		let dv0 = 'cuda:' + S.devs[0];
		let prev = ent_d.ids.narrow(0, 0, 1).cuda(S.devs[0]).view([1]);
		let outs = [], embs = [];

		for (let i = 0; i < B; i++) {
			let emb = mk1.index(prev);                                          // (1, rank)
			let row = logits.narrow(0, i, 1).add(torch.matmul(emb, mk2.t()).to('float32'));

			push(embs, emb);
			prev = row.argmax(-1).view([1]);
			push(outs, prev);
		}

		// per-position confidence: sigmoid(proj([pre-norm state, markov embedding]))
		ent_d.conf = torch.matmul(torch.cat([pre0.to('float16'), torch.cat(embs, 0)], 1), confw.t())
			.to('float32').sigmoid().view([B]);

		return torch.cat(outs, 0);
	};

	/* capture the draft graph and, per verify window V (drafts verified per round),
	 * the update graph next to the target graph `graph` (tp.graphs(cache, [ 1, V + 1 ... ]));
	 * must run after it (shares the position states) and before any prefill (the
	 * warm-up clobbers the caches) */
	d.build = function (graph, cache, windows) {
		let sts = tp.dec_states;
		let lst = slice(sts, S.id * Gs, (S.id + 1) * Gs);

		for (let st in lst)
			dsv4.dec_set_pos(st, 0);

		// the draft chain
		let draft_cap = function () {ent_d.drafts = d.draft_body(lst);};

		draft_cap();                                         // eager warm-up
		torch.synchronize();

		d.gdraft = torch.graph();
		d.gdraft.capture_begin(dev0, S.devs);
		draft_cap();
		d.gdraft.capture_end();
		torch.synchronize();

		// the main-row update from the verify block's taps, per window
		d.gupd = {};

		for (let V in windows) {
			let vent = graph.ent(V + 1);

			if (!vent)
				die(`dspark_tp: the target graph for ${V + 1} rows is missing`);

			let upd = function () {d.update_rows(torch.cat(vent.taps, 1), lst);};

			upd();
			torch.synchronize();

			let gu = torch.graph();

			gu.capture_begin(S.devs[0], slice(S.devs, 1));
			upd();
			gu.capture_end();
			torch.synchronize();
			d.gupd[V] = gu;
		}

		d.graph = graph;
		d.lst = lst;
		d.cache = cache;
	};

	/* d.round with a synchronize after every phase: seconds of {draft, verify, update} */
	d.round_timed = function (seed, p0, V) {
		let vent = d.graph.ent(V + 1);
		let T = [ 0, 0, 0 ];
		let t0 = clock(true);

		function lap(k) {
			torch.synchronize();

			let t1 = clock(true);

			T[k] += (t1[0] - t0[0]) + (t1[1] - t0[1]) / 1e9;
			t0 = t1;
		}

		for (let st in d.lst)
			dsv4.dec_set_pos(st, p0);

		ent_d.ids.narrow(0, 0, 1).fill_(seed);
		torch.synchronize();
		t0 = clock(true);
		d.gdraft.replay();
		lap(0);

		vent.ids.narrow(0, 0, 1).fill_(seed);
		vent.ids.narrow(0, 1, V).copy_(ent_d.drafts.narrow(0, 0, V));
		d.graph.replay_size(V + 1, p0);
		lap(1);

		d.gupd[V].replay();
		lap(2);

		return T;
	};

	/* prompt rows: taps (n, 3 * hidden) for positions pos .. pos + n - 1 (eager) */
	d.prefill_rows = function (taps, pos) {
		let n = taps.shape()[0];
		let tmp = map(d.lst, (st) => ({dev: st.dev, pos_i: torch.from_array([ pos ], 'int64', 'cuda:' + st.dev),
			pos_f: torch.from_array([ pos * 1.0 ], 'float32', 'cuda:' + st.dev).view([1, 1])}));

		d.update_rows(taps, tmp);
	};

	/* draft + verify one round from the seed token (not yet in the target cache) at
	 * position p0 (target rows < p0 valid). The verify window is the longest prefix of
	 * the drafts whose confidence reaches the threshold (at most `Vmax`; 0 = a plain
	 * one-row step): {V, dr: [d1 .. dV], am: [greedy token after each of the V + 1
	 * verified rows], logits: (V + 1, vocab) on the head device}. The target cache then
	 * holds V + 1 rows; the caller sets cache.length to p0 + accepted + 1 (a rollback is
	 * just that). */
	d.run = function (seed, p0, Vmax) {
		for (let st in d.lst)
			dsv4.dec_set_pos(st, p0);

		ent_d.ids.narrow(0, 0, 1).fill_(seed);
		torch.synchronize();                       // (the position fills ran eagerly on other streams)
		d.gdraft.replay();
		torch.synchronize();                       // (the graph's work on the other GPUs runs on its own streams)

		let drafts = ent_d.drafts;
		let conf = ent_d.conf.tolist();
		let dr = drafts.tolist();
		let V = 0;

		while (V < Vmax && conf[V] >= d.threshold)
			V++;

		let vent = d.graph.ent(V + 1);

		vent.ids.narrow(0, 0, 1).fill_(seed);

		if (V > 0)
			vent.ids.narrow(0, 1, V).copy_(drafts.narrow(0, 0, V));

		d.graph.replay_size(V + 1, p0);
		torch.synchronize();
		d.gupd[V].replay();

		return {V: V, dr: slice(dr, 0, V), am: vent.am.tolist(), logits: vent.out, conf: conf};
	};

	/* greedy round: {tokens: [accepted ...], bonus, n_acc}, cache.length updated */
	d.round = function (seed, p0, V) {
		let r = d.run(seed, p0, V);
		let k = 0;

		while (k < r.V && r.dr[k] == r.am[k])
			k++;

		d.cache.length = p0 + k + 1;

		return {tokens: slice(r.dr, 0, k), bonus: r.am[k], n_acc: k, V: r.V};
	};

	/* host-RAM prompt-cache store: the drafter's main-kv rings (positions p at slot p % rows).
	 * Only the rows below min(n, ring) hold anything. */
	d.ring_export = function (n, put) {
		let k = min(n, rk[0].cls[0].kv.shape()[0]);

		return {n: k, rings: map(rk, (R) => map(R.cls, (c) => put(c.kv.narrow(0, 0, k))))};
	};

	d.ring_import = function (o) {
		for (let g = 0; g < length(rk); g++)
			for (let b = 0; b < length(rk[g].cls); b++)
				rk[g].cls[b].kv.narrow(0, 0, o.n).copy_(o.rings[g][b]);
	};

	d.ring_bytes = function (n) {
		return length(rk) * nb * min(n, rk[0].cls[0].kv.shape()[0]) * cfg.head_dim * 2;
	};

	return d;
}
