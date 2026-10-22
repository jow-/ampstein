// ucode/expert_dyn.uc — dynamic residency of MoE experts kept in host memory (qwen4_tp host_experts.swaps).
//
// The set of experts in VRAM starts as the routing profile's (expert_host.uc) and follows the requests: the
// router's picks of every decode step (a graph replay) are logged on the device and added to a decayed score
// per (layer, expert), and every `interval` steps the cold expert with the highest score of a layer takes the
// place of the resident one with the lowest, if it is clearly ahead. Exchanges stay inside a layer, so the
// number of cold experts per layer, and with it the staging buffers and the pinned arenas, never change.
//
// The scores live on the device (one [layer, expert] table per rank, picks of other ranks go to a spare column):
// a step costs a few small kernels and no host sync, a plan two top-1 selections and one small read-back.
//
// One exchange of cold expert A (arena slot k) and resident expert B, all queued on the compute stream between
// two replays (a graph replay then never sees a half-done exchange):
//   H2D  A's trellis (arena slot k)  -> a spare device slot            A now lives there
//   D2H  B's trellis (its device slot) -> arena slot k                 queued behind the read of slot k above
//   B's old device slot becomes a spare
// and the entries of A and B in the decode pointer tables are rewritten in place, so the captured graphs are
// untouched (the prefill-staging tables are rebuilt at the next staged chunk). Only handles move
// (models.LinearEXL3Cold): no weight is copied twice, and no CPU copy is involved.
//
// An exchange moves one expert in each direction, so it only pays for an expert that is reused: the budget
// (swaps per step) and the margin keep the traffic well below what the saved misses would cost.

import {readfile} from 'fs';

const COOP_ROWS = 8;                 // decode steps log at most this many rows (glm5.COOP_ROWS)
const NAMES = ['gate', 'up', 'down'];
const TABLES = ['g_trellis', 'u_trellis', 'd_trellis'];

function env(name, dflt) {
	let v = getenv(name);

	return (v == null || v == '') ? dflt : +v;
}

export function create(tp, layers, nl, cfg, hx) {
	let NE = cfg.n_routed_experts, K = cfg.num_experts_per_tok;
	let G = length(layers[0]);
	let interval = hx.interval ?? env('AMP_EXPERT_DYN_INTERVAL', 8);
	let decay = hx.decay ?? env('AMP_EXPERT_DYN_DECAY', 0.99);          // per decode step
	let margin = hx.margin ?? env('AMP_EXPERT_DYN_MARGIN', 0.25);
	let prior = hx.prior ?? env('AMP_EXPERT_DYN_PRIOR', 64);             // the profile counts as this many steps
	let LOG = env('AMP_EXPERT_DYN_LOG', 0);
	let per_plan = max(1, int(hx.swaps * interval / G + 0.5));              // exchanges per rank and plan
	let grow = 1.0 / decay;
	let dev0 = layers[0][0].moe.dev;
	let log = torch.zeros([nl, COOP_ROWS, K], 'int64', 'cuda:' + dev0);

	// ---- the router's picks of every layer, copied into `log` inside the decode step (graph or eager)
	function logger(L) {
		return (idx) => {
			log.narrow(0, L, 1).view([COOP_ROWS, K]).narrow(0, 0, idx.shape()[0]).copy_(idx.to('int64'));
		};
	}

	for (let i = 0; i < nl; i++)
		layers[i][0].moe.sellog = logger(i);

	// ---- per rank: scores (the profile's counts as a prior, then the decayed picks, kept as a sum of grow^step so
	// that nothing has to be decayed per step), the cold mask, and the constants of the pick -> column mapping
	let prof = json(readfile(hx.profile)).counts;
	let R = [];

	for (let r = 0; r < G; r++) {
		let M0 = layers[0][r].moe, dv = 'cuda:' + M0.dev;
		let e0 = M0.e0, n = length(M0.experts), W = n + 1;         // column n collects the picks of the other ranks
		let seed = [], cold = [], base = [];

		for (let L = 0; L < nl; L++) {
			let tot = 0.0, ex = layers[L][r].moe.experts;

			for (let e = 0; e < NE; e++)
				tot += prof[L][e];

			for (let j = 0; j < n; j++) {
				push(seed, tot > 0 ? prof[L][e0 + j] / tot * K * prior : 0.0);
				push(cold, ex[j].slot != null ? 1.0 : 0.0);
			}

			push(seed, 0.0);
			push(cold, 0.0);
			push(base, L * W);
		}

		// (the spare device slots: sets of three trellis tensors)
		let sh = null, sets = [];

		for (let L = 0; L < nl && !sh; L++)
			if (layers[L][r].moe.stage)
				sh = map(NAMES, (nm) => layers[L][r].moe.experts[0][nm].trellis.shape());

		// (a freed set goes back on the list and the next exchange of the plan takes it: two sets are plenty)
		for (let k = 0; k < 2 && sh; k++)
			push(sets, map(sh, (s) => torch.empty(s, 'int16', dv)));

		push(R, {
			dv: dv, dvi: M0.dev, e0: e0, n: n, W: W,
			sc: torch.from_array(seed, 'float32', dv).reshape([nl, W]),
			cold: torch.from_array(cold, 'float32', dv).reshape([nl, W]),
			e0t: torch.full([1], e0, 'int64', dv), nt: torch.full([1], n, 'int64', dv),
			base: torch.from_array(base, 'int64', dv).reshape([nl, 1, 1]),
			ones: {}, spare: sets, miss: torch.zeros([1], 'float32', dv)
		});
	}

	let scale = 1.0, step = 0, picks = 0;
	let st = { swaps: 0, plans: 0, plan_ms: 0.0, exec_ms: 0.0 };

	function ptrs(ts) {
		return map(ts, (t) => t.data_ptr());
	}

	// exchange cold expert jc and resident expert jh of M with the spare set sp; returns B's old device set
	function exchange(M, jc, jh, sp) {
		let stg = M.stage, A = M.experts[jc], B = M.experts[jh];
		let dv = 'cuda:' + M.dev;
		let hostv = map(NAMES, (nm) => A[nm].trellis);
		let bdev = map(NAMES, (nm) => B[nm].dev_trellis);

		for (let m = 0; m < 3; m++) {
			torch.copy_h2d_async(sp[m], hostv[m]);
			torch.copy_async(hostv[m], bdev[m]);
		}

		let pa = ptrs(sp), ph = ptrs(hostv);
		let idx = torch.from_array([jc, jh], 'int64', dv);

		for (let m = 0; m < 3; m++)
			M.coop[TABLES[m]].index_copy_(0, idx, torch.from_array([pa[m], ph[m]], 'int64', dv));

		for (let m = 0; m < 3; m++) {
			A[NAMES[m]].trellis = sp[m];
			A[NAMES[m]].dev_trellis = sp[m];
			B[NAMES[m]].trellis = hostv[m];
			B[NAMES[m]].dev_trellis = null;
		}

		if (M.skip)
			M.skip.cold.index_copy_(0, torch.from_array([M.e0 + jc, M.e0 + jh], 'int64', dv),
				torch.from_array([0.0, 1.0], 'float32', dv));

		B.slot = A.slot;
		A.slot = null;
		stg.pf = null;         // the staging tables are rebuilt from the experts at the next staged chunk
		stg.hv = null;

		return bdev;
	}

	function now_ms() {
		let c = clock(true);

		return c[0] * 1000.0 + c[1] / 1e6;
	}

	// the cold mask of rank r follows the exchanges
	function mark(r, L, jc, jh) {
		let rk = R[r];

		rk.cold.view([-1]).index_copy_(0, torch.from_array([L * rk.W + jc, L * rk.W + jh], 'int64', rk.dv),
			torch.from_array([0.0, 1.0], 'float32', rk.dv));
	}

	// the best exchanges of rank r: per layer the cold expert with the highest score against the resident one
	// with the lowest, if it leads by the margin; the largest leads first
	function plan(r) {
		let rk = R[r], hot = rk.cold.le(0);
		let neg = torch.full([1], -1e30, 'float32', rk.dv);
		let cb = torch.where(hot, neg, rk.sc).topk(1, -1);
		let hb = torch.where(hot, rk.sc.neg(), neg).topk(1, -1);
		let cv = cb.values.reshape([-1]).cpu().tolist(), cj = cb.indices.reshape([-1]).cpu().tolist();
		let hv = hb.values.reshape([-1]).cpu().tolist(), hj = hb.indices.reshape([-1]).cpu().tolist();
		let cand = [];

		for (let L = 0; L < nl; L++) {
			let M = layers[L][r].moe;

			if (!M.stage || !M.coop || cv[L] < -1e29 || hj[L] >= rk.n)
				continue;

			let lo = -hv[L];

			if (cv[L] > lo * (1.0 + margin) + 0.5 * scale)
				push(cand, [cv[L] - lo, L, cj[L], hj[L]]);
		}

		sort(cand, (a, b) => b[0] - a[0]);

		return slice(cand, 0, per_plan);
	}

	function run_plan() {
		let t0 = now_ms();

		for (let r = 0; r < G; r++) {
			let todo = plan(r);
			let t1 = now_ms();

			st.plan_ms += t1 - t0;

			for (let c in todo) {
				let sp = pop(R[r].spare);

				if (!sp)
					break;

				push(R[r].spare, exchange(layers[c[1]][r].moe, c[2], c[3], sp));
				mark(r, c[1], c[2], c[3]);
				st.swaps++;
			}

			t0 = now_ms();
			st.exec_ms += t0 - t1;
		}

		st.plans++;

		if (LOG && st.plans % LOG == 0) {
			let miss = 0.0;

			for (let r = 0; r < G; r++)
				miss += R[r].miss.item();

			warn(sprintf('expert_dyn: step %d: %d exchanges in %d plans (plan %.2f ms, queue %.2f ms per plan), '
				+ 'live miss rate %.3f over %d picks\n', step, st.swaps, st.plans, st.plan_ms / st.plans,
				st.exec_ms / st.plans, picks ? miss / picks : 0.0, picks));
		}
	}

	return {
		interval: interval,
		swaps_per_step: hx.swaps,
		stats: st,

		// the S rows of the replay that has just been queued: add their picks to the scores, plan now and then
		after: function (S) {
			let lg0 = log.narrow(1, 0, S);

			for (let r = 0; r < G; r++) {
				let rk = R[r];
				let lg = (r == 0) ? lg0 : lg0.contiguous().cuda(rk.dvi);
				let loc = lg.sub(rk.e0t);
				let bad = loc.neg().le(0).to('int64').mul(loc.le(rk.n - 1).to('int64')).le(0);
				let flat = torch.where(bad, rk.nt, loc).add(rk.base).reshape([-1]);

				if (!rk.ones[S])
					rk.ones[S] = torch.ones([nl * S * K], 'float32', rk.dv);

				rk.sc.view([-1]).index_add_(0, flat, rk.ones[S].mul(scale));

				if (LOG)
					rk.miss.add_(rk.cold.view([-1]).gather(0, flat).sum());
			}

			picks += nl * S * K;
			scale *= grow;

			if (scale > 1e12) {
				for (let r = 0; r < G; r++)
					R[r].sc.mul_(1.0 / scale);

				scale = 1.0;
			}

			if (++step % interval == 0)
				run_plan();
		},

		// exchange cold expert jc with resident expert jh of layer L on rank r (tests)
		force: function (L, r, jc, jh) {
			let sp = pop(R[r].spare);

			if (!sp)
				die('expert_dyn: no spare slot');

			push(R[r].spare, exchange(layers[L][r].moe, jc, jh, sp));
			mark(r, L, jc, jh);
		}
	};
}
