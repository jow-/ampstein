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
// With pipeline stages (dsv41_tp: `stages` = [{first, end}, ...], layers [first, end) of a stage on its pair of GPUs)
// every stage keeps its own log and scores, `after_stage(s, S)` follows the graph replay of stage s, and the exchanges
// of a plan are not queued at once but behind the stage's next replay (`pend`): that stage's GPUs are idle while the
// later stages run, so the transfers (one H2D and one D2H of an expert, ~7 ms each exchange on jai's x4 links) hide
// there instead of delaying the next token.
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
import * as math from 'math';
import * as tune from './tune.uc';

const COOP_ROWS = 16;                // decode steps log at most this many rows (glm5.COOP_ROWS)
const SLOT = 2;                      // stream_side slot of the write-back (the all-reduces of the double-buffered prefill use it, off by default)
const NAMES = ['gate', 'up', 'down'];
const TABLES = ['g_trellis', 'u_trellis', 'd_trellis'];

function env(name, dflt) {
	let v = getenv(name);

	return (v == null || v == '') ? dflt : +v;
}

export function create(tp, layers, nl, cfg, hx, stages) {
	let NE = cfg.n_routed_experts, K = cfg.num_experts_per_tok;

	stages ??= [ {first: 0, end: nl} ];

	let NSG = length(stages);
	let G = length(layers[stages[0].first]);
	let interval = hx.interval ?? env('AMP_EXPERT_DYN_INTERVAL', 8);
	let decay = hx.decay ?? env('AMP_EXPERT_DYN_DECAY', 0.99);          // per decode step
	let margin = hx.margin ?? env('AMP_EXPERT_DYN_MARGIN', 0.25);
	let prior = hx.prior ?? env('AMP_EXPERT_DYN_PRIOR', 64);             // the profile counts as this many steps
	let LOG = env('AMP_EXPERT_DYN_LOG', 0);
	let per_plan = max(1, int(hx.swaps * interval / G + 0.5));              // exchanges per rank and plan
	let grow = 1.0 / decay;
	let ctl = getenv('AMP_EXPERT_DYN_CTL');          // a JSON file {interval, swaps, margin, decay}, re-read now and then (tuning)
	let swaps = hx.swaps;
	// prefill chunks feed the scores too (the prompt tells which experts the answer will use): a chunk ages the scores like
	// `pf_steps` decode steps, and a picked expert counts `pf_weight` of a decode pick
	let pf_steps = hx.pf_steps ?? env('AMP_EXPERT_DYN_PF_STEPS', 32);
	let pf_weight = hx.pf_weight ?? env('AMP_EXPERT_DYN_PF_WEIGHT', 1.0 / 32);
	let pf_grow = math.pow(1.0 / decay, pf_steps);
	let logs = [];

	// ---- the router's picks of every layer, copied into the stage's `log` inside the decode step (graph or eager)
	function logger(log, L) {
		return (idx) => {
			log.narrow(0, L, 1).view([COOP_ROWS, K]).narrow(0, 0, idx.shape()[0]).copy_(idx.to('int64'));
		};
	}

	for (let st in stages) {
		let nls = st.end - st.first;
		let log = torch.zeros([nls, COOP_ROWS, K], 'int64', 'cuda:' + layers[st.first][0].moe.dev);

		push(logs, log);

		for (let i = st.first; i < st.end; i++)
			layers[i][0].moe.sellog = logger(log, i - st.first);
	}

	// ---- per unit (stage s, rank r): scores (the profile's counts as a prior, then the decayed picks, kept as a sum of
	// grow^step so that nothing has to be decayed per step), the cold mask, and the constants of the pick -> column mapping
	let prof = json(readfile(hx.profile)).counts;
	let U = [];

	for (let s = 0; s < NSG; s++) {
		let st = stages[s], nls = st.end - st.first;

		for (let r = 0; r < G; r++) {
			let M0 = layers[st.first][r].moe, dv = 'cuda:' + M0.dev;
			let e0 = M0.e0, n = length(M0.experts), W = n + 1;         // column n collects the picks of the other ranks
			let seed = [], cold = [], base = [];

			for (let l = 0; l < nls; l++) {
				let L = st.first + l, tot = 0.0, ex = layers[L][r].moe.experts;

				for (let e = 0; e < NE; e++)
					tot += prof[L][e];

				for (let j = 0; j < n; j++) {
					push(seed, tot > 0 ? prof[L][e0 + j] / tot * K * prior : 0.0);
					push(cold, ex[j].slot != null ? 1.0 : 0.0);
				}

				push(seed, 0.0);
				push(cold, 0.0);
				push(base, l * W);
			}

			// (the spare device slots: sets of three trellis tensors)
			let sh = null, sets = [];

			for (let L = st.first; L < st.end && !sh; L++)
				if (layers[L][r].moe.stage)
					sh = map(NAMES, (nm) => layers[L][r].moe.experts[0][nm].trellis.shape());

			// (a freed set goes back on the list and the next exchange of the plan takes it: two sets are plenty)
			for (let k = 0; k < 2 && sh; k++)
				push(sets, map(sh, (sp) => torch.empty(sp, 'int16', dv)));

			push(U, {
				s: s, r: r, first: st.first, end: st.end, nls: nls,
				dv: dv, dvi: M0.dev, e0: e0, n: n, W: W,
				sc: torch.from_array(seed, 'float32', dv).reshape([nls, W]),
				cold: torch.from_array(cold, 'float32', dv).reshape([nls, W]),
				e0t: torch.full([1], e0, 'int64', dv), nt: torch.full([1], n, 'int64', dv),
				base: torch.from_array(base, 'int64', dv).reshape([nls, 1, 1]),
				ones: {}, spare: sets, miss: torch.zeros([1], 'float32', dv), pend: []
			});
		}
	}

	let scale = [], step = 0, picks = 0;
	let st = { swaps: 0, plans: 0, plan_ms: 0.0, exec_ms: 0.0 };

	for (let s = 0; s < NSG; s++)
		push(scale, 1.0);

	function ptrs(ts) {
		return map(ts, (t) => t.data_ptr());
	}

	// exchange cold expert jc and resident expert jh of M with the spare set sp; returns B's old device set. One native
	// call (torch.expert_exchange) queues it all: the copies (the link is full duplex: B's matrix m goes down while A's
	// matrix m + 1 comes up, B's bytes replacing A's in the arena slot after A's of the same matrix have been read) and
	// the small writes (pointer tables, masks, maps) as one kernel; `extra`: more [address, value, size] writes (mark)
	const F1 = 1065353216;          // the bits of 1.0f

	function exchange(M, jc, jh, sp, extra) {
		let stg = M.stage, A = M.experts[jc], B = M.experts[jh];
		let hostv = map(NAMES, (nm) => A[nm].trellis);
		let bdev = map(NAMES, (nm) => B[nm].dev_trellis);
		let pa = ptrs(sp), ph = ptrs(hostv);
		let k = A.slot, xd = [], xs = [], pk = [ ...(extra || []) ];

		for (let m = 0; m < 3; m++) {
			let tb = M.coop[TABLES[m]].data_ptr();

			push(pk, [ tb + jc * 8, pa[m], 8 ], [ tb + jh * 8, ph[m], 8 ]);
		}

		if (M.skip) {
			let cb = M.skip.cold.data_ptr();

			push(pk, [ cb + (M.e0 + jc) * 4, 0, 4 ], [ cb + (M.e0 + jh) * 4, F1, 4 ]);
		}

		// CPU experts (glm5.cpu_experts_begin): the CPU layer's entry of arena slot k now computes B — its scales are
		// copied into the entry's pinned tensors behind the trellis, and the map sends A's picks to the GPU, B's to the CPU
		if (M.cpu) {
			for (let m = 0; m < 3; m++) {
				push(xd, M.cpu.suh[m][k], M.cpu.svh[m][k]);
				push(xs, B[NAMES[m]].suh, B[NAMES[m]].svh);
			}

			let mb = M.cpu.map.data_ptr();

			push(pk, [ mb + (M.e0 + jc) * 4, -1, 4 ], [ mb + (M.e0 + jh) * 4, k, 4 ]);

			// (a CPU-only slot of the prefill split: B takes A's place there too)
			if (M.cpu.map_pf && k >= stg.nst) {
				let fb = M.cpu.map_pf.data_ptr();

				push(pk, [ fb + (M.e0 + jc) * 4, -1, 4 ], [ fb + (M.e0 + jh) * 4, k, 4 ]);
			}
		}

		torch.expert_exchange(M.dev, SLOT, sp, hostv, hostv, bdev, xd, xs, pk);

		for (let m = 0; m < 3; m++) {
			A[NAMES[m]].trellis = sp[m];
			A[NAMES[m]].dev_trellis = sp[m];
			B[NAMES[m]].trellis = hostv[m];
			B[NAMES[m]].dev_trellis = null;
		}

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

	// the cold mask of unit u follows the exchanges (the writes, for exchange())
	function mark(u, L, jc, jh) {
		let l = L - u.first, cb = u.cold.data_ptr();

		return [ [ cb + (l * u.W + jc) * 4, 0, 4 ], [ cb + (l * u.W + jh) * 4, F1, 4 ] ];
	}

	// the best exchanges of unit u: per layer the cold expert with the highest score against the resident one
	// with the lowest, if it leads by the margin; the largest leads first
	// the selection on the device: one tensor (values, then the indices as floats, exact below 2^24)
	function select(u) {
		let hot = u.cold.le(0);
		let neg = torch.full([1], -1e30, 'float32', u.dv);
		let cb = torch.where(hot, neg, u.sc).topk(1, -1);
		let hb = torch.where(hot, u.sc.neg(), neg).topk(1, -1);

		return torch.cat([ cb.values.reshape([-1]), cb.indices.reshape([-1]).to('float32'),
			hb.values.reshape([-1]), hb.indices.reshape([-1]).to('float32') ]);
	}

	function plan(u, rb) {
		let r = u.r, nls = u.nls;
		let cv = slice(rb, 0, nls), cj = map(slice(rb, nls, 2 * nls), int);
		let hv = slice(rb, 2 * nls, 3 * nls), hj = map(slice(rb, 3 * nls, 4 * nls), int);
		let cand = [];

		for (let l = 0; l < u.nls; l++) {
			let L = u.first + l, M = layers[L][r].moe;

			if (!M.stage || !M.coop || cv[l] < -1e29 || hj[l] >= u.n)
				continue;

			let lo = -hv[l];

			if (cv[l] > lo * (1.0 + margin) + 0.5 * scale[u.s])
				push(cand, [cv[l] - lo, L, cj[l], hj[l]]);
		}

		sort(cand, (a, b) => b[0] - a[0]);

		return slice(cand, 0, per_plan);
	}

	// queue the exchanges that a plan left for stage s (behind the replay that has just been queued, or at once)
	function exec_pending(s) {
		let t0 = now_ms();

		for (let r = 0; r < G; r++) {
			let u = U[s * G + r];

			for (let c in u.pend) {
				let ex = layers[c[1]][r].moe.experts;

				// (a plan read back late may name a pair that an exchange of an earlier plan has changed since)
				if (ex[c[2]].slot == null || ex[c[3]].slot != null)
					continue;

				let sp = pop(u.spare);

				if (!sp)
					break;

				push(u.spare, exchange(layers[c[1]][r].moe, c[2], c[3], sp, mark(u, c[1], c[2], c[3])));
				st.swaps++;
			}

			u.pend = [];
		}

		st.exec_ms += now_ms() - t0;
	}

	// the picks per local expert of a prefill chunk, as counted by the sorted dispatch of layer L on rank r
	function prefill_count(u, L, count) {
		let l = L - u.first, s = u.s;

		if (l == 0 && u.r == 0)
			scale[s] *= pf_grow;           // (once per chunk and stage)

		u.sc.view([-1]).narrow(0, l * u.W, u.n).add_(count.narrow(0, 0, u.n).to('float32').mul(pf_weight * scale[s]));

		if (scale[s] > 1e12) {
			for (let r = 0; r < G; r++)
				U[s * G + r].sc.mul_(1.0 / scale[s]);

			scale[s] = 1.0;
		}
	}

	for (let u in U)
		for (let L = u.first; L < u.end; L++)
			layers[L][u.r].moe.pflog = ((uu, LL) => (count) => prefill_count(uu, LL, count))(u, L);

	// A decode plan is asynchronous: the selection is copied into pinned memory behind the work queued on the unit's device
	// (a read-back with .cpu() would wait for the exchanges queued there, tens of ms, while the step's logits sit ready)
	// and a flag is copied after it; a later step finds the flag and takes the data. plan_sync is the blocking variant.
	function issue_plan(u) {
		let rb = select(u);

		if (!u.pbuf) {
			u.pbuf = torch.pinned_empty([4 * u.nls], 'float32');
			u.pflag = torch.pinned_empty([1], 'float32');
		}

		u.pflag.fill_(-1.0);
		torch.copy_async(u.pbuf, rb);
		torch.copy_async(u.pflag, torch.full([1], 1.0 + step, 'float32', u.dv));
		u.inflight = true;
		st.plans++;
	}

	function poll_plan(u) {
		if (!u.inflight || u.pflag.item() < 0)
			return;

		u.inflight = false;
		u.pend = plan(u, u.pbuf.tolist());
	}

	function plan_sync(u) {
		return plan(u, select(u).cpu().tolist());
	}

	// plan the units of stage s now and wait for the answer (after a prefill, tests, the single-stage models)
	function run_plan(s) {
		let t0 = now_ms();

		for (let r = 0; r < G; r++)
			U[s * G + r].pend = plan_sync(U[s * G + r]);

		st.plan_ms += now_ms() - t0;
		st.plans++;
	}

	// the S rows of the replay of stage s that has just been queued: add their picks to the scores, then queue the
	// exchanges that the last plan left for this stage
	function after_stage(s, S) {
		let lg0 = logs[s].narrow(1, 0, S);
		let nls = stages[s].end - stages[s].first;

		for (let r = 0; r < G; r++) {
			let u = U[s * G + r];
			let lg = (r == 0) ? lg0 : lg0.contiguous().cuda(u.dvi);
			let loc = lg.sub(u.e0t);
			let bad = loc.neg().le(0).to('int64').mul(loc.le(u.n - 1).to('int64')).le(0);
			let flat = torch.where(bad, u.nt, loc).add(u.base).reshape([-1]);

			if (!u.ones[S])
				u.ones[S] = torch.ones([nls * S * K], 'float32', u.dv);

			u.sc.view([-1]).index_add_(0, flat, u.ones[S].mul(scale[s]));

			if (LOG)
				u.miss.add_(u.cold.view([-1]).gather(0, flat).sum());
		}

		if (LOG)
			picks += nls * S * K;

		scale[s] *= grow;

		if (scale[s] > 1e12) {
			for (let r = 0; r < G; r++)
				U[s * G + r].sc.mul_(1.0 / scale[s]);

			scale[s] = 1.0;
		}

		exec_pending(s);
	}

	// (the control file, or the live tunables of tune.uc: interval, swaps, margin, decay, pf_steps, pf_weight)
	function reload() {
		let c = null;

		if (ctl) {
			try {
				c = json(readfile(ctl));
			}
			catch (e) {
			}
		}
		else
			c = {interval: tune.get('interval', interval), swaps: tune.get('swaps', swaps), margin: tune.get('margin', margin),
				decay: tune.get('decay', decay), pf_steps: tune.get('pf_steps', pf_steps), pf_weight: tune.get('pf_weight', pf_weight)};

		if (type(c) != 'object')
			return;

		if ((c.interval ?? interval) == interval && (c.swaps ?? swaps) == swaps && (c.margin ?? margin) == margin &&
		    (c.decay ?? decay) == decay && (c.pf_steps ?? pf_steps) == pf_steps && (c.pf_weight ?? pf_weight) == pf_weight)
			return;

		interval = c.interval ?? interval;
		swaps = c.swaps ?? swaps;
		margin = c.margin ?? margin;
		decay = c.decay ?? decay;
		grow = 1.0 / decay;
		pf_steps = c.pf_steps ?? pf_steps;
		pf_weight = c.pf_weight ?? pf_weight;
		pf_grow = math.pow(grow, pf_steps);
		per_plan = max(1, int(swaps * interval / G + 0.5));
		warn(`expert_dyn: control file: interval ${interval}, swaps ${swaps}, margin ${margin}, decay ${decay}, pf_steps ${pf_steps}, pf_weight ${pf_weight}\n`);
	}

	// the end of a decode step: take the plans that have arrived, ask for new ones (every stage once per `interval`
	// steps); sync: plan and wait (qwen4_tp)
	let last = {miss: 0.0, picks: 0, swaps: 0};

	// AMP_EXPERT_DYN_LOG=N: every N decode steps, the miss rate and the exchanges of the window since the last line
	function log_window() {
		let miss = 0.0;

		for (let u in U)
			miss += u.miss.item();

		warn(sprintf('expert_dyn: step %d: window miss rate %.4f over %d picks, %d exchanges (%.2f per step), '
			+ 'enqueue %.2f ms per exchange\n', step, picks > last.picks ? (miss - last.miss) / (picks - last.picks) : 0.0,
			picks - last.picks, st.swaps - last.swaps, (st.swaps - last.swaps) * 1.0 / LOG,
			st.swaps ? st.exec_ms / st.swaps : 0.0));
		last = {miss: miss, picks: picks, swaps: st.swaps};
	}

	function tick(sync) {
		step++;

		if (LOG && step % LOG == 0)
			log_window();

		if ((ctl || tune.active) && step % 64 == 0)
			reload();

		if (swaps <= 0)
			return;

		let ph = step % interval;

		for (let s = 0; s < NSG; s++) {
			if (sync) {
				if (ph == int(s * interval / NSG) % interval)
					run_plan(s);

				continue;
			}

			for (let r = 0; r < G; r++)
				poll_plan(U[s * G + r]);

			if (ph == int(s * interval / NSG) % interval)
				for (let r = 0; r < G; r++) {
					let u = U[s * G + r];

					if (!u.inflight && !length(u.pend))
						issue_plan(u);
				}
		}
	}

	return {
		interval: interval,
		swaps_per_step: hx.swaps,
		stats: st,
		after_stage: after_stage,
		tick: tick,

		// after a prefill: the scores know the prompt now; exchange up to `rounds` experts per layer at once (the pipeline is
		// idle, so the transfers just add to the time to the first token)
		settle: function (rounds) {
			let keep = per_plan;

			per_plan = 1000;

			for (let k = 0; k < (rounds ?? 2); k++) {
				for (let s = 0; s < NSG; s++)
					run_plan(s);

				for (let s = 0; s < NSG; s++)
					exec_pending(s);
			}

			per_plan = keep;
		},

		// one graph replay covering all stages (qwen4_tp): the scores, a plan, and its exchanges at once
		after: function (S) {
			for (let s = 0; s < NSG; s++)
				after_stage(s, S);

			tick(true);

			for (let s = 0; s < NSG; s++)
				exec_pending(s);
		},

		// exchange cold expert jc with resident expert jh of layer L on rank r (tests)
		force: function (L, r, jc, jh) {
			let s = 0;

			while (s < NSG - 1 && L >= stages[s].end)
				s++;

			let u = U[s * G + r], sp = pop(u.spare);

			if (!sp)
				die('expert_dyn: no spare slot');

			push(u.spare, exchange(layers[L][r].moe, jc, jh, sp, mark(u, L, jc, jh)));
		}
	};
}
