// ucode/engine.uc — single-sequence generation engine for the OpenAI
// server (qwen35 hybrid models).
//
// One KV cache / GDN state, allocated once at startup for max_ctx
// positions (no per-request allocation of the cache). Requests are
// queued and run one at a time as JOBS that advance in small steps from
// a uloop timer: one prefill chunk or one decode step per tick, so the
// event loop keeps serving I/O (new connections, disconnects) between
// steps.
//
// Prompt-cache reuse (single sequence):
//   - the cache holds `ids` (every processed token, in order)
//   - a new prompt that extends `ids` continues from the live state
//   - otherwise, if it extends the SNAPSHOT prefix (the conversation up
//     to the end of the last message of the previous request, before
//     its generation prompt), the GDN recurrent/conv states are restored
//     from the snapshot and the KV length rewound to it — attention KV
//     beyond that point is simply overwritten
//   - otherwise: full prefill from position 0
//   The GDN layers cannot rewind, hence the snapshot at a stable
//   boundary (see chat_qwen35.render: prefix/gen split).
//
// Job interface (see submit()):
//   job.ids        prompt token ids
//   job.snap_at    prefix length to snapshot (0/null: none)
//   job.images     optional [{start, grid, emb}]: images in the prompt,
//                  start = position of the image's first row in `ids`,
//                  grid = the vision tower's patch grid [1, gh, gw],
//                  emb = its (rows, hidden) output. `ids` carries a
//                  qwen35.MARK_BASE+ marker id on every image row.
//   job.max_new    generation limit
//   job.sampling   {temperature, top_k, top_p, min_p, seed,
//                    presence_penalty, frequency_penalty}
//   job.stop_ids   token ids that end generation (not emitted)
//   job.on_eos(n)  optional: a stop id was sampled (n earlier consultations
//                  of this job); return true to let the generation end, a
//                  number > 0 to redraw the position with the stop ids' logits
//                  lowered by that much, or a non-empty array of token ids to
//                  force them instead (the stop is discarded and the
//                  generation continues after them). job.eos_max (default 4)
//                  limits the consultations per job.
//   job.veto(prefix, id)  optional: constrained generation. A token the engine is
//                  about to accept after the not yet reported `prefix` tokens;
//                  return true to refuse it. It must have no side effects
//                  (called for every token and for draft windows, so keep
//                  it cheap). A refused token is replaced by
//                  job.veto_force(prefix), when that returns token ids (as
//                  for on_eos: the consumer shows them itself), or else
//                  the position is redrawn from the same logits with the
//                  refused tokens lowered
//   job.on_progress(done, total)   prefill progress, return false to abort
//   job.on_token(id)               return false to stop
//   job.on_done(info)              {reason, prompt_tokens, cached_tokens,
//                                    completion_tokens, timings}
//   job.on_error(msg)

import * as qwen35 from './qwen35.uc';
import * as uloop from 'uloop';
import * as math from 'math';
import { readfile, writefile } from 'fs';
import * as sampler from './sampler.uc';
import * as models from './models.uc';
import * as qgraph from './qwen35_graph.uc';

function now_ms() {
	let t = clock(true);

	return t[0] * 1000 + t[1] / 1e6;
}

/* Pipeline rounds per uloop tick during prefill: one round enqueues one
 * layer per device stage (host-side work only; the GPUs run behind). */
const PIPE_ROUNDS = 8;

// AMP_MTP_TIMING=1: per-phase MTP round times (synchronizing at phase
// boundaries), printed per request
const MTP_TIMING = getenv('AMP_MTP_TIMING') != null;
const MTP_TIMING_STEPS = getenv('AMP_MTP_NO_CHAIN') != null;     // draft step by step (a host read after each)

function lcp(a, b) {
	return torch.common_prefix(a, b);   // native: 100k+ ids per request
}

/* --- sampling -------------------------------------------------------------- */

/* The sampling distribution for one position: (…, V) logits (last
 * position used) -> {ids, probs} over the kept candidates, most
 * probable first, renormalized; null at temperature 0 (greedy).
 *
 * Candidates: the top_k logits (default and cap 1000); presence /
 * frequency penalties from `counts` (a per-request (V) fp32 DEVICE
 * tensor of generated-token counts, or null); temperature softmax over
 * the candidates; min_p against the best; top_p over the probability
 * order. The per-candidate work (penalty gather, re-ordering, exp) runs
 * as tensor ops on the GPU; the host only walks the sorted
 * probabilities to the cut — the former all-ucode version cost 4.5 ms
 * per call at top_k 1000 (8.6 ms with penalties), up to 4 calls per MTP
 * round (docs/LESSONS.md L72). */
export function dist(logits, s, counts) {
	let temp = (s.temperature != null) ? s.temperature : 1.0;

	if (temp <= 0)
		return null;

	let flat = logits.reshape([-1]);
	let k = (s.top_k != null && s.top_k > 0) ? s.top_k : 1000;

	if (k > 1000)
		k = 1000;

	let tk = flat.topk(k, 0);
	let vals = tk.values.to('float32');
	let idxs = tk.indices;
	let pp = s.presence_penalty || 0, fp = s.frequency_penalty || 0;

	if ((pp || fp) && counts) {
		let c = counts.gather(0, idxs);                // (k) counts
		let seen = c.le(0).to('float32').neg().add(1.0);   // c > 0

		vals = vals.sub(seen.mul(pp).add(c.mul(fp)));

		// penalties may reorder the candidates
		let re = vals.topk(k, 0);

		vals = re.values;
		idxs = idxs.gather(0, re.indices);
	}

	// probabilities in descending order; the min_p / top_p cut is found
	// on the GPU (counting the candidates inside it) and only the kept
	// prefix crosses to the host
	let p = vals.div(temp).softmax(0);
	let n = p.numel(), keep = n;
	let min_p = s.min_p || 0;
	let top_p = (s.top_p != null) ? s.top_p : 1.0;

	if (min_p > 0 || top_p < 1.0) {
		// keep i while p[i] >= min_p * p[0] and the mass BEFORE i is
		// < top_p (the candidate that crosses top_p stays in)
		let ok = null;

		// p[0] is the maximum: keep p >= min_p * p[0], i.e. not (p < thr)
		if (min_p > 0) {
			let thr = min_p * p.narrow(0, 0, 1).item();

			ok = p.le(thr * (1 - 1e-7) - 1e-38).to('float32').neg().add(1.0);
		}

		if (top_p < 1.0) {
			// fp64 prefix sum: the cut must not move with fp32 rounding
			// at the boundary (the host walk summed in double)
			let p64 = p.to('float64');
			let before = p64.cumsum(0).sub(p64);
			let in_p = before.le(top_p * (1 - 1e-15)).to('float32');

			ok = ok ? ok.mul(in_p) : in_p;
		}

		// both conditions are monotone prefixes: their product too
		keep = max(1, int(ok.sum().item()));
	}

	let kp = p.narrow(0, 0, keep);
	let probs = kp.div(kp.sum()).tolist();
	let ids = idxs.narrow(0, 0, keep).tolist();

	let out = {ids: ids, probs: probs};

	return out;
}

/* Probability of token x under a dist() result. */
function prob_of(d, x) {
	let i = index(d.ids, x);

	return (i >= 0) ? d.probs[i] : 0.0;
}

/* Draw from a dist() result, optionally excluding one token (the
 * speculative-sampling residual for a one-hot draft). */
function draw(d, exclude) {
	let total = 0.0;

	for (let i = 0; i < length(d.ids); i++)
		if (d.ids[i] !== exclude)
			total += d.probs[i];

	if (total <= 0)
		return d.ids[0];

	let r = math.rand(0, total);
	let last = d.ids[0];

	for (let i = 0; i < length(d.ids); i++) {
		if (d.ids[i] === exclude)
			continue;

		last = d.ids[i];
		r -= d.probs[i];

		if (r < 0)
			return d.ids[i];
	}

	return last;
}

/* One speculative-sampling decision for a one-hot (greedy) draft x
 * against the target distribution d: accept x with probability p(x),
 * else draw the replacement from p with x removed. The emitted token is
 * distributed exactly as p (tests/spec_sampling.uc). Returns
 * {accepted, token}. */
export function spec_step(d, x) {
	if (math.rand(0, 1.0) < prob_of(d, x))
		return {accepted: true, token: x};

	return {accepted: false, token: draw(d, x)};
}

/* --- async sampling (notifier-driven) --------------------------------------- */

/* The engine's decode loop runs against the event loop: a phase enqueues
 * device work, hands the small results to pinned host buffers with
 * torch.copy_async(), arms the completion notifier and returns — the
 * uloop callback continues the job once the results are visible, and
 * the event loop serves I/O while the GPUs run. These helpers split
 * dist() (above) along that line; they are exported for tests.
 *
 * bufs holds preallocated PINNED CPU tensors of the maximum candidate
 * count: {p: float32[K], idxs: int64[K], keep: float32[1],
 * tok: int64[1]}. Each call copies into narrow(0, 0, k) views — a
 * contiguous view of a pinned storage stays pinned, and copy_async
 * matches byte sizes against the view. */

/* GPU half of dist(): the same candidate pipeline, but the min_p
 * threshold is computed on the device (p[0] * min_p as a (1,) tensor;
 * le() broadcasts it over p), so building the mask needs no host read.
 * Ends with copy_async of the probabilities, the candidate ids and the
 * kept count into bufs, then arms note on devs (array of device
 * indices, or null for the current device). Returns the state object
 * for dist_finish(). */
export function dist_begin(note, bufs, devs, logits, s, counts) {
	let temp = (s.temperature != null) ? s.temperature : 1.0;
	let flat = logits.reshape([-1]);
	let k = (s.top_k != null && s.top_k > 0) ? s.top_k : 1000;

	if (k > 1000)
		k = 1000;

	let tk = flat.topk(k, 0);
	let vals = tk.values.to('float32');
	let idxs = tk.indices;
	let pp = s.presence_penalty || 0, fp = s.frequency_penalty || 0;

	if ((pp || fp) && counts) {
		let c = counts.gather(0, idxs);                // (k) counts
		let seen = c.le(0).to('float32').neg().add(1.0);   // c > 0

		vals = vals.sub(seen.mul(pp).add(c.mul(fp)));

		// penalties may reorder the candidates
		let re = vals.topk(k, 0);

		vals = re.values;
		idxs = idxs.gather(0, re.indices);
	}

	// probabilities in descending order; the min_p / top_p cut is found
	// on the GPU and only the kept prefix crosses to the host
	let p = vals.div(temp).softmax(0);
	let min_p = s.min_p || 0;
	let top_p = (s.top_p != null) ? s.top_p : 1.0;
	let ok = null;

	// keep i while p[i] >= min_p * p[0] (p[0] is the maximum) and the
	// mass BEFORE i is < top_p (the candidate that crosses top_p stays)
	if (min_p > 0) {
		let thr = p.narrow(0, 0, 1).mul(min_p * (1 - 1e-7)).sub(1e-38);

		ok = p.le(thr).to('float32').neg().add(1.0);
	}

	if (top_p < 1.0) {
		// fp64 prefix sum: the cut must not move with fp32 rounding
		// at the boundary (the host walk summed in double)
		let p64 = p.to('float64');
		let before = p64.cumsum(0).sub(p64);
		let in_p = before.le(top_p * (1 - 1e-15)).to('float32');

		ok = ok ? ok.mul(in_p) : in_p;
	}

	torch.copy_async(bufs.p.narrow(0, 0, k), p);
	torch.copy_async(bufs.idxs.narrow(0, 0, k), idxs);

	if (ok)
		torch.copy_async(bufs.keep, ok.sum().reshape([1]));
	else
		bufs.keep.fill_(k);    // host-side store: no mask kernel, no copy

	if (note)
		note.arm(devs);

	return {k: k};
}

/* Host half of dist(): cut + renormalize from the pinned buffers (plain
 * CPU reads — the wake-up guarantees the copies completed). Fills the
 * ids/probs of the state object dist_begin returned. */
export function dist_finish(bufs, d) {
	let keep = max(1, int(bufs.keep.item()));
	let raw = bufs.p.narrow(0, 0, keep).tolist();
	let ids = bufs.idxs.narrow(0, 0, keep).tolist();
	let total = 0.0;

	for (let i = 0; i < keep; i++)
		total += raw[i];

	let probs = [];

	for (let i = 0; i < keep; i++)
		push(probs, raw[i] / total);

	d.ids = ids;
	d.probs = probs;

	return d;
}

/* Greedy decode, async: argmax on the device, one int64 across to the
 * pinned buffer, wake. The token is bufs.tok.item() after the wake-up. */
export function argmax_begin(note, bufs, devs, logits) {
	torch.copy_async(bufs.tok, logits.reshape([-1]).argmax().reshape([1]));
	if (note)
		note.arm(devs);
}

/* --- engine ---------------------------------------------------------------- */

export function create(model, opts) {
	opts = opts || {};

	let cfg = model.cfg;
	let max_ctx = opts.max_ctx || 32768;
	let chunk = opts.prefill_chunk || 1024;
	let cache = model.new_cache(max_ctx);

	let snap = {};
	let tsnap = {};          // the turn checkpoint (see plan_reuse): second GDN snapshot
	let gdn_layers = [];

	// The turn checkpoint is the generic path's (per-layer states + KV rewind by length);
	// models with their own cache export or aux recurrent state keep one snapshot
	const TS_OK = !model.aux_get && !model.cache_export;

	/* GPU completion notifier + pinned result buffers for the async
	 * decode loop (see the async sampling section above). The engine
	 * runs one job at a time, and the wake-up consumes the buffers
	 * before the next arm — no second reader. */
	let note = torch.notifier();
	let waiting = false;      // armed, continuation pending
	let resume = null;
	let sbufs = {
		p: torch.zeros([1000], 'float32').pin(),
		idxs: torch.zeros([1000], 'int64').pin(),
		keep: torch.zeros([1], 'float32').pin(),
		tok: torch.zeros([1], 'int64').pin()
	};

	// distinct model devices: the wake-up is the AND of their streams,
	// which is exactly "the forward and the result copies completed"
	let devset = [];

	if (model.devs)
		for (let d in model.devs)
			if (index(devset, d) < 0)
				push(devset, d);

	// GDN state holders of layer i: the layer's cache entry, or one per
	// device under tensor parallelism (model.states)
	function states(i) {
		return model.states ? model.states(cache, i) : [ cache.layers[i] ];
	}

	function snap_bufs(i) {
		return map(states(i), (cl) => [
			torch.zeros(cl.rec_state.shape(), 'float32', 'cuda:' + cl.dev),
			torch.zeros(cl.conv_state.shape(), 'float16', 'cuda:' + cl.dev)
		]);
	}

	/* copy the GDN states of layer i into buffers b (save) or back */
	function save_states(i, b) {
		let st = states(i);

		for (let n = 0; n < length(st); n++) {
			b[n][0].copy_(st[n].rec_state);
			b[n][1].copy_(st[n].conv_state);
		}
	}

	function load_states(i, b) {
		let st = states(i);

		for (let n = 0; n < length(st); n++) {
			st[n].rec_state.copy_(b[n][0]);
			st[n].conv_state.copy_(b[n][1]);
		}
	}

	function zero_states() {
		for (let k in gdn_layers)
			for (let cl in states(k)) {
				cl.rec_state.fill_(0);
				cl.conv_state.fill_(0);
			}

		// recurrent state outside the layer entries (qwen4: PLE conv tail +
		// n-gram id context)
		if (model.aux_reset)
			model.aux_reset(cache);

		cache.length = 0;
	}

	// GDN state snapshot buffers (allocated once): the prompt-cache
	// snapshot (snap) and the MTP verify rollback (vsnap, below)
	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		if (cfg.layer_types[i] != 'linear_attention')
			continue;

		push(gdn_layers, i);
		snap[i] = snap_bufs(i);
		tsnap[i] = TS_OK ? snap_bufs(i) : null;
	}

	if (TS_OK && length(gdn_layers)) {
		let tb = 0;

		for (let k in gdn_layers)
			for (let b in tsnap[k])
				tb += b[0].numel() * 4 + b[1].numel() * 2;

		warn(sprintf('engine: turn checkpoint buffers: %.1f MiB\n', tb / 1048576.0));
	}

	let ids = [];          // tokens currently processed into the cache
	let snap_ids = null;   // token prefix the snapshot represents
	let tsnap_ids = null;  // ... the turn checkpoint (always a prefix of `ids`, else null)
	let tsnap_hidden = null;

	// CUDA-graph decode steps (opts.graphs != false): sizes 1 (plain
	// decode) and 1..W (MTP verify windows and replays). Captured now,
	// before any prefill — the eager warmup inside clobbers the cache,
	// which reset_state() below clears.
	// DFlash 2 block drafter (qwen35_tp + dflash.uc): replaces the draft CHAIN of the MTP round,
	// everything after it (verify window, GDN rollback, acceptance) is shared
	let DF = model.dflash || null;
	let mtp_planned = ((model.mtp || DF) && opts.mtp_window > 0) ? opts.mtp_window : 0;
	let graph = null;

	// per-position GDN states for MTP verify (W >= 2): the W-token verify
	// forward keeps every GDN layer's recurrent + conv state after each
	// of its first W-1 tokens (qwen35.gdn_forward, cl.spec_*), so a
	// rejection after token a restores them with one batched copy per GPU
	// (rollback[a]) — no GDN snapshot before the verify, no re-run of the
	// accepted tokens after it. Allocated before graph capture: the
	// verify graph writes these buffers.
	let spec = mtp_planned >= 2;

	// graph-replayed prefill tails (graph_runner): up to TAIL_MAX new tokens
	// through the decode / verify step graphs. A replayed row costs about
	// what a decode row does (~9-12 ms on GLM-5.3 TP 3x2: the MoE reads grow
	// with the rows), the eager path ~110-170 ms of launches at 18k tokens
	// (more at depth) plus ~2 ms per token: the graphs win up to ~12 tokens
	// (L135). Not for the drafters with their own prefill taps (DFlash,
	// DSpark) nor for models whose state lives outside the layer caches
	// (they hook the eager prefill)
	const TAIL_MAX = (getenv('AMP_TAIL_GRAPH_MAX') != null) ? +getenv('AMP_TAIL_GRAPH_MAX') : 12;
	let TAIL_OK = TAIL_MAX > 0 && opts.graphs != false && !DF && !model.dspark &&
		!model.aux_get && !model.cache_export;
	let rollback = [];

	if (spec) {
		for (let a = 0; a < mtp_planned - 1; a++)
			push(rollback, {d: [], s: []});

		for (let k in gdn_layers)
			for (let cl in states(k)) {
				let dev = 'cuda:' + cl.dev;

				cl.spec_w = mtp_planned;
				cl.spec_states = torch.zeros([ mtp_planned - 1, ...cl.rec_state.shape() ],
					'float32', dev);
				cl.spec_conv = [];

				for (let a = 0; a < mtp_planned - 1; a++) {
					push(cl.spec_conv, torch.zeros(cl.conv_state.shape(), 'float16', dev));
					push(rollback[a].d, cl.rec_state, cl.conv_state);
					push(rollback[a].s, cl.spec_states.narrow(0, a, 1), cl.spec_conv[a]);
				}
			}

		// model-owned rollback state (qwen4: the PLE conv tail) next to the GDN states
		if (model.spec_bufs) {
			let extra = model.spec_bufs(cache, mtp_planned);

			for (let a = 0; a < length(extra); a++) {
				push(rollback[a].d, ...extra[a].d);
				push(rollback[a].s, ...extra[a].s);
			}
		}
	}

	// DSpark block drafter (DeepSeek-V4, model.dspark): rounds of draft + verify with a
	// verify window of `dsv` drafts; greedy and sampled requests (one-hot draft rule)
	let SP = (model.dspark && opts.dspark_window !== 0) ? model.dspark : null;
	let dsv = SP ? (opts.dspark_window || 4) : 0;

	if (opts.graphs != false) {
		let sizes = [ 1 ];

		if (SP)
			for (let k = 2; k <= dsv + 1; k++)
				push(sizes, k);

		// with per-position states only the verify size replays; the
		// snapshot path re-runs 2..W-1 accepted tokens as well
		if (spec)
			push(sizes, mtp_planned);
		else
			for (let k = 2; k <= mtp_planned; k++)
				push(sizes, k);

		graph = model.graphs ? model.graphs(cache, sizes) :
			qgraph.create(model, cache, sizes);

		if (SP) {
			let wins = [ 0 ];

			for (let k = 1; k <= dsv; k++)
				push(wins, k);

			SP.build(graph, cache, wins);
		}

		if (DF && mtp_planned > 0) {
			if (!graph || !graph.feat || !graph.has(mtp_planned) || DF.W != mtp_planned)
				die(`engine: the DFlash drafter needs the captured verify graph (window ${DF.W})`);

			DF.build(() => graph.feat(mtp_planned));
		}

		zero_states();
	}

	// MTP speculative decoding (opts.mtp_window > 0 and a checkpoint MTP
	// head): the draft cache mirrors the target positions; generation
	// then runs in rounds (draft W tokens, verify them in one target
	// forward, accept by the speculative-sampling rule).
	let M = model.mtp || DF;
	let df_pprev = null;     // DFlash: position of the last verify's first row (its taps are fresh)

	let mtp_w = (M && opts.mtp_window > 0) ? opts.mtp_window : 0;
	let hid_w = model.hidden_width || cfg.hidden_size;     // width of the draft head's hidden input
	let dc = cache.mtp;

	// draft-step entry points: a model with its own MTP block (GLM)
	// provides model.mtp_chunk / mtp_chunk_ids, else the qwen35 block
	function mtp_run(token_ids, hidden, pos, head) {
		return model.mtp_chunk ?
			model.mtp_chunk(token_ids, hidden, dc, pos, head) :
			qwen35.mtp_chunk(cfg, M, model.embed_tokens, model.devs[0], token_ids,
				hidden, dc, pos, head);
	}

	function mtp_run_ids(ids_dev, hidden, pos, head) {
		return model.mtp_chunk_ids ?
			model.mtp_chunk_ids(ids_dev, hidden, dc, pos, head) :
			qwen35.mtp_chunk_ids(cfg, M, model.embed_tokens, model.devs[0], ids_dev,
				hidden, dc, pos, head);
	}
	let draft_len = 0;       // draft cache positions [0, draft_len) valid
	let vsnap = {};

	if (mtp_w && !spec)
		for (let k in gdn_layers)
			vsnap[k] = snap_bufs(k);

	// draft head (opt-in, opts.draft_vocab = N > 0): the lm_head
	// restricted to the first N token ids (BPE ids are roughly frequency-
	// ordered) plus the vocabulary tail with the added/special tokens —
	// the draft only needs an argmax, and the full 248k-column head costs
	// 1.1 ms per draft step. Measured on the 27B: N = 32768 lowers draft
	// acceptance 90% -> 68-73% on code prompts and does not pay (the head
	// is ~6% of a round); off by default (docs/LESSONS.md L66).
	let draft_head = null;

	if (mtp_w && opts.draft_vocab > 0 && model.lm_head.trellis) {
		let V = model.lm_head.out_features;
		let main = opts.draft_vocab;
		let tail = 512;

		if (main % 256 == 0 && main + tail < V && V % 256 == 0)
			draft_head = models.sub_columns(model.lm_head,
				[ [ 0, main ], [ V - tail, tail ] ]);
	}

	// Device-side draft chain state (async greedy rounds): c_dev holds
	// the drafted token ids — slot 0 the round's seed, slots 1..W the
	// chain's argmax results (cross-device copy_ from the draft head's
	// device); rb is the pinned host side of the round: the acceptance
	// count, the verify predictions (the bonus source) and the drafted
	// tokens, all delivered by one copy_async batch per round.
	let c_dev = null;
	let rb = null;

	if (mtp_w) {
		c_dev = torch.zeros([mtp_w + 1], 'int64', 'cuda:' + model.devs[0]);

		rb = {
			a: torch.zeros([1], 'float32').pin(),
			pred: torch.zeros([mtp_w], 'int64').pin(),
			c: torch.zeros([mtp_w + 1], 'int64').pin(),

			// the acceptance count: min over (i for mismatches, W for
			// matches) — this index vector is the i side
			idx: torch.arange(mtp_w, 'float32', 'cuda:' + model.devs[0])
		};
	}

	let last_hidden = null;  // target post-norm hidden at ids[-1]
	let snap_hidden = null;  // ... at the snapshot boundary - 1
	let snap_aux = null;     // model.aux_get() state at the snapshot boundary

	let queue = [];
	let job = null;
	let parked = [], pick_parked = false;
	const SLICE_MS = (opts.slice_ms == null) ? 0 : +opts.slice_ms;
	const SLICE_TOK = (opts.slice_tokens == null) ? 0 : +opts.slice_tokens;
	let timer = null;

	function restore_snapshot() {
		for (let k in gdn_layers)
			load_states(k, snap[k]);

		ids = slice(snap_ids);
		cache.length = length(ids);
		draft_len = length(ids);
		last_hidden = snap_hidden;

		// a checkpoint past the snapshot is no longer a prefix of the cache
		if (tsnap_ids != null && length(tsnap_ids) > length(ids)) {
			tsnap_ids = null;
			tsnap_hidden = null;
		}

		if (DF) {
			DF.reset();
			df_pprev = null;
		}

		if (model.aux_set && snap_aux)
			model.aux_set(cache, snap_aux);
	}

	/* Rewind to the turn checkpoint: GDN states from tsnap, the attention
	 * KV by length. The end-of-prompt snapshot lies past it and is dropped
	 * (its states no longer fit the rewound KV). */
	function restore_turn() {
		for (let k in gdn_layers)
			load_states(k, tsnap[k]);

		ids = slice(tsnap_ids);
		cache.length = length(ids);
		draft_len = length(ids);
		last_hidden = tsnap_hidden;
		snap_ids = null;
		snap_hidden = null;
		snap_aux = null;

		if (DF) {
			DF.reset();
			df_pprev = null;
		}
	}

	function reset_state() {
		zero_states();
		ids = [];
		tsnap_ids = null;
		tsnap_hidden = null;

		// a fresh prefill overwrites the attention KV from position 0:
		// an older snapshot's prefix is no longer in the cache
		snap_ids = null;
		snap_hidden = null;
		draft_len = 0;
		last_hidden = null;

		if (DF) {
			DF.reset();
			df_pprev = null;
		}
	}

	/* Target forward of a few tokens at the cache's current length
	 * through the captured graphs when one fits; else eager. Returns
	 * logits (1, S, V) and leaves the post-norm hidden in cache.hidden
	 * (graph buffers: copy what must outlive the next step). */
	function step_fwd(token_ids) {
		if (graph && graph.has(length(token_ids)))
			return graph.run(token_ids, cache.length).logits;

		return model.forward_chunk(token_ids, cache);
	}

	/* --- host-RAM prompt-cache store ----------------------------------
	 * The engine holds ONE live state (+ one snapshot). When a request
	 * does not continue it (another conversation, a side request of the
	 * client, a cancelled turn), the live state is moved to page-locked
	 * host RAM instead of being thrown away, and a later request that
	 * extends a stored state swaps it back (a 100k-token prefix is
	 * ~6.5 GB: ~1 s over PCIe against ~100 s of prefill). An entry keeps
	 * the token ids, the attention KV (chunks of KV_CHUNK tokens), the GDN
	 * states, the MTP draft cache and hidden, and the snapshot taken at
	 * the last stable prefix, so both reuse points of the old state
	 * survive. opts.cache_ram (GiB, 0: off) bounds the total; LRU. */
	const KV_CHUNK = 4096;
	const STORE_MIN_TOKENS = 2048;    // smaller states are cheaper to recompute
	const STORE_MIN_GAIN = 512;       // a stored hit must save at least this much
	let store = [];                   // oldest first
	let store_bytes = 0;
	let store_budget = int((opts.cache_ram || 0) * 1073741824);

	// device-resident parked slots in front of the host store (opts.slots: 'auto' = as many as fit
	// after everything else is allocated | N | 0; opts.slot_margin: MiB kept free on every GPU).
	// An entry in a slot swaps back device to device (milliseconds, no PCIe).
	let slots_free = [], slots_total = 0;
	let slots_cfg = (opts.slots == null) ? 'auto' : opts.slots;

	function elsize(t) {
		let d = t.dtype();

		return (d == 'float32' || d == 'int32') ? 4 : (d == 'int64' || d == 'float64') ? 8 :
			(d == 'int8' || d == 'uint8' || d == 'bool') ? 1 : 2;
	}

	function is_gdn(i) {
		return cfg.layer_types[i] == 'linear_attention';
	}

	function tbytes(t) {
		return t.numel() * elsize(t);
	}

	// every layer holds attention KV, an MLA latent cache (GLM) or GDN states:
	// else the store stays off
	let mla_store = false;     // MLA latent caches: generic device slots are available

	// a QSA attention cache (Qwen3.8-Flash-Next): K / V rows (max_seq, kvh, hd), raw indexer keys and the pooled blocks
	function is_qsa(c) {
		return c.raw_k != null && c.k_cache != null;
	}

	function is_mla(c) {
		return c.ckv != null || c.qk != null || is_qsa(c);
	}

	// the holder of a layer's attention cache among its per-rank entries (a replica under TP; the owner rank only
	// when the model keeps the cache on one rank), else null
	function kv_holder(sts) {
		for (let c in sts)
			if (is_mla(c))
				return c;

		return null;
	}

	// the draft cache holders when the MTP block has an MLA / QSA cache (replicas under TP, or the owner rank of a
	// row of per-rank entries), else null
	function mtp_mla() {
		let l = cache.mtp ? cache.mtp.layers : null;

		if (l && length(l) && type(l[0]) == 'array')
			l = l[0];

		if (!l || !length(l))
			return null;

		l = filter(l, (x) => is_mla(x));

		return length(l) ? l : null;
	}

	function store_supported() {
		let kv = false;

		// the model exports / imports its whole cache itself (DeepSeek-V4: compressed
		// pools + compressor / window aux state + drafter ring)
		if (model.cache_export)
			return !!(model.cache_import && model.cache_bytes);

		// recurrent state outside the layer entries (qwen4's PLE layer): aux_get / aux_set tensors and plain
		// values, stored with the entry (aux_to_host)
		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			if (is_gdn(i)) {
				for (let cl in sts)
					if (!cl.rec_state)
						return false;

				continue;
			}

			// the attention cache of a layer: on every rank (dense TP) or on one owner rank (QSA)
			let h = kv_holder(sts);

			if (h) {
				kv = true;
				mla_store = true;
				continue;
			}

			for (let cl in sts)
				if (cl.k_cache)
					kv = true;

			for (let cl in sts)
				if (!cl.k_cache)
					return false;
		}

		return kv && (!cache.mtp || cache.mtp.k_cache != null || mtp_mla() != null);
	}

	if ((store_budget > 0 || slots_cfg != 0) && !store_supported()) {
		if (store_budget > 0)
			warn('engine: host-RAM cache store not supported by this model, disabled\n');

		store_budget = 0;
		slots_cfg = 0;
	}

	function to_host(src, e) {
		if (src.numel() == 0)
			return src.cpu();

		let h = torch.pinned_empty(src.shape(), src.dtype());

		h.copy_(src);
		e.bytes += tbytes(h);

		return h;
	}

	// a model's aux state (tensors, numbers, id lists) to host memory
	function aux_to_host(v, e) {
		let t = type(v);

		if (t == 'array')
			return map(v, (x) => aux_to_host(x, e));

		if (t == 'object') {
			let o = {};

			for (let k in v)
				o[k] = aux_to_host(v[k], e);

			return o;
		}

		return (t == 'resource') ? to_host(v, e) : v;
	}

	/* Tiered precision of a stored KV region (a stored state is idle: the
	 * far past matters least to the next token). Chunk by chunk, by the
	 * distance of its end from the end of the state: the last `tail` tokens
	 * stay fp16, up to `q8` tokens back they are int8 (groups of 32,
	 * error ~0.7%), further back int4 (~11%); the first chunk (attention
	 * sinks) is never below int8, and a state shorter than `min` tokens is
	 * stored exactly (cheap to hold, cheap to lose). opts.cache_quant:
	 * 'off' | {min, tail, q8} (tokens). Returns 16, 8 or 4. */
	let qcfg = {min: 16384, tail: 8192, q8: 40960};

	if (opts.cache_quant == 'off')
		qcfg = null;
	else if (type(opts.cache_quant) == 'object')
		qcfg = {...qcfg, ...opts.cache_quant};

	function tier_of(o, m, n) {
		if (!qcfg || n < qcfg.min)
			return 16;

		let d = n - (o + m);
		let t = (d < qcfg.tail) ? 16 : (d < qcfg.q8) ? 8 : 4;

		return (o == 0 && t == 4) ? 8 : t;
	}

	// bytes of one KV chunk (k + v) of `m` tokens at tier t
	function chunk_bytes(shape, m, t) {
		let vals = 2 * shape[1] * m * shape[3];

		return (t == 16) ? vals * 2 : (t == 8) ? vals + vals / 16 : vals / 2 + vals / 16;
	}

	function kv_to_host(c, dev, n, e) {
		let chunks = [];
		let sh = c.k_cache.shape();
		let kvh = sh[1], hd = sh[3];

		for (let o = 0; o < n; o += KV_CHUNK) {
			let m = min(KV_CHUNK, n - o);
			let t = tier_of(o, m, n);
			let ch = {tier: t, m: m, x: []};

			for (let src in [ c.k_cache, c.v_cache ]) {
				let v = src.narrow(2, o, m);

				if (t == 16) {
					push(ch.x, to_host(v, e));
					continue;
				}

				let rows = v.contiguous().view([ kvh * m, hd ]);
				let q = torch.empty([ kvh * m, (t == 8) ? hd : hd / 2 ], 'int8', 'cuda:' + dev);
				let sc = torch.empty([ kvh * m, hd / 32 ], 'float16', 'cuda:' + dev);

				if (t == 8)
					torch.kv_q8_put(rows, q, sc, 0);
				else
					torch.kv_q4_put(rows, q, sc, 0);

				push(ch.x, [ to_host(q, e), to_host(sc, e) ]);
			}

			push(chunks, ch);
		}

		return chunks;
	}

	function kv_from_host(c, dev, chunks, n) {
		let sh = c.k_cache.shape();
		let kvh = sh[1], hd = sh[3];

		for (let k = 0; k * KV_CHUNK < n; k++) {
			let o = k * KV_CHUNK;
			let m = min(KV_CHUNK, n - o);
			let ch = chunks[k];
			let dst = [ c.k_cache, c.v_cache ];

			for (let x = 0; x < 2; x++) {
				let src = ch.x[x];

				if (ch.tier != 16) {
					let q = src[0].cuda(dev), sc = src[1].cuda(dev);

					src = (ch.tier == 8 ? torch.kv_q8_get(q, sc, 0, kvh * ch.m) :
						torch.kv_q4_get(q, sc, 0, kvh * ch.m)).view([ 1, kvh, ch.m, hd ]);
				}

				if (m < ch.m)
					src = src.narrow(2, 0, m);

				dst[x].narrow(2, o, m).copy_(src);
			}
		}
	}

	/* MLA latent cache (GLM): the ckv rows (fp16, tiered like KV; the Q8 form is kept as is), the
	 * indexer plane (per token) and the complete pooled keys, both always fp16 (they feed every
	 * selection). Under tensor parallelism the replicas of a layer are identical: one is stored. */
	function mla_pools(n) {
		return int(n / cfg.index_kpool);
	}

	function qsa_pools(n) {
		return int(n / cfg.indexer_compress_ratio) + 1;
	}

	function qsa_bytes(c, n) {
		let sh = c.k_cache.shape();

		return n * (2 * sh[1] * sh[2] + c.raw_k.shape()[1]) * 2 + qsa_pools(n) * c.pooled.shape()[1] * 2;
	}

	// the rows [0, n) of a QSA cache (K, V, raw indexer keys) and its pooled blocks, exact fp16
	function qsa_to_host(c, dev, n, e) {
		return {k: to_host(c.k_cache.narrow(0, 0, n), e), v: to_host(c.v_cache.narrow(0, 0, n), e),
			raw: to_host(c.raw_k.narrow(0, 0, n), e),
			pooled: to_host(c.pooled.narrow(0, 0, min(qsa_pools(n), c.pooled.shape()[0])), e)};
	}

	function qsa_from_host(c, dev, d, n) {
		c.k_cache.narrow(0, 0, n).copy_(d.k.narrow(0, 0, n));
		c.v_cache.narrow(0, 0, n).copy_(d.v.narrow(0, 0, n));
		c.raw_k.narrow(0, 0, n).copy_(d.raw.narrow(0, 0, n));

		let p = min(qsa_pools(n), d.pooled.shape()[0]);

		c.pooled.narrow(0, 0, p).copy_(d.pooled.narrow(0, 0, p));
	}

	// rows [0, n) of a QSA cache to another (slot <-> live, any device)
	function qsa_copy(d, s, n) {
		for (let k in [ 'k_cache', 'v_cache', 'raw_k' ])
			d[k].narrow(0, 0, n).copy_(s[k].narrow(0, 0, n).cuda(d.dev));

		let p = min(qsa_pools(n), s.pooled.shape()[0]);

		d.pooled.narrow(0, 0, p).copy_(s.pooled.narrow(0, 0, p).cuda(d.dev));
	}

	function qsa_alloc(c) {
		let o = {dev: c.dev};

		for (let k in [ 'k_cache', 'v_cache', 'raw_k', 'pooled' ])
			o[k] = torch.zeros(c[k].shape(), c[k].dtype(), 'cuda:' + c.dev);

		return o;
	}

	function mla_bytes(c, n) {
		if (is_qsa(c))
			return qsa_bytes(c, n);

		let dc = (c.ckv ? c.ckv : c.qk).shape()[1];
		let b = n * c.plane.shape()[1] * 2 + mla_pools(n) * c.pools.shape()[1] * 2;

		for (let o = 0; o < n; o += KV_CHUNK) {
			let m = min(KV_CHUNK, n - o);
			let t = c.qk ? 8 : tier_of(o, m, n);
			let vals = m * dc;

			b += (t == 16) ? vals * 2 : (t == 8) ? vals + vals / 16 : vals / 2 + vals / 16;
		}

		return b;
	}

	function mla_to_host(c, dev, n, e) {
		if (is_qsa(c))
			return qsa_to_host(c, dev, n, e);

		let chunks = [];
		let dc = (c.ckv ? c.ckv : c.qk).shape()[1];

		for (let o = 0; o < n; o += KV_CHUNK) {
			let m = min(KV_CHUNK, n - o);

			if (c.qk) {
				push(chunks, {tier: 8, m: m, x: [ to_host(c.qk.narrow(0, o, m), e), to_host(c.qs.narrow(0, o, m), e) ]});
				continue;
			}

			let t = tier_of(o, m, n);
			let v = c.ckv.narrow(0, o, m);

			if (t == 16) {
				push(chunks, {tier: 16, m: m, x: [ to_host(v, e) ]});
				continue;
			}

			let q = torch.empty([ m, (t == 8) ? dc : dc / 2 ], 'int8', 'cuda:' + dev);
			let sc = torch.empty([ m, dc / 32 ], 'float16', 'cuda:' + dev);

			if (t == 8)
				torch.kv_q8_put(v, q, sc, 0);
			else
				torch.kv_q4_put(v, q, sc, 0);

			push(chunks, {tier: t, m: m, x: [ to_host(q, e), to_host(sc, e) ]});
		}

		return {ch: chunks, plane: to_host(c.plane.narrow(0, 0, n), e),
			pools: to_host(c.pools.narrow(0, 0, mla_pools(n)), e)};
	}

	function mla_from_host(c, dev, d, n) {
		if (d.k != null) {
			qsa_from_host(c, dev, d, n);

			return;
		}

		for (let k = 0; k * KV_CHUNK < n; k++) {
			let o = k * KV_CHUNK;
			let m = min(KV_CHUNK, n - o);
			let ch = d.ch[k];

			if (c.qk) {
				c.qk.narrow(0, o, m).copy_(ch.x[0].narrow(0, 0, m));
				c.qs.narrow(0, o, m).copy_(ch.x[1].narrow(0, 0, m));
				continue;
			}

			let src = ch.x[0];

			if (ch.tier != 16) {
				let q = ch.x[0].cuda(dev), sc = ch.x[1].cuda(dev);

				src = (ch.tier == 8) ? torch.kv_q8_get(q, sc, 0, ch.m) : torch.kv_q4_get(q, sc, 0, ch.m);
			}

			if (m < ch.m)
				src = src.narrow(0, 0, m);

			c.ckv.narrow(0, o, m).copy_(src);
		}

		c.plane.narrow(0, 0, n).copy_(d.plane.narrow(0, 0, n));

		if (mla_pools(n) > 0)
			c.pools.narrow(0, 0, mla_pools(n)).copy_(d.pools.narrow(0, 0, mla_pools(n)));
	}

	// rows [0, n) of an MLA cache object to another (slot <-> live, any device)
	function mla_copy(d, s, n) {
		if (is_qsa(s)) {
			qsa_copy(d, s, n);

			return;
		}

		for (let k in [ 'ckv', 'qk', 'qs', 'plane' ])
			if (s[k] != null)
				d[k].narrow(0, 0, n).copy_(s[k].narrow(0, 0, n).cuda(d.dev));

		if (mla_pools(n) > 0)
			d.pools.narrow(0, 0, mla_pools(n)).copy_(s.pools.narrow(0, 0, mla_pools(n)).cuda(d.dev));
	}

	function mla_alloc(c) {
		if (is_qsa(c))
			return qsa_alloc(c);

		let o = {dev: c.dev};

		for (let k in [ 'ckv', 'qk', 'qs', 'plane', 'pools' ])
			if (c[k] != null)
				o[k] = torch.zeros(c[k].shape(), c[k].dtype(), 'cuda:' + c.dev);

		return o;
	}

	/* A resident slot of the generic (MLA) store path: device copies, at the live cache's capacity,
	 * of the latent cache rows (one replica per layer), the KDA states, the snapshot buffers and the
	 * draft cache. An entry in a slot is exact (no age tiers) and swaps back device to device. */
	function new_slot_generic() {
		let sl = {kv: {}, gdn: {}, snap: {}, tsnap: {}, draft: null};

		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			if (is_gdn(i))
				sl.gdn[i] = map(sts, (cl) => [
					torch.zeros(cl.rec_state.shape(), 'float32', 'cuda:' + cl.dev),
					torch.zeros(cl.conv_state.shape(), 'float16', 'cuda:' + cl.dev)
				]);
			else
				sl.kv[i] = mla_alloc(kv_holder(sts));
		}

		for (let k in gdn_layers) {
			sl.snap[k] = snap_bufs(k);
			sl.tsnap[k] = TS_OK ? snap_bufs(k) : null;
		}

		if (mtp_mla())
			sl.draft = mla_alloc(mtp_mla()[0]);

		return sl;
	}

	// host bytes a slot entry takes when demoted (at the tiers of its length)
	function slot_host_bytes(e) {
		let b = 0;

		for (let i in keys(e.slot.kv))
			b += mla_bytes(e.slot.kv[i], e.n);

		for (let i in keys(e.slot.gdn))
			for (let p in e.slot.gdn[i])
				b += tbytes(p[0]) + tbytes(p[1]);

		if (e.snap)
			for (let k in keys(e.slot.snap))
				for (let p in e.slot.snap[k])
					b += tbytes(p[0]) + tbytes(p[1]);

		if (e.tsnap)
			for (let k in keys(e.slot.tsnap))
				for (let p in e.slot.tsnap[k])
					b += tbytes(p[0]) + tbytes(p[1]);

		if (e.draft && e.slot.draft)
			b += mla_bytes(e.slot.draft, e.dl);

		return b;
	}

	function store_estimate() {
		let n = length(ids), b = 0;

		if (model.cache_bytes)
			return model.cache_bytes(cache, n);

		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			if (!is_gdn(i) && kv_holder(sts)) {
				b += mla_bytes(kv_holder(sts), n);
				continue;
			}

			for (let cl in sts) {
				if (!is_gdn(i)) {
					let sh = cl.k_cache.shape();

					for (let o = 0; o < n; o += KV_CHUNK) {
						let m = min(KV_CHUNK, n - o);

						b += chunk_bytes(sh, m, tier_of(o, m, n));
					}
				}
				else
					b += 2 * (tbytes(cl.rec_state) + tbytes(cl.conv_state));
			}
		}

		if (mtp_mla() && draft_len > 0)
			b += mla_bytes(mtp_mla()[0], draft_len);
		else if (cache.mtp && draft_len > 0) {
			let sh = cache.mtp.k_cache.shape();

			for (let o = 0; o < draft_len; o += KV_CHUNK) {
				let m = min(KV_CHUNK, draft_len - o);

				b += chunk_bytes(sh, m, tier_of(o, m, draft_len));
			}
		}

		return b;
	}

	function store_drop(k) {
		store_bytes -= store[k].bytes;

		if (store[k].slot)
			push(slots_free, store[k].slot);

		splice(store, k, 1);
	}

	// index of the oldest entry in the given tier (device: has a slot), -1 when none
	function oldest(dev) {
		for (let k = 0; k < length(store); k++)
			if (!!store[k].slot == dev && (dev || !store[k].owner))     // (a parked job's host entry is not evictable)
				return k;

		return -1;
	}

	// a model-specific entry tree (arrays / objects / tensors / numbers) copied to pinned host memory
	function host_tree(v, e) {
		if (v == null)
			return null;

		let t = type(v);

		if (t == 'array')
			return map(v, (x) => host_tree(x, e));

		if (t == 'object') {
			let o = {};

			for (let k in v)
				o[k] = host_tree(v[k], e);

			return o;
		}

		return (t == 'resource') ? to_host(v, e) : v;
	}

	function tree_bytes(v) {
		if (v == null)
			return 0;

		let t = type(v), b = 0;

		if (t == 'array') {
			for (let x in v)
				b += tree_bytes(x);
		}
		else if (t == 'object') {
			for (let k in v)
				b += tree_bytes(v[k]);
		}
		else if (t == 'resource')
			b = tbytes(v);

		return b;
	}

	/* Move a device-resident entry to host RAM (frees its slot), or drop it when the host
	 * budget cannot take it. */
	function demote(k) {
		let e = store[k];
		let b = e.model ? tree_bytes(e.model) : slot_host_bytes(e);

		if (store_budget <= 0 || b > store_budget) {
			store_drop(k);

			return;
		}

		store_bytes -= e.bytes;
		splice(store, k, 1);

		// make room among the host entries first
		while (store_bytes + b > store_budget) {
			let o = oldest(false);

			if (o < 0)
				break;

			store_drop(o);
		}

		let h = {bytes: e.model ? 0 : e.bytes};

		try {
			if (e.model)
				e.model = host_tree(e.model, h);
			else {
				// generic entry: the slot's rows into the host layout (age tiers apply)
				let sl = e.slot;

				e.kv = {};
				e.gdn = {};

				for (let i in keys(sl.kv))
					e.kv[i] = [ mla_to_host(sl.kv[i], sl.kv[i].dev, e.n, h) ];

				for (let i in keys(sl.gdn))
					e.gdn[i] = map(sl.gdn[i], (p) => [ to_host(p[0], h), to_host(p[1], h) ]);

				if (e.snap) {
					e.snap = {};

					for (let q in keys(sl.snap))
						e.snap[q] = map(sl.snap[q], (p) => [ to_host(p[0], h), to_host(p[1], h) ]);
				}

				if (e.tsnap) {
					e.tsnap = {};

					for (let q in keys(sl.tsnap))
						e.tsnap[q] = map(sl.tsnap[q], (p) => [ to_host(p[0], h), to_host(p[1], h) ]);
				}

				if (e.draft && sl.draft)
					e.draft = mla_to_host(sl.draft, sl.draft.dev, e.dl, h);
				else
					e.draft = null;
			}

			e.bytes = h.bytes;
			store_bytes += h.bytes;
			push(slots_free, e.slot);
			e.slot = null;
			unshift(store, e);                     // oldest
		}
		catch (err) {
			warn(`engine: cache store: demotion failed (${err.message}), entry dropped\n`);
			push(slots_free, e.slot);
		}
	}

	/* Move the live state (and its snapshot) to host RAM. Returns true when
	 * stored; any failure leaves the store as it was. */
	/* owner: the job whose live state is being parked (time slicing): any length, the entry is
	 * linked to the job (owner.park) and invisible to prefix lookups; else the plain store. */
	function store_put(owner) {
		let n = length(ids);
		let est = store_estimate();

		if ((!owner && n < STORE_MIN_TOKENS) || cache.length != n)
			return false;

		if (slots_total <= 0 && (store_budget <= 0 || est > store_budget))
			return false;

		// a stored state that is a prefix of this one is superseded
		for (let k = length(store) - 1; k >= 0; k--)
			if (!store[k].owner && lcp(store[k].ids, ids) == store[k].n)
				store_drop(k);

		// a free device slot (else demote the oldest device entry), then the host store
		let slot = null;

		if (slots_total > 0) {
			if (!length(slots_free) && oldest(true) >= 0)
				demote(oldest(true));

			if (length(slots_free))
				slot = pop(slots_free);
		}

		if (!slot) {
			if (store_budget <= 0 || est > store_budget)
				return false;

			while (store_bytes + est > store_budget && oldest(false) >= 0)
				store_drop(oldest(false));
		}

		let e = {ids: slice(ids), n: n, bytes: 0, kv: {}, gdn: {}, dl: draft_len,
			hid: null, snap_ids: null, snap: null, snap_hid: null, draft: null, slot: null,
			tsnap_ids: null, tsnap: null, tsnap_hid: null};
		let t0 = now_ms();

		try {
			torch.synchronize();

			if (model.cache_export) {
				// the model's own layout; the snapshot's aux state rides along when the
				// snapshot is a prefix of the live state
				let sa = (snap_ids != null && snap_aux != null && lcp(snap_ids, ids) == length(snap_ids)) ?
					snap_aux : null;

				e.model = slot ? model.cache_export(cache, n, (t) => t.add(0.0), sa, slot) :
					model.cache_export(cache, n, (t) => to_host(t, e), sa);

				if (slot) {
					torch.synchronize();
					e.slot = slot;
				}

				if (sa)
					e.snap_ids = slice(snap_ids);

				if (owner) {
					e.owner = owner;
					owner.park = e;
				}

				push(store, e);
				store_bytes += e.bytes;
				warn(sprintf('engine: cache store: %d tokens moved to %s in %.0f ms (%d entries, %d/%d slots free, %.2f GiB host)\n',
					n, slot ? 'a device slot' : 'host RAM', now_ms() - t0, length(store), length(slots_free),
					slots_total, store_bytes / 1073741824.0));

				return true;
			}

			for (let i = 0; i < cfg.num_hidden_layers; i++) {
				let sts = states(i);

				if (slot) {
					if (is_gdn(i))
						for (let r = 0; r < length(sts); r++) {
							slot.gdn[i][r][0].copy_(sts[r].rec_state);
							slot.gdn[i][r][1].copy_(sts[r].conv_state);
						}
					else
						mla_copy(slot.kv[i], kv_holder(sts), n);
				}
				else if (!is_gdn(i) && kv_holder(sts))
					e.kv[i] = [ mla_to_host(kv_holder(sts), kv_holder(sts).dev, n, e) ];
				else
					e[is_gdn(i) ? 'gdn' : 'kv'][i] = map(sts, (cl) => !is_gdn(i) ?
						kv_to_host(cl, cl.dev, n, e) : [ to_host(cl.rec_state, e), to_host(cl.conv_state, e) ]);
			}

			if (mtp_mla() && draft_len > 0) {
				if (slot)
					mla_copy(slot.draft, mtp_mla()[0], draft_len);
				else
					e.draft = mla_to_host(mtp_mla()[0], mtp_mla()[0].dev, draft_len, e);

				if (slot)
					e.draft = true;
			}
			else if (cache.mtp && draft_len > 0)
				e.draft = kv_to_host(cache.mtp, M.dev, draft_len, e);

			if (last_hidden)
				e.hid = to_host(last_hidden, e);

			if (model.aux_get)
				e.aux = aux_to_host(model.aux_get(cache), e);

			if (snap_ids != null && lcp(snap_ids, ids) == length(snap_ids)) {
				e.snap_ids = slice(snap_ids);
				e.snap = {};

				if (model.aux_get && snap_aux)
					e.snap_aux = aux_to_host(snap_aux, e);

				for (let k in gdn_layers) {
					if (slot) {
						for (let r = 0; r < length(snap[k]); r++) {
							slot.snap[k][r][0].copy_(snap[k][r][0]);
							slot.snap[k][r][1].copy_(snap[k][r][1]);
						}
					}
					else
						e.snap[k] = map(snap[k], (b) => [ to_host(b[0], e), to_host(b[1], e) ]);
				}

				if (snap_hidden)
					e.snap_hid = to_host(snap_hidden, e);
			}

			// the turn checkpoint rides along (a prefix of the live state)
			if (TS_OK && tsnap_ids != null && lcp(tsnap_ids, ids) == length(tsnap_ids)) {
				e.tsnap_ids = slice(tsnap_ids);
				e.tsnap = {};

				for (let k in gdn_layers) {
					if (slot) {
						for (let r = 0; r < length(tsnap[k]); r++) {
							slot.tsnap[k][r][0].copy_(tsnap[k][r][0]);
							slot.tsnap[k][r][1].copy_(tsnap[k][r][1]);
						}
					}
					else
						e.tsnap[k] = map(tsnap[k], (b) => [ to_host(b[0], e), to_host(b[1], e) ]);
				}

				if (tsnap_hidden)
					e.tsnap_hid = to_host(tsnap_hidden, e);
			}

			if (slot) {
				torch.synchronize();
				e.slot = slot;
			}
		}
		catch (err) {
			warn(`engine: cache store: offload failed (${err.message})\n`);

			if (slot)
				push(slots_free, slot);

			return false;
		}

		if (owner) {
			e.owner = owner;
			owner.park = e;
		}

		push(store, e);
		store_bytes += e.bytes;
		warn(sprintf('engine: cache store: %d tokens moved to %s in %.0f ms (%d entries, %.1f GiB host)\n',
			n, slot ? 'a device slot' : 'host RAM', now_ms() - t0, length(store), store_bytes / 1073741824.0));

		return true;
	}

	/* Swap stored entry e back into the live buffers up to `start` tokens:
	 * start == e.n restores the live state, else the entry's snapshot
	 * (start == length(e.snap_ids)). e must already be out of the list. */
	function store_restore(e, start) {
		let t0 = now_ms();
		let live = (start == e.n);

		if (e.model) {
			let r = model.cache_import(cache, e.model, start, live);

			if (e.snap_ids) {
				snap_ids = slice(e.snap_ids);
				snap_aux = r.snap;
			}
			else {
				snap_ids = null;
				snap_aux = null;
			}

			snap_hidden = null;
			last_hidden = null;
			tsnap_ids = null;
			tsnap_hidden = null;

			if (live) {
				ids = slice(e.ids);
				cache.length = e.n;
				draft_len = 0;
			}
			else
				restore_snapshot();

			torch.synchronize();

			let from_slot = !!e.slot;

			if (e.slot) {
				push(slots_free, e.slot);
				e.slot = null;
			}

			warn(sprintf('engine: cache store: %d tokens restored from %s in %.0f ms\n',
				start, from_slot ? 'a device slot' : 'host RAM', now_ms() - t0));

			return;
		}

		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			for (let r = 0; r < length(sts); r++) {
				if (!is_gdn(i) && is_mla(sts[r])) {
					if (e.slot)
						mla_copy(sts[r], e.slot.kv[i], start);
					else
						mla_from_host(sts[r], sts[r].dev, e.kv[i][0], start);
				}
				else if (!is_gdn(i) && !sts[r].k_cache) {
					// a rank without the layer's cache (QSA: the owner rank holds it)
				}
				else if (!is_gdn(i))
					kv_from_host(sts[r], sts[r].dev, e.kv[i][r], start);
				else if (live) {
					let g = e.slot ? e.slot.gdn[i][r] : e.gdn[i][r];

					sts[r].rec_state.copy_(g[0]);
					sts[r].conv_state.copy_(g[1]);
				}
			}
		}

		// restoring to the entry's turn checkpoint (when the snapshot lies past it)
		let turn = !live && e.tsnap_ids != null && start == length(e.tsnap_ids) &&
			!(e.snap_ids != null && start == length(e.snap_ids));

		if (e.tsnap) {
			let ts = e.slot ? e.slot.tsnap : e.tsnap;

			for (let k in gdn_layers)
				for (let r = 0; r < length(tsnap[k]); r++) {
					tsnap[k][r][0].copy_(ts[k][r][0]);
					tsnap[k][r][1].copy_(ts[k][r][1]);
				}

			tsnap_ids = slice(e.tsnap_ids);
			tsnap_hidden = e.tsnap_hid ? e.tsnap_hid.cuda(M.dev) : null;
		}
		else {
			tsnap_ids = null;
			tsnap_hidden = null;
		}

		if (e.snap) {
			let es = e.slot ? e.slot.snap : e.snap;

			for (let k in gdn_layers)
				for (let r = 0; r < length(snap[k]); r++) {
					snap[k][r][0].copy_(es[k][r][0]);
					snap[k][r][1].copy_(es[k][r][1]);
				}

			snap_ids = slice(e.snap_ids);
			snap_hidden = e.snap_hid ? e.snap_hid.cuda(M.dev) : null;
			snap_aux = e.snap_aux || null;
		}
		else {
			snap_ids = null;
			snap_hidden = null;
			snap_aux = null;
		}

		let dl = min(e.dl, start);

		if (mtp_mla() && e.draft && dl > 0) {
			for (let h in mtp_mla()) {
				if (e.slot)
					mla_copy(h, e.slot.draft, dl);
				else
					mla_from_host(h, h.dev, e.draft, dl);
			}
		}
		else if (cache.mtp && e.draft && dl > 0)
			kv_from_host(cache.mtp, M.dev, e.draft, dl);

		if (live) {
			ids = slice(e.ids);
			cache.length = e.n;
			last_hidden = e.hid ? e.hid.cuda(M.dev) : null;

			if (model.aux_set && e.aux)
				model.aux_set(cache, e.aux);
		}
		else if (turn)
			restore_turn();
		else
			restore_snapshot();

		// the draft cache only follows when the entry had it up to `start`
		draft_len = (cache.mtp && dl == start && (!live || e.dl == start)) ? start : 0;

		if (!draft_len)
			last_hidden = null;

		torch.synchronize();

		let from_slot = !!e.slot;

		if (e.slot) {
			push(slots_free, e.slot);
			e.slot = null;
		}

		warn(sprintf('engine: cache store: %d tokens restored from %s in %.0f ms\n',
			start, from_slot ? 'a device slot' : 'host RAM', now_ms() - t0));
	}

	/* Decide where the job's prompt starts: live state, snapshot, a stored
	 * state or 0. Returns {from, start, detail}. */
	function plan_reuse(prompt) {
		let live = lcp(ids, prompt);
		let snl = (snap_ids != null) ? lcp(snap_ids, prompt) : 0;
		let det = {ids: length(ids), live_lcp: live, snap: snap_ids ? length(snap_ids) : 0, snap_lcp: snl};
		let plan;

		// live continuation needs at least one new token to get logits;
		// an empty `ids` means unknown state (startup, failure, cancel):
		// always start fresh then
		if (length(ids) > 0 && live == length(ids) && live < length(prompt))
			return {from: 'live', start: live, detail: det};

		if (snap_ids != null && length(snap_ids) < length(prompt) && snl == length(snap_ids))
			plan = {from: 'snapshot', start: length(snap_ids), detail: det};
		else
			plan = {from: 'fresh', start: 0, detail: det};

		// the turn checkpoint (end of the last user message): when the template
		// re-renders the assistant / tool turns after it differently (reasoning
		// dropped once a new user message arrives), the divergence point lies
		// between this checkpoint and the end-of-prompt snapshot
		if (tsnap_ids != null) {
			let tl = lcp(tsnap_ids, prompt);

			det.turn = length(tsnap_ids);
			det.turn_lcp = tl;

			if (tl == length(tsnap_ids) && length(tsnap_ids) < length(prompt) &&
			    length(tsnap_ids) > plan.start)
				plan = {from: 'turn', start: length(tsnap_ids), detail: det};
		}

		if (live < length(ids) && live > 0) {
			det.div_old = slice(ids, max(live - 4, 0), live + 8);
			det.div_new = slice(prompt, max(live - 4, 0), live + 8);
		}

		if (store_budget <= 0 && slots_total <= 0)
			return plan;

		// best stored state that the prompt extends
		let hit = null, hit_start = 0, hit_k = -1;

		for (let k = 0; k < length(store); k++) {
			let e = store[k];

			if (e.owner)
				continue;            // a parked job's own state

			let l = lcp(e.ids, prompt);

			if (l == e.n && e.n < length(prompt) && e.n > hit_start) {
				hit = e; hit_start = e.n; hit_k = k;
			}
			else if (e.snap_ids && length(e.snap_ids) < length(prompt) &&
			         length(e.snap_ids) > hit_start &&
			         lcp(e.snap_ids, prompt) == length(e.snap_ids)) {
				hit = e; hit_start = length(e.snap_ids); hit_k = k;
			}
			else if (e.tsnap_ids && length(e.tsnap_ids) < length(prompt) &&
			         length(e.tsnap_ids) > hit_start &&
			         lcp(e.tsnap_ids, prompt) == length(e.tsnap_ids)) {
				hit = e; hit_start = length(e.tsnap_ids); hit_k = k;
			}
		}

		if (hit && hit_start < plan.start + STORE_MIN_GAIN)
			hit = null;

		// the live state is about to be (mostly) dropped — what it still
		// contributes is plan.start, whatever a stored hit adds on top —
		// keep it in RAM
		if (length(ids) >= STORE_MIN_TOKENS && plan.start < length(ids) / 2) {
			if (hit)
				splice(store, hit_k, 1);       // (out of the way of the LRU eviction)

			if (hit)
				store_bytes -= hit.bytes;

			det.stored = store_put();
		}
		else if (hit) {
			splice(store, hit_k, 1);
			store_bytes -= hit.bytes;
		}

		if (!hit)
			return plan;

		det.ram_from = hit_start;
		det.ram_tier = hit.slot ? 'device' : 'host';

		let tr = now_ms();

		store_restore(hit, hit_start);
		det.ram_ms = now_ms() - tr;

		return {from: 'ram', start: hit_start, detail: det};
	}

	function finish(reason) {
		let j = job;

		job = null;

		if (MTP_TIMING && j.tm && j.rounds) {
			let o = {};

			for (let k in j.tm)
				o[k] = (k == 'replays') ? j.tm[k] : sprintf('%.2f', j.tm[k] / j.rounds);

			warn(`mtp timing (ms/round over ${j.rounds} rounds): ${o}\n`);
		}

		let now = now_ms();

		j.on_done({
			reason: reason,
			prompt_tokens: length(j.ids),
			cached_tokens: j.reuse_start,
			reuse: j.reuse_from,
			reuse_detail: j.reuse_detail,
			completion_tokens: j.generated,
			preemptions: j.preempted || 0,
			mtp: j.rounds ? {rounds: j.rounds, drafted: j.drafted,
				accepted: j.accepted} : null,
			timings: {
				prefill_ms: j.t_prefill_done - j.t_start,
				prefill_tokens: length(j.ids) - j.reuse_start,
				decode_ms: now - j.t_prefill_done,
				decode_tokens: j.generated
			}
		});
	}

	function fail(msg) {
		let j = job;

		job = null;

		// the cache contents are undefined after a failed step
		ids = [];
		tsnap_ids = null;
		tsnap_hidden = null;
		snap_ids = null;
		snap_hidden = null;
		draft_len = 0;
		last_hidden = null;
		cache.length = 0;

		j.on_error(msg);
	}

	/* Penalty counts (presence/frequency penalties requested, sampled):
	 * a (V) fp32 device tensor next to the logits, created from the
	 * first logits row and bumped once per counted token. */
	function counts_for(j, logits) {
		if (!j.pen)
			return null;

		if (!j.cnt_t)
			j.cnt_t = logits.reshape([-1]).to('float32').mul(0.0);

		return j.cnt_t;
	}

	function count_tok(j, tok) {
		if (j.cnt_t)
			j.cnt_t.narrow(0, tok, 1).add_(1.0);
	}

	/* The prefill hands out the post-norm hidden of each finished chunk
	 * (opts.on_hidden) and the draft cache is seeded right then, so the
	 * hidden of the whole prompt is never held (3 GiB at 247k tokens — a
	 * late allocation that scaled with the prompt): seed_prev carries the
	 * last row of the previous chunk into the next. */
	// Two persistent buffers on the hidden's device, allocated once: prev
	// (the last hidden row of the chunk before) and seedbuf (the draft
	// block's input rows [prev, H[0 .. n-2]] of one sub-chunk). Per chunk
	// only copy_ runs — no cat, and no view that keeps a finished chunk's
	// hidden alive.
	let seed_prev = null, seedbuf = null;

	if (mtp_w) {
		seed_prev = torch.zeros([1, 1, hid_w], 'float16', 'cuda:' + M.dev);
		seedbuf = torch.zeros([1, chunk, hid_w], 'float16', 'cuda:' + M.dev);
	}

	function seed_start(j) {
		if (j.pos == 0)
			seed_prev.fill_(0);
		else if (last_hidden != seed_prev)
			seed_prev.copy_(last_hidden);

		j.snap_hid = null;
	}

	function seed_chunk(j, pos, H) {
		let n = H.shape()[1];

		for (let o = 0; o < n; o += chunk) {
			let len = (n - o < chunk) ? n - o : chunk;

			// row 0: the hidden before this sub-chunk; rows 1..: H shifted by one
			seedbuf.narrow(1, 0, 1).copy_((o == 0) ? seed_prev : H.narrow(1, o - 1, 1));

			if (len > 1)
				seedbuf.narrow(1, 1, len - 1).copy_(H.narrow(1, o, len - 1));

			mtp_run(slice(j.ids, pos + o, pos + o + len), seedbuf.narrow(1, 0, len),
				pos + o, null);
		}

		seed_prev.copy_(H.narrow(1, n - 1, 1));

		// the hidden at the snapshot boundary - 1 is the last row of the
		// chunk that ends there (the boundary is a chunk cut): one small copy
		if (j.snap_at && pos + n == j.snap_at)
			j.snap_hid = seed_prev.add(0.0);

		if (j.tcut && pos + n == j.tcut)
			j.tsnap_hid = seed_prev.add(0.0);
	}

	/* Prefill runner over the captured step graphs (same interface as the
	 * models' prefill_pipeline runners): the new tokens go through the
	 * largest captured size that ends before the next snapshot cut, one
	 * replay per step(). Per segment it seeds the draft cache from the
	 * rows' post-norm hidden (popts.on_hidden) and takes the snapshots at
	 * the cuts (popts.on_layer / on_aux for every layer). A 2-token tail on
	 * GLM-5.3 TP: ~45 ms to the first token instead of ~165 ms of eager
	 * launches (L135). The replays are only enqueued here: their device
	 * time shows up in the first decode step. */
	function graph_runner(j, popts) {
		let ids_new = slice(j.ids, j.pos), total = length(ids_new);
		let start = cache.length, done = 0, logits = null;
		let cuts = filter(popts.cuts || [], (c) => c > start && c < start + total);
		let sizes = sort(filter(uniq([ mtp_planned, 4, 3, 2, 1 ]), (k) => k > 0 && graph.has(k)),
			(a, b) => b - a);

		return {
			step: function () {
				if (done >= total)
					return false;

				let pos = start + done, lim = total - done, S = 1;

				for (let c in cuts)
					if (c > pos && c - pos < lim)
						lim = c - pos;

				for (let k in sizes)
					if (k <= lim) {
						S = k;
						break;
					}

				let r = graph.run(slice(ids_new, done, done + S), pos);
				let end = pos + S;

				if (popts.hidden && popts.on_hidden)
					popts.on_hidden(pos, cache.hidden.view([1, S, -1]));

				if (index(cuts, end) >= 0) {
					if (popts.on_layer)
						for (let li = 0; li < cfg.num_hidden_layers; li++)
							popts.on_layer(end, li);

					if (popts.on_aux)
						popts.on_aux(end);
				}

				done += S;

				// a copy: the graph's output buffer is reused by the next replay
				if (done >= total)
					logits = r.logits.view([S, -1]).narrow(0, S - 1, 1).add(0.0);

				return done < total;
			},

			done_tokens: function () {
				return done;
			},

			logits: function () {
				return logits;
			}
		};
	}

	/* Plan the reuse of the cache for j.ids and start the pipelined prefill
	 * of what is new (j.runner). begin() runs it for a new job; inject()
	 * runs it for a job whose generation is extended by forced tokens (the
	 * cache then holds all of j.ids but the tail: live reuse). */
	function start_prefill(j) {
		let plan = plan_reuse(j.ids);

		if (plan.from == 'snapshot')
			restore_snapshot();
		else if (plan.from == 'turn')
			restore_turn();
		else if (plan.from == 'fresh')
			reset_state();

		if (!j.injected) {     // (a forced continuation keeps the request's own reuse info)
			j.reuse_from = plan.from;
			j.reuse_start = plan.start;
			j.reuse_detail = plan.detail;
		}

		j.pos = plan.start;
		j.logits = null;
		j.snap_taken = false;

		// MTP needs the draft cache in step with the target up to the
		// reuse point, and the target hidden state just before it
		j.use_mtp = mtp_w > 0 && (DF || plan.start == 0 ||
			(last_hidden != null && draft_len == plan.start));
		j.use_dspark = !!(SP && graph && graph.has(1) && !j.no_dspark);

		if (j.use_dspark && !j.rounds) {
			j.rounds = 0;
			j.drafted = 0;
			j.accepted = 0;
		}

		// pipelined prefill of the new tokens (wavefront over the
		// layer split, see qwen35.prefill_pipeline); the snapshot
		// boundary becomes a chunk cut, and the GDN states are copied
		// per layer right after the boundary chunk ran that layer —
		// stream order puts the copy before the next chunk touches it
		let snap_at = (j.snap_at && j.snap_at > j.pos && j.snap_at < length(j.ids)) ?
			j.snap_at : 0;

		// the turn checkpoint is taken the same way, at the end of the last
		// user message, unless the cache already holds one there
		let t_at = (TS_OK && j.turn_at && j.turn_at > j.pos && j.turn_at < length(j.ids) &&
			length(gdn_layers) && !(tsnap_ids != null && length(tsnap_ids) == j.turn_at &&
			lcp(tsnap_ids, j.ids) == j.turn_at)) ? j.turn_at : 0;

		j.tcut = t_at;
		j.tsnap_taken = false;
		j.tsnap_hid = null;

		let cuts = [];

		if (snap_at)
			push(cuts, snap_at);

		if (t_at && t_at != snap_at)
			push(cuts, t_at);

		cuts = sort(cuts, (a, b) => a - b);

		if (j.use_mtp && !DF)
			seed_start(j);

		// DFlash: only the newest window of the prompt matters to the drafter
		let df_from = DF ? length(j.ids) - DF.window - 64 : 0;

		df_pprev = null;

		let popts = {
			chunk: chunk,
			cuts: cuts,
			hidden: j.use_mtp && !DF,
			on_hidden: (j.use_mtp && !DF) ? (pos, H) => seed_chunk(j, pos, H) : null,
			on_taps: (SP && !getenv('AMP_DSPARK_NO_PREFILL_TAPS')) ? (taps, pos) => SP.prefill_rows(taps, pos) :
				DF ? function (taps, pos) {
					if (pos + taps[0][0].shape()[1] > df_from)
						DF.prefill_rows(taps, pos);
				} : null,
			on_layer: function (chunk_end, li) {
				if (snap_at && chunk_end == snap_at && snap[li]) {
					save_states(li, snap[li]);
					j.snap_taken = true;
				}

				if (t_at && chunk_end == t_at && tsnap[li]) {
					save_states(li, tsnap[li]);
					j.tsnap_taken = true;
				}
			},
			on_aux: function (chunk_end) {
				if (snap_at && chunk_end == snap_at && model.aux_get) {
					j.snap_aux = model.aux_get(cache);

					// models without per-layer recurrent (GDN) state: the aux state IS the snapshot
					if (!length(gdn_layers))
						j.snap_taken = true;
				}
			}
		};

		let n_new = length(j.ids) - j.pos;

		j.runner = (TAIL_OK && graph && graph.run && graph.has(1) && !cache.mm &&
			n_new > 0 && n_new <= TAIL_MAX) ? graph_runner(j, popts) :
			model.prefill_pipeline ?
			model.prefill_pipeline(cache, slice(j.ids, j.pos), popts) :
			qwen35.prefill_pipeline(model, cache, slice(j.ids, j.pos), popts);
	}

	function begin(j) {
		let s = j.sampling || {};

		job = j;
		j.t_start = now_ms();
		j.generated = 0;
		j.cnt_t = null;
		j.pen = !!((s.presence_penalty || s.frequency_penalty) &&
			(s.temperature == null || s.temperature > 0));
		j.phase = 'prefill';

		if (length(j.ids) == 0)
			return fail('empty prompt');

		if (length(j.ids) >= max_ctx)
			return fail('context_length_exceeded');

		if (j.sampling && j.sampling.seed != null)
			math.srand(j.sampling.seed);

		// multimodal: the image spans of THIS prompt (null: text only).
		// The history `ids` carries a marker id on every image row, so
		// prefix reuse below can never match two different images; the
		// spans give the text layers their MRoPE positions and the
		// prefill its vision embeddings.
		try {
			let mm = model.mm_build ? model.mm_build(cfg, j.images) :
				qwen35.mm_build(cfg, j.images);

			for (let im in (j.images || []))
				if (im.start + (im.emb ? im.emb.shape()[0] : 0) > length(j.ids))
					die('image span exceeds the prompt');

			cache.mm = mm;

			if (cache.mtp)
				cache.mtp.mm = mm;
		}
		catch (e) {
			return fail('invalid image input: ' + e.message);
		}

		j.rounds = 0;
		j.drafted = 0;
		j.accepted = 0;

		start_prefill(j);
	}

	/* Forced tokens (j.force(): the server's reasoning budget): the
	 * generation continues as if the prompt had been extended by them. At
	 * a step boundary the cache holds every emitted token except, in the
	 * MTP phase, the pending t_next — so the continuation is the live-
	 * reuse prefill of [t_next] + forced; the GDN state, the KV and the
	 * draft cache follow through the ordinary prefill path, and MTP goes
	 * on afterwards. The forced tokens count as generated (usage, max_new)
	 * and as seen by the penalties; the caller has already shown them to
	 * its consumer. */
	function inject(j, forced, skip_next) {
		let room = j.max_new - j.generated - 1;

		if (max_ctx - cache.length - 2 < room)
			room = max_ctx - cache.length - 2;

		if (length(forced) > room)
			forced = slice(forced, 0, room);

		if (length(forced) == 0)
			return false;

		let tail = (j.phase == 'mtp' && !skip_next) ? [ j.t_next ] : [];

		push(tail, ...forced);

		for (let f in forced) {
			j.generated++;
			count_tok(j, f);
		}

		let keep = {rounds: j.rounds, drafted: j.drafted, accepted: j.accepted,
			t_prefill_done: j.t_prefill_done};

		j.ids = slice(ids, 0);
		push(j.ids, ...tail);
		j.snap_at = 0;
		j.turn_at = 0;
		j.phase = 'prefill';
		j.injected = (j.injected || 0) + length(forced);
		start_prefill(j);
		j.rounds = keep.rounds;
		j.drafted = keep.drafted;
		j.accepted = keep.accepted;
		j.t_prefill_done = keep.t_prefill_done;

		return true;
	}

	/* Draft cache for the newly prefilled positions: the MTP input at
	 * position i is (token i, target hidden at i-1); runs the MTP block
	 * without its head, in chunks. */

	/* A stop id was sampled and the consumer registered job.on_eos: the
	 * position is redrawn from logits (any shape holding one row) with
	 * the stop ids lowered by `bias` — exactly sampling the position
	 * from the biased distribution (penalties, cuts and all). */
	function biased_token(j, logits, bias) {
		let l = logits.reshape([-1]).to('float32').add(0.0);   // a copy
		let s = j.sampling || {};
		let temp = (s.temperature != null) ? s.temperature : 1.0;

		for (let id in j.stop_ids)
			l.narrow(0, id, 1).add_(-bias);

		if (temp <= 0)
			return int(l.argmax().item());

		return draw(dist(l, s, counts_for(j, l)), null);
	}

	/* job.veto(prefix, tok): the consumer's constrained generation (--call-guard)
	 * refuses `tok` after the pending `prefix` tokens. If job.veto_force(prefix)
	 * then has a deterministic continuation, those tokens are forced instead (the
	 * refused token is discarded, as for a stop id); otherwise the position is
	 * redrawn from logits (the row `tok` was drawn from, any shape holding one
	 * row) with the refused tokens lowered, until one passes; after 8 tries the
	 * last draw goes through. Returns {tok} or {force}. */
	function veto_resolve(j, prefix, tok, logits) {
		if (!j.veto || !j.veto(prefix, tok))
			return {tok: tok};

		// a deterministic continuation: the consumer shows it and hands it back
		if (j.veto_force) {
			let f = j.veto_force(prefix);

			if (type(f) == 'array' && length(f))
				return {force: f};
		}

		if (logits == null)
			return {tok: tok};

		let l = logits.reshape([-1]).to('float32').add(0.0);   // a copy
		let s = j.sampling || {};
		let temp = (s.temperature != null) ? s.temperature : 1.0;

		for (let n = 0; n < 8; n++) {
			l.narrow(0, tok, 1).add_(-1e30);
			tok = (temp <= 0) ? int(l.argmax().item()) : draw(dist(l, s, counts_for(j, l)), null);

			if (!j.veto(prefix, tok))
				break;
		}

		return {tok: tok};
	}

	/* Consult job.on_eos for a sampled stop token (logits: the row it was
	 * drawn from). Returns {tok} with the token to use instead (it may
	 * still be a stop id: the generation then ends), or {force} with
	 * token ids to continue with. */
	function eos_resolve(j, tok, logits) {
		let max = j.eos_max || 4;

		while (j.on_eos && index(j.stop_ids, tok) >= 0 && (j.eos_n || 0) < max) {
			let v = j.on_eos(j.eos_n || 0);

			j.eos_n = (j.eos_n || 0) + 1;

			if (type(v) == 'array' && length(v))
				return {force: v};

			if ((type(v) == 'int' || type(v) == 'double') && v > 0 && logits != null) {
				tok = biased_token(j, logits, v);
				continue;
			}

			break;
		}

		return {tok: tok};
	}

	/* Hand control back to the event loop until the device work
	 * completes: arm the notifier (the AND of the model's streams —
	 * the wake-up means the forward AND the result copies are done)
	 * and return without re-arming the timer. f runs from the handle
	 * callback below. DECLARED BEFORE ITS CALLERS (mtp_round,
	 * decode_after): ucode does not hoist nested functions — a call
	 * into a later-declared sibling resolves as an undeclared global
	 * at runtime (L64). */
	function yield_gpu(f) {
		waiting = true;
		resume = f;
		note.arm(devset.length ? devset : null);
	}

	/* Notifier callback: the results are visible — drain the eventfd
	 * (level-triggered: an undrained fd would spin the loop) and run
	 * the continuation. The continuation may arm again (next sampled
	 * token), finish the job, or hand control back to the timer for
	 * the next host-side phase. */
	function wake() {
		note.drain();

		/* The notifier fires when the streams reach the callback, yet on
		 * multi-GPU layer splits the pinned result copy was observed
		 * becoming visible ~0.5 ms later (stale token, cause unresolved,
		 * L86). The work is done by now, so this sync costs microseconds. */
		torch.synchronize();

		if (!waiting)
			return;

		waiting = false;

		let f = resume;
		resume = null;

		try {
			f();
		}
		catch (e) {
			let tr = '';

			for (let k = 0; e.stacktrace && k < length(e.stacktrace) && k < 6; k++)
				tr += `  ${e.stacktrace[k].filename}:${e.stacktrace[k].line}  ${e.stacktrace[k].context}\n`;

			warn(`engine: step failed: ${e.message}\n${tr}`);
			fail('internal error: ' + e.message);
		}

		if (!waiting && (job || length(queue) || length(parked)))
			timer.set(0);
	}

	/* Acceptance bookkeeping + emit, shared by the synchronous round and
	 * the async continuation. c: the round's token array (seed + W
	 * drafts); a/bonus: the acceptance decision; vhidden: the verify
	 * output's hidden (cache.hidden at verify time). mark/tm: MTP_TIMING
	 * hooks — null from the async path, whose marks would measure host
	 * enqueue, not device time. */
	function mtp_finish(j, p, c, a, bonus, vhidden, mark, tm, vrows) {
		let W = mtp_w;

		// a stop id among the accepted drafts ends the round there: it
		// becomes the round's last token (the bonus), so a consumer that
		// vetoes it (job.on_eos) finds the state consistent — the drafts
		// behind it were accepted conditional on it and are dropped
		for (let i = 1; i <= a; i++) {
			if (index(j.stop_ids, c[i]) >= 0 || (j.veto && j.veto(slice(c, 1, i), c[i]))) {
				a = i - 1;
				bonus = c[i];
				break;
			}
		}

		j.rounds++;
		j.drafted += W;
		j.accepted += a;

		if (mark)
			mark('accept');

		// caches: roll back past the rejected writes
		let p_new = p + ((a < W) ? a + 1 : W);
		let H;

		if (a < W - 1 && spec) {
			// restore the GDN states after token a; the attention KV
			// rolls back with the length, and row a of the verify output
			// is the accepted prefix's hidden (causal)
			torch.copy_many(rollback[a].d, rollback[a].s);
			cache.length = p + a + 2;
			H = vhidden.narrow(1, a, 1);

			if (mark)
				mark('replay');
		}
		else if (a < W - 1) {
			for (let k in gdn_layers)
				load_states(k, vsnap[k]);

			cache.length = p + 1;
			step_fwd(slice(c, 0, a + 1));
			H = cache.hidden.narrow(1, a, 1);

			if (mark)
				mark('replay');

			if (tm)
				tm.replays = (tm.replays || 0) + 1;
		}
		else {
			H = vhidden.narrow(1, (a < W) ? a : W - 1, 1);
		}

		for (let k = 0; k <= ((a < W) ? a : W - 1); k++)
			push(ids, c[k]);

		if (model.aux_ids)
			model.aux_ids(cache, ids);

		if (dc)
			dc.length = p_new + 1;

		draft_len = p_new + 1;
		last_hidden = H.add(0.0);     // a copy: graph buffers are reused
		j.t_next = (a < W) ? bonus : c[W];

		// emit: the accepted drafts, then the bonus token
		let emit = slice(c, 1, a + 1);

		if (bonus != null)
			push(emit, bonus);

		for (let k = 0; k < length(emit); k++) {
			let tok = emit[k];

			if (j.veto && vrows != null && k == a && k == length(emit) - 1) {
				// the bonus: drawn from row a (the tokens before it were emitted, so the consumer's
				// state already includes them), redrawn when the consumer refuses it
				let vr = veto_resolve(j, [], tok, vrows.narrow(0, a, 1));

				if (vr.force) {
					if (inject(j, vr.force, true))
						return;

					return finish('stop');
				}

				if (vr.tok != tok) {
					tok = vr.tok;
					j.t_next = tok;
				}
			}

			if (index(j.stop_ids, tok) >= 0) {
				// (only the bonus can be a stop id here; row a of the verify
				// output is the distribution it was drawn from)
				let r = eos_resolve(j, tok, vrows ? vrows.narrow(0, a, 1) : null);

				if (r.force) {
					if (inject(j, r.force, true))
						return;

					return finish('stop');
				}

				tok = r.tok;

				if (index(j.stop_ids, tok) >= 0)
					return finish('stop');

				j.t_next = tok;
			}

			j.generated++;

			if (k >= a)
				count_tok(j, tok);

			if (j.on_token(tok) == false)
				return finish((j.alive && !j.alive()) ? 'cancelled' : 'stop');

			if (j.generated >= j.max_new)
				return finish('length');
		}
	}

	/* One speculative round (EAGLE-1 with the checkpoint's MTP head, see
	 * qwen35 generate_mtp for the round structure). Drafting is greedy,
	 * so the draft distribution is one-hot: a drafted token x is accepted
	 * with probability p(x) under the target's sampling distribution, and
	 * a rejection draws from p with x removed — the output distribution
	 * equals plain sampling (Leviathan et al. / Chen et al.); at
	 * temperature 0 this is the exact-match rule. */
	function mtp_round(j) {
		let W = mtp_w;
		let p = cache.length - 1;
		let s = j.sampling || {};
		let tm = j.tm, t_last = 0;

		function mark(name) {
			if (!MTP_TIMING)
				return;

			torch.synchronize();

			let t = now_ms();

			// time since the previous round's last mark: emit, HTTP,
			// the event loop
			if (!t_last && j.t_mark)
				tm.between = (tm.between || 0) + (t - j.t_mark);
			else if (t_last)
				tm[name] = (tm[name] || 0) + (t - t_last);

			t_last = j.t_mark = t;
		}

		if (MTP_TIMING && !tm)
			tm = j.tm = {};

		mark(null);

		// near the context end or the token budget: plain decode for
		// the rest (the draft cache then lags; MTP resumes after the
		// next fresh prefill)
		if (cache.length + W + 1 >= max_ctx || j.max_new - j.generated < 2) {
			j.phase = 'decode';
			j.use_mtp = false;
			j.logits = step_fwd([j.t_next]);

			for (let i = 0; i < 1; i++)
				push(ids, j.t_next);

			return;
		}

		// draft chain
		let c = [j.t_next];
		let D = last_hidden;
		let temp = (s.temperature != null) ? s.temperature : 1.0;

		// (a tensor-parallel model scores the draft with its vocabulary-
		// sharded head: model.draft_argmax)
		let sharded = model.draft_argmax && !draft_head;

		// Async greedy round: the draft chain runs on the device (each
		// step's argmax feeds the next step's embedding lookup straight
		// from the device buffer), the verify replays with device-side
		// token ids, and the acceptance prefix-count is computed on the
		// device — one wake-up delivers a, the verify predictions and
		// the drafted tokens to pinned host buffers. Requires a captured
		// verify graph (the eager fallback scores host ids) and the plain
		// lm_head; the sampled walk and the TP sharded draft head stay
		// synchronous below.
		if (temp <= 0 && (DF || (!sharded && !draft_head)) && graph &&
		    graph.has(W) && graph.run_dev) {
			c_dev.narrow(0, 0, 1).fill_(j.t_next);

			if (DF) {
				// the block drafter's W ids land in c_dev[1 .. W] on the device
				DF.draft_async(j.t_next, df_pprev, cache.length, c_dev.narrow(0, 1, W));
				df_pprev = cache.length;
			}
			else
			for (let k = 1; k <= W; k++) {
				let r = mtp_run_ids(c_dev.narrow(0, k - 1, 1).view([1, 1]),
					D, p + k, model.lm_head);

				c_dev.narrow(0, k, 1).copy_(r.logits.reshape([-1]).argmax());
				D = r.hidden;
			}

			// verify: one target forward over c[0 .. W-1] (device ids;
			// advances cache.length like step_fwd)
			let vr = graph.run_dev(c_dev.narrow(0, 0, W).view([1, W]),
				cache.length);
			// (the logits live on the head's device, the draft ids on the
			// first one: bring the W predictions over)
			let vpred = vr.logits.view([W, -1]).argmax(-1).cuda(model.devs[0]);

			// acceptance: a = first index where vpred != c[1..W], or W
			// on a full match — min over (i where mismatch, W where
			// match); fp32 exact for W <= 2^24. NOT the mask sum: tokens
			// can repeat, so matches may occur after a mismatch.
			let cshift = c_dev.narrow(0, 1, W);
			let m = vpred.le(cshift).to('float32')
				.mul(cshift.le(vpred).to('float32'));
			let cand = rb.idx.add(m.mul(W));

			torch.copy_async(rb.a, cand.neg().max().neg().reshape([1]));
			torch.copy_async(rb.pred, vpred);
			torch.copy_async(rb.c, c_dev);

			yield_gpu(function () {
				let a = int(rb.a.item());
				let vp = rb.pred.tolist();
				let cc = rb.c.tolist();

				mtp_finish(j, p, cc, a, (a < W) ? vp[a] : null, vr.hidden,
					null, null, vr.logits.view([W, -1]));
			});

			return;
		}

		if (DF) {
			// one block-diffusion pass proposes the W drafts (greedy walk of the
			// candidate selector: a deterministic one-hot proposal, lossless under
			// the acceptance rules below); its context rows are the last verify's
			// taps, fresh in the graph's buffers when df_pprev is set
			push(c, ...DF.draft(j.t_next, df_pprev, cache.length));
			df_pprev = cache.length;
		}
		else if (model.mtp_chain && sharded && !MTP_TIMING_STEPS && model.mtp_g)
			push(c, ...model.mtp_chain(j.t_next, D, p, W));
		else
		for (let k = 1; k <= W; k++) {
			let r = mtp_run([c[k - 1]], D, p + k, sharded ? null : (draft_head || model.lm_head));

			if (sharded)
				push(c, (r.token != null) ? r.token : model.draft_argmax(r.collapsed || r.hidden));
			else {
				let am = sampler.argmax_last(r.logits);

				push(c, draft_head ? draft_head.map(am) : am);
			}

			D = r.hidden;
		}

		mark('draft');

		// verify: snapshot the GDN states (a rejection rolls them back),
		// then one target forward over c[0 .. W-1]
		if (!spec)
			for (let k in gdn_layers)
				save_states(k, vsnap[k]);

		mark('snapshot');

		let vlogits = step_fwd(slice(c, 0, W));
		let vrows = vlogits.view([W, -1]);
		let vhidden = cache.hidden;

		mark('verify');

		// acceptance: row k scores the token at p+k+2, i.e. c[k+1]
		let a = 0, bonus = null;

		if (temp <= 0) {
			let vpred = vrows.argmax(-1).tolist();

			while (a < W && vpred[a] == c[a + 1])
				a++;

			if (a < W)
				bonus = vpred[a];
		}
		else {
			// accepted drafts count toward the penalties of the next
			// row right away; the emit loop below counts only the bonus
			let cnt = counts_for(j, vrows);

			for (let k = 0; k < W; k++) {
				let x = c[k + 1];
				let r = spec_step(dist(vrows.narrow(0, k, 1), s, cnt), x);

				if (r.accepted) {
					a++;
					count_tok(j, x);
					continue;
				}

				bonus = r.token;
				break;
			}
		}

		mtp_finish(j, p, c, a, bonus, vhidden, mark, tm, vrows);
	}

	/* a speculative round keeps the first L rows: the cache length, and the model's own state of the rows
	 * it took back (pool counts, half-built compressor pairs) when it has any */
	function spec_len(L) {
		if (SP && SP.rollback)
			SP.rollback(L);

		cache.length = L;
	}

	/* One DSpark round: draft + verify through three CUDA graphs (dspark_tp), accept
	 * by the one-hot-draft rule (exact match at temperature 0, p(x) under the
	 * sampling distribution otherwise), roll back by resetting the cache length. */
	function dspark_round(j) {
		let V = dsv;
		let p0 = cache.length;
		let seed = j.t_next;
		let s = j.sampling || {};
		let temp = (s.temperature != null) ? s.temperature : 1.0;

		// near the context end or the token budget: plain decode for the rest
		if (p0 + V + 3 >= max_ctx || j.max_new - j.generated < 2) {
			j.phase = 'decode';
			j.use_dspark = false;
			j.logits = step_fwd([seed]);
			push(ids, seed);

			return;
		}

		// the drafter lacks the main rows of the last window (a reused prefix whose rows were overwritten):
		// one plain step; its taps refill the ring
		if (SP.ready && !SP.ready(p0)) {
			let rr = graph.run([ seed ], p0);

			if (rr.taps)
				SP.prefill_rows(rr.taps, p0);

			j.phase = 'decode';
			j.logits = rr.logits;
			push(ids, seed);

			return;
		}

		let r = SP.run(seed, p0, dsv);                  // r.V <= dsv: the confident prefix of the drafts
		let dr = r.dr;
		let a = 0, bonus = null;

		V = r.V;

		let vrows = r.logits;   // (V + 1) verify rows, when the drafter hands them out

		if (temp <= 0) {
			while (a < V && r.am[a] == dr[a])
				a++;

			bonus = r.am[a];
		}
		else {
			let cnt = counts_for(j, vrows);

			for (let k = 0; k < V; k++) {
				let x = dr[k];
				let rr = spec_step(dist(vrows.narrow(0, k, 1), s, cnt), x);

				if (rr.accepted) {
					a++;
					count_tok(j, x);
					continue;
				}

				bonus = rr.token;
				break;
			}

			if (bonus == null)
				bonus = draw(dist(vrows.narrow(0, V, 1), s, cnt), null);
		}

		// a stop id among the accepted drafts ends the round there and
		// becomes its bonus token (see mtp_finish)
		for (let i = 0; i < a; i++) {
			if (index(j.stop_ids, dr[i]) >= 0 || (j.veto && j.veto(slice(dr, 0, i), dr[i]))) {
				a = i;
				bonus = dr[i];
				break;
			}
		}

		// the cache holds the seed + the a accepted drafts
		let fed = [ seed, ...slice(dr, 0, a) ];
		let emit = slice(dr, 0, a);

		push(emit, bonus);

		spec_len(p0 + a + 1);
		j.rounds++;
		j.drafted += V;
		j.accepted += a;

		for (let k = 0; k < length(emit); k++) {
			let tok = emit[k];
			let stopped = null;

			if (j.veto && vrows != null && k == a && k == length(emit) - 1) {
				let vr = veto_resolve(j, [], tok, vrows.narrow(0, a, 1));

				if (vr.force) {
					// the seed and the accepted drafts are fed; the refused token is dropped
					push(ids, ...fed);

					if (inject(j, vr.force, true))
						return;

					return finish('stop');
				}

				if (vr.tok != tok) {
					tok = vr.tok;
					bonus = tok;
				}
			}

			if (index(j.stop_ids, tok) >= 0) {
				let rr = eos_resolve(j, tok, vrows ? vrows.narrow(0, a, 1) : null);

				if (rr.force) {
					// the seed and the accepted drafts are fed; the stop is dropped
					push(ids, ...fed);

					if (inject(j, rr.force, true))
						return;

					return finish('stop');
				}

				tok = rr.tok;
				bonus = tok;
			}

			if (index(j.stop_ids, tok) >= 0) {
				stopped = 'stop';
			}
			else {
				j.generated++;

				if (k >= a)
					count_tok(j, tok);

				if (j.on_token(tok) == false)
					stopped = (j.alive && !j.alive()) ? 'cancelled' : 'stop';
				else if (j.generated >= j.max_new)
					stopped = 'length';
			}

			if (stopped) {
				// rows kept: the seed and the drafts before this token
				spec_len(p0 + k + 1);
				push(ids, ...slice(fed, 0, k + 1));

				return finish(stopped);
			}
		}

		push(ids, ...fed);
		j.t_next = bonus;

		if (cache.length + 1 >= max_ctx)
			return finish('length');
	}

	/* Consume one decoded token: emit, bookkeeping, and the next
	 * forward (or the hand-over to the MTP phase). Runs from the
	 * notifier callback after the sampling results arrived. */
	function decode_after(j, tok, logits) {
		let vr = veto_resolve(j, [], tok, logits);

		if (vr.force)
			return inject(j, vr.force, true) ? null : finish('stop');

		tok = vr.tok;

		if (index(j.stop_ids, tok) >= 0) {
			let r = eos_resolve(j, tok, logits);

			if (r.force)
				return inject(j, r.force, true) ? null : finish('stop');

			tok = r.tok;

			if (index(j.stop_ids, tok) >= 0)
				return finish('stop');
		}

		j.generated++;
		count_tok(j, tok);

		if (j.on_token(tok) == false)
			return finish((j.alive && !j.alive()) ? 'cancelled' : 'stop');

		if (j.generated >= j.max_new)
			return finish('length');

		if (cache.length + 1 >= max_ctx)
			return finish('length');

		// MTP: the sampled token seeds the next round's draft chain
		if (j.use_mtp) {
			j.t_next = tok;
			j.phase = 'mtp';
			return;
		}

		// DSpark: the sampled token seeds the next round (it is not in the cache yet)
		if (j.use_dspark) {
			j.t_next = tok;
			j.phase = 'dspark';
			return;
		}

		j.logits = step_fwd([tok]);
		push(ids, tok);
	}

	/* One unit of work for the current job. */
	function step() {
		let j = job;

		if (j.phase == 'prefill') {
			let total = length(j.ids);
			let more = j.runner.step(PIPE_ROUNDS);

			if (j.on_progress &&
			    j.on_progress(j.pos + j.runner.done_tokens(), total) == false) {
				// the enqueued work still runs; the cache state is
				// consistent only once it has, so drop the reuse info
				ids = [];
				snap_ids = null;
				tsnap_ids = null;
				tsnap_hidden = null;
				return finish('cancelled');
			}

			if (more)
				return;

			j.logits = j.runner.logits();

			j.runner = null;

			// (the draft cache was seeded chunk by chunk, seed_chunk)
			if (j.use_mtp) {
				if (dc)
					dc.length = total;

				draft_len = total;
			}

			push(ids, ...slice(j.ids, j.pos, total));

			if (j.tsnap_taken) {
				tsnap_ids = slice(j.ids, 0, j.tcut);
				tsnap_hidden = (j.use_mtp && !DF) ? j.tsnap_hid : null;
			}

			j.tsnap_hid = null;

			if (j.snap_taken) {
				snap_ids = slice(j.ids, 0, j.snap_at);
				snap_hidden = (j.use_mtp && !DF) ? j.snap_hid : null;
				snap_aux = j.snap_aux;
			}

			if (j.use_mtp && !DF)
				last_hidden = seed_prev;           // (aliases the persistent buffer; copied on reuse)
			j.snap_hid = null;

			j.pos = total;

			if (!j.injected)
				j.t_prefill_done = now_ms();    // (a forced continuation does not restart the clock)
			j.phase = 'decode';
		}

		// forced tokens from the consumer (the server's reasoning budget):
		// polled at every step boundary, where the state is consistent
		if (j.force) {
			let f = j.force();

			if (f && length(f) && inject(j, f))
				return;
		}

		if (j.phase == 'mtp')
			return mtp_round(j);

		if (j.phase == 'dspark')
			return dspark_round(j);

		// decode: sample from the pending logits, emit, feed back.
		// The sampling runs on the GPU; the host only walks the kept
		// candidates once the notifier delivers the results, so the
		// event loop keeps running while the device works.
		let logits = j.logits;

		j.logits = null;

		let s = j.sampling || {};
		let temp = (s.temperature != null) ? s.temperature : 1.0;

		if (temp <= 0) {
			argmax_begin(null, sbufs, devset, logits);

			yield_gpu(function () {
				decode_after(j, sbufs.tok.item(), logits);
			});

			return;
		}

		let ds = dist_begin(null, sbufs, devset, logits, s,
			counts_for(j, logits));

		yield_gpu(function () {
			decode_after(j, draw(dist_finish(sbufs, ds), null), logits);
		});
	}

	/* --- time slicing: co-generation ----------------------------------------------------
	 * With opts.slice_tokens > 0 and / or opts.slice_ms > 0 a job that has generated that many
	 * tokens (or run that long) since it last got the GPU, while other sessions wait (queued, or
	 * parked earlier), is PARKED at the step boundary (a speculative round may overshoot the
	 * token count by a few tokens): its whole live state goes
	 * through the store (a device slot when there is one, else host RAM) as an entry owned by the
	 * job, the next session runs, and the parked job resumes later where it stopped (restoring
	 * the entry exactly; the pending logits of a plain decode are carried by the job). Sessions
	 * alternate round-robin, a new arrival and a parked job taking turns. Without contention
	 * nothing changes. Only decode-phase jobs are sliced; a prefill runs to completion. */
	function should_preempt() {
		let j = job;

		if ((SLICE_MS <= 0 && SLICE_TOK <= 0) || (slots_total <= 0 && store_budget <= 0))
			return false;

		if (j.phase != 'decode' && j.phase != 'dspark' && j.phase != 'mtp')
			return false;

		if (!length(queue) && !length(parked))
			return false;

		if (!j.t_slice) {
			j.t_slice = now_ms();
			j.g_slice = j.generated;

			return false;
		}

		return (SLICE_TOK > 0 && j.generated - j.g_slice >= SLICE_TOK) ||
			(SLICE_MS > 0 && now_ms() - j.t_slice >= SLICE_MS);
	}

	function preempt() {
		let j = job;
		let lg = (j.phase == 'decode' && j.logits) ? j.logits.add(0.0) : null;   // out of the graph buffer
		let saved_ids = slice(ids);
		let t0 = now_ms();

		j.park = null;

		if (!store_put(j) || !j.park) {
			j.t_slice = now_ms();       // cannot park (no room): carry on, ask again after another slice
			j.g_slice = j.generated;

			return false;
		}

		j.pend_logits = lg;
		j.park_ids = saved_ids;
		j.logits = null;
		j.preempted = (j.preempted || 0) + 1;
		j.t_slice = null;
		push(parked, j);
		job = null;

		// the live state is in the entry now: the engine starts empty (the buffers are overwritten)
		ids = [];
		tsnap_ids = null;
		tsnap_hidden = null;
		snap_ids = null;
		snap_aux = null;
		snap_hidden = null;
		draft_len = 0;
		last_hidden = null;
		cache.length = 0;

		warn(sprintf('engine: job parked after %d tokens (%.0f ms), %d queued, %d parked\n',
			j.generated, now_ms() - t0, length(queue), length(parked)));

		return true;
	}

	// a parked job that was cancelled while it waited: free its entry
	function release_parked(j) {
		for (let k = 0; k < length(store); k++)
			if (store[k] == j.park) {
				store_drop(k);
				break;
			}

		j.park = null;
		j.pend_logits = null;
		j.on_done({reason: 'cancelled', prompt_tokens: length(j.ids), cached_tokens: j.reuse_start,
			reuse: j.reuse_from, reuse_detail: j.reuse_detail, completion_tokens: j.generated,
			preemptions: j.preempted || 0});
	}

	function unpark(j) {
		let e = j.park;
		let k = -1;

		for (let i = 0; i < length(store); i++)
			if (store[i] == e) {
				k = i;
				break;
			}

		// the engine's live state belongs to whoever ran last: keep it as the plan would
		if (length(ids) >= STORE_MIN_TOKENS)
			store_put();

		// (the entry may have moved in the list)
		k = -1;

		for (let i = 0; i < length(store); i++)
			if (store[i] == e) {
				k = i;
				break;
			}

		job = j;
		j.park = null;
		j.t_slice = now_ms();
		j.g_slice = j.generated;

		try {
			let mm = model.mm_build ? model.mm_build(cfg, j.images) : qwen35.mm_build(cfg, j.images);

			cache.mm = mm;

			if (cache.mtp)
				cache.mtp.mm = mm;
		}
		catch (err) {
			return fail('invalid image input: ' + err.message);
		}

		if (k >= 0) {
			splice(store, k, 1);
			store_bytes -= e.bytes;
			store_restore(e, e.n);

			j.logits = j.pend_logits;
			j.pend_logits = null;

			// the MTP chain needs the draft cache and hidden of exactly this position; without
			// them (a state stored without a draft) the rest of the job decodes plainly
			if (DF) {
				// the drafter ring is not stored: it refills from the next rounds' taps
				DF.reset();
				df_pprev = null;
			}
			else if (j.phase == 'mtp' && !(cache.mtp && draft_len == cache.length && last_hidden != null)) {
				j.use_mtp = false;
				j.phase = 'decode';
				j.logits = step_fwd([ j.t_next ]);
				push(ids, j.t_next);
			}

			warn(sprintf('engine: job resumed at %d tokens (%d preemptions)\n', length(ids), j.preempted));

			return;
		}

		// the entry is gone: re-prefill what the cache held (+ the pending seed) and carry on
		let tail = (j.phase == 'mtp' || j.phase == 'dspark') ? [ j.t_next ] : [];
		let keep = {rounds: j.rounds, drafted: j.drafted, accepted: j.accepted, t_prefill_done: j.t_prefill_done};

		j.ids = [ ...j.park_ids, ...tail ];
		j.snap_at = 0;
		j.turn_at = 0;
		j.phase = 'prefill';
		j.injected = (j.injected || 0) + 1;
		j.pend_logits = null;
		warn(`engine: parked state lost, re-prefilling ${length(j.ids)} tokens\n`);
		start_prefill(j);
		j.rounds = keep.rounds;
		j.drafted = keep.drafted;
		j.accepted = keep.accepted;
		j.t_prefill_done = keep.t_prefill_done;
	}

	function tick() {
		/* a submit() during a pending GPU continuation re-arms the
		 * timer; the continuation's tail re-arms after it runs — do
		 * not step on top of the in-flight round */
		if (waiting)
			return;

		// a parked job whose client went away must not hold its slot
		for (let i = length(parked) - 1; i >= 0; i--)
			if (parked[i].alive && !parked[i].alive()) {
				release_parked(parked[i]);
				splice(parked, i, 1);
			}

		if (job && should_preempt())
			preempt();

		if (!job) {
			while ((length(queue) || length(parked)) && !job) {
				let from_parked = length(parked) && (!length(queue) || pick_parked);
				let j = from_parked ? shift(parked) : shift(queue);

				pick_parked = !from_parked;      // alternate while both wait

				if (j.alive && !j.alive()) {
					if (from_parked)
						release_parked(j);
					else
						j.on_done({reason: 'cancelled', prompt_tokens: length(j.ids),
							cached_tokens: 0, completion_tokens: 0});

					continue;
				}

				if (from_parked)
					unpark(j);
				else
					begin(j);
			}

			if (!job)
				return;     // idle: no re-arm
		}

		try {
			step();
		}
		catch (e) {
			warn(`engine: step failed: ${e.message}\n${e.stacktrace ? e.stacktrace[0].context : ''}\n`);
			fail('internal error: ' + e.message);
		}

		if (!waiting && (job || length(queue) || length(parked)))
			timer.set(0);
	}

	timer = uloop.timer(-1, tick);
	uloop.handle(note, wake, uloop.ULOOP_READ);

	/* Persist the live generation state to a safetensors file (hot
	 * reload: the server saves on SIGINT/SIGTERM and restores on
	 * start). Saves the token history, every layer's attention KV
	 * (used part only) and GDN recurrent/conv states (per rank under
	 * TP), the MTP draft cache and the last hidden state. The prompt
	 * snapshot is NOT saved (a divergent new prompt then costs a fresh
	 * prefill instead of a snapshot rewind). Refuses while a job is
	 * running (mid-job cache contents are inconsistent with ids).
	 * Returns the number of tokens saved. */
	function save_state_dense(path) {
		if (job)
			die('engine.save_state: a job is running');

		if (length(ids) == 0)
			die('engine.save_state: no cached state');

		let ents = [ [ 'ids', torch.from_array(ids, 'int64') ] ];

		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			for (let r = 0; r < length(sts); r++) {
				let cl = sts[r];
				let pfx = `l${i}.r${r}`;

				if (cl.k_cache) {
					push(ents, [ pfx + '.k',
						cl.k_cache.narrow(2, 0, cache.length) ]);
					push(ents, [ pfx + '.v',
						cl.v_cache.narrow(2, 0, cache.length) ]);
				}

				if (cl.rec_state) {
					push(ents, [ pfx + '.rec', cl.rec_state ]);
					push(ents, [ pfx + '.conv', cl.conv_state ]);
				}
			}
		}

		if (cache.mtp && draft_len > 0) {
			push(ents, [ 'draft.k',
				cache.mtp.k_cache.narrow(2, 0, draft_len) ]);
			push(ents, [ 'draft.v',
				cache.mtp.v_cache.narrow(2, 0, draft_len) ]);
		}

		if (M && last_hidden)
			push(ents, [ 'last_hidden', last_hidden ]);

		torch.st_save(path, ents);

		return length(ids);
	}

	/* Restore state saved by save_state(): reads the file, copies the
	 * tensors into the live caches/states and rewinds the token
	 * history. Missing optional entries (draft, last_hidden) skip
	 * gracefully; shape mismatches raise (the caller decides whether
	 * to continue fresh). Must run before any job is submitted. */
	function restore_state_dense(path) {
		if (job)
			die('engine.restore_state: a job is running');

		let st = torch.st(path);
		let names = st.list();
		let has = (n) => index(names, n) >= 0;

		if (!has('ids'))
			die('engine.restore_state: no ids in file');

		let ids_t = st.tensor('ids');
		let N = ids_t.shape()[0];

		if (N > cache.max_seq)
			die(`engine.restore_state: ${N} tokens exceed the ` +
				`cache capacity (${cache.max_seq})`);

		for (let i = 0; i < cfg.num_hidden_layers; i++) {
			let sts = states(i);

			for (let r = 0; r < length(sts); r++) {
				let cl = sts[r];
				let pfx = `l${i}.r${r}`;

				if (cl.k_cache && has(pfx + '.k')) {
					cl.k_cache.narrow(2, 0, N)
						.copy_(st.tensor(pfx + '.k'));
					cl.v_cache.narrow(2, 0, N)
						.copy_(st.tensor(pfx + '.v'));
				}

				if (cl.rec_state && has(pfx + '.rec')) {
					cl.rec_state.copy_(st.tensor(pfx + '.rec'));
					cl.conv_state.copy_(st.tensor(pfx + '.conv'));
				}
			}
		}

		ids = ids_t.tolist();
		cache.length = N;
		tsnap_ids = null;
		tsnap_hidden = null;
		snap_ids = null;
		snap_hidden = null;

		if (cache.mtp && has('draft.k')) {
			let dl = st.tensor('draft.k').shape()[2];

			cache.mtp.k_cache.narrow(2, 0, dl)
				.copy_(st.tensor('draft.k'));
			cache.mtp.v_cache.narrow(2, 0, dl)
				.copy_(st.tensor('draft.v'));
			cache.mtp.length = dl;
			draft_len = dl;
		}

		if (M && has('last_hidden'))
			last_hidden = st.tensor('last_hidden').cuda(M.dev);

		return N;
	}

	/* Hot reload through the cache store's entry layout. Models whose cache is
	 * not plain KV + GDN rows (GLM-5.3's MLA latent rows, indexer planes and
	 * pools, KDA states, the MTP draft cache; DeepSeek-V4's own export) are
	 * written as the host-RAM store entry the live state would become: the
	 * entry's tensors go to `path` (safetensors, named t0, t1, ...), its
	 * structure (arrays / objects / numbers / tensor references) to
	 * `path.tree`. The entry carries the snapshot and turn checkpoint, so a
	 * divergent first prompt after the restart still rewinds instead of
	 * prefilling. Restoring is the store's own swap-in. */
	function tree_pack(v, ents) {
		if (v == null)
			return null;

		let t = type(v);

		if (t == 'array')
			return map(v, (x) => tree_pack(x, ents));

		if (t == 'object') {
			let o = {};

			for (let k in v)
				o[k] = tree_pack(v[k], ents);

			return o;
		}

		if (t != 'resource')
			return v;

		let r = {'$t': length(ents), shape: v.shape(), dtype: v.dtype()};

		// an empty tensor has no bytes to store: shape and dtype recreate it
		if (v.numel() > 0)
			push(ents, ['t' + length(ents), v]);
		else
			r['$t'] = -1;

		return r;
	}

	function tree_unpack(v, st) {
		let t = type(v);

		if (t == 'array')
			return map(v, (x) => tree_unpack(x, st));

		if (t != 'object')
			return v;

		if (exists(v, '$t'))
			return (v['$t'] < 0) ? torch.empty(v.shape, v.dtype) : st.tensor('t' + v['$t']);

		let o = {};

		for (let k in v)
			o[k] = tree_unpack(v[k], st);

		return o;
	}

	function entry_state() {
		return !!(mla_store || model.cache_export);
	}

	function save_state(path) {
		if (!entry_state())
			return save_state_dense(path);

		if (job)
			die('engine.save_state: a job is running');

		if (length(ids) < STORE_MIN_TOKENS)
			die('engine.save_state: no cached state worth saving');

		// the entry the live state would become, in host RAM whatever the store's
		// budget and device slots are: borrow the store for one put
		let sb = store_budget, sf = slots_free, stot = slots_total;
		let before = length(store), ok = false;

		store_budget = 1 << 50;
		slots_free = [];
		slots_total = 0;

		try {
			ok = store_put();
		}
		catch (e) {
			store_budget = sb;
			slots_free = sf;
			slots_total = stot;

			die(`engine.save_state: ${e.message}`);
		}

		store_budget = sb;
		slots_free = sf;
		slots_total = stot;

		if (!ok || length(store) <= before)
			die('engine.save_state: the cache entry could not be exported');

		let e = pop(store);

		store_bytes -= e.bytes;

		let ents = [];
		let tree = tree_pack({...e, slot: null}, ents);

		torch.st_save(path, ents);
		writefile(path + '.tree', sprintf('%J', tree));

		return e.n;
	}

	function restore_state(path) {
		let tree;

		try {
			tree = readfile(path + '.tree');
		}
		catch (e) {
			tree = null;
		}

		if (!entry_state() || tree == null)
			return restore_state_dense(path);

		if (job)
			die('engine.restore_state: a job is running');

		let e = tree_unpack(json(tree), torch.st(path));

		if (type(e) != 'object' || type(e.ids) != 'array' || !(e.n > 0))
			die('engine.restore_state: malformed entry');

		if (e.n > cache.max_seq)
			die(`engine.restore_state: ${e.n} tokens exceed the cache capacity (${cache.max_seq})`);

		e.slot = null;
		e.owner = null;
		e.bytes = 0;
		store_restore(e, e.n);

		return e.n;
	}

	/* Startup warm-up (as exllamav3's "Warming up"): run the worst-case
	 * shapes once — a full prefill chunk at the END of the context (the
	 * sparse attention then sees its largest pool count), a decode step,
	 * and the MTP verify and draft shapes — so the caching allocator
	 * reserves its whole working set NOW. A context that does not fit then
	 * fails at load with a clear message instead of in the middle of the
	 * first long request, and requests afterwards reuse reserved blocks
	 * (no cudaMalloc on the serving path). The caches are reset after. */
	function warmup() {
		if (opts.warmup == false)
			return;

		let t0 = now_ms();
		let W = mtp_w, H = hid_w;
		let n = (chunk < max_ctx / 2) ? chunk : int(max_ctx / 2);
		let pos0 = max_ctx - n - W - 4;

		if (pos0 < 0)
			pos0 = 0;

		try {
			zero_states();
			cache.length = pos0;

			let wids = [];

			for (let i = 0; i < n; i++)
				push(wids, 0);

			let popts = {chunk: n, cuts: [], hidden: W > 0 && !DF};
			let run = model.prefill_pipeline ?
				model.prefill_pipeline(cache, wids, popts) :
				qwen35.prefill_pipeline(model, cache, wids, popts);

			while (run.step(1)) {}

			let hid = (W > 0) ? run.hidden() : null;

			// decode step, MTP verify window, draft steps
			step_fwd([ 0 ]);

			if (W > 0) {
				let w = [];

				for (let i = 0; i < W; i++)
					push(w, 0);

				step_fwd(w);

				if (!DF) {
					let D = torch.zeros([ 1, 1, H ], 'float16', 'cuda:' + M.dev);

					for (let k = 0; k < W; k++)
						D = mtp_run([ 0 ], D, pos0 + k, model.lm_head).hidden;

					if (hid)
						mtp_run(slice(wids, 0, 2), hid.narrow(1, 0, 2).contiguous(), pos0, null);
				}
			}

			torch.synchronize();
		}
		catch (e) {
			die(`engine warm-up failed (context ${max_ctx}, chunk ${n} does not fit in VRAM?): ${e.message}`);
		}

		reset_state();
		warn(`engine: warm-up done in ${sprintf('%.1f', (now_ms() - t0) / 1000.0)} s\n`);
	}

	warmup();

	/* Resident slots: after everything else is allocated (weights, graphs, the warm-up working
	 * set), take full-size pool arenas until a GPU is out of memory or has only the margin
	 * left (opts.slot_margin MiB, default 512). That count is the number of conversations
	 * that can be parked on the device besides the live one. */
	function alloc_slots() {
		if (slots_cfg == 0 || !(model.new_slot || mla_store) || !model.devs)
			return;

		let cap = (slots_cfg == 'auto') ? 256 : int(slots_cfg);   // (auto: as many as fit, 256 at most)
		let margin = (opts.slot_margin == null ? 512 : +opts.slot_margin) * 1048576;
		let t0 = now_ms();
		let why = 'requested count reached';

		function free_of(d) {
			return torch.mem_info(d)[2];
		}

		function min_free() {
			let f = null;

			for (let d in model.devs) {
				let m = free_of(d);

				if (f == null || m < f)
					f = m;
			}

			return f;
		}

		// per-device cost of the previous slot: the allocator keeps a freed block reserved, so a slot
		// cannot be given back to the driver; stop BEFORE the allocation that would breach the margin
		let cost = null;

		while (slots_total < cap) {
			if (cost) {
				let short = false;

				for (let i = 0; i < length(model.devs); i++)
					if (free_of(model.devs[i]) < cost[i] + margin)
						short = true;

				if (short) {
					why = sprintf('margin reached (%.0f MiB free, %.0f MiB per slot)', min_free() / 1048576.0,
						max(...cost) / 1048576.0);
					break;
				}
			}

			let before = map(model.devs, free_of);
			let s;

			try {
				s = model.new_slot ? model.new_slot(max_ctx) : new_slot_generic();
				torch.synchronize();
			}
			catch (e) {
				why = 'out of memory';
				break;
			}

			cost = map(model.devs, (d, i) => max(before[i] - free_of(d), 0));
			push(slots_free, s);
			slots_total++;
		}

		warn(sprintf('engine: %d resident cache slots (%s) allocated in %.1f s, %.0f MiB free on the fullest GPU\n',
			slots_total, why, (now_ms() - t0) / 1000.0, min_free() / 1048576.0));
	}

	alloc_slots();

	return {
		max_ctx: max_ctx,

		save_state: function (path) {
			return save_state(path);
		},

		restore_state: function (path) {
			return restore_state(path);
		},

		submit: function (j) {
			push(queue, j);
			timer.set(0);
		},

		busy: function () {
			return job != null;
		},

		queued: function () {
			return length(queue);
		},

		/* ENGINE_PROBE: scan every persistent cache tensor (KV rows, indexer,
		 * pooled blocks, GDN states, draft cache, PLE conv tail) for non-finite
		 * values. Returns ["name: first-bad-row N/total", ...] (empty = clean). */
		scan: function () {
			let out = [];

			function bad(t) {
				let x = t.to('float32');

				return x.sub(x).abs().max().item() != 0.0;     // NaN iff any element is inf/NaN
			}

			function first_bad(t) {
				let n = t.shape()[0], lo = 0;

				while (n > 1) {
					let h = int(n / 2);

					if (bad(t.narrow(0, lo, h)))
						n = h;
					else {
						lo += h;
						n -= h;
					}
				}

				return lo;
			}

			function walk(name, v) {
				if (v == null)
					return;

				if (type(v) == 'array') {
					for (let i = 0; i < length(v); i++)
						walk(name + `[${i}]`, v[i]);

					return;
				}

				let sh;

				try {sh = v.shape();} catch (e) {sh = null;}

				if (sh != null) {
					if (length(sh) && bad(v))
						push(out, sprintf('%s: first-bad-row %d/%d', name, first_bad(v), sh[0]));

					return;
				}

				if (type(v) == 'object')
					for (let k in v)
						if (k != 'cos' && k != 'sin')
							walk(name + '.' + k, v[k]);
			}

			walk('layers', cache.layers);
			walk('mtp', cache.mtp);
			walk('ple', cache.ple ? {conv: cache.ple.conv} : null);

			return out;
		},

		cached: function () {
			let c = {live: length(ids), snapshot: snap_ids ? length(snap_ids) : 0};

			c.draft = draft_len;
			c.stored = length(store);
			c.stored_bytes = store_bytes;
			c.slots = slots_total;
			c.slots_free = length(slots_free);
			c.parked = length(parked);

			return c;
		},

		mtp_window: mtp_w
	};
}
