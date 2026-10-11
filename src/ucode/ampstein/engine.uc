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
//   job.max_new    generation limit
//   job.sampling   {temperature, top_k, top_p, min_p, seed,
//                    presence_penalty, frequency_penalty}
//   job.stop_ids   token ids that end generation (not emitted)
//   job.on_progress(done, total)   prefill progress, return false to abort
//   job.on_token(id)               return false to stop
//   job.on_done(info)              {reason, prompt_tokens, cached_tokens,
//                                    completion_tokens, timings}
//   job.on_error(msg)

import * as qwen35 from './qwen35.uc';
import * as uloop from 'uloop';
import * as math from 'math';
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
	let gdn_layers = [];

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

		cache.length = 0;
	}

	// GDN state snapshot buffers (allocated once): the prompt-cache
	// snapshot (snap) and the MTP verify rollback (vsnap, below)
	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		if (cfg.layer_types[i] != 'linear_attention')
			continue;

		push(gdn_layers, i);
		snap[i] = snap_bufs(i);
	}

	let ids = [];          // tokens currently processed into the cache
	let snap_ids = null;   // token prefix the snapshot represents

	// CUDA-graph decode steps (opts.graphs != false): sizes 1 (plain
	// decode) and 1..W (MTP verify windows and replays). Captured now,
	// before any prefill — the eager warmup inside clobbers the cache,
	// which reset_state() below clears.
	let mtp_planned = (model.mtp && opts.mtp_window > 0) ? opts.mtp_window : 0;
	let graph = null;

	// per-position GDN states for MTP verify (W >= 2): the W-token verify
	// forward keeps every GDN layer's recurrent + conv state after each
	// of its first W-1 tokens (qwen35.gdn_forward, cl.spec_*), so a
	// rejection after token a restores them with one batched copy per GPU
	// (rollback[a]) — no GDN snapshot before the verify, no re-run of the
	// accepted tokens after it. Allocated before graph capture: the
	// verify graph writes these buffers.
	let spec = mtp_planned >= 2;
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
	}

	if (opts.graphs != false) {
		let sizes = [ 1 ];

		// with per-position states only the verify size replays; the
		// snapshot path re-runs 2..W-1 accepted tokens as well
		if (spec)
			push(sizes, mtp_planned);
		else
			for (let k = 2; k <= mtp_planned; k++)
				push(sizes, k);

		graph = qgraph.create(model, cache, sizes);

		zero_states();
	}

	// MTP speculative decoding (opts.mtp_window > 0 and a checkpoint MTP
	// head): the draft cache mirrors the target positions; generation
	// then runs in rounds (draft W tokens, verify them in one target
	// forward, accept by the speculative-sampling rule).
	let M = model.mtp;

	let mtp_w = (M && opts.mtp_window > 0) ? opts.mtp_window : 0;
	let hid_w = model.hidden_width || cfg.hidden_size;     // width of the draft head's hidden input
	let dc = cache.mtp;

	// draft-step entries: the qwen35 MTP block
	function mtp_run(token_ids, hidden, pos, head) {
		return qwen35.mtp_chunk(cfg, M, model.embed_tokens, model.devs[0], token_ids,
			hidden, dc, pos, head);
	}

	function mtp_run_ids(ids_dev, hidden, pos, head) {
		return qwen35.mtp_chunk_ids(cfg, M, model.embed_tokens, model.devs[0], ids_dev,
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

	let queue = [];
	let job = null;
	let timer = null;

	function restore_snapshot() {
		for (let k in gdn_layers)
			load_states(k, snap[k]);

		ids = slice(snap_ids);
		cache.length = length(ids);
		draft_len = length(ids);
		last_hidden = snap_hidden;
	}

	function reset_state() {
		zero_states();
		ids = [];

		// a fresh prefill overwrites the attention KV from position 0:
		// an older snapshot's prefix is no longer in the cache
		snap_ids = null;
		snap_hidden = null;
		draft_len = 0;
		last_hidden = null;
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

		if (live < length(ids) && live > 0) {
			det.div_old = slice(ids, max(live - 4, 0), live + 8);
			det.div_new = slice(prompt, max(live - 4, 0), live + 8);
		}

		return plan;
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
	}

	/* Plan the reuse of the cache for j.ids and start the pipelined prefill
	 * of what is new (j.runner). begin() runs it for a new job; inject()
	 * runs it for a job whose generation is extended by forced tokens (the
	 * cache then holds all of j.ids but the tail: live reuse). */
	function start_prefill(j) {
		let plan = plan_reuse(j.ids);

		if (plan.from == 'snapshot')
			restore_snapshot();
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
		j.use_mtp = mtp_w > 0 && (plan.start == 0 ||
			(last_hidden != null && draft_len == plan.start));

		// pipelined prefill of the new tokens (wavefront over the
		// layer split, see qwen35.prefill_pipeline); the snapshot
		// boundary becomes a chunk cut, and the GDN states are copied
		// per layer right after the boundary chunk ran that layer —
		// stream order puts the copy before the next chunk touches it
		let snap_at = (j.snap_at && j.snap_at > j.pos && j.snap_at < length(j.ids)) ?
			j.snap_at : 0;

		if (j.use_mtp)
			seed_start(j);

		let popts = {
			chunk: chunk,
			cuts: snap_at ? [ snap_at ] : [],
			hidden: j.use_mtp,
			on_hidden: j.use_mtp ? (pos, H) => seed_chunk(j, pos, H) : null,
			on_layer: function (chunk_end, li) {
				if (!snap_at || chunk_end != snap_at || !snap[li])
					return;

				save_states(li, snap[li]);
				j.snap_taken = true;
			}
		};

		j.runner = qwen35.prefill_pipeline(model, cache, slice(j.ids, j.pos), popts);
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

		j.rounds = 0;
		j.drafted = 0;
		j.accepted = 0;

		start_prefill(j);
	}

	/* Draft cache for the newly prefilled positions: the MTP input at
	 * position i is (token i, target hidden at i-1); runs the MTP block
	 * without its head, in chunks. */

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

		if (!waiting && (job || length(queue)))
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
			if (index(j.stop_ids, c[i]) >= 0) {
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

			if (index(j.stop_ids, tok) >= 0) {
				// (only the bonus can be a stop id here; row a of the verify
				// output is the distribution it was drawn from)
				return finish('stop');
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
		if (temp <= 0 && !sharded && !draft_head && graph &&
		    graph.has(W) && graph.run_dev) {
			c_dev.narrow(0, 0, 1).fill_(j.t_next);

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

	/* Consume one decoded token: emit, bookkeeping, and the next
	 * forward (or the hand-over to the MTP phase). Runs from the
	 * notifier callback after the sampling results arrived. */
	function decode_after(j, tok, logits) {
		if (index(j.stop_ids, tok) >= 0)
			return finish('stop');

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
				return finish('cancelled');
			}

			if (more)
				return;

			j.logits = j.runner.logits();

			j.runner = null;

			// (the draft cache was seeded chunk by chunk, seed_chunk)
			if (j.use_mtp) {
				dc.length = total;
				draft_len = total;
			}

			push(ids, ...slice(j.ids, j.pos, total));

			if (j.snap_taken) {
				snap_ids = slice(j.ids, 0, j.snap_at);
				snap_hidden = j.use_mtp ? j.snap_hid : null;
			}

			if (j.use_mtp)
				last_hidden = seed_prev;           // (aliases the persistent buffer; copied on reuse)
			j.snap_hid = null;

			j.pos = total;

			if (!j.injected)
				j.t_prefill_done = now_ms();    // (a forced continuation does not restart the clock)
			j.phase = 'decode';
		}

		if (j.phase == 'mtp')
			return mtp_round(j);

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

	function tick() {
		/* a submit() during a pending GPU continuation re-arms the
		 * timer; the continuation's tail re-arms after it runs — do
		 * not step on top of the in-flight round */
		if (waiting)
			return;

		if (!job) {
			while (length(queue) && !job) {
				let j = shift(queue);

				if (j.alive && !j.alive()) {
					j.on_done({reason: 'cancelled', prompt_tokens: length(j.ids),
						cached_tokens: 0, completion_tokens: 0});

					continue;
				}

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

		if (!waiting && (job || length(queue)))
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
	function save_state(path) {
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
	function restore_state(path) {
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

		cached: function () {
			let c = {live: length(ids), snapshot: snap_ids ? length(snap_ids) : 0};

			c.draft = draft_len;

			return c;
		},

		mtp_window: mtp_w
	};
}
