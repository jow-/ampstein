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
import * as qgraph from './qwen35_graph.uc';

function now_ms() {
	let t = clock(true);

	return t[0] * 1000 + t[1] / 1e6;
}

/* Pipeline rounds per uloop tick during prefill: one round enqueues one
 * layer per device stage (host-side work only; the GPUs run behind). */
const PIPE_ROUNDS = 8;

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
		return [ cache.layers[i] ];
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

	// GDN state snapshot buffers (allocated once): the prompt-cache snapshot
	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		if (cfg.layer_types[i] != 'linear_attention')
			continue;

		push(gdn_layers, i);
		snap[i] = snap_bufs(i);
	}

	let ids = [];          // tokens currently processed into the cache
	let snap_ids = null;   // token prefix the snapshot represents

	// CUDA-graph decode steps (opts.graphs != false), captured now, before
	// any prefill — the capture's warm-up runs clobber the cache, which
	// zero_states() below clears.
	let graph = null;

	if (opts.graphs != false) {
		let sizes = [ 1 ];

		graph = qgraph.create(model, cache, sizes);

		zero_states();
	}

	let queue = [];
	let job = null;
	let timer = null;

	function restore_snapshot() {
		for (let k in gdn_layers)
			load_states(k, snap[k]);

		ids = slice(snap_ids);
		cache.length = length(ids);
	}

	function reset_state() {
		zero_states();
		ids = [];

		// a fresh prefill overwrites the attention KV from position 0:
		// an older snapshot's prefix is no longer in the cache
		snap_ids = null;
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

		let now = now_ms();

		j.on_done({
			reason: reason,
			prompt_tokens: length(j.ids),
			cached_tokens: j.reuse_start,
			reuse: j.reuse_from,
			reuse_detail: j.reuse_detail,
			completion_tokens: j.generated,
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

		// pipelined prefill of the new tokens (wavefront over the
		// layer split, see qwen35.prefill_pipeline); the snapshot
		// boundary becomes a chunk cut, and the GDN states are copied
		// per layer right after the boundary chunk ran that layer —
		// stream order puts the copy before the next chunk touches it
		let snap_at = (j.snap_at && j.snap_at > j.pos && j.snap_at < length(j.ids)) ?
			j.snap_at : 0;

		let popts = {
			chunk: chunk,
			cuts: snap_at ? [ snap_at ] : [],
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

		start_prefill(j);
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

		if (!waiting && (job || length(queue)))
			timer.set(0);
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

			push(ids, ...slice(j.ids, j.pos, total));

			if (j.snap_taken) {
				snap_ids = slice(j.ids, 0, j.snap_at);
			}

			j.pos = total;

			if (!j.injected)
				j.t_prefill_done = now_ms();    // (a forced continuation does not restart the clock)
			j.phase = 'decode';
		}

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

			return c;
		},

		mtp_window: 0
	};
}
