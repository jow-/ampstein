// ucode/sampler.uc — token sampling over vocab logits.
//
// Works on the tail logits tensor produced by the forward graph — any
// shape that flattens to (vocab,): (V,), (1, V), (1, 1, V). Returns a
// single token id (integer).
//
//   temperature  0    -> greedy argmax (default; deterministic)
//                t    -> softmax(logits / t), sample from the result
//   top_k        0    -> off (default); k -> keep the k highest logits
//
// Seeding is NOT done here (a per-call seed would repeat one draw
// across a generation loop); seed the math module once before
// generating: math.srand(seed) — llama.generate(opts.seed) does it.
//
// The probability math runs on the CPU side in ucode (tolist of the
// filtered candidate set). With top_k set, only k values cross the
// GPU->CPU boundary per token; without it, the whole vocab does — fine
// for correctness-first phases, revisit when profiling.

import * as math from 'math';

/* Greedy argmax over a (…, V) logits tensor: returns the winning token
 * id per LAST position as an integer. Cheaper than topk(1) — no heap,
 * no values tensor. Used by the speculative verify path and any hot
 * greedy loop. */
export function argmax_last(logits) {
	let s = logits.shape();
	let flat = (length(s) == 1) ? logits :
		logits.narrow(length(s) - 2, s[length(s) - 2] - 1, 1)
			.reshape([-1]);

	return flat.argmax().item();
}

/* Sample one token id from `logits` (see above). opts keys:
 * temperature, top_k. */
export function sample(logits, opts) {
	opts = opts || {};

	let temperature = (opts.temperature != null) ? opts.temperature : 0;
	let top_k = (opts.top_k != null) ? opts.top_k : 0;

	let flat = logits.reshape([-1]);   // (V,)

	// greedy: argmax, no randomness involved
	if (temperature == 0)
		return flat.topk(1, 0).indices.item();

	// candidate set: top-k filtered, or the whole vocab
	let vals, idxs;

	if (top_k > 0) {
		// topk returns {values, indices} tensors; pull both to CPU
		// before crossing into ucode arrays
		let tk = flat.topk(top_k, 0);

		vals = tk.values.cpu().tolist();
		idxs = tk.indices.cpu().tolist();
	} else {
		vals = flat.cpu().tolist();
		idxs = null;
	}

	// softmax over the candidates, numerically stable via the max
	// (topk output is descending, so vals[0] IS the max; on the full
	// path scan for it)
	let maxv = vals[0];

	if (top_k == 0)
		for (let i = 1; i < length(vals); i++)
			if (vals[i] > maxv)
				maxv = vals[i];

	let n = length(vals);
	let w = [];
	let total = 0.0;

	for (let i = 0; i < n; i++) {
		let e = math.exp((vals[i] - maxv) / temperature);

		push(w, e);
		total += e;
	}

	// inverse-transform sample: r uniform in [0, total). NB: bare
	// math.rand() returns an INTEGER in [0, RAND_MAX); the two-arg
	// range form returns a double.
	let r = math.rand(0, total);

	for (let i = 0; i < n; i++) {
		r -= w[i];

		if (r < 0)
			return (idxs != null) ? idxs[i] : i;
	}

	// fp16 rounding may leave a sliver of r unspent — fall back to the
	// last candidate
	return (idxs != null) ? idxs[n - 1] : n - 1;
}