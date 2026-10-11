// tests/sampling_dist.uc — engine.dist (tensor-op sampling distribution)
// against the former all-ucode implementation (kept below as the
// reference): same candidates in the same order and the same
// probabilities for temperature / top_k / top_p / min_p / presence and
// frequency penalties, on synthetic logits. Also torch.common_prefix.
// Needs a GPU (skips otherwise).

import * as engine from 'ampstein.engine';
import * as math from 'math';

if (torch.device_count() < 1) {
	warn('skip: no GPU\n');
	exit(125);
}

/* the reference: engine.dist before it moved to tensor ops, host counts */
function dist_ref(logits, s, counts) {
	let temp = (s.temperature != null) ? s.temperature : 1.0;

	if (temp <= 0)
		return null;

	let flat = logits.reshape([-1]);
	let k = (s.top_k != null && s.top_k > 0) ? s.top_k : 1000;

	if (k > 1000)
		k = 1000;

	let tk = flat.topk(k, 0);
	let vals = tk.values.cpu().to('float32').tolist();
	let idxs = tk.indices.cpu().tolist();
	let n = length(vals);

	let pp = s.presence_penalty || 0, fp = s.frequency_penalty || 0;

	if ((pp || fp) && counts) {
		for (let i = 0; i < n; i++) {
			let c = counts[idxs[i]];

			if (c)
				vals[i] -= pp + fp * c;
		}
	}

	let maxv = max(...vals);

	let w = [], total = 0.0;

	for (let i = 0; i < n; i++) {
		let e = math.exp((vals[i] - maxv) / temp);

		push(w, e);
		total += e;
	}

	// order by probability (penalties may have reordered the candidates)
	let order = [];

	for (let i = 0; i < n; i++)
		push(order, i);

	if (pp || fp)
		order = sort(order, (a, b) => (w[b] > w[a]) ? 1 : ((w[b] < w[a]) ? -1 : 0));

	// min-p: drop candidates below min_p * p_max
	let keep = n;
	let min_p = s.min_p || 0;

	if (min_p > 0) {
		let thr = min_p * w[order[0]];

		keep = 0;

		while (keep < n && w[order[keep]] >= thr)
			keep++;
	}

	// top-p over the probability-ordered candidates
	let top_p = (s.top_p != null) ? s.top_p : 1.0;

	if (top_p < 1.0) {
		let acc = 0.0, lim = 0;

		while (lim < keep) {
			acc += w[order[lim]] / total;
			lim++;

			if (acc >= top_p)
				break;
		}

		keep = lim;
	}

	let sub = 0.0;

	for (let i = 0; i < keep; i++)
		sub += w[order[i]];

	let out = { ids: [], probs: [] };

	for (let i = 0; i < keep; i++) {
		push(out.ids, idxs[order[i]]);
		push(out.probs, w[order[i]] / sub);
	}

	return out;
}

let V = 248320, worst = 0, cases = 0;

for (let trial = 0; trial < 4; trial++) {
	let logits = torch.arange(V, 'float32', 'cuda:0').mul(0.37 + 0.11 * trial).sin()
		.mul(6.0 + 6 * trial).to('float16').view([1, 1, V]);
	let counts = {}, ct = torch.zeros([V], 'float32', 'cuda:0');
	let top = logits.reshape([-1]).topk(50, 0).indices.tolist();

	for (let i = 0; i < 50; i += 3) {
		counts[top[i]] = 1 + i % 3;
		ct.narrow(0, top[i], 1).add_(1 + i % 3);
	}

	for (let cfg in [ { temperature: 0.7 }, { temperature: 0.7, top_p: 0.95 },
	                  { temperature: 1.0, top_k: 20 }, { temperature: 0.6, top_p: 0.9, min_p: 0.05 },
	                  { temperature: 0.7, top_p: 0.95, presence_penalty: 0.5 },
	                  { temperature: 0.8, top_p: 0.95, frequency_penalty: 0.3, presence_penalty: 0.2 },
	                  { temperature: 0.7, top_k: 40, top_p: 0.8, min_p: 0.1, presence_penalty: 1.0 } ]) {
		let a = dist_ref(logits, cfg, counts), b = engine.dist(logits, cfg, ct);

		cases++;
		assert(length(a.ids) == length(b.ids), `${cfg}: kept ${length(a.ids)} vs ${length(b.ids)}`);

		for (let i = 0; i < length(a.ids); i++) {
			assert(a.ids[i] == b.ids[i], `${cfg}: candidate ${i}: ${a.ids[i]} vs ${b.ids[i]}`);
			worst = max(worst, math.abs(a.probs[i] - b.probs[i]));
		}
	}
}

assert(worst < 1e-6, `probabilities differ (${worst})`);
assert(engine.dist(torch.zeros([1, 1, 8], 'float16', 'cuda:0'), { temperature: 0 }, null) == null,
	'temperature 0 must return null (greedy)');

printf('dist: %d cases match the reference (worst |dp| %.1e)\n', cases, worst);

// common_prefix
assert(torch.common_prefix([1, 2, 3, 4], [1, 2, 5]) == 2, 'common_prefix');
assert(torch.common_prefix([], [1]) == 0 && torch.common_prefix([7, 8], [7, 8]) == 2, 'common_prefix edges');

print('sampling_dist: PASS\n');
