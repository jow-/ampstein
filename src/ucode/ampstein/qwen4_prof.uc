// ucode/qwen4_prof.uc — routing profile of the GGUF Qwen3.8-Flash-Next experts (which experts stay in VRAM).
//
// The resident set of a layer is the experts its router picks most. A profile can be given (--expert-profile), but none is
// required: while the server runs, every layer counts its router's picks on the device (one index_add_ per layer and step,
// inside the decode graphs too), and the counts are written next to the model now and then (<model>/expert-profile.json,
// --expert-profile-save). The next start places the experts by them. The running server keeps its set: exchanging experts
// between VRAM and host memory while decoding costs more than the misses it saves (docs/QWEN4.md).
//
// File formats read: {"format": "counts", "tokens": N, "counts": [[...E counts] per layer]} (what is written; the counts
// carry over from run to run) and the older {"<layer>": [expert ids, most picked first]} (ranking only).

import {readfile, writefile, rename} from 'fs';

const DECODE_ROWS = 64;                     // rows of a graph-captured step (the ones buffer of a layer is fixed: graphs keep its address)
const MAX_TOKENS = 4000000;                 // counts are halved when they cover more tokens than this (a changing workload wins in the end)

function order_of(counts) {
	let ids = [];

	for (let e = 0; e < length(counts); e++)
		push(ids, e);

	sort(ids, (a, b) => (counts[b] - counts[a]) || (a - b));

	return ids;
}

/* The profile in `path`: {order: {layer: [ids, most picked first]}, counts: [[...] per layer] | null, tokens}, or null. */
export function read(path) {
	let raw;

	try {
		raw = json(readfile(path));
	}
	catch (e) {
		return null;
	}

	if (type(raw) != 'object')
		return null;

	let order = {}, counts = null, tokens = 0;

	if (raw.format == 'counts' && type(raw.counts) == 'array') {
		counts = raw.counts;
		tokens = raw.tokens ?? 0;

		for (let l = 0; l < length(counts); l++)
			order[l] = order_of(counts[l]);
	}
	else {
		for (let l, ids in raw)
			if (type(ids) == 'array')
				order[+l] = ids;
	}

	return {order: order, counts: counts, tokens: tokens};
}

/* seed: the counts of an earlier run or null. */
export function create(E, K, seed, tokens) {
	let layers = {};                         // layer -> {c: counts (E) float32 on the layer's device, ones: (DECODE_ROWS * K)}
	let ones_of = {};                        // device -> ones buffer, shared by its layers
	let base_tokens = tokens ?? 0;

	function ones_for(dev) {
		ones_of[dev] ??= torch.ones([DECODE_ROWS * K], 'float32', 'cuda:' + dev);

		return ones_of[dev];
	}

	return {
		/* the function a layer's moe_forward calls with the picks (rows, K) of every step */
		hook: function (layer, dev) {
			let c = (seed && seed[layer] && length(seed[layer]) == E) ?
				torch.from_array(seed[layer], 'float32', 'cuda:' + dev) : torch.zeros([E], 'float32', 'cuda:' + dev);
			let ones = ones_for(dev);

			layers[layer] = {c: c, dev: dev};

			return (idx) => {
				let f = idx.reshape([-1]);
				let n = f.shape()[0];

				// (a prefill chunk is eager: its own buffer; the decode graphs use the fixed one)
				c.index_add_(0, f, (n <= DECODE_ROWS * K) ? ones.narrow(0, 0, n) : torch.ones([n], 'float32', 'cuda:' + dev));
			};
		},

		tokens: function () {
			for (let l, s in layers)
				return base_tokens + int(s.c.sum().item() / K);

			return 0;
		},

		/* Put the experts the router picked most into VRAM, in exchange for the least picked resident ones of the same layer, as far as
		 * an exchange pays back: Ms[layer] the layer's moe object, swap(M, cold expert, resident expert) the exchange, sync(M) after a
		 * layer's batch. opts: {min_rate: picks per token an expert must gain over the one it replaces (0.01: an exchange costs about
		 * a millisecond, a miss ~85 us, so it pays back within ~1200 tokens), max_ms: time budget, decay: factor the counts are
		 * multiplied with afterwards (the placement follows the recent workload), min_tokens: layers that counted fewer are left alone}. Returns {swaps, ms, gain: expected fewer misses per token}. */
		rebalance: function (Ms, swap, sync, opts) {
			opts ??= {};

			let min_rate = opts.min_rate ?? 0.01, max_ms = opts.max_ms ?? 1500, decay = opts.decay ?? 0.7;
			let plan = [], n = length(layers);

			for (let l = 0; l < n; l++) {
				let M = Ms[l];

				if (!M || !layers[l] || !M.gq_cold)
					continue;

				let c = layers[l].c.cpu().tolist(), tot = 0;

				for (let v in c)
					tot += v;

				// (rates from a few tokens are noise)
				if (tot / K < (opts.min_tokens ?? 1000))
					continue;

				let R = M.gq.w1.shape()[0], order = order_of(c), want = [], vict = [];

				for (let i = 0; i < R; i++)
					if (M.res_of[order[i]] < 0)
						push(want, order[i]);

				for (let i = length(order) - 1; i >= R; i--)
					if (M.res_of[order[i]] >= 0)
						push(vict, order[i]);

				for (let i = 0; i < length(want) && i < length(vict); i++) {
					let gain = (c[want[i]] - c[vict[i]]) * K / tot;

					if (gain < min_rate)
						break;

					push(plan, {layer: l, a: want[i], b: vict[i], gain: gain});
				}
			}

			sort(plan, (x, y) => y.gain - x.gain);

			let t0 = clock(true), swaps = 0, gain = 0, touched = {};

			for (let p in plan) {
				let t = clock(true);

				if ((t[0] - t0[0]) * 1000 + (t[1] - t0[1]) / 1000000 > max_ms)
					break;

				if (swap(Ms[p.layer], p.a, p.b)) {
					swaps++;
					gain += p.gain;
					touched[p.layer] = true;
				}
			}

			for (let l in touched)
				sync(Ms[+l]);

			torch.synchronize();

			for (let l = 0; l < n; l++)
				if (layers[l])
					layers[l].c.mul_(decay);

			let t = clock(true);

			return {swaps: swaps, planned: length(plan), gain: gain, ms: int((t[0] - t0[0]) * 1000 + (t[1] - t0[1]) / 1000000)};
		},

		/* write the counts to `path` (through a temporary file); returns the number of tokens they cover */
		dump: function (path) {
			let counts = [], n = length(layers), tot = 0;

			for (let l = 0; l < n; l++) {
				if (!layers[l])
					return -1;

				counts[l] = layers[l].c.cpu().tolist();
			}

			for (let v in counts[0])
				tot += v;

			tot = int(tot / K);

			// (the seed's tokens are in the counts already; the live total is the sum)
			if (tot > MAX_TOKENS) {
				for (let l = 0; l < n; l++) {
					counts[l] = map(counts[l], (v) => v / 2);
					layers[l].c.mul_(0.5);
				}

				tot = int(tot / 2);
			}

			let tmp = path + '.tmp';

			writefile(tmp, sprintf('%J', {format: 'counts', experts: E, k: K, tokens: tot, counts: counts}));
			rename(tmp, path);

			return tot;
		}
	};
}
