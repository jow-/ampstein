// Test: torch.gdn_scan with tiled value heads (the order of a GGUF checkpoint converted by llama.cpp: value head
// j = r * Hk + k belongs to key head j % Hk) against the grouped order (j = k * R + r, key head j / R) on the same
// data with the value heads permuted. Synthetic data, no model.
//
//   ucode -L <build-dir> -L <src>/ucode tests/gdn_scan_tiled.uc

import * as math from 'math';

let seed = 4242;

function rnd() {
	seed = (seed * 1103515245 + 12345) % 2147483648;

	return (seed / 1073741824.0) - 1.0;
}

function tensor(shape, lo, hi) {
	let n = 1;

	for (let i = 0; i < length(shape); i++)
		n *= shape[i];

	let vals = [];

	for (let i = 0; i < n; i++)
		push(vals, lo + (rnd() + 1.0) * 0.5 * (hi - lo));

	return torch.from_array(vals).view(shape).cuda(0);
}

for (let dk in [ 64, 128 ]) {
	let S = 29, Hk = 4, R = 3, Hv = Hk * R, dv = 128, scale = 1.0 / math.sqrt(dk);
	let q = tensor([ S, Hk, dk ], -1, 1), k = tensor([ S, Hk, dk ], -1, 1);
	let v = tensor([ S, Hv, dv ], -1, 1), beta = tensor([ S, Hv ], 0.1, 0.9), gexp = tensor([ S, Hv ], 0.8, 0.99);
	let st = tensor([ Hv, dk, dv ], -0.5, 0.5);

	let perm_id = [];

	for (let j = 0; j < Hv; j++)
		push(perm_id, j);

	// perm[j_tiled] = j_grouped: tiled head r * Hk + kk is grouped head kk * R + r
	let perm = [];

	for (let j = 0; j < Hv; j++)
		push(perm, (j % Hk) * R + int(j / Hk));

	// (heads along dim `dim`, reordered: result head i = t's head order[i])
	function pick(t, dim, order) {
		let parts = [];

		for (let i in order)
			push(parts, t.narrow(dim, i, 1));

		return torch.cat(parts, dim).contiguous();
	}

	// grouped data = tiled data with the value heads gathered into the grouped order: grouped head g is tiled head inv[g]
	let inv = [];

	for (let g = 0; g < Hv; g++)
		inv[perm[g]] = g;

	let out_t = torch.gdn_scan(q, k, v, beta, gexp, pick(st, 0, perm_id), scale, null, true);
	let out_g = torch.gdn_scan(q, k, pick(v, 1, inv), pick(beta, 1, inv), pick(gexp, 1, inv), pick(st, 0, inv), scale);

	// the tiled output head j equals the grouped output head perm-inverse of j
	let d = out_t.sub(pick(out_g, 1, perm)).abs().max().item();

	printf('gdn_scan tiled dk=%d: max |tiled - permuted grouped| = %g\n', dk, d);
	assert(d < 1e-5, `gdn_scan tiled dk=${dk} differs from the grouped scan by ${d}`);
}

print('gdn_scan_tiled: ok\n');
