// tests/dsv4_store_state.uc — DeepSeek-V4 host-RAM store round trip at the model level (no
// engine): a cache runs A, exports its state (live, and the snapshot aux state at SNAP), is
// contaminated by a long other conversation B, and imports it back (live and snapshot reuse
// points). The restored state and its next-token logits must be as close to a fresh prefill
// as two fresh prefills are to each other: with AMP_MOE_ATOMIC set the prefill is not
// bit-deterministic (fp noise of a few 1e-3 in the early layers growing to ~20 at the
// massive-activation rows), so the bound is the measured fresh-vs-fresh noise floor; by default
// the floor is exactly zero and the restored state must match to the bit.
import * as dsv4 from 'ampstein.dsv4';
import * as dsv4_tp from 'ampstein.dsv4_tp';

import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const BOS = '<｜begin▁of▁sentence｜>';

if (!stat(MODEL) || torch.device_count() < 6) {
	warn('skip: needs the DSV4 model and 6 GPUs\n');
	exit(125);
}

let tok = torch.tokenizer(MODEL);
let model = dsv4_tp.load(MODEL, { layers: null, devices: 6, stages: 3, drafter: false, prefill_rows: 512 });

function text(seed, n) {
	let s = BOS;

	for (let i = 0; i < n; i++)
		s += `Item ${seed}-${i}: the ${seed} committee reviewed record number ${i * 7 + length(seed)} and noted nothing unusual. `;

	return s;
}

let A = tok.encode(text('alpha', 150), true);
let B = tok.encode(text('bravo', 150), true);
let TAIL = tok.encode(' Summarize the findings in one sentence:', false);
let SNAP = length(A) - 20;

function prefill(cache, ids) {
	for (let o = 0; o < length(ids); o += 512)
		model.forward_chunk(slice(ids, o, min(o + 512, length(ids))), cache, 0);
}   // (continues at cache.length)

function put(t) {
	let h = torch.pinned_empty(t.shape(), t.dtype());

	h.copy_(t);

	return h;
}

function mdiff(a, b) {
	if (a == null && b == null)
		return 0;

	if (a == null || b == null)
		return 'NULL-MISMATCH';

	return a.to('float32').sub(b.to('float32')).abs().max().item();
}

// compare two caches tensor by tensor (aux_get entries + pools + kw)
function cmp(label, X, Y) {
	let auxr = model.aux_get(X), auxc = model.aux_get(Y);
	let bad = 0, worst = 0, mism = 0, byL = {};

	function note(layer, v) {
		if (type(v) == 'string')
			mism++;

		if (type(v) == 'string' || v > 0) {
			bad++;
			byL[layer] = (type(v) == 'string') ? 1e9 : (v > (byL[layer] || 0) ? v : byL[layer]);
		}

		if (type(v) != 'string' && v > worst)
			worst = v;
	}

	for (let i = 0; i < length(X.layers); i++)
		for (let g = 0; g < length(X.layers[i]); g++) {
			let r = X.layers[i][g], x = Y.layers[i][g];

			for (let k = 0; k < 2; k++) {
				let ar = auxr[i][g][k], ax = auxc[i][g][k];

				if ((ar == null) != (ax == null)) {
					note(i, 'NULL-MISMATCH');
					continue;
				}

				if (ar == null)
					continue;

				if (ar.n != ax.n)
					note(i, 'NULL-MISMATCH');

				for (let f in [ 'bk', 'bg', 'ok', 'og' ])
					note(i, mdiff(ar[f], ax[f]));

				let Sr = k ? r.idx : r.comp, Sx = k ? x.idx : x.comp;

				if (ar.n > 0)
					note(i, mdiff(Sr.pool.narrow(0, 0, ar.n), Sx.pool.narrow(0, 0, ar.n)));
			}

			note(i, mdiff(auxr[i][g][2].kw, auxc[i][g][2].kw));
		}

	let w = [];

	for (let i = 0; i < length(X.layers); i += 6)
		push(w, sprintf('L%d:%.3g', i, byL[i] || 0));

	printf('%s: %d differing tensors, worst %g; per-layer worst %s\n', label, bad, worst, join(' ', w));

	return { worst: worst, mismatch: mism };
}


let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + (detail != null ? ' (' + detail + ')' : '') + '\n');
}

// the cache under test: run A with the snapshot aux state taken at SNAP, export both
let c = model.new_cache(8192);

prefill(c, slice(A, 0, SNAP));

let snap_aux = model.aux_get(c);

prefill(c, slice(A, SNAP));

let em = model.cache_export(c, length(A), put, snap_aux);

torch.synchronize();

// bit-exactness: a restored state must equal the state that was exported, bit for bit, however
// much was overwritten in between (this is independent of the prefill noise measured below).
// Snapshot the originals first: the live state and the snapshot aux state with their pool rows.
function copy_aux(aux) {
	return map(aux, (row) => map(row, (e) => map(e, function (x) {
		if (x == null)
			return null;

		let o = {};

		for (let k in x)
			o[k] = (k == 'n' || x[k] == null) ? x[k] : x[k].add(0.0);

		return o;
	})));
}

function copy_pools(cache) {
	return map(cache.layers, (row) => map(row, (cl) => map([ cl.comp, cl.idx ],
		(S) => (S && S.n > 0) ? S.pool.narrow(0, 0, S.n).add(0.0) : null)));
}

let live_aux = copy_aux(model.aux_get(c));       // c is at the exported live state here
let live_pools = copy_pools(c);

function exact(name, cache, aux, pools, rowsof) {
	let now = model.aux_get(cache), bad = 0, total = 0;

	for (let i = 0; i < length(cache.layers); i++)
		for (let g = 0; g < length(cache.layers[i]); g++) {
			let cl = cache.layers[i][g];

			for (let k = 0; k < 2; k++) {
				let a = aux[i][g][k], x = now[i][g][k];

				if ((a == null) != (x == null) || (a && a.n != x.n)) {
					bad++;
					continue;
				}

				if (!a)
					continue;

				for (let f in [ 'bk', 'bg', 'ok', 'og' ]) {
					total++;

					if (mdiff(a[f], x[f]) != 0)
						bad++;
				}

				let S = k ? cl.idx : cl.comp, P = pools[i][g][k];

				total++;

				if (a.n > 0 && mdiff(P.narrow(0, 0, a.n), S.pool.narrow(0, 0, a.n)) != 0)
					bad++;
			}

			total++;

			if (mdiff(aux[i][g][2].kw, now[i][g][2].kw) != 0)
				bad++;
		}

	check(`${name}: restored state is bit-identical to the exported one`, bad == 0, `${bad} of ${total} tensors differ`);
}

// the live state: contaminate, import, compare with the copies taken above
model.aux_reset(c);
c.length = 0;
prefill(c, B);
model.cache_import(c, em, length(A), true);
torch.synchronize();
exact('live state', c, live_aux, live_pools);

// the snapshot state: its aux is the snapshot's; pools below its entry counts equal the live ones
model.aux_reset(c);
c.length = 0;
prefill(c, B);
model.cache_import(c, em, SNAP, false);
torch.synchronize();

let snap_copy = copy_aux(snap_aux);
let snap_pools = map(live_pools, (row) => row);
exact('snapshot', c, snap_copy, snap_pools);

for (let live in [ true, false ]) {
	let L = live ? length(A) : SNAP;
	let name = live ? 'live state' : 'snapshot';
	let ref = model.new_cache(8192), ref2 = model.new_cache(8192);

	// the same chunking as the state under test (A[:SNAP], then the rest): the prefill numerics
	// depend on where the chunks are cut, so a uniform split is not the same computation
	prefill(ref, slice(A, 0, min(L, SNAP)));
	prefill(ref2, slice(A, 0, min(L, SNAP)));

	if (L > SNAP) {
		prefill(ref, slice(A, SNAP, L));
		prefill(ref2, slice(A, SNAP, L));
	}

	// contaminate: the long other conversation overwrites ring slots and pool rows
	model.aux_reset(c);
	c.length = 0;
	prefill(c, B);
	model.cache_import(c, em, L, live);
	torch.synchronize();

	let floor = cmp(`${name}: fresh vs fresh`, ref, ref2);
	let got = cmp(`${name}: restored vs fresh`, ref, c);
	let l1 = model.forward_chunk(TAIL, ref, 1);
	let l2 = model.forward_chunk(TAIL, ref2, 1);
	let l3 = model.forward_chunk(TAIL, c, 1);
	let dfl = mdiff(l1, l2), dgl = mdiff(l1, l3);

	printf('%s: next-token logits: fresh vs fresh %g, restored vs fresh %g (max |logit| %g)\n', name, dfl, dgl,
		l1.to('float32').abs().max().item());

	check(`${name}: no tensor is missing or has a different entry count`, got.mismatch == 0, `${got.mismatch}`);
	check(`${name}: restored state within the fresh-vs-fresh noise (x2 + 0.1)`, got.worst <= 2 * floor.worst + 0.1,
		sprintf('%g vs floor %g', got.worst, floor.worst));
	check(`${name}: next-token logits within the noise (x2 + 0.1)`, dgl <= 2 * dfl + 0.1,
		sprintf('%g vs floor %g', dgl, dfl));
}

if (fails)
	die(`${fails} check(s) failed`);

print('dsv4_store_state: ok\n');
