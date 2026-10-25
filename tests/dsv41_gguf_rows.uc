// tests/dsv41_gguf_rows.uc — multi-row (speculative verify) steps through the decode graphs when only some sizes are captured: S rows through
// the graph of the next larger size (the surplus rows are taken back with rollback) give the logits of an eager S-row step, leave the cache at
// pos + S and do not disturb the steps after.
//   AMP_MODEL_DSV41_GGUF  AMP_DSV41_LAYERS (default 4)  AMP_DSV41_GPUS (default 1)

import * as dsv41 from 'ampstein.dsv41';
import * as ggufw from 'ampstein.ggufw';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41_GGUF') || '/data/models/DSV4.1-Flash-1017-GSQ-RCO-IQ2_XXS';

if (!stat(MODEL + '/config.json') || !ggufw.is_gguf_dir(MODEL)) {
	warn('skip: DSV4.1 GGUF model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 4), NG = +(getenv('AMP_DSV41_GPUS') || 1);

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: 1024, prefill_rows: 64});
let ids = [], x = 99;

for (let i = 0; i < 100; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let cb = m.new_cache(1024);
let g = m.graphs(cb, [ 1, 2, 3, 4, 5 ]);                 // (a GGUF model captures 1, 3 and 5 rows; AMP_DSV41_ROW_SIZES overrides)

m.aux_reset(cb);
cb.length = 0;
m.prefill(ids, cb, 64, 0);

assert(g.has(2) && g.has(3) && g.has(4) && g.has(5) && !g.has(6), 'has() covers 2 .. the largest captured size');

let toks = [ 4242, 1717, 2323, 3131, 777 ];
let P = cb.length, snap = m.aux_get(cb);

// from the same state: the rows of the exact-size graph, then S rows through a larger graph (S = 2 -> 3, S = 4 -> 5)
function from_state(S) {
	cb.length = P;
	m.aux_set(cb, snap);

	return g.rows(slice(toks, 0, S), P);
}

let full5 = from_state(5), full3 = from_state(3);
let worst = 0;

for (let S in [ 2, 4, 3, 5 ]) {
	let r = from_state(S);
	let ref = (S <= 3) ? full3 : full5;
	let e = rel_rms(r.logits, ref.logits.narrow(0, 0, S));

	printf('%d rows (%s): logits rel rms %.2e vs the exact graph, cache length %d (want %d)\n', S, (S == 3 || S == 5) ? 'own graph' : 'padded', e, cb.length, P + S);
	assert(cb.length == P + S, 'cache length after the rows');
	assert(r.logits.shape()[0] == S && r.am.shape()[0] == S && (r.taps == null || r.taps.shape()[0] == S), 'S rows come back');
	assert(e < 0.02, 'the padded rows differ from the exact-size graph');
	assert(join(',', r.am.tolist()) == join(',', ref.am.narrow(0, 0, S).tolist()), 'argmax of the rows');
	worst = max(worst, e);

}

// the step after S padded rows equals the step after the same S rows through the exact graph (only S = 3 has both)
{
	cb.length = P;
	m.aux_set(cb, snap);
	g.rows([ toks[0], toks[1] ], P);                // padded to 3

	let a = g.run([ 55 ], cb.length).logits.to('float32').cpu();

	cb.length = P;
	m.aux_set(cb, snap);
	g.rows(slice(toks, 0, 3), P);                   // exact graph, then take the third row back
	m.rollback(cb, P + 2);

	let b = g.run([ 55 ], cb.length).logits.to('float32').cpu();

	printf('the step after 2 rows: padded vs exact-then-rollback rel rms %.2e\n', rel_rms(a, b));
	assert(rel_rms(a, b) < 0.02, 'the step after the padded rows differs');
}

printf('dsv41_gguf_rows: all ok (worst %.2e)\n', worst);
