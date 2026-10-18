// tests/dsv41_verify.uc — multi-row decode graphs (the speculative verify step) against single-token decode:
// S rows through graph.rows() give the logits of S eager steps, and after a rollback the next step
// continues exactly like an eager run that never saw the rejected rows.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 8)  AMP_DSV41_GPUS (default 2)  AMP_DSV41_PREFILL (default 4093)

import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(MODEL + '/model.safetensors.index.json')) {
	warn('skip: DSV4.1 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 8);
const NG = +(getenv('AMP_DSV41_GPUS') || 2);
const NP = +(getenv('AMP_DSV41_PREFILL') || 4093);
const S = 5;
const MAXSEQ = NP + 64;

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: MAXSEQ, prefill_rows: 512});
let ids = [], x = 12345;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let ca = m.new_cache(MAXSEQ), cb = m.new_cache(MAXSEQ), cc = m.new_cache(MAXSEQ);

m.prefill(ids, ca, 512, 0);
m.prefill(ids, cc, 512, 0);

let g = m.graphs(cb, [ 1, S, 3 ]);

m.aux_reset(cb);
cb.length = 0;
m.prefill(ids, cb, 512, 0);

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let toks = [ 4242, 777, 12345, 999, 31337 ];
let worst = 0;

// 1. S rows at once against S eager steps
let r = g.rows(toks, NP);

assert(cb.length == NP + S, 'rows() leaves cache.length at pos + S');

for (let k = 0; k < S; k++) {
	let la = m.forward_chunk([ toks[k] ], ca, 1);
	let e = rel_rms(r.logits.narrow(0, k, 1), la);
	let ta = la.to('float32').argmax(-1).tolist()[0];

	printf('row %d pos %d: logits rel rms %.2e  argmax %d %d\n', k, NP + k, e, ta, r.am.tolist()[k]);
	worst = max(worst, e);
}

// the noise floor of the eager comparison grows with the pool (top-512 boundary of the index scores): 0.05 up to
// 17k positions, 0.1 beyond
assert(worst < (NP > 20000 ? 0.1 : 0.05), 'multi-row decode differs from single-token decode');

// 2. keep 3 rows, continue with a different token: like an eager run that only saw those 3
m.rollback(cb, NP + 3);
assert(cb.length == NP + 3, 'rollback sets the length');

for (let k = 0; k < 3; k++)
	m.forward_chunk([ toks[k] ], cc, 0);

let alt = 55555;
let lc = m.forward_chunk([ alt ], cc, 1);
let lb = g.run([ alt ], cb.length).logits;
let e2 = rel_rms(lb, lc);

printf('after rollback: logits rel rms %.2e\n', e2);
assert(e2 < (NP > 20000 ? 0.1 : 0.05), 'decode after a rollback differs');

// 3. a 3-row step right after (pair parity flips)
let r3 = g.rows([ 11, 22, 33 ], cb.length);
let l3 = [];

for (let t in [ 11, 22, 33 ])
	push(l3, m.forward_chunk([ t ], cc, 1));

for (let k = 0; k < 3; k++) {
	let e = rel_rms(r3.logits.narrow(0, k, 1), l3[k]);

	printf('3-row step, row %d: rel rms %.2e\n', k, e);
	assert(e < (NP > 20000 ? 0.1 : 0.05), 'second multi-row step differs');
}

// 4. the same rows one by one through the single-row GRAPH: only the batching differs, so it must agree closely
{
	let pos4 = cb.length;
	let toks4 = [ 8001, 9002, 10003, 11004, 12005 ];
	let r4 = g.rows(toks4, pos4);
	let rows_logits = r4.logits.to('float32').add(0.0);

	m.rollback(cb, pos4 + 1);

	for (let k = 1; k < S; k++) {
		let l1 = g.run([ toks4[k] ], cb.length).logits;
		let e1 = rel_rms(rows_logits.narrow(0, k, 1), l1);

		printf('row %d vs single-row graph: rel rms %.2e\n', k, e1);
		assert(e1 < 0.1, 'batched rows differ from single-row graph steps');     // the eager comparisons above are the tight ones
	}
}

printf('dsv41_verify: all ok\n');
