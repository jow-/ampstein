// tests/dsv41_graph.uc — CUDA-graph decode (model.graphs) against eager decode: the same logits for a
// run of steps that crosses a pool bucket and a pair boundary of the ratio-2 compressors.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 8)  AMP_DSV41_GPUS (default 2)  AMP_DSV41_PREFILL (default 4090)
//   AMP_DSV41_STEPS (default 12)

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
const NP = +(getenv('AMP_DSV41_PREFILL') || 4090);
const NS = +(getenv('AMP_DSV41_STEPS') || 12);
const MAXSEQ = +(getenv('AMP_DSV41_MAXSEQ') || (NP + NS + 64));

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: MAXSEQ, prefill_rows: 512});
let ids = [], x = 12345;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let ca = m.new_cache(MAXSEQ);
let cb = m.new_cache(MAXSEQ);

m.prefill(ids, ca, 512, 0);
m.prefill(ids, cb, 512, 0);

let g = m.graphs(cb, [ 1 ]);

// graph warm-up reset the cache: prefill again
m.aux_reset(cb);
cb.length = 0;
m.prefill(ids, cb, 512, 0);

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let tok = 4242;
let worst = 0;

for (let i = 0; i < NS; i++) {
	let la = m.forward_chunk([ tok ], ca, 1);
	let lb = g.run([ tok ], cb.length).logits;
	let e = rel_rms(lb, la);
	let ta = la.to('float32').argmax(-1).tolist()[0], tb = lb.to('float32').argmax(-1).tolist()[0];

	printf('step %d pos %d: logits rel rms %.2e  argmax %d %d\n', i, NP + i, e, ta, tb);
	worst = max(worst, e);
	tok = ta;
}

assert(worst < 0.05, 'graph decode differs from eager decode');
printf('dsv41_graph: all ok\n');
