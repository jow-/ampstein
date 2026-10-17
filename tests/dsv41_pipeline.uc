// tests/dsv41_pipeline.uc — the wavefront prefill (model.prefill_pipeline, one stage per chunk per round) against the
// sequential chunked prefill: the same logits for the last token and the same next steps.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 40)  AMP_DSV41_GPUS (default 6)  AMP_DSV41_PREFILL (default 2500)

import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(MODEL + '/model.safetensors.index.json')) {
	warn('skip: DSV4.1 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 40);
const NG = +(getenv('AMP_DSV41_GPUS') || 6);
const NP = +(getenv('AMP_DSV41_PREFILL') || 2500);

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: NP + 64, prefill_rows: 512});
let ids = [], x = 4321;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let ca = m.new_cache(NP + 64), cb = m.new_cache(NP + 64);
let la = m.prefill(ids, ca, 512, 1);
let run = m.prefill_pipeline(cb, ids, {chunk: 512});

while (run.step(1)) {}

let lb = run.logits();
let e = rel_rms(lb, la);

printf('wavefront vs sequential prefill: logits rel rms %.2e (cache lengths %d %d)\n', e, ca.length, cb.length);
assert(e < 0.01 && ca.length == cb.length, 'pipelined prefill differs');

let tok = 77;

for (let i = 0; i < 4; i++) {
	let a = m.forward_chunk([ tok ], ca, 1), b = m.forward_chunk([ tok ], cb, 1);
	let ee = rel_rms(b, a);

	printf('step %d: rel rms %.2e\n', i, ee);
	assert(ee < 0.01, 'decode after the pipelined prefill differs');
	tok = a.to('float32').argmax(-1).tolist()[0];
}

printf('dsv41_pipeline: all ok\n');
