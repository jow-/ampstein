// tests/dsv41_gguf.uc — DeepSeek-V4.1 from GGUF files (ggufw.uc, ggufq.uc): the first layers of the model, one prompt through the
// chunked prefill (routed experts: tokens sorted by expert, dequantized matrices through cuBLAS) and through single-token steps
// (the GEMV kernels): the same logits up to fp16 / int8-activation rounding and the same greedy token.
//   AMP_MODEL_DSV41_GGUF  (a directory with the .gguf shards + config.json + tokenizer.json; the Engram tables from the shards
//                          of an EXL3 pack symlinked in, or AMP_DSV41_NO_ENGRAM=1)
//   AMP_DSV41_LAYERS (default 4)  AMP_DSV41_GPUS (default 1)  AMP_DSV41_PREFILL (default 48)

import * as dsv41 from 'ampstein.dsv41';
import * as ggufw from 'ampstein.ggufw';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41_GGUF') || '/data/models/DSV4.1-Flash-1017-GSQ-RCO-IQ2_XXS';

if (!stat(MODEL + '/config.json') || !ggufw.is_gguf_dir(MODEL)) {
	warn('skip: DSV4.1 GGUF model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 4);
const NG = +(getenv('AMP_DSV41_GPUS') || 1);
const NP = +(getenv('AMP_DSV41_PREFILL') || 48);

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: NP + 64, prefill_rows: 64});
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
let la = m.forward_chunk(ids, ca, 1), lb = null;

for (let i = 0; i < NP; i++)
	lb = m.forward_chunk([ ids[i] ], cb, 1);

let e = rel_rms(lb, la);
let aa = la.argmax(-1).tolist()[0], ab = lb.argmax(-1).tolist()[0];

printf('chunked prefill vs token-by-token steps: logits rel rms %.2e, argmax %d %d\n', e, aa, ab);
assert(e < 0.05, 'chunked prefill and single-token steps differ');
assert(aa == ab, 'greedy token differs');

// a step after both caches: the same
let tok = aa;
let sa = m.forward_chunk([ tok ], ca, 1), sb = m.forward_chunk([ tok ], cb, 1);

printf('next step: rel rms %.2e\n', rel_rms(sb, sa));
assert(rel_rms(sb, sa) < 0.05, 'the step after differs');

printf('dsv41_gguf: all ok\n');
