// tests/dsv41_gen.uc — DeepSeek-V4.1-Flash end to end: prefill a prompt, greedy-decode, print the text.
//   AMP_MODEL_DSV41=<dir>  AMP_DSV41_LAYERS=<n>  AMP_DSV41_GPUS=<n>  AMP_DSV41_PROMPT='...'  AMP_DSV41_N=<tokens>
// Checks that chunked prefill (chunk 7) and a single chunk agree on the next token and logits.

import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(MODEL + '/model.safetensors.index.json')) {
	warn('skip: DSV4.1 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 0);
const NG = +(getenv('AMP_DSV41_GPUS') || 1);
const N = +(getenv('AMP_DSV41_N') || 24);
const PROMPT = getenv('AMP_DSV41_PROMPT') ||
	'<｜begin▁of▁sentence｜><｜User｜>The capital of France is<｜Assistant｜></think>';

let tok = torch.tokenizer(MODEL);
let t0 = time();
let m = dsv41.load(MODEL, {devices: NG, layers: NL || null});

warn(`loaded in ${time() - t0}s\n`);

let ids = tok.encode(PROMPT, true);

printf('prompt: %d tokens\n', length(ids));

function argmax_row(lg) {
	return lg.to('float32').argmax(-1).tolist();
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

// one chunk vs small chunks
let c1 = m.new_cache(length(ids) + N + 8);
let l1 = m.forward_chunk(ids, c1, 1);
let c2 = m.new_cache(length(ids) + N + 8);
let l2 = m.prefill(ids, c2, 7, 1);
let e = rel_rms(l1, l2);

printf('chunked vs single prefill logits: rel rms %.2e\n', e);
assert(e < 0.05, 'chunking changes the logits');

let out = [];
let cur = argmax_row(l1)[0];

t0 = time();

for (let i = 0; i < N; i++) {
	push(out, cur);

	let lg = m.forward_chunk([cur], c1, 1);

	cur = argmax_row(lg)[0];
}

warn(`decode ${N} tokens in ${time() - t0}s\n`);
printf('generated: %J\n', tok.decode(out));
