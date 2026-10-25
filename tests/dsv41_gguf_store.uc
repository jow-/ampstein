// tests/dsv41_gguf_store.uc — the host-RAM prompt-cache store of the single-chain model (cache_export / cache_import): a conversation exported to host
// memory, the cache then used for an unrelated prompt, and the conversation imported again continues with the logits it had before.
//   AMP_MODEL_DSV41_GGUF  AMP_DSV41_LAYERS (default 8)  AMP_DSV41_GPUS (default 2)  AMP_DSV41_PREFILL (default 300)

import * as dsv41 from 'ampstein.dsv41';
import * as ggufw from 'ampstein.ggufw';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41_GGUF') || '/data/models/DSV4.1-Flash-1017-GSQ-RCO-IQ2_XXS';

if (!stat(MODEL + '/config.json') || !ggufw.is_gguf_dir(MODEL)) {
	warn('skip: DSV4.1 GGUF model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 8), NG = +(getenv('AMP_DSV41_GPUS') || 2), NP = +(getenv('AMP_DSV41_PREFILL') || 300);
const MAXSEQ = NP + 64;

let m = dsv41.load(MODEL, {devices: NG, layers: NL, max_seq: MAXSEQ, prefill_rows: 128});

function prompt(seed, n) {
	let ids = [], x = seed;

	for (let i = 0; i < n; i++) {
		x = (x * 1103515245 + 12345) % 2147483648;
		push(ids, 1000 + (x % 60000));
	}

	return ids;
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let cache = m.new_cache(MAXSEQ);

m.prefill(prompt(12345, NP), cache, 128, 1);

let em = m.cache_export(cache, NP, (t) => t.cpu(), null);
let toks = [ 4242, 777, 12345, 999 ];
let ref = [];

for (let t in toks)
	push(ref, m.forward_chunk([ t ], cache, 1).to('float32').cpu());

// an unrelated prompt replaces the conversation
m.aux_reset(cache);
cache.length = 0;
m.prefill(prompt(777, NP - 50), cache, 128, 1);

m.cache_import(cache, em, NP, true);

let worst = 0;

for (let i = 0; i < length(toks); i++) {
	let l = m.forward_chunk([ toks[i] ], cache, 1).to('float32').cpu();
	let e = rel_rms(l, ref[i]);

	printf('step %d: logits rel rms %.2e  argmax %d %d\n', i, e, l.argmax(-1).tolist()[0], ref[i].argmax(-1).tolist()[0]);
	worst = max(worst, e);
}

assert(worst < 1e-3, 'an imported conversation continues differently');
printf('dsv41_gguf_store: all ok (%d bytes exported)\n', m.cache_bytes(cache, NP));
