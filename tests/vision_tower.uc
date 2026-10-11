// tests/vision_tower.uc — Qwen3.8 vision tower vs exllamav3.
// Fixture: tests/fixtures/vision_oracle.safetensors + vision_test.png
// (tests/oracle/make_vision_oracle.py). Input is the oracle's own
// preprocessed patches (so tower numerics are isolated from the
// preprocessing, which vision_preprocess.uc locks bit-exactly), then the
// full path from the PNG bytes.
//
// The reference runs fp32 intermediates where ours is fp16: expect
// rel-RMS ~1e-3, not bit-exactness.

import * as vision from 'ampstein.qwen35_vision';
import * as math from 'math';
import { stat, readfile } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';
const DIR = `${sourcepath(0, true)}/fixtures`;

if (!stat(MODEL_DIR) || !stat(`${DIR}/vision_oracle.safetensors`)) {
	warn('skip: model dir or vision fixtures missing\n');
	exit(125);
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));
	let num = d.mul(d).mean().item();
	let den = b.to('float32').mul(b.to('float32')).mean().item();

	return math.sqrt(num / den);
}

let ref = torch.st(`${DIR}/vision_oracle.safetensors`);
let patches = ref.tensor('patches');
let grid = ref.tensor('grid').tolist();
let want_b0 = ref.tensor('block0').cuda(0);
let want = ref.tensor('emb').cuda(0);

let v = vision.load(MODEL_DIR, { device: 0 });

assert(v.cfg.depth == 27 && v.cfg.out_hidden_size == 5120, 'config');
assert(vision.n_tokens(v.cfg, grid) == want.shape()[0], 'n_tokens');

let trace = [];
let emb = v.forward(patches, grid, trace);

assert(emb.dtype() == 'float16', 'output dtype');
assert(emb.shape()[0] == want.shape()[0] && emb.shape()[1] == 5120, 'shape');

let e0 = rel_rms(trace[0], want_b0);
let e = rel_rms(emb, want);
let cosv = emb.to('float32').mul(want.to('float32')).sum().item() /
	math.sqrt(emb.to('float32').pow(2).sum().item() *
	          want.to('float32').pow(2).sum().item());

printf('block0 rel-rms %g, final rel-rms %g, cosine %g\n', e0, e, cosv);
assert(e0 < 0.01, 'block 0 matches the oracle');
assert(e < 0.05, 'embeddings match the oracle');
assert(cosv > 0.999, 'embedding direction');

/* full path from the PNG bytes (native preprocessing is bit-exact, so
 * the result must be identical to the patches path) */
let r = v.embed(readfile(`${DIR}/vision_test.png`));

assert(r.n_tokens == want.shape()[0], 'embed(): token count');
assert(rel_rms(r.emb, emb) == 0, 'embed() == forward(oracle patches)');

print('vision_tower: ok\n');
