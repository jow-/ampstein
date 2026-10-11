// tests/glm5_vision.uc — GLM-5.3-Flash vision tower vs exllamav3: the native
// canvas preprocessing (bit-exact patches), the residual after block 0 and
// the final merger embeddings (fixture: make_glm5_vision_oracle.py). Needs
// ~2 GiB on GPU GLM5_VIS_DEV (default 0).
import * as math from 'math';
import * as glm5_vision from 'ampstein.glm5_vision';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const DIR = sourcepath(0, true);
const FIX = `${DIR}/fixtures/glm5_vision_oracle.safetensors`;
const PNG = `${DIR}/fixtures/vision_test.png`;

if (!stat(MODEL) || !stat(FIX) || !stat(PNG)) {
	warn('skip: model, image or oracle fixture missing\n');
	exit(125);
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / b.to('float32').mul(b.to('float32')).mean().item());
}

function cosine(a, b) {
	let x = a.to('float32').reshape([-1]), y = b.to('float32').reshape([-1]);

	return x.mul(y).sum().item() / math.sqrt(x.mul(x).sum().item() * y.mul(y).sum().item());
}

let fx = torch.st(FIX);
let T = glm5_vision.load(MODEL, { device: +(getenv('AMP_GLM5_VIS_DEV') || 0) });
let bytes = readfile(PNG);
let p = T.preprocess(bytes);
let want = fx.tensor('patches');
let grid = fx.tensor('grid').tolist();

printf('grid %s (oracle %s), %d patches\n', join('x', p.grid), join('x', grid), p.patches.shape()[0]);
assert(join(',', p.grid) == join(',', grid), 'patch grid matches');
assert(p.patches.shape()[0] == want.shape()[0] && p.patches.shape()[1] == want.shape()[1], 'patch shape');

let dmax = p.patches.to('float32').sub(want.to('float32')).abs().max().item();

printf('preprocessing: max |diff| vs exllamav3 = %g\n', dmax);
assert(dmax == 0, 'patches are bit-exact');

let trace = [];
let emb = T.forward(p.patches, p.grid, trace);

printf('block 0: rel-rms %.2e, cosine %.6f\n', rel_rms(trace[0], fx.tensor('block0').cuda(T.dev)),
	cosine(trace[0], fx.tensor('block0').cuda(T.dev)));

if (fx.has_tensor ? fx.has_tensor('blocks_out') : true) {
	printf('after block 23: rel-rms %.2e, cosine %.6f\n',
		rel_rms(trace[1], fx.tensor('blocks_out').cuda(T.dev)),
		cosine(trace[1], fx.tensor('blocks_out').cuda(T.dev)));
	printf('after downsample: rel-rms %.2e, cosine %.6f\n',
		rel_rms(trace[2], fx.tensor('down').cuda(T.dev)),
		cosine(trace[2], fx.tensor('down').cuda(T.dev)));
}

let e = rel_rms(emb, fx.tensor('emb').cuda(T.dev));
let c = cosine(emb, fx.tensor('emb').cuda(T.dev));

printf('final embeddings %s: rel-rms %.2e, cosine %.6f\n', join('x', emb.shape()), e, c);
assert(rel_rms(trace[0], fx.tensor('block0').cuda(T.dev)) < 0.005, 'block 0 matches');
// the tower amplifies input noise strongly (a 1e-3 perturbation of our own input
// already moves the final embeddings by ~8e-2, tests/glm5_vision_trace.uc), so the
// bound is the amplification, not the fp16 noise floor of a single block
assert(c > 0.995, 'final embeddings match (cosine)');
assert(e < 0.15, 'final embeddings match (rel-rms)');
print('glm5_vision: ok\n');
