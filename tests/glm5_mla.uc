// tests/glm5_mla.uc — GLM-5.3-Flash NoPE MLA layers (dense form) vs
// exllamav3: teacher-forced from the fixture's attn_in on layers 3, 7, 11,
// plus chunk invariance through the latent cache.

import * as glm5 from 'ampstein.glm5';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') ||
	'/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_oracle.safetensors`;

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: GLM model dir or oracle fixture missing\n');
	exit(125);
}

const DEV = 0;
let cfg = glm5.load_config(MODEL);
let ws = weights.open(MODEL);
let fx = torch.st(FIX);
let n = length(fx.tensor('ids').tolist());

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() /
		b.to('float32').mul(b.to('float32')).mean().item());
}

for (let i in [3, 7, 11]) {
	assert(cfg.layer_types[i] == 'deepseek_sparse_attention', `layer ${i} is MLA`);

	let A = glm5.build_mla(ws, cfg, i, DEV);
	let x = fx.tensor(`b${i}_attn_in`).cuda(DEV).reshape([n, -1]);
	let want = fx.tensor(`b${i}_attn_out`).cuda(DEV).reshape([n, -1]);

	let cl = glm5.new_mla_cache(cfg, 64, DEV);
	let y = glm5.mla_forward(cfg, A, x, cl, 0);
	let e = rel_rms(y, want);

	printf('layer %d MLA: rel-rms %g\n', i, e);
	assert(e < 0.02, `layer ${i}: MLA output`);

	// two chunks through the cache == one pass
	let cl2 = glm5.new_mla_cache(cfg, 64, DEV);
	let h = int(n / 2);
	let y1 = glm5.mla_forward(cfg, A, x.narrow(0, 0, h).contiguous(), cl2, 0);
	let y2 = glm5.mla_forward(cfg, A, x.narrow(0, h, n - h).contiguous(), cl2, h);
	let e2 = rel_rms(torch.cat([y1, y2], 0), y);

	printf('layer %d MLA chunked (%d+%d) vs whole: rel-rms %g\n', i, h, n - h, e2);
	assert(e2 < 5e-3, `layer ${i}: chunked MLA`);
}

try { torch.synchronize(); } catch (e) {}
print('glm5_mla: ok\n');
