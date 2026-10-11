// tests/glm5_kda.uc — GLM-5.3-Flash KDA (linear attention) layers vs
// exllamav3: replays the attention sublayer of the KDA layers present in
// the oracle (teacher-forced from the fixture's attn_in) and compares
// attn_out. Layers 0, 1, 2, 4 (the first KDA layers; no GPU memory beyond
// ~1 GiB needed).

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

for (let i in [0, 1, 2, 4]) {
	assert(cfg.layer_types[i] == 'linear_attention', `layer ${i} is KDA`);

	let K = glm5.build_kda(ws, cfg, i, DEV);
	let cl = glm5.new_kda_state(cfg, K, DEV);
	let x = fx.tensor(`b${i}_attn_in`).cuda(DEV).reshape([n, -1]);
	let want = fx.tensor(`b${i}_attn_out`).cuda(DEV).reshape([n, -1]);

	let y = glm5.kda_forward(cfg, K, x, cl);
	let e = rel_rms(y, want);

	printf('layer %d KDA: rel-rms %g\n', i, e);
	assert(e < 0.02, `layer ${i}: KDA output`);

	// chunk invariance: the same tokens split in two chunks through the
	// carried conv/recurrent state must reproduce the single-chunk output
	let cl2 = glm5.new_kda_state(cfg, K, DEV);
	let h = int(n / 2);
	let y1 = glm5.kda_forward(cfg, K, x.narrow(0, 0, h).contiguous(), cl2);
	let y2 = glm5.kda_forward(cfg, K, x.narrow(0, h, n - h).contiguous(), cl2);
	let e2 = rel_rms(torch.cat([y1, y2], 0), y);

	printf('layer %d KDA chunked (%d+%d) vs whole: rel-rms %g\n', i, h, n - h, e2);
	assert(e2 < 1e-3, `layer ${i}: chunked KDA`);
}

try { torch.synchronize(); } catch (e) {}
print('glm5_kda: ok\n');
