// tests/glm5_moe.uc — GLM-5.3-Flash MoE layer vs exllamav3: the router's
// selection and weights, then the full routed + shared output, teacher-
// forced from the fixture's mlp_in. First MoE layer (3) by default;
// GLM5_MOE_LAYER overrides. Needs ~3 GiB free on GPU GLM5_DEV (default 0).

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

const DEV = +(getenv('AMP_GLM5_DEV') || 0);
const LAYER = +(getenv('AMP_GLM5_MOE_LAYER') || 3);
let cfg = glm5.load_config(MODEL);
let ws = weights.open(MODEL);
let fx = torch.st(FIX);
let n = length(fx.tensor('ids').tolist());

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() /
		b.to('float32').mul(b.to('float32')).mean().item());
}

assert(cfg.mlp_layer_types[LAYER] == 'sparse', 'layer is a MoE layer');

let pfx = `${glm5.LP}${LAYER}.mlp`;

ws.upload((k) => index(k, pfx) == 0 ? DEV : null);

let t0 = time();
let M = glm5.build_moe(ws, cfg, LAYER, DEV);

printf('layer %d MoE loaded in %.1fs\n', LAYER, time() - t0);

let y = fx.tensor(`b${LAYER}_mlp_in`).cuda(DEV).reshape([n, -1]);

// routing
let r = glm5.moe_route(cfg, M, y);
let wi = fx.tensor(`b${LAYER}_route_idx`).reshape([n, -1]).tolist();
let ww = fx.tensor(`b${LAYER}_route_w`).reshape([n, -1]).tolist();
let ri = r.idx.reshape([n, -1]).tolist();
let rw = r.w.reshape([n, -1]).tolist();
let bad = 0, wmax = 0.0;

for (let t = 0; t < n; t++) {
	let a = sort(slice(wi[t])), b = sort(slice(ri[t]));

	for (let k = 0; k < length(a); k++)
		if (a[k] != b[k]) { bad++; break; }

	// weights by expert id
	for (let k = 0; k < length(wi[t]); k++) {
		let kk = index(ri[t], wi[t][k]);

		if (kk >= 0) {
			let d = math.abs ? math.abs(rw[t][kk] - ww[t][k]) :
				((rw[t][kk] > ww[t][k]) ? rw[t][kk] - ww[t][k] : ww[t][k] - rw[t][kk]);

			if (d > wmax) wmax = d;
		}
	}
}

printf('routing: %d/%d tokens with a different expert set, max |dw| %g\n', bad, n, wmax);
assert(bad == 0, 'router selects the same experts');
assert(wmax < 2e-3, 'router weights (reference rounds them to fp16)');

let out = glm5.moe_forward(cfg, M, y);
let e = rel_rms(out, fx.tensor(`b${LAYER}_mlp_out`).cuda(DEV).reshape([n, -1]));

printf('layer %d MoE output: rel-rms %g\n', LAYER, e);
assert(e < 0.02, 'MoE output');

try { torch.synchronize(); } catch (e2) {}
print('glm5_moe: ok\n');

// fused decode path (torch.moe_coop) vs the composite path, 1 to 16 tokens
for (let rows in [1, 5, 12, 16]) {
	let yy = y.narrow(0, 0, rows).contiguous();
	let a = glm5.moe_forward(cfg, M, yy).mul(1.0);
	let b = glm5.moe_forward_ref(cfg, M, yy);
	let ef = rel_rms(a, b);

	printf('fused vs composite, %d token(s): rel-rms %.2e (coop %s)\n', rows, ef,
		M.coop ? 'on' : 'OFF');
	assert(M.coop, 'experts share one bitrate: fused path built');
	assert(ef < 5e-3, 'fused MoE matches the composite path');
}

// token-sorted fused path (prefill) vs the composite path on all prompt rows
{
	let a = glm5.moe_forward(cfg, M, y).mul(1.0);
	let b = glm5.moe_forward_ref(cfg, M, y);
	let ef = rel_rms(a, b);

	printf('sorted fused vs composite, %d tokens: rel-rms %.2e\n', n, ef);
	assert(ef < 5e-3, 'sorted fused MoE matches the composite path');
}
