// tests/glm5_blocks.uc — GLM-5.3-Flash mHC sites + dense MLP vs exllamav3.
// Fixture tests/fixtures/glm5_oracle.safetensors (make_glm5_oracle.py).
//
// Teacher-forced per stage for the dense layers 0..2: every stage takes the
// reference's own input from the fixture, so errors cannot accumulate and a
// failure names its stage (hc mix post/comb, collapse+norm, MLP, residual).

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

function T(name) { return fx.tensor(name).cuda(DEV); }

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));
	let num = d.mul(d).mean().item();
	let den = b.to('float32').mul(b.to('float32')).mean().item();

	return math.sqrt(num / den);
}

function maxabs(a, b) {
	return a.to('float32').sub(b.to('float32')).abs().max().item();
}

let ids = fx.tensor('ids').tolist();
let n = length(ids);
let H = cfg.hidden_size;

function flat(t) { return t.reshape([n, -1]); }

/* streams after the embedding: the fp32 embedding broadcast 4x */
let x = T('h_embed').to('float32').reshape([n, 1, H]).expand([n, 4, H]).contiguous();

for (let i = 0; i < 3; i++) {
	let L = `${glm5.LP}${i}`;
	let hca = glm5.build_hc(ws, i, 'attn', DEV);
	let hcf = glm5.build_hc(ws, i, 'ffn', DEV);
	let in_ln = ws.gpu16(`${L}.input_layernorm.weight`, DEV);
	let post_ln = ws.gpu16(`${L}.post_attention_layernorm.weight`, DEV);

	// attention site: mix on the reference's input streams
	let r = glm5.hc_mix(cfg, hca, x);
	let e1 = maxabs(r[0], flat(T(`b${i}_hc_attn_post`)));
	let e2 = maxabs(r[1], T(`b${i}_hc_attn_comb`).reshape([n, 4, 4]));

	let ain = torch.rms_norm(r[2], in_ln, cfg.rms_norm_eps);
	let e3 = rel_rms(ain, flat(T(`b${i}_attn_in`)));

	printf('layer %d attn site: post %g comb %g attn_in rel-rms %g\n', i, e1, e2, e3);
	assert(e1 < 1e-4 && e2 < 1e-4, `layer ${i}: hc attn post/comb`);
	assert(e3 < 2e-3, `layer ${i}: attn input`);

	// residual update with the reference attention output
	glm5.hc_apply(x, flat(T(`b${i}_attn_out`)), r[0], r[1]);

	// ffn site
	r = glm5.hc_mix(cfg, hcf, x);
	e1 = maxabs(r[0], flat(T(`b${i}_hc_ffn_post`)));
	e2 = maxabs(r[1], T(`b${i}_hc_ffn_comb`).reshape([n, 4, 4]));

	let min = torch.rms_norm(r[2], post_ln, cfg.rms_norm_eps);
	e3 = rel_rms(min, flat(T(`b${i}_mlp_in`)));

	printf('layer %d ffn site:  post %g comb %g mlp_in rel-rms %g\n', i, e1, e2, e3);
	assert(e1 < 1e-4 && e2 < 1e-4, `layer ${i}: hc ffn post/comb`);
	assert(e3 < 2e-3, `layer ${i}: mlp input`);

	// dense MLP on the reference's own input
	let mlp = glm5.build_mlp(ws, `${L}.mlp`, DEV);
	let y = glm5.mlp_forward(cfg, mlp, flat(T(`b${i}_mlp_in`)));
	let e4 = rel_rms(y, flat(T(`b${i}_mlp_out`)));

	printf('layer %d dense MLP: rel-rms %g\n', i, e4);
	assert(e4 < 0.02, `layer ${i}: dense mlp`);

	glm5.hc_apply(x, flat(T(`b${i}_mlp_out`)), r[0], r[1]);

	let e5 = rel_rms(x, T(`b${i}_x`).reshape([n, 4, H]));

	printf('layer %d block output streams: rel-rms %g\n', i, e5);
	assert(e5 < 1e-3, `layer ${i}: streams after the block`);

	// continue from the reference's streams (no accumulation of our error)
	x = T(`b${i}_x`).reshape([n, 4, H]).contiguous();
}

try { torch.synchronize(); } catch (e) {}
print('glm5_blocks: ok\n');
