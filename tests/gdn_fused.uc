// Test: the fused GatedDeltaNet layer (qwen35.gdn_forward: gdn_gates +
// gdn_prep + gdn_scan + gdn_out_norm kernels) vs its composite ATen
// reference (qwen35.gdn_forward_ref) on synthetic data, no model.
//
// The layer's projections are stubbed to return fixed random tensors,
// so both paths see identical inputs. Checks the layer output and the
// carried recurrent + conv state, for a first chunk, a short chunk
// (S < K-1: the conv state update reads old state slots) and a
// continuation chunk.
//
//   ucode -L <build-dir> -L <src>/ucode tests/gdn_fused.uc

import * as qwen35 from 'ampstein.qwen35';

let seed = 4242;

function rnd() {
	seed = (seed * 1103515245 + 12345) % 2147483648;

	return (seed / 1073741824.0) - 1.0;
}

function tensor(shape, lo, hi, dtype) {
	let n = 1;

	for (let i = 0; i < length(shape); i++)
		n *= shape[i];

	let vals = [];

	for (let i = 0; i < n; i++)
		push(vals, lo + (rnd() + 1.0) * 0.5 * (hi - lo));

	return torch.from_array(vals, 'float32', 'cuda:0').to(dtype || 'float32').view(shape);
}

const Hk = 4, Hv = 12, hd = 128, K = 4;
const D = (2 * Hk + Hv) * hd;

let cfg = {
	ln_num_k_heads: Hk, ln_k_head_dim: hd, ln_num_v_heads: Hv, ln_v_head_dim: hd,
	ln_k_dim: Hk * hd, ln_v_dim: Hv * hd, fdim_qkv: D, conv_kernel: K,
	rms_norm_eps: 1e-6
};

function stub(t) {
	return { forward: function (x) { return t; } };
}

function layer(S) {
	return {
		in_proj_qkv: stub(tensor([1, S, D], -2, 2, 'float16')),
		in_proj_z: stub(tensor([1, S, Hv * hd], -2, 2, 'float16')),
		in_proj_b: stub(tensor([1, S, Hv], -3, 3, 'float16')),
		in_proj_a: stub(tensor([1, S, Hv], -3, 3, 'float16')),
		out_proj: { forward: function (x) { return x; } }
	};
}

let L0 = {
	dt_bias: tensor([Hv], -1, 1),
	neg_exp_a_log: tensor([Hv], -2, -0.1),
	conv_w: tensor([D, K], -0.5, 0.5, 'float16'),
	norm_w: tensor([hd], 0.5, 1.5, 'float16')
};

function state() {
	return {
		rec_state: tensor([Hv, hd, hd], -0.1, 0.1),
		conv_state: tensor([1, D, K - 1], -1, 1, 'float16')
	};
}

function rel(a, b) {
	let af = a.to('float32'), bf = b.to('float32');

	return af.sub(bf).pow(2).mean().sqrt().item() / bf.pow(2).mean().sqrt().item();
}

let cs_f = state();
let cs_r = { rec_state: cs_f.rec_state.add(0.0), conv_state: cs_f.conv_state.add(0.0) };

for (let S in [ 37, 2, 19 ]) {
	let L = { ...layer(S), ...L0 };
	let x = tensor([1, S, 16], 0, 1, 'float16');     // unused by the stubs
	let y_f = qwen35.gdn_forward(cfg, L, x, cs_f);
	let y_r = qwen35.gdn_forward_ref(cfg, L, x, cs_r);

	let e_out = rel(y_f, y_r);
	let e_rec = rel(cs_f.rec_state, cs_r.rec_state);
	let e_conv = cs_f.conv_state.to('float32').sub(cs_r.conv_state.to('float32')).abs().max().item();

	printf('S=%2d: out rel-rms %.2e, rec state %.2e, conv state maxdiff %.2e\n',
		S, e_out, e_rec, e_conv);

	// fp16 conv taps were rounded per op in the composite path; the
	// fused kernel accumulates in fp32
	assert(e_out < 5e-3, `S=${S}: output rel-rms ${e_out}`);
	assert(e_rec < 5e-3, `S=${S}: recurrent state rel-rms ${e_rec}`);
	assert(e_conv == 0, `S=${S}: conv state differs (${e_conv})`);
}

print('gdn_fused: PASS\n');
