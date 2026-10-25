// examples/gguf_check_dense.uc — does the GGUF's dense tensor mapping (ggufw.uc) hold? Compares dequantized GGUF matrices
// with the block-scaled FP8 matrices of a safetensors pack of the same model (DeepSeek-V4.1-Flash-1017 SAGE EXL3: dense
// weights FP8, the original's precision up to the block scales): the same matrix quantized differently, so the cosine
// similarity of every pair is close to 1 when the layout and naming agree, ~0 when not.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/gguf_check_dense.uc GGUF_DIR SAFETENSORS_DIR [layers...]

import * as weights from 'ampstein.weights';

let gdir = ARGV[0], sdir = ARGV[1];
let layers = (length(ARGV) > 2) ? map(slice(ARGV, 2), (x) => +x) : [ 0, 1, 2, 20, 39 ];
let g = weights.open(gdir), s = weights.open(sdir);
let scratch = torch.empty([64 * 1024 * 1024], 'float16', 'cuda:0');
let worst = 1.0;

function cmp(key, ga, sa) {
	let a = torch.gguf_dequant(g.gpu(ga + '.weight', 0), g.qtype(ga + '.weight'), 'float32');
	let w = s.gpu(sa + '.weight', 0), sc = s.gpu(sa + '.scale', 0);
	let b = torch.fp8_dequant(w, sc, scratch).to('float32');

	a = a.reshape([-1]);
	b = b.reshape([-1]);

	let dot = a.mul(b).sum().item(), na = a.mul(a).sum().item(), nb = b.mul(b).sum().item();
	let cs = dot / ((na * nb) ** 0.5);

	printf('%-44s %-7s cos %.5f  |gguf|/|fp8| %.4f\n', key, g.qtype(ga + '.weight'), cs, ((na / nb) ** 0.5));
	worst = min(worst, cs);
}

for (let l in layers) {
	for (let m in [ [ 'attn.wq_a', 'attn.wq_a' ], [ 'attn.wq_b', 'attn.wq_b' ], [ 'attn.wkv', 'attn.wkv' ], [ 'attn.wo_a', 'attn.wo_a' ],
	                [ 'attn.wo_b', 'attn.wo_b' ], [ 'ffn.shared_experts.w1', 'ffn.shared_experts.w1' ],
	                [ 'ffn.shared_experts.w3', 'ffn.shared_experts.w3' ], [ 'ffn.shared_experts.w2', 'ffn.shared_experts.w2' ] ]) {
		let k = `layers.${l}.${m[0]}`;

		if (s.has(k + '.scale') && g.has(k + '.weight'))
			cmp(k, k, k);
	}
}

printf('worst cosine %.5f\n', worst);
