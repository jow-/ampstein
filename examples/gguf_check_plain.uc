// examples/gguf_check_plain.uc — the GGUF's plain (bf16 / f32) tensors against the same tensors of a safetensors pack of the same model: they
// are the original's values in both, so the maximum difference should be 0 (anything else: a transformed or mis-mapped tensor).
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/gguf_check_plain.uc GGUF_DIR SAFETENSORS_DIR [layers...]

import * as weights from 'ampstein.weights';

let g = weights.open(ARGV[0]), s = weights.open(ARGV[1]);
let layers = (length(ARGV) > 2) ? map(slice(ARGV, 2), (x) => +x) : [ 0, 1, 2, 8, 14, 20, 24, 39 ];
let n = 0, bad = 0;

function cmp(key) {
	if (!g.has(key) || !s.has(key))
		return;

	let a = g.tensor(key).to('float32'), b = s.tensor(key).to('float32');
	let same = join('x', a.shape()) == join('x', b.shape());
	let d = same ? a.sub(b).abs().max().item() : -1;

	n++;

	if (!same || d != 0) {
		bad++;
		printf('%-52s gguf %s pack %s  max diff %g\n', key, join('x', a.shape()), join('x', b.shape()), d);
	}
}

for (let l in layers) {
	for (let k in [ 'attn_norm.weight', 'ffn_norm.weight', 'hc_attn_fn', 'hc_attn_base', 'hc_attn_scale', 'hc_ffn_fn', 'hc_ffn_base', 'hc_ffn_scale',
	                'attn.q_norm.weight', 'attn.kv_norm.weight', 'attn.attn_sink', 'ffn.gate.weight', 'ffn.gate.bias',
	                'attn.compressor.wkv.weight', 'attn.compressor.wgate.weight', 'attn.compressor.norm.weight',
	                'attn.indexer.wk.weight', 'attn.indexer.k_norm.weight', 'attn.indexer.weights_proj.weight',
	                'engram.q_weight', 'engram.k_weight' ])
		cmp(`layers.${l}.${k}`);
}

cmp('norm.weight');
printf('%d tensors compared, %d differ\n', n, bad);
