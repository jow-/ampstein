// examples/gguf_vs_exl3.uc — the first N layers of a GGUF checkpoint against the same layers of an EXL3 / FP8 pack of the same model
// (DeepSeek-V4.1-Flash-1017): one prompt through both, the final hidden states and logits compared. The two quantize the
// same weights differently, so the numbers are close (cosine ~0.99 for a few layers) when naming, layout and kernels agree
// and unrelated when anything is mixed up. Not a CTest: it needs both models.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/gguf_vs_exl3.uc GGUF_DIR EXL3_DIR [layers=4] [tokens=96] [gpus=1]

import * as dsv41 from 'ampstein.dsv41';

let gdir = ARGV[0], edir = ARGV[1];
let NL = +(ARGV[2] || 4), NT = +(ARGV[3] || 96), NG = +(ARGV[4] || 1);

let ids = [], x = 4321;

for (let i = 0; i < NT; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function run(dir, tag) {
	let m = dsv41.load(dir, {devices: NG, layers: NL, max_seq: NT + 64, prefill_rows: 512});
	let cache = m.new_cache(NT + 64);
	let c = m.chunk_begin(ids, cache);

	m.chunk_layers(c, cache, 0, NL);

	let h = dsv41.hc_collapse(c.h, c.pre).to('float32').cpu();
	let lg = m.chunk_head(c, 4).to('float32').cpu();

	printf('%s: hidden |x| %.3f, logits |x| %.3f\n', tag, h.mul(h).mean().item() ** 0.5, lg.mul(lg).mean().item() ** 0.5);
	m = null;

	return {h: h, l: lg};
}

function cos(a, b) {
	a = a.reshape([-1]);
	b = b.reshape([-1]);

	return a.mul(b).sum().item() / ((a.mul(a).sum().item() * b.mul(b).sum().item()) ** 0.5);
}

let a = run(gdir, 'gguf');
let b = run(edir, 'exl3');

printf('cosine: hidden %.5f, logits %.5f; argmax of the last row: gguf %d, exl3 %d\n', cos(a.h, b.h), cos(a.l, b.l),
	a.l.narrow(0, 3, 1).argmax(-1).tolist()[0], b.l.narrow(0, 3, 1).argmax(-1).tolist()[0]);
