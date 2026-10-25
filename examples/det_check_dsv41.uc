// examples/det_check_dsv41.uc — is a greedy run reproducible? The same prompt, prefilled and decoded twice (graph steps of 1 row and of 3 rows) into
// fresh caches: the logits of the two runs must be bit-identical.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/det_check_dsv41.uc MODEL_DIR [layers=40] [gpus=6] [steps=24]

import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], NL = +(ARGV[1] || 40), NG = +(ARGV[2] || 6), NS = +(ARGV[3] || 24);
let m = dsv41.load(dir, {devices: NG, layers: NL, max_seq: 2048, prefill_rows: 256, spark: false});
let ids = [], x = 4242;

for (let i = 0; i < 700; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function run() {
	let cache = m.new_cache(2048);
	let g = m.graphs(cache, [ 1, 3 ]);
	let out = [];

	m.aux_reset(cache);
	cache.length = 0;
	m.prefill(ids, cache, 256, 0);
	push(out, m.forward_chunk([ 4242 ], cache, 1).to("float32").cpu());          // (eager)

	let tok = 4242;

	for (let i = 0; i < NS; i++) {
		let l = g.run([ tok ], cache.length).logits.to('float32').cpu();

		push(out, l);
		tok = l.argmax(-1).tolist()[0];

		if (i % 4 == 3) {
			let r = g.rows([ tok, tok + 1, tok + 2 ], cache.length).logits.to('float32').cpu();

			push(out, r);
			m.rollback(cache, cache.length - 2);
		}
	}

	return out;
}

let a = run(), b = run(), bad = 0;

for (let i = 0; i < length(a); i++) {
	let d = a[i].sub(b[i]).abs().max().item();

	if (d != 0) {
		bad++;
		printf('result %d differs: max abs diff %g\n', i, d);
	}
}

printf('%d results compared, %d differ\n', length(a), bad);
