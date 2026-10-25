// examples/bench_dsv41_rows.uc — a multi-row (speculative verify) step through the decode graphs: milliseconds per step for 1, 3 and 5 rows.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/bench_dsv41_rows.uc MODEL_DIR [layers=40] [gpus=6] [steps=100]

import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], NL = +(ARGV[1] || 40), NG = +(ARGV[2] || 6), NS = +(ARGV[3] || 100);

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

let m = dsv41.load(dir, {devices: NG, layers: NL, max_seq: 4096, prefill_rows: 256, spark: false});
let ids = [], x = 12345;

for (let i = 0; i < 64; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let cache = m.new_cache(4096);
let g = m.graphs(cache, [ 1, 2, 3, 4, 5 ]);

m.aux_reset(cache);
cache.length = 0;
m.prefill(ids, cache, 256, 0);

let P = cache.length, snap = m.aux_get(cache);
let toks = [ 4242, 1717, 2323, 3131, 777 ];

for (let S in (ARGV[4] ? [ +ARGV[4] ] : [ 1, 3, 5 ])) {
	function once() {
		cache.length = P;
		m.aux_set(cache, snap);

		return (S == 1) ? g.run([ toks[0] ], P) : g.rows(slice(toks, 0, S), P);
	}

	for (let i = 0; i < 5; i++)
		once();

	torch.synchronize();

	let t0 = now();

	for (let i = 0; i < NS; i++)
		once().logits.argmax(-1).tolist();

	printf('%d row%s: %.2f ms per step\n', S, S == 1 ? '' : 's', (now() - t0) / NS * 1000);
}
