// examples/bench_dsv41_attn.uc — time of the V4.1 attention layers per prefill chunk and per decode step
// (random inputs, layers 2 / 3 / 20 / 21 / 24 of the pack: AMP_DSV41_ATTN_W, AMP_DSV41_CFG as in tests/dsv41_attn.uc)
//   ucode -L .../lib -L .../src/ucode -ltorch=ampstein.torch examples/bench_dsv41_attn.uc [chunk] [positions]

import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';

const WF = getenv('AMP_DSV41_ATTN_W') || '/data/models/attn_l2_3_20_21_24.safetensors';
const CFGDIR = getenv('AMP_DSV41_CFG') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';
const CHUNK = +(ARGV[0] || 1024);
const NPOS = +(ARGV[1] || 16384);
const LAYERS = [ 2, 3, 20, 21, 24 ];

let cfg = dsv41.load_config(CFGDIR);
let i = rindex(WF, '/');
let ws = weights.open(substr(WF, 0, i), substr(WF, i + 1));
let A = {}, cache = {pools: {}, layers: {}};

for (let l in LAYERS) {
	A[l] = dsv41.build_attn(ws, cfg, l, 0);
	cache.layers[l] = dsv41.new_layer_cache(cfg, A[l], NPOS + CHUNK);

	if (A[l].kv_source)
		cache.pools[l] = {owner: cache.layers[l], mirrors: []};
}

function now() {
	let c = clock(true);

	return c[0] * 1000.0 + c[1] / 1e6;
}

let x = torch.rand([CHUNK, 5120], 'float16', 'cuda:0').sub(0.5);
let times = {};

for (let l in LAYERS)
	times[l] = 0;

let n = 0;

for (let pos = 0; pos + CHUNK <= NPOS; pos += CHUNK) {
	let ctx = {topk: null, cand: null};

	for (let l in LAYERS) {
		torch.synchronize();

		let t0 = now();
		let y = dsv41.attn_forward(cfg, A[l], ctx, cache, cache.layers[l], x, pos);

		torch.synchronize();

		if (pos > 0)
			times[l] += now() - t0;
	}

	if (pos > 0)
		n++;
}

printf('chunk %d, positions up to %d: ms per chunk and layer\n', CHUNK, NPOS);

for (let l in LAYERS)
	printf('  layer %2d (ratio %d): %.1f ms\n', l, cfg.ratios[l], times[l] / n);

if (dsv41.PROFILE) {
	printf('phases (ms per chunk, summed over the %d layers):\n', length(LAYERS));

	for (let k in sort(keys(dsv41.PROFILE)))
		printf('  %-12s %.1f\n', k, dsv41.PROFILE[k] / (n + 1));
}
