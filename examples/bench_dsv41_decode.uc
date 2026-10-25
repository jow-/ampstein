// examples/bench_dsv41_decode.uc — decode speed of the single-chain DeepSeek-V4.1 model (CUDA graphs): milliseconds per token and
// tokens per second after a short prompt, optionally with the per-op host profile of one eager step (AMP_DSV41_PROF=1).
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/bench_dsv41_decode.uc MODEL_DIR [layers=40] [gpus=6] [prompt=64] [steps=60]

import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0];
let NL = +(ARGV[1] || 40), NG = +(ARGV[2] || 6), NP = +(ARGV[3] || 64), NS = +(ARGV[4] || 60);

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

let m = dsv41.load(dir, {devices: NG, layers: NL, max_seq: 4096, prefill_rows: 256});
let ids = [], x = 12345;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let cache = m.new_cache(4096);

m.prefill(ids, cache, 256, 0);

let g = m.graphs(cache, [ 1 ]);

m.aux_reset(cache);
cache.length = 0;
m.prefill(ids, cache, 256, 0);

let tok = 4242;

for (let i = 0; i < 5; i++)
	tok = g.run([ tok ], cache.length).logits.argmax(-1).tolist()[0];

torch.synchronize();

let t0 = now();

for (let i = 0; i < NS; i++)
	tok = g.run([ tok ], cache.length).logits.argmax(-1).tolist()[0];

let dt = (now() - t0) / NS;

printf('%d layers on %d GPUs: %.2f ms per token = %.1f tok/s (graph decode, %d steps after a %d-token prompt)\n', NL, NG, dt * 1000, 1 / dt, NS, NP);

// AMP_DSV41_PROF=1: the host-side counters (they synchronize, so the step time above grows), ms per step over the timed steps
for (let k in sort(keys(dsv41.PROFILE)))
	printf('  %s: %.3f ms/step\n', k, dsv41.PROFILE[k] * 1000 / (NS + 5));
