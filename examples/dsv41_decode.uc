// examples/dsv41_decode.uc — decode speed of the V4.1 pack through the CUDA graphs (random context).
//   ucode ... examples/dsv41_decode.uc <model_dir> <context tokens> <steps> [gpus] [max_seq]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], ctxn = +ARGV[1], steps = +ARGV[2], gpus = +(ARGV[3] || 6), ctx = +(ARGV[4] || 65536);
let m = dsv41.load(dir, {devices: gpus, max_seq: ctx, prefill_rows: 1024, layers: +(getenv('AMP_DSV41_LAYERS') || 0) || null});
let cache = m.new_cache(ctx);
let g = m.graphs(cache, [ 1 ]);
let ids = [], x = 7;

for (let i = 0; i < ctxn; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let run = m.prefill_pipeline(cache, ids, {chunk: 1024});

while (run.step(1)) {}

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

let tok = 4242;

g.run([ tok ], cache.length);
torch.synchronize();

for (let k in keys(dsv41.PROFILE))
	delete dsv41.PROFILE[k];

let t0 = now();

for (let i = 0; i < steps; i++) {
	let r = g.run([ tok ], cache.length);

	tok = r.logits.to('float32').argmax(-1).tolist()[0];
}

let dt = now() - t0;

printf('context %d, %d steps: %.1f ms/token, %.1f tok/s\n', ctxn, steps, dt * 1000 / steps, steps / dt);

for (let k in sort(keys(dsv41.PROFILE)))
	printf('  %-16s %.2f ms/token\n', k, dsv41.PROFILE[k] / steps);
