// examples/dsv41_mem.uc — memory of the V4.1 pack for a context size: load, cache, decode graphs.
//   ucode ... examples/dsv41_mem.uc <model_dir> <ctx> [gpus]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], ctx = +ARGV[1], gpus = +(ARGV[2] || 6);
let m = dsv41.load(dir, {devices: gpus, max_seq: ctx, prefill_rows: 512});
let cache = m.new_cache(ctx);

dsv41.mem_report(m.devs, `cache for ${ctx}`);

let g = m.graphs(cache, [ 1 ]);

printf('devices: %J\n', m.devs);

// the engine's warm-up: a full chunk at the end of the context, then a decode step
let n = +(ARGV[3] || 2048);

cache.length = ctx - n - 8;

let wids = [];

for (let i = 0; i < n; i++)
	push(wids, 0);

let run = m.prefill_pipeline(cache, wids, {chunk: n});

while (run.step(1)) {}

g.run([ 0 ], cache.length);
torch.synchronize();

for (let d = 0; d < gpus; d++) {
	let mi = torch.mem_info(d);

	printf('cuda:%d allocated %.2f reserved %.2f free %.2f GiB\n', d, mi[0] / 1073741824.0, mi[1] / 1073741824.0, mi[2] / 1073741824.0);
}
