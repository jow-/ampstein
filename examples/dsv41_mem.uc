// examples/dsv41_mem.uc — memory of the V4.1 pack for a context size: load, cache, decode graphs.
//   ucode ... examples/dsv41_mem.uc <model_dir> <ctx> [gpus]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], ctx = +ARGV[1], gpus = +(ARGV[2] || 6);
let m = dsv41.load(dir, {devices: gpus, max_seq: ctx, prefill_rows: 512});
let cache = m.new_cache(ctx);

dsv41.mem_report(m.devs, `cache for ${ctx}`);

let g = m.graphs(cache, [ 1 ]);

printf('devices: %J\n', m.devs);
