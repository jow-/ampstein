// examples/dsv41_prefill.uc — prefill throughput of the V4.1 pack (random tokens), with AMP_DSV41_PROF=1 the
// time of the phases.   ucode ... examples/dsv41_prefill.uc <model_dir> <tokens> [chunk] [gpus] [max_seq]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], n = +ARGV[1], chunk = +(ARGV[2] || 512), gpus = +(ARGV[3] || 6), ctx = +(ARGV[4] || 65536);
let m = dsv41.load(dir, {devices: gpus, max_seq: ctx, prefill_rows: chunk});
let cache = m.new_cache(ctx);
let ids = [], x = 7;

for (let i = 0; i < n; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

// warm-up chunk, then the timed run
m.forward_chunk(slice(ids, 0, chunk), cache, 0);
torch.synchronize();
m.aux_reset(cache);
cache.length = 0;

for (let k in keys(dsv41.PROFILE))
	delete dsv41.PROFILE[k];

let t0 = now();
let run = m.prefill_pipeline(cache, ids, {chunk: chunk});

while (run.step(1)) {}

let th = now() - t0;

torch.synchronize();

let dt = now() - t0;

printf('host issue time %.1f s\n', th);

printf('%d tokens, chunk %d: %.1f s, %.0f tok/s\n', n, chunk, dt, n / dt);

if (dsv41.PROFILE) {
	let tot = 0;

	for (let k in keys(dsv41.PROFILE))
		tot += dsv41.PROFILE[k];

	for (let k in sort(keys(dsv41.PROFILE)))
		printf('  %-14s %8.0f ms  %4.1f%%\n', k, dsv41.PROFILE[k], 100.0 * dsv41.PROFILE[k] / tot);
}

if (dsv41.HOSTPROF) {
	let per = {};

	for (let e in dsv41.HOSTPROF)
		per[e[0]] = (per[e[0]] || 0) + e[1];

	for (let l in sort(keys(per), (a, b) => a - b))
		printf('  layer %2d host %.0f ms\n', l, per[l]);
}
