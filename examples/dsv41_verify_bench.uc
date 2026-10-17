// examples/dsv41_verify_bench.uc — time of an S-row decode step (the speculative verify) through the graphs.
//   ucode ... examples/dsv41_verify_bench.uc <model_dir> <context tokens> [steps] [max_seq]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], ctxn = +ARGV[1], steps = +(ARGV[2] || 20), ctx = +(ARGV[3] || 32768);
let m = dsv41.load(dir, {devices: 6, max_seq: ctx, prefill_rows: 1024});
let cache = m.new_cache(ctx);
let g = m.graphs(cache, [ 1, 2, 3, 4, 5 ]);
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

for (let S = 1; S <= 5; S++) {
	let tk = [];

	for (let i = 0; i < S; i++)
		push(tk, 4242 + i);

	let p0 = cache.length;
	let call = () => (S == 1) ? g.run(tk, p0) : g.rows(tk, p0);

	call();
	torch.synchronize();

	let t0 = now();

	for (let i = 0; i < steps; i++) {
		let r = call();

		r.logits.to('float32').argmax(-1).tolist();
	}

	let dt = now() - t0;

	printf('S=%d: %.1f ms/step\n', S, dt * 1000 / steps);
}
