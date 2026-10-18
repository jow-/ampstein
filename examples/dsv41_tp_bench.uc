// examples/dsv41_tp_bench.uc — speed of the tensor-parallel model: prefill (wavefront) and the 1 .. 5 row decode steps.
//   ucode ... examples/dsv41_tp_bench.uc <model_dir> <context tokens> [steps] [max_seq] [chunk]
import * as dsv41_tp from 'ampstein.dsv41_tp';

let dir = ARGV[0], ctxn = +ARGV[1], steps = +(ARGV[2] || 20), ctx = +(ARGV[3] || 32768), chunk = +(ARGV[4] || 1024);
let m = dsv41_tp.load(dir, {devices: 6, stages: 3, max_seq: ctx, prefill_rows: chunk});
let cache = m.new_cache(ctx);
let g = m.graphs(cache, [ 1, 2, 3, 4, 5 ]);
let ids = [], x = 7;

for (let i = 0; i < ctxn; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

let t0 = now();
let run = m.prefill_pipeline(cache, ids, {chunk: chunk});

while (run.step(1)) {}

torch.synchronize();
printf('prefill %d tokens: %.1f s, %.0f tok/s\n', ctxn, now() - t0, ctxn / (now() - t0));

for (let S = 1; S <= 5; S++) {
	let tk = [];

	for (let i = 0; i < S; i++)
		push(tk, 4242 + i);

	let p0 = cache.length;
	let call = () => (S == 1) ? g.run(tk, p0) : g.rows(tk, p0);

	call();
	torch.synchronize();

	let t1 = now();

	for (let i = 0; i < steps; i++) {
		let r = call();

		r.logits.to('float32').argmax(-1).tolist();
	}

	printf('S=%d: %.1f ms/step\n', S, (now() - t1) * 1000 / steps);
}
