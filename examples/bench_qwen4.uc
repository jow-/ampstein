// examples/bench_qwen4.uc — Qwen3.8-Flash-Next decode / prefill timing.
//
//   bench_qwen4.uc [--model DIR] [--ctx N] [--prefill N] [--steps N] [--no-graph]
//
// Prefills N synthetic tokens through the wavefront runner (timed), then runs
// `steps` dependent decode steps (each token feeds the next), graphs or eager.

let torch = require('ampstein.torch');
global.torch = torch;
import * as qwen4 from 'ampstein.qwen4';
import * as qwen4_tp from 'ampstein.qwen4_tp';

let model_dir = '/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
let ctx = 8192, nprefill = 512, steps = 40, use_graph = true, use_tp = false, stages = 1, chunk = 512;

for (let i = 0; i < length(ARGV); i++) {
	switch (ARGV[i]) {
	case '--model': model_dir = ARGV[++i]; break;
	case '--ctx': ctx = +ARGV[++i]; break;
	case '--prefill': nprefill = +ARGV[++i]; break;
	case '--steps': steps = +ARGV[++i]; break;
	case '--no-graph': use_graph = false; break;
	case '--tp': use_tp = true; break;
	case '--stages': stages = +ARGV[++i]; break;
	case '--chunk': chunk = +ARGV[++i]; break;
	}
}

function elapsed(c0) {
	let c1 = clock(true);

	return (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1000000000.0;
}

let t0 = clock(true);
let model = use_tp ? qwen4_tp.load(model_dir, { stages: stages, prefill_rows: chunk }) : qwen4.load(model_dir, { mtp: false });

printf('loaded in %.1fs\n', elapsed(t0));

let cache = model.new_cache(ctx);
let g = use_graph ? model.graphs(cache, [ 1 ]) : null;

if (g) {
	for (let cl in cache.layers)
		if (cl.rec_state) {
			cl.rec_state.fill_(0);
			cl.conv_state.fill_(0);
		}

	if (use_tp)
		model.reset(cache);
	else
		model.aux_reset(cache);

	cache.length = 0;
}

// a varied synthetic prompt (repeating patterns make the QSA top-k ambiguous)
let ids = [];

for (let i = 0; i < nprefill; i++)
	push(ids, 1000 + (i * 7919) % 50000);

for (let rep = 0; rep < 2; rep++) {
	let c = model.new_cache(ctx);
	let r = model.prefill_pipeline(c, ids, { chunk: chunk });
	let t1 = clock(true);

	while (r.step(2));

	torch.synchronize();

	if (rep == 1 || nprefill >= 4096)
		printf('prefill %d tokens: %.2f s (%.0f tok/s)\n', nprefill, elapsed(t1), nprefill / elapsed(t1));
}

let r = model.prefill_pipeline(cache, ids, { chunk: chunk });

while (r.step(2));

let tok = r.logits().view([1, -1]).argmax(-1).tolist()[0];

function step() {
	let l = g ? g.run([tok], cache.length).logits : model.forward_chunk([tok], cache);

	tok = l.view([1, -1]).argmax(-1).tolist()[0];
}

for (let i = 0; i < 4; i++)
	step();

torch.synchronize();

let t2 = clock(true);

for (let i = 0; i < steps; i++)
	step();

torch.synchronize();

let tot = elapsed(t2);

printf('%s%s decode at context ~%d: %.2f ms/token (%.1f tok/s) over %d dependent steps\n',
	use_tp ? 'tp ' : '', g ? 'graph' : 'eager', nprefill, tot * 1000 / steps, steps / tot, steps);
