// examples/mem_glm5.uc — per-GPU memory after loading GLM-5.3-Flash.
//   uc examples/mem_glm5.uc -- [--tp] [--gpus N] [--ctx N] [--q8] [--no-mtp]
// Prints, per device, what the allocator hands to tensors (allocated), what
// it holds from the driver (reserved) and the driver's used memory (context
// + non-PyTorch), after the load, after a gc(), and after the cache is
// allocated.
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as glm5_tp from 'ampstein.glm5_tp';

let tp = false, gpus = 6, ctx = 16384, q8 = false, mtp = true, stages = 1, chunk = 1024;
let argv = ARGV, st = index(argv, '--');

if (st >= 0) argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	if (argv[i] == '--tp') tp = true;
	else if (argv[i] == '--gpus') gpus = int(argv[++i]);
	else if (argv[i] == '--ctx') ctx = int(argv[++i]);
	else if (argv[i] == '--q8') q8 = true;
	else if (argv[i] == '--no-mtp') mtp = false;
	else if (argv[i] == '--stages') stages = int(argv[++i]);
	else if (argv[i] == '--chunk') chunk = int(argv[++i]);
}

const GIB = 1073741824.0;
let dir = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';

function report(tag) {
	let a = '', r = '', u = '';

	for (let d = 0; d < gpus; d++) {
		let m = torch.mem_info(d);

		a += sprintf(' %5.1f', m[0] / GIB);
		r += sprintf(' %5.1f', m[1] / GIB);
		u += sprintf(' %5.1f', (m[3] - m[2]) / GIB);
	}

	print(`${tag}\n  allocated:${a}\n  reserved: ${r}\n  driver:   ${u}\n`);
}

let opts = { devices: gpus, mtp: mtp, kv_bits: q8 ? 8 : null, max_seq: ctx, stages: stages, prefill_rows: chunk };
let model = tp ? glm5_tp.load(dir, opts) : glm5.load(dir, opts);

report('after load');
gc();
report('after gc');

let cache = model.new_cache(ctx);

report(`after new_cache(${ctx}${q8 ? ', Q8' : ''})`);

if (tp && model.graphs) {
	model.graphs(cache, [ 1, 3 ]);
	report('after graphs (sizes 1, 3)');
}

// worst-case prefill chunk at the end of the context
let wids = [];

for (let i = 0; i < chunk; i++)
	push(wids, 0);

cache.length = ctx - chunk - 8;

let run = model.prefill_pipeline(cache, wids, { chunk: chunk, hidden: false });

while (run.step(1)) {}

torch.synchronize();
report(`after a ${chunk}-token prefill chunk at the end of the context`);

report(`after new_cache(${ctx}${q8 ? ', Q8' : ''})`);
