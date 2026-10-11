// examples/bench_prefill_glm5.uc — prefill throughput of GLM-5.3-Flash.
//   uc examples/bench_prefill_glm5.uc -- [--tp] [--stages N] [--tokens N] [--chunk N]
// Prefills a deterministic token pattern through the engine's pipelined
// path (prefill_pipeline) and reports tokens/s; run twice, the first run
// warms the allocator / autotune.
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as glm5_tp from 'ampstein.glm5_tp';

let tp = false, gpus = 6, stages = 1, tokens = 8192, chunk = 2048, start = 0, q8 = false, ov = 'plain', stage_end = null;
let argv = ARGV, st = index(argv, '--');

if (st >= 0) argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	if (argv[i] == '--tp') tp = true;
	else if (argv[i] == '--stages') stages = int(argv[++i]);
	else if (argv[i] == '--tokens') tokens = int(argv[++i]);
	else if (argv[i] == '--chunk') chunk = int(argv[++i]);
	else if (argv[i] == '--gpus') gpus = int(argv[++i]);
	else if (argv[i] == '--start') start = int(argv[++i]);
	else if (argv[i] == '--q8') q8 = true;
	else if (argv[i] == '--ov') ov = argv[++i];       // plain | sync | async (TP stages)
	else if (argv[i] == '--stage-end') stage_end = map(split(argv[++i], ','), (x) => int(x));
}

let dir = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
let max_seq = start + tokens + 64;
let opts = { devices: gpus, stages: stages, max_seq: max_seq, prefill_rows: chunk, mtp: false, kv_bits: q8 ? 8 : null,
	overlap: ov != 'plain', stage_end: stage_end };
let model = tp ? glm5_tp.load(dir, opts) : glm5.load(dir, opts);
let ids = [];

for (let i = 0; i < tokens; i++)
	push(ids, 1000 + (i * 7919) % 50000);

function once() {
	let cache = model.new_cache(max_seq);

	cache.length = start;           // (a deep position: the cache content is irrelevant for timing)
	let t0 = clock(true);
	let run = model.prefill_pipeline(cache, ids, { chunk: chunk, overlap: ov != 'plain', ov_sync: ov == 'sync' });

	while (run.step(1)) {}

	torch.synchronize();

	let t1 = clock(true);
	let s = (t1[0] - t0[0]) + (t1[1] - t0[1]) / 1000000000.0;

	printf('prefill %d tokens, chunk %d: %.2f s = %.0f tok/s\n', tokens, chunk, s, tokens / s);
}

once();
once();
