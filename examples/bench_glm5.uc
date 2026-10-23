// examples/bench_glm5.uc — serial decode latency of GLM-5.3-Flash on the
// layer split (every token waits for the previous one, as in the server).
//   ucode -L ... examples/bench_glm5.uc -- [--steps N] [--gpus N] [--no-graphs]
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as glm5_tp from 'ampstein.glm5_tp';

let steps = 32, gpus = 6, graphs = true, tp = false, S = 1, stages = 1;
let argv = ARGV, st = index(argv, '--');

if (st >= 0) argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	if (argv[i] == '--steps') steps = int(argv[++i]);
	else if (argv[i] == '--gpus') gpus = int(argv[++i]);
	else if (argv[i] == '--no-graphs') graphs = false;
	else if (argv[i] == '--tp') tp = true;
	else if (argv[i] == '--stages') stages = int(argv[++i]);
	else if (argv[i] == '--s') S = int(argv[++i]);
}

let dir = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
let model = tp ? glm5_tp.load(dir, { devices: gpus, stages: stages, max_seq: 2048 }) : glm5.load(dir, { devices: gpus });
let cache = model.new_cache(2048);
let G = graphs ? model.graphs(cache, S > 1 ? [1, S] : (tp && model.mtp ? [1, 3] : [1])) : null;
if (S != 1 && !G) die('--s needs graphs');
let ids = [];

for (let i = 0; i < 128; i++)
	push(ids, 1000 + (i * 7919) % 50000);

model.prefill(ids, cache, 128, 1);

let tok = 5;

function step() {
	if (S > 1) {
		let pos = cache.length;
		let ids = [];

		for (let i = 0; i < S; i++)
			push(ids, tok + i);

		let lg = G.run(ids, pos).logits;

		tok = lg.argmax(-1).tolist()[0];
		cache.length = pos + 1;
		return;
	}

	let lg = G ? G.run([tok], cache.length).logits : model.forward_chunk([tok], cache, 1);

	tok = lg.argmax(-1).item();
}

for (let i = 0; i < 4; i++)
	step();

let c0 = clock(true);

for (let i = 0; i < steps; i++)
	step();

let c1 = clock(true);
let tot = (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1000000000.0;

printf('%s%s decode: %.2f ms/token (%.1f tok/s) over %d serial steps\n',
	tp ? 'tp ' : '', graphs ? 'graph' : 'eager', tot * 1000 / steps, steps / tot, steps);

// MTP round components (tensor parallel): verify step of 3 tokens, one draft step
if (tp && model.mtp && G) {
	let H = model.cfg.hidden_size;
	let hid = torch.zeros([1, 1, H], 'float16', 'cuda:' + model.mtp.dev);

	function timeit(label, f) {
		for (let i = 0; i < 3; i++) f();
		torch.synchronize();

		let c0 = clock(true);

		for (let i = 0; i < 24; i++) {
			f();
			torch.synchronize();
		}

		let c1 = clock(true);

		printf('%s: %.2f ms\n', label, ((c1[0] - c0[0]) + (c1[1] - c0[1]) / 1e9) * 1000 / 24);
	}

	cache.length = 130;
	timeit('plain step (S=1)', () => { cache.length = 130; G.run([5], 130).logits.argmax(-1).item(); });
	timeit('verify step (S=3)', () => { cache.length = 130; G.run([5, 6, 7], 130).logits.argmax(-1); });
	timeit('draft step (mtp graph)', () => { model.mtp_chunk([5], hid, cache.mtp, 131, true).logits.argmax(-1); });
}
