// examples/bench_prefill.uc — qwen35 prefill throughput: sequential
// chunk-by-chunk vs pipelined over the layer split
// (qwen35.prefill_pipeline).
//
//   ucode -L build -L ucode examples/bench_prefill.uc [chunk] [tokens] [modes]
//
// Defaults: chunk 1024, 4096 tokens, modes "plain,pipe,pipe" (the first
// run of each process includes warmup; compare the later runs). Model:
// $AMP_MODEL_QWEN35 or /data/models/Qwen3.8-27B-exl3-5.00bpw,
// layers 0-31 on cuda:0, 32-63 on cuda:1.

let torch = require('ampstein.torch');
global.torch = torch;
import * as qwen35 from 'ampstein.qwen35';
import * as sampler from 'ampstein.sampler';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const C = int(ARGV[0] || 1024);
const TOTAL = int(ARGV[1] || 4096);
const MODES = split(ARGV[2] || 'plain,pipe,pipe', ',');

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let m = qwen35.load(MODEL, { devices: devs, mtp: false });
let cache = m.new_cache(TOTAL + 16);
let ids = [];

for (let i = 0; i < TOTAL; i++)
	push(ids, 1000 + (i * 37) % 5000);

function reset() {
	for (let l = 0; l < m.cfg.num_hidden_layers; l++) {
		let cl = cache.layers[l];

		if (cl.rec_state) {
			cl.rec_state.fill_(0);
			cl.conv_state.fill_(0);
		}
	}

	cache.length = 0;
}

for (let mode in MODES) {
	reset();
	torch.synchronize();

	let t0 = now_ms(), lg;

	if (mode == 'plain') {
		// one chunk at a time: a single-chunk pipeline is a plain forward
		for (let p = 0; p < TOTAL; p += C) {
			let n = (TOTAL - p < C) ? TOTAL - p : C;
			let r = qwen35.prefill_pipeline(m, cache, slice(ids, p, p + n), { chunk: n });

			while (r.step(64))
				;

			lg = r.logits();
		}
	}
	else {
		let r = qwen35.prefill_pipeline(m, cache, ids, { chunk: C });

		while (r.step(8))
			;

		lg = r.logits();
	}

	let a = sampler.argmax_last(lg);

	torch.synchronize();

	let ms = now_ms() - t0;

	printf('%-5s %d tokens, chunk %d: %7.0f ms = %5.0f tok/s (argmax %d)\n',
		mode, TOTAL, C, ms, TOTAL * 1000 / ms, a);
}
