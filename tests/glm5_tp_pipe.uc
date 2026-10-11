// tests/glm5_tp_pipe.uc — the double-buffered prefill pipeline of glm5_tp
// (micro-chunks interleaved layer by layer, all-reduces on side streams).
// Three runs over the same 3000-token prompt on the same model:
//   plain       opts.overlap = false           (one chunk per stage)
//   ov-sync     micro-chunks, reductions synchronous (opts.ov_sync)
//   ov-async    micro-chunks, reductions on side streams
// ov-async and ov-sync execute the same kernels in the same per-stream order,
// so they must agree to fp16 noise: that proves the side streams introduce
// no race (stale input, premature reuse). Plain vs overlapped is only
// reported — a different row batching moves router near-ties and the model
// amplifies that (L85), quality is checked on real text by glm5_ppl_pipe.
// Needs the six GPUs.
import * as glm5 from 'ampstein.glm5_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';

if (!stat(MODEL)) { warn('skip: model missing\n'); exit(125); }

let stages = +(getenv('AMP_GLM5_STAGES') || 3);
let N = 3000, CH = 1024;
let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6), stages: stages,
	max_seq: N + 64, prefill_rows: CH, mtp: false, overlap: true });
let cfg = model.cfg;
let ids = [];

for (let i = 0; i < N; i++)
	push(ids, 1000 + (i * 7919) % 50000);

function reset(cache) {
	for (let i = 0; i < cfg.num_hidden_layers; i++)
		for (let cl in cache.layers[i])
			if (cl.rec_state) { cl.rec_state.fill_(0); cl.conv_state.fill_(0); }

	cache.length = 0;
}

function run(cache, popts) {
	reset(cache);

	let parts = [];
	let t0 = clock(true);
	let r = model.prefill_pipeline(cache, ids, { chunk: CH, hidden: true, overlap: popts.overlap,
		ov_sync: popts.ov_sync, on_hidden: (pos, H) => push(parts, H.view([-1, cfg.hidden_size])) });

	while (r.step(1)) {}

	torch.synchronize();

	let t1 = clock(true);

	return { logits: r.logits(), hidden: torch.cat(parts, 0), s: (t1[0] - t0[0]) + (t1[1] - t0[1]) / 1000000000.0 };
}

function rel(x, y) {
	let a = x.to('float32'), b = y.to('float32');
	let d = a.sub(b);

	return math.sqrt(d.mul(d).mean().item() / a.mul(a).mean().item());
}

let cache = model.new_cache(N + 64);

run(cache, { overlap: false });            // warm-up (allocator, autotune)

let plain = run(cache, { overlap: false });
let plain2 = run(cache, { overlap: false });
let sync = run(cache, { ov_sync: true });
let async = run(cache, {});

printf('prefill %d tokens: plain %.2f s (%.0f tok/s), overlapped %.2f s (%.0f tok/s)\n', N,
	plain.s, N / plain.s, async.s, N / async.s);
printf('plain vs plain (determinism): logits rel-rms %.2e, hidden rel-rms %.2e\n',
	rel(plain.logits, plain2.logits), rel(plain.hidden, plain2.hidden));
printf('async vs sync: logits rel-rms %.2e, hidden rel-rms %.2e\n',
	rel(async.logits, sync.logits), rel(async.hidden, sync.hidden));
printf('plain vs overlapped: logits rel-rms %.4f, hidden rel-rms %.4f (informational)\n',
	rel(plain.logits, async.logits), rel(plain.hidden, async.hidden));

assert(async.hidden.shape()[0] == N && sync.hidden.shape()[0] == N, 'one hidden row per prompt row');
// (the prefill itself is not deterministic: two plain runs differ by the noise floor)
let floor = rel(plain.hidden, plain2.hidden);

assert(rel(async.hidden, sync.hidden) < 3.0 * floor + 1e-3, 'side-stream reductions: no more difference than run-to-run noise');
assert(rel(async.logits, sync.logits) < 3.0 * rel(plain.logits, plain2.logits) + 1e-3, 'same for the logits');
print('glm5_tp_pipe: ok\n');
