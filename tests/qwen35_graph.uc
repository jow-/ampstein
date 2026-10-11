// tests/qwen35_graph.uc — CUDA-graph-captured qwen35 steps
// (ucode/qwen35_graph.uc) vs the eager forward on the real Qwen3.8 model
// split over two GPUs: same prompt prefilled eagerly into two caches,
// then decode steps of S=1 (and one S=3 window) through the graph on
// one cache and eagerly on the other, teacher-forced with the eager
// argmax. Checks logits per step (fp16 noise: the graph's KV-cache
// attention kernel differs from the dense one) and argmax agreement,
// then reports the graph decode rate.

import * as qwen35 from 'ampstein.qwen35';
import * as qgraph from 'ampstein.qwen35_graph';
import * as sampler from 'ampstein.sampler';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

if (!stat(MODEL_DIR)) {
	warn('skip: model dir not found: ' + MODEL_DIR + '\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let m = qwen35.load(MODEL_DIR, { devices: devs, mtp: false });
let tok = torch.tokenizer(MODEL_DIR);

function reset(cache) {
	for (let i = 0; i < m.cfg.num_hidden_layers; i++) {
		let cl = cache.layers[i];

		if (cl.rec_state) {
			cl.rec_state.fill_(0);
			cl.conv_state.fill_(0);
		}
	}

	cache.length = 0;
}

let ce = m.new_cache(512);
let cg = m.new_cache(512);
let G = qgraph.create(m, cg, [1, 3]);

reset(cg);

let ids = tok.encode('The quick brown fox jumps over the lazy dog. In computer science, a hash table is', true);
let le = m.forward_chunk(ids, ce);

m.forward_chunk(ids, cg);

let t = sampler.argmax_last(le);
let worst = 0, agree = 0, N = 12;

function rel(a, b) {
	let af = a.to('float32'), bf = b.to('float32');

	return math.sqrt(af.sub(bf).pow(2).mean().item() / bf.pow(2).mean().item());
}

for (let i = 0; i < N; i++) {
	let a = m.forward_chunk([t], ce);
	let b = G.run([t], cg.length).logits;
	let ta = sampler.argmax_last(a), tb = sampler.argmax_last(b);

	worst = max(worst, rel(b, a));

	if (ta == tb)
		agree++;

	t = ta;
}

// one S=3 window (MTP verify shape)
let w = [t, 13, 220];
let a3 = m.forward_chunk(w, ce);
let b3 = G.run(w, cg.length).logits;
let r3 = rel(b3, a3);

printf('S=1: %d steps, worst logits rel-rms %.2e, argmax agree %d/%d; S=3: rel-rms %.2e\n',
	N, worst, agree, N, r3);

// graph steps attend through torch.attn_decode, eager ones through
// FA2: different fp16 summation order, rel-rms ~2e-2 on the logits
assert(worst < 0.04 && r3 < 0.04, 'graph logits deviate from eager');
assert(agree >= N - 1, 'graph argmax disagrees with eager');
assert(ce.length == cg.length, 'cache lengths differ');

// decode rate through the graph (greedy, host argmax each step)
torch.synchronize();

let t0 = now_ms(), K = 48;

for (let i = 0; i < K; i++)
	t = sampler.argmax_last(G.run([t], cg.length).logits);

let ms = (now_ms() - t0) / K;

printf('graph decode: %.1f ms/token (%.1f tok/s)\n', ms, 1000 / ms);
print('qwen35_graph: PASS\n');
