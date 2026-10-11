// tests/qwen35_pipeline.uc — pipelined multi-chunk prefill
// (qwen35.prefill_pipeline, wavefront over the layer split) vs plain
// sequential forward_chunk calls over the same prompt, on the real Qwen3.8 model
// split over two GPUs (layers 0-31 cuda:0, 32-63 cuda:1).
//
// Checks: last-position logits are bit-exact vs a sequential prefill
// with the same chunk boundaries, the argmax is identical, and greedy
// decoding continued from both caches produces the same tokens (so the KV cache and the GDN recurrent/conv states
// the pipeline leaves behind are consistent). Uses small chunks plus an
// extra cut so several chunks are in flight on both devices at once.

import * as qwen35 from 'ampstein.qwen35';
import * as sampler from 'ampstein.sampler';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL_DIR)) {
	warn('skip: model dir not found: ' + MODEL_DIR + '\n');
	exit(125);
}

const N = 300, CHUNK = 48, CUT = 100, DECODE = 6;

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let m = qwen35.load(MODEL_DIR, { devices: devs, mtp: false });
let tok = torch.tokenizer(MODEL_DIR);
let text = '';

for (let i = 0; i < 40; i++)
	text += `Item ${i}: the value of ${i} squared is ${i * i}. `;

let ids = slice(tok.encode(text, true), 0, N);

assert(length(ids) == N, `prompt has ${length(ids)} tokens, need ${N}`);

function greedy(cache, logits) {
	let out = [];
	let lg = logits;

	for (let i = 0; i < DECODE; i++) {
		let t = sampler.argmax_last(lg);

		push(out, t);
		lg = m.forward_chunk([t], cache);
	}

	return out;
}

// reference: plain sequential prefill with the SAME chunk boundaries
// (identical kernels in identical per-layer order -> bit-exact expected;
// one big chunk would differ by fp16 noise, since >144-row chunks take
// the reconstruct+cuBLAS gemm path)
let bounds = [];

for (let p = CHUNK; p < N; p += CHUNK)
	push(bounds, p);

push(bounds, CUT);
bounds = sort(bounds, (x, y) => x - y);
push(bounds, N);

let c_ref = m.new_cache(N + DECODE + 8);
let lg_ref, prev = 0;

for (let i = 0; i < length(bounds); i++) {
	lg_ref = m.forward_chunk(slice(ids, prev, bounds[i]), c_ref);
	prev = bounds[i];
}

// the pipeline runs lm_head on ONE row; do the same here (a 44-row head
// gemm takes another kernel and differs by fp16 noise)
let hid = c_ref.hidden;

lg_ref = m.lm_head.forward(hid.narrow(1, hid.shape()[1] - 1, 1).contiguous());

// pipelined: several chunks in flight, plus an extra cut
let c_pipe = m.new_cache(N + DECODE + 8);
let r = qwen35.prefill_pipeline(m, c_pipe, ids, { chunk: CHUNK, cuts: [ CUT ] });

while (r.step(3))
	;

let lg_pipe = r.logits();

assert(c_pipe.length == N, `pipeline cache length ${c_pipe.length}`);

let a = lg_pipe.to('float32'), b = lg_ref.to('float32');
let rel = math.sqrt(a.sub(b).pow(2).mean().item() / b.pow(2).mean().item());
let am_pipe = sampler.argmax_last(lg_pipe), am_ref = sampler.argmax_last(lg_ref);

printf('last-position logits: rel-rms %.2e, argmax %d vs %d\n', rel, am_pipe, am_ref);
assert(rel < 1e-6, `logits rel-rms ${rel} (expected bit-exact)`);
assert(am_pipe == am_ref, `argmax ${am_pipe} vs ${am_ref}`);

let g_ref = greedy(c_ref, lg_ref);
let g_pipe = greedy(c_pipe, lg_pipe);

printf('greedy continuation: %J vs %J\n', g_pipe, g_ref);
assert(sprintf('%J', g_pipe) == sprintf('%J', g_ref), 'continuations differ');

print('qwen35_pipeline: PASS\n');
