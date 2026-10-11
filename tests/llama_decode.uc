// Phase 3 test: KV-cache decode path vs the exllamav3 logits oracle.
//
// Fixture: tests/fixtures/llama_logits.safetensors (see
// tests/oracle/make_llama_logits_oracle.py) — per-position logits of
// turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw) from exllamav3's own
// Model.forward.
//
// Checks:
//   - feeding the oracle ids ONE TOKEN AT A TIME through the cached
//     path (ucode/kvcache.uc + llama.forward_chunk) reproduces the
//     oracle's argmax at every position, within fp16 noise
//   - chunked prefill (2-token chunks) agrees with the oracle too
//   - greedy generate() continues with the oracle's argmax token at
//     the last prompt position
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/llama_decode.uc

import * as llama from 'ampstein.llama';
import { stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/llama_logits.safetensors`;
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

const MAX_ABS_DIFF = 0.15;   // fp16 logits noise; oracle is fp32

let oracle = torch.st(FIXTURE);

let ids = oracle.tensor('ids').cpu().tolist();
let ref = oracle.tensor('logits');          // (S, V) fp32 CPU
let S = length(ids), V = ref.shape()[1];

let m = llama.load(MODEL);

// --- 1. single-token decode steps against the oracle ------------------
// Each step attends to cache slots [0, pos+1); position i's logits
// must match the oracle's row i.
let cache = m.new_cache(S + 4);

assert(cache.length == 0, 'fresh cache length');

for (let i = 0; i < S; i++) {
	let logits = m.forward_chunk([ids[i]], cache);   // (1, 1, V)

	assert(cache.length == i + 1, `cache length after step ${i}`);

	let mine = logits.view([1, V]).cpu().to('float32');
	let diff = mine.sub(ref.narrow(0, i, 1)).abs().max().item();

	assert(diff <= MAX_ABS_DIFF,
		`decode step ${i}: max abs diff ${diff} exceeds ${MAX_ABS_DIFF}`);

	let am_mine = logits.topk(1, 2).indices.item();
	let am_ref = ref.narrow(0, i, 1).topk(1, 1).indices.item();

	assert(am_mine == am_ref,
		`decode step ${i}: argmax ${am_mine} vs oracle ${am_ref}`);
}

// --- 2. chunked prefill (2-token chunks) against the oracle ------------
let cache2 = m.new_cache(S + 4);
let pos = 0;

for (let start = 0; start < S; start += 2) {
	let n = (start + 2 <= S) ? 2 : S - start;
	let chunk = [];

	for (let j = 0; j < n; j++)
		push(chunk, ids[start + j]);

	let logits = m.forward_chunk(chunk, cache2);   // (1, n, V)

	let mine = logits.view([n, V]).cpu().to('float32');
	let rref = ref.narrow(0, start, n);
	let diff = mine.sub(rref).abs().max().item();

	assert(diff <= MAX_ABS_DIFF,
		`chunk at ${start}: max abs diff ${diff} exceeds ${MAX_ABS_DIFF}`);

	pos += n;
}

assert(cache2.length == S, 'chunked prefill filled the cache');

// --- 3. greedy generation continues with the oracle's argmax -----------
let out = m.generate(ids, 4);

assert(length(out) == S + 4, 'generate returns prompt ++ new tokens');

for (let i = 0; i < S; i++)
	assert(out[i] == ids[i], `generate echoes prompt at ${i}`);

let last_ref = ref.narrow(0, S - 1, 1).topk(1, 1).indices.item();

assert(out[S] == last_ref,
	`first generated token ${out[S]} vs oracle argmax ${last_ref}`);

printf('llama_decode: %d cached steps + chunked prefill + greedy gen, '
	+ 'argmax exact, diffs within %.2f\n', S, MAX_ABS_DIFF);