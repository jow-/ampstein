// Phase 4 test: CUDA-graph-captured decode step vs the exllamav3
// logits oracle.
//
// Fixture: tests/fixtures/llama_logits.safetensors (same as
// llama_decode.uc). The decoder captures ONE S=1 step whose token id,
// position and KV write slot all live in device buffers; replaying
// re-reads their CURRENT values, so this test also guards the
// capture-safety invariants:
//
//   - every tensor the graph reads stays reachable for the decoder's
//     whole lifetime (ucode GC + caching allocator would otherwise
//     free and REUSE that memory between replays — lesson L38)
//   - value-dependent ops (embedding gather, fused attention's
//     device-side position read, KV slot write) follow the buffers on
//     every replay
//
// Each oracle token is teacher-forced through the captured step
// (start_at + run(1)); the step's logits must match the oracle row.
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/graphdec.uc

import * as llama from 'ampstein.llama';
import * as graphdec from 'ampstein.graphdec';
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
let dec = graphdec.create(m, S + 8);

// Teacher-force the ORACLE's tokens one at a time through the
// captured step (no eager prefill — every position goes through the
// graph). Step i appends ids[i] at slot i and produces the logits
// for position i+1, which must match the oracle's row i.
for (let i = 0; i < S; i++) {
	dec.start_at(ids[i]);
	dec.run(1);

	assert(dec.length() == i + 1, `cache length after step ${i}`);

	let logits = dec.logits();
	let mine = logits.view([1, V]).cpu().to('float32');
	let diff = mine.sub(ref.narrow(0, i, 1)).abs().max().item();

	assert(diff <= MAX_ABS_DIFF,
		`graph step ${i}: max abs diff ${diff} exceeds ${MAX_ABS_DIFF}`);

	let am_mine = logits.topk(1, 2).indices.item();
	let am_ref = ref.narrow(0, i, 1).topk(1, 1).indices.item();

	assert(am_mine == am_ref,
		`graph step ${i}: argmax ${am_mine} vs oracle ${am_ref}`);
}

// the last captured step's argmax IS the greedy continuation
let tok = ref.narrow(0, S - 1, 1).topk(1, 1).indices.item();

assert(dec.next_token() == tok,
	`continuation ${dec.next_token()} vs oracle ${tok}`);

oracle.free();

printf('graphdec: %d captured steps argmax-exact, continuation %d\n',
	S, tok);