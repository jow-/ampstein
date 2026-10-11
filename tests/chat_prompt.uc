// Phase 3 regression test: full llama-3 CHAT prompt prefill logits.
//
// Fixture: tests/fixtures/chat_prompt_logits.safetensors (see
// tests/oracle/make_chat_prompt_oracle.py) — BOS + atomized chat
// template (42 tokens) with exllamav3 Model.forward logits.
//
// This is the test that catches the all-fp16 RMSNorm variance bug:
// token 128000's embedding row is tiny (~3e-3), mean(x^2) lands near
// eps and fp16 squares go subnormal — an all-fp16 norm produces garbage
// logits for BOS-prefixed prompts while every normal-magnitude token
// still matches the plain-prompt oracles.
//
// Criterion: max |diff| <= 0.5 over all positions (fp16 noise; the
// RMSNorm bug produced ~7.0), and the final position's argmax must
// match exactly (it drives generation). Near-tie argmax flips at
// intermediate positions are legitimate fp16 noise (top-2 gaps of
// ~0.02 with diffs of ~0.19) and are not asserted.
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/chat_prompt.uc

import * as llama from 'ampstein.llama';
import { stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/chat_prompt_logits.safetensors`;
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

const MAX_ABS_DIFF = 0.5;

let oracle = torch.st(FIXTURE);

let ids = oracle.tensor('ids').cpu().tolist();
let ref = oracle.tensor('logits');          // (S, V) fp16 CPU
let S = length(ids);

assert(ids[0] == 128000, 'fixture prompt starts with BOS');

let m = llama.load(MODEL);

let cache = m.new_cache(S + 4);
let logits = m.forward_chunk(ids, cache);   // (1, S, V)

let mine = logits.view([S, -1]).cpu().to('float32');
let V = mine.shape()[1];

let maxdiff = 0;

for (let i = 0; i < S; i++) {
	let row = mine.narrow(0, i, 1);
	let d = row.sub(ref.narrow(0, i, 1)).abs().max().item();

	if (d > maxdiff)
		maxdiff = d;
}

assert(maxdiff <= MAX_ABS_DIFF,
	`max abs diff ${maxdiff} exceeds ${MAX_ABS_DIFF}`);

let last_mine = mine.narrow(0, S - 1, 1).topk(1, 1).indices.item();
let last_ref = ref.narrow(0, S - 1, 1).topk(1, 1).indices.item();

assert(last_mine == last_ref,
	`final position argmax ${last_mine} vs oracle ${last_ref}`);

printf('chat_prompt: %d positions, max diff %.3f <= %.2f, '
	+ 'final argmax exact\n', S, maxdiff, MAX_ABS_DIFF);