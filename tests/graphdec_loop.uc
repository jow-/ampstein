// Multi-step device-loop test: the captured decode path chains steps ON
// DEVICE (argmax -> ids -> next embedding gather -> position advance)
// across replay(n > 1). tests/graphdec.uc teacher-forces every step with
// start_at + run(1), so that chaining was never compared end-to-end
// (a testing gap).
//
// Checks: graphdec prefill + start_at + run(32) + tokens() reproduces
// the eager path's greedy generate() TOKEN-EXACTLY.
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/graphdec_loop.uc

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

const STEPS = 32;

let oracle = torch.st(FIXTURE);
let ids = oracle.tensor('ids').cpu().tolist();

let m = llama.load(MODEL);

/* Reference: the eager path's greedy continuation. */
let want = m.generate(ids, STEPS);
want = slice(want, length(ids), length(want));

/* Captured path: prefill, seed the device loop with the argmax of the
 * last prefill position, then run(32) as ONE host call. */
let dec = graphdec.create(m, length(ids) + STEPS + 8);

let plog = dec.prefill(ids);
let first = plog.narrow(1, length(ids) - 1, 1)
	.view([1, -1]).topk(1, 1).indices.item();

dec.start_at(first);
dec.run(STEPS);

let got = dec.tokens();

assert(length(got) == STEPS,
	`expected ${STEPS} tokens, got ${length(got)}`);

for (let i = 0; i < STEPS; i++)
	assert(got[i] == want[i],
		`token ${i}: graph path ${got[i]} vs eager ${want[i]}`);

printf('graphdec_loop: %d chained steps token-exact vs eager greedy\n',
	STEPS);