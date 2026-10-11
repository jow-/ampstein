// Phase 2 test: full 16-layer EXL3 linear stack from a REAL model dir vs
// the exllamav3 oracle.
// Fixture: tests/fixtures/exl3_stack_oracle.safetensors — per-layer
// outputs of all 7 EXL3 linears (q/k/v/o + gate/up/down) of
// turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw), computed by exllamav3
// itself. Regenerate with `.venv/bin/python tests/oracle/make_exl3_stack_oracle.py`.
//
// Checks:
//   - all 16 layers construct from the model file with correct shapes
//     (112 linears: 7 per layer)
//   - each layer's dataflow (q = q_proj(X); attn = o_proj(q);
//     gate/up from attn; act = silu(gate) * up; Y = down_proj(act))
//     matches the oracle per-layer output bit-exactly
//
// Run via CTest or manually:
//   ucode -L <build-dir> tests/exl3_stack.uc

import * as models from 'ampstein.models';
import * as weights from 'ampstein.weights';
import { stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/exl3_stack_oracle.safetensors`;
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

const NUM_LAYERS = 16;
const M = 2, HIDDEN = 2048;

let oracle = torch.st(FIXTURE);
let model = weights.open(MODEL);

// construct all layers from the real model file
let layers = [];

for (let i = 0; i < NUM_LAYERS; i++)
	push(layers, models.layer_linears(model, `model.layers.${i}`));

assert(length(layers) == NUM_LAYERS, 'all layers constructed');

let total = 0;

for (let i = 0; i < length(layers); i++)
	total += length(layers[i]);

assert(total == NUM_LAYERS * 7, `112 linears, got ${total}`);

// deterministic input, same as the oracle's
let x = torch.full([M, HIDDEN], 0.1, 'float16').cuda();

// per-layer dataflow, bit-exact vs oracle
let X = oracle.tensor('X').cuda();

for (let i = 0; i < NUM_LAYERS; i++) {
	let L = layers[i];
	let q = L['self_attn.q_proj'].forward(X);
	let attn = L['self_attn.o_proj'].forward(q);
	let gate = L['mlp.gate_proj'].forward(attn);
	let up = L['mlp.up_proj'].forward(attn);
	let act = gate.silu().mul(up);
	let y = L['mlp.down_proj'].forward(act);

	let got = y.cpu().tolist();
	let want = oracle.tensor(`layer${i}_Y`).tolist();

	for (let r = 0; r < M; r++)
		for (let c = 0; c < HIDDEN; c++)
			assert(got[r][c] == want[r][c],
				`layer ${i} output [${r}][${c}]: got ${got[r][c]}, want ${want[r][c]}`);

	printf('layer %2d: bit-exact\n', i);
}

oracle.free();

printf('exl3_stack: all assertions passed\n');