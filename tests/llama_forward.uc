// Phase 3 test: full llama-family prefill forward from a REAL model dir
// vs the exllamav3 oracle.
//
// Fixture: tests/fixtures/llama_logits.safetensors — per-position logits
// of turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw) computed by
// exllamav3's own Model.forward (the complete reference graph: embedding,
// RMSNorm, RoPE incl. llama3 scaling, GQA attention, SwiGLU, quantized
// lm_head). Regenerate with:
//   .venv/bin/python tests/oracle/make_llama_logits_oracle.py
//
// Checks:
//   - ucode/llama.uc loads the model dir (config + weights + all 112
//     EXL3 linears + norms + embed + quantized lm_head)
//   - forward() over the oracle's token ids produces logits whose
//     argmax matches the oracle at EVERY position
//   - max |logit diff| vs the fp32 oracle stays within fp16 noise
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/llama_forward.uc

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

assert(S > 0 && V > 0, 'fixture sane');

// --- config parsing -------------------------------------------------
let cfg = llama.load_config(MODEL);

assert(cfg.model_type == 'llama', 'model_type');
assert(cfg.num_hidden_layers == 16, 'num_hidden_layers');
assert(cfg.hidden_size == 2048, 'hidden_size');
assert(cfg.num_attention_heads == 32, 'num_attention_heads');
assert(cfg.num_key_value_heads == 8, 'num_key_value_heads');
assert(cfg.head_dim == 64, 'head_dim');
assert(cfg.rope_scaling.rope_type == 'llama3', 'rope_scaling type');

// --- RoPE cache ------------------------------------------------------
let rc = llama.rope_cache(cfg, S);

assert(length(rc.cos.shape()) == 2, 'cos rank');
assert(rc.cos.shape()[0] == S && rc.cos.shape()[1] == cfg.head_dim,
	`cos shape (${S}, ${cfg.head_dim})`);

// --- full model load + forward ---------------------------------------
let m = llama.load(MODEL);

assert(length(m.layers) == cfg.num_hidden_layers, 'all layers constructed');

let logits = m.forward(ids);                // (1, S, V) fp16 GPU
let sh = logits.shape();

assert(length(sh) == 3 && sh[0] == 1 && sh[1] == S && sh[2] == V,
	`logits shape (1, ${S}, ${V}), got ${sh}`);

// --- argmax must match the oracle at every position -------------------
let am_mine = logits.topk(1, 2).indices.cpu().tolist();
let am_ref = ref.topk(1, 1).indices.cpu().tolist();

for (let i = 0; i < S; i++)
	assert(am_mine[0][i][0] == am_ref[i][0],
		`argmax mismatch at position ${i}: ` +
		`${am_mine[0][i][0]} vs ${am_ref[i][0]}`);

// --- numeric agreement within fp16 noise ------------------------------
let mine = logits.view([S, V]).cpu().to('float32');
let diff = mine.sub(ref).abs().max().item();

assert(diff <= MAX_ABS_DIFF,
	`max abs logit diff ${diff} exceeds bound ${MAX_ABS_DIFF}`);

printf('llama_forward: %d positions, argmax exact, max |diff| = %.4f\n',
	S, diff);