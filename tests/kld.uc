// Phase 4 test: per-layer numerics harness vs the exllamav3 oracle.
// Fixture: tests/fixtures/kld_oracle.safetensors — residual-stream
// hidden states after every TransformerBlock plus final logits of
// turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw), captured through
// exllamav3's own module chain (see tests/oracle/make_kld_oracle.py).
//
// This is NOT a bit-exactness test: the reference graph runs fp32
// intermediates (exllamav3 llama.py sets out_dtype=torch.float on
// attn/mlp) while our engine is fp16 end-to-end. The metrics are the
// ones exllamav3's eval tooling uses, ported to ucode in
// ucode/metrics.uc:
//
//   - per-layer rel-RMS + cosine error of the residual stream
//   - KLD(softmax(oracle) || softmax(ours)), mean over positions
//   - perplexity agreement: our ppl formula on the stored logits must
//     reproduce the oracle's ppl_f32; our engine's logits must give a
//     ppl close to it
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L ucode/ tests/kld.uc

import * as llama from 'ampstein.llama';
import * as metrics from 'ampstein.metrics';
import * as math from 'math';
import { stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/kld_oracle.safetensors`;
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

const NUM_LAYERS = 16;
const S = 128, HIDDEN = 2048;

// rel-RMS error: ||a - b|| / ||b|| over all elements.
function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));
	let num = d.mul(d).mean().item();
	let den = b.to('float32').mul(b.to('float32')).mean().item();

	return math.sqrt(num / den);
}

// cosine similarity over all elements (1 = identical direction).
function cosine(a, b) {
	let af = a.to('float32'), bf = b.to('float32');

	let dot = af.mul(bf).sum().item();
	let na = af.mul(af).sum().item();
	let nb = bf.mul(bf).sum().item();

	return dot / math.sqrt(na * nb);
}

let oracle = torch.st(FIXTURE);
let ids = oracle.tensor('ids').tolist();

assert(length(ids) == S, `expected ${S} ids, got ${length(ids)}`);

let model = llama.load(MODEL);

// --- per-layer residual stream -------------------------------------------

let trace = [];
let logits3 = model.forward(ids, trace);

assert(length(trace) == NUM_LAYERS, `trace has ${length(trace)} layers`);

let worst_rms = 0.0, worst_cos = 1.0;

for (let i = 0; i < NUM_LAYERS; i++) {
	let want = oracle.tensor(`layer${i}_h`).cuda();          // (S, hidden) fp16
	let got = trace[i].reshape([S, HIDDEN]);                // (1, S, hidden) -> (S, hidden)

	let rms = rel_rms(got, want);
	let cs = cosine(got, want);

	printf('layer %2d: rel-rms %.5f  cosine %.6f\n', i, rms, cs);

	assert(rms < 0.02, `layer ${i} rel-rms ${rms} >= 0.02`);
	assert(cs > 0.999, `layer ${i} cosine ${cs} <= 0.999`);

	if (rms > worst_rms)
		worst_rms = rms;
	if (cs < worst_cos)
		worst_cos = cs;
}

// --- KLD on the final logits ----------------------------------------------

let ours = logits3.reshape([S, -1]);                        // (S, V) fp16
let ref = oracle.tensor('logits').cuda();                   // (S, V) fp16

let rows = metrics.kld_rows(ours, ref);
let kld_mean = rows.mean().item();
let kld_max = rows.max().item();

printf('kld: mean %.6f  max %.6f\n', kld_mean, kld_max);

assert(kld_mean < 0.001, `mean KLD ${kld_mean} >= 0.001`);
assert(kld_max < 0.01, `max KLD ${kld_max} >= 0.01`);

// --- perplexity agreement --------------------------------------------------

// our ppl formula on the oracle's own logits must reproduce the oracle
// number (validates the metric implementation, not the model)
let ref_ppl = metrics.ppl(ref, ids);
let oracle_ppl = oracle.tensor('ppl_f32').item();

printf('ppl: oracle %.6f  ours-from-oracle-logits %.6f\n', oracle_ppl, ref_ppl.ppl);

assert(math.abs(ref_ppl.ppl - oracle_ppl) / oracle_ppl < 1e-4,
	`ppl formula mismatch: got ${ref_ppl.ppl}, want ${oracle_ppl}`);

// our engine's logits must give a close ppl
let our_ppl = metrics.ppl(ours, ids);

printf('ppl: our-engine %.6f  (delta %.4f%%)\n',
	our_ppl.ppl, 100.0 * (our_ppl.ppl - oracle_ppl) / oracle_ppl);

assert(math.abs(our_ppl.ppl - oracle_ppl) / oracle_ppl < 0.02,
	`engine ppl ${our_ppl.ppl} differs from oracle ${oracle_ppl} by > 2%`);

oracle.free();

printf('kld: all assertions passed (worst rel-rms %.5f, worst cosine %.6f)\n',
	worst_rms, worst_cos);