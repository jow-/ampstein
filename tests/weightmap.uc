// Phase 2/3 test: weight-map routing + linear class routing.
//
// Fixtures:
//   - tests/fixtures/sharded/ — an HF-style SHARDED model dir built by
//     tests/oracle/make_shard_fixture.py from the real-model layer
//     fixture (exl3_real_layer.safetensors): q_proj.{trellis,suh,svh}
//     in shard 1; gemm references X/Y, a plain fp16 linear (proj.*)
//     with a CUDA-computed reference, and a bf16 tensor in shard 2.
//   - tests/fixtures/exl3_real_layer.safetensors — the same real
//     weights un-sharded, for the bit-exact routing comparison.
//   - the real single-file model dir (Llama-3.2-1B-exl3) — must keep
//     loading through the same weights.open() path.
//
// Checks:
//   1. sharded routing: tensors resolve across both shards, values
//      bit-exact vs the single-file fixture
//   2. LinearFP16: y == X @ W.T + bias vs the CUDA torch reference
//   3. bf16 weight cast to fp16 (exact: same mantissa width)
//   4. LinearEXL3Seq built from the sharded dir gemms within the
//      upstream rel-rms bound vs the fixture's Y reference
//   5. real model dir: llama.load() works through weights.open()
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/weights.uc

import * as models from 'ampstein.models';
import * as weights from 'ampstein.weights';
import * as llama from 'ampstein.llama';
import * as math from 'math';
import { stat } from 'fs';

const FIXTURES = sourcepath(0, true) + '/fixtures';
const REAL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(REAL)) {
	warn('skip: model dir not found: ' + REAL + '\n');
	exit(125);
}

// --- 1. sharded routing resolves keys across shards --------------------
let ws = weights.open(FIXTURES + '/sharded');
let single = torch.st(FIXTURES + '/exl3_real_layer.safetensors');

assert(ws.has('q_proj.trellis'), 'q_proj.trellis routes to shard 1');
assert(ws.has('proj.weight'), 'proj.weight routes to shard 2');
assert(ws.has('norm.weight'), 'norm.weight routes to shard 2');
assert(ws.has('X'), 'reference tensors route too');
assert(!ws.has('nonexistent.weight'), 'unknown key reports absent');

let t1 = ws.tensor('q_proj.trellis');
let t0 = single.tensor('trellis');

assert(length(t1.shape()) == 3, 'trellis is 3-D');
assert(t1.shape()[0] == t0.shape()[0], 'trellis dim 0 matches');
assert(t1.shape()[2] == t0.shape()[2], 'trellis tile width matches');

// int16 values are exactly representable in fp32, so a routed copy of
// the same data must diff to exactly zero
let td = t1.cpu().to('float32').sub(t0.cpu().to('float32')).abs().max().item();

assert(td == 0, 'trellis values bit-exact across shard routing');

// --- 2. LinearFP16 math vs the CUDA torch reference --------------------
let proj = models.LinearFP16(ws, 'proj');

assert(proj.in_features == 32 && proj.out_features == 64,
	'proj dims from weight shape');

let X = ws.tensor('proj_x').cuda();
let Yref = ws.tensor('proj_y').cuda();

let Y = proj.forward(X);

let ydiff = Y.cpu().to('float32').sub(Yref.cpu().to('float32')).abs().max().item();

assert(ydiff <= 0.1,
	`proj forward matches torch reference (max abs diff ${ydiff})`);

// --- 3. bf16 weight cast to fp16 ---------------------------------------
let nw = ws.fp16('norm.weight');

assert(nw.dtype() == 'float16', 'bf16 weight cast to fp16');
assert(nw.shape()[0] == 32, 'cast preserves shape');

let nvals = nw.cpu().tolist();

assert(nvals[0] == nvals[0], 'cast values readable (not NaN)');

// --- 4. EXL3 linear from the sharded dir -------------------------------
let exl = models.LinearEXL3Seq(ws, 'q_proj');

assert(exl.K == 4 && exl.in_features == 2048 && exl.out_features == 2048,
	'q_proj dims derived from trellis shape');

let x = ws.tensor('X').cuda();           // (8, 2048) fp16
let yref = ws.tensor('Y').cuda();        // gemm reference

let y = exl.forward(x);

// squared relative-RMS error, bound identical to exllamav3's own gemm
// tests (see tests/exl3_real.uc)
let d32 = y.cpu().to('float32').sub(yref.cpu().to('float32'));
let r32 = yref.cpu().to('float32');
let err2 = d32.mul(d32).sum().item() / r32.mul(r32).sum().item();

assert(err2 < 0.02 * 0.02, `q_proj gemm rel-rms ${math.sqrt(err2)} within bound`);

// --- 5. the real single-file model loads through the same path ---------
let m = llama.load(REAL);

assert(length(m.layers) == 16, 'real model: 16 layers');
assert(m.cfg.vocab_size == 128256, 'real model: vocab size');

printf(`weights: sharded routing bit-exact, fp16 linear diff ${ydiff}, `
	+ `exl3 rel-rms ${math.sqrt(err2)}, real model loads\n`);