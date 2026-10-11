// tests/glm5_long.uc — GLM-5.3-Flash beyond index_topk: the k-pool DSA
// indexer. Fixture tests/fixtures/glm5_long_oracle.safetensors
// (make_glm5_long_oracle.py, ~3.7k-token retrieval prompt).
//
// Part A (one layer, ~1 GiB): layer 3's attention over the whole prompt,
//   chunked like a real prefill (512) through the latent + indexer cache:
//   the selected token sets of the last 64 queries and the attention
//   output vs exllamav3's.
// Part B (AMP_GLM5_FULL=1, all six GPUs): last-16 logits and the greedy
//   continuation of the whole model.

import * as glm5 from 'ampstein.glm5';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') ||
	'/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_long_oracle.safetensors`;
const DEV = +(getenv('AMP_GLM5_DEV') || 0);

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: GLM model dir or long oracle fixture missing\n');
	exit(125);
}

let cfg = glm5.load_config(MODEL);
let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let T = length(ids);

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() /
		b.to('float32').mul(b.to('float32')).mean().item());
}

assert(T > cfg.index_topk, 'fixture prompt exceeds index_topk');

/* ---- Part A: layer 3 -------------------------------------------------- */
let ws = weights.open(MODEL);
let A = glm5.build_mla(ws, cfg, 3, DEV);
let x = fx.tensor('l3_attn_in').cuda(DEV).reshape([T, -1]);
let want = fx.tensor('l3_attn_out').cuda(DEV).reshape([T, -1]);
let cl = glm5.new_mla_cache(cfg, T + 8, DEV);

cl.dbg = [];

let outs = [];
const CHUNK = 512;

for (let p = 0; p < T; p += CHUNK) {
	let n = (T - p < CHUNK) ? T - p : CHUNK;

	push(outs, glm5.mla_forward(cfg, A, x.narrow(0, p, n).contiguous(), cl, p));
}

let y = torch.cat(outs, 0);
let e = rel_rms(y, want);

printf('layer 3 sparse MLA over %d tokens: rel-rms %g (last 64 rows: %g)\n', T, e,
	rel_rms(y.narrow(0, T - 64, 64), want.narrow(0, T - 64, 64)));

// selected-index sets of the last 64 queries
let got = torch.cat(cl.dbg, 0);
let rows = got.shape()[0];
let mine = got.narrow(0, rows - 64, 64).cpu().tolist();
let ref = fx.tensor('l3_idx').tolist();
let exact = 0, worst = 1.0;

for (let r = 0; r < 64; r++) {
	let a = {}, b = {}, na = 0, nb = 0, inter = 0;

	for (let v in mine[r]) if (v >= 0) { a[v] = 1; na++; }
	for (let v in ref[r]) if (v >= 0) { b[v] = 1; nb++; if (a[v]) inter++; }

	let frac = (nb > 0) ? (inter * 1.0) / nb : 1.0;   // NB: float division

	if (na == nb && inter == nb) exact++;
	if (frac < worst) worst = frac;
}

printf('indexer: %d/64 query rows select exactly the reference set, worst overlap %.4f\n',
	exact, worst);

let failures = [];

if (!(e < 0.03)) push(failures, 'layer 3 sparse output');
if (!(worst > 0.99)) push(failures, 'indexer selection overlap');

/* ---- Part B: whole model ----------------------------------------------- */
if (getenv('AMP_GLM5_FULL')) {
	let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6) });
	let cache = model.new_cache(T + 64);
	let t0 = time();
	let logits = model.prefill(ids, cache, 512, 16);

	torch.synchronize();
	printf('prefill %d tokens: %.1fs\n', T, time() - t0);

	let lw = fx.tensor('logits');
	let le = rel_rms(logits.cpu(), lw);
	let same = 0;
	let ag = logits.cpu().argmax(-1).tolist(), aw = lw.argmax(-1).tolist();

	for (let i = 0; i < 16; i++) if (ag[i] == aw[i]) same++;

	// KL(reference || ours) per position
	let pr = lw.to('float32').softmax(-1), pq = logits.cpu().to('float32').softmax(-1);
	let kl = pr.mul(pr.add(1e-30).log().sub(pq.add(1e-30).log())).sum(-1);
	let klm = kl.mean().item(), klx = kl.max().item();

	printf('last-16 logits: rel-rms %g, argmax equal %d/16, KLD mean %g max %g\n',
		le, same, klm, klx);

	if (!(klm < 0.02)) push(failures, 'KLD');

	let cont = fx.tensor('cont').tolist();
	let out = [];
	let lg = logits.narrow(0, 15, 1);

	for (let k = 0; k < length(cont); k++) {
		let t = lg.reshape([-1]).argmax().item();

		push(out, t);
		lg = model.forward_chunk([t], cache, 1);
	}

	printf('continuation: %s\noracle      : %s\n', join(',', out), join(',', cont));

	let fd = -1;

	for (let k = 0; k < length(cont); k++) if (out[k] != cont[k]) { fd = k; break; }

	if (!(fd < 0 || fd >= 6)) push(failures, `continuation diverges at ${fd}`);
	if (same < 12) push(failures, 'argmax agreement');
}

try { torch.synchronize(); } catch (e2) {}
assert(!length(failures), 'glm5_long: ' + join('; ', failures));
print('glm5_long: ok\n');
