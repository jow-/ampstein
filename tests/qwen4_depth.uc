// tests/qwen4_depth.uc — Qwen3.8-Flash-Next quality vs exllamav3 as a function
// of context depth (fixture: tests/oracle/make_qwen4_depth_oracle.py). The same
// text is prefilled in 1024-token chunks; at every checkpoint depth D the logits
// of the last ROWS positions are compared with the reference: mean NLL, KLD
// (ref || ours), top-1 agreement. Needs all six GPUs free (~100 GB of weights).
// Env: AMP_MODEL_QWEN4, AMP_QWEN4_DEPTH_FIX (fixture), AMP_QWEN4_BUDGET (indexer
// budget override for sparsity ablations; default = the checkpoint's 2048),
// AMP_QWEN4_CHUNK (default 1024), AMP_QWEN4_DEPTHS (comma list, subset of the fixture's).

import * as qwen4 from 'ampstein.qwen4';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') ||
	'/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
const FIX = getenv('AMP_QWEN4_DEPTH_FIX') || `${sourcepath(0, true)}/fixtures/qwen4_depth.safetensors`;
const ROWS = 256;
const CHUNK = +(getenv('AMP_QWEN4_CHUNK') || 1024);

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: Qwen4 model dir or depth fixture missing\n');
	exit(125);
}

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let depths = [];

for (let nm in fx.list())
	if (match(nm, /^D[0-9]+_logits$/))
		push(depths, +substr(nm, 1, length(nm) - 8));

depths = sort(depths);

if (getenv('AMP_QWEN4_DEPTHS')) {
	let want = map(split(getenv('AMP_QWEN4_DEPTHS'), ','), (x) => +x);
	depths = filter(depths, (d) => index(want, d) >= 0);
}

let dmax = depths[length(depths) - 1];

for (let d in depths)
	assert(d % CHUNK == 0, 'checkpoint depths must be multiples of the chunk size');

let t0 = time();
let model = qwen4.load(MODEL, { });

printf('loaded in %.1fs\n', time() - t0);

if (getenv('AMP_QWEN4_BUDGET')) {
	model.cfg.indexer_budget = +getenv('AMP_QWEN4_BUDGET');
	model.cfg.sparse_threshold = 4 * (model.cfg.indexer_budget / model.cfg.indexer_compress_ratio) + 3;
	printf('indexer budget overridden: %d tokens\n', model.cfg.indexer_budget);
}

let cache = model.new_cache(dmax + 64);
let dev = 'cuda:' + model.mixer.dev;
let tp0 = clock(true);
let rows = {};

for (let c0 = 0; c0 < dmax; c0 += CHUNK) {
	let lg = model.forward_chunk(slice(ids, c0, c0 + CHUNK), cache).view([CHUNK, -1]);

	if (index(depths, c0 + CHUNK) >= 0)
		rows[c0 + CHUNK] = lg.narrow(0, CHUNK - ROWS, ROWS).to('float32').contiguous();

	if ((c0 / CHUNK) % 16 == 15) {
		torch.synchronize();
		let c1 = clock(true);
		printf('  %6d tokens, %.0f tok/s\n', c0 + CHUNK,
			(c0 + CHUNK) / ((c1[0] - tp0[0]) + (c1[1] - tp0[1]) / 1000000000.0));
	}
}

torch.synchronize();
print('depth    NLL ours     NLL ref      diff    KLD mean   KLD max   top1\n');

for (let d in depths) {
	let ours = rows[d];
	let ref = fx.tensor(`D${d}_logits`).cuda(model.mixer.dev).to('float32');
	let V = min(ours.shape()[1], ref.shape()[1]);

	ours = ours.narrow(1, 0, V).contiguous();
	ref = ref.narrow(1, 0, V).contiguous();

	let po = ours.softmax(-1), pr = ref.softmax(-1);
	let tgt = torch.from_array(slice(ids, d - ROWS + 1, d), 'int64', dev).view([-1, 1]);
	let nll_o = po.narrow(0, 0, ROWS - 1).gather(1, tgt).log().neg().view([-1]);
	let nll_r = fx.tensor(`D${d}_nll`).cuda(model.mixer.dev);
	let kl = pr.mul(pr.add(1e-30).log().sub(po.add(1e-30).log())).sum(-1);
	let agree = ours.argmax(-1).sub(ref.argmax(-1)).abs().le(0).to('float32').sum().item();

	printf('%6d  %10.5f  %10.5f  %+8.5f  %9.5f  %9.5f  %3d/%d\n', d, nll_o.mean().item(),
		nll_r.mean().item(), nll_o.mean().item() - nll_r.mean().item(), kl.mean().item(),
		kl.max().item(), agree, ROWS);
}

try { torch.synchronize(); } catch (e) {}
print('qwen4_depth done\n');
