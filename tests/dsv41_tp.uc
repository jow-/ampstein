// tests/dsv41_tp.uc — DeepSeek-V4.1 as 2 stages x 2 tensor-parallel ranks (dsv41_tp) against the single-chain model:
// chunked prefill logits, wavefront prefill, graph decode, a multi-row step and its rollback, the host-RAM store.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 8)  AMP_DSV41_PREFILL (default 300)

import * as dsv41 from 'ampstein.dsv41';
import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV41') || '/data/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw';

if (!stat(MODEL + '/model.safetensors.index.json')) {
	warn('skip: DSV4.1 model dir missing\n');
	exit(125);
}

if (torch.device_count() < 4) {
	warn('skip: needs 4 GPUs\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV41_LAYERS') || 8);
const NP = +(getenv('AMP_DSV41_PREFILL') || 300);
const MAXSEQ = NP + 64;

let ids = [], x = 12345;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let toks = [ 4242, 777, 12345, 999, 31337 ];

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

/* run the sequence on a model: prefill logits, 5 eager decode logits (fresh cache), graph rows */
function run(m, tp) {
	let out = {};
	let cache = m.new_cache(MAXSEQ);

	out.pre = m.prefill(ids, cache, 128, 1).to('float32').cpu();

	let cache2 = m.new_cache(MAXSEQ);
	let pl = m.prefill_pipeline(cache2, ids, {chunk: 128});

	while (pl.step(1)) {}

	out.pipe = pl.logits().to('float32').cpu();

	// the host-RAM store: the prefilled cache to host memory, another sequence over it, the cache back; a step
	// then gives what it gave before (the same computation: identical up to the order of the all-reduces)
	if (tp && m.cache_export) {
		let put = function (t) {
			let h = torch.pinned_empty(t.shape(), t.dtype());

			h.copy_(t);

			return h;
		};

		let n2 = cache2.length;
		let em = m.cache_export(cache2, n2, put, null, null);

		out.st0 = m.forward_chunk([ toks[1] ], cache2, 1).to('float32').cpu();
		m.reset_cache(cache2);
		m.prefill(map(ids, (t) => 1000 + (t * 7) % 60000), cache2, 128, 0);
		m.cache_import(cache2, em, n2, true);
		out.st1 = m.forward_chunk([ toks[1] ], cache2, 1).to('float32').cpu();
	}

	out.steps = [];

	for (let t in toks)
		push(out.steps, m.forward_chunk([ t ], cache, 1).to('float32').cpu());

	// graphs on a third cache: single steps and a 5-row step
	let cache3 = m.new_cache(MAXSEQ);
	let g = m.graphs(cache3, [ 1, 5 ]);

	m.prefill(ids, cache3, 128, 0);
	out.g1 = g.run([ toks[0] ], cache3.length).logits.to('float32').cpu();

	let r5 = g.rows(toks, cache3.length);

	out.g5 = r5.logits.to('float32').cpu();
	m.rollback(cache3, cache3.length - 3);               // keep the first 2 of the 5 rows
	out.gl = g.run([ 55555 ], cache3.length).logits.to('float32').cpu();

	return out;
}

let a = run(dsv41.load(MODEL, {devices: 2, layers: NL, max_seq: MAXSEQ, prefill_rows: 128}), false);

gc();

let b = run(dsv41_tp.load(MODEL, {devices: 4, stages: 2, layers: NL, max_seq: MAXSEQ, prefill_rows: 128}), true);
let worst = 0;

function cmp(name, x, y) {
	let e = rel_rms(x, y);

	printf('%-10s rel rms %.2e  argmax %d %d\n', name, e, x.argmax(-1).tolist()[0], y.argmax(-1).tolist()[0]);
	worst = max(worst, e);
}

cmp('prefill', b.pre, a.pre);
cmp('pipeline', b.pipe, a.pipe);

for (let i = 0; i < length(toks); i++)
	cmp('step ' + i, b.steps[i], a.steps[i]);

cmp('graph 1', b.g1, a.g1);
cmp('rows 5', b.g5, a.g5);

assert(worst < 0.1, 'tensor-parallel logits differ from the single-chain model');

if (b.st0) {
	let e = rel_rms(b.st1, b.st0);

	printf('%-10s rel rms %.2e  argmax %d %d\n', 'store', e, b.st1.argmax(-1).tolist()[0], b.st0.argmax(-1).tolist()[0]);
	assert(e < 1e-3, 'a step after the cache store round trip differs from the step before it');
}

printf('dsv41_tp: all ok\n');
