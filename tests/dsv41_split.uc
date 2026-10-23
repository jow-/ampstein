// tests/dsv41_split.uc — DeepSeek-V4.1 with the first layers on a worker (docs/RPC.md, DeepSeek-V4.1): the
// coordinator (dsv41_tp with opts.remote) and the worker (a partial model with opts.no_head) talk through
// rpc_stage over loopback TCP, in this process; the logits must equal those of the whole model (same ranks, same
// layers): chunked prefill, wavefront prefill, eager steps, graph steps, a multi-row step and its rollback, and
// a snapshot of the state (aux_get / aux_set) that is restored after other work.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 12)  AMP_DSV41_REMOTE (layers on the worker, default 8)
//   AMP_DSV41_PREFILL (default 300)    Needs 4 GPUs.

import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as rpc from 'ampstein.rpc';
import * as stage from 'ampstein.rpc_stage';
import * as socket from 'socket';
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

const NL = +(getenv('AMP_DSV41_LAYERS') || 12);
const NR = +(getenv('AMP_DSV41_REMOTE') || 8);
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

/* the sequence on a model: prefill logits, wavefront prefill, 5 eager steps, graph rows with a rollback, and the
 * snapshot of the state restored after other work */
function run(m) {
	let out = {};
	let cache = m.new_cache(MAXSEQ);

	out.pre = m.prefill(ids, cache, 128, 1).to('float32').cpu();

	let cache2 = m.new_cache(MAXSEQ);
	let pl = m.prefill_pipeline(cache2, ids, {chunk: 128});

	while (pl.step(1)) {}

	out.pipe = pl.logits().to('float32').cpu();
	out.steps = [];

	for (let t in toks)
		push(out.steps, m.forward_chunk([ t ], cache, 1).to('float32').cpu());

	let cache3 = m.new_cache(MAXSEQ);
	let g = m.graphs(cache3, [ 1, 5 ]);

	m.aux_reset(cache3);
	cache3.length = 0;
	m.prefill(ids, cache3, 128, 0);

	let snap = m.aux_get(cache3), at = cache3.length;

	out.g1 = g.run([ toks[0] ], cache3.length).logits.to('float32').cpu();

	let r5 = g.rows(toks, cache3.length);

	out.g5 = r5.logits.to('float32').cpu();
	m.rollback(cache3, cache3.length - 3);               // keep the first 2 of the 5 rows
	out.gl = g.run([ 55555 ], cache3.length).logits.to('float32').cpu();

	// back to the snapshot (the state of the pools and windows at `at`), and the same step again
	cache3.length = at;
	m.aux_set(cache3, snap);
	out.gs = g.run([ toks[0] ], at).logits.to('float32').cpu();

	return out;
}

let a = run(dsv41_tp.load(MODEL, {devices: 2, stages: 1, layers: NL, max_seq: MAXSEQ, prefill_rows: 128}));

gc();
torch.empty_cache();

function pair() {
	let l = socket.listen('127.0.0.1', 0, null, 4);
	let c = socket.connect('127.0.0.1', l.sockname().port);
	let s = l.accept();

	l.close();

	return [ rpc.conn(c), rpc.conn(s) ];
}

let p = pair();
let worker_model = dsv41_tp.load(MODEL, {devices: [ 0, 1 ], stages: 1, layers: NR, no_head: true, max_seq: MAXSEQ, prefill_rows: 128});
let W = stage.worker(p[1], worker_model), K = stage.client(p[0]);

// (one process: the worker takes every request that is queued before the coordinator waits for an answer)
K.pump = function () {
	for (;;) {
		let f = p[1].recv(20);

		if (!f)
			break;

		W.step(f);
	}
};

let b = run(dsv41_tp.load(MODEL, {devices: [ 2, 3 ], stages: 1, layers: NL, max_seq: MAXSEQ, prefill_rows: 128,
	remote: K, remote_layers: NR}));
let worst = 0;

function cmp(name, x, y) {
	let e = rel_rms(x, y);

	printf('%-10s rel rms %.2e  argmax %d %d\n', name, e, x.argmax(-1).tolist()[0], y.argmax(-1).tolist()[0]);
	worst = max(worst, e);

	if (x.argmax(-1).tolist()[0] != y.argmax(-1).tolist()[0])
		die(`FAIL: ${name}: another greedy token\n`);
}

cmp('prefill', b.pre, a.pre);
cmp('pipeline', b.pipe, a.pipe);

for (let i = 0; i < length(toks); i++)
	cmp('step ' + i, b.steps[i], a.steps[i]);

cmp('graph 1', b.g1, a.g1);
cmp('rows 5', b.g5, a.g5);
cmp('rollback', b.gl, a.gl);
cmp('snapshot', b.gs, b.g1);       // the same step as g1, from the restored state

assert(worst < 0.05, 'the split model\'s logits differ from the whole model\'s');
printf('dsv41_split: all ok\n');
