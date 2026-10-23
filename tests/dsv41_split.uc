// tests/dsv41_split.uc — DeepSeek-V4.1 with the first layers on a worker (docs/RPC.md, DeepSeek-V4.1): the
// coordinator (dsv41_tp with opts.remote) and the worker (a partial model with opts.no_head) talk through
// rpc_stage over loopback TCP, in this process; the logits must equal those of the whole model (same ranks, same
// layers): chunked prefill, wavefront prefill, eager steps, graph steps, a multi-row step and its rollback, and
// a snapshot of the state (aux_get / aux_set) that is restored after other work.
//   AMP_MODEL_DSV41  AMP_DSV41_LAYERS (default 12)  AMP_DSV41_REMOTE (layers on the worker, default 8)
//   AMP_DSV41_PREFILL (default 300)    Needs 4 GPUs (jai: 3-4 for the worker, 0-1 here).
//   AMP_SPLIT_CMD  how to start the installed server script (the worker is a process of its own: it cannot share
//     this one, a hop is megabytes and the socket would block), e.g. "ucode R/prefix/bin/ampstein".
//     AMP_DSV41_REF=file keeps the whole model's results, which
//     are made in a first run (the GPUs are then free of its state for the split model's run).

import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as rpc_coord from 'ampstein.rpc_coord';
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

const PORT = +(getenv('AMP_DSV41_PORT') || 7760);

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

	warn('run: prefill\n');
	out.pre = m.prefill(ids, cache, 128, 1).to('float32').cpu();

	warn('run: pipeline\n');
	let cache2 = m.new_cache(MAXSEQ);
	let pl = m.prefill_pipeline(cache2, ids, {chunk: 128});

	while (pl.step(1)) {}

	out.pipe = pl.logits().to('float32').cpu();
	warn('run: steps\n');
	out.steps = [];

	for (let t in toks)
		push(out.steps, m.forward_chunk([ t ], cache, 1).to('float32').cpu());

	warn('run: graphs\n');
	let cache3 = m.new_cache(MAXSEQ);
	let g = m.graphs(cache3, [ 1, 5 ]);

	warn('run: graphs captured\n');
	m.aux_reset(cache3);
	cache3.length = 0;
	m.prefill(ids, cache3, 128, 0);

	warn('run: decode\n');
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

// the whole model's results; AMP_DSV41_REF=file keeps them, so that a rerun (in a fresh process: the GPUs are
// then free of the whole model's state) loads the split model only
const REF = getenv('AMP_DSV41_REF');
let a = null;

if (REF && stat(REF)) {
	let fx = torch.st(REF);

	a = {pre: fx.tensor('pre'), pipe: fx.tensor('pipe'), g1: fx.tensor('g1'), g5: fx.tensor('g5'), gl: fx.tensor('gl'), gs: fx.tensor('gs'), steps: []};

	for (let i = 0; i < length(toks); i++)
		push(a.steps, fx.tensor('step' + i));

	warn(`reference from ${REF}\n`);
}
else {
	a = run(dsv41_tp.load(MODEL, {devices: 2, stages: 1, layers: NL, max_seq: MAXSEQ, prefill_rows: 128}));

	if (REF) {
		let l = [ [ 'pre', a.pre ], [ 'pipe', a.pipe ], [ 'g1', a.g1 ], [ 'g5', a.g5 ], [ 'gl', a.gl ], [ 'gs', a.gs ] ];

		for (let i = 0; i < length(toks); i++)
			push(l, [ 'step' + i, a.steps[i] ]);

		torch.st_save(REF, l);
		warn(`reference saved to ${REF}: run again for the split model\n`);
		exit(0);
	}

	gc();
	torch.empty_cache();
}

if (!getenv('AMP_SPLIT_CMD')) {
	warn('skip: AMP_SPLIT_CMD missing (see the header)\n');
	exit(125);
}

// the worker, a process of its own on GPUs 3 and 4 (it sees them as 0 and 1: a one-stage model on other
// devices than 0, 1 fails in the p2p communicator, 'unspecified launch failure', on jai); this process takes GPUs 0 and 1
system(`CUDA_VISIBLE_DEVICES=3,4 ${getenv('AMP_SPLIT_CMD')} --worker --connect 127.0.0.1:${PORT} --model ${MODEL} --gpus 2 > /tmp/dsv41_split_worker.log 2>&1 &`);

let ex = rpc_coord.attach({listen: `127.0.0.1:${PORT}`, model: MODEL, gpus: 2, stages: 1, max_seq: MAXSEQ, prefill_rows: 128,
	family: 'dsv41', layers: NR, wait: 900});

let b = run(dsv41_tp.load(MODEL, {devices: [ 0, 1 ], stages: 1, layers: NL, max_seq: MAXSEQ, prefill_rows: 128,
	...ex.load}));
let worst = 0;

function cmp(name, x, y) {
	let e = rel_rms(x, y);

	// (a failing check ends the test: the worker process is told first, else it waits for the next coordinator)

	printf('%-10s rel rms %.2e  argmax %d %d\n', name, e, x.argmax(-1).tolist()[0], y.argmax(-1).tolist()[0]);
	worst = max(worst, e);

	if (x.argmax(-1).tolist()[0] != y.argmax(-1).tolist()[0]) {
		ex.client.bye();
		die(`FAIL: ${name}: another greedy token\n`);
	}
}

cmp('prefill', b.pre, a.pre);
cmp('pipeline', b.pipe, a.pipe);

for (let i = 0; i < length(toks); i++)
	cmp('step ' + i, b.steps[i], a.steps[i]);

cmp('graph 1', b.g1, a.g1);
cmp('rows 5', b.g5, a.g5);
cmp('rollback', b.gl, a.gl);
cmp('whole: snapshot', a.gs, a.g1);       // (the whole model's own: is the premise of the next check sound?)
cmp('snapshot', b.gs, b.g1);       // the same step as g1, from the restored state

ex.client.bye();

// The bound of dsv41_tp.uc: the worker's GPUs and the coordinator's differ in kind on jai (GPU 3 is an A5500, the rest A5000: another SM
// count, another split of the kernels' work), and the whole model on GPUs 3,4 differs from the one on 0,1 by exactly the
// amount seen here (logits rel-rms 7e-3 .. 8e-2 after 12 layers); the greedy tokens agree at every step.
assert(worst < 0.1, 'the split model\'s logits differ from the whole model\'s');
printf('dsv41_split: all ok\n');
