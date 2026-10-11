// tests/dsv4_verify.uc — the multi-row (speculative verify) graph step of dsv4_tp:
//   (a) one S-row step equals S sequential single-token steps (logits per row),
//   (b) a rollback (position reset to the accepted prefix) followed by new tokens equals a
//       cache that never saw the rejected rows (the compressor ring / pool counters are
//       position-derived, so there is nothing to repair).
// Layer subset (DSV4_LAYERS, default 10) over 6 GPUs in 3 stages; the prompt is long enough
// for the CSA indexer top-k regime and the steps cross compressor window boundaries.

import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';

if (!stat(MODEL)) {
	warn('skip: DSV4 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV4_LAYERS') || 10);
const S = +(getenv('AMP_VERIFY_S') || 6);
const N = +(getenv('AMP_VERIFY_N') || 2301);              // 2301: the steps straddle windows of 4 and 128 (2304)

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / b.to('float32').mul(b.to('float32')).mean().item());
}

let tp = dsv4_tp.load(MODEL, { layers: NL, devices: 6, stages: 3 });
let ids = [ 0 ], x = 99;

for (let i = 1; i < N; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let ms = N + 64;

function fresh() {
	let c = tp.new_cache(ms);
	let run = tp.prefill_pipeline(c, ids, { chunk: 512 });

	while (run.step(1)) {}

	return c;
}

// graphs capture on a cache: build one per cache (the warm-up clobbers it, so before the prefill)
function with_graphs(sizes) {
	let c = tp.new_cache(ms);
	let g = tp.graphs(c, sizes);

	tp.aux_reset(c);
	c.length = 0;

	let run = tp.prefill_pipeline(c, ids, { chunk: 512 });

	while (run.step(1)) {}

	return { c: c, g: g };
}

let toks = [], y = 5;

for (let i = 0; i < S + 4; i++) {
	y = (y * 1103515245 + 12345) % 2147483648;
	push(toks, 1000 + (y % 60000));
}

let A = with_graphs([ 1 ]);
let B = with_graphs([ 1, S ]);
let C = with_graphs([ 1 ]);

torch.synchronize();

let ok = true;

// (a)
let rows = [];

for (let i = 0; i < S; i++)
	push(rows, A.g.run([ toks[i] ], N + i).logits.add(0.0));

let rb = B.g.run(slice(toks, 0, S), N).logits;
let worst = 0.0, agree = 0;

for (let i = 0; i < S; i++) {
	let r = rel_rms(rb.narrow(0, i, 1), rows[i]);

	if (r > worst) worst = r;
	if (rb.narrow(0, i, 1).view([-1]).argmax(-1).item() == rows[i].view([-1]).argmax(-1).item()) agree++;
}

printf('S=%d step vs sequential: worst rel-rms %.2e, argmax agree %d/%d\n', S, worst, agree, S);
ok = ok && worst < 0.1 && agree >= S - 1;

// (b) accept 3 rows, continue with new tokens
let keep = 3;
let ra = [], rc = [];

B.c.length = N + keep;

for (let i = 0; i < 3; i++) {
	push(ra, B.g.run([ toks[S + i] ], N + keep + i).logits.add(0.0));
}

for (let i = 0; i < keep; i++)
	C.g.run([ toks[i] ], N + i);

for (let i = 0; i < 3; i++)
	push(rc, C.g.run([ toks[S + i] ], N + keep + i).logits.add(0.0));

let wb = 0.0;

for (let i = 0; i < 3; i++) {
	let r = rel_rms(ra[i], rc[i]);

	if (r > wb) wb = r;
}

printf('after rollback to %d accepted rows: worst rel-rms vs a clean cache %.2e\n', keep, wb);
ok = ok && wb < 0.1;

try { torch.synchronize(); } catch (e) {}

print(ok ? 'dsv4_verify ok\n' : 'dsv4_verify FAILED\n');
exit(ok ? 0 : 1);
