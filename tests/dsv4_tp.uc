// tests/dsv4_tp.uc — dsv4_tp (pairs of GPUs, pipeline of stages) vs the layer-split model on a
// layer subset: prefill logits, eager decode and graph decode. Env: DSV4_LAYERS (default 10),
// DSV4_TP_DEVICES (6), DSV4_TP_STAGES (3), TP_N (prompt tokens, default 700), TP_STEPS (24).
// The layer-split reference sits on the first 2 GPUs (layers/2 each), the TP model on all.

import * as dsv4 from 'ampstein.dsv4';
import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';

if (!stat(MODEL)) {
	warn('skip: DSV4 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV4_LAYERS') || 10);
const ND = +(getenv('AMP_DSV4_TP_DEVICES') || 6);
const NS = +(getenv('AMP_DSV4_TP_STAGES') || 3);
const N = +(getenv('AMP_TP_N') || 700);
const STEPS = +(getenv('AMP_TP_STEPS') || 24);

function rel_rms(a, b) {
	let x = a.to('float32'), y = b.to('float32').cuda(0);
	let d = x.cuda(0).sub(y);

	return math.sqrt(d.mul(d).mean().item() / y.mul(y).mean().item());
}

let ids = [ 0 ], x = 777;

for (let i = 1; i < N; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let ref = dsv4.load(MODEL, { layers: NL, devices: 2 });
let ca = ref.new_cache(N + STEPS + 16);
let la = ref.prefill(ids, ca, 512, 1);

let tp = dsv4_tp.load(MODEL, { layers: NL, devices: ND, stages: NS });
let cb = tp.new_cache(N + STEPS + 16);
let run = tp.prefill_pipeline(cb, ids, { chunk: 512 });

while (run.step(1)) {}

let lb = run.logits();

printf('prefill logits TP vs split: rel-rms %.2e, argmax %d vs %d\n', rel_rms(lb, la),
	lb.view([-1]).argmax(-1).item(), la.view([-1]).argmax(-1).item());

let ok = (rel_rms(lb, la) < 0.1);
let tok = la.view([-1]).argmax(-1).item();
let cc = tp.new_cache(N + STEPS + 16);

tp.prefill(ids, cc, 512, 0);

let graph = tp.graphs(cc, [ 1 ]);

torch.synchronize();

// base: eager decode of TP against eager decode of the split model, graph decode against TP eager
let worst_e = 0.0, worst_g = 0.0, agree = 0;

for (let i = 0; i < STEPS; i++) {
	let le = ref.forward_chunk([ tok ], ca).view([-1]);
	let lt = tp.forward_chunk([ tok ], cb).view([-1]);
	let lg = graph.run([ tok ], cc.length).logits.view([-1]);
	let re = rel_rms(lt, le), rg = rel_rms(lg, lt);

	if (re > worst_e) worst_e = re;
	if (rg > worst_g) worst_g = rg;
	if (lt.argmax(-1).item() == le.argmax(-1).item()) agree++;

	tok = le.argmax(-1).item();
}

printf('decode over %d steps: TP eager vs split worst rel-rms %.2e (argmax agree %d), TP graph vs TP eager worst %.2e\n',
	STEPS, worst_e, agree, worst_g);
try { torch.synchronize(); } catch (e) {}

ok = ok && agree >= STEPS - 6 && worst_e < 0.3 && worst_g < 0.3;
print(ok ? 'dsv4_tp ok\n' : 'dsv4_tp FAILED\n');
exit(ok ? 0 : 1);
