// Side-stream stage hand-off vs the plain copy: same prompt through the layer-split wavefront
// prefill, logits compared (the prefill is not bit-deterministic: MoE atomics, so a noise baseline
// from two identical runs is measured first). A race on the hand-off buffers would show as garbage.
import * as dsv4 from 'ampstein.dsv4';
import { readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';

if (!readfile(`${MODEL}/config.json`)) {
	warn('skip: DSV4 model dir missing\n');
	exit(125);
}

const NL = +(getenv('AMP_DSV4_LAYERS') || 12), ND = +(getenv('AMP_DSV4_DEVICES') || 6);
let model = dsv4.load(MODEL, { layers: NL, devices: ND, prefill_rows: 1024 });
let ids = [ 0 ];

for (let i = 1; i < 4096; i++)
	push(ids, 1000 + (i * 7919) % 90000);

function run(side) {
	dsv4.set_side_ho(side);

	let cache = model.new_cache(8192);
	let r = model.prefill_pipeline(cache, ids, { chunk: 1024 });

	while (r.step(8)) ;

	torch.synchronize();

	return r.logits().to('float32').cpu();
}

function rel(a, b) {
	let d = a.sub(b);

	return (d.mul(d).mean().item() ** 0.5) / (b.mul(b).mean().item() ** 0.5);
}

let a = run(false), b = run(false), c = run(true), d = run(true);
let noise = rel(a, b);
let cross = rel(c, a);
let am = a.argmax(-1).item(), cm = c.argmax(-1).item(), dm = d.argmax(-1).item();

warn(`noise (plain vs plain) ${noise}, side vs plain ${cross}, side vs side ${rel(c, d)}; argmax ${am} ${cm} ${dm}\n`);

let ok = cross < 10 * noise + 1e-3 && am == cm && am == dm;

warn(ok ? 'dsv4_sideho ok\n' : 'dsv4_sideho FAILED\n');
exit(ok ? 0 : 1);
