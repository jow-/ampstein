// DeepSeek-V4 vision tower vs exllamav3 (fixtures from tests/oracle/make_dsv4_vision_oracle.py,
// DSV4_VFIX = fixture directory, images next to them).
import * as dv from 'ampstein.dsv4_vision';
import { readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const FIX = getenv('AMP_DSV4_VFIX') || `${getenv('HOME')}/dsv4/fixtures`;

if (!readfile(`${MODEL}/config.json`) || !readfile(`${FIX}/dsv4_vision_odd.png`)) {
	warn('skip: DSV4 model dir or vision fixtures missing\n');
	exit(125);
}

let fails = 0;

function check(name, ok, info) {
	warn((ok ? 'ok: ' : 'FAIL: ') + name + (info ? ' — ' + info : '') + '\n');

	if (!ok)
		fails++;
}

function rel(a, b) {
	a = a.to('float32').cuda(0);
	b = b.to('float32').cuda(0);

	return a.sub(b).pow(2).mean().item() ** 0.5 / (b.pow(2).mean().item() ** 0.5 + 1e-12);
}

function cosine(a, b) {
	a = a.to('float32').cuda(0).view([-1]);
	b = b.to('float32').cuda(0).view([-1]);

	return a.mul(b).sum().item() / ((a.mul(a).sum().item() ** 0.5) * (b.mul(b).sum().item() ** 0.5));
}

let T = dv.load(MODEL, { device: 0 });

for (let name in [ 'odd', 'wide' ]) {
	let fx = torch.st(`${FIX}/dsv4_vision_oracle_${name}.safetensors`);
	let dims = fx.tensor('dims').tolist();
	let png = readfile(`${FIX}/dsv4_vision_${name}.png`);
	let p = T.preprocess(png);

	check(`[${name}] geometry`, p.nvh == dims[0] && p.nvw == dims[1] && p.nh == dims[2] && p.nw == dims[3],
		`${p.nvh}x${p.nvw} / ${p.nh}x${p.nw} vs ${dims[0]}x${dims[1]} / ${dims[2]}x${dims[3]}`);

	let ref = fx.tensor('patches');
	let d = p.patches.to('float32').sub(ref.to('float32')).abs().max().item();

	check(`[${name}] patches vs PIL`, d <= 0.02, `max |diff| ${d}`);

	let tr = [];
	let out = T.forward(p.patches, p.nvh, p.nvw, tr);

	check(`[${name}] block 0`, rel(tr[0], fx.tensor('b0')) < 0.02, `rel-rms ${rel(tr[0], fx.tensor('b0'))}`);
	check(`[${name}] last block`, rel(tr[1], fx.tensor('bl')) < 0.1, `rel-rms ${rel(tr[1], fx.tensor('bl'))} cosine ${cosine(tr[1], fx.tensor('bl'))}`);
	check(`[${name}] unfold`, rel(tr[2], fx.tensor("unf")) < 0.13, `rel-rms ${rel(tr[2], fx.tensor('unf'))}`);
	check(`[${name}] aligner output`, cosine(out, fx.tensor("out")) > 0.99,
		`cosine ${cosine(out, fx.tensor('out'))} rel-rms ${rel(out, fx.tensor('out'))}`);

	p.out = out;

	let b = T.block(p, 0);
	let types = fx.tensor('types').tolist();

	check(`[${name}] block layout`, join(',', b.types) == join(',', types) && b.lead == dims[6],
		`${length(b.types)} rows vs ${length(types)}`);
	check(`[${name}] block rows`, cosine(b.rows, fx.tensor("block")) > 0.99,
		`cosine ${cosine(b.rows, fx.tensor('block'))}`);

	// the lead pad count follows the start position (START lands on p % 4 == 3)
	for (let sp in [ 0, 1, 2, 3, 6 ]) {
		let bb = dv.build_image_block(p.nh, p.nw, sp);

		check(`[${name}] START phase at start_pos ${sp}`, (sp + bb.lead) % 4 == 3, `lead ${bb.lead}`);
	}
}

warn(fails ? 'dsv4_vision FAILED\n' : 'dsv4_vision ok\n');
exit(fails ? 1 : 0);
