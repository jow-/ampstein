// tests/dsv41_hc2.uc — the fused one-site-late mHC pass (torch.hc_mix2 via dsv41.hc_site) against the composite
// form (hc_mixes + hc_collapse): post, comb, pre and the collapsed input for R = 1 .. 2048 rows.

import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;
const D = 5120;
const cfg = {hidden_size: D, rms_norm_eps: 1.0e-20, hc_eps: 1.0e-6, hc_sinkhorn_iters: 20};

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let fn = torch.rand([24, 4 * D], 'float32', dv).sub(0.5).mul(0.02);
let hc = {fn: fn, fn_pre_t: fn.narrow(0, 0, 4).t().contiguous(), base: torch.rand([24], 'float32', dv).sub(0.5),
	base_pre: null, scale: torch.from_array([ 0.5, 0.7, 0.9 ], 'float32', dv), scale_pre: null};

hc.base_pre = hc.base.narrow(0, 0, 4).contiguous();
hc.scale_pre = hc.scale.narrow(0, 0, 1).contiguous();

for (let R in [ 1, 2, 5, 64, 2048 ]) {
	let st = torch.rand([R, 4, D], 'float32', dv).sub(0.5).mul(R == 5 ? 30000.0 : 4.0);     // R = 5: saturating rows
	let pp = torch.rand([R, 4], 'float32', dv);
	let a = dsv41.hc_site(cfg, hc, st, pp);
	let m = dsv41.hc_mixes(cfg, hc, st);
	let col = dsv41.hc_collapse(st, pp);
	let e = [ rel_rms(a.post, m.post), rel_rms(a.comb, m.comb), rel_rms(a.pre, m.pre), rel_rms(a.col, col) ];

	printf('R=%d: post %.1e comb %.1e pre %.1e col %.1e\n', R, ...e);

	for (let x in e)
		assert(x < 1e-4, `R=${R}: fused hc differs from the composite form`);
}

printf('dsv41_hc2: all ok\n');
