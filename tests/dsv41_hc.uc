// tests/dsv41_hc.uc — the hyper-connection mixes (pre / post / comb) and the collapse against DeepSeek's
// Block.hc_mixes (tests/oracle/make_dsv41_hc_oracle.py).
//   AMP_DSV41_HC_ORACLE=<fixture>  AMP_DSV41_MISC_W=<layers.2.hc_* safetensors>  AMP_DSV41_CFG=<dir with config.json>

import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const FIX = getenv('AMP_DSV41_HC_ORACLE') || '/data/models/dsv41_hc_oracle.safetensors';
const WF = getenv('AMP_DSV41_MISC_W') || '/data/models/misc_l2.safetensors';
const CFGDIR = getenv('AMP_DSV41_CFG') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(FIX) || !stat(WF) || !stat(CFGDIR + '/config.json')) {
	warn('skip: hc oracle fixture, weights or config missing\n');
	exit(125);
}

let cfg = dsv41.load_config(CFGDIR);
let i = rindex(WF, '/');
let ws = weights.open(substr(WF, 0, i), substr(WF, i + 1));
let fx = torch.st(FIX);
let x = fx.tensor('x').cuda(0);

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

for (let tag in [ 'attn', 'ffn' ]) {
	let hc = dsv41.build_hc(ws, 'layers.2', tag, 0);
	let r = dsv41.hc_mixes(cfg, hc, x);
	let ep = rel_rms(r.pre, fx.tensor('pre_' + tag).cuda(0));
	let eo = rel_rms(r.post, fx.tensor('post_' + tag).cuda(0));
	let ec = rel_rms(r.comb, fx.tensor('comb_' + tag).cuda(0));
	let col = dsv41.hc_collapse(x, r.pre);
	let el = rel_rms(col, fx.tensor('col_' + tag).cuda(0));

	printf('%s: pre %.2e post %.2e comb %.2e collapse %.2e\n', tag, ep, eo, ec, el);
	assert(ep < 1e-4 && eo < 1e-4 && ec < 1e-4 && el < 2e-3, `hc ${tag}`);
}

printf('dsv41_hc: all ok\n');
