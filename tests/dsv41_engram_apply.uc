// tests/dsv41_engram_apply.uc — the Engram gate and residual update (projection of the gathered rows, key /
// value, signed-sqrt sigmoid gate) against DeepSeek's Engram.forward
// (tests/oracle/make_dsv41_engram_apply_oracle.py).

import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const FIX = getenv('AMP_DSV41_EA_ORACLE') || '/data/models/dsv41_engram_apply_oracle.safetensors';
const WF = getenv('AMP_DSV41_ENGRAM_W') || '/data/models/engram_l1.safetensors';
const CFGDIR = getenv('AMP_DSV41_CFG') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(FIX) || !stat(WF) || !stat(CFGDIR + '/config.json')) {
	warn('skip: engram apply oracle fixture, weights or config missing\n');
	exit(125);
}

let cfg = dsv41.load_config(CFGDIR);
let i = rindex(WF, '/');
let ws = weights.open(substr(WF, 0, i), substr(WF, i + 1));
let fx = torch.st(FIX);
let E = dsv41.build_engram(ws, cfg, 1, 0);
let h = fx.tensor('h').cuda(0).contiguous();
let emb = fx.tensor('emb').cuda(0).to('float16');

dsv41.engram_apply(cfg, E, h, emb);

let want = fx.tensor('y').cuda(0);
let d = h.sub(want);
let e = math.sqrt(d.mul(d).mean().item() / want.mul(want).mean().item());

printf('engram apply: rel rms %.2e\n', e);
assert(e < 5e-3, 'engram update differs from the reference');
printf('dsv41_engram_apply: all ok\n');
