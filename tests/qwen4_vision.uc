// tests/qwen4_vision.uc — the vision tower on the Qwen3.8-Flash-Next checkpoint
// (Qwen3.5's tower, out_hidden 2560) against exllamav3's image embeddings
// (tests/oracle/make_qwen4_vision_oracle.py). Needs ~1.5 GiB on one GPU.
// Env: AMP_MODEL_QWEN4, QWEN4_VIS_FIX, QWEN4_VIS_IMG, QWEN4_VIS_DEV.

import * as vision from 'ampstein.qwen35_vision';
import * as math from 'math';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') || '/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
const FIX = getenv('AMP_QWEN4_VIS_FIX') || `${sourcepath(0, true)}/fixtures/qwen4_vision_oracle.safetensors`;
const IMG = getenv('AMP_QWEN4_VIS_IMG') || `${sourcepath(0, true)}/fixtures/vision_test.png`;

if (!stat(MODEL) || !stat(FIX) || !stat(IMG)) {
	warn('skip: model, fixture or image missing\n');
	exit(125);
}

let dev = +(getenv('AMP_QWEN4_VIS_DEV') || 0);
let v = vision.load(MODEL, { device: dev });
let r = v.embed(readfile(IMG));
let want = torch.st(FIX).tensor('emb').cuda(dev).to('float32');
let got = r.emb.to('float32');

printf('embeddings: ours %J, reference %J\n', got.shape(), want.shape());

let d = got.sub(want);
let rel = math.sqrt(d.mul(d).mean().item() / want.mul(want).mean().item());
let dot = got.mul(want).sum().item();
let cos = dot / math.sqrt(got.mul(got).sum().item() * want.mul(want).sum().item());

printf('rel-rms %.3e, cosine %.6f\n', rel, cos);

let ok = (cos > 0.995);

print(ok ? 'qwen4_vision ok\n' : 'qwen4_vision FAILED\n');
exit(ok ? 0 : 1);
