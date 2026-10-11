// tests/qwen4_depth_decode.uc — the DECODE path of Qwen3.8-Flash-Next at depth:
// prefill the first D-256 tokens of the depth fixture's text, then run the last
// 255 positions TEACHER-FORCED through the CUDA-graph decode step (S = 1, or
// S = 3 as the MTP verify) and compare the logits with the exllamav3 rows of the
// same window (tests/oracle/make_qwen4_depth_oracle.py). Same metrics as
// qwen4_depth.uc. Env: AMP_QWEN4_DEPTH (default 131072), AMP_QWEN4_STEP (1|3),
// AMP_QWEN4_EAGER=1 (forward_chunk instead of the graph), AMP_MODEL_QWEN4,
// AMP_QWEN4_DEPTH_FIX.

import * as qwen4 from 'ampstein.qwen4';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') ||
	'/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
const FIX = getenv('AMP_QWEN4_DEPTH_FIX') || `${sourcepath(0, true)}/fixtures/qwen4_depth.safetensors`;
const D = +(getenv('AMP_QWEN4_DEPTH') || 131072);
const S = +(getenv('AMP_QWEN4_STEP') || 1);
const EAGER = (getenv('AMP_QWEN4_EAGER') == '1');
const ROWS = 256, NR = ROWS - 1, CHUNK = 1024;

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: Qwen4 model dir or depth fixture missing\n');
	exit(125);
}

assert(NR % S == 0, 'step must divide 255');

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let P = D - ROWS;

let t0 = time();
let model = qwen4.load(MODEL, { });

printf('loaded in %.1fs\n', time() - t0);

let cache = model.new_cache(D + 64);
let g = null;

if (!EAGER) {
	g = model.graphs(cache, [ S ]);

	// graph warm-up touched the recurrent/conv/PLE state: start clean
	for (let cl in cache.layers)
		if (cl.rec_state) {
			cl.rec_state.fill_(0);
			cl.conv_state.fill_(0);
		}

	model.aux_reset(cache);
	cache.length = 0;
}

for (let c0 = 0; c0 < P; c0 += CHUNK)
	model.forward_chunk(slice(ids, c0, min(c0 + CHUNK, P)), cache);

torch.synchronize();
printf('prefilled %d tokens in %.1fs\n', P, time() - t0);

let outs = [];

for (let i = 0; i < NR; i += S) {
	let tk = slice(ids, P + i, P + i + S);
	let lg = EAGER ? model.forward_chunk(tk, cache) : g.run(tk, cache.length).logits;

	push(outs, lg.view([S, -1]).to('float32'));
}

let ours = torch.cat(outs, 0);
let ref = fx.tensor(`D${D}_logits`).narrow(0, 0, NR).cuda(model.mixer.dev).to('float32');
let V = min(ours.shape()[1], ref.shape()[1]);

ours = ours.narrow(1, 0, V).contiguous();
ref = ref.narrow(1, 0, V).contiguous();

let po = ours.softmax(-1), pr = ref.softmax(-1);
let tgt = torch.from_array(slice(ids, P + 1, D), 'int64', 'cuda:' + model.mixer.dev).view([-1, 1]);
let nll_o = po.gather(1, tgt).log().neg().view([-1]);
let nll_r = fx.tensor(`D${D}_nll`).cuda(model.mixer.dev);
let kl = pr.mul(pr.add(1e-30).log().sub(po.add(1e-30).log())).sum(-1);
let agree = ours.argmax(-1).sub(ref.argmax(-1)).abs().le(0).to('float32').sum().item();

printf('%s S=%d depth %d: NLL ours %.5f ref %.5f (diff %+.5f)  KLD mean %.5f max %.5f  top1 %d/%d\n',
	EAGER ? 'eager' : 'graph', S, D, nll_o.mean().item(), nll_r.mean().item(),
	nll_o.mean().item() - nll_r.mean().item(), kl.mean().item(), kl.max().item(), agree, NR);

try { torch.synchronize(); } catch (e) {}
print('qwen4_depth_decode done\n');
