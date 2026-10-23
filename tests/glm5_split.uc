// tests/glm5_split.uc — partial loading of the GLM tensor-parallel model (docs/RPC.md phase 3):
// the pipeline built as two halves that own different stages (opts.own), run by hand layer by
// layer with the hand-off between them done as a copy over the host, as the RPC will, must
// give the same logits as the one model that owns every stage.
//   AMP_GLM5_GPUS (6)  AMP_GLM5_STAGES (3) GPUs per stage = GPUS / STAGES; the halves are stage 0
//   and the rest. Needs the GPUs free (~20 GiB each); skips (125) without the model or the memory.

import * as glm5 from 'ampstein.glm5_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const NGPU = +(getenv('AMP_GLM5_GPUS') || 6);
const NS = +(getenv('AMP_GLM5_STAGES') || 3);
const NEED = +(getenv('AMP_GLM5_NEED_GIB') || 18);

if (!stat(MODEL)) {
	warn('skip: GLM model dir missing\n');
	exit(125);
}

if (torch.device_count() < NGPU) {
	warn(`skip: ${torch.device_count()} GPUs, ${NGPU} needed\n`);
	exit(125);
}

for (let d = 0; d < NGPU; d++)
	if (torch.mem_info(d)[2] < NEED * 1073741824) {
		warn(`skip: GPU ${d} has ${sprintf('%.1f', torch.mem_info(d)[2] / 1073741824.0)} GiB free, ${NEED} needed\n`);
		exit(125);
	}

let tok = torch.tokenizer(MODEL);
let ids = tok.encode('[gMASK]<sop><|user|>\nWhat is the capital of France? Answer in one sentence.<|assistant|>\n</think>', true);
let n = length(ids);
let Gs = NGPU / NS;

function check(what, ok, extra) {
	if (!ok)
		die(`FAIL: ${what}${extra ? ' (' + extra + ')' : ''}\n`);

	warn(`ok: ${what}\n`);
}

// the whole model
let full = glm5.load(MODEL, { devices: NGPU, stages: NS, max_seq: 1024 });
let ends = [];

for (let i = 1; i < length(full.stage_of); i++)
	if (full.stage_of[i] != full.stage_of[i - 1])
		push(ends, i);

push(ends, length(full.stage_of));

let ref = full.forward_chunk(ids, full.new_cache(512), n).cpu();

check('reference logits', ref.shape()[ref.shape().length - 1] > 1000);

let stage_of = full.stage_of;

full = null;
gc();
torch.empty_cache();

// the halves: stage 0 on the first Gs GPUs, the others on the rest
let devs = [];

for (let d = 0; d < NGPU; d++)
	push(devs, d);

let A = glm5.load(MODEL, { devices: slice(devs, 0, Gs), stages: NS, own: [ 0, 1 ], stage_end: ends, max_seq: 1024 });
let B = glm5.load(MODEL, { devices: slice(devs, Gs), stages: NS, own: [ 1, NS ], stage_end: ends, max_seq: 1024 });

check('the halves plan the same stages', join(',', A.stage_of) == join(',', stage_of) && join(',', B.stage_of) == join(',', stage_of));
check('A owns the first stage only', A.own_layers[0] == 0 && A.own_layers[1] == ends[0] && A.embed_tokens != null && A.lm_head == null && A.mtp == null);
check('B owns the rest, with the head', B.own_layers[0] == ends[0] && B.own_layers[1] == length(stage_of) && B.embed_tokens == null && B.lm_head != null);
check('...and A has no layers of B', A.layers[ends[0]] == null && B.layers[0] == null);

let ca = A.new_cache(512), cb = B.new_cache(512);

check('caches are partial too', ca.layers[0] != null && ca.layers[ends[0]] == null && cb.layers[0] == null && cb.layers[ends[0]] != null);

let h = A.embed(ids, ca.mm, 0);
let cur = 0;

for (let i = 0; i < length(stage_of); i++) {
	let M = (i < ends[0]) ? A : B, c = (i < ends[0]) ? ca : cb;

	if (stage_of[i] != cur) {
		if (stage_of[i] == 1) {
			// the hop A -> B: through the host, a replica per rank (rank r to rank r)
			let host = map(h, (x) => x.cpu());

			h = map(B.stages[1].devs, (d, r) => host[r].cuda(d));
		}
		else
			h = B.handoff(h, cur, stage_of[i]);

		cur = stage_of[i];
	}

	h = M.layer(i, h, c, 0);
}

let got = B.head(h, n).cpu();
let d = got.to('float32').sub(ref.to('float32'));

check('same logits as the whole model', d.abs().max().item() == 0, sprintf('max |diff| %g', d.abs().max().item()));
check('same greedy token', got.reshape([-1, got.shape()[got.shape().length - 1]]).argmax(-1).tolist()[n - 1] ==
	ref.reshape([-1, ref.shape()[ref.shape().length - 1]]).argmax(-1).tolist()[n - 1]);

warn('glm5_split: ok\n');
