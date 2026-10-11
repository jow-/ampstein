// tests/qwen35_tp.uc — tensor-parallel qwen35 (ucode/qwen35_tp.uc) vs
// the layer-split model on the real Qwen3.8 checkpoint over two GPUs.
//
// Reference first (layer split): prompt prefill logits + a few greedy
// decode steps. Then the same model is sharded in place, and the TP
// forward is compared step by step (teacher-forced with the reference
// tokens): logits agree to fp16 noise (row-split partial sums round
// differently) and the argmax matches. Reports the eager TP decode rate.

import * as qwen35 from 'ampstein.qwen35';
import * as qwen35_tp from 'ampstein.qwen35_tp';
import * as sampler from 'ampstein.sampler';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

if (!stat(MODEL_DIR)) {
	warn('skip: model dir not found: ' + MODEL_DIR + '\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

function rel(a, b) {
	let af = a.to('float32'), bf = b.to('float32');

	return math.sqrt(af.sub(bf).pow(2).mean().item() / bf.pow(2).mean().item());
}

// ranks: $AMP_TP_DEVICES (default "0,1"); the reference layer split
// spreads the layers evenly over the same GPUs
let ranks = map(split(getenv('AMP_TP_DEVICES') || '0,1', ','), (x) => int(x));

if (torch.device_count() < length(ranks)) {
	warn('skip: needs ' + length(ranks) + ' GPUs\n');
	exit(125);
}

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, ranks[int(i * length(ranks) / 64)]);

let m = qwen35.load(MODEL_DIR, { devices: devs, mtp: false });
let tok = torch.tokenizer(MODEL_DIR);
let ids = tok.encode('The quick brown fox jumps over the lazy dog. In computer science, a hash table is', true);
const STEPS = 6;

// reference (layer split)
let ce = m.new_cache(256);
let ref = [ m.forward_chunk(ids, ce).narrow(1, length(ids) - 1, 1).to('float32').cpu() ];
let toks = [ sampler.argmax_last(ref[0]) ];

for (let i = 0; i < STEPS; i++) {
	push(ref, m.forward_chunk([toks[i]], ce).to('float32').cpu());
	push(toks, sampler.argmax_last(ref[i + 1]));
}

ce = null;
gc();

// shard and compare
let t0 = now_ms();
let tp = qwen35_tp.shard(m, { devices: ranks,
	f0: getenv('AMP_TP_F0') ? +getenv('AMP_TP_F0') : null });

printf('sharded over %d ranks in %.1f s (mlp %s, gdn k heads %s, kv heads %s; all-reduce %s)\n',
	length(ranks), (now_ms() - t0) / 1000,
	join('/', map(tp.plan.mlp, (r) => r[1])), join('/', map(tp.plan.gdn_k, (r) => r[1])),
	join('/', map(tp.plan.attn_kv, (r) => r[1])), tp.comm.info().mode);

let ct = tp.new_cache(256);
let TG = tp.graphs(ct, [ 1, 3 ]);

tp.reset(ct);

// eager TP on a second cache: prefill + the same teacher-forced steps
let ce2 = tp.new_cache(256);
let eag = [ tp.forward(ids, ce2, 1).logits ];

for (let i = 0; i < STEPS; i++)
	push(eag, tp.forward([toks[i]], ce2, 1).logits);

// graph TP: eager prefill, then the decode steps through the graph
let out = [ tp.forward(ids, ct, 1).logits ];

for (let i = 0; i < STEPS; i++)
	push(out, TG.run([toks[i]], ct.length).logits.add(0.0));

let worst = 0, agree = 0, gworst = 0;

for (let i = 0; i <= STEPS; i++) {
	worst = max(worst, rel(out[i].to('float32').cpu(), ref[i]));

	if (i > 0) {
		gworst = max(gworst, rel(out[i], eag[i]));
	}

	if (sampler.argmax_last(out[i]) == toks[i])
		agree++;
}

// one S=3 window both ways
let w = [ toks[STEPS], 13, 220 ];
let r3 = rel(TG.run(w, ct.length).logits, tp.forward(w, ce2).logits);

printf('graph TP vs eager TP: worst rel-rms %.2e (S=1), %.2e (S=3)\n', gworst, r3);
// graph steps attend through torch.attn_decode (mma for small S),
// eager forward through the prefill-style flash path (any S): different
// fp16 summation orders, fp16 noise, not bit-identity
assert(gworst < 0.04 && r3 < 0.04, 'graph TP deviates from eager TP');

printf('TP vs layer split: %d positions, worst logits rel-rms %.2e, argmax agree %d/%d\n',
	STEPS + 1, worst, agree, STEPS + 1);

assert(worst < 0.03, `TP logits deviate (${worst})`);
assert(agree >= STEPS, 'TP argmax disagrees');

// sharded draft head: argmax over the vocabulary shards == full-head
// argmax (exactly), on a few post-norm hidden rows
for (let i = 0; i < STEPS; i++) {
	let D = tp.forward([toks[i]], ce2, 1).hidden;
	let full = sampler.argmax_last(tp.lm_head.forward(D.cuda(tp.lm_head.dev)));

	assert(tp.draft_argmax(D) == full, `draft_argmax ${tp.draft_argmax(D)} != full head ${full}`);
}

// decode rates: eager TP, graph TP
let t = toks[STEPS];

torch.synchronize();
t0 = now_ms();

for (let i = 0; i < 24; i++)
	t = sampler.argmax_last(tp.forward([t], ce2, 1).logits);

let ms = (now_ms() - t0) / 24;

printf('eager TP decode: %.1f ms/token (%.1f tok/s)\n', ms, 1000 / ms);

torch.synchronize();
t0 = now_ms();

for (let i = 0; i < 48; i++)
	t = sampler.argmax_last(TG.run([t], ct.length).logits);

ms = (now_ms() - t0) / 48;

printf('graph TP decode: %.1f ms/token (%.1f tok/s)\n', ms, 1000 / ms);
print('qwen35_tp: PASS\n');
