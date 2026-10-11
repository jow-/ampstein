// tests/glm5_full.uc — GLM-5.3-Flash TENSOR-PARALLEL model vs exllamav3 (oracle
// fixture tests/fixtures/glm5_oracle.safetensors): tokenizer ids, prefill
// logits of every prompt position, and the greedy continuation through the
// cached decode path. Needs the six GPUs free (~20 GiB each).

import * as glm5 from 'ampstein.glm5_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') ||
	'/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_oracle.safetensors`;
const NGPU = +(getenv('AMP_GLM5_GPUS') || 6);

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: GLM model dir or oracle fixture missing\n');
	exit(125);
}

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let n = length(ids);
let cont = fx.tensor('cont').tolist();

// tokenizer
let tok = torch.tokenizer(MODEL);
let tids = tok.encode('[gMASK]<sop><|user|>\nWhat is the capital of France? Answer in one sentence.<|assistant|>\n</think>', true);

printf('tokenizer: %d ids (oracle %d)\n', length(tids), n);
assert(join(',', tids) == join(',', ids), 'tokenizer ids match the oracle');

let t0 = time();
let model = glm5.load(MODEL, { devices: NGPU, stages: +(getenv('AMP_GLM5_STAGES') || 1), max_seq: 1024 });

printf('model loaded in %.1fs, ranks: %s\n', time() - t0, join('', model.devs));

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() /
		b.to('float32').mul(b.to('float32')).mean().item());
}

// prefill, all rows
let cache = model.new_cache(512);
t0 = time();
let logits = model.forward_chunk(ids, cache, n);
torch.synchronize();
printf('prefill %d tokens: %.2fs\n', n, time() - t0);

let want = fx.tensor('logits').cuda(logits.get_device ? logits.get_device() : 0);
let am_got = logits.cpu().argmax(-1).tolist();
let am_want = fx.tensor('logits').argmax(-1).tolist();
let same = 0;

for (let i = 0; i < n; i++)
	if (am_got[i] == am_want[i])
		same++;

let e = rel_rms(logits.cpu(), fx.tensor('logits'));

printf('logits: rel-rms %g, argmax equal at %d/%d positions\n', e, same, n);
let failures = [];

if (!(e < 0.15)) push(failures, 'logits rel-rms');
if (am_got[n - 1] != am_want[n - 1]) push(failures, 'final-position argmax');

// greedy continuation through the cached decode path
let out = [];
let lg = logits.narrow(0, n - 1, 1);

for (let k = 0; k < length(cont); k++) {
	let t = lg.reshape([-1]).argmax().item();

	push(out, t);
	lg = model.forward_chunk([t], cache, 1);
}

printf('continuation: %s\n', join(',', out));
printf('oracle      : %s\n', join(',', cont));

let first_diff = -1;

for (let k = 0; k < length(cont); k++)
	if (out[k] != cont[k]) { first_diff = k; break; }

if (!(first_diff < 0 || first_diff >= 8))
	push(failures, `continuation diverges at token ${first_diff}`);

assert(!length(failures), 'glm5_tp: ' + join('; ', failures));

print('glm5_tp: ok\n');
