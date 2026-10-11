// tests/qwen35_full.uc — end-to-end test of the Qwen3.8 hybrid
// architecture with a multi-GPU layer split: layers 0-31 on cuda:0,
// 32-63 on cuda:1. Compares full prefill logits and a greedy
// continuation against tests/fixtures/qwen35_full.safetensors
// (tests/oracle/make_qwen35_full_oracle.py, exllamav3).

import * as qwen35 from 'ampstein.qwen35';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL_DIR)) {
	warn('skip: model dir not found: ' + MODEL_DIR + '\n');
	exit(125);
}

const FIXTURE = `${sourcepath(0, true)}/fixtures/qwen35_full.safetensors`;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));
	let num = d.mul(d).mean().item();
	let den = b.to('float32').mul(b.to('float32')).mean().item();

	return math.sqrt(num / den);
}

let fx = torch.st(FIXTURE);
let ids = fx.tensor('ids').tolist();
let cont_want = fx.tensor('cont').tolist();
let S = length(ids);

printf('prompt: %d tokens\n', S);

// layer split: first half on device 0, second half on device 1
let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let t0 = clock();
let model = qwen35.load(MODEL_DIR, { devices: devs });
let t_load = (clock()[0] - t0[0]) + (clock()[1] - t0[1]) / 1e9;

printf('loaded in %.2f s\n', t_load);

let cache = model.new_cache(S + length(cont_want) + 2);

t0 = clock();
let logits = model.forward_chunk(ids, cache);
torch.synchronize();
let t_prefill = (clock()[0] - t0[0]) + (clock()[1] - t0[1]) / 1e9;

printf('prefill: %.2f s\n', t_prefill);

let want = fx.tensor('logits').cuda(1).view([S, -1]);
let got = logits.view([S, -1]);

printf('logits: rel-rms %.6f\n', rel_rms(got, want));

let am_got = got.topk(1, 1).indices.view([S]).tolist();
let am_want = want.topk(1, 1).indices.view([S]).tolist();
let am_ok = 0;

for (let i = 0; i < S; i++)
	if (am_got[i] == am_want[i])
		am_ok++;

printf('argmax: %d/%d positions exact\n', am_ok, S);

if (am_ok < S - 1)
	die(`only ${am_ok}/${S} argmax positions match`);

// greedy continuation from the last position
let last = logits.narrow(1, S - 1, 1);
let cont = [];

for (let n = 0; n < length(cont_want); n++) {
	let tok = last.view([-1]).topk(1, 0).indices.item();

	push(cont, tok);

	if (n + 1 >= length(cont_want))
		break;

	last = model.forward_chunk([tok], cache);
}

printf('continuation got:  %J\n', cont);
printf('continuation want: %J\n', cont_want);

let cont_ok = true;

for (let i = 0; i < length(cont_want); i++)
	if (cont[i] != cont_want[i])
		cont_ok = false;

if (!cont_ok)
	die('greedy continuation mismatch');

printf('PASS\n');