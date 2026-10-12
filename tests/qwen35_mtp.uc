// tests/qwen35_mtp.uc — MTP speculative decoding on the Qwen3.8-27B
// hybrid model: the speculative path must produce EXACTLY the same
// tokens as plain greedy decoding (the draft only proposes; the target
// verifies). Also checks the oracle continuation from
// fixtures/qwen35_full.safetensors.
//
// The interesting failure mode here is hybrid-model state rollback: the
// verify window advances every GDN layer's recurrent + conv state
// through all W tokens; rejected candidates must be rolled back
// (snapshot + replay) or every subsequent token diverges.

import * as qwen35 from 'ampstein.qwen35';
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

let fx = torch.st(FIXTURE);
let ids = fx.tensor('ids').tolist();
let cont_want = fx.tensor('cont').tolist();

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let model = qwen35.load(MODEL_DIR, { devices: devs });

if (!model.mtp)
	die('model has no MTP head — test is pointless');

printf('prompt: %d tokens\n', length(ids));

// plain greedy reference (longer than the oracle continuation so the
// speculative path runs several rounds, incl. rejected ones)
const N = 24;
let plain = model.generate(ids, N, {});
let spec = model.generate_mtp(ids, N, { window: 4 });

if (length(plain) != length(spec))
	die(`length mismatch: plain ${length(plain)} vs spec ${length(spec)}`);

let bad = -1;

for (let i = 0; i < length(plain); i++)
	if (plain[i] != spec[i]) {
		bad = i;
		break;
	}

/* fp16 logits near 16 have a 0.0078 grain and the network amplifies any
 * change of summation order to a few of them: two paths may legitimately
 * split a near-tie differently (docs/LESSONS.md L37, L65, L88). A mismatch
 * therefore fails only when the first differing token is not a tie in the
 * plain model's own logits at that position. */
const TIE = 0.02;

function margin(prefix, a, b) {
	let cache = model.new_cache(length(prefix) + 2);
	let lg = model.forward_chunk(prefix, cache);
	let S = length(prefix);
	let row = lg.view([S, -1]).narrow(0, S - 1, 1).view([-1]).to('float32').tolist();

	return row[a] - row[b];
}

if (bad >= 0) {
	let m = margin(slice(plain, 0, bad), plain[bad], spec[bad]);

	if (m > TIE || m < -TIE)
		die(`token mismatch at ${bad}: plain ${plain[bad]} vs spec ${spec[bad]} ` +
			`(logit margin ${m})`);

	printf('speculative == plain greedy up to %d; token %d differs by a near-tie ' +
		'(plain %d vs spec %d, margin %.4f)\n',
		bad - length(ids), bad - length(ids), plain[bad], spec[bad], m);
}
else
	printf('speculative == plain greedy: %d/%d tokens exact\n',
		length(plain) - length(ids), N);

// oracle continuation (first 8 generated tokens), up to a near-tie
for (let i = 0; i < length(cont_want); i++) {
	let got = spec[length(ids) + i];

	if (bad >= 0 && length(ids) + i >= bad)
		break;

	if (got != cont_want[i]) {
		let pre = slice(spec, 0, length(ids) + i);
		let m = margin(pre, cont_want[i], got);

		if (m > TIE)
			die(`oracle mismatch at ${i}: got ${got}, want ${cont_want[i]} ` +
				`(logit margin ${m})`);

		printf('oracle token %d: near-tie (got %d, want %d, margin %.4f)\n',
			i, got, cont_want[i], m);
		break;
	}
}

printf('oracle continuation checked: %d tokens\n', length(cont_want));

printf('PASS\n');