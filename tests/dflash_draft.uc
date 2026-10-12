// tests/dflash_draft.uc — DFlash 2 drafter on the Qwen3.8-27B TP model:
// prefill a chat prompt with the drafter taps on, then check that the
// drafted block matches the target's own greedy continuation (the
// acceptance a round would see). Skips when the models are absent.
//
//   AMP_MODEL_QWEN35, AMP_MODEL_DFLASH  model directories

import * as qwen35 from 'ampstein.qwen35';
import * as qwen35_tp from 'ampstein.qwen35_tp';
import * as dflash from 'ampstein.dflash';
import * as chat from 'ampstein.chat_qwen35';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const DRAFT = getenv('AMP_MODEL_DFLASH') || '/data/models/Qwen3.8-27B-DFlash2-EXL3-4.00bpw';

if (!stat(MODEL) || !stat(DRAFT)) {
	warn('skip: model dirs not found\n');
	exit(125);
}

const W = 7, N = 40;

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let m = qwen35.load(MODEL, { devices: devs, mtp: false });
let tp = qwen35_tp.shard(m, { devices: [ 0, 1 ] });
let df = dflash.load(DRAFT, tp, { window: W });

tp.set_taps(df.layer_ids);

let tok = torch.tokenizer(MODEL);
let r = chat.render([
	{ role: 'user', content: 'Write a Python function that checks whether a string is a palindrome, ' +
		'ignoring case and punctuation, and explain how it works.' }
], null, { enable_thinking: false });
let prompt = tok.encode(r.prefix + r.gen, true);

printf('prompt: %d tokens\n', length(prompt));

// eager draft round (no graph), taps through forward()
let zf = map(df.layer_ids, () => torch.zeros([1, W, 5120], 'float16', 'cuda:0'));

df.build(() => zf, false);

let cache = tp.new_cache(length(prompt) + N + 16);
let res = tp.forward(prompt, cache, 1, null, false);

df.prefill_rows(res.feat, 0);

// greedy continuation of the target, one token at a time (taps unused here)
tp.set_taps(null);

let cont = [];
let logits = res.logits;

for (let i = 0; i < N; i++) {
	let t = logits.view([-1]).argmax().item();

	push(cont, t);
	logits = tp.forward([ t ], cache, 1, null, false).logits;
}

printf('continuation: %s\n', tok.decode(cont));

// reference drafts: restart from the prompt with the first token as the seed
let hits = 0, total = 0;

function check(seed, start, expect) {
	let d = df.draft(seed, null, start);
	let a = 0;

	while (a < W && d[a] == expect[a])
		a++;

	printf('  seed %d @%d: accepted %d/%d  drafts %s\n', seed, start, a, W,
		tok.decode(d));
	hits += a;
	total += W;
}

for (let k = 0; k < length(res.feat); k++)
	printf('tap %d absmax %.4g\n', k, res.feat[k].to('float32').abs().max().item());

df.reset();
df.prefill_rows(res.feat, 0);
check(cont[0], length(prompt), slice(cont, 1, 1 + W));

printf('acceptance %d/%d\n', hits, total);
assert(hits >= 4, 'drafter accepted fewer than 4 of 7 tokens');
printf('PASS\n');
