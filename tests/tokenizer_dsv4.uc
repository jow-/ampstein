// tests/tokenizer_dsv4.uc — the native tokenizer on DeepSeek-V4's tokenizer.json,
// whose pre-tokenizer is a Sequence of three Split(Isolated) stages (digits,
// CJK, letters/punctuation) + regex-less ByteLevel. Ids must equal the HF
// `tokenizers` reference (tests/oracle/make_tokenizer_dsv4_oracle.py).

import { readfile, stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/tokenizer_dsv4.json`;
const MODEL = getenv('AMP_MODEL_DSV4') ||
	'/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';

if (!stat(MODEL) || !stat(FIXTURE)) {
	warn('skip: DSV4 model dir or fixture missing\n');
	exit(125);
}

let cases = json(readfile(FIXTURE)).cases;
let t = torch.tokenizer(MODEL);
let n = 0;

for (let i = 0; i < length(cases); i++) {
	let c = cases[i];
	let want = c.ids || c.ids_special;
	let ids = t.encode(c.text, c.ids_special ? true : false);

	assert(length(ids) == length(want),
		`case ${i} (${c.text}): id count ${length(ids)} vs oracle ${length(want)}`);

	for (let j = 0; j < length(ids); j++)
		assert(ids[j] == want[j], `case ${i}: id ${j} ${ids[j]} vs oracle ${want[j]}`);

	assert(t.decode(ids) == c.text, `case ${i}: decode roundtrip mismatch`);
	n++;
}

printf('tokenizer_dsv4: %d cases, ids exact, decode roundtrips\n', n);
