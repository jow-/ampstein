// Phase 3 test: native tokenizer vs the HF `tokenizers` oracle.
//
// Fixture: tests/fixtures/tokenizer_oracle.json — text -> ids pairs for
// a battery of strings (ASCII, contractions, digit runs, UTF-8, emoji,
// CJK, whitespace runs, special tokens) produced by exllamav3's
// Tokenizer (HF `tokenizers` backend). Regenerate with:
//   .venv/bin/python tests/oracle/make_tokenizer_oracle.py
//
// Checks, for every case:
//   - encode(text) reproduces the oracle ids EXACTLY
//   - decode(ids) roundtrips back to the original text
//
// Run via CTest or manually:
//   ucode -L <build-dir> tests/tokenizer.uc

import { readfile, stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/tokenizer_oracle.json`;
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

let fixture = json(readfile(FIXTURE));

let cases = fixture.cases;

assert(length(cases) >= 20, `expected >= 20 cases, got ${length(cases)}`);

let t = torch.tokenizer(MODEL);

let n = 0;

for (let i = 0; i < length(cases); i++) {
	let c = cases[i];
	let ids = t.encode(c.text);

	assert(length(ids) == length(c.ids),
		`case ${i}: id count ${length(ids)} vs oracle ${length(c.ids)}`);

	for (let j = 0; j < length(ids); j++)
		assert(ids[j] == c.ids[j],
			`case ${i}: id ${j} ${ids[j]} vs oracle ${c.ids[j]}`);

	let back = t.decode(ids);

	assert(back == c.text, `case ${i}: decode roundtrip mismatch`);

	/* cases carrying special-token strings also check the atomic path */
	if (c.ids_special) {
		let sp = t.encode(c.text, true);

		assert(length(sp) == length(c.ids_special),
			`case ${i}: special id count ${length(sp)} vs oracle ` +
			`${length(c.ids_special)}`);

		for (let j = 0; j < length(sp); j++)
			assert(sp[j] == c.ids_special[j],
				`case ${i}: special id ${j} ${sp[j]} vs oracle ` +
				`${c.ids_special[j]}`);
	}

	n++;
}

printf('tokenizer: %d cases, ids exact, decode roundtrips\n', n);

/* --- pair-form merges ----------------------------------------------------
 *
 * Newer `tokenizers` releases write model.merges as [["a", "b"], ...]
 * pairs. The fixture carries a tokenizer.json rewritten into the pair
 * encoding plus HF reference ids for the SAME file; the engine must
 * load it and reproduce those ids. */

const PF_FIXTURE = `${sourcepath(0, true)}/fixtures/tokenizer_pairform.json`;
const PF_DIR = `${sourcepath(0, true)}/fixtures/tokenizer_pairform`;

if (stat(PF_FIXTURE)) {
	let pf = json(readfile(PF_FIXTURE));
	let tp = torch.tokenizer(PF_DIR);

	for (let i = 0; i < length(pf.cases); i++) {
		let c = pf.cases[i];
		let ids = tp.encode(c.text);

		assert(length(ids) == length(c.ids),
			`pairform case ${i}: id count ${length(ids)} vs oracle ` +
			`${length(c.ids)}`);

		for (let j = 0; j < length(ids); j++)
			assert(ids[j] == c.ids[j],
				`pairform case ${i}: id ${j} ${ids[j]} vs oracle ` +
				`${c.ids[j]}`);
	}

	printf('tokenizer: %d pair-form merge cases, ids exact\n',
		length(pf.cases));
}
else
	warn('tokenizer: pair-form fixture missing, skipping ' +
		'(regenerate with tests/oracle/make_tokenizer_oracle.py)\n');