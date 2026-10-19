// Gemma 4 tokenizer (SentencePiece-style BPE: spaces -> U+2581, merges over the whole run, <0xXX> byte fallback, special
// tokens matched) vs the HF `tokenizers` oracle: encode reproduces the ids exactly, decode round-trips.
//
// Fixture: tests/fixtures/gemma4_tokenizer_oracle.json, regenerate with
//   python3 tests/oracle/make_gemma4_tokenizer_oracle.py [model_dir]
import { readfile, stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/gemma4_tokenizer_oracle.json`;
const MODEL = getenv('AMP_MODEL_GEMMA4') || '/data/models/gemma-4-26B-A4B-it-exl3-6.10bpw';

if (!stat(MODEL) || !stat(FIXTURE)) {
	warn('skip: needs the Gemma 4 model dir and ' + FIXTURE + '\n');
	exit(125);
}

let tok = torch.tokenizer(MODEL);
let cases = json(readfile(FIXTURE));
let bad = 0;

for (let c in cases) {
	let ids = tok.encode(c.text, true);
	let same = (length(ids) == length(c.ids));

	for (let i = 0; same && i < length(ids); i++)
		if (ids[i] != c.ids[i])
			same = false;

	if (!same) {
		bad++;
		warn(`FAIL encode ${sprintf('%J', substr(c.text, 0, 40))}: got ${sprintf('%J', slice(ids, 0, 16))} want ${sprintf('%J', slice(c.ids, 0, 16))}\n`);
	}

	if (tok.decode(c.ids) != c.text) {
		bad++;
		warn(`FAIL decode ${sprintf('%J', substr(c.text, 0, 40))}\n`);
	}
}

if (bad)
	die(`${bad} check(s) failed`);

printf('gemma4_tokenizer: %d cases ok\n', length(cases));
