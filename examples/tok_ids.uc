// examples/tok_ids.uc — encode the strings of a JSON file ([{text, ids}, ...]) with the model directory's tokenizer
// (special tokens matched) and report the entries that differ from the reference ids; also decode round trips.
//   ucode ... examples/tok_ids.uc <model_dir> <ref.json>
import { readfile } from 'fs';

let tok = torch.tokenizer(ARGV[0]);
let ref = json(readfile(ARGV[1]));
let bad = 0;

for (let r in ref) {
	let ids = tok.encode(r.text, true);
	let same = (length(ids) == length(r.ids));

	for (let i = 0; same && i < length(ids); i++)
		if (ids[i] != r.ids[i])
			same = false;

	let back = tok.decode(r.ids);

	printf('%s %d ids  roundtrip %s\n', same ? 'ok  ' : 'DIFF', length(ids), (back == r.text) ? 'ok' : 'differs');

	if (!same) {
		bad++;
		printf('  got  %J\n  want %J\n', slice(ids, 0, 24), slice(r.ids, 0, 24));
	}
}

exit(bad ? 1 : 0);
