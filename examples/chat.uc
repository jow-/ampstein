// examples/chat.uc — minimal terminal chat on a local EXL3 model dir.
//
// The Phase 3 exit gate: prefill + decode + tokenizer + sampling wired
// into an interactive loop, no oracle fixtures involved.
//
// Usage:
//   ucode -L <build-dir> -L <src>/ucode examples/chat.uc -- <model-dir>
//
// Options via -D key=value (ucode standalone):
//   -D chat.system="You are a helpful assistant."
//   -D chat.maxnew=128
//   -D chat.temp=0.7
//   -D chat.topk=40
//   -D chat.seed=42
//
// Llama-3 chat template (the model this was built against); a plain
// completion mode kicks in for dirs without <|eot_id|> in the vocab.

let torch = require('ampstein.torch');
global.torch = torch;
import * as llama from 'ampstein.llama';
import { stdin } from 'fs';

const BOS = 128000;   // <|begin_of_text|>
const EOT = 128009;   // <|eot_id|>

function die(msg) {
	warn('chat: %s\n', msg);
	exit(1);
}

/* Read the model dir from ARGV (after --) or die with usage. */
function model_dir() {
	for (let i = 0; i < length(ARGV); i++)
		if (ARGV[i] == '--')
			return ARGV[i + 1];

	die('usage: chat.uc -- <model-dir>');
}

/* Template one user turn (llama-3 prompt format). */
function user_turn(text) {
	return `<|start_header_id|>user<|end_header_id|>\n\n${text}<|eot_id|>` +
		`<|start_header_id|>assistant<|end_header_id|>\n\n`;
}

/* Default system preamble the llama-3.2 template injects when the
 * caller supplies no system message (rendered from tokenizer_config.json
 * with jinja2; hardcoding it here — a real template engine is Phase 5). */
function system_turn() {
	return `<|start_header_id|>system<|end_header_id|>\n\n` +
		`Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n` +
		`<|eot_id|>`;
}

/* Generate a continuation; returns the generated token ids. */
function respond(m, tk, ids, opts) {
	let out = m.generate(ids, opts.maxnew, opts);

	return slice(out, length(ids), length(out));
}

/* --- main --------------------------------------------------------------- */

let dir = model_dir();
let tk = torch.tokenizer(dir);
let m = llama.load(dir);

let has_eot = tk.decode([EOT]) == '<|eot_id|>';

// options come from -D chat.{...} JSON defines (all optional)
let opts = {
	maxnew: (type(chat) == 'object' && chat.maxnew != null) ?
		int(chat.maxnew) : 128,
	temperature: (type(chat) == 'object' && chat.temp != null) ?
		chat.temp * 1.0 : 0.0,
	top_k: (type(chat) == 'object' && chat.topk != null) ?
		int(chat.topk) : 0,
	seed: (type(chat) == 'object' && chat.seed != null) ?
		int(chat.seed) : null
};

if (!has_eot)
	die('this demo expects a llama-3 model (special <|eot_id|> not found)');

warn('chat: model loaded, %d layers, greedy=%s\n',
	m.cfg.num_hidden_layers, opts.temperature == 0);

let history = system_turn();

while (true) {
	warn('\n> ');

	let line = stdin.read('line');

	if (line == null)
		break;

	// strip the trailing newline; skip empties
	line = replace(line, /\n$/, '');

	if (length(line) == 0)
		continue;

	history += user_turn(line);

	// special=true: the template's <|start_header_id|> etc. must map
	// to their atomic ids — plain-text encoding confuses the model
	// into echoing the prompt
	let ids = tk.encode(history, true);

	// prefill + decode; ids carry no BOS — llama-3 templates start
	// with <|begin_of_text|>, so prepend it
	let full = [BOS];
	for (let i = 0; i < length(ids); i++)
		push(full, ids[i]);

	let gen = respond(m, tk, full, opts);

	// stop at EOT if the sampler produced one
	let text_ids = [];

	for (let i = 0; i < length(gen); i++) {
		if (gen[i] == EOT)
			break;

		push(text_ids, gen[i]);
	}

	print(tk.decode(text_ids));

	/* the template closes the assistant turn with <|eot_id|> even when
	 * generation hit the length limit without producing one — without
	 * it turn 2's prompt is malformed (the model sees an assistant
	 * message that never ended) */
	history += tk.decode(text_ids) + '<|eot_id|>';
}

warn('\nchat: bye\n');