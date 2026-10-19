// tests/chat_gemma4.uc — the Gemma 4 chat format (chat_gemma4.jinja, the checkpoint's template with its recursive macros
// unrolled, + chat_gemma4.prepare through chat_jinja) against the checkpoint's own chat_template.jinja rendered by jinja2
// (tests/oracle/make_chat_gemma4_oracle.py), byte-exact; plus the streaming parser (thought channel, tool calls).

import * as chat_jinja from 'ampstein.chat_jinja';
import * as chat from 'ampstein.chat_gemma4';
import { stat, readfile } from 'fs';

const FIX = `${sourcepath(0, true)}/fixtures/chat_gemma4_oracle.json`;

if (!stat(FIX)) {
	warn('skip: fixture not found: ' + FIX + '\n');
	exit(125);
}

let render = chat_jinja.make_renderer(`${sourcepath(0, true)}/../src/ucode/ampstein/chat_gemma4.jinja`, chat.tpl_opts);
let cases = json(readfile(FIX));
let bad = 0;

for (let c in cases) {
	let r = render(c.messages, c.tools, {enable_thinking: c.think});
	let got = '<bos>' + r.text + r.gen;

	if (got != c.expect) {
		bad++;

		let n = 0;

		while (n < length(got) && substr(got, n, 1) == substr(c.expect, n, 1))
			n++;

		warn(`FAIL ${c.name}: differs at byte ${n}\n  got  ${sprintf('%J', substr(got, max(n - 40, 0), 120))}\n  want ${sprintf('%J', substr(c.expect, max(n - 40, 0), 120))}\n`);
	}
	else
		printf('ok %s\n', c.name);
}

assert(bad == 0, `${bad} conversations render differently`);

// the parser: a thought channel, text, a tool call with a string and a number, fed in awkward pieces
function run_parser(thinking, text, step) {
	let p = chat.parser(thinking, []);
	let evs = [];

	for (let i = 0; i < length(text); i += step)
		push(evs, ...p.feed(substr(text, i, step)));

	push(evs, ...p.finish());

	return evs;
}

let stream = '<|channel>thought\nlet me think<channel|>Sure.<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:2}<tool_call|>';

for (let step in [ 1, 3, 1000 ]) {
	let ev = run_parser(true, stream, step);
	let reasoning = '', content = '', calls = [];

	for (let e in ev)
		if (e.type == 'reasoning')
			reasoning += e.text;
		else if (e.type == 'content')
			content += e.text;
		else if (e.type == 'tool_call')
			push(calls, e);

	assert(index(reasoning, 'let me think') >= 0, `step ${step}: reasoning ${sprintf('%J', reasoning)}`);
	assert(content == 'Sure.', `step ${step}: content ${sprintf('%J', content)}`);
	assert(length(calls) == 1 && calls[0].name == 'get_weather', `step ${step}: calls ${sprintf('%J', calls)}`);

	let args = chat.typed_arguments(calls[0], null);

	assert(args.city == 'Paris' && args.days == 2, `step ${step}: arguments ${sprintf('%J', args)}`);
}

printf('chat_gemma4: all ok\n');
