// tests/glm5_chat.uc — GLM-5.x chat prompts: ucode/chat_glm5.render() vs the
// checkpoint's jinja template (tests/fixtures/chat_glm5.json, produced by
// tests/oracle/make_chat_glm5_oracle.py), byte for byte; plus the streaming
// output parser. Pure ucode: no GPU, no model.

import * as chat from 'ampstein.chat_glm5';
import { stat, readfile } from 'fs';

const FIX = `${sourcepath(0, true)}/fixtures/chat_glm5.json`;

if (!stat(FIX)) {
	warn('skip: chat_glm5.json fixture missing (run the oracle script)\n');
	exit(125);
}

let cases = json(readfile(FIX));
let bad = 0;

for (let i = 0; i < length(cases); i++) {
	let c = cases[i];
	let r = chat.render(c.messages, c.tools, c.kwargs);
	let got = r.prefix + r.gen;

	if (got != c.prompt) {
		bad++;
		warn(`case ${i} differs:\n  want ${sprintf('%J', c.prompt)}\n  got  ${sprintf('%J', got)}\n`);
	}
}

assert(!bad, `${bad} of ${length(cases)} prompts differ from the jinja template`);
printf('render: %d/%d prompts identical to the template\n', length(cases), length(cases));

// parser: reasoning, content, tool calls split across arbitrary chunks
function run(text, thinking, step) {
	let p = chat.parser(thinking, []);
	let ev = [];

	for (let i = 0; i < length(text); i += step)
		ev = [...ev, ...p.feed(substr(text, i, step))];

	return [...ev, ...p.finish()];
}

let text = 'let me think</think>The answer.<tool_call>get_weather<arg_key>city</arg_key>' +
	'<arg_value>Paris</arg_value><arg_key>n</arg_key><arg_value>3</arg_value></tool_call>' +
	'<tool_call>calc<arg_key>expr</arg_key><arg_value>2+2</arg_value></tool_call>';

for (let step in [1, 3, 7, 1000]) {
	let ev = run(text, true, step);
	let reasoning = '', content = '', calls = [];

	for (let e in ev) {
		if (e.type == 'reasoning') reasoning += e.text;
		else if (e.type == 'content') content += e.text;
		else if (e.type == 'tool_call') push(calls, e);
	}

	assert(reasoning == 'let me think', `reasoning (chunk ${step}): ${reasoning}`);
	assert(content == 'The answer.', `content (chunk ${step}): ${content}`);
	assert(length(calls) == 2 && calls[0].name == 'get_weather' &&
	       calls[0].params.city == 'Paris' && calls[0].params.n == '3' &&
	       calls[1].name == 'calc' && calls[1].params.expr == '2+2',
		`tool calls (chunk ${step})`);

	let typed = chat.typed_arguments(calls[0], [{ type: 'function', function: {
		name: 'get_weather', parameters: { properties: { city: { type: 'string' }, n: { type: 'integer' } } } } }]);

	assert(typed.city == 'Paris' && typed.n === 3, 'typed arguments');
}

// no-thinking generation prompt: the parser starts in content mode
let ev = run('plain answer', false, 5);
let t = '';

for (let e in ev) if (e.type == 'content') t += e.text;
assert(t == 'plain answer', 'content-mode parser');

print('chat_glm5: ok\n');
