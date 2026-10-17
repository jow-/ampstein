// tests/chat_dsv41.uc — the DeepSeek-V4.1 chat format (chat_dsv41.jinja + chat_dsv41.prepare through
// chat_jinja) against DeepSeek's encoding/encoding.py (tests/oracle/make_dsv41_chat_oracle.py), and the
// tool-call parser on the spaced DSML grammar.

import * as chat_jinja from 'ampstein.chat_jinja';
import * as chat from 'ampstein.chat_dsv41';
import { stat, readfile } from 'fs';

const FIX = `${sourcepath(0, true)}/fixtures/dsv41_chat_oracle.json`;

if (!stat(FIX)) {
	warn('skip: chat oracle fixture missing\n');
	exit(125);
}

let render = chat_jinja.make_renderer(`${sourcepath(0, true)}/../src/ucode/ampstein/chat_dsv41.jinja`, chat.tpl_opts);
let cases = json(readfile(FIX));
let bad = 0;

for (let c in cases) {
	let ropts = {enable_thinking: c.mode == 'thinking'};

	if (c.effort != null)
		ropts.reasoning_effort = c.effort;

	let r = render(c.messages, c.tools, ropts);
	let last = c.messages[length(c.messages) - 1].role;
	let got = r.text + ((last == 'assistant') ? '' : r.gen);

	if (got != c.expect) {
		bad++;
		warn(`FAIL ${c.name}\n--- got\n${got}\n--- want\n${c.expect}\n`);
	}
	else
		printf('ok %s\n', c.name);
}

assert(bad == 0, `${bad} conversations render differently`);

// the parser: reasoning, content, a call block with a string and a JSON parameter
let p = chat.parser(true, []);
let ev = p.feed('think hard</think>Sure.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n' +
	'<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n' +
	'<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>');

push(ev, ...p.finish());

let types = join(',', map(ev, (e) => e.type));

assert(types == 'reasoning,content,tool_call', `events: ${types}`);
assert(ev[2].name == 'get_weather' && ev[2].params.city == 'Paris' && ev[2].params.days == 2, 'tool call arguments');
printf('chat_dsv41: all ok\n');
