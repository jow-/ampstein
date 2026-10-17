// ucode/chat_dsv4.uc — DeepSeek-V4.1 chat format: factory options for the
// transpiled checkpoint template (chat_jinja.make_renderer) and the streaming
// output parser (the interface of chat_glm5 / chat_qwen35).
//
// Output grammar:
//   <think> reasoning </think> content
//   <｜DSML｜ calls>
//   <｜DSML｜ invoke name="NAME">
//   <｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>
//   </｜DSML｜ invoke>
//   </｜DSML｜ calls>
//   ended by <｜end▁of▁sentence｜> (token id 1, a stop id of the server).
// string="true" parameters are the raw text, string="false" ones JSON.

import {make_text as qwen_make_text} from './chat_qwen35.uc';

export const make_text = qwen_make_text;

const TAG_THINK_END = '</think>';
const TAG_CALLS = '<｜DSML｜ calls>';
const TAG_CALLS_END = '</｜DSML｜ calls>';
const TAG_INVOKE = '<｜DSML｜ invoke name="';
const TAG_INVOKE_END = '</｜DSML｜ invoke>';
const TAG_PARAM = '<｜DSML｜ parameter name="';
const TAG_PARAM_END = '</｜DSML｜ parameter>';

/* What the model writes to end its reasoning (the template puts exactly
 * `reasoning_content </think>` into earlier turns). */
export const THINK_END = '</think>';

export const REASONING_NUDGE = '\n\nI have been thinking for quite a while now. I should wrap up my reasoning and move on to the answer.\n\n';
export const REASONING_CLOSE = '\n\nI have reached my reasoning budget, so I will stop here and answer directly based on what I have worked out so far.' + THINK_END;

/* The generation prompt the template appends (assistant_leader): the
 * driver adds it after the rendered messages; thinking opens a <think>
 * block, chat mode closes it empty. */
export function gen(thinking) {
	return '<｜Assistant｜>' + (thinking ? '<think>' : '</think>');
}

/* Factory options for chat_jinja.make_renderer. The template has no image
 * branch: image parts are swapped for the placeholder token (trusted text), which
 * the server expands into the image block. */
export const IMAGE_PLACEHOLDER = '<｜deepseek_image｜>';

/* Text of a message content: a string as is, an array of parts as its text parts joined by blank lines */
function text_of(c) {
	if (type(c) != 'array')
		return (c == null) ? '' : c;

	let parts = [];

	for (let p in c)
		if (type(p) == 'object' && p.type == 'text')
			push(parts, p.text);
		else if (type(p) == 'string')
			push(parts, p);

	return join('\n\n', parts);
}

/* The normalization encoding.py does before rendering: tool results become blocks of the user turn
 * that follows (consecutive user turns merge), in the order of the tool calls; top-level tools go
 * onto the leading system message (a new empty one when there is none). Returns {messages, tools}. */
export function prepare(msgs, tools, ropts) {
	let out = [];

	for (let m in msgs) {
		let prev = length(out) ? out[length(out) - 1] : null;

		if (m.role == 'tool') {
			let blk = {type: 'tool_result', tool_use_id: m.tool_call_id ?? '', content: text_of(m.content)};

			if (prev && prev.role == 'user')
				push(prev.content_blocks, blk);
			else
				push(out, {role: 'user', content_blocks: [blk]});
		}
		else if (m.role == 'user') {
			let blocks = [];

			if (type(m.content) == 'array') {
				for (let p in m.content) {
					if (type(p) == 'object' && p.type == 'text')
						push(blocks, {type: 'text', text: p.text});
					else if (type(p) == 'object' && (p.type == 'image' || p.type == 'image_url'))
						push(blocks, {type: 'image'});
					else if (type(p) == 'string')
						push(blocks, {type: 'text', text: p});
				}
			}
			else {
				push(blocks, {type: 'text', text: m.content ?? ''});
			}

			if (prev && prev.role == 'user')
				for (let b in blocks)
					push(prev.content_blocks, b);
			else
				push(out, {...m, content_blocks: blocks});
		}
		else {
			push(out, {...m, content: (m.role == 'system' || m.role == 'assistant') ? text_of(m.content) : m.content});
		}
	}

	// results follow the order of the calls of the assistant turn before them
	let ids = [];

	for (let m in out) {
		if (m.role == 'assistant' && type(m.tool_calls) == 'array') {
			ids = [];

			for (let tc in m.tool_calls)
				push(ids, tc.id ?? tc.function?.id ?? '');
		}
		else if (m.role == 'user') {
			let res = [], slots = [];

			for (let i = 0; i < length(m.content_blocks); i++)
				if (m.content_blocks[i].type == 'tool_result') {
					push(slots, i);
					push(res, m.content_blocks[i]);
				}

			if (length(res) > 1) {
				let keyed = map(res, (b) => {
					let k = index(ids, b.tool_use_id);

					return {k: (k < 0) ? 0 : k, b: b};
				});
				let order = [];

				for (let i = 0; i < length(keyed); i++)
					push(order, i);

				order = sort(order, (a, c) => (keyed[a].k - keyed[c].k) || (a - c));

				for (let i = 0; i < length(slots); i++)
					m.content_blocks[slots[i]] = keyed[order[i]].b;
			}
		}
	}

	if (type(tools) == 'array' && length(tools)) {
		if (length(out) && out[0].role == 'system' && !length(out[0].tools ?? []))
			out[0] = {...out[0], tools: tools};
		else
			out = [{role: 'system', content: '', tools: tools}, ...out];
	}

	return {messages: out, tools: null};
}

export const tpl_opts = {
	prepare: prepare,
	gen: gen,
	vision_marker: IMAGE_PLACEHOLDER,
	always_swap: true,
	system_image_error: 'images are not supported in system messages'
};

/* Parameters arrive typed (string="false" values are parsed from JSON), so
 * the arguments of a call are the event's params as they are. */
export function typed_arguments(ev, tools) {
	return ev.params;
}

/* Length of the longest suffix of buf that is a proper prefix of one of
 * tags (text held back because it may turn into a tag). */
function holdback(buf, tags) {
	let n = length(buf);
	let best = 0;

	for (let tag in tags) {
		for (let k = (length(tag) - 1 < n) ? length(tag) - 1 : n; k > best; k--) {
			if (substr(buf, n - k) == substr(tag, 0, k)) {
				best = k;
				break;
			}
		}
	}

	return best;
}

/* The inside of one tool_calls block -> [{name, params}, ...]. Tolerant:
 * an unterminated last parameter / invoke is taken as far as it goes. */
export function parse_calls(text) {
	let calls = [];
	let rest = text;

	while (true) {
		let is = index(rest, TAG_INVOKE);

		if (is < 0)
			break;

		rest = substr(rest, is + length(TAG_INVOKE));

		let q = index(rest, '">');

		if (q < 0)
			break;

		let name = substr(rest, 0, q);

		rest = substr(rest, q + 2);

		let ie = index(rest, TAG_INVOKE_END);
		let body = (ie < 0) ? rest : substr(rest, 0, ie);
		let params = {};

		while (true) {
			let ps = index(body, TAG_PARAM);

			if (ps < 0)
				break;

			body = substr(body, ps + length(TAG_PARAM));

			let kq = index(body, '"');

			if (kq < 0)
				break;

			let key = substr(body, 0, kq);
			let tail = substr(body, kq + 1);
			let str = true;
			let sm = index(tail, ' string="');

			if (sm >= 0 && sm < index(tail, '>')) {
				str = (substr(tail, sm + 9, 4) == 'true');
			}

			let gt = index(tail, '>');

			if (gt < 0)
				break;

			tail = substr(tail, gt + 1);

			let pe = index(tail, TAG_PARAM_END);
			let raw = (pe < 0) ? tail : substr(tail, 0, pe);

			if (!str) {
				try {
					params[key] = json(raw);
				}
				catch (e) {
					params[key] = raw;
				}
			}
			else
				params[key] = raw;

			body = (pe < 0) ? '' : substr(tail, pe + length(TAG_PARAM_END));
		}

		push(calls, {name: name, params: params});

		if (ie < 0)
			break;

		rest = substr(rest, ie + length(TAG_INVOKE_END));
	}

	return calls;
}

/* Streaming parser: feed(text) -> events [{type: 'reasoning'|'content'|
 * 'tool_call'|'stop', ...}], finish() flushes. thinking: the generation
 * prompt ended inside <think>. stops: extra stop strings. */
export function parser(thinking, stops) {
	stops = stops || [];

	let mode = thinking ? 'reason' : 'content';
	let buf = '';

	function step(events) {
		while (true) {
			if (mode == 'reason') {
				let i = index(buf, TAG_THINK_END);

				if (i >= 0) {
					if (i > 0)
						push(events, {type: 'reasoning', text: substr(buf, 0, i)});

					buf = substr(buf, i + length(TAG_THINK_END));
					mode = 'content';
					continue;
				}

				let emit = length(buf) - holdback(buf, [TAG_THINK_END]);

				if (emit > 0) {
					push(events, {type: 'reasoning', text: substr(buf, 0, emit)});
					buf = substr(buf, emit);
				}

				return;
			}

			if (mode == 'content') {
				let at = -1, which = null;
				let ci = index(buf, TAG_CALLS);

				if (ci >= 0) {
					at = ci;
					which = 'call';
				}

				for (let s in stops) {
					let si = index(buf, s);

					if (si >= 0 && (at < 0 || si < at)) {
						at = si;
						which = 'stop';
					}
				}

				if (which == 'call') {
					// the template separates content and calls with a blank line
					let before = rtrim(substr(buf, 0, at));

					if (before != '')
						push(events, {type: 'content', text: before});

					buf = substr(buf, at + length(TAG_CALLS));
					mode = 'tool';
					continue;
				}

				if (which == 'stop') {
					if (at > 0)
						push(events, {type: 'content', text: substr(buf, 0, at)});

					push(events, {type: 'stop'});
					buf = '';
					mode = 'done';
					return;
				}

				let emit = length(buf) - holdback(buf, [TAG_CALLS, ...stops]);

				// trailing whitespace may precede a tool call: hold it back
				while (emit > 0 && index(' \t\r\n', substr(buf, emit - 1, 1)) >= 0)
					emit--;

				if (emit > 0) {
					push(events, {type: 'content', text: substr(buf, 0, emit)});
					buf = substr(buf, emit);
				}

				return;
			}

			if (mode == 'tool') {
				let j = index(buf, TAG_CALLS_END);

				if (j < 0)
					return;

				let calls = parse_calls(substr(buf, 0, j));

				for (let c in calls)
					push(events, {type: 'tool_call', name: c.name, params: c.params});

				if (!length(calls))
					push(events, {type: 'content', text: TAG_CALLS + substr(buf, 0, j) + TAG_CALLS_END});

				buf = substr(buf, j + length(TAG_CALLS_END));
				mode = 'after_tool';
				continue;
			}

			// after the calls: only the end of sentence follows
			buf = '';
			return;
		}
	}

	return {
		feed: function (text) {
			let events = [];

			buf += text;
			step(events);

			return events;
		},

		finish: function () {
			let events = [];

			step(events);

			if (buf != '') {
				if (mode == 'reason')
					push(events, {type: 'reasoning', text: buf});
				else if (mode == 'content')
					push(events, {type: 'content', text: buf});
				else if (mode == 'tool') {
					let calls = parse_calls(buf);

					for (let c in calls)
						push(events, {type: 'tool_call', name: c.name, params: c.params});

					if (!length(calls))
						push(events, {type: 'content', text: TAG_CALLS + buf});
				}

				buf = '';
			}

			return events;
		},

		mode: function () {
			return mode;
		}
	};
}
