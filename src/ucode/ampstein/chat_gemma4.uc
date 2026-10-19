// ucode/chat_gemma4.uc — Gemma 4 chat format: factory options for the checkpoint's chat_template.jinja
// (chat_jinja.make_renderer) and the streaming output parser (the interface of chat_dsv41 / chat_qwen35).
//
// Output grammar:
//   <|channel>thought\n REASONING <channel|> CONTENT                (thinking on; off: the prompt already ends in
//                                                                    <|channel>thought\n<channel|>)
//   <|tool_call>call:NAME{key:<|"|>string<|"|>,n:1,flag:true,list:[..],obj:{..}}<tool_call|>   (one or more)
//   ended by <turn|> (token 106, a stop id of the server).
// A tool result goes back inside the SAME model turn: the template reads `tool_responses` of the assistant message
// that made the calls, so prepare() folds the OpenAI tool messages into it and the generation prompt then does not
// open a new turn.

import {make_text as qwen_make_text} from './chat_qwen35.uc';

export const make_text = qwen_make_text;

const TAG_CH = '<|channel>';
const TAG_CH_END = '<channel|>';
const TAG_CALL = '<|tool_call>';
const TAG_CALL_END = '<tool_call|>';
const STR = '<|"|>';

export const THINK_END = TAG_CH_END;
export const REASONING_NUDGE = '\n\nI have been thinking for quite a while now. I should wrap up my reasoning and move on to the answer.\n\n';
export const REASONING_CLOSE = '\n\nI have reached my reasoning budget, so I will stop here and answer directly based on what I have worked out so far.' + THINK_END;

let last_tool = false;           // the rendered messages end inside a model turn (tool results just came back)

/* The generation prompt: a new model turn (unless the turn is open after tool results) and, with thinking off,
 * the empty thought channel the template pre-fills. */
export function gen(thinking) {
	return (last_tool ? '' : '<|turn>model\n') + (thinking ? '' : '<|channel>thought\n<channel|>');
}

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

/* OpenAI messages -> the template's: the tool messages after an assistant message with calls become its
 * `tool_responses` (name, response); the answer that follows is merged into the same message */
export function prepare(msgs, tools, ropts) {
	let out = [];

	for (let i = 0; i < length(msgs); i++) {
		let m = msgs[i];

		if (m.role == 'assistant' && type(m.tool_calls) == 'array' && length(m.tool_calls)) {
			let names = {}, res = [];

			for (let tc in m.tool_calls)
				names[tc.id ?? ''] = tc.function.name;

			while (i + 1 < length(msgs) && msgs[i + 1].role == 'tool') {
				let tm = msgs[++i];
				let raw = text_of(tm.content);
				let val = raw;

				try {
					let j = json(raw);

					if (type(j) == 'object')
						val = j;
				}
				catch (e) {}

				push(res, {name: names[tm.tool_call_id ?? ''] ?? tm.name ?? 'unknown', response: val});
			}

			let nm = {...m, content: text_of(m.content), tool_responses: res};

			// the model's answer after its tool results continues the same turn
			if (length(res) && i + 1 < length(msgs) && msgs[i + 1].role == 'assistant' &&
			    !(type(msgs[i + 1].tool_calls) == 'array' && length(msgs[i + 1].tool_calls)) && nm.content == '') {
				nm.content = text_of(msgs[++i].content);
			}

			push(out, nm);
		}
		else if (m.role == 'tool') {
			// a tool result without the call before it: show it as a user turn
			push(out, {role: 'user', content: text_of(m.content)});
		}
		else if (type(m.content) == 'array') {
			push(out, {...m, content: (m.role == 'assistant' || m.role == 'system') ? text_of(m.content) : m.content});
		}
		else
			push(out, m);
	}

	let lm = length(out) ? out[length(out) - 1] : null;

	last_tool = lm && type(lm.tool_responses) == 'array' && length(lm.tool_responses) && (lm.content ?? '') == '';

	return {messages: out, tools: tools};
}

export const tpl_opts = {
	prepare: prepare,
	gen: gen
};

/* ------------------------------------------------------------------ */
/* tool calls                                                          */

/* a value of the call syntax at s[i]: {v: value, i: next index} */
function parse_value(s, i) {
	let n = length(s);

	while (i < n && index(' \t\r\n', substr(s, i, 1)) >= 0)
		i++;

	if (substr(s, i, length(STR)) == STR) {
		let e = index(substr(s, i + length(STR)), STR);

		if (e < 0)
			return {v: substr(s, i + length(STR)), i: n};

		return {v: substr(s, i + length(STR), e), i: i + length(STR) + e + length(STR)};
	}

	let c = substr(s, i, 1);

	if (c == '{') {
		let o = {};

		i++;

		while (i < n) {
			while (i < n && index(' \t\r\n,', substr(s, i, 1)) >= 0)
				i++;

			if (substr(s, i, 1) == '}')
				return {v: o, i: i + 1};

			let key, ke;

			if (substr(s, i, length(STR)) == STR) {
				ke = index(substr(s, i + length(STR)), STR);
				key = substr(s, i + length(STR), ke);
				i += length(STR) + ke + length(STR);
			}
			else {
				ke = index(substr(s, i), ':');

				if (ke < 0)
					break;

				key = substr(s, i, ke);
				i += ke;
			}

			if (substr(s, i, 1) != ':')
				break;

			let r = parse_value(s, i + 1);

			o[trim(key)] = r.v;
			i = r.i;
		}

		return {v: o, i: i};
	}

	if (c == '[') {
		let a = [];

		i++;

		while (i < n) {
			while (i < n && index(' \t\r\n,', substr(s, i, 1)) >= 0)
				i++;

			if (substr(s, i, 1) == ']')
				return {v: a, i: i + 1};

			let r = parse_value(s, i);

			push(a, r.v);
			i = r.i;
		}

		return {v: a, i: i};
	}

	// a bare token: number, true / false / null, else text
	let j = i;

	while (j < n && index(',}]', substr(s, j, 1)) < 0)
		j++;

	let tok = trim(substr(s, i, j - i));
	let v = tok;

	if (tok == 'true')
		v = true;
	else if (tok == 'false')
		v = false;
	else if (tok == 'null')
		v = null;
	else if (match(tok, /^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/))
		v = +tok;

	return {v: v, i: j};
}

/* "call:NAME{args}" -> {name, params} */
export function parse_call(text) {
	let t = trim(text);

	if (substr(t, 0, 5) == 'call:')
		t = substr(t, 5);

	let b = index(t, '{');

	if (b < 0)
		return {name: trim(t), params: {}};

	let r = parse_value(t, b);

	return {name: trim(substr(t, 0, b)), params: (type(r.v) == 'object') ? r.v : {}};
}

/* Arguments typed after the tool's schema where the call text is ambiguous (a number the schema calls a string) */
export function typed_arguments(ev, tools) {
	let params = ev.params;
	let schema = null;

	for (let t in (tools || []))
		if ((t.function ?? t).name == ev.name)
			schema = (t.function ?? t).parameters?.properties;

	if (type(schema) != 'object')
		return params;

	let out = {};

	for (let k in keys(params)) {
		let v = params[k], ty = schema[k]?.type;

		if (ty == 'string' && type(v) != 'string' && v != null)
			v = (type(v) == 'object' || type(v) == 'array') ? sprintf('%J', v) : '' + v;
		else if ((ty == 'integer' || ty == 'number') && type(v) == 'string' && match(v, /^-?[0-9]+(\.[0-9]+)?$/))
			v = +v;

		out[k] = v;
	}

	return out;
}

/* Length of the longest suffix of buf that is a proper prefix of one of tags */
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

/* Streaming parser: feed(text) -> events [{type: 'reasoning'|'content'|'tool_call'|'stop', ...}], finish() flushes.
 * thinking: the generation prompt left the thought channel open (the model writes `<|channel>thought\n`). */
export function parser(thinking, stops) {
	stops = stops || [];

	let mode = thinking ? 'pre' : 'content';
	let buf = '';

	function step(events) {
		while (true) {
			if (mode == 'pre') {
				// the thought channel opens with <|channel>thought\n; a model that answers at once skips it
				let lead = ltrim(buf);

				if (substr(lead, 0, length(TAG_CH)) == TAG_CH) {
					let rest = substr(lead, length(TAG_CH));
					let nl = index(rest, '\n');

					if (nl < 0 && length(rest) < 12 && substr('thought\n', 0, length(rest)) == rest)
						return;

					buf = (nl >= 0 && substr(rest, 0, nl) == 'thought') ? substr(rest, nl + 1) : rest;
					mode = 'reason';
					continue;
				}

				if (length(lead) < length(TAG_CH) && substr(TAG_CH, 0, length(lead)) == lead)
					return;

				mode = 'content';
				continue;
			}

			if (mode == 'reason') {
				let i = index(buf, TAG_CH_END);

				if (i >= 0) {
					if (i > 0)
						push(events, {type: 'reasoning', text: substr(buf, 0, i)});

					buf = substr(buf, i + length(TAG_CH_END));
					mode = 'content';
					continue;
				}

				let emit = length(buf) - holdback(buf, [TAG_CH_END]);

				if (emit > 0) {
					push(events, {type: 'reasoning', text: substr(buf, 0, emit)});
					buf = substr(buf, emit);
				}

				return;
			}

			if (mode == 'content') {
				let at = -1, which = null;
				let ci = index(buf, TAG_CALL);

				if (ci >= 0) {
					at = ci;
					which = 'call';
				}

				let hi = index(buf, TAG_CH);

				if (hi >= 0 && (at < 0 || hi < at)) {
					at = hi;
					which = 'think';
				}

				for (let s in stops) {
					let si = index(buf, s);

					if (si >= 0 && (at < 0 || si < at)) {
						at = si;
						which = 'stop';
					}
				}

				if (which == 'call') {
					let before = rtrim(substr(buf, 0, at));

					if (before != '')
						push(events, {type: 'content', text: before});

					buf = substr(buf, at + length(TAG_CALL));
					mode = 'tool';
					continue;
				}

				if (which == 'think') {
					if (at > 0)
						push(events, {type: 'content', text: substr(buf, 0, at)});

					buf = substr(buf, at);
					mode = 'pre';
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

				let emit = length(buf) - holdback(buf, [TAG_CALL, TAG_CH, ...stops]);

				while (emit > 0 && index(' \t\r\n', substr(buf, emit - 1, 1)) >= 0)
					emit--;

				if (emit > 0) {
					push(events, {type: 'content', text: substr(buf, 0, emit)});
					buf = substr(buf, emit);
				}

				return;
			}

			if (mode == 'tool') {
				let j = index(buf, TAG_CALL_END);

				if (j < 0)
					return;

				let c = parse_call(substr(buf, 0, j));

				push(events, {type: 'tool_call', name: c.name, params: c.params});

				buf = ltrim(substr(buf, j + length(TAG_CALL_END)));
				mode = 'content';
				continue;
			}

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
				if (mode == 'reason' || mode == 'pre')
					push(events, {type: 'reasoning', text: buf});
				else if (mode == 'content')
					push(events, {type: 'content', text: buf});
				else if (mode == 'tool') {
					let c = parse_call(buf);

					push(events, {type: 'tool_call', name: c.name, params: c.params});
				}

				buf = '';
			}

			return events;
		},

		mode: function () {
			return (mode == 'pre') ? 'reason' : mode;
		}
	};
}
