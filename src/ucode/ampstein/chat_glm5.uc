// ucode/chat_glm5.uc — GLM-5.x chat format: prompt rendering and streaming
// output parsing (the interface of chat_qwen35: render / parser /
// typed_arguments).
//
// render() is a hand port of the checkpoint's chat_template.jinja; it is
// checked byte-for-byte against the jinja template by tests/chat_glm5.uc
// (oracle: tests/oracle/make_chat_glm5_oracle.py).
//
// Output grammar:
//   <think> reasoning </think> content
//   <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
//   ended by one of the eos tokens (<|user|>, <|observation|>, ...).

import {pyjson, typed_arguments as qwen_typed_arguments} from './chat_qwen35.uc';

export const typed_arguments = qwen_typed_arguments;

const TOOLS_HEAD = '<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n';
const TOOLS_TAIL = '</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>';

const TAG_THINK_END = '</think>';

/* What the model writes to end its reasoning, in the template's own form: the
 * server appends this to its closing sentence when it forces the end of
 * thinking (the template puts nothing between </think> and the content). */
export const THINK_END = '\n</think>';

/* Default reasoning-budget texts (the server's --reasoning-nudge / -limit): a
 * soft nudge appended to the thinking, and the sentence that forces the end of
 * it (closing in this model's template form). */
export const REASONING_NUDGE = '\n\nI have been thinking for quite a while now. I should wrap up my reasoning and move on to the answer.\n\n';
export const REASONING_CLOSE = '\n\nI have reached my reasoning budget, so I will stop here and answer directly based on what I have worked out so far.' + THINK_END;
const TAG_CALL = '<tool_call>';
const TAG_CALL_END = '</tool_call>';

/* The template's visible_text: strings as-is, content-part arrays
 * concatenated (text parts; other media become a reminder), null -> ''. */
/* An image as the HF processor lays it out; the server expands the single
 * <|image|> into one row per vision-tower output. */
export const IMAGE_PLACEHOLDER = '<|begin_of_image|><|image|><|end_of_image|>';

/* The checkpoint template's add_generation_prompt tail:
 * unconditional '<|assistant|><think>' (no thinking branch in the
 * template; the hand port's closed-think variant was never oracle-
 * exercised). The transpiled renderer (chat_jinja) hands this back as
 * `gen` for the engine to tokenize with special tokens enabled. */
export function gen(thinking) {
	return '<|assistant|><think>';
}

/* A vision content part (with vision on, the factory swaps these for
 * IMAGE_PLACEHOLDER text; off, the template renders a reminder and
 * nothing is collected). */
export function image_part(part, ropts) {
	return !!ropts.vision && (type(part) == 'object') &&
	       (part.type == 'image' || part.type == 'image_url' ||
	        part.image != null || part.image_url != null);
}

/* Factory options for the transpiled renderer (chat_jinja.make_renderer
 * in the server). The template has no image branch: with vision on the
 * driver swaps image parts for IMAGE_PLACEHOLDER (and re-taints the
 * spans trusted); with vision off the template renders its reminder. */
export const tpl_opts = {
	gen: gen,
	image_part: image_part,
	vision_marker: IMAGE_PLACEHOLDER
};

function visible_text(content, sink) {
	let t = type(content);

	if (content == null)
		return '';

	if (t == 'string')
		return content;

	if (t == 'array') {
		let out = '';

		for (let item in content) {
			if (type(item) == 'string') {
				out += item;
			}
			else if (type(item) == 'object') {
				if (item.type == 'text') {
					out += item.text;
				}
				else if (sink && (item.type == 'image' || item.type == 'image_url' ||
				         item.image != null || item.image_url != null)) {
					push(sink, item);

					out += IMAGE_PLACEHOLDER;
				}
				else if (item.type == 'image' || item.type == 'image_url' ||
				         item.type == 'video' || item.type == 'video_url' ||
				         item.type == 'audio' || item.type == 'audio_url' ||
				         item.type == 'input_audio') {
					let media = replace(replace(item.type, '_url', ''), 'input_', '');

					out += '<reminder>You are unable to process this ' + media +
						" because you don't have multi-modal input ability. Try different methods.</reminder>";
				}
			}
		}

		return out;
	}

	return '' + content;
}

/* tool_to_json: the tool dict without defer_loading / strict, keys in
 * order, values as Python-style JSON. */
function tool_to_json(tool) {
	let parts = [];

	for (let k in tool) {
		if (k == 'defer_loading' || k == 'strict')
			continue;

		push(parts, sprintf('"%s": ', k) + pyjson(tool[k]));
	}

	return '{' + join(', ', parts) + '}';
}

function id_of(o) {
	if (type(o) != 'object')
		return null;

	if (o.tool_call_id)
		return o.tool_call_id;

	return o.id || null;
}

/* Tool-call arguments arrive as a JSON string on the wire. */
function call_arguments(tc) {
	let a = tc.arguments;

	if (type(a) == 'string') {
		if (trim(a) == '')
			return {};

		try {
			a = json(a);
		}
		catch (e) {
			return {arguments: tc.arguments};
		}
	}

	return (type(a) == 'object') ? a : {};
}

function tool_response(m) {
	return '<tool_response>' + visible_text(m.content) + '</tool_response>';
}

/* The model's own tool call answered by `text` in place of the tool (the
 * server's call hook): the observation block of the template and the next
 * generation prompt, forced into its context. */
export function tool_feedback(text, thinking) {
	return '<|observation|>' + tool_response({content: text}) +
		(thinking ? '<|assistant|><think>' : '<|assistant|><think></think>');
}

/* Render a conversation. opts:
 *   reasoning_effort  low | high (anything else: max, per the template)
 *   enable_thinking   default true; false ends the generation prompt with
 *                     an empty think block
 *   clear_thinking    default true (template): reasoning of earlier turns
 *                     is dropped; false keeps it
 *
 * Returns {prefix, gen, thinking, images}: prefix is everything up to the
 * end of the last message, gen the generation prompt. */
export function render(messages, tools, opts) {
	opts = opts || {};

	if (type(messages) != 'array' || length(messages) == 0)
		die('No messages provided.');

	let effort = lc('' + (opts.reasoning_effort == null ? '' : opts.reasoning_effort));

	if (effort != 'low' && effort != 'high')
		effort = 'max';

	let out = '[gMASK]<sop><|system|>Reasoning Effort: ' +
		uc(substr(effort, 0, 1)) + substr(effort, 1);

	if (type(tools) == 'array' && length(tools) > 0) {
		out += TOOLS_HEAD;

		for (let t in tools) {
			let f = (type(t) == 'object' && t.function != null) ? t.function : t;

			if (!f.defer_loading)
				out += tool_to_json(f) + '\n';
		}

		out += TOOLS_TAIL;
	}

	let last_user = -1;

	for (let i = 0; i < length(messages); i++)
		if (messages[i].role == 'user')
			last_user = i;

	let clear = (opts.clear_thinking == null) ? true : !!opts.clear_thinking;
	let images = [];

	for (let i = 0; i < length(messages); i++) {
		let m = messages[i];

		if (type(m) != 'object')
			die('message must be an object');

		if (m.role == 'user') {
			// with opts.vision the user's images become placeholders (the
			// template itself would turn them into "unable to process" notes)
			out += '<|user|>' + visible_text(m.content, opts.vision ? images : null);
		}
		else if (m.role == 'system' || m.role == 'developer') {
			out += '<|system|>' + visible_text(m.content);
		}
		else if (m.role == 'assistant') {
			let content = visible_text(m.content);
			let rc = null;

			if (type(m.reasoning_content) == 'string') {
				rc = m.reasoning_content;
			}
			else if (index(content, TAG_THINK_END) >= 0) {
				let before = split(content, TAG_THINK_END);
				let head = before[0];
				let ti = rindex(head, '<think>');

				rc = (ti >= 0) ? substr(head, ti + 7) : head;
				content = split(content, TAG_THINK_END);
				content = content[length(content) - 1];
			}

			out += '<|assistant|>';

			if ((!clear || i > last_user) && rc != null)
				out += '<think>' + rc + '</think>';
			else
				out += '<think></think>';

			let c = trim(content);

			if (c != '')
				out += c;

			if (type(m.tool_calls) == 'array') {
				for (let tc in m.tool_calls) {
					if (tc.function != null)
						tc = tc.function;

					out += '<tool_call>' + tc.name;

					let args = call_arguments(tc);

					for (let k in args) {
						let v = args[k];

						out += '<arg_key>' + k + '</arg_key><arg_value>' +
							((type(v) == 'string') ? v : pyjson(v)) + '</arg_value>';
					}

					out += '</tool_call>';
				}
			}
		}
		else if (m.role == 'tool') {
			// one <|observation|> block per run of tool messages
			if (i > 0 && messages[i - 1].role == 'tool')
				continue;

			let end = i;

			while (end + 1 < length(messages) && messages[end + 1].role == 'tool')
				end++;

			out += '<|observation|>';

			// answer in the order of the preceding assistant's calls when
			// every call and every response carries an id
			let calls = (i > 0 && messages[i - 1].role == 'assistant' &&
				type(messages[i - 1].tool_calls) == 'array' &&
				length(messages[i - 1].tool_calls)) ? messages[i - 1].tool_calls : null;
			let can_sort = (calls != null);

			if (can_sort) {
				for (let tc in calls)
					if (!id_of(tc))
						can_sort = false;

				for (let k = i; k <= end; k++)
					if (!id_of(messages[k]))
						can_sort = false;
			}

			if (can_sort) {
				for (let tc in calls)
					for (let k = i; k <= end; k++)
						if (id_of(messages[k]) == id_of(tc))
							out += tool_response(messages[k]);
			}
			else {
				for (let k = i; k <= end; k++)
					out += tool_response(messages[k]);
			}
		}
	}

	let thinking = (opts.enable_thinking == null) ? true : !!opts.enable_thinking;

	return {
		prefix: out,
		gen: thinking ? '<|assistant|><think>' : '<|assistant|><think></think>',
		thinking: thinking,
		images: images
	};
}

/* --- output parsing ------------------------------------------------------ */

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

/* "NAME<arg_key>K</arg_key><arg_value>V</arg_value>..." -> {name, params}
 * (values are the raw strings; typed_arguments() coerces them). */
export function parse_tool_call(text) {
	let ai = index(text, '<arg_key>');
	let name = trim(ai < 0 ? text : substr(text, 0, ai));

	if (name == '')
		return null;

	let params = {};
	let rest = (ai < 0) ? '' : substr(text, ai);

	while (true) {
		let ks = index(rest, '<arg_key>');

		if (ks < 0)
			break;

		let ke = index(rest, '</arg_key>');

		if (ke < 0)
			break;

		let key = trim(substr(rest, ks + 9, ke - ks - 9));

		rest = substr(rest, ke + 10);

		let vs = index(rest, '<arg_value>');
		let ve = index(rest, '</arg_value>');

		if (vs < 0)
			break;

		if (ve < 0) {
			params[key] = substr(rest, vs + 11);
			break;
		}

		params[key] = substr(rest, vs + 11, ve - vs - 11);
		rest = substr(rest, ve + 12);
	}

	return {name: name, params: params};
}

/* Streaming parser: push(text) -> events [{type: 'reasoning'|'content'|
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
				let ci = index(buf, TAG_CALL);

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
					let before = substr(buf, 0, at);

					if (before != '')
						push(events, {type: 'content', text: before});

					buf = substr(buf, at + length(TAG_CALL));
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

				let emit = length(buf) - holdback(buf, [TAG_CALL, ...stops]);

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

				let call = parse_tool_call(substr(buf, 0, j));

				if (call)
					push(events, {type: 'tool_call', name: call.name, params: call.params});
				else
					push(events, {type: 'content', text: TAG_CALL + substr(buf, 0, j) + TAG_CALL_END});

				buf = substr(buf, j + length(TAG_CALL_END));
				mode = 'after_tool';
				continue;
			}

			if (mode == 'after_tool') {
				let t = ltrim(buf);

				if (index(t, TAG_CALL) == 0) {
					buf = substr(t, length(TAG_CALL));
					mode = 'tool';
					continue;
				}

				if (t == '' || holdback(t, [TAG_CALL]) == length(t)) {
					buf = t;
					return;
				}

				buf = '';          // stray text after the calls: dropped
				return;
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
				if (mode == 'reason')
					push(events, {type: 'reasoning', text: buf});
				else if (mode == 'content')
					push(events, {type: 'content', text: buf});
				else if (mode == 'tool') {
					let call = parse_tool_call(buf);

					if (call && index(buf, '</arg_value>') >= 0)
						push(events, {type: 'tool_call', name: call.name, params: call.params});
					else
						push(events, {type: 'content', text: TAG_CALL + buf});
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
