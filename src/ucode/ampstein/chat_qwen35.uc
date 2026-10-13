// ucode/chat_qwen35.uc — Qwen3.5 / Qwen3.8 chat format: prompt rendering
// and streaming output parsing.
//
// render() is a hand port of the checkpoint's chat_template.jinja (the
// official Qwen3.5 template). There is no Jinja engine in ucode; a
// Jinja -> ucode-template transpiler would make this generic across
// models, and render() is kept behind a small interface (messages,
// tools, opts -> {prefix, gen}) so it can be swapped for that later.
//
// parser() turns the generated text stream back into OpenAI-shaped
// pieces: reasoning (<think> ... </think>), content, and tool calls in
// Qwen's XML format:
//
//   <tool_call>
//   <function=NAME>
//   <parameter=P>
//   VALUE
//   </parameter>
//   </function>
//   </tool_call>

import * as struct from 'struct';

/* --- Python-compatible JSON (Jinja's tojson) ------------------------- */

/* json.dumps(v, ensure_ascii=False) with the default separators
 * (", " and ": "). ucode's %J already escapes strings the same way;
 * only the structural spacing differs. */
export function pyjson(v) {
	let t = type(v);

	if (t == 'object') {
		let parts = [];

		for (let k in v)
			push(parts, sprintf('%J', k) + ': ' + pyjson(v[k]));

		return '{' + join(', ', parts) + '}';
	}

	if (t == 'array') {
		let parts = [];

		for (let i = 0; i < length(v); i++)
			push(parts, pyjson(v[i]));

		return '[' + join(', ', parts) + ']';
	}

	return sprintf('%J', v);
}

/* --- rendering ----------------------------------------------------------- */

const REASONING_TEXT = {
	xhigh: 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
	medium: '',
	low: 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.'
};

const TOOLS_PREAMBLE = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

const TOOLS_FORMAT = '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>';

/* An image placeholder as the template renders it; the server expands
 * the single <|image_pad|> into one pad per vision-tower row. */
export const IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>';

/* The template's add_generation_prompt tail, per thinking mode. The
 * transpiled renderer (chat_jinja) renders the conversation with
 * add_generation_prompt: false and hands this back as `gen`; the hand
 * ported render() appends the same string. */
export function gen(thinking) {
	return '<|im_start|>assistant\n' +
	       (thinking ? '<think>\n'
	                 : '<think>\n\n</think>\n\n');
}

/* A content part the template renders as an 锚|image锛 placeholder (the
 * hand ported render_content sink condition); collected in order of
 * appearance for the vision pipeline. */
export function image_part(part) {
	return (type(part) == 'object') &&
	       (part.type == 'image' || part.type == 'image_url' ||
	        part.image != null || part.image_url != null);
}

/* Factory options for the transpiled renderer (chat_jinja.make_renderer
 * in the server). The template's image branch lives in a macro and
 * macro outputs are tainted wholesale (utpl.uc), so the placeholder
 * takes the driver's vision_marker re-taint path (always_swap);
 * system_image_error mirrors the template's own rejection. */
export const tpl_opts = {
	gen: gen,
	image_part: image_part,
	vision_marker: IMAGE_PLACEHOLDER,
	always_swap: true,
	system_image_error: 'System message cannot contain images.'
};

/* The template's render_content macro: strings as-is, content-part
 * arrays concatenated, null -> ''. Image parts (OpenAI image_url /
 * Qwen image) render as IMAGE_PLACEHOLDER; `sink` (array, optional)
 * collects the parts in order of appearance. Video parts are rejected. */
function render_content(content, sink) {
	let t = type(content);

	if (content == null)
		return '';

	if (t == 'string')
		return content;

	if (t == 'array') {
		let out = '';

		for (let i = 0; i < length(content); i++) {
			let item = content[i];

			if (type(item) == 'string') {
				out += item;
				continue;
			}

			if (type(item) != 'object')
				die('unexpected content part');

			if (item.type == 'video' || item.video != null)
				die('video content is not supported');

			if (item.type == 'image' || item.type == 'image_url' ||
			    item.image != null || item.image_url != null) {
				if (sink)
					push(sink, item);

				out += IMAGE_PLACEHOLDER;
				continue;
			}

			if (item.text != null)
				out += item.text;
			else
				die('unexpected content part type');
		}

		return out;
	}

	die('unexpected content type');
}

/* Map OpenAI-style reasoning_effort onto the template's levels. */
function resolve_effort(effort) {
	if (effort == null)
		return 'xhigh';     // template default

	effort = lc('' + effort);

	if (effort == 'low' || effort == 'minimal')
		return 'low';

	if (effort == 'medium')
		return 'medium';

	return 'xhigh';         // high, xhigh, max, ...
}

/* Tool-call arguments arrive as a JSON string (OpenAI wire format); the
 * template iterates them as a mapping. */
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

/* Render a conversation. opts:
 *   enable_thinking   default true
 *   reasoning_effort  low | medium | xhigh (default xhigh, per template)
 *   preserve_thinking default true
 *   add_generation_prompt default true
 *
 * Returns {prefix, gen, thinking, images}: images are the image
 * content parts in prompt order (see IMAGE_PLACEHOLDER). prefix is everything up to and
 * including the last message's <|im_end|>\n, gen the generation prompt.
 * Tokenizing them separately and concatenating gives the same ids as
 * tokenizing the whole string (gen starts with a special token), and the
 * prefix end is a stable boundary for prompt-cache reuse. */
export function render(messages, tools, opts) {
	opts = opts || {};

	if (type(messages) != 'array' || length(messages) == 0)
		die('No messages provided.');

	// lenient: OpenAI's newer "developer" role means system
	let msgs = [];

	for (let i = 0; i < length(messages); i++) {
		let m = messages[i];

		if (type(m) != 'object')
			die('message must be an object');

		if (m.role == 'developer') {
			m = {...m, role: 'system'};
		}

		push(msgs, m);
	}

	let thinking = (opts.enable_thinking == null) ? true : !!opts.enable_thinking;
	let preserve = (opts.preserve_thinking == null) ? true : !!opts.preserve_thinking;
	let reasoning = thinking ? REASONING_TEXT[resolve_effort(opts.reasoning_effort)] : '';
	let has_tools = (type(tools) == 'array' && length(tools) > 0);

	let out = struct.buffer();

	if (has_tools) {
		out.put('*', '<|im_start|>system\n');

		if (reasoning)
			out.put('*', reasoning + '\n\n');

		out.put('*', TOOLS_PREAMBLE);

		for (let i = 0; i < length(tools); i++)
			out.put('*', '\n' + pyjson(tools[i]));

		out.put('*', '\n</tools>' + TOOLS_FORMAT);

		if (msgs[0].role == 'system') {
			let content = trim(render_content(msgs[0].content));

			if (index(content, IMAGE_PLACEHOLDER) >= 0)
				die('System message cannot contain images.');

			if (content)
				out.put('*', '\n\n' + content);
		}

		out.put('*', '<|im_end|>\n');
	}
	else {
		if (msgs[0].role == 'system') {
			let content = trim(render_content(msgs[0].content));

			if (index(content, IMAGE_PLACEHOLDER) >= 0)
				die('System message cannot contain images.');

			if (content)
				out.put('*', '<|im_start|>system\n' + (reasoning ? reasoning + '\n\n' : '') +
					content + '<|im_end|>\n');
			else if (reasoning)
				out.put('*', '<|im_start|>system\n' + reasoning + '<|im_end|>\n');
		}
		else if (reasoning) {
			out.put('*', '<|im_start|>system\n' + reasoning + '<|im_end|>\n');
		}
	}

	// last user query (tool responses packed as user turns don't count)
	let multi_step_tool = true;
	let last_query_index = length(msgs) - 1;

	for (let qi = length(msgs) - 1; qi >= 0; qi--) {
		let m = msgs[qi];

		if (multi_step_tool && m.role == 'user') {
			let content = trim(render_content(m.content));

			if (!(index(content, '<tool_response>') == 0 &&
			      substr(content, -16) == '</tool_response>')) {
				multi_step_tool = false;
				last_query_index = qi;
			}
		}
	}

	if (multi_step_tool)
		die('No user query found in messages.');

	let images = [];

	for (let i = 0; i < length(msgs); i++) {
		let m = msgs[i];
		let content = trim(render_content(m.content, images));

		if (m.role == 'system') {
			if (i != 0)
				die('System message must be at the beginning.');
		}
		else if (m.role == 'user') {
			out.put('*', '<|im_start|>user\n' + content + '<|im_end|>\n');
		}
		else if (m.role == 'assistant') {
			// accept both field spellings clients use for reasoning
			let rc = m.reasoning_content;

			if (type(rc) != 'string')
				rc = m.reasoning;

			rc = (type(rc) == 'string') ? trim(rc) : '';

			if (preserve || i > last_query_index)
				out.put('*', '<|im_start|>assistant\n<think>\n' + rc + '\n</think>\n\n' + content);
			else
				out.put('*', '<|im_start|>assistant\n' + content);

			if (type(m.tool_calls) == 'array') {
				for (let j = 0; j < length(m.tool_calls); j++) {
					let tc = m.tool_calls[j];

					if (tc.function != null)
						tc = tc.function;

					if (j == 0)
						out.put('*', (trim(content) ? '\n\n' : '') +
							'<tool_call>\n<function=' + tc.name + '>\n');
					else
						out.put('*', '\n<tool_call>\n<function=' + tc.name + '>\n');

					let args = call_arguments(tc);

					for (let k in args) {
						let v = args[k];

						out.put('*', '<parameter=' + k + '>\n');
						out.put('*', (type(v) == 'string') ? v : pyjson(v));
						out.put('*', '\n</parameter>\n');
					}

					out.put('*', '</function>\n</tool_call>');
				}
			}

			out.put('*', '<|im_end|>\n');
		}
		else if (m.role == 'tool') {
			let prev = (i > 0) ? msgs[i - 1] : null;
			let next = (i + 1 < length(msgs)) ? msgs[i + 1] : null;

			if (prev && prev.role != 'tool')
				out.put('*', '<|im_start|>user');

			out.put('*', '\n<tool_response>\n' + content + '\n</tool_response>');

			if (next && next.role != 'tool')
				out.put('*', '<|im_end|>\n');
			else if (!next)
				out.put('*', '<|im_end|>\n');
		}
		else {
			die('Unexpected message role.');
		}
	}

	let gen = '';

	if (opts.add_generation_prompt != false) {
		gen = '<|im_start|>assistant\n';
		gen += thinking ? '<think>\n' : '<think>\n\n</think>\n\n';
	}

	return {prefix: out.pull(), gen: gen, thinking: thinking, images: images};
}

/* What the model reads when a tool call of its own is answered by `text`
 * instead of being sent: the end of its turn, a tool message in the template's
 * form, and the next generation prompt. The server forces this (call hook)
 * so the model sees the objection as a tool result and answers it. */
export function tool_feedback(text, thinking) {
	return '<|im_end|>\n<|im_start|>user\n<tool_response>\n' + text +
		'\n</tool_response><|im_end|>\n<|im_start|>assistant\n' +
		(thinking ? '<think>\n' : '<think>\n\n</think>\n\n');
}

/* --- output parsing ------------------------------------------------------ */

const TAG_THINK_END = '</think>';

/* What the model writes to end its reasoning, in the template's own form: the
 * server appends this to its closing sentence when it forces the end of
 * thinking (the template separates </think> and the content by a blank line). */
export const THINK_END = '\n</think>\n\n';

/* Default reasoning-budget texts (the server's --reasoning-nudge / -limit): a
 * soft nudge appended to the thinking, and the sentence that forces the end of
 * it (Qwen3's documented budget-forcing wording, closing in the template form). */
export const REASONING_NUDGE = '\n\nI have been thinking for quite a while now. I should wrap up my reasoning and move on to the answer.\n\n';
export const REASONING_CLOSE = '\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.' + THINK_END;
const TAG_CALL = '<tool_call>';
const TAG_CALL_END = '</tool_call>';

/* Length of the longest suffix of buf that is a proper prefix of one of
 * tags — text that must be held back because it might turn into a tag. */
function holdback(buf, tags) {
	let n = length(buf);
	let best = 0;

	for (let t = 0; t < length(tags); t++) {
		let tag = tags[t];
		for (let k = min(length(tag) - 1, n); k > best; k--) {
			if (substr(buf, n - k) == substr(tag, 0, k)) {
				best = k;
				break;
			}
		}
	}

	return best;
}

/* Parse the inside of one <tool_call> ... </tool_call> block. Returns
 * {name, params} with raw string values, or null if malformed. */
export function parse_tool_call(text) {
	let fs = index(text, '<function=');

	if (fs < 0)
		return null;

	let fe = index(substr(text, fs), '>');

	if (fe < 0)
		return null;

	let name = trim(substr(text, fs + 10, fe - 10));
	let rest = substr(text, fs + fe + 1);
	let params = {};

	while (true) {
		let ps = index(rest, '<parameter=');

		if (ps < 0)
			break;

		let pe = index(substr(rest, ps), '>');

		if (pe < 0)
			break;

		let pname = trim(substr(rest, ps + 11, pe - 11));
		let body = substr(rest, ps + pe + 1);
		let ve = index(body, '</parameter>');
		let value;

		if (ve < 0) {
			// unterminated: take up to </function> or the end
			let fe2 = index(body, '</function>');

			value = (fe2 >= 0) ? substr(body, 0, fe2) : body;
			rest = '';
		}
		else {
			value = substr(body, 0, ve);
			rest = substr(body, ve + 12);
		}

		// the format puts the value on its own lines
		if (substr(value, 0, 1) == '\n')
			value = substr(value, 1);

		if (substr(value, -1) == '\n')
			value = substr(value, 0, length(value) - 1);

		params[pname] = value;
	}

	return {name: name, params: params};
}

/* Convert raw string parameters to typed values using the tool's JSON
 * schema: declared strings stay strings, everything else is parsed as
 * JSON when possible (falls back to the raw string). */
export function typed_arguments(call, tools) {
	let props = null;

	if (type(tools) == 'array') {
		for (let i = 0; i < length(tools); i++) {
			let f = tools[i].function || tools[i];

			if (f.name == call.name) {
				props = (f.parameters && f.parameters.properties) || null;
				break;
			}
		}
	}

	let out = {};

	for (let k in call.params) {
		let raw = call.params[k];
		let ty = (props && props[k]) ? props[k].type : null;

		if (type(ty) == 'array')
			ty = (index(ty, 'string') >= 0) ? 'string' : ty[0];

		if (ty == 'string') {
			out[k] = raw;
			continue;
		}

		let parsed = raw;

		try {
			parsed = json(raw);
		}
		catch (e) {
			parsed = raw;
		}

		// unknown schema: keep plain words as strings, structured
		// values parsed
		if (ty == null && type(parsed) != 'object' && type(parsed) != 'array' &&
		    type(parsed) != 'int' && type(parsed) != 'double' &&
		    type(parsed) != 'bool')
			parsed = raw;

		out[k] = parsed;
	}

	return out;
}

/* Streaming parser. thinking: whether generation starts inside a
 * <think> block (the generation prompt opened it). stops: array of stop
 * strings (applied to content only).
 *
 * feed(text) and finish() return arrays of events:
 *   {type: 'reasoning', text}
 *   {type: 'content', text}
 *   {type: 'tool_call', name, params}    (raw string params)
 *   {type: 'stop'}                       (a stop string was hit)
 */
export function parser(thinking, stops) {
	stops = stops || [];

	let mode = thinking ? 'reason' : 'content_start';
	let buf = '';

	function content_tags() {
		return [TAG_CALL, ...stops];
	}

	function step(events) {
		while (true) {
			if (mode == 'reason') {
				let i = index(buf, TAG_THINK_END);

				if (i >= 0) {
					if (i > 0)
						push(events, {type: 'reasoning', text: substr(buf, 0, i)});

					buf = substr(buf, i + length(TAG_THINK_END));
					mode = 'content_start';
					continue;
				}

				let keep = holdback(buf, [TAG_THINK_END]);
				let emit = length(buf) - keep;

				if (emit > 0) {
					push(events, {type: 'reasoning', text: substr(buf, 0, emit)});
					buf = substr(buf, emit);
				}

				return;
			}

			if (mode == 'content_start') {
				// the template separates </think> and content with "\n\n"
				let t = ltrim(buf, '\n');

				if (t == '')
					return;

				buf = t;
				mode = 'content';
				continue;
			}

			if (mode == 'content') {
				// earliest of: tool call opener, any stop string
				let at = -1, which = null;
				let ci = index(buf, TAG_CALL);

				if (ci >= 0) {
					at = ci;
					which = 'call';
				}

				for (let s = 0; s < length(stops); s++) {
					let si = index(buf, stops[s]);

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

				if (which == 'stop') {
					if (at > 0)
						push(events, {type: 'content', text: substr(buf, 0, at)});

					push(events, {type: 'stop'});
					buf = '';
					mode = 'done';
					return;
				}

				let keep = holdback(buf, content_tags());
				let emit = length(buf) - keep;

				// hold back trailing whitespace too: if a tool call
				// follows, it is dropped (the template's "\n\n" before
				// <tool_call>), otherwise it goes out with the next text
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
				// only whitespace or further tool calls are expected
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

				// stray text after the calls: drop it (the format says
				// "NO suffix")
				buf = '';
				return;
			}

			// done
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
				else if (mode == 'content' || mode == 'content_start') {
					let t = rtrim(buf);

					if (t != '')
						push(events, {type: 'content', text: t});
				}
				else if (mode == 'tool') {
					// unterminated call (e.g. length limit): best effort
					let call = parse_tool_call(buf);

					if (call && index(buf, '</function>') >= 0)
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

/* --- raw text with stop strings (the /v1/completions endpoint) --------- */

/* Buffer-backed raw output text with stop-string truncation. The text is
 * appended byte-wise in pieces; each piece costs one copy into the buffer
 * (O(n) total, not the O(n^2) of `text += piece`), and the stop search is
 * incremental: a stop cannot begin inside the already scanned range except
 * in the final (max_stop_len - 1) overlap, so each piece only searches its
 * own bytes plus that tail (buf.slice(base), small). The buffer itself is
 * never rewritten — `base` just advances past the confirmed bytes — and
 * done() pulls the (truncated) text out with zero copy.
 *
 * Returns {feed, stopped, done}:
 *   feed(text)   append a piece; returns the newly visible text ('' once a
 *                stop string has truncated the text)
 *   stopped()    a stop string was hit
 *   done()       the full text (truncated at the stop string) */
export function make_text(stops) {
	stops = stops || [];
	let buf = struct.buffer();
	let base = 0;         // start of the unconfirmed window (search start)
	let msl = 0;

	for (let i = 0; i < length(stops); i++)
		if (length(stops[i]) > msl)
			msl = length(stops[i]);

	let tail = (msl > 0) ? msl - 1 : 0;
	let stopped = (msl == 0 && length(stops) > 0);   // an empty stop: stop at 0

	return {
		feed: function (t) {
			if (stopped || t == '')
				return '';

			let before = buf.length();

			buf.put('*', t);

			if (msl > 0) {
				let region = buf.slice(base);
				let best = -1;

				for (let i = 0; i < length(stops); i++) {
					let at = index(region, stops[i]);

					if (at >= 0 && (best < 0 || base + at < best))
						best = base + at;
				}

				if (best >= 0) {
					buf.length(best);
					stopped = true;
				}
			}

			let vis = (buf.length() > before) ? buf.slice(before) : '';

			if (!stopped) {
				let n = buf.length();

				base = (n > tail) ? n - tail : 0;
			}

			return vis;
		},

		stopped: function () {
			return stopped;
		},

		done: function () {
			return buf.pull();
		}
	};
}
