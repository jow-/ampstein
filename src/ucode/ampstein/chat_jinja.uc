// ucode/chat_jinja.uc — transpiled chat-template renderer (factory).
//
// make_renderer(tpl, opts) reads the checkpoint's chat_template.jinja
// (tpl: a model directory or a .jinja path), transpiles it to a ucode
// template ONCE (jinja2utpl) and compiles it with loadstring (utpl.load),
// and returns a closure that renders one request:
//
//     let render_prompt = chat_jinja.make_renderer(args.model, {
//         gen: (thinking) => '...',    // the add_generation_prompt tail
//         image_part: (part, ropts) => bool,
//         vision_marker: '<|...|>',    // optional, text-only templates
//    });
//     let r = render_prompt(messages, tools, {enable_thinking, ...});
//     // r = {segments: [{t, s}], text, gen, thinking, images}
//
// The template renders with add_generation_prompt: FALSE — the
// conversation only, what the KV / prompt caches store — and the
// model-specific generation tail comes back as `gen` (the server
// tokenizes it with special tokens enabled). User content lands in
// TAINTED segments (ucode/utpl.uc), so a user string containing a
// special-token sequence can never tokenize as a special token.
//
// Vision, two template shapes:
//   - the template renders its own placeholder (Qwen:
//     '<image>', a trusted template literal): image_part only collects
//     the parts, in order of appearance;
//   - the template has no image branch (text-only checkpoint, GLM-5):
//     with vision on, image parts are swapped for a text part carrying
//     opts.vision_marker before the render, collected, and the marker
//     spans are re-tainted trusted after it (they must tokenize as the
//     special pad ids the vision pipeline scans for). A template whose
//     image branch lives in a macro (Qwen) uses the same swap with
//     always_swap; system_image_error mirrors that template's own
//     system-image rejection.

import * as utpl from './utpl.uc';
import {readfile} from 'fs';

/* Tool-call arguments arrive as a JSON string on the wire; the
 * templates iterate them as a mapping (HF semantics). */
function arguments_map(a) {
	if (type(a) != 'string')
		return a;

	if (trim(a) == '')
		return {};

	try {
		return json(a);
	}
	catch (e) {
		return {arguments: a};
	}
}

/* Split every occurrence of `marker` in the segment stream into its own
 * trusted segment (the span must tokenize with special tokens enabled);
 * the surrounding text keeps its taint. The marker always sits inside a
 * single segment (it is one untrusted() value). */
function mark_trusted(segs, marker) {
	let out = [];

	for (let i = 0; i < length(segs); i++) {
		let s = segs[i], from = 0;

		while (true) {
			let j = index(s.s, marker, from);

			if (j < 0) {
				if (from < length(s.s))
					push(out, {t: s.t, s: substr(s.s, from)});
				break;
			}

			if (j > from)
				push(out, {t: s.t, s: substr(s.s, from, j - from)});
			push(out, {t: true, s: marker});
			from = j + length(marker);
		}
	}

	return out;
}

export function make_renderer(tpl, opts) {
	opts = opts || {};

	let src = readfile(index(tpl, '.jinja') >= 0 ? tpl
	                                             : tpl + '/chat_template.jinja');
	if (src == null)
		die(`chat_jinja: cannot read the chat template of ${tpl}`);

	let t = utpl.load(src);

	return function(messages, tools, ropts) {
		ropts = ropts || {};
		let thinking = (ropts.enable_thinking == null) ? true
		                                           : !!ropts.enable_thinking;
		let images = [];

		if (type(messages) != 'array' || length(messages) == 0)
			die('No messages provided.');

		/* wire-format normalization (the templates iterate these) */
		let msgs = [];

		for (let i = 0; i < length(messages); i++) {
			let m = messages[i];

			if (type(m) != 'object')
				die('message must be an object');

			if (m.role == 'developer')
				m = {...m, role: 'system'};

			let tcs = m.tool_calls;

			if (type(tcs) == 'array')
				for (let j = 0; j < length(tcs); j++)
					tcs[j].function.arguments =
						arguments_map(tcs[j].function.arguments);

			push(msgs, m);
		}

		if (opts.vision_marker && (opts.always_swap || ropts.vision)) {
			/* swap image parts for the marker text (it renders as plain
			 * user content); the driver re-taints the marker spans
			 * trusted after the render */
			for (let i = 0; i < length(msgs); i++) {
				let c = msgs[i].content;

				if (type(c) != 'array')
					continue;

				for (let j = 0; j < length(c); j++) {
					if (opts.image_part ? opts.image_part(c[j], ropts)
				                       : (type(c[j]) == 'object' &&
				                          (c[j].type == 'image' ||
				                           c[j].type == 'image_url'))) {
						if (msgs[i].role == 'system' && opts.system_image_error)
							die(opts.system_image_error);
						push(images, c[j]);
						c[j] = {type: 'text', text: opts.vision_marker};
					}
				}
			}
		}
		else if (opts.image_part) {
			/* the template renders its own placeholder: collect the
			 * parts in order of appearance */
			for (let i = 0; i < length(msgs); i++) {
				let c = msgs[i].content;

				if (type(c) != 'array')
					continue;

				for (let j = 0; j < length(c); j++)
					if (opts.image_part(c[j], ropts))
						push(images, c[j]);
			}
		}

		/* model-specific normalization of the message list (DeepSeek-V4.1 merges tool results into
		 * user turns) */
		if (opts.prepare) {
			let pr = opts.prepare(msgs, tools, ropts);

			msgs = pr.messages;
			tools = pr.tools;
		}

		let ctx = {messages: msgs, tools: tools, add_generation_prompt: false};

		/* the template reads its knobs as context roots; absent options
		 * must stay UNDEFINED (jinja `is undefined` / |default), so pass
		 * only the ones the request actually set */
		for (let k in ropts)
			if (k != 'vision' && k != 'add_generation_prompt' &&
			    ropts[k] != null)
				ctx[k] = ropts[k];

		let r = utpl.render_tpl(t, ctx);

		if (length(images) && opts.vision_marker)
			r.segments = mark_trusted(r.segments, opts.vision_marker);

		let text = '';

		for (let i = 0; i < length(r.segments); i++)
			text += r.segments[i].s;

		return {
			segments: r.segments,
			text: text,
			gen: (ropts.add_generation_prompt == false) ? ''
			     : (opts.gen ? opts.gen(thinking) : ''),
			thinking: thinking,
			images: images
		};
	};
}
