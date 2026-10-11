// ucode/ampstein_core.uc - Ampstein engine host logic, imported by the
// ampstein entry. The server/engine logic lives here as a module so the
// entry stays lean (arg parsing, --help/--version, REQUIRE_SEARCH_PATH bootstrap).
//
// Loaded via runtime import('ampstein.core'): the entry pushes
// <prefix>/share/*.uc onto REQUIRE_SEARCH_PATH first. The imports below are
// ADJACENT-RELATIVE ('./x.uc'), because a module's own static imports are
// resolved at compile time against the importing file's directory or -L,
// never against REQUIRE_SEARCH_PATH (L117) — this keeps the installed tree
// relocatable with no -L. Native modules (torch) are found via the entry's
// REQUIRE_SEARCH_PATH and exposed as the global `torch`.

import * as struct from 'struct';
import * as uloop from 'uloop';
import * as math from 'math';
import {readfile, writefile, unlink} from 'fs';

import * as httpd from './httpd.uc';
import * as engine from './engine.uc';
import * as utpl from './utpl.uc';
import * as chat_jinja from './chat_jinja.uc';
import * as chat_qwen35 from './chat_qwen35.uc';
import * as qwen35 from './qwen35.uc';

function now_s() {
	return time();
}

function rand_id(prefix) {
	let s = prefix;

	for (let i = 0; i < 24; i++)
		s += substr('0123456789abcdefghijklmnopqrstuvwxyz', int(math.rand(0, 36)), 1);

	return s;
}

function basename(p) {
	p = rtrim(p, '/');

	let i = rindex(p, '/');

	return (i >= 0) ? substr(p, i + 1) : p;
}

/* Number of leading bytes of s that form complete UTF-8 sequences. */
function utf8_complete(s) {
	let n = length(s);
	let i = n - 1, back = 0;

	while (i >= 0 && back < 4) {
		let c = ord(s, i);

		if ((c & 0xC0) != 0x80) {
			// lead byte: how long should the sequence be?
			let need = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0) ? 2 :
				((c & 0xF0) == 0xE0) ? 3 : ((c & 0xF8) == 0xF0) ? 4 : 1;

			return (n - i >= need) ? n : i;
		}

		i--;
		back++;
	}

	return n;
}

export function serve(args) {
	let served_name = args.name || basename(args.model);

	let gen_cfg = {};

	try {
		gen_cfg = json(readfile(args.model + '/generation_config.json')) || {};
	}
	catch (e) {
		gen_cfg = {};
	}

	// default device count: two (the dense Qwen layer-splits over a pair)
	if (args.gpus == null)
		args.gpus = 2;

	let chat = chat_qwen35;
	let cfg = qwen35.load_config(args.model);

	/* The transpiled chat template: chat_template.jinja -> jinja2utpl ->
	   loadstring, compiled ONCE here; this closure is invoked for every
	   chat request. User content lands in tainted segments, so a user
	   string containing a special-token sequence can never tokenize as a
	   special token (ucode/chat_jinja.uc, ucode/utpl.uc). */
	let render_prompt = chat_jinja.make_renderer(args.model, chat.tpl_opts);

	let devs = [];
	let per = args.split || int((cfg.num_hidden_layers + args.gpus - 1) / args.gpus);

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let d = int(i / per);

		push(devs, (d >= args.gpus) ? args.gpus - 1 : d);
	}

	warn(`ampstein: loading ${args.model} over ${args.gpus} GPU(s), ctx ${args.ctx}\n`);

	uloop.init();

	let t0 = time();
	let tok = torch.tokenizer(args.model);

	let model;

	model = qwen35.load(args.model, {devices: devs, mtp: args.mtp > 0});

	let eng = engine.create(model, {
		max_ctx: args.ctx,
		prefill_chunk: args.chunk,
		mtp_window: args.mtp,
		draft_vocab: args.draft_vocab,
	});

	let banner = `ampstein: model loaded in ${time() - t0}s`;

	if (eng.mtp_window)
		banner += `, MTP speculative decoding (window ${eng.mtp_window})`;

	warn(banner + '\n');

	/* --- hot reload: restore the cache state saved by a previous shutdown ---
	 *
	 * On SIGINT/SIGTERM the server writes its live generation state (token
	 * history, attention KV, GDN states, MTP draft cache) to a scratch
	 * safetensors file and restores it on the next start — so an agentic
	 * session survives server restarts without re-prefilling the context.
	 * The port-qualified path keeps concurrent servers apart. */
	const CACHE_ST = `/tmp/Ampstein-cache-${args.port}.safetensors`;
	const CACHE_META = `${CACHE_ST}.json`;

	(function restore_cache() {
		let meta;

		try {
			meta = json(readfile(CACHE_META));
		}
		catch (e) {
			return;     // no saved state
		}

		if (meta.format != 1 || meta.model != args.model) {
			warn('ampstein: ignoring saved cache state ' +
				'(model or format changed)\n');
			return;
		}

		try {
			let n = eng.restore_state(CACHE_ST);

			warn(`ampstein: restored ${n} cached tokens from ${CACHE_ST}\n`);
		}
		catch (e) {
			warn(`ampstein: cache restore failed: ${e.message}\n`);

			try {
				unlink(CACHE_ST);
				unlink(CACHE_META);
			} catch (e2) {}
		}
	})();

	// Prompt tokenization cache (one entry): agent sessions re-send the
	// whole conversation every turn, so the rendered prefix usually
	// extends the previous one (the new part starts where the old render's
	// generation tail meets the new assistant content). The extension
	// point is a segment boundary and each side encodes independently,
	// so cached ids + encoding of the NEW segments (each with its own
	// taint) equals a full encode — and the tainted user content in the
	// new segments is never atomized. 645 KB / 210k tokens: a full
	// re-encode costs 268 ms.
	let prefix_cache = utpl.prefix_cache();

	function encode_prefix(r) {
		return prefix_cache.run(
			(segs) => utpl.encode_segments(tok, {segments: segs}),
			r, null);
	}

	// stop tokens: generation_config's eos list, plus <|im_end|>
	let stop_ids = [];
	let eos = gen_cfg.eos_token_id;

	if (type(eos) == 'array')
		stop_ids = eos;
	else if (eos != null)
		stop_ids = [eos];

	let im_end = tok.encode('<|im_end|>', true);

	if (length(im_end) == 1 && index(stop_ids, im_end[0]) < 0)
		push(stop_ids, im_end[0]);

	let created_at = now_s();

	/* --- helpers ------------------------------------------------------------------ */

	function api_error(res, status, message, type_, code, param) {
		res.json(status, {
			error: {
				message: message,
				type: type_ || 'invalid_request_error',
				param: param || null,
				code: code || null
			}
		});
	}

	function model_card() {
		return {
			id: served_name,
			object: 'model',
			created: created_at,
			owned_by: 'Ampstein',
			root: args.model,
			max_model_len: eng.max_ctx
		};
	}

	/* Sampling parameters: request values, defaults from
	 * generation_config.json (as vLLM does). */
	function sampling_of(body) {
		function pick(k, def) {
			return (body[k] != null) ? body[k] : ((gen_cfg[k] != null) ? gen_cfg[k] : def);
		}

		return {
			temperature: pick('temperature', 1.0),
			top_p: pick('top_p', 1.0),
			top_k: pick('top_k', 0),
			min_p: pick('min_p', 0),
			presence_penalty: body.presence_penalty || 0,
			frequency_penalty: body.frequency_penalty || 0,
			seed: body.seed
		};
	}

	function stops_of(body) {
		let s = body.stop;

		if (s == null)
			return [];

		if (type(s) == 'string')
			return [s];

		return (type(s) == 'array') ? filter(s, (x) => type(x) == 'string' && x != '') : [];
	}

	function usage_of(info) {
		return {
			prompt_tokens: info.prompt_tokens,
			completion_tokens: info.completion_tokens,
			total_tokens: info.prompt_tokens + info.completion_tokens,
			prompt_tokens_details: {cached_tokens: info.cached_tokens || 0}
		};
	}

	/* decode token ids for a log line: image marker ids (>= 2^30) stand for the
	 * placeholder token, and a failure must never break the request */
	function safe_decode(ids) {
		try {
			return tok.decode(map(ids, (x) => (x >= 1073741824) ? (cfg.image_token_id || 0) : x));
		}
		catch (e) {
			return '?';
		}
	}

	function log_done(kind, info) {
		let t = info.timings;

		if (!t) {
			warn(`${kind}: ${info.reason}\n`);
			return;
		}

		let pf = (t.prefill_ms > 0) ? t.prefill_tokens * 1000 / t.prefill_ms : 0;
		let dc = (t.decode_ms > 0) ? t.decode_tokens * 1000 / t.decode_ms : 0;

		let mtp = info.mtp ? sprintf(', MTP %d rounds, %.0f%% of drafts accepted, %.2f tok/round',
			info.mtp.rounds, 100.0 * info.mtp.accepted / info.mtp.drafted,
			info.completion_tokens * 1.0 / info.mtp.rounds) : '';

		let why = '';
		let d = info.reuse_detail;

		// not a plain continuation of the live state: say why (client-side
		// re-rendering of the previous turn shows as a short live match)
		if (d && info.reuse != 'live') {
			why = sprintf(' [live %d/%d matched, snapshot %d/%d matched', d.live_lcp, d.ids, d.snap_lcp, d.snap);

			if (d.div_old)
				why += `, diverges: ${sprintf('%J', safe_decode(d.div_old))} -> ${sprintf('%J', safe_decode(d.div_new))}`;

			why += ']';
		}

		warn(sprintf('%s: prompt %d (cached %d via %s%s), prefill %d tok in %.0f ms (%.1f tok/s), ' +
			'decode %d tok in %.0f ms (%.1f tok/s)%s, finish %s\n',
			kind, info.prompt_tokens, info.cached_tokens, info.reuse, why,
			t.prefill_tokens, t.prefill_ms, pf, t.decode_tokens, t.decode_ms, dc,
			mtp, info.reason));
	}

	/* Intermediate prefill progress: the engine calls on_progress once per
	 * processed chunk; log a line whenever another ~1024 tokens completed
	 * (and the final one), but only for prompts long enough that
	 * intermediate state is interesting (more than one chunk). */
	function progress_logger(kind, id) {
		let logged = 0;
		let t0 = 0, done0 = 0;   // clock reference: the first callback

		return function (done, total) {
			let t = clock(true);
			let now = t[0] * 1000 + t[1] / 1e6;

			if (!t0) {
				t0 = now;
				done0 = done;
			}

			if (total >= 2048 && done > logged && now > t0 &&
			    (done - logged >= 1024 || done == total)) {
				// rate over the observed span (from the first callback:
				// the very first chunk's time is not included)
				let dt = now - t0;
				let rate = (dt > 0) ? (done - done0) * 1000 / dt : 0;

				warn(sprintf('%s %s: prefill %d/%d tok (%d%%), %.1f tok/s\n',
					kind, id, done, total, int(100 * done / total), rate));
			}

			logged = done;
		};
	}

	/* Incremental detokenizer: per-token byte pieces (byte-level BPE
	 * concatenates exactly), emitted at UTF-8 boundaries. */
	function detok() {
		let pending = '';

		return {
			push: function (id) {
				pending += tok.decode([id]);

				let n = utf8_complete(pending);
				let out = substr(pending, 0, n);

				pending = substr(pending, n);

				return out;
			},

			flush: function () {
				let out = pending;

				pending = '';

				return out;
			}
		};
	}

	/* Limit max_new to the remaining context. */
	function max_new_of(body, prompt_len) {
		let room = eng.max_ctx - prompt_len - 1;
		let want = body.max_completion_tokens || body.max_tokens;

		if (want == null || want > room)
			want = room;

		return want;
	}

	/* --- chat completions ------------------------------------------------------- */

	function chat_completions(req, res, body) {
		let kw = body.chat_template_kwargs || {};
		let effort = body.reasoning_effort;
		let enable_thinking = kw.enable_thinking;

		if (effort != null && (lc('' + effort) == 'none' || lc('' + effort) == 'off'))
			enable_thinking = false;

		let tools = body.tools;

		if (body.tool_choice == 'none')
			tools = null;

		let r;

		try {
			r = render_prompt(body.messages, tools, {
				enable_thinking: enable_thinking,
				reasoning_effort: kw.reasoning_effort || effort,
				preserve_thinking: kw.preserve_thinking,
			});
		}
		catch (e) {
			return api_error(res, 400, e.message, 'invalid_request_error', null, 'messages');
		}

		let prefix_ids = encode_prefix(r);

		let gen_ids = tok.encode(r.gen, true);
		let prefix_len = length(prefix_ids);   // BEFORE extending: ids aliases prefix_ids (L81)
		let ids = prefix_ids;     // fresh array (encode / expansion): extend in place

		push(ids, ...gen_ids);

		if (length(ids) >= eng.max_ctx)
			return api_error(res, 400,
				`This model's maximum context length is ${eng.max_ctx} tokens. ` +
				`However, your messages resulted in ${length(ids)} tokens. ` +
				`Please reduce the length of the messages.`,
				'invalid_request_error', 'context_length_exceeded', 'messages');

		let stream = !!body.stream;
		let include_usage = stream && body.stream_options && body.stream_options.include_usage;
		let id = rand_id('chatcmpl-');
		let progress = progress_logger('chat', id);
		let created = now_s();
		let p = chat.parser(r.thinking, stops_of(body));
		let dt = detok();

		// non-stream only: the final JSON needs the full text; the streamed
		// path never reads it. Buffer-backed so a long completion costs O(n)
		// total instead of O(n^2) string concatenation.
		let content = struct.buffer(), reasoning = struct.buffer();
		let calls = [];
		let stopped = false;

		function chunk(delta, finish) {
			return {
				id: id, object: 'chat.completion.chunk', created: created,
				model: served_name,
				choices: [{index: 0, delta: delta, logprobs: null,
					finish_reason: finish || null}]
			};
		}

		function emit(events) {
			for (let i = 0; i < length(events); i++) {
				let ev = events[i];

				if (ev.type == 'reasoning') {
					if (!stream)
						reasoning.put('*', ev.text);

					if (stream)
						res.sse(chunk({reasoning_content: ev.text}));
				}
				else if (ev.type == 'content') {
					if (!stream)
						content.put('*', ev.text);

					if (stream)
						res.sse(chunk({content: ev.text}));
				}
				else if (ev.type == 'tool_call') {
					let call = {
						id: rand_id('call_'),
						type: 'function',
						function: {
							name: ev.name,
							arguments: sprintf('%J', chat.typed_arguments(ev, body.tools))
						}
					};

					if (stream)
						res.sse(chunk({tool_calls: [{index: length(calls), ...call}]}));

					push(calls, call);
				}
				else if (ev.type == 'stop') {
					stopped = true;
				}
			}
		}

		if (stream) {
			res.stream(200);
			res.sse(chunk({role: 'assistant', content: ''}));
		}

		eng.submit({
			ids: ids,
			snap_at: prefix_len,
			max_new: max_new_of(body, length(ids)),
			sampling: sampling_of(body),
			stop_ids: stop_ids,

			alive: function () {
				return res.alive();
			},

			on_progress: function (done, total) {
				progress(done, total);

				// keep the stream warm and notice disconnects during long
				// prefills
				if (stream)
					return res.ping(`prefill ${done}/${total}`);

				return res.alive();
			},

			on_token: function (t) {
				emit(p.feed(dt.push(t)));

				if (stopped)
					return false;

				return stream ? !res.finished() && res.alive() : res.alive();
			},

			on_done: function (info) {
				emit(p.feed(dt.flush()));
				emit(p.finish());
				log_done('chat', info);

				if (info.reason == 'cancelled')
					return res.end();

				let finish = (length(calls) > 0) ? 'tool_calls' :
					((info.reason == 'length') ? 'length' : 'stop');

				if (stream) {
					res.sse(chunk({}, finish));

					if (include_usage)
						res.sse({id: id, object: 'chat.completion.chunk', created: created,
							model: served_name, choices: [], usage: usage_of(info)});

					res.sse('[DONE]');
					res.end();
					return;
				}

				let msg = {role: 'assistant', content: (content.length() == 0 && length(calls)) ? null : content.pull()};

				// trim() copies unconditionally (amp_trim_common always allocates),
				// so only pay for it when an edge byte is actually whitespace —
				// the common case pulls the buffer out with zero copies
				if (reasoning.length() != 0) {
					let rs = reasoning.pull();

					if (index(' \t\r\n', substr(rs, 0, 1)) >= 0 ||
					    index(' \t\r\n', substr(rs, -1)) >= 0)
						rs = trim(rs);

					msg.reasoning_content = rs;
				}

				if (length(calls))
					msg.tool_calls = calls;

				res.json(200, {
					id: id, object: 'chat.completion', created: created, model: served_name,
					choices: [{index: 0, message: msg, logprobs: null, finish_reason: finish}],
					usage: usage_of(info)
				});
			},

			on_error: function (msg) {
				warn(`chat: error: ${msg}\n`);

				if (stream) {
					res.sse({error: {message: msg, type: 'server_error', code: null}});
					res.end();
				}
				else {
					api_error(res, 500, msg, 'server_error');
				}
			}
		});
	}

	/* --- completions (raw prompt) ------------------------------------------------ */

	function completions(req, res, body) {
		let prompt = body.prompt;

		if (type(prompt) == 'array')
			prompt = (type(prompt[0]) == 'string') ? prompt[0] : null;

		if (type(prompt) != 'string')
			return api_error(res, 400, 'prompt must be a string', 'invalid_request_error', null, 'prompt');

		let ids = tok.encode(prompt, true);

		if (length(ids) >= eng.max_ctx)
			return api_error(res, 400,
				`This model's maximum context length is ${eng.max_ctx} tokens, ` +
				`your prompt has ${length(ids)} tokens.`,
				'invalid_request_error', 'context_length_exceeded', 'prompt');

		let stream = !!body.stream;
		let id = rand_id('cmpl-');
		let progress = progress_logger('completion', id);
		let created = now_s();
		let dt = detok();
		let stops = stops_of(body);
		// buffer-backed accumulation + incremental stop search (O(n) total,
		// not the O(n^2) of per-piece string concatenation + full re-scans)
		let text = chat.make_text(stops);
		let stopped = false;

		function piece(t) {
			stopped = text.stopped();

			return text.feed(t);
		}

		function chunk(t, finish) {
			return {id: id, object: 'text_completion', created: created, model: served_name,
				choices: [{index: 0, text: t, logprobs: null, finish_reason: finish || null}]};
		}

		if (stream)
			res.stream(200);

		eng.submit({
			ids: ids,
			snap_at: 0,
			max_new: max_new_of(body, length(ids)),
			sampling: sampling_of(body),
			stop_ids: stop_ids,

			alive: function () {return res.alive();},

			on_progress: function (done, total) {
				progress(done, total);

				return stream ? res.ping(`prefill ${done}/${total}`) : res.alive();
			},

			on_token: function (t) {
				let s = piece(dt.push(t));

				if (stream && s != '')
					res.sse(chunk(s));

				return !stopped && res.alive();
			},

			on_done: function (info) {
				if (!stopped) {
					let s = piece(dt.flush());

					if (stream && s != '')
						res.sse(chunk(s));
				}

				log_done('completion', info);

				if (info.reason == 'cancelled')
					return res.end();

				let finish = (info.reason == 'length') ? 'length' : 'stop';

				if (stream) {
					res.sse(chunk('', finish));
					res.sse('[DONE]');
					res.end();
					return;
				}

				res.json(200, {
					id: id, object: 'text_completion', created: created, model: served_name,
					choices: [{index: 0, text: text.done(), logprobs: null, finish_reason: finish}],
					usage: usage_of(info)
				});
			},

			on_error: function (msg) {
				if (stream) {
					res.sse({error: {message: msg, type: 'server_error', code: null}});
					res.end();
				}
				else {
					api_error(res, 500, msg, 'server_error');
				}
			}
		});
	}

	/* --- routing --------------------------------------------------------------------- */

	function route(req, res) {
		let path = req.path;

		if (index(path, '/v1/') == 0)
			path = substr(path, 3);

		// request length: Content-Length when present (chunked bodies have
		// none), human-readable
		let cl = req.headers['content-length'];
		let len = '';

		if (cl != null) {
			let n = int(cl);

			len = (n >= 1048576) ? sprintf(' (%.1f MiB body)', n / 1048576) :
				(n >= 1024) ? sprintf(' (%.1f KiB body)', n / 1024) :
				` (${n} B body)`;
		}

		warn(`${req.method} ${req.target}${len}\n`);

		if (req.method == 'OPTIONS')
			return res.send(204, {
				'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
				'Access-Control-Allow-Headers': '*'
			}, '');

		if (req.method == 'GET' && (path == '/health' || path == '/'))
			return res.json(200, {status: 'ok', model: served_name,
				busy: eng.busy(), queued: eng.queued(), cache: eng.cached()});

		if (req.method == 'GET' && path == '/models')
			return res.json(200, {object: 'list', data: [model_card()]});

		if (req.method == 'GET' && index(path, '/models/') == 0) {
			// lenient: any id resolves to the loaded model
			return res.json(200, model_card());
		}

		if (req.method == 'POST' && (path == '/chat/completions' || path == '/completions')) {
			let body;

			try {
				body = json(req.body);
			}
			catch (e) {
				return api_error(res, 400, 'invalid JSON body: ' + e.message);
			}

			if (type(body) != 'object')
				return api_error(res, 400, 'request body must be a JSON object');

			if (body.n != null && body.n > 1)
				return api_error(res, 400, 'n > 1 is not supported', 'invalid_request_error', null, 'n');

			if (path == '/chat/completions')
				return chat_completions(req, res, body);

			return completions(req, res, body);
		}

		return api_error(res, 404, `unknown endpoint ${req.method} ${req.path}`, 'invalid_request_error', 'not_found');
	}

	httpd.listen(args.host, args.port, route);
	warn(`ampstein: serving '${served_name}' on http://${args.host}:${args.port}/v1\n`);

	/* --- shutdown: save the cache state on SIGINT/SIGTERM ---------------------- */

	let shutting_down = false;

	function shutdown_cache() {
		if (shutting_down) {
			exit(0);    // second signal: force
			return;
		}

		shutting_down = true;

		try {
			let n = eng.save_state(CACHE_ST);

			writefile(CACHE_META, sprintf('%J', {
				format: 1, model: args.model, tokens: n,
				max_ctx: eng.max_ctx, mtp: eng.mtp_window}));

			warn(`ampstein: saved ${n} cached tokens to ${CACHE_ST}\n`);
		}
		catch (e) {
			warn(`ampstein: cache state not saved (${e.message})\n`);
		}

		// let in-flight device work (and host-func callbacks) drain before
		// tearing down — exiting mid-callback corrupts the heap
		try {
			torch.synchronize();
		}
		catch (e2) {}

		uloop.end();
	}

	// keep the references: a collected signal object unregisters its handler
	let sig_int = uloop.signal('SIGINT', shutdown_cache);
	let sig_term = uloop.signal('SIGTERM', shutdown_cache);

	uloop.run();
}
