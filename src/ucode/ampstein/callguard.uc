// ucode/callguard.uc — constrained generation of tool calls (--call-guard).
//
// A model that has seen failing calls in its context can keep producing the
// same malformed one (Qwen3.8-27B: `write` with `content` and `content2` ..
// `content468` but no `path`, 205 times in one session). Telling it so in text
// did not bring it back. This guard acts on the sampling itself: it follows the
// generated text, and once `<function=NAME>` identifies a tool of the request it
// vetoes every token that would leave the tool's schema:
//
//   - the function name must be (a prefix of) one of the request's tools;
//   - a parameter name must be one of the tool's properties, not used before;
//   - `</function>` is not allowed while a required parameter is missing;
//   - bulk parameters (content, text, edits, arrays, objects ...) come after
//     the short required ones: `path` first, `content` last.
//
// The engine asks `veto(prefix, tok)` for each token it is about to accept and
// redraws the position from the same logits with the vetoed tokens lowered.
// The guard keeps no model state; `feed(text)` follows the emitted text.
//
// Format: Qwen's XML calls,
//   <tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>
//
//   let g = callguard.qwen(tools, thinking);   // tools: the request's tool list
//   g.feed(text);                              // every emitted piece, in order
//   g.veto([piece, ...], piece)                // would `piece` (after the pending
//                                              // `prefix` pieces) violate the schema?

const BULK = {
	content: true, contents: true, text: true, body: true, data: true, code: true,
	script: true, patch: true, diff: true, edits: true, new_string: true, old_string: true,
	new_text: true, old_text: true, newText: true, oldText: true, file_text: true
};

const T_CALL = '<tool_call>';
const T_ENDCALL = '</tool_call>';
const T_FN = '<function=';
const T_ENDFN = '</function>';
const T_PARAM = '<parameter=';
const T_ENDPARAM = '</parameter>';
const T_ENDTHINK = '</think>';

function schemas(tools, ban, seen) {
	let m = {}, any = false;

	// names in `ban` are left out (not when that would leave no tool at all);
	// so are tools the conversation never called: a model that is refused `write`
	// would otherwise be pushed by the shared first token onto `web_*` (the
	// refused name's lexical neighbours), not onto the tool it was using
	let others = filter(tools ?? [], (t) => index(ban ?? [], (t.function ?? t).name) < 0);

	if (length(ban) && length(seen)) {
		let used = filter(others, (t) => index(seen, (t.function ?? t).name) >= 0);
		let safe = filter(others, (t) => index(['edit', 'write'], (t.function ?? t).name) < 0);

		if (length(used) > 0)
			others = used;

		// a ban means the model is looping or confused, and a wrong call is then
		// safest where it cannot change anything: the shell and `read` first (a
		// model refused `write` mostly announced a command), otherwise anything
		// but the file writers
		let run = filter(used, (t) => index(['bash', 'read'], (t.function ?? t).name) >= 0);

		if (length(run) > 0)
			others = run;
		else if (length(safe) > 0)
			others = safe;
	}

	if (length(others) > 0)
		tools = others;

	for (let t in tools ?? []) {
		let f = t.function ?? t;

		if (type(f.name) != 'string')
			continue;

		let pr = f.parameters?.properties;
		let props = (type(pr) == 'object') ? keys(pr) : [];
		let req = (type(f.parameters?.required) == 'array') ? f.parameters.required : [];
		let bulk = {};

		for (let k in props) {
			let ty = pr[k]?.type;

			if (BULK[k] || ty == 'array' || ty == 'object')
				bulk[k] = true;
		}

		m[f.name] = { props: props, req: req, bulk: bulk };
		any = true;
	}

	return any ? m : null;
}

function keep(buf, n) {
	return (length(buf) > n) ? substr(buf, length(buf) - n) : buf;
}

function starts(s, p) {
	return substr(s, 0, length(p)) == p;
}

/* The strings that may follow in `between`: [forms, can_close] (forms: complete
 * `<parameter=NAME>` openers; the names are unconstrained when the tool lists no
 * properties). */
function allowed(g) {
	let fn = g.fn;
	let pend = false, missing = false;

	for (let r in fn.req) {
		if (!g.used[r]) {
			missing = true;

			if (!fn.bulk[r])
				pend = true;
		}
	}

	let forms = [];

	for (let n in fn.props) {
		if (g.used[n] || (pend && fn.bulk[n]))
			continue;

		push(forms, T_PARAM + n + '>');
	}

	return { forms: forms, close: !missing };
}

/* Advance g by text; true when the text leaves the schema (g then continues
 * unconstrained until the call ends). */
function adv(g, text) {
	let bad = false;

	g.buf += text;

	for (let round = 0; round < 64; round++) {
		let m = g.mode;

		if (m == 'think') {
			let i = index(g.buf, T_ENDTHINK);

			if (i < 0) {
				g.buf = keep(g.buf, length(T_ENDTHINK));
				return bad;
			}

			g.buf = substr(g.buf, i + length(T_ENDTHINK));
			g.mode = 'out';
		}
		else if (m == 'out') {
			let i = index(g.buf, T_CALL);

			if (i < 0) {
				g.buf = keep(g.buf, length(T_CALL));
				return bad;
			}

			g.buf = substr(g.buf, i + length(T_CALL));
			g.mode = 'call';
		}
		else if (m == 'call') {
			let s = ltrim(g.buf);

			if (s == '')
				return bad;

			if (length(s) < length(T_FN)) {
				if (!starts(T_FN, s))
					g.mode = 'free';
				else
					return bad;

				continue;
			}

			if (!starts(s, T_FN)) {
				g.mode = 'free';
				continue;
			}

			let rest = substr(s, length(T_FN));
			let k = index(rest, '>');

			if (k < 0) {
				// the name so far must lead to one of the tools
				let ok = false;

				for (let n in keys(g.tools)) {
					if (starts(n, rest)) {
						ok = true;
						break;
					}
				}

				if (!ok) {
					g.mode = 'free';     // (never trap the model in a name no tool continues)
					return true;
				}

				return bad;
			}

			let name = substr(rest, 0, k);

			g.fn = g.tools[name];

			if (g.fn == null)
				bad = true;

			g.buf = substr(rest, k + 1);
			g.used = {};
			g.mode = g.fn ? 'between' : 'free';
		}
		else if (m == 'between') {
			let s = ltrim(g.buf);

			if (s == '')
				return bad;

			let a = allowed(g);
			let named = length(g.fn.props) > 0;
			let done = null, partial = false;

			if (!named && starts(s, T_PARAM)) {
				// unconstrained names: any `<parameter=...>`
				let k = index(s, '>');

				if (k < 0)
					return bad;

				g.used[substr(s, length(T_PARAM), k - length(T_PARAM))] = true;
				g.buf = substr(s, k + 1);
				g.mode = 'value';
				continue;
			}

			for (let f in a.forms) {
				if (starts(f, s))
					partial = true;
				else if (starts(s, f) && done == null)
					done = f;
			}

			if (a.close) {
				if (starts(T_ENDFN, s))
					partial = true;
				else if (starts(s, T_ENDFN))
					done = T_ENDFN;
			}

			if (!named && starts(T_PARAM, s))
				partial = true;

			if (done == T_ENDFN) {
				g.buf = substr(s, length(T_ENDFN));
				g.mode = 'post';
			}
			else if (done != null) {
				g.used[substr(done, length(T_PARAM), length(done) - length(T_PARAM) - 1)] = true;
				g.buf = substr(s, length(done));
				g.mode = 'value';
			}
			else if (partial) {
				return bad;
			}
			else {
				// nothing in the schema starts like this
				if (length(a.forms) == 0 && !a.close)
					g.mode = 'free';     // (no way to satisfy the schema: do not trap the model)
				else {
					bad = true;
					g.mode = 'free';
				}
			}
		}
		else if (m == 'value') {
			let i = index(g.buf, T_ENDPARAM);

			if (i < 0) {
				g.buf = keep(g.buf, length(T_ENDPARAM));
				return bad;
			}

			g.buf = substr(g.buf, i + length(T_ENDPARAM));
			g.mode = 'between';
		}
		else {      // 'post', 'free': to the end of the call
			let i = index(g.buf, T_ENDCALL);

			if (i < 0) {
				g.buf = keep(g.buf, length(T_ENDCALL));
				return bad;
			}

			g.buf = substr(g.buf, i + length(T_ENDCALL));
			g.mode = 'out';
		}
	}

	return bad;
}

/* The text that must follow in the state of g, for a position whose token
 * was refused: the rest of the only legal opener (the first by property order;
 * required short parameters before bulk ones), or the end of the call when
 * everything required is there. null when no deterministic continuation exists. */
function suggest(g) {
	if (g.mode == 'call') {
		let s = ltrim(g.buf);

		if (!starts(s, T_FN))
			return null;

		let rest = substr(s, length(T_FN));

		// (nothing typed yet: which tool is the model's choice, the caller redraws)
		if (rest == '' || index(rest, '>') >= 0)
			return null;

		for (let n in keys(g.tools))
			if (starts(n, rest))
				return substr(n, length(rest)) + '>\n';

		return null;
	}

	if (g.mode == 'between') {
		let s = ltrim(g.buf);
		let a = allowed(g);
		let lead = (g.buf == '') ? '\n' : '';

		for (let f in a.forms)
			if (starts(f, s))
				return lead + substr(f, length(s)) + '\n';

		if (a.close && starts(T_ENDFN, s))
			return lead + substr(T_ENDFN, length(s)) + '\n' + T_ENDCALL;
	}

	return null;
}

/* The conversation as the model should see it: earlier assistant tool calls
 * without the arguments their tool does not declare. A stray `content2` that a
 * harness silently ignores stays in the context as an example; the model copies
 * it, the copies multiply, and a few hundred calls later the real `path` is
 * crowded out (openwrt-minibuild session: one `content2` at 15:53, 93 per call
 * at 17:00, no `path` at 17:23). The cleaning is a pure function of the request,
 * applied to the prompt only; the client's own history is untouched. Only calls
 * of tools with declared properties change, and only by dropping unknown keys.
 * Returns `messages` itself when nothing changes. */
export function clean_messages(messages, tools) {
	if (type(messages) != 'array' || type(tools) != 'array')
		return messages;

	let props = {};

	for (let t in tools) {
		let f = t.function ?? t;
		let pr = f.parameters?.properties;

		if (type(f.name) == 'string' && type(pr) == 'object' && length(keys(pr)) > 0)
			props[f.name] = pr;
	}

	let out = null;

	for (let i = 0; i < length(messages); i++) {
		let m = messages[i];

		if (type(m) != 'object' || m.role != 'assistant' || type(m.tool_calls) != 'array')
			continue;

		let calls = null;

		for (let j = 0; j < length(m.tool_calls); j++) {
			let tc = m.tool_calls[j];
			let fn = tc?.function;
			let pr = (type(fn?.name) == 'string') ? props[fn.name] : null;

			if (pr == null)
				continue;

			let raw = fn.arguments, a = raw;

			if (type(raw) == 'string') {
				try {
					a = json(raw);
				}
				catch (e) {
					continue;
				}
			}

			if (type(a) != 'object')
				continue;

			let keep = {}, dropped = 0;

			for (let k in keys(a)) {
				if (pr[k] != null)
					keep[k] = a[k];
				else
					dropped++;
			}

			// (a call with nothing declared in it is not made emptier)
			if (dropped == 0 || length(keys(keep)) == 0)
				continue;

			calls ??= [ ...m.tool_calls ];
			calls[j] = { ...tc, function: { ...fn, arguments: (type(raw) == 'string') ? sprintf('%J', keep) : keep } };
		}

		if (calls != null) {
			out ??= [ ...messages ];
			out[i] = { ...m, tool_calls: calls };
		}
	}

	return out ?? messages;
}

/* Besides no-op `edit`s (see noop_edit), only `write` is watched: a model that writes the same script again and again
 * under a new name (each answered by success, so no failure policy sees it).
 * Repeated `edit`s are not banned: they are legitimate often enough (many
 * similar fixes, probing after a failed match), and a ban there cost more than
 * it saved. */
const REPEAT_TOOLS = ['write'];

/* An `edit` that cannot change anything: no replacement at all, `/dev/null` as
 * the target, or every `newText` equal to its `oldText`. A model that announces
 * another tool and emits `edit` anyway fills the call with such placeholders;
 * they fail ("No changes made") or are refused one by one, which no equality of
 * arguments sees. One alone happens, two within a few calls are the loop. */
function noop_edit(h) {
	if (h.name != 'edit')
		return false;

	let a = null;

	try {
		a = json(h.arguments);
	}
	catch (e) {
		return false;
	}

	if (type(a) != 'object')
		return false;

	if (a.path == '/dev/null')
		return true;

	let e = a.edits;

	if (type(e) == 'string') {
		try {
			e = json(e);
		}
		catch (err) {
			return false;
		}
	}

	if (type(e) != 'array')
		return false;

	if (length(e) == 0)
		return true;

	for (let r in e)
		if (type(r) != 'object' || r.oldText != r.newText)
			return false;

	return true;
}

/* A call's arguments without its target and with digits folded: what is being
 * done, not to which numbered file. null when not an object. */
function body_of(h) {
	let a = null;

	try {
		a = json(h.arguments);
	}
	catch (e) {
		return null;
	}

	if (type(a) != 'object')
		return null;

	let o = {};

	for (let k in keys(a))
		if (k != 'path' && k != 'file' && k != 'file_path' && k != 'filepath')
			o[k] = a[k];

	return replace(sprintf('%J', o), /[0-9]+/g, '#');
}

/* Paths that differ only by a stray suffix, digit or letter: same directory, and the
 * names without extension equal after folding digits, one a prefix of the other
 * with a tail of at most 3 characters, or within edit distance 2. */
function similar_paths(p, q) {
	if (type(p) != 'string' || type(q) != 'string' || p == '' || q == '')
		return false;

	if (p == q)
		return true;

	let dp = match(p, /^(.*\/)?([^\/]*)$/), dq = match(q, /^(.*\/)?([^\/]*)$/);
	let da = dp[1] ?? '', db = dq[1] ?? '';

	if (da != db)
		return false;

	let a = replace(dp[2], /\.[^.]*$/, ''), b = replace(dq[2], /\.[^.]*$/, '');

	if (a == b || replace(a, /[0-9]+/g, '#') == replace(b, /[0-9]+/g, '#'))
		return true;

	let la = length(a), lb = length(b);

	if ((index(b, a) == 0 && lb - la <= 3) || (index(a, b) == 0 && la - lb <= 3))
		return true;

	// (stems of a letter or two are all within distance 2 of each other)
	if (la > lb + 2 || lb > la + 2 || la > 64 || lb > 64 || (la < 4 && lb < 4))
		return false;

	// edit distance <= 2 (two rows)
	let prev = [], cur = [];

	for (let j = 0; j <= lb; j++)
		prev[j] = j;

	for (let i = 1; i <= la; i++) {
		cur[0] = i;

		for (let j = 1; j <= lb; j++) {
			let c = (substr(a, i - 1, 1) == substr(b, j - 1, 1)) ? 0 : 1;
			let m = prev[j - 1] + c;

			if (prev[j] + 1 < m)
				m = prev[j] + 1;

			if (cur[j - 1] + 1 < m)
				m = cur[j - 1] + 1;

			cur[j] = m;
		}

		prev = cur;
		cur = [];
	}

	return prev[lb] <= 2;
}

function same_file_again(a, b) {
	let x = null, y = null;

	try {
		x = json(a.arguments);
		y = json(b.arguments);
	}
	catch (e) {
		return false;
	}

	if (type(x) != 'object' || type(y) != 'object')
		return false;

	if (type(x.content) != 'string' || type(y.content) != 'string' || rtrim(x.content) != rtrim(y.content))
		return false;

	// the same content, byte for byte (trailing whitespace aside). Of some size it
	// is the repeat itself that matters, whatever the new name is (`_a`, `_final`,
	// `_again`): any file of the same directory. Short or empty contents (`.gitkeep`,
	// `__init__.py`) are legitimately equal, so they need a nearly equal path too.
	if (length(rtrim(x.content)) >= 24 && type(x.path) == 'string' && type(y.path) == 'string' &&
	    match(x.path, /^(.*\/)?/)[0] == match(y.path, /^(.*\/)?/)[0])
		return true;

	return similar_paths(x.path, y.path);
}

/* The tool names to ban for the next response, from the conversation's calls
 * (toolhist.uc form: {name, arguments}, oldest first). Banning the name at the
 * sampling level is what the text of a note could not do: the model announces
 * `read` and emits the previous `edit` again, because the copy is the likeliest
 * continuation; with the name gone, the announced tool is. */
export function repeated(history) {
	let n = length(history ?? []);

	// two no-op edits among the last four calls
	if (n >= 2 && history[n - 1].name == 'edit') {
		let k = 0;

		for (let i = (n > 4) ? n - 4 : 0; i < n; i++)
			if (noop_edit(history[i]))
				k++;

		if (k >= 2)
			return ['edit'];
	}

	if (n < 2 || index(REPEAT_TOOLS, history[n - 1].name) < 0)
		return [];

	let a = history[n - 1], b = history[n - 2];

	// the same content written again to a nearly identical path (`/tmp/pyval.sh`,
	// `/tmp/pyval2.sh`): a model that announces "now I run it" and copies its
	// previous call instead. Two such writes are enough.
	if (a.name == 'write' && b.name == 'write' && same_file_again(b, a))
		return ['write'];

	// the same call aimed at a new file each time (`/tmp/stat7.sh`, `/tmp/stat8.sh`):
	// equal once the target and the digits are taken out
	if (n >= 3) {
		let sig = body_of(a);

		if (sig != null && history[n - 2].name == a.name && history[n - 3].name == a.name &&
		    body_of(history[n - 2]) == sig && body_of(history[n - 3]) == sig)
			return [a.name];
	}

	return [];
}

/* tools: the request's tool list (OpenAI form); thinking: generation starts
 * inside a reasoning block; ban: tool names not to be called in this response;
 * seen: names of the tools the conversation has called (a ban then leaves only
 * those). Returns null without tool definitions. */
export function qwen(tools, thinking, ban, seen) {
	let tl = schemas(tools, ban, seen);

	if (tl == null)
		return null;

	let g = { mode: thinking ? 'think' : 'out', buf: '', tools: tl, fn: null, used: {} };

	return {
		feed: function (text) {
			if (text != null && text != '')
				adv(g, text);
		},

		/* true when `piece`, following the pending `prefix` pieces, leaves the
		 * schema of the call in progress */
		veto: function (prefix, piece) {
			if (piece == null || piece == '')
				return false;

			let text = join('', prefix) + piece;

			// outside a constrained span the text must not even start one
			if ((g.mode == 'out' || g.mode == 'think' || g.mode == 'post' || g.mode == 'free') &&
			    index(g.buf + text, T_CALL) < 0 && index(g.buf + text, T_ENDTHINK) < 0)
				return false;

			if (g.mode == 'value' && index(g.buf + text, T_ENDPARAM) < 0)
				return false;

			let c = { ...g, used: { ...g.used } };

			for (let p in prefix)
				adv(c, p);

			return adv(c, piece);
		},

		/* the text to force after the pending `prefix` pieces when the next token
		 * was refused (null: none, the caller redraws) */
		fix: function (prefix) {
			let c = { ...g, used: { ...g.used } };

			for (let p in prefix)
				adv(c, p);

			return suggest(c);
		},

		/* the call in progress (null outside one), for logs */
		state: function () {
			return { mode: g.mode, fn: g.fn ? true : false, used: keys(g.used) };
		}
	};
}
