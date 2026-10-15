// examples/prompt_policy_compaction.uc — make oversized summarization requests
// fit the context (ampstein --prompt-hook).
//
// An agent harness that compacts its history sends the whole conversation to the
// model, as text, and asks for a summary; the answer replaces the history. pi
// builds that request from `serializeConversation()`, which includes every
// thinking block verbatim: a session whose live context is 262k tokens (the
// chat template drops earlier turns' thinking) serializes to well over 500k
// tokens, and the request that was meant to shrink the session fails with the
// same "maximum context length" error. Nothing is lost by answering such a
// request from a reduced copy: the client keeps only the summary.
//
// The script runs for every chat request, once its prompt is rendered (this one
// only acts on fits == false); the request state arrives as globals:
//   messages    [{role, content, ...}] the request's messages (do not mutate)
//   tools       the request's tool definitions
//   tokens      the rendered prompt's length
//   max_tokens  the room the request asks for (0: none)
//   context     the engine's context length
//   limit       the prompt length that leaves that room (context - 1 - max_tokens)
//   fits        tokens + max_tokens < context
//   round       0, 1, ...: how often the script already rewrote this request
// and the return value is the verdict: an array of messages (or an object with
// `messages` and / or `tools`) replaces the request's; the prompt is rendered
// again and the script asked again, up to --prompt-hook-max rounds. Anything
// else leaves the request alone (an oversized one then gets the usual 400).
//
// Recognised: pi's summarizer request (agent-session `compact()`): a system
// message "You are a context summarization assistant. ...", one user message
// `<conversation>\n...\n</conversation>\n\n[<previous-summary>...]\n\n` plus the
// SUMMARIZATION_PROMPT ("The messages above are a conversation to summarize.
// ...") or the UPDATE one ("The messages above are NEW conversation messages
// ..."), max_tokens 0.8 x reserveTokens. The conversation is
// `[User]: ` / `[Assistant thinking]: ` / `[Assistant]: ` / `[Assistant tool
// calls]: ` / `[Tool result]: ` entries joined by blank lines. It is reduced,
// step by step, until its estimated size fits: thinking dropped, tool results
// and calls cut, then the oldest entries left out (the newest are what the
// summary has to be right about).

const KEEP_HEAD = 0.1;       // share of the character budget kept for the oldest entries
const RESULT_MAX = [600, 200];     // characters of a tool result / call, per step
const SAFETY = 0.85;         // of the estimate, per round (it compounds)

const OPEN = '<conversation>\n';
const CLOSE = '\n</conversation>\n\n';

function text_of(c) {
	if (type(c) == 'string')
		return c;

	return join('', map(filter(c ?? [], (p) => p?.type == 'text'), (p) => p.text));
}

function cut(s, n) {
	return (length(s) > n) ? substr(s, 0, n) + `\n[... ${length(s) - n} more characters omitted]` : s;
}

// '[Kind]: text' entries; a piece that does not start one belongs to the entry before.
function entries(conv) {
	let out = [];

	for (let piece in split('\n\n' + conv, '\n\n[')) {
		let m = match(piece, /^(User|Assistant thinking|Assistant tool calls|Assistant|Tool result)\]: /);

		if (m || !length(out))
			push(out, {kind: m ? m[1] : 'User', text: m ? substr(piece, length(m[0])) : piece});
		else
			out[-1].text += '\n\n[' + piece;
	}

	return out;
}

function render(es) {
	return join('\n\n', map(es, (e) => `[${e.kind}]: ${e.text}`));
}

function size(es) {
	let n = 0;

	for (let e in es)
		n += length(e.text) + length(e.kind) + 6;

	return n;
}

if (fits)
	return null;

let last = messages[length(messages) - 1];

if (last?.role != 'user' || !match(text_of(messages[0]?.content), /^You are a context summarization assistant\./))
	return null;

let text = text_of(last.content);
let a = index(text, OPEN), z = rindex(text, CLOSE);

if (a < 0 || z < a || !match(substr(text, z), /\n\nThe messages above are (a conversation to summarize|NEW conversation messages)/))
	return null;

let head = substr(text, 0, a + length(OPEN));
let tail = substr(text, z);
let es = entries(substr(text, a + length(OPEN), z - a - length(OPEN)));

// the characters one token is worth in this request, and the tokens that are not conversation
let chars = 0;

for (let m in messages)
	chars += length(text_of(m.content));

let per_token = chars / (tokens || 1);
let fixed = (length(head) + length(tail)) / per_token + (length(messages) - 1) * 16;
let budget = (limit - fixed) * per_token;

for (let i = 0; i <= round; i++)
	budget *= SAFETY;

if (budget < 2000)
	return null;

let before = size(es);

// 1. thinking
es = filter(es, (e) => e.kind != 'Assistant thinking');

// 2. long tool results and calls
for (let n in RESULT_MAX) {
	if (size(es) <= budget)
		break;

	for (let e in es)
		if (e.kind == 'Tool result' || e.kind == 'Assistant tool calls')
			e.text = cut(e.text, n);
}

// 3. oldest entries out, a little of the very first ones kept
if (size(es) > budget) {
	let keep_head = budget * KEEP_HEAD, h = [], t = [], hs = 0, ts = 0;

	for (let i = 0; i < length(es) && hs + length(es[i].text) <= keep_head; i++) {
		push(h, es[i]);
		hs += length(es[i].text) + 20;
	}

	for (let i = length(es) - 1; i >= length(h) && hs + ts + length(es[i].text) <= budget; i--) {
		unshift(t, es[i]);
		ts += length(es[i].text) + 20;
	}

	let skipped = length(es) - length(h) - length(t);

	if (skipped > 0)
		push(h, {kind: 'User', text: `[... ${skipped} entries of the conversation omitted to fit the context ...]`});

	es = [...h, ...t];
}

if (size(es) >= before)
	return null;

let out = [...slice(messages, 0, -1), {...last, content: head + render(es) + tail}];

return out;
