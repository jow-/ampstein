// Test: ucode/chat_qwen35.uc render() vs the checkpoint's own
// chat_template.jinja rendered by jinja2 (fixture from
// tests/oracle/make_chat_qwen35_oracle.py), byte-exact; plus the
// streaming output parser.
//
//   ucode -L <src>/ucode tests/chat_qwen35.uc

import * as chat from 'ampstein.chat_qwen35';
import { readfile, stat } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/chat_qwen35_cases.json`;

/* The fixture is generated (gitignored): skip, not fail, when it is
 * absent (CTest SKIP_RETURN_CODE) — regenerate with
 * tests/oracle/make_chat_qwen35_oracle.py. */
if (!stat(FIXTURE)) {
	warn('skip: fixture not found: ' + FIXTURE + '\n');
	exit(125);
}

let cases = json(readfile(FIXTURE));
let fails = 0;

for (let i = 0; i < length(cases); i++) {
	let c = cases[i];
	let kw = c.kwargs || {};
	let r = chat.render(c.messages, c.tools, kw);
	let got = r.prefix + r.gen;

	if (got != c.expected) {
		fails++;

		let n = 0;

		while (n < length(got) && substr(got, n, 1) == substr(c.expected, n, 1))
			n++;

		warn(`case ${c.name}: MISMATCH at byte ${n}\n  got: %J\n  exp: %J\n`,
			substr(got, n - 40, 120), substr(c.expected, n - 40, 120));
	}
	else {
		// the prefix must end exactly at the last message boundary
		assert(substr(r.prefix, -11) == '<|im_end|>\n', `case ${c.name}: prefix boundary`);
		printf('case %s: ok (%d bytes)\n', c.name, length(got));
	}
}

/* parser: feed the stream in awkward pieces (tags split across feeds) */
function run_parser(thinking, text, step, stops) {
	let p = chat.parser(thinking, stops);
	let evs = [];

	for (let i = 0; i < length(text); i += step)
		push(evs, ...p.feed(substr(text, i, step)));

	push(evs, ...p.finish());

	// merge adjacent text events
	let out = [];

	for (let i = 0; i < length(evs); i++) {
		let e = evs[i];
		let last = length(out) ? out[length(out) - 1] : null;

		if (last && last.type == e.type && (e.type == 'reasoning' || e.type == 'content'))
			last.text += e.text;
		else
			push(out, { ...e });
	}

	return out;
}

let gen = "Let me think.\nOK.\n</think>\n\nI'll read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run>\n<parameter=cmd>\nls\n-la\n</parameter>\n</function>\n</tool_call>";

for (let step = 1; step <= 7; step++) {
	let ev = run_parser(true, gen, step);

	assert(length(ev) == 4, `parser step ${step}: ${length(ev)} events: ${sprintf('%J', ev)}`);
	assert(ev[0].type == 'reasoning' && ev[0].text == 'Let me think.\nOK.\n', `step ${step}: reasoning`);
	assert(ev[1].type == 'content' && ev[1].text == "I'll read it.", `step ${step}: content ${sprintf('%J', ev[1])}`);
	assert(ev[2].type == 'tool_call' && ev[2].name == 'read_file' &&
	       ev[2].params.path == 'README.md' && ev[2].params.limit == '20', `step ${step}: call 1`);
	assert(ev[3].type == 'tool_call' && ev[3].params.cmd == 'ls\n-la', `step ${step}: call 2`);
}

let tools = cases[6].tools;
let args = chat.typed_arguments({ name: 'read_file', params: { path: '42', limit: '20' } }, tools);

assert(args.path === '42' && args.limit === 20, `typed args: ${sprintf('%J', args)}`);

// stop strings and no-thinking mode; text that merely looks like a tag
let ev = run_parser(false, 'Hello <tool world. END here', 3, ['END']);

assert(length(ev) == 2 && ev[0].text == 'Hello <tool world. ' && ev[1].type == 'stop',
	`stop: ${sprintf('%J', ev)}`);

// unterminated reasoning (length cut) stays reasoning
ev = run_parser(true, 'thinking forever', 4);
assert(length(ev) == 1 && ev[0].type == 'reasoning', `unterminated think: ${sprintf('%J', ev)}`);

/* make_text: buffer-backed raw text with incremental stop search */

function feed_all(t, pieces) {
	let vis = [];

	for (let i = 0; i < length(pieces); i++)
		push(vis, t.feed(pieces[i]));

	return vis;
}

// stop split across feeds: 'Hello END' truncates to 'Hello '
let t = chat.make_text(['END'], false);
let vis = feed_all(t, ['He', 'll', 'o E', 'ND']);
assert(vis[0] == 'He' && vis[1] == 'll' && vis[2] == 'o E' && vis[3] == '',
	`make_text vis: ${sprintf('%J', vis)}`);
assert(t.stopped() && t.done() == 'Hello ', `make_text truncation: ${sprintf('%J', t.done())}`);

// the earliest stop wins, regardless of list order
t = chat.make_text(['XXX', 'YY'], false);
vis = feed_all(t, ['aXXYYbXXX']);
assert(vis[0] == 'aXX' && t.stopped() && t.done() == 'aXX',
	`earliest stop: ${sprintf('%J', vis)} ${sprintf('%J', t.done())}`);

// a 3-char stop completed by the second piece
t = chat.make_text(['ABC'], false);
vis = feed_all(t, ['AB', 'C']);
assert(vis[0] == 'AB' && vis[1] == '' && t.stopped() && t.done() == '',
	`overlap stop: ${sprintf('%J', vis)}`);

// already-scanned bytes never re-trigger a stop
t = chat.make_text(['AB'], false);
vis = feed_all(t, ['x', 'AB', 'AB']);
assert(vis[0] == 'x' && vis[1] == '' && vis[2] == '' && t.done() == 'x',
	`confirmed range: ${sprintf('%J', vis)}`);

// no stop present
t = chat.make_text(['\n'], false);
vis = feed_all(t, ['a', 'b']);
assert(vis[0] == 'a' && vis[1] == 'b' && !t.stopped() && t.done() == 'ab',
	`no stop: ${sprintf('%J', vis)}`);

// empty pieces and empty text
t = chat.make_text(['x'], false);
assert(t.feed('') == '' && t.done() == '', `empty feed`);

// an empty stop string stops at position 0
t = chat.make_text([''], false);
assert(t.feed('x') == '' && t.stopped() && t.done() == '', `empty stop`);

// the stop-overlap tail is carried across feeds: 'ZZ' completes only
// because the previous piece's tail was kept; done() returns the
// truncated full text
t = chat.make_text(['END']);
vis = feed_all(t, ['He', 'll', 'o E', 'ND']);
assert(vis[0] == 'He' && vis[1] == 'll' && vis[2] == 'o E' && vis[3] == '' &&
	t.stopped() && t.done() == 'Hello ', `truncation: ${sprintf('%J', vis)}`);

t = chat.make_text(['ZZ']);
vis = feed_all(t, ['a', 'b', 'c', 'Z', 'Z']);
assert(vis[0] == 'a' && vis[1] == 'b' && vis[2] == 'c' && vis[3] == 'Z' &&
	vis[4] == '' && t.stopped() && t.done() == 'abc',
	`tail carry: ${sprintf('%J', vis)}`);

t = chat.make_text([]);
vis = feed_all(t, ['xy', 'zw']);
assert(vis[0] == 'xy' && vis[1] == 'zw' && !t.stopped() && t.done() == 'xyzw',
	`no stops: ${sprintf('%J', vis)}`);

// differential fuzz against the original (string-concatenation) algorithm:
// per-piece visible text, the stop point and the final text must all match
function ref_state(stops) {
	return { stops: stops, text: '', stopped: false };
}

function ref_feed(st, t) {
	if (st.stopped)   // the server never feeds after a stop
		return '';

	let before = length(st.text);

	st.text += t;

	for (let i = 0; i < length(st.stops); i++) {
		let at = index(st.text, st.stops[i]);

		if (at >= 0) {
			st.text = substr(st.text, 0, at);
			st.stopped = true;
		}
	}

	return (length(st.text) > before) ? substr(st.text, before) : '';
}

{
	// Park-Miller LCG: stays below 2^53 so the arithmetic is exact
	let seed = 123456789;
	let next = function () {
		seed = (seed * 48271) % 2147483647;

		return seed;
	};
	let stops = ['ABC', 'XY', 'ZQZ'];
	let ref = ref_state(stops);
	let nw = chat.make_text(stops);
	let alpha = 'AXYZQBCx y\n.';

	for (let i = 0; i < 4000; i++) {
		let n = 1 + next() % 5;
		let piece = '';

		for (let j = 0; j < n; j++)
			piece += substr(alpha, next() % length(alpha), 1);

		let rv = ref_feed(ref, piece);
		let nv = nw.feed(piece);

		assert(rv == nv, `fuzz ${i}: vis ${sprintf('%J', rv)} != ${sprintf('%J', nv)}`);
		assert(ref.stopped == nw.stopped(), `fuzz ${i}: stop point ${sprintf('%J', piece)}`);
	}

	assert(ref.text == nw.done(), `fuzz: final text`);
	printf('make_text: fuzz ok (%d pieces, %d bytes, stop %s)\n',
		4000, length(ref.text), ref.stopped ? 'hit' : 'missed');
}

if (fails)
	die(`${fails} template case(s) mismatched`);

print('chat_qwen35: all checks passed\n');
