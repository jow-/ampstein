// Test: the jinja2 -> utpl transpiler on REAL checkpoint templates.
// Each family renders the checkpoint's own chat_template.jinja
// (captured into tests/fixtures/ by the matching oracle script) for a
// battery of conversations; every output must match the jinja2 oracle
// byte-for-byte, and the taint segmentation must keep user content
// separate from template text.
//
// Families:
//   Qwen3.8  (tests/oracle/make_chat_qwen35_oracle.py): macros, |items,
//               namespace, [::-1], |default, reasoning-effort validation
//   GLM-5.3  (tests/oracle/make_chat_glm5_oracle.py): .items(), .split(),
//               .strip(), |capitalize, |replace, range() + break,
//               tool-response re-sorting, string-truthiness macros
//
// This is the regression lock for the macro/|trim bug (a macro that
// returned its untrusted() placeholder text made |trim and truthiness
// operate on the marker instead of the value) and for |items fidelity
// (mapping -> pairs, undefined -> empty, anything else raises, as
// jinja2's do_items).
//
//   ucode -L <src>/ucode tests/utpl_template.uc

import * as utpl from 'ampstein.utpl';
import { readfile, stat } from 'fs';

const DIR = `${sourcepath(0, true)}/fixtures`;

let FAMS = [
	{ tpl: DIR + '/chat_qwen35_template.jinja',
	  cases: DIR + '/chat_qwen35_cases.json', field: 'expected' },
];

/* Generated fixtures (gitignored): skip, not fail, when absent (CTest
 * SKIP_RETURN_CODE) — regenerate with the matching oracle script. */
let have = 0;

for (let i = 0; i < length(FAMS); i++) {
	if (!stat(FAMS[i].tpl) || !stat(FAMS[i].cases))
		FAMS[i] = null;
	else
		have++;
}

if (have == 0) {
	warn('skip: no template fixtures in ' + DIR + '\n');
	exit(125);
}

/* Tool-call arguments arrive as a JSON string (OpenAI wire format); the
 * templates iterate them as a mapping (HF semantics). GLM fixture cases
 * already carry mappings — the conversion is a no-op for them. Mirror
 * the server's template_ctx normalization. */
function call_arguments(tc) {
	let a = tc.arguments;

	if (type(a) == 'string') {
		if (trim(a) == '')
			return {};

		try {
			a = json(a);
		}
		catch (e) {
			return { arguments: tc.arguments };
		}
	}

	return (type(a) == 'object') ? a : {};
}

function fix_messages(msgs) {
	for (let i = 0; i < length(msgs); i++) {
		let tcs = msgs[i].tool_calls;

		if (!tcs)
			continue;

		for (let j = 0; j < length(tcs); j++)
			tcs[j].function.arguments = call_arguments(tcs[j].function);
	}

	return msgs;
}

let fails = 0, total = 0;

for (let fi = 0; fi < length(FAMS); fi++) {
	let F = FAMS[fi];

	if (F == null) {
		warn('skip family: fixtures not found\n');
		continue;
	}

	let t = utpl.load(readfile(F.tpl));
	let cases = json(readfile(F.cases));

	for (let i = 0; i < length(cases); i++) {
		let c = cases[i];
		total++;
		let ctx = {
			messages: fix_messages(c.messages),
			tools: c.tools,
			add_generation_prompt: true
		};

		if (c.kwargs)
			for (let k in c.kwargs)
				ctx[k] = c.kwargs[k];

		let r = utpl.render_tpl(t, ctx);

		if (r.text != c[F.field]) {
			fails++;

			let n = min(length(r.text), length(c[F.field]));
			let j = 0;

			while (j < n && ord(r.text, j) == ord(c[F.field], j))
				j++;

			warn(`case ${c.name || i}: MISMATCH at byte ${j} ` +
			     `(got ${length(r.text)} bytes, expected ${length(c[F.field])})\n` +
			     `  got: ${sprintf('%J', substr(r.text, max(0, j - 40), 120))}\n` +
			     `  exp: ${sprintf('%J', substr(c[F.field], max(0, j - 40), 120))}\n`);
		}
		else {
			/* the template text is trusted + user content tainted: every
			 * case renders both kinds of segment */
			let taints = 0;

			for (let s = 0; s < length(r.segments); s++)
				taints += r.segments[s].t ? 1 : 0;

			assert(length(r.segments) > 1,
			       `case ${c.name || i}: expected >1 segment, got ${length(r.segments)}`);
			assert(taints > 0,
			       `case ${c.name || i}: no tainted segment (user content must be taint-tracked)`);
			printf('case %s: ok (%d bytes, %d segments, %d tainted)\n',
				c.name || i, length(r.text), length(r.segments), taints);
		}
	}
}

if (fails)
	die(`${fails} case(s) mismatched`);

print(`utpl_template: all ${total} cases byte-exact vs jinja2\n`);
