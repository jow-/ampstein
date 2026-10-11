// Test: utpl + jinja2utpl — taint-aware Jinja -> utpl transpilation.
//
//   ucode -L <src>/ucode tests/utpl.uc
//
// Part 1 (model-free): embedded templates are transpiled and rendered;
//   the text must be exact and the SEGMENT TAINTS must be exact:
//   template literals trusted (their special tokens tokenize as special
//   tokens), interpolated values tainted (user content stays plain text).
//   Covers: mixed-literal concat splitting, macros (private output
//   buffer, defaults, nested-placeholder splicing), for/set redecl,
//   loop.index, while, filters (tojson/length).
//
// Part 2 (needs a model dir for the tokenizer, else skipped): a tainted
//   segment containing a special-token string must not tokenize into
//   the special token id (the prompt-injection case); the same string
//   in a trusted segment must. The marker is the model's own special
//   token (Llama-3.2: <|eot_id|>, id 128009) — GLM's fence tokens
//   (U+2554 U+256D and friends, written as \u escapes so the raw bytes
//   never pass through editing tools) take the same path.

import * as utpl from 'ampstein.utpl';
import { stat } from 'fs';

const M0 = '\u2554\u256d';   // U+2554 U+256d (GLM fence, as \u escapes)
const M1 = '\u2554\u256d';

let fails = 0;

function check(cond, msg) {
	if (!cond) {
		fails++;
		warn('FAIL: ' + msg + '\n');
	}
}

function seg_str(segs) {
	let s = '';

	for (let i = 0; i < length(segs); i++)
		s += segs[i].s;

	return s;
}

function cmp_segs(name, r, exp_text, exp_segs) {
	check(r.text == exp_text,
	       name + ': text mismatch\n  got: ' + r.text +
	       '\n  exp: ' + exp_text);

	if (length(r.segments) != length(exp_segs)) {
		fails++;
		warn(name + ': ' + length(r.segments) + ' segments, ' +
		       'expected ' + length(exp_segs) + '\n  got: ' +
		       r.segments + '\n');
		return;
	}

	for (let i = 0; i < length(exp_segs); i++) {
		let e = exp_segs[i];
		let g = r.segments[i];

		if (g.t != e[0] || g.s != e[1]) {
			fails++;
			warn(name + ': segment ' + i + ' mismatch\n' +
			       '  got: [' + (g.t ? 't' : 'T') + '] ' + g.s +
			       '\n  exp: [' + (e[0] ? 't' : 'T') + '] ' + e[1] +
			       '\n');
		}
	}
}

let C = {
	who: 'there',
	names: ['a', 'b'],
	msg: 'user message \u2554\u256d content',
	tools: [{ a: 1 }]
};

function run(name, tpl, ctx, exp_text, exp_segs) {
	let t = utpl.load(tpl);
	let r = utpl.render_tpl(t, ctx);

	cmp_segs(name, r, exp_text, exp_segs);
}

/* A: plain text + tainted interpolation */
run('A', 'Hi {{ who }}', C,
    'Hi there',
    [[ 1, 'Hi ' ], [ 0, 'there' ]]);

/* B: mixed-literal concat — the literal markers stay trusted */
run('B', '{{- "' + M0 + '" + who + "' + M0 + '" -}}', C,
    M0 + 'there' + M0,
    [[ 1, M0 ], [ 0, 'there' ], [ 1, M0 ]]);

/* C: macro with a default parameter; all-literal args stay trusted */
run('C',
    '{%- macro greet(who, punct="?") -%}{{- "hey " ~ who ~ punct -}}\n' +
    '{%- endmacro -%}\n' +
    '{{- greet("bob", "!") -}}\n' +
    '{{- greet("ann") -}}\n',
    C,
    'hey bob!hey ann?',
    [[ 1, 'hey bob!hey ann?' ]]);

/* D: for over a ctx sequence (tainted), then a set that shadows the
 * ctx root with a trusted literal — the set must be a plain assignment
 * (redecl) to the preamble's let, and the later read is trusted */
run('D',
    '{%- for x in names -%}{{- x -}}\n{%- endfor -%}' +
    '{%- set names = "done" -%}\n{{- names -}}\n',
    C,
    'abdone',
    [[ 0, 'ab' ], [ 1, 'done' ]]);

/* E: a macro whose output (with its own markers) is spliced into a
 * tainted concat — the macro body INLINES at the call site, so each
 * leaf keeps its own taint: the template literals (incl. the markers)
 * stay trusted, only the user content (msg) is tainted */
run('E',
    '{%- macro q(role, body) -%}{{- "' + M0 + '" + role + "' + M1 +
    '" + body + "' + M0 + '" -}}\n' +
    '{%- endmacro -%}\n' +
    '{{- "' + M0 + 'system" + q("u", msg) -}}\n',
    C,
    M0 + 'system' + M0 + 'u' + M1 + C.msg + M0,
    [[ 1, M0 + 'system' + M0 + 'u' + M1 ],
     [ 0, C.msg ],
     [ 1, M0 ]]);

/* F: filters — tojson keeps the taint, length is clean, literals
 * trusted */
run('F', '{{ tools | tojson }} {{ tools | length }}\n', C,
    '[{"a": 1}] 1',
    [[ 0, '[{"a": 1}]' ], [ 1, ' 1' ]]);

/* G: loop.index is a trusted counter */
run('G',
    '{%- for x in names -%}{{- loop.index -}}:{{- x -}}\n' +
    '{%- endfor -%}\n',
    C,
    '1:a2:b',
    [[ 1, '1:' ], [ 0, 'a' ], [ 1, '2:' ], [ 0, 'b' ]]);

/* H: while over a trusted local counter */
run('H',
    '{%- set i = 0 -%}{%- while i < 3 -%}{{- i -}}' +
    '{%- set i = i + 1 -%}{%- endwhile -%}\n',
    C,
    '012',
    [[ 1, '012' ]]);

/* ------------------------------------------------------------------ */
/* Part 2: tokenizer-level injection check (needs a model dir)        */
/* ------------------------------------------------------------------ */

const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

if (!stat(MODEL)) {
	warn('skip: part 2 (tokenizer) — model dir not found: ' + MODEL +
	       '\n');
}
else {
	/* the marker must be a special token of this model's tokenizer:
	 * user content carrying it, if encoded with special tokens enabled,
	 * would become protocol syntax */
	let MK = '<|eot_id|>';
	let SP = 128009;

	let tok = torch.tokenizer(MODEL);

	/* the marker is ONE special token when trusted */
	let sp = tok.encode(MK, true);

	check(length(sp) == 1 && sp[0] == SP, 'part 2: expected the marker '
	      + 'to encode as the single special id ' + SP + ', got: ' + sp);

	/* same string as plain text: subword pieces, no special id */
	let plain = tok.encode(MK);

	check(index(plain, SP) < 0, 'part 2: plain encode of the marker '
	      + 'contains the special id: ' + plain);

	/* render a template that interpolates it into a tainted segment */
	let t = utpl.load('{{- "Q: " + q -}}\n');
	let r = utpl.render_tpl(t, { q: 'say ' + MK });

	check(length(r.segments) == 2, 'part 2: expected 2 segments, got: '
	      + r.segments);
	check(!r.segments[1].t, 'part 2: value segment must be tainted');

	let tainted_ids = tok.encode(r.segments[1].s);
	let trusted_ids = tok.encode(r.segments[1].s, true);

	check(index(tainted_ids, SP) < 0,
	      'part 2: INJECTION \u2014 tainted encode contains the special '
	      + 'id: ' + tainted_ids);
	check(index(trusted_ids, SP) >= 0,
	      'part 2: trusted encode lacks the special id: ' + trusted_ids);

	/* the full segment-wise encode of the render */
	let ids = utpl.encode_segments(tok, r);

	check(length(ids) == length(tok.encode('Q: ', true)) +
	      length(tainted_ids), 'part 2: encode_segments length ' +
	      length(ids) + ' != ' + (length(tok.encode('Q: ', true)) +
	      length(tainted_ids)));
}

if (fails) {
	warn('utpl: ' + fails + ' failure(s)\n');
	exit(1);
}

print('utpl: all checks passed\n');
