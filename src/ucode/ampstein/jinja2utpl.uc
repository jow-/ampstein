// ucode/jinja2utpl.uc — Jinja (chat_template.jinja) -> utpl transpiler.
//
// The transpiler
//   1. lexes Jinja (text / {{}} / {% %} / {# #}, with the `-` strip flags),
//   2. parses a Jinja subset: for/if/while/set/macro + the full expression
//      grammar (filters, tests, ~, in, is, ternaries, calls, literals),
//   3. resolves jinja2 whitespace (trim_blocks, lstrip_blocks, keep the
//      trailing-newline rule, explicit `-` strips) into the emitted text,
//   4. runs a static taint analysis: every value originating in the render
//      context (a free variable: messages, tools, ...) is TAINTED and the
//      taint propagates through concatenation, filters, assignments and
//      macro calls. Every {{...}} output that can carry context data is
//      emitted as tpl_out_untrusted(<expr>); template literals and
//      context-free values as tpl_out(<expr>),
//   5. emits a PLAIN ucode program (no ucode template mode) that
//      loadstring(src, {raw_mode: true}) compiles. The program starts
//      with explicit import lines of INDIVIDUAL symbols (no wildcards:
//      imported names become stack slots in the VM instead of namespace
//      property lookups):
//
//          import {...} from 'ampstein.tv';      value ops (ucode/tv.uc)
//          import {...} from 'ampstein.utpl';    output + capture helpers
//
//      Every output byte is an explicit call into the sink (ucode/sink.uc
//      via ucode/utpl.uc): the value carries the trust (MIXED runs), and
//      the entry point is a failsafe — tpl_out_untrusted() forces a
//      single tainted run on anything that arrives without the tag, so a
//      static-analysis gap fails closed, never open. A jinja macro is an
//      ad-hoc template: its body is INLINED at each call site (output
//      position: a statement block on the live stream; value position:
//      an IIFE under a sink capture, whose cap_end() is a MIXED value
//      with the internal trust split intact).
//
// Unsupported constructs are rejected loudly (die) — never miscompiled.

import * as struct from 'struct';

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

function is_digit(c) {
	return c >= '0' && c <= '9';
}

function is_ident_start(c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

function is_ident_cont(c) {
	return is_ident_start(c) || is_digit(c);
}

function is_ws(c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
	       c == '\f';
}

/* jinja2 `-` strips: all whitespace (incl. newlines) */
function rstrip_ws(s) {
	let n = length(s);

	while (n > 0 && is_ws(substr(s, n - 1, 1)))
		n--;

	return substr(s, 0, n);
}

function lstrip_ws(s) {
	let n = 0;

	while (n < length(s) && is_ws(substr(s, n, 1)))
		n++;

	return substr(s, n);
}

/* jinja2 lstrip_blocks: strip the trailing run of spaces/tabs ONLY when
 * it reaches the start of the line (preceded by \n or the start of the
 * source); the newline itself stays. */
function lstrip_line_ws(s) {
	let n = length(s);

	while (n > 0 && (substr(s, n - 1, 1) == ' ' || substr(s, n - 1, 1) == '\t'))
		n--;

	if (n > 0 && substr(s, n - 1, 1) != '\n')
		return s;

	return substr(s, 0, n);
}

/* ucode string literal from a value */
function ustr(v) {
	let s = '', i, c;

	for (i = 0; i < length(v); i++) {
		c = substr(v, i, 1);

		if (c == '"')
			s += '\\"';
		else if (c == '\\')
			s += '\\\\';
		else if (c == '\n')
			s += '\\n';
		else if (c == '\t')
			s += '\\t';
		else if (c == '\r')
			s += '\\r';
		else if (c == '\f')
			s += '\\f';
		else if (c == '\v')
			s += '\\v';
		else
			s += c;
	}

	return '"' + s + '"';
}

function hexval(c) {
	if (is_digit(c))
		return int(c);

	return lc(c) - 'a' + 10;
}

/* Jinja string literal at i (pointing at the opening quote).
 * Returns {s: value, end: offset after the closing quote}. */
function parse_jinja_string(s, i) {
	let q = substr(s, i, 1);
	let j = i + 1, n = length(s), out = '', c, e, hex, v, k;

	while (j < n) {
		c = substr(s, j, 1);

		if (c == q)
			return {s: out, end: j + 1};

		if (c == '\\') {
			e = substr(s, j + 1, 1);

			if (e == 'n') {out += '\n'; j += 2; continue;}
			if (e == 't') {out += '\t'; j += 2; continue;}
			if (e == 'r') {out += '\r'; j += 2; continue;}
			if (e == 'f') {out += '\f'; j += 2; continue;}
			if (e == 'v') {out += '\v'; j += 2; continue;}
			if (e == 'b') {out += '\b'; j += 2; continue;}
			if (e == q || e == '\\') {out += e; j += 2; continue;}
			if (e == '0')
				die('jinja2utpl: unsupported \\0 escape in string');

			if (e == 'u') {
				hex = substr(s, j + 2, 4);
				v = 0;

				for (k = 0; k < 4; k++) {
					let hc = substr(hex, k, 1);
					let hv = (hc >= '0' && hc <= '9') || (hc >= 'a' && hc <= 'f') ||
					          (hc >= 'A' && hc <= 'F') ? hexval(hc) : -1;

					if (hv < 0)
						die('jinja2utpl: bad \\u escape at ' + (j + 2));

					v = v * 16 + hv;
				}

				out += chr(v);
				j += 6;
				continue;
			}

			if (e == 'x') {
				hex = substr(s, j + 2, 2);
				v = hexval(substr(hex, 0, 1)) * 16 + hexval(substr(hex, 1, 1));

				if (v < 32 || v > 126)
					die('jinja2utpl: bad \\x escape at ' + (j + 2));

				out += chr(v);
				j += 4;
				continue;
			}

			die('jinja2utpl: unsupported escape \\' + e + ' in string');
		}

		out += c;
		j++;
	}

	die('jinja2utpl: unterminated string literal');
}

/* ------------------------------------------------------------------ */
/* Jinja lexer                                                        */
/* ------------------------------------------------------------------ */

/* Find the end of a block starting at p. Returns the offset of the
 * closing sequence. String-aware: '}}' inside a quoted string does not
 * end a block. Scans byte-wise: the structural characters (quotes,
 * backslash, '}' '%' '#' '-') are ASCII and never occur inside a
 * multi-byte UTF-8 sequence. */
function find_block_end(src, p, kind) {
	/* closing sequences: '}}' (125 125), '%}' (37 125), '#}' (35 125) */
	let c0 = kind == 'expr' ? 125 : kind == 'stmt' ? 37 : 35;
	let n = length(src), i = p, q = 0;

	while (i < n) {
		let c = ord(src, i);

		if (q != 0) {
			/* inside a '...' or "..." literal: a backslash escapes the
			 * next character (\n, \' , \\, ...), a matching quote closes
			 * the literal (and is consumed) */
			if (c == 92 && i + 1 < n) {
				i += 2;
				continue;
			}

			i++;

			if (c == q)
				q = 0;

			continue;
		}

		if (c == 39 || c == 34) {
			q = c;
			i++;
			continue;
		}

		if (c == c0 && i + 1 < n && ord(src, i + 1) == 125)
			return i;

		i++;
	}

	die('jinja2utpl: unterminated ' + kind + ' block');
}

/* Tokenize the Jinja source. Tokens:
 *   {k: 'text',    s}
 *   {k: 'expr',    s, dl, dr}   {{...}}  (dl: {{-  dr: -}})
 *   {k: 'stmt',    s, dl, dr}   {% ... %}
 *   {k: 'comment', s, dl, dr}   {# ... #}
 * jinja2 keep_trailing_newline=False: one trailing \n is dropped. */
export function jinja_lex(src) {
	if (substr(src, -1) == '\n')
		src = substr(src, 0, length(src) - 1);

	let toks = [];
	let i = 0, n = length(src);

	while (i < n) {
		let j = i;

		while (j < n) {
			let two = substr(src, j, 2);

			if (two == '{{' || two == '{%' || two == '{#')
				break;

			j++;
		}

		if (j > i)
			push(toks, {k: 'text', s: substr(src, i, j - i)});

		if (j >= n)
			break;

		let open = substr(src, j, 2);
		let kind = open == '{{' ? 'expr' : open == '{%' ? 'stmt' : 'comment';
		let p = j + 2;
		let dl = substr(src, p, 1) == '-';

		if (dl)
			p++;

		let q = find_block_end(src, p, kind);
		let s = substr(src, p, q - p);

		/* the '-' of a -}}/-%} strip flag sits immediately before the
		 * closing tag (we are outside any string literal there: the
		 * string-aware end scan would not have stopped inside one) */
		let dr = substr(s, -1) == '-';

		if (dr)
			s = substr(s, 0, length(s) - 1);

		push(toks, {k: kind, s: s, dl: dl, dr: dr});
		i = q + 2;
	}

	return toks;
}

/* ------------------------------------------------------------------ */
/* expression tokenizer                                               */
/* ------------------------------------------------------------------ */

/* Tokenize a Jinja expression. Tokens: {k: 'str'|'num'|'name'|'op', v} */
export function jinja_toks(s) {
	let toks = [];
	let i = 0, n = length(s), c, j, two;

	while (i < n) {
		c = substr(s, i, 1);

		if (is_ws(c)) {
			i++;
			continue;
		}

		if (c == "'" || c == '"') {
			let v = parse_jinja_string(s, i);

			push(toks, {k: 'str', v: v.s});
			i = v.end;
			continue;
		}

		if (is_digit(c)) {
			j = i + 1;

			while (j < n && (is_digit(substr(s, j, 1)) || substr(s, j, 1) == '.'))
				j++;

			// exponent
			if (j < n && (substr(s, j, 1) == 'e' || substr(s, j, 1) == 'E')) {
				let j2 = j + 1;

				if (j2 < n && (substr(s, j2, 1) == '+' || substr(s, j2, 1) == '-'))
					j2++;

				if (j2 < n && is_digit(substr(s, j2, 1))) {
					j = j2;

					while (j < n && is_digit(substr(s, j, 1)))
						j++;
				}
			}

			push(toks, {k: 'num', v: substr(s, i, j - i)});
			i = j;
			continue;
		}

		if (c == '.' && i + 1 < n && is_digit(substr(s, i + 1, 1))) {
			/* .5 is a float literal, but after a name / ) / ] it is a
			 * digit attribute access (foo.0) */
			let last = length(toks) > 0 ? toks[length(toks) - 1] : null;

			if (last != null &&
			    (last.k == 'name' ||
			     (last.k == 'op' && (last.v == ')' || last.v == ']')))) {
				push(toks, {k: 'op', v: '.'});
				j = i + 1;

				while (j < n && is_digit(substr(s, j, 1)))
					j++;

				push(toks, {k: 'num', v: substr(s, i + 1, j - i - 1)});
				i = j;
				continue;
			}

			j = i + 1;

			while (j < n && (is_digit(substr(s, j, 1)) || substr(s, j, 1) == '.'))
				j++;

			push(toks, {k: 'num', v: substr(s, i, j - i)});
			i = j;
			continue;
		}

		if (is_ident_start(c)) {
			j = i + 1;

			while (j < n && is_ident_cont(substr(s, j, 1)))
				j++;

			push(toks, {k: 'name', v: substr(s, i, j - i)});
			i = j;
			continue;
		}

		two = substr(s, i, 2);

		if (two == '==' || two == '!=' || two == '<=' || two == '>=' ||
		    two == '//' || two == '**') {
			push(toks, {k: 'op', v: two});
			i += 2;
			continue;
		}

		if (index('+-*/%<>|~()[]{}:,.=', c) >= 0) {
			push(toks, {k: 'op', v: c});
			i++;
			continue;
		}

		die('jinja2utpl: unexpected character ' + ustr(c) +
		    ' in expression: ' + ustr(s));
	}

	return toks;
}

/* ------------------------------------------------------------------ */
/* expression parser                                                  */
/* ------------------------------------------------------------------ */

/* Parse a whole expression. Returns the AST node. */
export function jinja_expr(s) {
	let toks = jinja_toks(s);
	let pos = 0;

	/* The parser functions are mutually recursive (the ternary's else
	 * branch, slice bounds and call args all re-enter parse_ternary)
	 * and ucode nested function declarations value-capture sibling
	 * bindings at their own declaration point (later siblings are null),
	 * so every parser lives on one holder object and crosses through it. */
	let P = {};

	P.peek = function() {
		return pos < length(toks) ? toks[pos] : null;
	};

	P.next = function() {
		return toks[pos++];
	};

	P.expect_op = function(op) {
		let t = P.peek();

		if (t == null || t.k != 'op' || t.v != op)
			die('jinja2utpl: expected "' + op + '" in expression: ' + ustr(s));

		return P.next();
	};

	P.expect_name = function(name) {
		let t = P.peek();

		if (t == null || t.k != 'name' || t.v != name)
			die('jinja2utpl: expected "' + name + '" in expression: ' + ustr(s));

		return P.next();
	};

	P.opt_name = function(name) {
		let t = P.peek();

		if (t != null && t.k == 'name' && t.v == name) {
			pos++;
			return true;
		}

		return false;
	};

	/* call args after '(': [ {k: name|null, v: expr} ] */
	P.parse_args = function() {
		P.expect_op('(');
		let args = [];

		if (P.peek() != null && P.peek().v == ')') {
			P.next();
			return args;
		}

		while (true) {
			let kw = null;
			let t = P.peek();

			if (t.k == 'name' && pos + 1 < length(toks) &&
			    toks[pos + 1].k == 'op' && toks[pos + 1].v == '=') {
				kw = t.v;
				pos += 2;
			}

			push(args, {k: kw, v: P.parse_ternary()});

			t = P.peek();

			if (t == null)
				die('jinja2utpl: unterminated argument list: ' + ustr(s));

			if (t.k == 'op' && t.v == ',') {
				P.next();

				if (P.peek() != null && P.peek().v == ')') {
					P.next();
					break;
				}

				continue;
			}

			P.expect_op(')');
			break;
		}

		return args;
	};

	P.parse_ternary = function() {
		let a = P.parse_or();
		let t = P.peek();

		if (t != null && t.k == 'name' && t.v == 'if') {
			P.next();
			let cond = P.parse_or();
			P.expect_name('else');
			let b = P.parse_ternary();

			return {t: 'ternary', a: a, cond: cond, b: b};
		}

		return a;
	};

	P.parse_or = function() {
		let e = P.parse_and();

		while (P.peek() != null && P.peek().k == 'name' && P.peek().v == 'or') {
			P.next();
			e = {t: 'bin', op: 'or', l: e, r: P.parse_and()};
		}

		return e;
	};

	P.parse_and = function() {
		let e = P.parse_not();

		while (P.peek() != null && P.peek().k == 'name' && P.peek().v == 'and') {
			P.next();
			e = {t: 'bin', op: 'and', l: e, r: P.parse_not()};
		}

		return e;
	};

	P.parse_not = function() {
		let t = P.peek();

		if (t != null && t.k == 'name' && t.v == 'not') {
			P.next();
			return {t: 'not', e: P.parse_not()};
		}

		return P.parse_cmp();
	};

	P.parse_cmp = function() {
		let e = P.parse_add();
		let t = P.peek();

		if (t == null)
			return e;

		// ordinary comparison operators
		if (t.k == 'op' && (t.v == '==' || t.v == '!=' || t.v == '<' ||
		    t.v == '>' || t.v == '<=' || t.v == '>=')) {
			P.next();
			return {t: 'bin', op: t.v, l: e, r: P.parse_add()};
		}

		// membership: x in y / x not in y
		if (t.k == 'name' && t.v == 'in') {
			P.next();
			return {t: 'in', l: e, r: P.parse_add()};
		}

		if (t.k == 'name' && t.v == 'not' &&
		    pos + 1 < length(toks) && toks[pos + 1].k == 'name' &&
		    toks[pos + 1].v == 'in') {
			pos += 2;
			return {t: 'notin', l: e, r: P.parse_add()};
		}

		// tests: x is [not] test
		if (t.k == 'name' && t.v == 'is') {
			pos++;
			let neg = P.opt_name('not');
			let tt = P.peek();

			if (tt == null || tt.k != 'name')
				die('jinja2utpl: expected test name after is: ' + ustr(s));

			pos++;
			return {t: 'is', e: e, test: tt.v, neg: neg};
		}

		return e;
	};

	P.parse_add = function() {
		let e = P.parse_mul();

		while (true) {
			let t = P.peek();

			if (t == null || t.k != 'op' || (t.v != '+' && t.v != '-' &&
			    t.v != '~'))
				break;

			P.next();
			e = {t: 'bin', op: t.v, l: e, r: P.parse_mul()};
		}

		return e;
	};

	P.parse_mul = function() {
		let e = P.parse_unary();

		while (true) {
			let t = P.peek();

			if (t == null || t.k != 'op' || (t.v != '*' && t.v != '/' &&
			    t.v != '//' && t.v != '%'))
				break;

			P.next();
			e = {t: 'bin', op: t.v, l: e, r: P.parse_unary()};
		}

		return e;
	};

	P.parse_unary = function() {
		let t = P.peek();

		if (t != null && t.k == 'op' && (t.v == '-' || t.v == '+')) {
			P.next();
			return {t: 'neg', e: P.parse_unary(), pos: t.v == '+'};
		}

		return P.parse_power();
	};

	P.parse_power = function() {
		let e = P.parse_postfix();
		let t = P.peek();

		if (t != null && t.k == 'op' && t.v == '**') {
			P.next();
			return {t: 'pow', l: e, r: P.parse_power()};
		}

		return e;
	};

	P.parse_postfix = function() {
		let e = P.parse_primary();

		while (true) {
			let t = P.peek();

			if (t == null || t.k != 'op')
				break;

			if (t.v == '|') {
				P.next();
				let name = P.next();

				if (name == null || name.k != 'name')
					die('jinja2utpl: expected filter name after | : ' +
					    ustr(s));

				let args = [];

				if (P.peek() != null && P.peek().v == '(')
					args = P.parse_args();

				e = {t: 'filter', e: e, name: name.v, args: args};
				continue;
			}

			if (t.v == '[') {
				P.next();
				e = P.parse_index(e);
				continue;
			}

			if (t.v == '.') {
				P.next();
				let name = P.next();

				/* digit attributes are jinja's numeric getitem
				 * (foo.0); emitted as ?. [0] */
				if (name == null ||
				    (name.k != 'name' && name.k != 'num'))
					die('jinja2utpl: expected attribute name after . : ' +
					    ustr(s));

				e = {t: 'attr', e: e, n: '' + name.v};
				continue;
			}

			if (t.v == '(') {
				e = {t: 'call', e: e, args: P.parse_args()};
				continue;
			}

			break;
		}

		return e;
	};

	P.parse_index = function(base) {
		/* [ expr ] | [ -1 ] | [ ::-1 ] */
		let t = P.peek();

		if (t == null || t.k != 'op' || t.v != ':') {
			let e = P.parse_ternary();
			P.expect_op(']');

			return {t: 'index', e: base, idx: e};
		}

		// slice: only ::-1 is supported
		P.next();

		if (P.peek() == null || P.peek().v != ':')
			die('jinja2utpl: unsupported slice (only ::-1): ' + ustr(s));

		P.next();

		/* the step is '-1': tokenizer yields op '-' + num '1' */
		if (P.peek() == null || P.peek().k != 'op' || P.peek().v != '-')
			die('jinja2utpl: unsupported slice (only ::-1): ' + ustr(s));

		P.next();

		if (P.peek() == null || P.peek().k != 'num' || P.peek().v != '1')
			die('jinja2utpl: unsupported slice (only ::-1): ' + ustr(s));

		P.next();
		P.expect_op(']');

		return {t: 'rev', e: base};
	};
	P.parse_primary = function() {
		let t = P.next();

		if (t == null)
			die('jinja2utpl: unexpected end of expression: ' + ustr(s));

		if (t.k == 'num') {
			// keep raw text; int vs double decided by transform
			return {t: 'num', raw: t.v};
		}

		if (t.k == 'str')
			return {t: 'str', s: t.v};

		if (t.k == 'name') {
			switch (t.v) {
			case 'true':
			case 'True':
				return {t: 'bool', v: true};
			case 'false':
			case 'False':
				return {t: 'bool', v: false};
			case 'none':
			case 'None':
			case 'null':
				return {t: 'null'};
			}

			return {t: 'name', n: t.v};
		}

		if (t.k == 'op') {
			if (t.v == '(') {
				// tuple or parenthesized expression
				let first = P.parse_ternary();

				if (P.peek() != null && P.peek().v == ',') {
					let items = [ first ];

					while (true) {
						P.next();

						if (P.peek() != null && P.peek().v == ')') {
							P.next();
							break;
						}

						push(items, P.parse_ternary());

						let t2 = P.peek();

						if (t2 == null || t2.v != ',')
							break;
					}

					P.expect_op(')');
					return {t: 'list', items: items};
				}

				P.expect_op(')');
				return first;
			}

			if (t.v == '[') {
				let items = [];

				if (P.peek() != null && P.peek().v == ']') {
					P.next();
					return {t: 'list', items: items};
				}

				while (true) {
					push(items, P.parse_ternary());
					let t2 = P.peek();

					if (t2 == null || t2.v != ',')
						break;

					P.next();

					if (P.peek() != null && P.peek().v == ']') {
						P.next();
						break;
					}
				}

				P.expect_op(']');
				return {t: 'list', items: items};
			}

			if (t.v == '{') {
				let items = [];

				if (P.peek() != null && P.peek().v == '}') {
					P.next();
					return {t: 'dict', items: items};
				}

				while (true) {
					let kt = P.next();
					let key;

					if (kt.k == 'str')
						key = kt.v;
					else if (kt.k == 'name' || kt.k == 'num')
						key = kt.v;
					else
						die('jinja2utpl: expected dict key: ' + ustr(s));

					P.expect_op(':');
					push(items, [ key, P.parse_ternary() ]);
					let t2 = P.peek();

					if (t2 == null || t2.v != ',')
						break;

					P.next();

					if (P.peek() != null && P.peek().v == '}') {
						P.next();
						break;
					}
				}

				P.expect_op('}');
				return {t: 'dict', items: items};
			}
		}

		die('jinja2utpl: unexpected token in expression: ' + ustr(s));
	};

	let e = P.parse_ternary();

	if (pos != length(toks))
		die('jinja2utpl: trailing tokens in expression: ' + ustr(s));

	return e;
}

/* ------------------------------------------------------------------ */
/* statement + program parser                                         */
/* ------------------------------------------------------------------ */

function read_ident(s, i) {
	let n = length(s);

	while (i < n && is_ws(substr(s, i, 1)))
		i++;

	if (i >= n || !is_ident_start(substr(s, i, 1)))
		die('jinja2utpl: expected an identifier at: ' + ustr(s));

	let j = i + 1;

	while (j < n && is_ident_cont(substr(s, j, 1)))
		j++;

	return {name: substr(s, i, j - i), end: j};
}

function skip_ws(s, i) {
	let n = length(s);

	while (i < n && is_ws(substr(s, i, 1)))
		i++;

	return i;
}

function expect_char(s, i, c, what) {
	if (substr(s, i, 1) != c)
		die('jinja2utpl: expected "' + c + '" in ' + what + ': ' + ustr(s));

	return i + 1;
}

function expect_word(s, i, word, what) {
	i = skip_ws(s, i);
	let j = i;

	while (j < length(s) && is_ident_cont(substr(s, j, 1)))
		j++;

	if (substr(s, i, j - i) != word)
		die('jinja2utpl: expected "' + word + '" in ' + what + ': ' + ustr(s));

	return j;
}

/* Everything after the leading keyword of a statement. */
function rest_after(s, word) {
	let i = skip_ws(s, 0);
	let j = i;

	while (j < length(s) && is_ident_cont(substr(s, j, 1)))
		j++;

	if (substr(s, i, j - i) != word)
		die('jinja2utpl: internal word mismatch: ' + ustr(s));

	return substr(s, j);
}

/* for TARGETS in EXPR — targets: one or two names */
function parse_for_head(s) {
	let rest = rest_after(s, 'for');
	let i = skip_ws(rest, 0);
	let targets = [];

	if (substr(rest, i, 1) == '(') {
		i++;
		let a = read_ident(rest, i);

		push(targets, a.name);
		i = expect_char(rest, a.end, ',', 'for target list');
		i = skip_ws(rest, i);
		let b = read_ident(rest, i);

		push(targets, b.name);
		i = expect_char(rest, b.end, ')', 'for target list');
		i = skip_ws(rest, i);
	}
	else {
		let a = read_ident(rest, i);

		push(targets, a.name);
		i = skip_ws(rest, a.end);

		/* bare two-target form: for a, b in seq */
		if (substr(rest, i, 1) == ',') {
			let b = read_ident(rest, i + 1);

			push(targets, b.name);
			i = skip_ws(rest, b.end);
		}
	}

	i = expect_word(rest, i, 'in', 'for statement');
	i = skip_ws(rest, i);

	return {targets: targets, seq: jinja_expr(substr(rest, i))};
}

function parse_set_head(s) {
	let rest = rest_after(s, 'set');
	let i = skip_ws(rest, 0);
	let a = read_ident(rest, i);
	let name = a.name;
	i = a.end;

	/* dotted target: set image_count.value = ..., set ns.multi_step_tool = ... */
	while (substr(rest, i, 1) == '.') {
		i++;
		let b = read_ident(rest, i);

		name += '.' + b.name;
		i = b.end;
	}

	i = skip_ws(rest, i);

	if (substr(rest, i, 1) == ',')
		die('jinja2utpl: multi-target set (a, b = ...) is not supported');

	let j = expect_char(rest, i, '=', 'set statement');
	j = skip_ws(rest, j);

	return {name: name, e: jinja_expr(substr(rest, j))};
}

/* macro NAME (p1, p2 = expr, ...) */
function parse_macro_head(s) {
	let rest = rest_after(s, 'macro');
	let i = skip_ws(rest, 0);
	let a = read_ident(rest, i);
	let j = expect_char(rest, a.end, '(', 'macro header');
	j = skip_ws(rest, j);
	let n = length(rest);
	let params = [];

	if (substr(rest, j, 1) != ')') {
		while (true) {
			j = skip_ws(rest, j);
			let p = read_ident(rest, j);
			j = skip_ws(rest, p.end);
			let def = null;

			if (substr(rest, j, 1) == '=') {
				// default: up to the next top-level ',' or ')'
				let depth = 0, k = j + 1, str = null;

				while (k < n) {
					let c = substr(rest, k, 1);

					if (str != null) {
						if (c == '\\')
							k++;
						else if (c == str)
							str = null;
					}
					else if (c == "'" || c == '"')
						str = c;
					else if (c == '(' || c == '[' || c == '{')
						depth++;
					else if (c == ')' || c == ']' || c == '}') {
						if (depth == 0)
							break;

						depth--;
					}
					else if (depth == 0 && c == ',')
						break;

					k++;
				}

				def = jinja_expr(substr(rest, j + 1, k - j - 1));
				j = k;
			}

			push(params, {name: p.name, def: def});
			j = skip_ws(rest, j);

			if (substr(rest, j, 1) == ',') {
				j++;
				continue;
			}

			j = expect_char(rest, j, ')', 'macro header');
			break;
		}
	}

	return {name: a.name, params: params};
}

/* Parse the whole template into a node list. */
export function jinja_parse(src) {
	let toks = jinja_lex(src);
	let pos = 0;

	function first_word(tk) {
		if (tk.k != 'stmt')
			return null;

		let w = jinja_toks(tk.s);

		return (length(w) > 0 && w[0].k == 'name') ? w[0].v : null;
	}

	function parse_program(until) {
		let nodes = [];
		let stop = null;

		while (true) {
			if (pos >= length(toks)) {
				if (until == null)
					break;

				die('jinja2utpl: unexpected end of template' +
				    (until ? ' (expected ' + join(' / ', until) + ')' : ''));
			}

			let tk = toks[pos];

			if (tk.k == 'text') {
				pos++;
				push(nodes, {t: 'text', s: tk.s});
				continue;
			}

			if (tk.k == 'comment') {
				pos++;
				continue;
			}

			if (tk.k == 'expr') {
				pos++;
				push(nodes, {t: 'out', dl: tk.dl, dr: tk.dr,
				           e: jinja_expr(tk.s)});
				continue;
			}

			let word = first_word(tk);

			if (word == null)
				die('jinja2utpl: statement must start with a keyword: ' +
				    ustr(tk.s));

			if (until != null && index(until, word) >= 0) {
				stop = pos;
				break;
			}

			if (word == 'endif' || word == 'endfor' || word == 'endwhile' ||
			    word == 'endmacro' || word == 'elif' || word == 'else')
				die('jinja2utpl: unexpected ' + word);

			if (word == 'if') {
				pos++;
				let cond = jinja_expr(rest_after(tk.s, 'if'));
				let r = parse_program(['elif', 'else', 'endif']);
				let branches = [];
				let st = r.stop;

				while (true) {
					if (first_word(toks[st]) == 'elif') {
						let er = rest_after(toks[st].s, 'elif');
						st++;
						pos = st;
						let b = parse_program(['elif', 'else', 'endif']);

						push(branches, {dl: toks[st - 1].dl, dr: toks[st - 1].dr,
						                  cond: jinja_expr(er), body: b.nodes,
						                  stop: b.stop});
						st = b.stop;
						continue;
					}

					if (first_word(toks[st]) == 'else') {
						st++;
						pos = st;
						let b = parse_program(['endif']);

						push(branches, {dl: toks[st - 1].dl, dr: toks[st - 1].dr,
						                  cond: null, body: b.nodes,
						                  stop: b.stop});
						st = b.stop;
						continue;
					}

					if (first_word(toks[st]) != 'endif')
						die('jinja2utpl: unterminated if block');

					push(nodes, {t: 'if',
					           open: {dl: tk.dl, dr: tk.dr}, cond: cond,
					           body: r.nodes,
					           branches: branches,
					           close: {dl: toks[st].dl, dr: toks[st].dr}});
					pos = st + 1;
					break;
				}
			}
			else if (word == 'for') {
				pos++;
				let m = parse_for_head(tk.s);
				let r = parse_program(['endfor']);

				push(nodes, {t: 'for',
				           open: {dl: tk.dl, dr: tk.dr}, targets: m.targets,
				           seq: m.seq, body: r.nodes,
				           close: {dl: toks[r.stop].dl, dr: toks[r.stop].dr}});
				pos = r.stop + 1;
			}
			else if (word == 'while') {
				pos++;
				let cond = jinja_expr(rest_after(tk.s, 'while'));
				let r = parse_program(['endwhile']);

				push(nodes, {t: 'while',
				           open: {dl: tk.dl, dr: tk.dr}, cond: cond,
				           body: r.nodes,
				           close: {dl: toks[r.stop].dl, dr: toks[r.stop].dr}});
				pos = r.stop + 1;
			}
			else if (word == 'macro') {
				pos++;
				let m = parse_macro_head(tk.s);
				let r = parse_program(['endmacro']);

				push(nodes, {t: 'macro',
				           open: {dl: tk.dl, dr: tk.dr}, name: m.name,
				           params: m.params, body: r.nodes,
				           close: {dl: toks[r.stop].dl, dr: toks[r.stop].dr}});
				pos = r.stop + 1;
			}
			else if (word == 'set') {
				pos++;
				let m = parse_set_head(tk.s);

				push(nodes, {t: 'set', dl: tk.dl, dr: tk.dr,
				           name: m.name, e: m.e});
			}
			else if (word == 'break' || word == 'continue') {
				pos++;
				push(nodes, {t: word, dl: tk.dl, dr: tk.dr});
			}
			else {
				// bare expression statement
				pos++;
				push(nodes, {t: 'exprstmt', dl: tk.dl, dr: tk.dr,
				           e: jinja_expr(tk.s)});
			}
		}

		return {nodes: nodes, stop: stop};
	}

	let top = parse_program(null);

	return top.nodes;
}

/* ------------------------------------------------------------------ */
/* names, filters, tests, methods                                     */
/* ------------------------------------------------------------------ */

/* ucode names the transpiled code may reference directly */
let AMP_NAMES = {
	die: 1, assert: 1, length: 1, type: 1, keys: 1, index: 1, substr: 1,
	slice: 1, join: 1, push: 1, trim: 1, ltrim: 1, rtrim: 1, lc: 1, uc: 1,
	sprintf: 1, printf: 1, print: 1, warn: 1, json: 1, int: 1, abs: 1,
	min: 1, max: 1, floor: 1, ceil: 1, round: 1, global: 1, exit: 1,
	getenv: 1, readfile: 1, stat: 1, loadstring: 1, loadfile: 1, include: 1,
	render: 1, chr: 1, struct: 1, random: 1, uloop: 1
};

/* helpers imported into the emitted program (ucode/utpl.uc) — bare
 * names after the import, so they are reserved for template
 * declarations just like the builtins */
let UTPL_HELPERS = {
	tpl_out: 1, tpl_out_untrusted: 1, tpl_range: 1, tpl_rev: 1,
	pyjson: 1, cap_begin: 1, cap_end: 1, cap_abort: 1
};

/* value ops imported into the emitted program (ucode/tv.uc) */
let TV_NAMES = {
	taint: 1, cat: 1, truthy: 1, get: 1, idx: 1, eq: 1, ne: 1,
	in_op: 1, kind: 1, plain: 1, capitalize: 1, replace: 1, trim: 1,
	lower: 1, upper: 1, tostr: 1, length_of: 1, items: 1, deflt: 1,
	join: 1, startswith: 1, endswith: 1, split: 1
};

let KEYWORDS = {
	if: 1, elif: 1, else: 1, for: 1, while: 1, function: 1, let: 1,
	return: 1, break: 1, continue: 1, true: 1, false: 1, null: 1, in: 1,
	of: 1, new: 1, delete: 1, import: 1, export: 1, switch: 1, case: 1,
	try: 1, catch: 1, finally: 1, throw: 1, this: 1, undefined: 1, void: 1,
	do: 1, with: 1, class: 1, const: 1, var: 1, async: 1, await: 1,
	yield: 1, static: 1, prototype: 1, default: 1, endfor: 1, endif: 1,
	endwhile: 1, endfunction: 1, endmacro: 1
};

function is_uc_name(n) {
	return index(keys(AMP_NAMES), n) >= 0 || index(keys(UTPL_HELPERS), n) >= 0
	       || index(keys(TV_NAMES), n) >= 0;
}

/* filters whose result carries the taint of their arguments */
let FILTERS = {
	tojson: 'ident', trim: 'ident', lower: 'ident', upper: 'ident',
	string: 'ident', safe: 'ident', default: 'ident', join: 'ident',
	length: 'clean', items: 'ident', capitalize: 'ident',
	replace: 'ident'
};

/* methods on strings (Jinja calls Python str methods) */
let METHODS = {
	startswith: 'clean', endswith: 'clean', strip: 'ident', trim: 'ident',
	upper: 'ident', lower: 'ident', items: 'ident', split: 'ident'
};

function check_filter(name) {
	if (index(keys(FILTERS), name) < 0)
		die('jinja2utpl: unsupported filter: | ' + name);
}

function check_test(name) {
	switch (name) {
	case 'string': case 'integer': case 'int': case 'float': case 'double':
	case 'number': case 'boolean': case 'bool': case 'mapping':
	case 'iterable': case 'none': case 'undefined': case 'defined':
	case 'true': case 'false':
		return;
	}

	die('jinja2utpl: unsupported test: is ' + name);
}

function check_reserved(name, what) {
	if (KEYWORDS[name] || is_uc_name(name) || name == '__ctx' ||
	    name == '__utpl_state' || name == 'loop')
		die('jinja2utpl: ' + what + ' "' + name + '" collides with a ' +
		    'reserved name');
}

/* ------------------------------------------------------------------ */
/* taint analysis                                                     */
/* ------------------------------------------------------------------ */

function new_scope(parent) {
	return {names: {}, lets: {}, parent: parent, loop_seq: null};
}

function is_declared(scope, name) {
	while (scope) {
		if (index(keys(scope.names), name) >= 0)
			return true;

		scope = scope.parent;
	}

	return false;
}

function lookup_taint(scope, name) {
	while (scope) {
		if (index(keys(scope.names), name) >= 0)
			return scope.names[name];

		scope = scope.parent;
	}

	return 1;
}

/* Taint of an expression: 0 = trusted, 1 = tainted (context data).
 * call_sinks: when non-null, macro call sites are reported into it. */
function taint_expr(n, scope, macros, free, call_sinks) {
	switch (n.t) {
	case 'num':
	case 'str':
	case 'bool':
	case 'null':
		return 0;

	case 'name': {
		let name = n.n;

		if (is_declared(scope, name))
			return lookup_taint(scope, name);

		if (is_uc_name(name) || macros[name])
			return 0;

		/* free variable: a value from the render context */
		if (index(free, name) < 0)
			push(free, name);

		check_reserved(name, 'context variable');
		return 1;
	}

	case 'list': {
		let m = 0;

		for (let i = 0; i < length(n.items); i++)
			m = max(m, taint_expr(n.items[i], scope, macros, free,
			                       call_sinks));

		return m;
	}

	case 'dict': {
		let m = 0;

		for (let i = 0; i < length(n.items); i++)
			m = max(m, taint_expr(n.items[i][1], scope, macros, free,
			                       call_sinks));

		return m;
	}

	case 'not':
		return 0;

	case 'neg':
		return taint_expr(n.e, scope, macros, free, call_sinks);

	case 'pow':
		die('jinja2utpl: ** (exponent) is not supported');

	case 'bin':
		switch (n.op) {
		case '==':
		case '!=':
		case '<':
		case '<=':
		case '>':
		case '>=':
			taint_expr(n.l, scope, macros, free, call_sinks);
			taint_expr(n.r, scope, macros, free, call_sinks);
			return 0;
		}

		/* store the leaf taints: the emitter splits mixed-literal
		 * concatenation output per leaf and needs each leaf's taint */
		n.l._t = taint_expr(n.l, scope, macros, free, call_sinks);
		n.r._t = taint_expr(n.r, scope, macros, free, call_sinks);

		return max(n.l._t, n.r._t);

	case 'in':
	case 'notin':
		taint_expr(n.l, scope, macros, free, call_sinks);
		taint_expr(n.r, scope, macros, free, call_sinks);
		return 0;

	case 'is':
		check_test(n.test);
		taint_expr(n.e, scope, macros, free, call_sinks);
		return 0;

	case 'ternary':
		taint_expr(n.cond, scope, macros, free, call_sinks);

		return max(taint_expr(n.a, scope, macros, free, call_sinks),
		           taint_expr(n.b, scope, macros, free, call_sinks));

	case 'filter': {
		check_filter(n.name);
		let t = taint_expr(n.e, scope, macros, free, call_sinks);

		for (let i = 0; i < length(n.args); i++)
			t = max(t, taint_expr(n.args[i].v, scope, macros, free,
			                       call_sinks));

		return FILTERS[n.name] == 'clean' ? 0 : t;
	}

	case 'index':
		taint_expr(n.idx, scope, macros, free, call_sinks);
		return taint_expr(n.e, scope, macros, free, call_sinks);

	case 'rev':
		return taint_expr(n.e, scope, macros, free, call_sinks);

	case 'attr': {
		if (n.e.t == 'name' && n.e.n == 'loop' &&
		    (n.n == 'previtem' || n.n == 'nextitem')) {
			let s = scope;

			while (s && s.loop_seq == null)
				s = s.parent;

			if (s)
				return s.loop_seq;
		}

		return taint_expr(n.e, scope, macros, free, call_sinks);
	}

	case 'call': {
		let t = 0;

		for (let i = 0; i < length(n.args); i++) {
			let a = n.args[i];

			if (a.k != null && !(n.e.t == 'name' && n.e.n == 'namespace'))
				die('jinja2utpl: keyword argument ' + a.k + '= is only ' +
				    'supported in namespace()');

			t = max(t, taint_expr(a.v, scope, macros, free, call_sinks));
		}

		if (n.e.t != 'name') {
			/* method call or call of an expression */
			if (n.e.t == 'attr' && index(keys(METHODS), n.e.n) >= 0) {
				taint_expr(n.e.e, scope, macros, free, call_sinks);
				return METHODS[n.e.n] == 'clean' ? 0 : t;
			}

			taint_expr(n.e, scope, macros, free, call_sinks);
			return t;
		}

		let name = n.e.n;

		if (name == 'namespace')
			return 0;

		if (name == 'raise_exception')
			return 0;

		if (macros[name]) {
			if (call_sinks)
				push(call_sinks, {name: name, args: n.args, scope: scope});

			return t;
		}

		if (is_uc_name(name))
			return (name == 'pyjson' || name == 'tostr' ||
			        name == 'capitalize' || name == 'lower' ||
			        name == 'upper' || name == 'replace' || name == 'cat' ||
			        name == 'deflt' || name == 'trim' || name == 'lc' ||
			        name == 'uc' || name == 'tpl_rev' || name == 'items' ||
			        name == 'join') ? t : 0;

		/* free variable used as a function */
		if (!is_declared(scope, name) && !macros[name]) {
			if (index(free, name) < 0)
				push(free, name);

			check_reserved(name, 'context variable');
		}

		return t;
	}
	}

	die('jinja2utpl: internal: unknown expression node ' + n.t);
}

/* Pass A: build the scope tree and the macro registry. */
function collect_scopes(nodes, scope, macros) {
	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		switch (n.t) {
		case 'if':
			collect_scopes(n.body, scope, macros);

			for (let b = 0; b < length(n.branches); b++)
				collect_scopes(n.branches[b].body, scope, macros);

			break;

		case 'while':
			collect_scopes(n.body, scope, macros);
			break;

		case 'for': {
			let sc = new_scope(scope);

			n._scope = sc;
			sc.kind = 'for';

			for (let t = 0; t < length(n.targets); t++)
				sc.names[n.targets[t]] = 0;

			sc.names['loop'] = 0;
			collect_scopes(n.body, sc, macros);
			break;
		}

		case 'macro': {
			let sc = new_scope(scope);

			n._scope = sc;
			sc.kind = 'macro';

			for (let p2 = sc.parent; p2; p2 = p2.parent)
				if (p2.kind == 'macro')
					die('jinja2utpl: nested macros are not supported '
					    + '(' + n.name + ')');

			/* the scope tree is rebuilt per fixed-point iteration: the
			 * same macro node is re-registered, which is not a duplicate */
			if (macros[n.name] && macros[n.name] != n)
				die('jinja2utpl: duplicate macro ' + n.name);

			macros[n.name] = n;

			for (let p = 0; p < length(n.params); p++)
				sc.names[n.params[p].name] = 0;

			collect_scopes(n.body, sc, macros);
			break;
		}
		}
	}
}

/* Single deterministic pass after the taint fixed point: compute
 * n.redecl for every simple-name set and mark fresh lets. */
function mark_redecl(nodes, scope, free) {
	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		switch (n.t) {
		case 'set': {
			/* every simple-name set target is pre-declared (the
			 * per-render preamble, the macro's function scope, or a for
			 * target), so a set is always a bare assignment into the
			 * existing binding — jinja's set-leaking semantics */
			if (index(n.name, '.') < 0)
				n.redecl = true;
			break;
		}

		case 'if':
			mark_redecl(n.body, scope, free);

			for (let b = 0; b < length(n.branches); b++)
				mark_redecl(n.branches[b].body, scope, free);
			break;

		case 'while':
			mark_redecl(n.body, scope, free);
			break;

		case 'for': {
			let sc = n._scope;

			for (let t = 0; t < length(n.targets); t++)
				sc.lets[n.targets[t]] = true;

			sc.lets['loop'] = true;
			mark_redecl(n.body, sc, free);
			break;
		}

		case 'macro': {
			let sc = n._scope;

			for (let p = 0; p < length(n.params); p++)
				sc.lets[n.params[p].name] = true;

			mark_redecl(n.body, sc, free);
			break;
		}
		}
	}
}

/* Pass B: compute taints. Macro parameter taints are a fixed point
 * over the call sites (a parameter is tainted if any call passes
 * tainted data). */
function analyze(nodes) {
	let macros = {};
	let free = [];
	let mod = new_scope(null);

	collect_scopes(nodes, mod, macros);

	/* name hygiene */
	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		if (n.t == 'macro') {
			check_reserved(n.name, 'macro');

			for (let p = 0; p < length(n.params); p++)
				check_reserved(n.params[p].name, 'macro parameter');
		}
	}

	function walk(nodes, scope) {
		for (let i = 0; i < length(nodes); i++) {
			let n = nodes[i];

			switch (n.t) {
			case 'out':
				n._t = taint_expr(n.e, scope, macros, free, null);
				break;

			case 'set': {
				check_reserved(n.name, 'set variable');

				/* a dotted target reads its base object */
				let dot = index(n.name, '.');

				if (dot >= 0)
					taint_expr({t: 'name', n: substr(n.name, 0, dot)},
					           scope, macros, free, null);

				scope.names[n.name] = n._t =
					taint_expr(n.e, scope, macros, free, null);
				break;
			}

			case 'exprstmt':
				taint_expr(n.e, scope, macros, free, null);
				break;

			case 'if':
				taint_expr(n.cond, scope, macros, free, null);
				walk(n.body, scope);

				for (let b = 0; b < length(n.branches); b++) {
					let br = n.branches[b];

					if (br.cond)
						taint_expr(br.cond, scope, macros, free, null);

					walk(br.body, scope);
				}

				break;

			case 'while':
				taint_expr(n.cond, scope, macros, free, null);
				walk(n.body, scope);
				break;

			case 'for': {
				let st = taint_expr(n.seq, scope, macros, free, null);

				for (let t = 0; t < length(n.targets); t++) {
					check_reserved(n.targets[t], 'for target');
					n._scope.names[n.targets[t]] = st;
				}

				n._scope.loop_seq = st;
				walk(n.body, n._scope);
				break;
			}

			case 'macro':
				walk(n.body, n._scope);
				break;
			}
		}
	}

	function walk_calls(nodes, scope, sinks) {
		for (let i = 0; i < length(nodes); i++) {
			let n = nodes[i];

			switch (n.t) {
			case 'out':
			case 'exprstmt':
				taint_expr(n.e, scope, macros, free, sinks);
				break;

			case 'set':
				taint_expr(n.e, scope, macros, free, sinks);
				break;

			case 'if':
				taint_expr(n.cond, scope, macros, free, sinks);
				walk_calls(n.body, scope, sinks);

				for (let b = 0; b < length(n.branches); b++) {
					let br = n.branches[b];

					if (br.cond)
						taint_expr(br.cond, scope, macros, free, sinks);

					walk_calls(br.body, scope, sinks);
				}

				break;

			case 'while':
				taint_expr(n.cond, scope, macros, free, sinks);
				walk_calls(n.body, scope, sinks);
				break;

			case 'for':
				taint_expr(n.seq, scope, macros, free, sinks);
				walk_calls(n.body, n._scope, sinks);
				break;

			case 'macro':
				walk_calls(n.body, n._scope, sinks);
				break;
			}
		}
	}

	/* fixed point over macro parameters: a parameter is tainted if any
	 * call site passes tainted data */
	let prev = null, stable = false;

	function pt_init(pt) {
		for (let k in macros) {
			let m = macros[k];

			for (let p = 0; p < length(m.params); p++) {
				let key = k + '/' + m.params[p].name;

				pt[key] = m.params[p].def
					? taint_expr(m.params[p].def, m._scope, macros, free,
					             null)
					: 0;
			}
		}
	}

	function pt_apply(pt) {
		for (let k in macros) {
			let m = macros[k];

			for (let p = 0; p < length(m.params); p++)
				m._scope.names[m.params[p].name] =
					pt[k + '/' + m.params[p].name];
		}
	}

	function equal(a, b) {
		let ks = keys(a);

		if (length(ks) != length(keys(b)))
			return false;

		for (let i = 0; i < length(ks); i++)
			if (a[ks[i]] != b[ks[i]])
				return false;

		return true;
	}

	for (let iter = 0; iter < 16 && !stable; iter++) {
		let cur = {};

		/* a fresh scope tree per pass: `set` bindings are registered in
		 * source order during the walk (Jinja2's `set` binds from that
		 * point on), so stale registrations from the previous pass must
		 * not leak into this one */
		mod = new_scope(null);
		collect_scopes(nodes, mod, macros);

		if (prev == null) {
			pt_init(cur);
		}
		else {
			/* taint only ever grows: refine the previous estimate */
			for (let k in prev)
				cur[k] = prev[k];
		}

		pt_apply(cur);
		walk(nodes, mod);

		let sinks = [];

		walk_calls(nodes, mod, sinks);

		for (let i = 0; i < length(sinks); i++) {
			let s = sinks[i];
			let m = macros[s.name];
			let pi = 0;

			for (let a = 0; a < length(s.args); a++) {
				if (s.args[a].k != null)
					continue;

				if (pi >= length(m.params))
					break;

				let key = s.name + '/' + m.params[pi].name;

				cur[key] = max(cur[key],
				               taint_expr(s.args[a].v, s.scope, macros,
				                           free, null));
				pi++;
			}
		}

		stable = prev != null && equal(prev, cur);
		prev = cur;
	}

	if (!stable)
		die('jinja2utpl: macro parameter taint did not converge');

	for (let i = 0; i < length(free); i++)
		check_reserved(free[i], 'context variable');

	mark_redecl(nodes, mod, free);

	return {macros: macros, free: free};
}

/* ------------------------------------------------------------------ */
/* emitter                                                            */
/* ------------------------------------------------------------------ */

let RANK = {
	or: 2, and: 3,
	'==': 5, '!=': 5, '<': 5, '<=': 5, '>': 5, '>=': 5,
	'+': 7, '-': 7, '~': 7,
	'*': 8, '/': 8, '//': 8, '%': 8
};

function is_primary(n) {
	switch (n.t) {
	case 'num':
	case 'str':
	case 'bool':
	case 'null':
	case 'name':
	case 'list':
	case 'dict':
	case 'index':
	case 'rev':
	case 'attr':
	case 'call':
		return true;
	}

	return false;
}

let EM = {};   /* emit functions: mutual recursion (no hoisting) */

/* emit state + inlining bookkeeping (set by EM.emit_nodes).
 * LETS, not function declarations: ucode captures function
 * declarations by VALUE at the defining point (a later
 * declaration is still null there), but let cells are live
 * - every closure sees later assignments. The EM object
 * methods cross-reference each other and these through the
 * cells. */
let EM_MACROS = null;   /* name -> macro node (set by EM.emit_nodes) */
let EM_ST = null;       /* the emit state (set by EM.emit_nodes) */
let EM_STACK = [];      /* inlining cycle guard (macro nodes) */

/* Emit a block with the jinja2 whitespace rules applied.
 * synth blocks are transpiler insertions: source-tag strip flags are
 * still honored when carried, but the trim_blocks/lstrip_blocks
 * defaults never apply to them. */

/* Record a helper symbol for the import lines; the emitted program
 * references every helper by BARE name (an import is a stack slot in
 * the VM, faster than a namespace property lookup). Declared before
 * every emitter that calls it (no hoisting, L47). */
function use_tv(n) {
	if (EM_ST != null)
		EM_ST.tv_used[n] = 1;

	return n;
}

function use_utpl(n) {
	if (EM_ST != null)
		EM_ST.utpl_used[n] = 1;

	return n;
}

/* Emit the buffered verbatim text as an explicit output call and clear
 * the buffer. Declared before emit_block (no function hoisting, L47/
 * L64). */
function flush_text(st) {
	if (st.pend_text == '')
		return;

	let s = st.pend_text;

	st.pend_text = '';

	/* every verbatim byte is an explicit output; template text is
	 * TRUSTED by construction (the transpiler emits it, the user never
	 * writes it) */
	use_utpl('tpl_out');
	st.out.put('*', "tpl_out(" + ustr(s) + ");\n");
}

function emit_block(s, dl, dr, is_stmt, synth, st) {
	/* a jinja block tag is a plain ucode statement now.  A left-dash /
	 * lstrip_blocks strip applies to the verbatim text that precedes the
	 * tag, which is still buffered in st.pend_text (not yet emitted), so
	 * we can rstrip the content directly. */
	if (dl && st.pend_text != '')
		st.pend_text = rstrip_ws(st.pend_text);
	else if (is_stmt && !synth && st.opts.lstrip && st.pend_text != '')
		st.pend_text = lstrip_line_ws(st.pend_text);

	flush_text(st);
	st.out.put('*', s + '\n');

	if (dr)
		st.pend = 'all';
	else if (is_stmt && !synth && st.opts.trim)
		st.pend = 'nl';
}

function emit_text(s, st) {
	if (st.pend == 'all')
		s = lstrip_ws(s);
	else if (st.pend == 'nl' && substr(s, 0, 1) == '\n')
		s = substr(s, 1);

	st.pend = null;

	if (s == '')
		return;

	st.pend_text += s;
}


function is_ident(s) {
	for (let i = 0; i < length(s); i++)
		if (!is_ident_cont(substr(s, i, 1)))
			return false;

	return length(s) > 0;
}

EM.emit_args = function(args) {
	let parts = [];

	for (let i = 0; i < length(args); i++) {
		if (args[i].k != null)
			die('jinja2utpl: keyword argument ' + args[i].k + '= is only ' +
			    'supported in namespace()');

		push(parts, EM.emit_e(args[i].v, 0));
	}

	return join(', ', parts);
};

EM.emit_filter = function(n) {
	let x = EM.emit_e(n.e, 9);
	let args = [];

	for (let i = 0; i < length(n.args); i++)
		push(args, EM.emit_e(n.args[i].v, 0));

	let a0 = length(args) > 0 ? args[0] : ustr('');

	switch (n.name) {
	case 'tojson': {
		/* pyjson over the PLAIN view (MIXED would expose its tag): the
		 * result is a plain string; the static taint ('ident') routes the
		 * output through tpl_out_untrusted, which taints it there */
		for (let i = 0; i < length(n.args); i++) {
			let a = n.args[i];

			if (a.k == null)
				die('jinja2utpl: tojson: positional arguments are not supported');

			if (!(a.k == 'ensure_ascii' &&
			       EM.emit_e(a.v, 0) == 'false'))
				die('jinja2utpl: unsupported tojson argument: ' + a.k);
		}

		use_utpl('pyjson');
		return 'pyjson(' + use_tv('plain') + '(' + x + '))';
	}
	case 'capitalize':
		return use_tv('capitalize') + '(' + x + ')';
	case 'replace':
		return use_tv('replace') + '(' + x + ', ' + a0 + ', ' +
		       (length(args) > 1 ? args[1] : ustr('')) +
		       (length(args) > 2 ? ', ' + args[2] : '') + ')';
	case 'trim':
		return use_tv('trim') + '(' + x + ')';
	case 'lower':
		return use_tv('lower') + '(' + x + ')';
	case 'upper':
		return use_tv('upper') + '(' + x + ')';
	case 'string':
		return use_tv('tostr') + '(' + x + ')';
	case 'safe':
		return x;
	case 'length':
		return use_tv('length_of') + '(' + x + ')';
	case 'items':
		return use_tv('items') + '(' + x + ')';
	case 'default':
		return use_tv('deflt') + '(' + x + ', ' + a0 + ')';
	case 'join':
		return use_tv('join') + '(' + x + ', ' + a0 + ')';
	}

	die('jinja2utpl: unsupported filter: | ' + n.name);
};

EM.emit_is = function(n) {
	let x = '(' + EM.emit_e(n.e, 0) + ')';
	let kind = use_tv('kind');
	let s;

	let kr = kind + '(' + use_tv('raw') + '(' + x + '))';

	switch (n.test) {
	case 'string':
		s = kr + ' == ' + ustr('s');
		break;
	case 'integer':
	case 'int':
		s = kr + ' == ' + ustr('n');
		break;
	case 'float':
	case 'double':
		s = kr + ' == ' + ustr('n');
		break;
	case 'number':
		s = kr + ' == ' + ustr('n');
		break;
	case 'boolean':
	case 'bool':
		s = kr + ' == ' + ustr('b');
		break;
	case 'mapping':
		s = kr + ' == ' + ustr('o');
		break;
	case 'iterable':
		s = '(' + kr + ' == ' + ustr('a') + ' || ' +
		    kr + ' == ' + ustr('s') + ')';
		break;
	case 'none':
	case 'undefined':
		/* missing ctx roots are taint cells wrapping undefined
		 * (kind 'c', v null) — a cell is undefined iff its raw
		 * value is null; the x.v read is short-circuited behind
		 * the kind check (a .v on a primitive would raise) */
		s = '(' + x + ' == null || ' + kind + '(' + x + ') == ' +
		    ustr('z') + ' || (' + kind + '(' + x + ') == ' +
		    ustr('c') + ' && ' + x + '.v == null))';
		break;
	case 'defined':
		s = '(' + x + ' != null && ' + kind + '(' + x + ') != ' +
		    ustr('z') + ' && !(' + kind + '(' + x + ') == ' +
		    ustr('c') + ' && ' + x + '.v == null))';
		break;
	case 'true':
		/* jinja constant tests are strict identity on the RAW value
		 * (cells unwrapped); ucode === does no coercion, so
		 * 1 === true and "true" === true are already false */
		s = use_tv('raw') + '(' + x + ') === true';
		break;
	case 'false':
		s = use_tv('raw') + '(' + x + ') === false';
		break;
	}

	if (n.neg)
		s = '!(' + s + ')';

	return s;
};

/* ------------------------------------------------------------------ */
/* macro inlining                                                      */
/* ------------------------------------------------------------------ */
/* A jinja macro is an ad-hoc template fragment: the transpiler does NOT
 * emit a function for {% macro %}. Every CALL site inlines the body as
 * a new scope (parameter bindings + the body's set targets pre-declared)
 * plus the body's emit statements — the same statement stream the
 * function form used, so whitespace and output semantics are the
 * function's:
 *
 *   output position  the body's runs flow into the live stream
 *   value position   the body runs under a sink CAPTURE (an IIFE so the
 *                    call stays an expression); cap_end() is the value —
 *                    a MIXED string with the body's per-run trust split
 *
 * jinja macros hoist (a body may call a macro defined later in the
 * template); a SELF-recursive or mutually recursive macro would inline
 * forever, so cycles die loudly at transpile time (recursive macros are
 * rejected, not approximated). */


/* is this a call of a (transpile-known) macro? */
function is_macro_call(n) {
	return n.t == 'call' && n.e.t == 'name' &&
	       EM_MACROS != null && EM_MACROS[n.e.n] != null;
}

/* The body's macro-local set targets, pre-declared (jinja's macro
 * namespace: a set inside a macro is local to it). */
function macro_scope_lines(m) {
	let lines = [];

	for (let name in m._sets) {
		let is_param = false;

		for (let p = 0; p < length(m.params); p++)
			if (m.params[p].name == name)
				is_param = true;

		if (!is_param)
			push(lines, 'let ' + name + ' = null;');
	}

	return lines;
}

/* The call's argument expressions, evaluated ONCE at the call site
 * (jinja evaluates arguments before entering the macro), passed as the
 * function's arguments. Returns [ [name, argExpr], ... ]; an omitted
 * arg is the default EXPRESSION (evaluated here, in the caller's scope,
 * as in the old inlined design) or null. A present arg — even one
 * holding none — is a real value and must not trigger the default, so
 * the test is structural (arg present or not), not a null check. */
function macro_bindings(m, n) {
	let out = [];
	let pi = 0;

	for (let i = 0; i < length(n.args); i++) {
		let a = n.args[i];

		if (a.k != null)
			die('jinja2utpl: keyword argument ' + a.k + '= is not '
			    + 'supported in macro calls');

		if (pi >= length(m.params))
			die('jinja2utpl: macro ' + m.name + ' takes '
			    + length(m.params) + ' argument(s)');

		push(out, [ m.params[pi].name, EM.emit_e(a.v, 0) ]);
		pi++;
	}

	for (; pi < length(m.params); pi++) {
		let p = m.params[pi];

		push(out, [ p.name, p.def ? EM.emit_e(p.def, 0) : 'null' ]);
	}

	return out;
}

function macro_cycle_die(m) {
	die('jinja2utpl: recursive macro ' + m.name +
	    ' (recursive macros are not supported)');
}

/* the emitted function identifier of a defined macro (name -> id) */
let MACRO_FN = {};
let EM_MACRO_N = 0;

/* Emit the macro as a function expression at its definition site. The
 * body is emitted through the shared emitter straight into the main
 * buffer (appends are O(1) amortized puts); the finished function
 * source is sliced out and the buffer truncated back to its pre-
 * definition length — no re-copying of the accumulated source. The
 * body is its own whitespace unit: st.pend starts as if the definition
 * tag had just been emitted (see below), and st.pend_text is parked
 * because body tags call flush_text(), which would otherwise flush the
 * OUTER pending template text into the function body. */
function emit_macro_def(n, st) {
	let id = '__macro_' + (EM_MACRO_N++);

	MACRO_FN[n.name] = id;

	/* definition stack: a macro body being emitted that CALLS a macro
	 * still on the stack is a (self|mutual) recursion — die at
	 * transpile time (a runtime infinite loop is worse) */
	push(EM_STACK, n);

	let before = st.out.length();
	let saved_pend = st.pend;
	let saved_ptext = st.pend_text;

	/* the body is its own whitespace unit: start it as if the macro
	 * definition tag had just been emitted (jinja's trim_blocks drops
	 * the first newline after a block tag; what PRECEDED the
	 * definition must not leak in) */
	st.pend = st.opts.trim ? 'nl' : null;
	st.pend_text = '';
	st.out.put('*', 'let ' + id + ' = function(');

	for (let i = 0; i < length(n.params); i++) {
		if (i)
			st.out.put('*', ', ');

		st.out.put('*', n.params[i].name);
	}

	st.out.put('*', ') {');

	let lines = macro_scope_lines(n);

	for (let i = 0; i < length(lines); i++)
		st.out.put('*', lines[i] + '\n');

	EM.emit_nodes(n.body, st, n.close.dl);
	st.out.put('*', '};');
	pop(EM_STACK);

	/* slice the function source out of the main buffer and truncate
	 * back (length(n) sets the length); re-emit it as ONE statement.
	 * The definition tag is a real BLOCK tag: lstrip/trim_blocks apply
	 * to it exactly as before (synth false); the multi-line function
	 * source rides along as the statement text, whitespace-inert
	 * inside */
	let fn_src = st.out.slice(before);
	st.out.length(before);
	st.pend = saved_pend;
	st.pend_text = saved_ptext;

	emit_block(fn_src, n.open.dl, n.close.dr, true, false, st);
}

/* Output position: a call to the macro's function expression. The
 * arguments evaluate in the CALLER's scope (jinja semantics — a self-
 * referring arg like tool_to_json(tool) reads the loop var, never the
 * parameter); dl/dr belong to the CALL TAG and are applied to the
 * surrounding buffered text, not to the body. */
EM.emit_macro_block = function(n, dl, dr, st) {
	let m = EM_MACROS[n.e.n];

	if (index(EM_STACK, m) >= 0)
		macro_cycle_die(m);

	let binds = macro_bindings(m, n);
	let s = MACRO_FN[m.name] + '(';

	for (let i = 0; i < length(binds); i++) {
		if (i)
			s += ', ';

		s += binds[i][1];
	}

	s += ');';
	/* the call is an OUTPUT tag, not a block tag: synth keeps the
	 * statement whitespace-neutral (no lstrip, no trim_blocks pend),
	 * exactly as the old inlined block markers were — only the tag's
	 * own dashes (dl/dr) shape the surrounding text */
	emit_block(s, dl, dr, true, true, st);
};

EM.emit_macro_expr = function(n, minrank, st) {
	let m = EM_MACROS[n.e.n];

	if (index(EM_STACK, m) >= 0)
		macro_cycle_die(m);

	let binds = macro_bindings(m, n);
	let args = '';

	for (let i = 0; i < length(binds); i++) {
		if (i)
			args += ', ';

		args += binds[i][1];
	}

	use_utpl('cap_begin');
	use_utpl('cap_end');
	use_utpl('cap_abort');
	/* value position: the macro emits into a capture (cap_begin/
	 * cap_end live on the per-render state, visible through the
	 * function scope chain); the captured runs — template literals
	 * trusted, user content tainted — come back as a MIXED value, so
	 * filters and tests see the real per-run value. On a body error
	 * the capture is abandoned and the error propagates. */
	let s = '(function() {\ncap_begin();\ntry {\n' +
	        MACRO_FN[m.name] + '(' + args + ');\n' +
	        'return cap_end();\n} catch (e) {\ncap_abort();\n' +
	        'die(e);\n}\n})()';

	return 7 < minrank ? '(' + s + ')' : s;
};


EM.emit_e = function(n, minrank) {
	let s;

	switch (n.t) {
	case 'num':
		return n.raw;

	case 'str':
		return ustr(n.s);

	case 'bool':
		return n.v ? 'true' : 'false';

	case 'null':
		return 'null';

	case 'list': {
		let parts = [];

		for (let i = 0; i < length(n.items); i++)
			push(parts, EM.emit_e(n.items[i], 0));

		return '[' + join(', ', parts) + ']';
	}

	case 'dict': {
		let parts = [];

		for (let i = 0; i < length(n.items); i++) {
			let k = n.items[i][0];
			let key = is_ident_start(substr(k, 0, 1)) &&
				is_ident(k) ? k : ustr(k);

			push(parts, `${key}: ${EM.emit_e(n.items[i][1], 0)}`);
		}

		return '{' + join(', ', parts) + '}';
	}

	case 'name':
		return n.n;

	case 'not':
		/* ucode treats a tagged MIXED scalar as a truthy OBJECT: the
		 * jinja truthiness must go through tv.truthy (plain bool) */
		return '!(' + use_tv('truthy') + '(' + EM.emit_e(n.e, 0) + '))';

	case 'neg':
		return (n.pos ? '' : '-') +
		       (is_primary(n.e) ? EM.emit_e(n.e, 0) :
		                          '(' + EM.emit_e(n.e, 0) + ')');

	case 'bin': {
		if (n.op == '~') {
			/* jinja ~: MIXED concatenation, per-run trust preserved */
			s = `${use_tv('cat')}(${EM.emit_e(n.l, 8)}, ${EM.emit_e(n.r, 8)})`;

			return 7 < minrank ? '(' + s + ')' : s;
		}

		if (n.op == '==' || n.op == '!=') {
			/* MIXED values compare by their value (strings by visible
			 * text, never by object identity); taint never affects the
			 * result */
			let f = n.op == '==' ? use_tv('eq') : use_tv('ne');
			s = f + '(' + EM.emit_e(n.l, 5) + ', ' + EM.emit_e(n.r, 5)
			    + ')';

			return 5 < minrank ? '(' + s + ')' : s;
		}

		if (n.op == '<' || n.op == '<=' || n.op == '>' || n.op == '>=') {
			/* get()/idx() hand out TAGGED MIXED scalars; ucode's ordering
			 * operators compare the OBJECTS (reference-ish), not the values
			 * (probe: tagged(1) > tagged(2) == true). Unwrap both sides with
			 * raw() so the comparison is on the plain value, as in jinja. */
			let w = use_tv('raw');
			s = w + '(' + EM.emit_e(n.l, 5) + ') ' + n.op + ' ' +
			    w + '(' + EM.emit_e(n.r, 6) + ')';

			return 5 < minrank ? '(' + s + ')' : s;
		}

		if (n.op == 'and' || n.op == 'or') {
			/* jinja and/or return their OPERAND (not a bool), with
			 * MIXED-aware truthiness — a bare && would treat a tagged
			 * scalar as a truthy object */
			let t = use_tv('truthy');
			s = n.op == 'and'
			    ? '(' + t + '(' + EM.emit_e(n.l, 0) + ') ? ' +
			      EM.emit_e(n.r, 0) + ' : ' + EM.emit_e(n.l, 0) + ')'
			    : '(' + t + '(' + EM.emit_e(n.l, 0) + ') ? ' +
			      EM.emit_e(n.l, 0) + ' : ' + EM.emit_e(n.r, 0) + ')';

			return 3 < minrank ? '(' + s + ')' : s;
		}

		let r = RANK[n.op];

		/* ordering and arithmetic must run on the PLAIN values: bare ucode
		 * operators on a MIXED tag are broken — `>` compares by object
		 * identity (a tagged 1 is never > a tagged 2), and `-`/`+` give
		 * wrong results when chained ((n-1)-2 != n-3) because the first op
		 * returns a kind-'n' tag the second mis-reads. raw() unwraps to the
		 * plain value where the operators behave. `==`/`!=` already go
		 * through tv.eq/tv.ne; `~` through tv.cat; and/or through truthy. */
		if (n.op == '<' || n.op == '<=' || n.op == '>' || n.op == '>=' ||		    n.op == '+' || n.op == '-' || n.op == '*' || n.op == '/' ||
		    n.op == '//' || n.op == '%') {
			let rr = use_tv('raw');
			s = rr + '(' + EM.emit_e(n.l, r) + ') ' + n.op + ' ' +			    rr + '(' + EM.emit_e(n.r, r + 1) + ')';

			return r < minrank ? '(' + s + ')' : s;
		}

		s = EM.emit_e(n.l, r) + ' ' + n.op + ' ' + EM.emit_e(n.r, r + 1);

		return r < minrank ? '(' + s + ')' : s;
	}

	case 'in':
		return `${use_tv('in_op')}(${EM.emit_e(n.l, 0)}, ${EM.emit_e(n.r, 0)})`;

	case 'notin':
		return `!${use_tv('in_op')}(${EM.emit_e(n.l, 0)}, ${EM.emit_e(n.r, 0)})`;

	case 'is':
		return EM.emit_is(n);

	case 'ternary':
		return `(${use_tv('truthy')}(${EM.emit_e(n.cond, 0)}) ? ${EM.emit_e(n.a, 0)} : ${EM.emit_e(n.b, 0)})`;

	case 'filter':
		return EM.emit_filter(n);

	case 'index':
		/* tv.idx: MIXED-safe + negative-index wrap (jinja a[-1]);
		 * a missing index is a tagged nil (jinja Undefined) */
		return `${use_tv('idx')}(${EM.emit_e(n.e, 9)}, ${EM.emit_e(n.idx, 0)})`;

	case 'rev':
		return `${use_utpl('tpl_rev')}(${EM.emit_e(n.e, 0)})`;

	case 'attr':
		/* tv.get: MIXED-safe attribute; a missing attribute is a tagged
		 * nil (jinja Undefined — falsy, not an error) */
		return use_tv('get') + '(' + EM.emit_e(n.e, 9) + ', ' + ustr(n.n) + ')';

	case 'call': {
		/* a macro call INLINES its body at the call site: a new scope
		 * (parameter + set-target bindings) plus the body's emit
		 * statements — captured (an IIFE) in value position, a plain
		 * block in output position (handled by the 'out' case). */
		if (is_macro_call(n))
			return EM.emit_macro_expr(n, minrank, EM_ST);

		if (n.e.t == 'name') {
			let name = n.e.n;

			if (name == 'raise_exception')
				return `die(${EM.emit_args(n.args)})`;

			if (name == 'range')
				return `${use_utpl('tpl_range')}(${EM.emit_args(n.args)})`;

			if (name == 'namespace') {
				let parts = [];

				for (let i = 0; i < length(n.args); i++) {
					let a = n.args[i];

					if (a.k == null)
						die('jinja2utpl: namespace() requires keyword ' +
						    'arguments');

					push(parts, a.k + ': ' + EM.emit_e(a.v, 0));
				}

				return '{' + join(', ', parts) + '}';
			}

			return name + '(' + EM.emit_args(n.args) + ')';
		}

		if (n.e.t == 'attr' && index(keys(METHODS), n.e.n) >= 0) {
			let base = EM.emit_e(n.e.e, 9);
			let a0 = length(n.args) > 0 ? EM.emit_e(n.args[0].v, 0) : null;

			switch (n.e.n) {
			case 'startswith':
				return `${use_tv('startswith')}(${base}, ${a0})`;
			case 'endswith':
				return `${use_tv('endswith')}(${base}, ${a0})`;
			case 'strip':
			case 'trim':
				return `${use_tv('trim')}(${base})`;
			case 'upper':
				return `${use_tv('upper')}(${base})`;
			case 'lower':
				return `${use_tv('lower')}(${base})`;
			case 'items':
				return `${use_tv('items')}(${base})`;
			case 'split':
				return use_tv('split') + '(' + base + ', ' + a0 + ')';
			}
		}

		return `${EM.emit_e(n.e, 0)}(${EM.emit_args(n.args)})`;
	}
	}

	die('jinja2utpl: internal: unknown expression node ' + n.t);
};


function loop_obj(i, s) {
	use_tv('idx');
	use_tv('length_of');

	return '{index0: ' + i + ', index: ' + i + ' + 1, first: ' + i +
	       ' == 0, last: ' + i + ' == length_of(' + s + ') - 1, length: ' +
	       'length_of(' + s + '), revindex: length_of(' + s + ') - ' + i + ', ' +
	       'revindex0: length_of(' + s + ') - 1 - ' + i + ', previtem: ' + i +
	       ' > 0 ? idx(' + s + ', ' + i + ' - 1) : null, nextitem: ' +
	       i + ' + 1 < length_of(' + s + ') ? idx(' + s + ', ' + i +
	       ' + 1) : null}';
}

function is_const_leaf(e) {
	return e.t == 'num' || e.t == 'str' || e.t == 'bool' || e.t == 'null';
}

/* Flatten a left-deep '+'/'~' chain into leaves; false when the top is
 * not a concatenation. */
function out_split(e, leaves) {
	if (e.t == 'bin' && (e.op == '+' || e.op == '~')) {
		if (e.l.t == 'bin' && (e.l.op == '+' || e.l.op == '~')) {
			if (!out_split(e.l, leaves))
				return false;
		}
		else
			push(leaves, e.l);

		push(leaves, e.r);
		return true;
	}

	return false;
}

/* the left dash of a tag node (block tags carry it under .open) */
function tag_dl(n) {
	if (n.t == 'if' || n.t == 'while' || n.t == 'for' || n.t == 'macro')
		return n.open.dl;

	return n.dl;
}

EM.emit_nodes = function(nodes, st, close_dl) {
	if (EM_MACROS == null) {
		/* top-level call (transpile): register the macro table for the
		 * inliner. Nested calls (macro bodies inlining, control flow)
		 * inherit the registry; EM_STACK is balanced by the inliner. */
		EM_MACROS = st.macros;
		EM_ST = st;
		EM_STACK = [];
	}

	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		switch (n.t) {
		case 'text': {
			let s = n.s;
			let dl = (i + 1 < length(nodes)) ? tag_dl(nodes[i + 1])
			                                    : close_dl;

			if (dl)
				s = rstrip_ws(s);

			emit_text(s, st);
			break;
		}

		case 'out': {
			/* a macro call in OUTPUT position inlines its body as a
			 * statement block: the body's emit statements run in the
			 * live stream (their own per-leaf trust) */
			if (is_macro_call(n.e)) {
				EM.emit_macro_block(n.e, n.dl, n.dr, st);
				break;
			}

			/* a concatenation emits leaf by leaf: each macro leaf as its
			 * inlined block, literals as trusted output, other leaves by
			 * their own taint (values through the untrusted failsafe) */
			let leaves = [];

			if (out_split(n.e, leaves) && length(leaves) > 1) {
				for (let k = 0; k < length(leaves); k++) {
					let L = leaves[k];
					let dl = (k == 0) ? n.dl : 0;
					let dr = (k + 1 == length(leaves)) ? n.dr : 0;

					if (is_macro_call(L)) {
						EM.emit_macro_block(L, dl, dr, st);
						continue;
					}

					let trusted = (is_const_leaf(L) || L._t == 0);
					let e = EM.emit_e(L, 0);
					let call = trusted ? 'tpl_out(' + e + ');'
					          : 'tpl_out_untrusted(' + e + ');';

					use_utpl('tpl_out');
					use_utpl('tpl_out_untrusted');

					emit_block(call, dl, dr, false, false, st);
				}

				break;
			}

			let e = EM.emit_e(n.e, 0);
			let call = n._t
				? `tpl_out_untrusted(${e});`
				: `tpl_out(${e});`;

			use_utpl('tpl_out');
			use_utpl('tpl_out_untrusted');

			emit_block(call, n.dl, n.dr, false, false, st);
			break;
		}

		case 'set': {
			let dotted = index(n.name, '.') >= 0;
			let decl = (!dotted && !n.redecl) ? 'let ' : '';
			let body = decl + n.name + ' = ' + EM.emit_e(n.e, 0) + ';';

			emit_block(body, n.dl, n.dr, true, false, st);
			break;
		}

		case 'exprstmt':
			emit_block(EM.emit_e(n.e, 0) + ';', n.dl, n.dr, true, false, st);
			break;

		case 'if':
			emit_block('if (' + use_tv('truthy') + '(' + EM.emit_e(n.cond, 0) + ')) {',
			         n.open.dl, n.open.dr, true, false, st);
			EM.emit_nodes(n.body, st, length(n.branches) ? n.branches[0].dl
			                                              : n.close.dl);

			for (let b = 0; b < length(n.branches); b++) {
				let br = n.branches[b];
				let head = br.cond ? '} else if (' + use_tv('truthy') + '(' +
				           EM.emit_e(br.cond, 0) + ')) {' : '} else {';

				emit_block(head, br.dl, br.dr, true, false, st);
				EM.emit_nodes(br.body, st,
				           b + 1 < length(n.branches) ? n.branches[b + 1].dl
				                                        : n.close.dl);
			}

			emit_block('}', n.close.dl, n.close.dr, true, false, st);
			break;

		case 'while':
			emit_block('while (' + use_tv('truthy') + '(' + EM.emit_e(n.cond, 0) + ')) {',
			         n.open.dl, n.open.dr, true, false, st);
			EM.emit_nodes(n.body, st, n.close.dl);
			emit_block('}', n.close.dl, n.close.dr, true, false, st);
			break;

		case 'for': {
			let k = st.seqN;
			let sn = '__s' + k, ii = '__i' + k;

			st.seqN++;
			emit_block('let ' + sn + ' = ' + EM.emit_e(n.seq, 0) + ';',
			         n.open.dl, 0, true, true, st);
			emit_block('for (let ' + ii + ' = 0; ' + ii +
			         ' < ' + use_tv('length_of') + '(' + sn + '); ' +
			         ii + '++) {',
			         0, n.open.dr, true, true, st);
			let decls = [];

			for (let t = 0; t < length(n.targets); t++)
				push(decls, 'let ' + n.targets[t] + ' = ' + use_tv('idx') +
				       '(' + sn + ', ' + ii + ');');

			if (length(n.targets) > 1) {
				/* idx() tags the element (a pair array becomes a MIXED
				 * 'a'), so the plain [0]/[1] subscript would miss — index
				 * through tv.idx again */
				let ix = use_tv('idx');

				decls[0] = 'let ' + n.targets[0] + ' = ' + ix + '(' + ix +
				           '(' + sn + ', ' + ii + '), 0);';
				decls[1] = 'let ' + n.targets[1] + ' = ' + ix + '(' + ix +
				           '(' + sn + ', ' + ii + '), 1);';
			}

			push(decls, 'let loop = ' + loop_obj(ii, sn) + ';');
			emit_block(join(' ', decls), 0, 0, true, true, st);
			EM.emit_nodes(n.body, st, n.close.dl);
			emit_block('}', n.close.dl, n.close.dr, true, false, st);
			break;
		}

		case 'break':
		case 'continue':
			emit_block(n.t + ';', n.dl, n.dr, true, false, st);
			break;

		case 'macro': {
			/* jinja has NO macro hoisting: the definition must lexically
			 * precede its first call, or the template errors (undefined
			 * macro). The definition emits the macro as a function
			 * expression; a call before it is a TDZ compile error,
			 * which is the faithful failure. The definition's own tag
			 * dashes handle the surrounding template text (emit_block);
			 * the BODY's whitespace is self-contained (close_dl + the
			 * body's own tag dashes) and is emitted once, here. */
			emit_macro_def(n, st);
			break;
		}

		}
	}

	flush_text(st);
};

/* ------------------------------------------------------------------ */
/* shadowing builtins: lexical rename                                 */
/* ------------------------------------------------------------------ */

/* A template declaration whose name collides with a ucode keyword,
 * builtin or utpl helper would shadow it in the emitted code (e.g.
 * `{% set index = ... %}` would break every `index()` the emitted
 * program relies on).  Such declarations — and every reference that
 * lexically binds to them — are renamed to __tN.  The emitted
 * program's scoping matches the taint walk's (if/while share a scope,
 * for/macro start one), so a top-down walk with a per-scope name map
 * is exact.  Macro names are hoisted function definitions and cannot
 * be renamed per scope — those are rejected. */

let RENAME_N = 0;

function is_shadowing(name) {
	if (index(name, '__') == 0)
		return true;

	return KEYWORDS[name] || is_uc_name(name) || name == 'loop';
}

function rename_map_copy(map) {
	let m = {};

	for (let k in map)
		m[k] = map[k];

	return m;
}

function rename_e(e, map) {
	if (e == null)
		return;

	switch (e.t) {
	case 'num':
	case 'str':
	case 'bool':
	case 'null':
		break;

	case 'name':
		if (index(keys(map), e.n) >= 0)
			e.n = map[e.n];
		break;

	case 'list':
		for (let i = 0; i < length(e.items); i++)
			rename_e(e.items[i], map);
		break;

	case 'dict':
		for (let i = 0; i < length(e.items); i++)
			rename_e(e.items[i][1], map);
		break;

	case 'not':
	case 'neg':
	case 'rev':
		rename_e(e.e, map);
		break;

	case 'bin':
	case 'in':
	case 'notin':
		rename_e(e.l, map);
		rename_e(e.r, map);
		break;

	case 'is':
		rename_e(e.e, map);
		break;

	case 'ternary':
		rename_e(e.cond, map);
		rename_e(e.a, map);
		rename_e(e.b, map);
		break;

	case 'filter':
		rename_e(e.e, map);

		for (let i = 0; i < length(e.args); i++)
			rename_e(e.args[i].v, map);

		break;

	case 'index':
		rename_e(e.e, map);
		rename_e(e.idx, map);
		break;

	case 'attr':
		rename_e(e.e, map);
		break;

	case 'call':
		rename_e(e.e, map);

		for (let i = 0; i < length(e.args); i++)
			rename_e(e.args[i].v, map);

		break;

	die('jinja2utpl: internal: unknown expression node in rename: ' +
	    e.t);
	}
}

function rename_stmts(nodes, map) {
	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		switch (n.t) {
		case 'text':
			break;

		case 'out':
			rename_e(n.e, map);
			break;

		case 'set': {
			rename_e(n.e, map);

			/* dotted target: only the leading segment is a reference */
			let dot = index(n.name, '.');

			if (dot >= 0) {
				let base = substr(n.name, 0, dot);

				if (index(keys(map), base) >= 0)
					n.name = map[base] + substr(n.name, dot);
			}
			else if (is_shadowing(n.name)) {
				let m2 = rename_map_copy(map);

				m2[n.name] = '__t' + RENAME_N;
				RENAME_N++;
				n.name = m2[n.name];
				map = m2;
			}
			break;
		}

		case 'exprstmt':
			rename_e(n.e, map);
			break;

		case 'if':
			rename_e(n.cond, map);
			rename_stmts(n.body, map);

			for (let b = 0; b < length(n.branches); b++) {
				if (n.branches[b].cond)
					rename_e(n.branches[b].cond, map);

				rename_stmts(n.branches[b].body, map);
			}
			break;

		case 'while':
			rename_e(n.cond, map);
			rename_stmts(n.body, map);
			break;

		case 'for': {
			rename_e(n.seq, map);
			let m2 = rename_map_copy(map);

			for (let t = 0; t < length(n.targets); t++) {
				if (is_shadowing(n.targets[t])) {
					m2[n.targets[t]] = '__t' + RENAME_N;
					RENAME_N++;
					n.targets[t] = m2[n.targets[t]];
				}
			}

			rename_stmts(n.body, m2);
			break;
		}

		case 'macro': {
			if (is_shadowing(n.name))
				die('jinja2utpl: macro name "' + n.name + '" collides ' +
				    'with a reserved name (macros are hoisted and ' +
				    'cannot be renamed)');

			let m2 = rename_map_copy(map);

			for (let p = 0; p < length(n.params); p++) {
				let pm = n.params[p];

				rename_e(pm.def, m2);

				if (is_shadowing(pm.name)) {
					m2[pm.name] = '__t' + RENAME_N;
					RENAME_N++;
					pm.name = m2[pm.name];
				}
			}

			rename_stmts(n.body, m2);
			break;
		}

		case 'break':
		case 'continue':
			break;

		die('jinja2utpl: internal: unknown statement node in rename: ' +
		    n.t);
		}
	}
}

/* jinja has ONE namespace per render (macros add their own): a set
 * leaks across if/for blocks and is visible to all later code. ucode
 * block-scopes `let` (a set inside an if branch would be invisible
 * after the endif), so every set target is pre-declared once in its
 * namespace — `let name = null` in the per-render preamble (top level)
 * or in the macro's function scope — and every set is a bare
 * assignment that binds to it. */
function collect_set_names(nodes, bucket) {
	for (let i = 0; i < length(nodes); i++) {
		let n = nodes[i];

		switch (n.t) {
		case 'set':
			if (index(n.name, '.') < 0)
				bucket[n.name] = 1;
			break;

		case 'macro':
			n._sets = {};
			collect_set_names(n.body, n._sets);
			break;

		case 'if':
			collect_set_names(n.body, bucket);

			for (let b = 0; b < length(n.branches); b++)
				collect_set_names(n.branches[b].body, bucket);
			break;

		case 'while':
		case 'for':
			collect_set_names(n.body, bucket);
			break;
		}
	}
}

/* ------------------------------------------------------------------ */
/* entry point                                                        */
/* ------------------------------------------------------------------ */

/* Transpile a Jinja chat template to utpl source.
 * opts: {trim_blocks, lstrip_blocks} (default true, like HF/exllamav3).
 * Returns {source, free}: the utpl source (feed to utpl.load) and the
 * render-context variables (free variables) the template reads. */
export function transpile(src, opts) {
	opts = opts || {};
	let st = {
		out: struct.buffer(),
		pend: null,
		seqN: 0,
		/* fresh per-inlining temp suffix (never reset by emit_nodes, which
		 * resets seqN/EM_STACK for nested bodies): nested macro IIFEs share
		 * one scope and would otherwise reuse __m0 and shadow the outer
		 * argument temp, losing the evaluated argument */
		pend_text: '',
		tv_used: {},
		utpl_used: {},
		opts: {
			trim: opts.trim_blocks == null ? true : !!opts.trim_blocks,
			lstrip: opts.lstrip_blocks == null ? true : !!opts.lstrip_blocks}
	};
	let nodes = jinja_parse(src);
	RENAME_N = 0;
	rename_stmts(nodes, {});
	let a = analyze(nodes);

	st.macros = a.macros;
	EM_ST = st;
	EM_MACROS = null;
	EM_STACK = [];

	/* control characters would break the emitted string literals or the
	 * sink's run bookkeeping */
	for (let i = 0; i < length(src); i++) {
		let code = ord(src, i);

		if (code < 32 && code != 9 && code != 10 && code != 13)
			die('jinja2utpl: template contains a forbidden control '
			    + 'character ' + code);
	}

	/* context roots, one statement each in a single synthetic block.
	 * Every root is a lazy MIXED CELL (the O(1) environment boundary,
	 * L74): taint materializes on access (tv.get / tv.idx / string
	 * ops), so no deep traversal of the prompt structure per render.
	 * A value derived from a root is MIXED-tagged from that point on;
	 * the trust travels in the value. */
	let roots = [];

	for (let i = 0; i < length(a.free); i++)
		push(roots, a.free[i] + ' = ' + use_tv('taint') + '(__utpl_state.ctx.'
		       + a.free[i] + ')');

	/* pre-declare every set target in the template namespace: fresh
	 * `let ... = null` per render (the preamble re-runs on every
	 * render), so jinja's set-leaking reads see the per-render value
	 * and never a stale one from an earlier render */
	let set_names = {};
	collect_set_names(nodes, set_names);

	for (let name in set_names)
		if (index(a.free, name) < 0)
			push(roots, name + ' = null');

	if (length(roots) > 0)
		st.out.put('*', 'let ' + join('; let ', roots) + ';\n');
	EM.emit_nodes(nodes, st);

	/* symbol prelude: the transpiled program is eval'd at runtime
	 * (loadstring has no module source to resolve), so instead of a
	 * static `import {...} from 'ampstein.tv'/'ampstein.utpl'` it binds each
	 * used symbol FROM the injected namespaces: __utpl_state.tv.<n> and
	 * __utpl_state.utpl.<n>. __utpl_state is put in the call scope by
	 * utpl.run_state and carries the two module namespaces.
	 * (L120: eval'd code cannot "refer to sources" — inject symbols.)
	 * Names come from the emitter's use records; sorted for determinism,
	 * one comma-separated `let` per module. */
	let pre = [];

	if (length(st.tv_used) > 0)
		push(pre, 'let ' + join(', ', map(sort(keys(st.tv_used)),
			n => n + ' = __utpl_state.tv.' + n)) + ';');

	if (length(st.utpl_used) > 0)
		push(pre, 'let ' + join(', ', map(sort(keys(st.utpl_used)),
			n => n + ' = __utpl_state.utpl.' + n)) + ';');

	return {source: (length(pre) > 0 ? join('\n', pre) + '\n' : '')
	                 + st.out.pull(),
	         free: a.free};
}
