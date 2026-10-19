// ucode/tv.uc — MIXED: the template's universal value type.
//
// The template renderer separates TRUSTED text (template literals, which
// carry the protocol: special tokens) from UNTRUSTED data (the render
// context: messages, tool results, ...). Flattening both into one string
// and tokenizing it lets user content containing a special-token string
// be interpreted as protocol syntax (in-band signalling). tv.uc keeps the
// provenance IN THE VALUE: MIXED is the universal template value type and
// taint is a value property that SURVIVES EVERY CALL —
//
//   split(MIXED.string#t)             -> MIXED.array#t   (#t pieces)
//   join(MIXED.string, MIXED.array#t) -> MIXED.string#t
//   get(MIXED.object#t, key)          -> MIXED.#t        (lazy)
//   cat(lit, MIXED#t)                 -> MIXED.string, per-run split intact
//
// Representation — ONE prototype, the variant in `kind`:
//
//   's'  { runs: [{u, s}, ...] }  string: per-run trust (u == true means
//                                  TAINTED / plain-text tokenization)
//   'a'  { u, items: [...] }      array: taint bit, elements are MIXED
//   'o'  { u, v: {...} }          object: taint bit, fields are MIXED
//   'n'  { u, v }                 number
//   'b'  { u, v }                 bool
//   'z'  { u }                    null
//   'c'  { u: 1, v: raw }         cell: LAZY environment boundary — one
//                                  wrapper per context root (O(1) per
//                                  render; no deep traversal of the 600 KB
//                                  prompt structure, L74). Access through
//                                  get()/idx() materializes child MIXEDs
//                                  on demand; the cell never mutates the
//                                  wrapped container.
//
// Universal (this version): EVERY value the template code touches is a
// tagged MIXED, including clean ones — u=0. Plain ucode values appear only
// on the other side of the boundary (raw env data before wrapping, the
// `raw()`/`text()` views). Leaving primitive scalars (numbers/bools/null —
// which cannot be unsafe: they carry no text) untagged is the LATER
// optimization; the ops here are written so that path is a one-line
// change in tag_raw()/the builders, not a redesign.
//
// Forgery is impossible from plain data: values built from JSON can never
// acquire a prototype, and an own-property dunder in plain data is inert
// (metamethods are looked up on the PROTOTYPE CHAIN only,
// ucv_metamethod_lookup() in ucode types.c).
//
// String ops that can be computed per-run (cat, trim, join) preserve the
// run structure. Ops that must cross run boundaries (split, replace, in,
// tojson, capitalize, ...) flatten to the visible text and OVER-TAINT the
// result — over-taint is always safe: it can only withhold special tokens
// a value cannot provably own.
//
// Taint never affects a CONDITION or a COMPARISON result (they are clean
// 'b'/'z' values) — only the text that reaches the sink carries it.
//
// NOTE (L47): ucode does not hoist — every helper below is declared
// before its callers.

/* --- the universal prototype ------------------------------------------------ */

/* NOTE (L47): ucode does not hoist, and a nested function only sees the
 * top-level bindings declared before its own definition. Everything is
 * reordered callee-before-caller; the ONE cycle — MIXED.__tostring__
 * needs the str view, which needs the MIXED prototype check — is broken
 * with a forward declaration. */

function tv_str_inner;

export const MIXED = {
	__tv: 'mixed',
	__tostring__: function() {
		/* coercion shows the visible value (debug/log output); emit sites
		 * must still branch on kind() — never on type() alone */
		return tv_str_inner(this);
	}
};

/* --- inspection ------------------------------------------------------------- */

export function is_mixed(v) {
	return type(v) == 'object' && proto(v) == MIXED;
}

/* jinja str() for any value; containers die (forward-declared above for
 * the __tostring__ metamethod) */
function tv_str_inner(v) {
	if (v == null)
		return '';

	if (type(v) == 'object' && proto(v) == MIXED) {
		if (v.kind == 's') {
			let s = '';

			for (let i = 0; i < length(v.runs); i++)
			s += v.runs[i].s;

			return s;
		}

		if (v.kind == 'z')
			return '';

		if (v.kind == 'c')
			return tv_str_inner(v.v);

		if (v.kind == 'n' || v.kind == 'b')
			return tv_str_inner(v.v);

		die('tv: container used as string');
	}

	let t = type(v);

	if (t == 'string')
		return v;

	if (t == 'int' || t == 'double')
		return '' + v;

	if (t == 'bool')
		return v ? 'True' : 'False';

	die('tv: cannot render ' + t + ' as template text');
}

/* --- inspection ------------------------------------------------------------- */

/* legacy alias: a 'c' cell (the old TAINT prototype) */
export function is_taint(v) {
	return is_mixed(v) && v.kind == 'c';
}

export function is_tv(v) {
	return is_mixed(v);
}

/* variant of a value: 's'/'a'/'o'/'n'/'b'/'z'/'c' for MIXED; plain inputs
 * resolve to the string kinds (an untagged string IS the clean 's' view,
 * an untagged array the clean 'a' view, etc.) */
export function kind(v) {
	if (is_mixed(v))
		return v.kind;

	if (v == null)
		return 'z';

	let t = type(v);

	if (t == 'string')
		return 's';

	if (t == 'array')
		return 'a';

	if (t == 'object')
		return 'o';

	if (t == 'int' || t == 'double')
		return 'n';

	if (t == 'bool')
		return 'b';

	return t;
}

/* Underlying value: cells unwrap lazily, tagged scalars return their v,
 * arrays/objects return their items/v, strings their text, plain values
 * themselves. Logic sites (comparisons, iteration, dict keys) work on
 * raw(). */
export function raw(v) {
	if (!is_mixed(v))
		return v;

	if (v.kind == 's')
		return tv_str_inner(v);

	if (v.kind == 'a')
		return v.items;

	if (v.kind == 'o')
		return v.v;

	return v.v;
}

/* Does this value carry any taint at all? */
export function tainted(v) {
	if (!is_mixed(v))
		return false;

	if (v.kind == 's') {
		for (let i = 0; i < length(v.runs); i++)
			if (v.runs[i].u)
				return true;

		return false;
	}

	return !!v.u;
}

/* --- construction ------------------------------------------------------------- */

/* Lazy environment cell. */
export function taint(v) {
	return proto({ kind: 'c', u: 1, v: v }, MIXED);
}

export function num(v) {
	return proto({ kind: 'n', u: 0, v: v }, MIXED);
}

export function tnum(v) {
	return proto({ kind: 'n', u: 1, v: v }, MIXED);
}

export function bln(v) {
	return proto({ kind: 'b', u: 0, v: v }, MIXED);
}

export function tbln(v) {
	return proto({ kind: 'b', u: 1, v: v }, MIXED);
}

export function nil() {
	return proto({ kind: 'z', u: 0 }, MIXED);
}

/* tagged array / object (u is the container taint bit) */
export function arr(items, u) {
	return proto({ kind: 'a', u: u ? 1 : 0, items: items }, MIXED);
}

export function obj(v, u) {
	return proto({ kind: 'o', u: u ? 1 : 0, v: v }, MIXED);
}

/* one tainted run over a string (the MIXED carrier for a tainted scalar
 * string result); an empty string is a zero-run 's' value, NOT nil — the
 * two must stay distinct values */
export function tstr(s) {
	if (s == '')
		return proto({ kind: 's', runs: [] }, MIXED);

	return proto({ kind: 's', runs: [{ u: true, s: s }] }, MIXED);
}

/* a clean one-run string value (zero-run for '') */
export function cstr(s) {
	if (s == '')
		return proto({ kind: 's', runs: [] }, MIXED);

	return proto({ kind: 's', runs: [{ u: false, s: s }] }, MIXED);
}

/* collapse / coalesce string runs: drop empty runs, merge adjacent
 * equal-u runs, all-clean -> a single clean run, all-empty -> a zero-run
 * 's' value ('').
 * Always builds fresh run objects so input lists are never aliased. */
function norm_runs(rs) {
	let out = [];

	for (let i = 0; i < length(rs); i++) {
		if (rs[i].s == '')
			continue;

		let last = length(out) > 0 ? out[length(out) - 1] : null;

		if (last != null && last.u == !!rs[i].u)
			last.s += rs[i].s;
		else
			push(out, { u: !!rs[i].u, s: rs[i].s });
	}

	let any = false;

	for (let i = 0; i < length(out); i++)
		if (out[i].u)
			any = true;

	if (!any) {
		let s = '';

		for (let i = 0; i < length(out); i++)
			s += out[i].s;

		return cstr(s);
	}

	return proto({ kind: 's', runs: out }, MIXED);
}

/* Build a string value from explicit runs [{u, s}, ...]; adjacent equal-u
 * runs merge, empty runs drop, an all-clean result collapses to a single
 * clean run. */
export function mixed(runs) {
	return norm_runs(runs);
}

/* --- string views --------------------------------------------------------------- */

/* Visible text (or scalar value) of any value (comparisons, lengths,
 * dict keys, capture sinks). Drops the run structure; never use for
 * tokenization. */
export function text(v) {
	return tv_str_inner(v);
}

/* Output run list of any value: 's' -> a COPY of its runs, 'z' / plain
 * null -> no runs (jinja Undefined renders ''), a cell over a
 * string/scalar -> one tainted run, clean scalars -> one clean run.
 * Containers die (they are not string values). Always returns fresh
 * {u,s} objects: callers may mutate the result. */
export function runs(v) {
	let out = [];

	if (v != null && type(v) == 'object' && proto(v) == MIXED) {
		if (v.kind == 's') {
			for (let i = 0; i < length(v.runs); i++)
				push(out, { u: !!v.runs[i].u, s: v.runs[i].s });

			return out;
		}

		if (v.kind == 'c') {
			let rv = v.v;

			if (rv != null && (type(rv) == 'array' || type(rv) == 'object'))
				die('tv: tainted container used as string');

			push(out, { u: true, s: tv_str_inner(v) });

			return out;
		}

		if (v.kind == 'a' || v.kind == 'o')
			die('tv: container used as string');

		if (v.kind == 'z')
			return out;

		if (v.kind == 'n' || v.kind == 'b') {
			push(out, { u: !!v.u, s: tv_str_inner(v) });

			return out;
		}
	}

	if (v == null)
		return out;

	if (type(v) == 'array' || type(v) == 'object')
		die('tv: container used as string');

	push(out, { u: false, s: tv_str_inner(v) });

	return out;
}

/* --- string-valued operations ------------------------------------------------------
 * Every op returns a MIXED string value: the run structure when it can be
 * computed per-run, one tainted run when the op crosses boundaries
 * (over-taint), nil() when empty. Taint survives the call. String values
 * are what {{ output }} and the tokenizer sink consume. */

/* jinja ~ (and any concatenation): preserves the per-run split */
export function cat(a, b) {
	return norm_runs([...runs(a), ...runs(b)]);
}

function is_cls(c) {
	return c == 32 || c == 9 || c == 10 || c == 13;
}

/* jinja strip() / |trim over runs: whitespace peels from the ends across
 * run boundaries; a whitespace-only run consumed entirely vanishes and the
 * cascade continues into the next run (a trusted whitespace run between
 * two tainted runs disappearing must not merge their taint states —
 * norm_runs() merges only equal-u neighbours). l/r default true when both
 * omitted. */
export function trim(v, l, r) {
	if (l == null && r == null)
		l = r = true;

	let rs = runs(v);

	if (l)
		for (let i = 0; i < length(rs); i++) {
			let s = rs[i].s, j = 0;

			while (j < length(s) && is_cls(ord(s, j)))
				j++;

			if (j == 0)
				break;

			rs[i].s = substr(s, j);
		}

	if (r)
		for (let i = length(rs) - 1; i >= 0; i--) {
			let s = rs[i].s, j = length(s);

			while (j > 0 && is_cls(ord(s, j - 1)))
				j--;

			if (j == length(s))
				break;

			rs[i].s = substr(s, 0, j);
		}

	return norm_runs(rs);
}

export function ltrim(v) {
	return trim(v, true, false);
}

export function rtrim(v) {
	return trim(v, false, true);
}

/* jinja str.split(sep): pieces are the visible text split on sep; the
 * boundary-crossing op OVER-TAINTS (a piece may straddle a run boundary it
 * cannot be proven to own). Clean input -> clean pieces; any taint ->
 * tainted pieces (tstr). Returns a MIXED array whose taint bit is the
 * input's. Taint survives the call. */
export function split(v, sep) {
	let s = text(v), out = [];

	if (sep == null || text(sep) == '')
		die('tv: split requires a separator');

	let rest = s, i = 0;

	while (true) {
		i = index(rest, sep);

		if (i < 0) {
			push(out, rest);
			break;
		}

		push(out, substr(rest, 0, i));
		rest = substr(rest, i + length(sep));
	}

	let t = tainted(v) ? 1 : 0, items = [];

	for (let k = 0; k < length(out); k++)
		push(items, t ? tstr(out[k]) :
		             proto({ kind: 's', runs: [{ u: false, s: out[k] }] }, MIXED));

	return proto({ kind: 'a', u: t, items: items }, MIXED);
}

/* over-tainting flattener for ops that cannot be computed per-run
 * (find/regex/format/...): result is the visible text, tainted iff any
 * input carried taint. */
export function flatten_op(any, s) {
	if (!tainted(any))
		return proto({ kind: 's', runs: [{ u: false, s: s }] }, MIXED);

	if (type(s) == 'string')
		return tstr(s);

	return taint(s);
}


function tag_raw(v, u) {
	if (v == null)
		return proto({ kind: 'z', u: u ? 1 : 0 }, MIXED);

	/* already tagged (MIXED): return as-is. A MIXED value is a ucode
	 * object, so without this guard the object branch below would
	 * double-tag it (re-wrap its own kind/runs fields as a fresh 'o'),
	 * corrupting idx() over items() pairs. Idempotent tagging is the
	 * correct semantics: the child already carries its own taint. */
	if (is_mixed(v))
		return v;

	let t = type(v);

	if (t == 'string')
		return u ? tstr(v)
		         : proto({ kind: 's', runs: [{ u: false, s: v }] }, MIXED);

	if (t == 'int' || t == 'double')
		return proto({ kind: 'n', u: u ? 1 : 0, v: v }, MIXED);

	if (t == 'bool')
		return proto({ kind: 'b', u: u ? 1 : 0, v: v }, MIXED);

	if (t == 'array') {
		let items = [];

		for (let i = 0; i < length(v); i++)
			push(items, tag_raw(v[i], u));

		return proto({ kind: 'a', u: u ? 1 : 0, items: items }, MIXED);
	}

	if (t == 'object') {
		let o = {};

		for (let k in v)
			o[k] = tag_raw(v[k], u);

		return proto({ kind: 'o', u: u ? 1 : 0, v: o }, MIXED);
	}

	die('tv: cannot tag ' + t);
}

/* jinja |lower / |upper / |capitalize: visible-text ops that over-taint
 * (they cross run boundaries) */
export function lower(v) {
	return flatten_op(v, lc(text(v)));
}

export function upper(v) {
	return flatten_op(v, uc(text(v)));
}

export function capitalize(v) {
	let s = text(v);

	/* jinja capitalize: first char UPPERCASE, the rest lowercase. The
	 * first char must NOT be wrapped in lc() again — that would undo the
	 * uc() (and yield all-lowercase, jinja's opposite) */
	return flatten_op(v,
		uc(substr(s, 0, 1)) + lc(substr(s, 1)));
}

/* jinja |string: str() of any value — the visible text as a string
 * value (the run split is flattened, so the result over-taints) */
export function tostr(v) {
	return flatten_op(v, text(v));
}

/* jinja |default: replace undefined (null / tagged nil) only */
export function deflt(v, d) {
	return raw(v) == null ? d : v;
}

/* jinja |items: mapping -> [ [k, v], ... ] (insertion order); null ->
 * empty; a non-mapping raises like jinja2's do_items. Keys and values
 * are tagged (they came from the mapping); the pair array itself is
 * plain so the emitted for-loop can iterate it directly. */
export function items(o) {
	if (raw(o) == null)
		return [];

	let r = raw(o);

	if (type(r) != 'object')
		die('tv: items: can only get item pairs from a mapping (got '
		    + type(r) + ')');

	let out = [];

	for (let k in r)
		push(out, [ tstr(k),
		               tag_raw(r[k], tainted(o) ? 1 : 0) ]);

	return out;
}

/* Jinja |list: an array as it is, a string as its characters, a mapping as its keys */
export function tolist(v) {
	let r = raw(v);

	if (type(r) == 'array')
		return v;

	if (type(r) == 'string') {
		let out = [];

		for (let c in split(r, ''))
			push(out, c);

		return arr(out, tainted(v));
	}

	if (type(r) == 'object')
		return arr(map(keys(r), (k) => tstr(k)), tainted(v));

	return arr([], 0);
}

/* Jinja |map('name'): the filter `name` (upper / lower / trim / string) applied to every item */
export function mapf(v, name) {
	let out = [];

	for (let x in raw(v)) {
		let e = tag_raw(x, tainted(v) ? 1 : 0);

		push(out, (name == 'upper') ? upper(e) : (name == 'lower') ? lower(e) : (name == 'trim') ? trim(e) :
			(name == 'string') ? tostr(e) : die('tv: map: unsupported filter ' + name));
	}

	return arr(out, tainted(v));
}

/* Jinja |dictsort: the item pairs of a mapping sorted by key */
export function dictsort(o) {
	if (raw(o) == null)
		return [];

	let r = raw(o);

	if (type(r) != 'object')
		die('tv: dictsort: can only sort a mapping (got ' + type(r) + ')');

	let out = [];

	for (let k in sort(keys(r)))
		push(out, [ tstr(k), tag_raw(r[k], tainted(o) ? 1 : 0) ]);

	return out;
}

/* deep-convert a MIXED value to a PLAIN ucode value (the view pyjson
 * and the driver consume): strings -> visible text, containers ->
 * plain containers, scalars -> their value */
export function plain(v) {
	if (v == null)
		return null;

	if (is_mixed(v)) {
		if (v.kind == 's' || v.kind == 'c' || v.kind == 'n' ||
		    v.kind == 'b')
			return raw(v);

		if (v.kind == 'z')
			return null;

		if (v.kind == 'a') {
			let out = [];

			for (let i = 0; i < length(v.items); i++)
				push(out, plain(v.items[i]));

			return out;
		}

		if (v.kind == 'o') {
			let out = {};

			for (let k in v.v)
				out[k] = plain(v.v[k]);

			return out;
		}
	}

	let t = type(v);

	if (t == 'array') {
		let out = [];

		for (let i = 0; i < length(v); i++)
			push(out, plain(v[i]));

		return out;
	}

	if (t == 'object') {
		let out = {};

		for (let k in v)
			out[k] = plain(v[k]);

		return out;
	}

	return v;
}

/* jinja str.replace(old, new[, count]) — over-taints (crosses runs) */
export function replace(v, find, rep, count) {
	let s = text(v);

	if (s == '')
		return nil();

	if (text(find) == '')
		die('tv: replace requires a non-empty search string');

	let out = '';
	let rest = s;
	let n = 0;

	while (true) {
		if (count != null && n >= count)
			break;

		let i = index(rest, text(find));

		if (i < 0)
			break;

		out += substr(rest, 0, i) + text(rep);
		rest = substr(rest, i + length(find));
		n++;
	}

	return flatten_op(v, out + rest);
}

/* jinja `in` for string values: substring test over the visible text.
 * Taint-transparent (the result is a clean bool). */
/* jinja startswith / endswith on string values (visible text) */
export function startswith(v, p) {
	let s = text(v);

	return bln(length(p) > length(s) ? false : substr(s, 0, length(p)) == p);
}

export function endswith(v, p) {
	let s = text(v);

	return bln(length(p) > length(s) ? false
	      : substr(s, length(s) - length(p)) == p);
}

/* jinja |join over a list of values with a (possibly mixed) separator;
 * the taint of the result is the join of separator and elements */
export function join(list, sep) {
	let sr = runs(sep == null ? '' : sep), out = [];

	for (let i = 0; i < length(list); i++) {
		if (i > 0)
			for (let j = 0; j < length(sr); j++)
				push(out, sr[j]);

		let er = runs(list[i]);

		for (let j = 0; j < length(er); j++)
			push(out, er[j]);
	}

	return norm_runs(out);
}

/* --- logic (taint never affects a condition or a comparison) ---------------------- */

export function truthy(v) {
	if (v == null)
		return false;

	if (is_mixed(v)) {
		if (v.kind == 's')
			return length(v.runs) > 0 && v.runs[0].s != '';

		if (v.kind == 'z')
			return false;

		if (v.kind == 'b')
			return v.v;

		if (v.kind == 'n')
			return v.v != 0;

		return truthy(raw(v));
	}

	if (type(v) == 'bool')
		return v;

	let t = type(v);

	if (t == 'string' || t == 'array' || t == 'object')
		return length(v) > 0;

	if (t == 'int' || t == 'double')
		return v != 0;

	return true;
}

/* jinja ==: string values compare by visible text; everything else
 * compares raw, exactly like the emitted bare `==` did (ucode's own ==
 * semantics, incl. 5 == "5"). Taint never affects the result. */
export function eq(a, b) {
	/* string equality only when BOTH sides are strings (jinja:
	 * 'str' == [1] is false, not an error); everything else compares
	 * raw, exactly like the emitted bare == did */
	let ka = kind(raw(a)), kb = kind(raw(b));

	if (ka == 's' && kb == 's')
		return text(a) == text(b);

	if (ka == 's' || kb == 's')
		return false;

	return raw(a) == raw(b);
}

export function ne(a, b) {
	return !eq(a, b);
}


/* jinja `in`: substring test for string values, element test for
 * arrays, key test for mappings — over MIXED and plain values alike.
 * Taint-transparent (the result is a clean bool). */
export function in_op(x, y) {
	if (y == null)
		return false;

	if (is_mixed(y)) {
		if (y.kind == 's')
			return index(text(y), text(x)) >= 0;

		if (y.kind == 'a') {
			for (let i = 0; i < length(y.items); i++)
				if (eq(y.items[i], x))
					return true;

			return false;
		}

		if (y.kind == 'o')
			return index(keys(y.v), text(x)) >= 0;

		if (y.kind == 'c')
			return in_op(x, y.v);

		return false;
	}

	let t = type(y);

	if (t == 'string')
		return index(y, text(x)) >= 0;

	if (t == 'array') {
		for (let i = 0; i < length(y); i++)
			if (eq(y[i], x))
				return true;

		return false;
	}

	if (t == 'object')
		return index(keys(y), text(x)) >= 0;

	die('tv: in_op on ' + t);
}

/* visible-text byte length of string values; element length of
 * containers/arrays; 0 for scalars */
export function length_of(v) {
	let k = kind(v);

	if (k == 's')
		return length(text(v));

	if (k == 'c') {
		let r = v.v;

		if (r == null)
			return 0;

		if (type(r) == 'array' || type(r) == 'object')
			return length(r);

		return length(tv_str_inner(v));
	}

	if (k == 'a' || k == 'o')
		return length(raw(v));

	if (v == null)
		return 0;

	if (type(v) == 'array' || type(v) == 'object')
		return length(v);

	return length(tv_str_inner(v));
}

/* --- taint propagation through access chains -----------------------------------------
 * The emitter compiles name lookups to frame reads (cells for environment
 * roots) and every subsequent .attr / [index] step through get()/idx():
 * the taint of the base is inherited by the result, so {% set x =
 * user_dict.foo.0 %} hands back a MIXED.#t value without the template
 * saying so. Materialization is LAZY (a cell stays a cell until a string
 * op or output forces it) — the driver boundary stays O(1) per render.
 * Taint survives the call. */

/* tag a raw value as the MIXED of its kind, carrying taint bit u (the
 * universal wrapper: EVERYTHING gets tagged, scalars included — the
 * later primitive-scalar exemption changes this one function only) */
export function get(base, key) {
	if (base == null)
		return proto({ kind: 'z', u: 0 }, MIXED);

	if (is_mixed(base)) {
		/* a tagged object's children are ALREADY tagged — hand them out
		 * as-is (re-tagging would wrap the tag itself) */
		if (base.kind == 'o') {
			let v = base.v[key];

			return v == null ? proto({ kind: 'z', u: 0 }, MIXED) : v;
		}

		if (base.kind == 'c') {
			let r = base.v;
			let v = r == null ? null : r[key];

			return tag_raw(v, 1);
		}

		return proto({ kind: 'z', u: 0 }, MIXED);
	}

	return tag_raw(base[key], 0);
}

export function idx(base, i) {
	if (base == null)
		return proto({ kind: 'z', u: 0 }, MIXED);

	/* jinja x['key'] on a mapping is a property lookup: when the index is a
	 * STRING and the base is an object (tagged 'o', a lazy cell wrapping an
	 * object, or a plain object) do a key access instead of an array index.
	 * The transpiler emits idx() for ALL subscripts, so tool['function']
	 * (a mapping subscript) must not fall through to the array path, which
	 * would return null. Mirrors get(); the plain-object branch tags the
	 * child ucode-style (get() does not reach plain objects here because it
	 * re-tags an already-tagged 'o' child). */
	if (type(i) == 'string') {
		if (is_mixed(base)) {
			if (base.kind == 'o') {
				let v = base.v[i];

				return v == null ? proto({ kind: 'z', u: 0 }, MIXED) : v;
			}

			if (base.kind == 'c')
				return idx(base.v, i);

			return proto({ kind: 'z', u: 0 }, MIXED);
		}

		return type(base) == 'object' ? tag_raw(base[i], 0)
			                            : proto({ kind: 'z', u: 0 }, MIXED);
	}

	let items;
	let tagged_items;

	/* the index may itself be MIXED (a tagged number coming from another
	 * idx() result, a loop bound or ns counter). A bare ucode array fetch
	 * with a tag object retrieves nothing, and the bounds checks below
	 * (`i < 0`) compare the tag by identity — unwrap to the plain number
	 * first. Non-numeric (null / container) indices are out of range. */
	i = raw(i);

	if (type(i) != 'int')
		return proto({ kind: 'z', u: 0 }, MIXED);

	if (is_mixed(base)) {
		if (base.kind == 'a') {
			items = base.items;
			tagged_items = true;
		} else if (base.kind == 'c') {
			items = base.v;
			tagged_items = false;
		} else
			return proto({ kind: 'z', u: 0 }, MIXED);
	} else {
		items = base;
		tagged_items = false;
	}

	if (items == null || type(items) != 'array')
		return proto({ kind: 'z', u: 0 }, MIXED);

	let n = length(items);

	if (i < 0)
		i += n;

	if (i < 0 || i >= n)
		return proto({ kind: 'z', u: 0 }, MIXED);

	/* tagged arrays carry already-tagged children — hand out as-is */
	if (tagged_items)
		return items[i];

	return tag_raw(items[i], tainted(base) ? 1 : 0);
}

/* eager environment wrapper for tests/small values: the LAZY cell — the
 * O(1) boundary (L74); children materialize on access */
export function wrap_env(v) {
	return taint(v);
}

/* --- frames: the template variable registry ---------------------------------------------
 * A frame is a plain object whose PROTOTYPE is the parent frame, so name
 * resolution walks the scope chain via ucode's own property lookup
 * (verified: inherited reads return the identical value — taint identity
 * survives; own writes never leak to the parent; keys() stays own-only).
 * {% set %} / for-loop bindings / macro scopes map onto frame choice;
 * since taint lives in the values, frames only encode scope and
 * shadowing. Root frames have no prototype. */

export function frame(parent) {
	return parent == null ? {} : proto({}, parent);
}

/* own-slot write (never writes through the chain) */
export function set(f, name, v) {
	f[name] = v;
}

/* read through the chain */
export function lookup(f, name) {
	return f[name];
}

/* is the name bound in this frame or any ancestor? (distinguishes a bound
 * null from unbound, which a bare read cannot) */
export function has(f, name) {
	for (let p = f; p != null; p = proto(p))
		if (index(keys(p), name) >= 0)
			return true;

	return false;
}
