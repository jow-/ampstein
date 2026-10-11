// tests/taint_values.uc — tv.uc MIXED: the universal template value type.
// Representation, propagation, string ops, frames. Model-free, GPU-free;
// the security invariants of the template renderer's runtime taint layer.
//
// Representation (one prototype, variant in `kind`):
//   's' { runs: [{u, s}] }  string, per-run trust   'n' { u, v } number
//   'a' { u, items }        array                   'b' { u, v } bool
//   'o' { u, v }            object                  'z' { u }    null
//   'c' { u:1, v }          lazy environment cell
// A value carries taint iff some run/bit is set; a clean value is still
// tagged (the universal type; the primitive-scalar exemption is a later
// optimization). Plain ucode values are only ever the raw()/text() views.
import * as tv from 'ampstein.tv';

let fails = 0, total = 0;

function check(name, cond) {
	total++;

	if (cond)
		printf('ok   %s\n', name);
	else {
		fails++;
		printf('FAIL %s\n', name);
	}
}

function str_runs(v) {
	if (!tv.is_mixed(v) || v.kind != 's')
		return null;

	return v.runs;
}

/* --- representation ---------------------------------------------------- */

let cell = tv.taint('alice');
check('cell is tv', tv.is_tv(cell) && tv.is_taint(cell) && cell.kind == 'c');
check('cell raw', tv.raw(cell) == 'alice');
check('cell text', tv.text(cell) == 'alice');
check('cell tainted', tv.tainted(cell));
check('kind of cell', tv.kind(cell) == 'c');
check('plain not tv', !tv.is_tv('x') && !tv.tainted('x') && tv.raw('x') == 'x');
check('plain kinds', tv.kind('x') == 's' && tv.kind([1]) == 'a'
      && tv.kind({}) == 'o' && tv.kind(5) == 'n' && tv.kind(true) == 'b'
      && tv.kind(null) == 'z');

/* a cell over a container stays a cell; not usable as a string */
let dict = tv.taint({ a: 'x', b: ['y'] });
check('container cell raw type', type(tv.raw(dict)) == 'object');
let raised = false;

try {
	tv.text(dict);
}
catch (e) {
	raised = true;
}

check('container as text raises', raised);

/* forged taint from plain data is impossible: a plain object with a
 * tainted-looking key is not a MIXED, and own dunders are inert */
check('plain lookalike not tv', !tv.is_tv({ value: 'x', __tv: 'mixed' })
      && !tv.is_tv({ kind: 'c', u: 1, v: 'x', __tv: 'mixed' }));
check('own dunder inert', ('' + { __tostring__: function() { return 'X'; } })
      != 'X');

/* --- scalar tagging (universal type) ---------------------------------- */

let n0 = tv.num(5), n1 = tv.tnum(5);
check('num tagged', tv.kind(n0) == 'n' && !tv.tainted(n0) && tv.raw(n0) == 5);
check('tnum tagged', tv.kind(n1) == 'n' && tv.tainted(n1) && tv.raw(n1) == 5);
let b0 = tv.bln(true), b1 = tv.tbln(false);
check('bln tagged', tv.kind(b0) == 'b' && tv.raw(b0) == true
      && tv.kind(b1) == 'b' && tv.tainted(b1) && tv.raw(b1) == false);
let z = tv.nil();
check('nil tagged', tv.kind(z) == 'z' && tv.raw(z) == null && !tv.tainted(z)
      && !tv.truthy(z) && tv.text(z) == '');

/* clean string values are tagged too */
let cs = tv.mixed([{ u: false, s: 'ab' }]);
check('clean string tagged', tv.kind(cs) == 's' && !tv.tainted(cs)
      && tv.text(cs) == 'ab');
check('coercion shows text', ('' + cs) == 'ab' && ('' + n1) == '5'
      && ('' + b1) == 'False');

/* --- runs ---------------------------------------------------------------- */

check('runs of plain', (function(r) { return length(r) == 1 && !r[0].u
      && r[0].s == 'ab'; })(tv.runs('ab')));
check('runs of cell', (function(r) { return length(r) == 1 && r[0].u
      && r[0].s == 'ab'; })(tv.runs(tv.taint('ab'))));

let m = tv.mixed([{ u: false, s: 'Dear ' }, { u: true, s: 'a<b' },
                  { u: false, s: '!' }]);
check('mixed kept', tv.is_mixed(m) && tv.tainted(m) && tv.kind(m) == 's');
check('mixed text', tv.text(m) == 'Dear a<b!');
check('mixed runs copy not aliased', (function() {
	let r1 = tv.runs(m);
	r1[0].s = 'MUT';

	return tv.text(m) == 'Dear a<b!';
})());

/* norm: merge adjacent equal-u, drop empties, all-clean collapses to a
 * single clean run, all-empty -> nil */
check('norm merges', (function(v) {
	let r = str_runs(v);

	return length(r) == 2 && r[0].s == 'ab' && !r[0].u
	    && r[1].s == 'cd' && r[1].u;
})(tv.mixed([{ u: false, s: 'a' }, { u: false, s: 'b' },
             { u: true, s: 'c' }, { u: true, s: 'd' }])));
check('norm drops empty', (function(v) {
	let r = str_runs(v);

	return length(r) == 1 && r[0].s == 'ab' && !r[0].u;
})(tv.mixed([{ u: false, s: 'a' }, { u: false, s: '' },
             { u: false, s: 'b' }])));
check('norm all-clean single clean run', (function(v) {
	let r = str_runs(v);

	return length(r) == 1 && r[0].s == 'abc' && !r[0].u
	    && !tv.tainted(v);
})(tv.mixed([{ u: false, s: 'a' }, { u: false, s: 'bc' }])));
check('norm empty is zero-run string', (function(v) {
	return tv.kind(v) == 's' && length(v.runs) == 0 && !tv.truthy(v)
	    && tv.text(v) == '';
})(tv.mixed([])));

/* --- concatenation --------------------------------------------------------
 * The core trust-preservation property: literal + user + literal keeps the
 * literals clean per-run so only the user span tokenizes as plain text. */

let c1 = tv.cat('Dear ', tv.taint('a<b'));
check('cat lit+user mixed', tv.is_mixed(c1) && tv.kind(c1) == 's');
check('cat runs', (function(r) {
	let rr = str_runs(c1);

	return length(rr) == 2 && !rr[0].u && rr[0].s == 'Dear ' && rr[1].u
	    && rr[1].s == 'a<b';
})());

let c2 = tv.cat(c1, ' wrote: ');
check('cat +lit keeps mixed', tv.is_mixed(c2));
check('cat +lit runs', (function() {
	let rr = str_runs(c2);

	return length(rr) == 3 && rr[2].s == ' wrote: ' && !rr[2].u;
})());

check('cat user+user merges to single run', (function(v) {
	let rr = str_runs(v);

	return tv.is_mixed(v) && length(rr) == 1 && rr[0].u;
})(tv.cat(tv.taint('x'), tv.taint('y'))));

check('cat clean+clean stays clean', (function(v) {
	let rr = str_runs(v);

	return tv.kind(v) == 's' && !tv.tainted(v) && length(rr) == 1
	    && rr[0].s == 'ab';
})(tv.cat('a', 'b')));

/* coercion of scalars through cat */
check('cat with number', tv.text(tv.cat('n=', 42)) == 'n=42');

/* --- trim ----------------------------------------------------------------
 * Whitespace peels across run boundaries; a consumed trusted whitespace
 * run must not merge the taint states of its neighbours. */

check('trim plain', tv.text(tv.trim(' ab ')) == 'ab');

/* trim peels only the ENDS: the clean whitespace run between the two
 * tainted runs is interior and must survive untouched */
let t1 = tv.cat(tv.cat(tv.cat('  ', tv.taint('x')), '  \n  '), tv.taint(' y'));
let tt = tv.trim(t1);
check('trim cascade runs', (function() {
	let r = str_runs(tt);

	return length(r) == 3 && r[0].u && r[0].s == 'x' && !r[1].u
	    && r[1].s == '  \n  ' && r[2].u && r[2].s == ' y';
})());

/* leading trusted whitespace disappears, the trusted tail stays clean */
let t2 = tv.cat(tv.cat(' \t', tv.taint('v')), '!');
check('trim l side', (function() {
	let r = str_runs(tv.ltrim(t2));

	return length(r) == 2 && r[0].u && r[0].s == 'v' && !r[1].u
	    && r[1].s == '!';
})());
check('rtrim keeps right ws', tv.text(tv.ltrim(t2)) == 'v!');

let t3 = tv.cat(tv.taint('v'), '  ');
check('rtrim user+ws', (function(v) {
	let r = str_runs(v);

	return length(r) == 1 && r[0].u && r[0].s == 'v';
})(tv.rtrim(t3)));

check('trim all-ws is empty string', (function(v) {
	return tv.kind(v) == 's' && length(v.runs) == 0;
})(tv.trim(tv.cat('  ', tv.taint(' ')))));

/* trim of a tainted run with internal ws only strips edges */
check('trim user edges', (function(r) {
	let rr = str_runs(tv.trim(tv.taint('  a b  ')));

	return length(rr) == 1 && rr[0].u && rr[0].s == 'a b';
})());

/* --- split: taint survives the call ------------------------------------ */

let sp = tv.split(tv.taint('a\nb\nc'), '\n');
check('split result is MIXED array', tv.is_mixed(sp) && sp.kind == 'a');
check('split container taint survives', tv.tainted(sp));
check('split pieces tainted', (function() {
	if (length(sp.items) != 3)
		return false;

	for (let i = 0; i < 3; i++)
		if (!tv.tainted(sp.items[i]))
			return false;

	return tv.text(sp.items[1]) == 'b';
})());
check('split text', tv.text(sp.items[0]) == 'a'
      && tv.text(sp.items[2]) == 'c');

let spc = tv.split('x,y', ',');
check('split clean stays clean', !tv.tainted(spc)
      && !tv.tainted(spc.items[0]) && tv.text(spc.items[1]) == 'y');

/* --- join ------------------------------------------------------------------
 * The template-level separator (trusted) must interleave clean runs
 * between tainted elements. */

let lst = tv.arr([tv.taint('a'), tv.taint('b'), 'c'], 1);
let j = tv.join(lst.items, tv.mixed([{ u: false, s: ', ' }]));
check('join mixed', tv.is_mixed(j) && tv.kind(j) == 's');
/* norm merges adjacent equal-u runs, so the clean separator before the
 * clean element 'c' forms one run: [a][, ][b][, c] */
check('join runs', (function(r) {
	return length(r) == 4 && r[0].u && r[0].s == 'a' && !r[1].u
	    && r[1].s == ', ' && r[2].u && r[2].s == 'b' && !r[3].u
	    && r[3].s == ', c';
})(str_runs(j)));
check('join text', tv.text(j) == 'a, b, c');
check('join empty is empty string', tv.kind(tv.join([], ', ')) == 's'
      && length(tv.join([], ', ').runs) == 0);
check('join clean stays clean', !tv.tainted(tv.join(['a', 'b'], '-'))
      && tv.text(tv.join(['a', 'b'], '-')) == 'a-b');
check('join no sep', tv.text(tv.join([tv.taint('a'), 'b'])) == 'ab');

/* --- replace / in / startswith / endswith -------------------------------- */

let rp = tv.replace(tv.taint('a>b'), '>', 'x');
check('replace over-taints', tv.tainted(rp) && tv.text(rp) == 'axb');
check('replace clean clean', !tv.tainted(tv.replace('a>b', '>', '|'))
      && tv.text(tv.replace('a>b', '>', '|')) == 'a|b');

check('in_op found', tv.in_op('\n', tv.taint('a\nb')));
check('in_op absent', !tv.in_op('z', tv.taint('ab')));
check('in_op taint-transparent', !tv.tainted(tv.in_op('a', tv.taint('ab'))));

check('startswith', tv.truthy(tv.startswith(tv.taint('system'), 'sys'))
      && !tv.truthy(tv.startswith(tv.taint('sys'), 'system')));
check('endswith', tv.truthy(tv.endswith(tv.taint('end.'), '.'))
      && !tv.truthy(tv.endswith(tv.taint('end.'), 'end')));

/* --- logic ------------------------------------------------------------------ */

check('truthy cell str', tv.truthy(tv.taint('0')));
check('truthy empty cell', !tv.truthy(tv.taint('')));
check('truthy cell list', tv.truthy(tv.taint([1])));
check('truthy cell obj', !tv.truthy(tv.taint({})));
check('truthy mixed', tv.truthy(m));
check('truthy nil', !tv.truthy(tv.nil()));
check('truthy clean string', tv.truthy(tv.mixed([{ u: false, s: 'x' }])));
check('truthy tagged scalars', tv.truthy(tv.num(1)) && !tv.truthy(tv.num(0))
      && tv.truthy(tv.tbln(true)) && !tv.truthy(tv.tbln(false)));
check('truthy plain', tv.truthy('x') && !tv.truthy('') && !tv.truthy(null));

check('eq text across taint', tv.eq(tv.taint('system'), 'system'));
check('eq mixed vs text', tv.eq(tv.cat('<|', tv.taint('x|>')), '<|x|>'));
check('eq differs', !tv.eq(tv.taint('a'), 'b'));
check('eq nil both sides', tv.eq(tv.nil(), null) && tv.eq(null, tv.nil()));
check('eq number across tag', tv.eq(tv.tnum(5), 5));
check('eq plain passthrough', tv.eq('a', 'a') && tv.ne('a', 'b'));
check('ne', tv.ne(tv.taint('a'), 'b') && !tv.ne(tv.taint('a'), 'a'));

check('length_of text', tv.length_of(tv.cat('ab', tv.taint('cd'))) == 4);
check('length_of list cell', tv.length_of(tv.taint([1, 2, 3])) == 3);
check('length_of tagged arr', tv.length_of(tv.arr([1, 2], 1)) == 2);

/* --- taint survives calls through the container lattice -------------------- */

let env = { user_dict: { foo: ['zero', 'one', 'two'] }, plain: 'p' };
let root = tv.taint(env);
let v1 = tv.get(root, 'user_dict');
check('get propagates taint', tv.tainted(v1) && tv.kind(v1) == 'o');
let v2 = tv.get(v1, 'foo');
check('get to array', tv.kind(v2) == 'a' && tv.tainted(v2));
let v3 = tv.idx(v2, 1);
check('chain to leaf', tv.text(v3) == 'one' && tv.tainted(v3)
      && tv.kind(v3) == 's');
check('idx negative', tv.text(tv.idx(v2, -1)) == 'two');
check('idx oob nil', tv.kind(tv.idx(v2, 9)) == 'z');
check('get missing nil', tv.kind(tv.get(v1, 'nope')) == 'z');

let arr_cell = tv.taint(['a', 'b']);
check('idx over cell container', tv.text(tv.idx(arr_cell, 1)) == 'b'
      && tv.tainted(tv.idx(arr_cell, 1)));

let clean_root = tv.obj({ lst: ['a'] }, 0);
check('clean chain stays clean', !tv.tainted(tv.idx(tv.get(clean_root,
      'lst'), 0)) && tv.text(tv.idx(tv.get(clean_root, 'lst'), 0)) == 'a');

/* split -> join round-trip keeps the taint end to end */
let rt = tv.join(tv.split(tv.taint('a\nb'), '\n').items,
                 tv.mixed([{ u: false, s: ',' }]));
check('split->join taint survives', tv.tainted(rt)
      && tv.text(rt) == 'a,b');

/* --- wrap_env ----------------------------------------------------------------- */

let w = tv.wrap_env({ s: 'x', n: 5, l: ['y', { z: 'q' }] });
check('wrap is cell', tv.is_taint(w) && tv.kind(w) == 'c');
let ws = tv.get(w, 's');
check('wrap leaf taint on access', tv.tainted(ws) && tv.text(ws) == 'x');
check('wrap container on access', tv.kind(tv.get(w, 'l')) == 'a');
check('wrap deep text', tv.text(tv.get(tv.idx(tv.get(w, 'l'), 1), 'z')) == 'q'
      && tv.kind(tv.idx(tv.get(w, 'l'), 1)) == 'o');

/* --- flatten_op ---------------------------------------------------------------- */

check('flatten keeps taint', tv.tainted(tv.flatten_op(m, 'replaced')));
check('flatten clean stays clean', !tv.tainted(tv.flatten_op('a', 'b'))
      && tv.text(tv.flatten_op('a', 'b')) == 'b');

/* --- frames: prototype-chain scopes ----------------------------------------------
 * Verified VM semantics this builds on: inherited reads return the same
 * cell object (taint identity survives), own writes never touch the
 * parent, shadowing is own-only. */

let base = tv.frame(null);
tv.set(base, 'messages', tv.taint([{ role: 'user', content: 'hi' }]));
tv.set(base, 'tools', tv.taint([]));

let inner = tv.frame(base);
check('frame inherits cell', tv.lookup(inner, 'messages')
      == tv.lookup(base, 'messages'));
check('inherited cell still tainted', tv.tainted(tv.lookup(inner,
      'messages')));

tv.set(inner, 'x', tv.mixed([{ u: false, s: 'lit' }]));
check('own write not in parent', !tv.has(base, 'x') && tv.has(inner, 'x'));

tv.set(inner, 'messages', tv.mixed([{ u: false, s: 'shadow' }]));
check('shadow own-only', tv.text(tv.lookup(inner, 'messages')) == 'shadow'
      && tv.tainted(tv.lookup(base, 'messages')));

/* {% set helper = user_dict.foo.0 %}: the assigned value carries its own
 * taint into the frame — template variables inherit taint for free */
let helper = tv.get(tv.idx(tv.lookup(base, 'messages'), 0), 'content');
tv.set(inner, 'helper', helper);
check('set inherits taint', tv.tainted(tv.lookup(inner, 'helper'))
      && tv.text(tv.lookup(inner, 'helper')) == 'hi');

/* a set variable can launder ONLY if it reads a raw value: the MIXED is
 * the taint carrier, so the escape hatch is tv.raw(), which no emitted
 * template code calls at an output site. Two-set chain stays tainted. */
let v_lit = tv.cat(tv.taint('user'), tv.taint('-supplied'));
tv.set(inner, 'a', v_lit);
tv.set(inner, 'b', tv.lookup(inner, 'a'));
check('set->set chain stays tainted', tv.tainted(tv.lookup(inner, 'b'))
      && tv.text(tv.lookup(inner, 'b')) == 'user-supplied');
check('literal set is clean', (function() {
	tv.set(inner, 'c', tv.cat('template-', 'literal'));

	return !tv.tainted(tv.lookup(inner, 'c'))
	    && tv.text(tv.lookup(inner, 'c')) == 'template-literal';
})());
/* mixed set: literal spliced with user content keeps its trust split
 * through two more assignments */
let v_mix = tv.cat(tv.cat('To ', tv.lookup(inner, 'helper')), ':');
tv.set(inner, 'd', v_mix);
check('mixed survives set', (function(v) {
	let r = str_runs(v);

	return length(r) == 3 && !r[0].u && r[0].s == 'To ' && r[1].u
	    && r[1].s == 'hi' && !r[2].u && r[2].s == ':';
})(tv.lookup(inner, 'd')));

check('has distinguishes bound null', (function() {
	let f = tv.frame(null);
	tv.set(f, 'n', tv.nil());

	return tv.has(f, 'n') && !tv.has(f, 'm');
})());

/* root frames have no prototype: keys cannot resolve through any
 * ambient object prototype */
check('root has no proto', tv.lookup(tv.frame(null), 'toString') == null
      && !tv.has(tv.frame(null), 'toString'));

printf('%d/%d passed\n', total - fails, total);

if (fails > 0)
	die('tv: ' + fails + ' failures');
