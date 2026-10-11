// ucode/sink.uc — the output sink of the template renderer.
//
// One output stream per render. Every emission — template-literal text,
// an interpolated value, a macro's output — goes through emit(), which
// appends to the currently active run buffer and MERGES adjacent runs of
// the same taint, so the render result is a list of whole runs rather
// than one segment per emission. There is no string concatenation
// anywhere on the render path (the buffer is the only accumulator), and
// there is no second representation of the output: the run list IS the
// result (docs/TEMPLATE.md §3.2).
//
// Taint travels with the VALUE, not with the emitted code path:
//
//   plain string   -> TRUSTED. The only way a bare string reaches emit()
//                     is as a template literal in generated code: every
//                     value derived from the render context is wrapped
//                     (tv.taint -> cell, tv.mixed/c -> MIXED) and every
//                     filter is taint-preserving, so a string is not
//                     reachable from context data.
//   tv.MIXED       -> its own per-run trusted/tainted structure
//   tv cell        -> visible text, TAINTED
//   anything else  -> str(), TAINTED (fail closed)
//
// Because the representation carries the trust, the transpiler's static
// taint analysis is an optimization (it may skip wrappers), never a
// security guarantee (docs/TEMPLATE.md §6).
//
// Capture (docs/TEMPLATE.md §3.3/§3.4): a macro whose output is used as a
// VALUE ({{(m(x))|trim}}, {% set s = m(x) %}) is compiled with a
// capture wrapper — capture_begin() diverts the sink into a fresh run
// buffer, the body emits normally, capture_end() materializes the
// captured runs as a MIXED value (trust split intact) and restores the
// parent buffer exactly as it was. Captures nest. An aborted render
// resets the sink (reset()).

import * as tv from './tv.uc';
import * as struct from 'struct';

/* segment taint flags (kept compatible with the historical {t} field:
 * t == true means trusted / special-token-enabled) */
export const TRUSTED = 1, TAINTED = 0;

/* one active run buffer: `buf` accumulates the run in progress, `segs`
 * the runs already closed. Declared before new(). */
function new_act() {
	return {buf: struct.buffer(), t: TRUSTED, segs: [], closed: false};
}

export function new() {
	return {top: new_act(), stack: [], depth: 0};
}


export function trusted(s) {
	return {t: TRUSTED, s: s};
}

export function tainted(s) {
	return {t: TAINTED, s: s};
}

/* Maximum capture nesting (macro-as-value inside macro-as-value). A
 * template that nests deeper is pathological: raise rather than
 * recurse unboundedly. */
const MAX_CAPTURE = 32;

/* struct.buffer's '*' format refuses non-strings, so a value that is not
 * a bare string goes through tv.text() (the visible text of a cell or
 * MIXED value; every run the sink buffers is a plain string anyway) */
function bput(buf, s) {
	if (s == null || s == '')
		return;

	buf.put('*', type(s) == 'string' ? s : tv.text(s));
}

/* Append one text run, merging with the active buffer when the taint
 * matches. O(1) amortized. */
export function run(sk, t, s) {
	if (s == null || s == '')
		return;

	let a = sk.depth > 0 ? sk.stack[sk.depth - 1] : sk.top;

	if (a.t != t) {
		let old = a.buf.pull();

		if (old != '')
			push(a.segs, {t: a.t, s: old});

		a.t = t;
		a.buf = struct.buffer();
	}
	bput(a.buf, s);
}

/* Emit any value with its own trust (see the header for the mapping). */
export function emit(sk, v) {
	if (v == null)
		return;

	if (tv.is_mixed(v)) {
		let rs = tv.runs(v);

		for (let i = 0; i < length(rs); i++)
			run(sk, rs[i].u ? TAINTED : TRUSTED, rs[i].s);

		return;
	}

	if (tv.is_taint(v)) {
		run(sk, TAINTED, tv.text(v));
		return;
	}

	let t = type(v);

	if (t == 'string') {
		run(sk, TRUSTED, v);
		return;
	}

	if (t == 'int' || t == 'double' || t == 'bool') {
		run(sk, TAINTED, tv.text(v));
		return;
	}

	die('sink: cannot emit ' + t);
}

/* Divert the sink into a fresh run buffer. */
export function capture_begin(sk) {
	if (sk.depth >= MAX_CAPTURE)
		die('sink: capture nesting exceeds ' + MAX_CAPTURE);

	push(sk.stack, new_act());
	sk.depth++;
	return sk.depth;	/* a template function may only RETURN, so the
	                     * wrapper keeps the depth in a render-global */
}

/* Close the active buffer of an accumulator: pull the run in progress
 * into its run list. Idempotent. A closed accumulator has buf == null
 * and must therefore NEVER stay in a reachable slot: sk.stack is grown
 * by reference with push(), and ucode turns an explicit null slot into
 * undefined for the next grow — which would poison unrelated element
 * reads far from here (the crash seen during bring-up). Both
 * capture_end() and capture_abort() replace their slot with a fresh
 * accumulator right away; finish() replaces sk.top. */
function act_close(a) {
	if (!a.closed) {
		let last = a.buf.pull();

		if (last != '')
			push(a.segs, {t: a.t, s: last});

		a.closed = true;

		/* the tripwire: a stale reference dereferencing this raises
		 * instead of silently writing into a dead buffer */
		a.buf = null;
	}
}

/* The finished run list of an accumulator, DETACHED from it: ucode
 * assigns arrays by reference, so a plain `let segs = sk.top.segs`
 * alias would let the NEXT phase's emissions extend the list a previous
 * caller is still holding. */
function act_runs(a) {
	let segs = a.segs;

	a.segs = [];

	return segs;
}

/* The captured runs of a closed accumulator as a MIXED value (tv run
 * polarity: u == true means TAINTED / plain text — the same convention
 * as tv.runs() of a cell). */
function capture_value(a) {
	let r = a.segs;
	let runs = [];

	for (let i = 0; i < length(r); i++)
		push(runs, {u: r[i].t == TAINTED, s: r[i].s});

	return tv.mixed(runs);
}

/* End the innermost capture: its runs as a MIXED value (empty capture ->
 * '' , which is the all-clean collapse of an empty run list — callers
 * must not have to special-case it), parent buffer exactly as it was. */
export function capture_end(sk) {
	if (sk.depth == 0)
		die('sink: capture_end without capture_begin');

	sk.depth--;
	let a = sk.stack[sk.depth];

	/* Replace the slot BEFORE closing: the closed accumulator keeps a
	 * null buffer, and a null inside sk.stack would poison the next
	 * grow-by-reference push (see act_close), so it never stays. */
	sk.stack[sk.depth] = new_act();
	act_close(a);

	return capture_value(a);
}


/* Discard the innermost capture (an aborted render). */
export function capture_abort(sk) {
	if (sk.depth == 0)
		return;

	sk.depth--;
	act_close(sk.stack[sk.depth]);
	sk.stack[sk.depth] = new_act();
}

/* Finish the render: the ordered merged run list. The active buffers are
 * pulled into it, so a second finish() on the same sink returns only
 * what was emitted since (the runtime always makes a fresh sink per
 * render). */
export function finish(sk) {
	if (sk.depth != 0)
		die('sink: render finished inside a capture (' + sk.depth +
		    ' unclosed)');

	act_close(sk.top);

	/* Hand out the finished runs (act_runs detaches the list, so the
	 * returned array is never extended by later emissions) and install
	 * a FRESH top accumulator: the sink stays usable, and a second
	 * finish() must not deref the closed buffer. */
	let segs = act_runs(sk.top);

	sk.top = new_act();

	return segs;
}

/* Visible text of a run list (the render's .text; spliced from the
 * segments so it always carries real values). */
export function text_of(segs) {
	let b = struct.buffer();

	for (let i = 0; i < length(segs); i++)
		bput(b, segs[i].s);

	return b.pull();
}

/* Reset a sink to empty (called when a render aborts, so a half-open
 * capture can never leak into the next render). */
export function reset(sk) {
	sk.top = new_act();
	sk.stack = [];
	sk.depth = 0;
}
