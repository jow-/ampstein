// ucode/utpl.uc — taint-aware template rendering runtime for the
// transpiled chat templates (ucode/jinja2utpl.uc), paired with
// ucode/sink.uc (the output sink) and ucode/tv.uc (MIXED: the template's
// universal value type).
//
// Why: a chat template mixes TRUSTED text (the template literals, which
// carry the protocol: special tokens like </think>) with
// UNTRUSTED data (the render context: messages, tool results, ...).
// Flattening both into one string and tokenizing it lets user content
// containing a special-token string be interpreted as protocol syntax
// (in-band signalling). utpl keeps the provenance:
//
//   - the transpiled program is PLAIN ucode: every output byte is an
//     explicit call (tpl_out / tpl_out_untrusted) into the sink,
//   - the value domain is tv.MIXED (the universal type): the preamble
//     binds every context root as a lazy cell, access materializes
//     tagged children on demand, and every op is MIXED -> MIXED with
//     taint surviving the call. The trust travels IN THE VALUE; the
//     transpiler's static taint analysis is an optimization, not a
//     guarantee — a value that arrives under-tagged still taints at the
//     tpl_out_untrusted failsafe,
//   - the stream is ucode/sink.uc: one run buffer, adjacent same-taint
//     runs merge, the run list IS the result (no markers, no second
//     representation),
//   - a jinja macro is an ad-hoc template: the transpiler inlines its
//     body at each call site (a new scope, emit statements). In output
//     position the body's runs flow into the live stream; in value
//     position the body runs under a sink CAPTURE and the gathered runs
//     are the value — a MIXED string (runs of text + trust, or a
//     zero-run value for an empty render),
//   - the rendered text is split into segments; trusted segments are
//     tokenized with special tokens ENABLED, tainted segments as plain
//     text (enable_special: false).
//
// API:
//   load(jinja_src, opts?)        transpile + compile a chat template
//   render_tpl(tpl, ctx)          render ctx -> {text, segments}
//   encode_segments(tok, r)       tokenized ids with per-segment flags
//   new_state/run_state/finish_state  low-level (custom drivers)
//
// The transpiled program reaches this module only through its OWN import
// line (ucode/jinja2utpl.uc emits it): the sink entry points
// (tpl_out / tpl_out_untrusted / cap_begin / cap_end / cap_abort /
// pyjson / tpl_range / tpl_rev) and the value ops from ucode/tv.uc —
// individual symbols, no wildcards, no installed globals. The render
// state travels per call through call()'s scope override
// (__utpl_state): the program imports this module as a SEPARATE
// instance (a loadstring'd program compiles its imports separately),
// and only the global environment is shared dynamically — see
// run_state().

import * as jinja2utpl from './jinja2utpl.uc';
import * as sink from './sink.uc';
import * as tv from './tv.uc';

/* --- helpers (imported by the transpiled programs) ---------------------- */

/* Jinja range(stop) / range(start, stop) / range(start, stop, step) */
export function tpl_range(a, b, step) {
	/* the transpiled program passes MIXED values for the bounds (loop
	 * vars are tv tags); unwrap them to plain values — bare `<`/`+=` on a
	 * MIXED tag is broken (identity compare / wrong step), so the loop
	 * would never run correctly */
	a = tv.raw(a);
	b = tv.raw(b);
	step = tv.raw(step);

	if (step == null) {
		if (b == null) {
			b = a;
			a = 0;
		}

		step = 1;
	}

	if (step == 0)
		die('utpl: tpl_range: step must not be zero');

	let out = [];

	if (step > 0)
		for (let i = a; i < b; i += step)
			push(out, i);
	else
		for (let i = a; i > b; i += step)
			push(out, i);

	return out;
}

/* Jinja [::-1]. Element access through tv.idx (MIXED-safe, taint-
 * preserving); the result is a MIXED array carrying the input's taint. */
export function tpl_rev(a) {
	let n = tv.length_of(a);
	let out = [];

	for (let i = n - 1; i >= 0; i--)
		push(out, tv.idx(a, i));

	return tv.arr(out, tv.tainted(a));
}

/* Jinja a[lo:hi] (step 1; negative bounds count from the end): arrays and strings */
export function tpl_slice(a, lo, hi) {
	let n = tv.length_of(a);
	let l = (lo == null) ? 0 : lo;
	let h = (hi == null) ? n : hi;

	if (l < 0)
		l = max(n + l, 0);

	if (h < 0)
		h = max(n + h, 0);

	l = min(l, n);
	h = min(h, n);

	if (type(a) == 'string')
		return (h > l) ? substr(a, l, h - l) : '';

	let out = [];

	for (let i = l; i < h; i++)
		push(out, tv.idx(a, i));

	return tv.arr(out, tv.tainted(a));
}

/* Jinja |tojson == json.dumps(v, ensure_ascii=False) with the default
 * separators (", " and ": "). ucode's %J matches the escaping; only the
 * structural spacing differs. */
export function pyjson(v) {
	let t = type(v);

	if (t == 'object') {
		let parts = [];

		for (let k in v)
			push(parts, sprintf('%J', k) + ': ' + pyjson(v[k]));

		return '{' + join(', ', parts) + '}';
	}

	if (t == 'array') {
		let parts = [];

		for (let i = 0; i < length(v); i++)
			push(parts, pyjson(v[i]));

		return '[' + join(', ', parts) + ']';
	}

	return sprintf('%J', v);
}

/* --- the value carrier: tv.MIXED -------------------------------------- */

/* The transpiled program imports the value ops it needs from ucode/tv.uc
 * by bare name (individual symbols — the emitter records each use). Plain
 * ucode values flow through every op unchanged (the clean optimization);
 * a tainted value is always tagged, so the sink can never launder it. */

/* --- output entry points ------------------------------------------------ */

/* The two ways a template output reaches the sink. The value carries
 * the trust (MIXED runs); the entry point is a FAILSAFE:
 *
 *   tpl_out(v)           emit per the value's own trust. Plain strings
 *                        (template literals) are trusted by
 *                        construction; a MIXED value emits its runs.
 *   tpl_out_untrusted(v) the transpiler's static analysis says the
 *                        expression is context-derived. A MIXED string
 *                        still emits its runs (its trust split is the
 *                        authority — it got there through the value
 *                        lattice); anything that arrives WITHOUT the tag
 *                        is forced to a single TAINTED run, so a static
 *                        analysis bug fails closed, never open.
 *
 * Both return '' so the emitted calls are plain statements. */
function emit_value(sk, v) {
	if (v == null)
		return '';

	let rs = tv.runs(v);

	for (let i = 0; i < length(rs); i++)
		sink.run(sk, rs[i].u ? sink.TAINTED : sink.TRUSTED, rs[i].s);

	return '';
}

function emit_untrusted(sk, v) {
	if (v == null)
		return '';

	if (tv.is_mixed(v))
		return emit_value(sk, v);

	sink.run(sk, sink.TAINTED, tv.text(v));

	return '';
}

export function tpl_out(v) {
	let st = __utpl_state;

	if (st == null)
		die('utpl: template output outside a render');

	return emit_value(st.sink, v);
}

export function tpl_out_untrusted(v) {
	let st = __utpl_state;

	if (st == null)
		die('utpl: template output outside a render');

	return emit_untrusted(st.sink, v);
}

/* --- macro capture (value position) -------------------------------------
 * A macro call in value position runs its inlined body under a sink
 * capture; cap_end() hands back the body's runs as a MIXED string value
 * (zero-run for an empty render — falsy, jinja's empty Markup). The
 * capture is re-entrant (nested captures) and exceptions unwind it
 * (sink.capture_abort via cap_cleanup on raise). */

export function cap_begin() {
	let st = __utpl_state;

	if (st == null)
		die('utpl: capture outside a render');

	sink.capture_begin(st.sink);
}

export function cap_end() {
	let st = __utpl_state;

	if (st == null)
		die('utpl: capture outside a render');

	return sink.capture_end(st.sink);
}

/* abandon the innermost capture (an inlined body that raised) */
export function cap_abort() {
	let st = __utpl_state;

	if (st == null)
		return;

	sink.capture_abort(st.sink);
}

/* --- render state --------------------------------------------------------- */

/* The state object of a render ({ctx, sink}). The transpiled program
 * sees it as __utpl_state through run_state()'s scope override; the
 * preamble reads __utpl_state.ctx and wraps each context root as a lazy
 * MIXED cell. */
/* Module namespaces injected into the transpiled program's scope. The
 * transpiled program is eval'd at runtime (loadstring) and cannot
 * `import` a module, so it binds each used symbol from these namespaces
 * via __utpl_state.tv.<n> / __utpl_state.utpl.<n> (jinja2utpl emit).
 * `tv` is the imported module namespace; __ops is a hand-built namespace
 * of utpl's own output/capture/value ops. */
let __ops = {tpl_range, tpl_rev, tpl_slice, pyjson, tpl_out, tpl_out_untrusted,
	cap_begin, cap_end, cap_abort};

export function new_state(ctx) {
	return {ctx: ctx, sink: sink.new(), tv: tv, utpl: __ops};
}

/* Run a compiled program under the state's scope override.
 *
 * The program imports this module as a SEPARATE instance (a
 * loadstring'd program compiles its imports separately — module state
 * is not shared), so the state cannot be a module local. It travels
 * through the GLOBAL environment instead, which call() overrides for
 * the duration of the call: every helper the program invokes — in
 * whichever module instance it landed in — resolves __utpl_state
 * dynamically against this scope. On a program error the state (and
 * any half-open capture with it) is dropped with the object; nothing
 * process-global is ever written. */
export function run_state(tpl, st) {
	call(tpl.fn, null, {__utpl_state: st});
}

/* sink segments [{t: 1|0, s}] -> the public [{t: bool, s}] shape
 * (polarity preserved from the old path: t: true = TRUSTED) */
function sink_segments(segs) {
	let out = [];

	for (let i = 0; i < length(segs); i++)
		push(out, {t: segs[i].t == sink.TRUSTED, s: segs[i].s});

	return out;
}

/* End a render: the sink's run list is the result. Returns
 * {text, segments}; the state object is single-use. */
export function finish_state(st) {
	let segs = sink_segments(sink.finish(st.sink));
	let text = [];

	for (let i = 0; i < length(segs); i++)
		push(text, segs[i].s);

	return {text: join('', text), segments: segs};
}

/* --- high level -------------------------------------------------------------
 * Transpile (jinja2utpl) + compile a Jinja chat template. Returns
 * {fn, source, free}. The emitted program is plain ucode that reaches
 * the runtime through the installed globals only. */
export function load(jinja_src, opts) {
	let r = jinja2utpl.transpile(jinja_src, opts);
	let fn = loadstring(r.source, {raw_mode: true, trim_blocks: false,
	                                lstrip_blocks: false});

	return {fn: fn, source: r.source, free: r.free};
}

/* Render ctx through a compiled program. Returns {text, segments}.
 * If the program raises, the state is left as-is; the next render_tpl()
 * resets it (the helpers are only reachable from inside a render). */
export function render_tpl(tpl, ctx) {
	let st = new_state(ctx);

	run_state(tpl, st);

	return finish_state(st);
}

/* Tokenize a rendered template segment-wise. tok: a torch.tokenizer()
 * resource. Trusted segments are encoded with special tokens enabled,
 * tainted segments as plain text — a user string containing a
 * special-token sequence can never become a special token. */
export function encode_segments(tok, r) {
	let segs = r.segments;
	let ids = [];

	for (let i = 0; i < length(segs); i++) {
		let s = segs[i];

		if (s.s == '')
			continue;

		// one native call per segment (a per-token push loop cost more than
		// the encoding itself)
		push(ids, ...(s.t ? tok.encode(s.s, true) : tok.encode(s.s)));
	}

	return ids;
}

/* --- prompt tokenization cache --------------------------------------------
 * Segment helpers for the cache below. Segments are ordered {t, s}
 * runs; a render extends an earlier one when the old segments are a
 * structural prefix (taint AND bytes) of the new ones. */

export function seg_equal(a, b) {
	if (length(a) != length(b))
		return false;

	for (let i = 0; i < length(a); i++)
		if (a[i].t != b[i].t || a[i].s != b[i].s)
			return false;

	return true;
}

/* Number of bytes of `a` that `b` carries verbatim at the start
 * (taint and bytes), or -1 if `a` is not a structural prefix. */
export function seg_prefix(b, a) {
	let alen = 0;

	for (let i = 0; i < length(a); i++)
		alen += length(a[i].s);

	let ai = 0, bi = 0, aoff = 0, boff = 0, matched = 0;

	while (ai < length(a) && bi < length(b)) {
		let A = a[ai], B = b[bi];
		let take = min(length(A.s) - aoff, length(B.s) - boff);

		if (A.t != B.t || substr(A.s, aoff, take) != substr(B.s, boff, take))
			return -1;

		matched += take;
		aoff += take;
		boff += take;

		if (aoff == length(A.s)) {
			ai++;
			aoff = 0;
		}

		if (boff == length(B.s)) {
			bi++;
			boff = 0;
		}
	}

	return (ai < length(a)) ? -1 : matched;
}

/* The segments of `b` from byte offset `at` (splitting the segment
 * the offset falls inside). */
export function seg_tail(b, at) {
	let out = [], off = 0;

	for (let i = 0; i < length(b); i++) {
		let sv = b[i].s;

		if (off + length(sv) <= at) {
			off += length(sv);
			continue;
		}

		let t = substr(sv, max(0, at - off));

		if (t != '')
			push(out, {t: b[i].t, s: t});

		off += length(sv);
	}

	return out;
}

/* A one-entry prompt-tokenization cache (the server's prefix path):
 * agent sessions re-send the whole conversation every turn, so a
 * render usually extends the previous one (the template renders the
 * same bytes for the same leading messages; the new part starts where
 * the old render's generation tail met the new assistant content).
 * When the OLD segments are a structural prefix of the new ones, the
 * boundary is a SEGMENT boundary and each side encodes independently
 * (taint per segment, never across one): cached ids + encoding of the
 * NEW segments equals a full encode, and the tainted user content in
 * the new segments is never atomized.
 *
 * encode: (segs) -> ids; scan_pads: (ids, from) -> positions (may be
 * null on the first hit). Returns the prefix ids; the cache's pads
 * follow the same one-entry lifecycle. */
export function prefix_cache() {
	let c = {segs: null, text: null, ids: null, pads: null};

	return {
	run: function(encode, r, scan_pads) {
		let segs = r.segments, text = r.text;
		let ids = null, pads = null;

		if (c.segs != null) {
			let plen = seg_prefix(segs, c.segs);

			if (plen >= 0) {
				ids = slice(c.ids);
				let n0 = length(ids);

				push(ids, ...encode(seg_tail(segs, plen)));

				// pad positions carry over; only the new part is scanned
				if (c.pads != null)
					pads = [ ...c.pads, ...scan_pads(ids, n0) ];
			}
			else if (c.text == text && seg_equal(segs, c.segs)) {
				ids = slice(c.ids);
				pads = c.pads;
			}
		}

		if (ids == null)
			ids = encode(segs);

		c = {segs: segs, text: text, ids: slice(ids), pads: pads};
		return ids;
		},
	get: function() {
		return c;
	}
	};
}
