# Template rework — design document

Status: **implemented** (the oracle suite is `tests/chat_jinja.uc` /
`tests/utpl_template.uc`). This document is
kept as the design rationale: what the system is for, what was wrong with
the earlier placeholder-capture design, and the shape of the replacement.

Scope: `tv.uc`, `utpl.uc`, `jinja2utpl.uc`, `chat_jinja.uc` and the
tokenizer sink (`utpl.uc::encode_segments`), all under `src/ucode/ampstein/`.

---

## 1. What the system is for

Chat templates in HF checkpoints ship as Jinja (`chat_template.jinja` or the
`chat_template` field of `tokenizer_config.json`). We render them **without
Python** (invariant 1) and without a Jinja interpreter at runtime:

```
chat_template.jinja --(jinja2utpl, offline-ish, once per model load)--> utpl source
                                        --(loadstring)--> compiled template function
messages/tools/etc. --(utpl runtime + tv taint values)--> segment stream
segment stream --(encode_segments)--> token ids
```

Two hard requirements, in priority order:

1. **Byte-exactness.** Rendered prompts must match the jinja2 oracle
   byte-for-byte. The oracle suite is 23 cases over the real Qwen3.8 and
   GLM-5.3 templates (`tests/chat_jinja.uc`, `tests/utpl_template.uc`,
   `tests/oracle/make_chat_*_oracle.py`).
2. **Taint safety.** Text that came from the request (`messages`, `tools`,
   tool results, image captions, anything a model could have written) must
   never be tokenized with special-token atomization. Text that came from
   the *template itself* must be, because a template that renders its own
   `<|im_start|>` as literal text produces a prompt the model cannot parse.

Everything below is in service of those two.

## 2. What is wrong with the current design

Three mechanisms are fighting each other, and the last one is the root
cause of the current mess:

**(a) Placeholder capture.** `untrusted(v)` records `v` in a side array and
returns a `\x01<idx>\x01` marker string, which is spliced into the captured
output. Afterwards `scan_markers()` walks the captured string, splits it on
`\x01`, and recursively expands the recorded values. Consequences:

* the rendered output exists **twice** (marker form and real form);
* the whole prompt is a `+=` string, which is exactly the O(n²) value-churn
  pattern removed elsewhere in the engine (L72);
* taint round-trips through *string serialization*, so anything that isn't
  a string has to be pre-flattened to text, losing information.

**(b) The macro private buffer (`__mbuf`).** Macro bodies were compiled to
`{% let __mbuf = tpl_buffer(); %} … __mbuf.put('*', …) … {{ return
__mbuf.pull() }}` because we believed filters (`|trim`), truthiness and
string methods must see the macro's real value. Consequences:

* macro output loses its internal trust split — the whole return value
  becomes one tainted blob, so template literals emitted *inside* a macro
  (a vision marker, a role header) get tainted and must be re-trusted by
  hand in the driver (`chat_jinja.vision_marker`). That is a correctness
  hazard that lives outside the type system.
* two output mechanisms (shared stream vs private buffer) instead of one.

**(c) The static `_t` lattice is load-bearing for correctness.** The
transpiler computes a per-expression taint bit and bakes the choice of
`tpl_str()` vs `untrusted()` into the emitted code. Static analysis is
needed anyway for macro-parameter fixpoint propagation, but using it to
decide *runtime representation* means any analysis gap is a security bug.

**The goal of the rework:** taint travels with the **value**, output is a
**single event sink**, and macros are **ordinary functions that emit**.

## 3. Target architecture

### 3.1 Values: `tv.uc` (already committed, keep)

`tv.uc` implements two prototype-tagged value kinds on top of plain ucode
objects (verified semantics: `proto(v, p)` sets, `proto(v)` reads, property
lookup walks the prototype chain, `__tostring__` is consulted on the
prototype chain only, own writes shadow without mutating the parent):

| kind | shape | meaning |
|---|---|---|
| cell   | `{ value: v }` with proto `TAINT` | one opaque value that must tokenize as plain text |
| MIXED  | `{ runs: [{u, s}] }` with proto `MIXED` | a string with an internal trusted/tainted split |

Operations: `cat` (preserves the split), `trim`/`ltrim`/`rtrim` (cascade
across runs), `join`, `get`/`idx` (taint propagates through access chains),
`text` (visible text, drops taint), `truthy`, `eq`, `length_of`,
`mixed`, `runs`, `taint`, `raw`. `tests/taint_values.uc` covers this.

`tv.uc` is fine and is **not** part of the rework.

### 3.2 Sink: one output stream, no capture (new)

`utpl` installs one global sink. The compiled template — and every macro,
which are now plain functions — *prints* through it:

```uc
/* installed as global __utpl_out(v) */
emit(v):
    if type(v) == 'string':       run(TRUSTED, v)      # only template literals arrive as plain strings
    else if MIXED:                for r in runs(v): run(r.u, r.s)
    else if cell:                 run(TAINTED, text(v))
    else:                         run(TAINTED, str(v)) # numbers, containers: rendered then tainted

run(t, s):                        # O(1) amortized: adjacent same-taint runs merge
    if s == '': return
    if buf == null or cur_t != t: flush(); buf = struct.buffer(); cur_t = t
    buf.put('*', s)
```

The result of a render **is** `{ segs: [{t, s}], text }` — no marker
string, no capture, no second representation. `text` is spliced from the
segments at the end, so it always carries real values.

Emission sites in the generated code:

| Jinja | emitted |
|---|---|
| raw text | `__utpl_out("<literal>")` |
| `{{ x }}`, `x` provably context-free | `__utpl_out(<expr>)` |
| `{{ x }}`, `x` may carry context | `untrusted(<expr>)` → `emit(value)` |
| `{{ macro(a) }}` | `__utpl_out(macro(a))` — the call emits internally; its return value is `null`, which the sink skips |

`untrusted()` survives as a name (so the transpiler's taint-wrapped outputs
need no new syntax) but is now just "emit this value with its own trust".
`emit_void(call)` wraps macro calls used as statements when needed.

**Why plain strings mean TRUSTED and nothing else does.** The only way a
plain string reaches the sink is from emitted code (a literal). Every value
that came from the context has passed through `tv.get`/`tv.idx`/`tv.cat` and
is therefore a cell or MIXED — never a bare string. Making the *representation
carry the trust* is what removes class (c) above as a security dependency:
if the static analysis is wrong about an expression, the value is still a
cell, and the sink still taints it. Static analysis degrades to an
optimization (choosing the wrapper), not a guarantee.

### 3.3 Macros: functions that emit (new)

Delete `__mbuf`. A macro compiles to a template function whose body is
compiled exactly like the top level: text → `__utpl_out(literal)`,
`{{ v }}` → `untrusted(v)`. The macro call is an expression whose value is
`null`; its *output* already went into the stream at the call site, in the
right order, with the right per-run trust.

The one thing we give up is treating a macro's output as a value:

```jinja
{{ ("x" ~ m(a)) | upper }}     # m's output used as a string, then filtered
{% set s = m(a) %}             # assigned, tested, compared
{{ m(a).startswith("…") }}
```

Handling: **capturing macros, on demand, per macro.** During AST collection
we already know every macro and every use site. A macro whose calls appear
*only* in output position compiles as an emitting function (the common case:
all of Qwen3.8's and GLM-5.3's). A macro used as a value compiles as a
capturing function:

```
function m(a):
    saved = sink_capture_begin()      # divert the sink into a temp run list
    <body, compiled normally>
    return sink_capture_end(saved)    # MIXED value with the internal split intact
```

Because capture is a *sink-level* diversion, not a string, the captured
value is MIXED — the trust split survives, and `|trim` on it cascades
across runs (that is exactly what `tv.trim` does). So the reason for
`__mbuf` disappears, and (b) is fixed rather than worked around.

This also removes the wholesale-taint problem for a macro that is used as a
value: the driver's `vision_marker` re-trusting hack can go away, because
the marker the template emits inside the macro stays a trusted run inside
the MIXED value.

### 3.4 Sink re-entrancy (the one subtle part)

`capture_begin`/`capture_end` must nest, must not lose merge state, and
must restore the parent's partially-filled buffer. Contract:

* capture is a stack in the sink; `begin` pushes the current
  `{buf, t}` and starts a fresh one; `end` pops, materializes the captured
  runs into a MIXED value, and leaves the parent buffer exactly as it was;
* a capture that is aborted by a template exception must not corrupt the
  outer render. `render_tpl` catches, resets the sink, re-raises (same
  "next begin() resets" contract as today);
* `end` on an empty capture returns the empty MIXED value, not `''`, so
  callers never have to special-case it;
* capture depth is bounded (nested macro-as-value calls) — assert < 32 and
  raise, don't recurse unboundedly.

### 3.5 Registry, not `with`

ucode has no JS-style `with`, and ucode functions cannot be called with an
arbitrary `this`, so template locals are a **frame chain**:

```uc
begin(ctx):
    frame = tv.frame(root_frame)          # prototype-chained off the global helpers
    global.__utpl_ctx = frame
    root = tv_taint(ctx)                  # reading messages/messages[i] propagates taint
```

The preamble binds each free root with `let <name> = tv_get(root, "<name>");`
(each root name is a compile-time constant, so this is both a binding and a
whitelist). `{% set x = e %}` becomes `x = tv_wrap(e)` in the frame so a
tainted assignment stays tainted; `loop.index` and friends are looked up in
the frame **before** falling back to ucode's builtin `loop` (shadowing: the
frame write shadows the inherited builtin without mutating it).

`tv.frame/set/lookup/wrap` already exist and are tested.

## 4. What changes, file by file

| File | Change |
|---|---|
| `ucode/tv.uc` | none (may gain `tv.wrap` deep-conversion helper if capture needs it) |
| `ucode/utpl.uc` | replace `untrusted()`/`scan_markers()`/`segments(captured)`/`state.parts` with the sink (§3.2, §3.4); `begin()` sets up frame + root cell; `render_tpl(tpl, ctx)` calls the compiled function directly and finishes the sink; drop `core_render` string capture and `tpl_buffer` |
| `ucode/jinja2utpl.uc` | text nodes → `__utpl_out(lit)`; macro bodies compiled like top level; capture wrapper for value-position macros (§3.3); keep the `_t` analysis for wrapper choice + macro-param fixpoint |
| `ucode/chat_jinja.uc` | drop `vision_marker` re-trusting; render result already has correct per-run trust |
| `core.uc` | only if the render-result shape changes (it should not: still `{text, segments}`) |
| `tests/*` | expectations on segment *granularity* change (finer runs, no macro blobs); byte-exactness expectations must not change |

## 5. Migration plan (each step lands green)

1. **Test the sink in isolation.** `tests/tv_sink.uc`: nested capture,
   merge behavior, MIXED passthrough, exception abort. No template involved.
   (Existing template tests: `tests/utpl.uc`, `tests/utpl_template.uc`,
   `tests/chat_jinja.uc`, `tests/glm5_chat.uc`, `tests/chat_qwen35.uc`,
   `tests/chat_prompt.uc`; `tv.uc` coverage is `tests/taint_values.uc`
   (CTest `taint_values`). Confirm the current list with
   `ctest --test-dir build -N` before writing new tests.)
2. **Sink in `utpl` behind a flag.** Keep the old path; add
   `render_sink_tpl()`. Both produce the same `{text, segments}` for the
   fixture set. Now we have an A/B.
3. **Transpiler emits to the sink** for text + non-macro output. Macro
   bodies still use `__mbuf`. Run `chat_jinja`, `utpl_template`,
   `tv`, `chat_qwen35`, `chat_qwen35_template`.
4. **Macros emit.** Flip macros one model at a time (Qwen3.8, then GLM-5.3).
   Segment granularity changes → update expectations with the diff *shown*,
   not silently. Byte-exactness must stay exact.
5. **Value-position macros** get the capture wrapper; a test with
   `{{ (m(x))|trim }}` and `{% set s = m(x) % }` + `if s` (synthetic
   template, not a real checkpoint — those don't use macros as values, but
   the mechanism must be tested).
6. **Delete the old path** (`core_render` capture, `scan_markers`,
   `tpl_buffer`, `state.parts`), drop the flag, remove `vision_marker`.

Revert strategy: each step is one commit; steps 1–3 are additive, so
`git revert` of any single commit is safe.

## 6. Invariants this design must not break

* **No Python at runtime** (invariant 1). The transpiler is ucode; the
  oracle Python scripts stay in `tests/oracle/`.
* **Taint fails closed.** A value of unknown origin must be a cell/MIXED,
  never a bare string. Context objects entering the frame are wrapped
  (`tv.wrap_env`); the only bare strings in the system are template
  literals and internal control values.
* **`_t` analysis is an optimization, not the guarantee** (§3.2's last
  paragraph). Any change that makes a security property depend on the
  analysis being complete needs a written rationale here first.
* **Byte-exactness is measured against jinja2**, never against our own
  previous output. If the two disagree, jinja2 is right (see the oracle
  scripts) — and record the reasoning when they must not.
* **O(n) rendering.** No `+=` over the whole prompt; the sink buffer is the
  only accumulator, and segments hold whole runs (L85a `struct.buffer`
  semantics).
* Existing engine tests that consume `{text, segments}` keep working: the
  *shape* is fixed, the *granularity* is allowed to get finer.

## 7. Open questions

1. **Segment granularity vs. the prefix cache.** `diff_segments` /
   `common_prefix_segments` (in `utpl.uc`) compare segment lists. Finer
   runs from macros could change incremental-prompt comparison behavior.
   Decide: normalize (merge adjacent same-taint runs — already done by the
   sink) and rely on that, or compare by `(taint, text)` prefix instead of
   list position. Verify with `tests/*prefix*`.
2. **Are bare strings ever ambiguous?** After step 6 the answer must be no;
   the check is a test that walks a rendered prompt and asserts every
   trusted segment traces back to literal bytes in the template source.
3. **`{% raw %}`/`{% filter %}`/`{% include %}`** remain rejected loudly.
   Confirm none of the target checkpoints need them before we invest.
4. **`|e` / autoescape**: irrelevant for LLM prompts, currently rejected.
   Keep rejecting rather than silently no-op'ing.
5. **Performance target.** The sink trades one string concat per output for
   one sink call per output. Microbenchmark in step 2 (`examples/` or a
   test) — we care about a 4000-message render, which was 60 ms pre-buffer
   work.

## 8. Testing contract

Every step runs, at minimum:

```sh
cmake --build build -j$(nproc)
ctest --test-dir build -R '(tv|taint_values|utpl|utpl_template|chat_jinja|chat_qwen35|chat_qwen35_template|httpd)' --output-on-failure
```

`chat_jinja` (23 cases) is the oracle-backed one: **it must stay byte-exact
across the whole rework.** New mechanism ⇒ new test, in the same commit:
`sink` (step 1), macro-as-value capture (step 5), and a taint-safety case
proving a user string containing `<|im_start|>` still tokenizes as plain
text while the template's own marker does not — that last one is the test
this entire design exists to keep passing.
