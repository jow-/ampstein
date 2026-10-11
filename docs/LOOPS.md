# Loop detection and nudging

Status: design proposal (not implemented). Lives in the
server/generation layer, in pure ucode.

## Problem

Models sometimes get stuck repeating themselves. Two shapes matter:

- **Across messages:** the same assistant message again and again, or a cycle
  of up to five messages. In agentic use this is typically the same tool call
  hitting the same error.
- **Within one reply:** a phrase or paragraph repeating until `max_tokens`.
  This is the worse one for wasted time.

Interrupting with a nudge such as "You appear to be looping" usually lets the
model recover. The engine should do that automatically.

**Policy decisions:**

- Tool calls, including their arguments, are part of the fingerprint.
- The engine never stops generation because of a loop.
- **Exact repeats are disallowed.** An assistant message identical to one of
  the last N can't be emitted. Deliberate repetition (polling, retries) stays
  possible, but must carry a varying element, which the nudge asks for.
- A second, soft detector catches loops that merely vary a counter. It
  escalates nudges and never bans.

## Detection across messages

**Fingerprint** per assistant message: `[length, digest.fnv1a64(normalized)]`.
ucode's `digest` module provides `fnv1a64`, so no native code is needed.

Normalization:

- The visible content, with `<think>…</think>` removed (some clients send
  reasoning back) and whitespace collapsed.
- Plus each tool call as `name` + arguments re-serialized as sorted-key JSON.
  Tool-call IDs are dropped, because they differ every turn and would defeat
  exact matching.

**Tail periodicity.** Over the last N = 20 assistant messages, find the
smallest period p ∈ 1..5 and the largest repeat count r such that the last
p·r fingerprints are r copies of the same p-sequence. Messages from the user
or from tools between them are ignored: re-issuing the same call after the
same error is exactly the loop to catch.

**Thresholds:**

- p = 1 with a short message (under 16 characters) and no tool call needs
  r ≥ 3, so that "OK" / "Done" doesn't trigger.
- Everything else triggers at r ≥ 2.
- Deliberate retries of a tool call are therefore tolerated once (r = 2 gets
  only the mildest nudge).

**No server state.** The client resends the whole history with every request,
so everything above is computed from the request. Because only the tail
counts, a model that breaks out stops being nudged automatically. If it falls
back into the loop, r grows again, and that drives escalation.

## No exact repeats

**Rule:** the full fingerprint (text + canonical tool calls) of a new
assistant message must differ from those of the last N = 20. Short messages
with no tool call (under 16 characters, e.g. "OK") are exempt.

**Enforcement at the end of the message, not the start.** Banning a repeated
message's first tokens would also block legitimate messages that merely start
the same way. Instead:

1. Decode chunks already stop at a message terminator (EOS, `<|im_end|>`,
   `</tool_call>`), and the host checks the tail there.
2. At a terminator, decode the message so far, normalize it, and fingerprint
   it.
3. If the fingerprint matches one of the last N: roll back the terminator
   token (KV is a length reset; GDN restores the latest chunk-boundary
   snapshot and re-runs the few tokens since), mask that terminator for the
   next step, and resume. The model has to keep writing, and that
   continuation is where the varying element lands.
4. Cap the forced continuations (e.g. 3 per message). After that, apply the
   sampling escalation below instead, so that a model that can only re-emit
   the same text isn't spun forever.

**The nudge asks for the variation, and says where to put it.** Injected
when the history's tail shows a repeat (r ≥ 2, detection as above) and before
any forced continuation:

> Your last p messages repeat earlier ones. If you are repeating on purpose
> (for example waiting or polling), say so and include something that
> changes each time, such as "(attempt 3)", in your message text. Otherwise,
> try a different approach.

The variation belongs in the message **text**, not in the tool arguments.
Strict tool schemas may reject an extra field, and a counter in a path breaks
the call. Because the fingerprint covers the text and the tool call together,
a varying sentence before an unchanged call is enough. A `# attempt 3`
comment in a shell command is fine, but never required.

## Soft detector and escalation

Exact-repeat banning is easy to satisfy mindlessly: a model can loop forever
while incrementing a counter. A second fingerprint catches that:

- **Soft fingerprint:** only the canonical tool calls (or the text, for
  messages without tools), with digit runs normalized (`attempt 3` →
  `attempt #`) and whitespace collapsed.
- The same tail periodicity (p ∈ 1..5, repeat count r) on soft fingerprints.
- It never bans; it escalates:

| r (soft) | Nudge | Sampling |
| --- | --- | --- |
| 3 | "You have called `<tool>` with the same arguments r times. If this is intended waiting, continue; otherwise, change approach." | Unchanged |
| 5 | A firmer note listing the repeated call and its last result. For thinking models, also an assistant-side thought prefix: "I've repeated this step several times. Is it working?" | Repetition penalty on the repeated messages' tokens; small temperature bump (+0.2, capped) |
| ≥ 8 | The same note | Stronger penalty |

The thresholds are higher than for exact repeats, because legitimate polling
does produce long runs. Every step is capped, never terminal, and
configurable: N, thresholds, note texts, sampling deltas, the
forced-continuation cap, and an off switch per detector.

## Injection

- **Inject after rendering the template**, as the very last thing before the
  assistant generation header. The client's messages are never modified.
- **Prefix caching is preserved.** Reuse (`MEMORY.md`) compares token
  prefixes. The next request won't contain the nudge, but it still shares
  everything before it.
- **The turn-boundary GDN snapshot is taken before the nudge tokens**, so the
  saved state never includes text the client hasn't seen.
- **Visible to you, not to the model's history:** a log line, and a response
  header such as `x-Ampstein-loop: period=2 repeats=3 action=nudge+penalty`.

## Loops within one reply

**Detection.** Decode runs in chunks that already check the token tail for
EOS (`MEMORY.md`). Add a periodicity check on the same token ids: the last
tokens repeat with some period p ≤ 512 at least 3 times, over a minimum span
of 64 tokens. It's host-side work on integers, negligible next to the
decode step.

**Action, never a stop:**

1. **Preferred: roll back and nudge.** Truncate to the start of the first
   repetition (KV is just a length reset; GDN layers restore the snapshot of
   the chunk boundary at or before that point, then re-run the few tokens up
   to the cut), inject the nudge, and resume. This needs a GDN state snapshot
   at every chunk boundary, the same mechanism the MTP verify step already
   uses (`gdn_snap`). Keeping two or three ring-buffered snapshots bounds the
   memory (~150 MiB each on the 27B).
2. **Until snapshots exist: nudge in place.** Append the nudge to the stream
   without rollback. In a thinking model's reasoning block: "Wait, I'm
   repeating myself. Let me approach this differently." In visible output:
   apply only the sampling escalation, and don't inject text the client would
   show.
3. Escalate exactly as across messages: each further detection in the same
   reply moves one step up the table above.

## Prior art

Surveyed 2026-10-07. The exllamav3 entries come from the local checkout (`~/devel/exllamav3` at `d3739fd`, the same commit as `vendor/exl3`); vLLM and llama.cpp entries from their documentation and search results, not their source. "Not found" means not found, not proven absent.

| Idea | vLLM | llama.cpp | exllamav3 |
| --- | --- | --- | --- |
| Penalties against token recurrence (repetition, presence, frequency) | Yes | Yes | Yes |
| **DRY** (penalizes extending an exact repeat of earlier context, with sequence breakers that protect chat-template tokens) | Not found in core | **Yes**, in the default sampler chain | **Yes**: sampler presets, and `generator/dry.cu` in the vendored tree |
| Message-level fingerprints, nudge injection, exact-repeat ban | Not found | Not found | Not found |

**Consequence for this design:** DRY already covers much of "Loops within
one reply". It is published (it roughly halves the loop rate on large models
while preserving benchmark scores), tuned, and its kernel is already vendored.
Adopt it as the first line of defense within a reply, and keep the
detection-plus-rollback path for loops DRY misses. The message-level part
(fingerprints, nudges, exact-repeat ban, soft detector) is not found in any of
the three engines; it belongs to agent-harness territory, which suits an engine
that sees the whole conversation on every request.

## Tests

- Unit tests on fingerprint sequences: periods 1–5, near-misses, tool-ID
  changes that must not matter, argument changes that must, short messages,
  and digit-only variations that change the full fingerprint but not the soft
  one.
- Terminator masking: a prompt that reliably re-emits the previous message
  gets forced past its terminator, and the final message differs from the
  banned one.
- A replayed agent transcript with a known loop: the nudge appears at the
  right turn and disappears once the history stops repeating.
- In-reply: a forced repetition (seeded sampling on a prompt that reliably
  loops) gets detected within a bounded number of tokens, and the rollback
  restores state bit-exactly (compare logits after rollback against a fresh
  prefill of the same tokens).

## Open questions

- The wording of the nudges. Start with the texts above and refine them from
  logged cases.
- How models react to a masked terminator. Forced continuations might
  produce awkward trailing text rather than a useful variation; the nudge
  before generation should make that rare. Measure on logged cases.
- Soft-detector thresholds for long polling runs, e.g. a build that takes 20
  checks. A per-tool threshold override may be needed.
