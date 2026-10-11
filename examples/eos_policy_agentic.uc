// examples/eos_policy_agentic.uc — end-of-turn policy for agent harnesses
// (ampstein --eos-hook).
//
// Some checkpoints (DeepSeek-V4-Flash at 3 bpw, measured) end a turn right
// after announcing an action with no tool call; an agent loop that treats a
// text-only turn as the final answer then stops. Two shapes are recognised
// at the end of the text:
//   - a trailing colon ("Let me find which test failed:")
//   - a closing sentence that opens with "Let me" / "Now let me" and ends
//     with a period ("Now let me rebuild and run the tests.")
// ("Let me know if ..." is a closing remark, not an announcement.) While no
// tool call has been made and tools are on offer, the end token is then
// discouraged: the position is redrawn with its logit lowered. The bias
// grows with each consultation (overrides), so a model that insists still
// gets through within --eos-hook-max rounds and a greedy request is not
// stuck on the same margin.
//
// The script runs once per sampled end token; the request state arrives as
// globals (text, reasoning, calls, tools, mode, tokens, overrides, finish)
// and the return value is the verdict: true = let the turn end, a number =
// soft penalty (logits), a string = text to force. See ampstein.

if (!length(tools) || length(calls) || mode != 'content')
	return true;

let announced = match(text, /:[ \t\r\n]*$/);

if (!announced) {
	// the last sentence: starts the text or follows a line break / sentence
	// end, and holds no sentence end of its own (a period inside a word,
	// "src/main.c", is not one)
	let m = match(text, /(^|[\n.!?][ \t]*)((now,?[ \t]+)?let me ([^\n.]|\.[^ \t\r\n])*\.)[ \t\r\n]*$/i);

	announced = m && !match(m[2], /let me know/i);
}

return announced ? 4 * (overrides + 1) : true;
