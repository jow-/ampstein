// examples/call_policy_loop.uc — tool-call loop breaker for agent harnesses
// (ampstein --call-hook).
//
// A model that has a failing tool call in its context can fall into
// resending it: the same edit with the same stale text, or a handful of
// variants of it, each answered by the same error (measured on Qwen3.8-27B:
// ~190 consecutive `edit` calls, its own reasoning saying "I'm stuck in a
// loop" before every one of them). Telling the model in the next user turn
// works, slowly; this policy acts before the call leaves the server.
//
// The script runs once per completed tool call (up to --call-hook-max
// verdicts per request); the request state arrives as globals:
//   call      {name, arguments}   the call just completed (arguments: JSON text)
//   calls     [{name, arguments}] calls already completed in this response
//   history   [{name, arguments, result}] the conversation's earlier tool
//             calls, oldest first, with the text their tool message returned
//             (null: unanswered)
//   text, reasoning, tools, tokens, overrides (earlier verdicts of this request)
// and the return value is the verdict: a string drops the call and answers it
// with that text as its tool result, in the same response (formats with a
// tool_feedback(): Qwen, GLM; others see it as text in their turn); the model
// answers with a new call or a change of plan and the client sees only that.
// Anything else lets the call through. Only --call-hook-max verdicts per
// request are asked for: raise it when the script is a filter.
//
// Here: a call is a repeat when the immediately preceding calls to the same
// tool (a run of at least STREAK, or ONE_SHOT when this call is byte-identical
// to the last one) all got the same failure text.

const STREAK = 3;        // same tool, same failure, in a row (variants of the call)
const ONE_SHOT = 1;      // the call is identical to a failed call just before it

const FAILED = /(could not|couldn't|cannot|can't|failed|failure|error|not found|no such|invalid|denied|refused|exited with code [1-9]|rc=[1-9])/i;

function head(s, n) {
	s = trim(s ?? '');
	s = replace(s, /[\r\n]+/g, ' ');

	return (length(s) > n) ? substr(s, 0, n) + '...' : s;
}

let n = 0, last = null, same_call = false;

for (let i = length(history) - 1; i >= 0; i--) {
	let h = history[i];

	if (h.name != call.name || h.result == null || !match(h.result, FAILED))
		break;

	if (last == null)
		last = h.result;
	else if (h.result != last)
		break;

	if (n == 0)
		same_call = (h.arguments == call.arguments);

	n++;
}

if (n < (same_call ? ONE_SHOT : STREAK))
	return true;

let rounds = overrides + 1;

if (rounds == 1)
	return `\n[call guard] That \`${call.name}\` call was not sent. ${n} \`${call.name}\` call(s) in a row already ` +
		`failed with the same result ("${head(last, 120)}"); sending it ${same_call ? 'again' : 'in another variant'} ` +
		`fails the same way. Look at the current state first (read the file, list the directory), ` +
		`then change the approach.\n`;

if (rounds == 2)
	return `\n[call guard] Again not sent: \`${call.name}\` keeps failing with "${head(last, 120)}". ` +
		`Do not call \`${call.name}\` with this target again before you have re-read it with a different tool. ` +
		`If the file already has the content you want, report that and move on.\n`;

return `\n[call guard] Still looping on \`${call.name}\` (${n} identical failures). Stop calling it, ` +
	`summarise where things stand in plain text and ask the user how to proceed.\n`;
