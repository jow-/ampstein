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
// to the last one) all got the same failure. "The same failure" compares the
// failure's signature, not its bytes: a validation error echoes the arguments it
// received (a different script each time), and Qwen3.8-27B sent 205 `write`
// calls without `path` that way, each answered by an error that differed only
// in that echo.
//
// A second check needs no history: a call that lacks a parameter its tool
// requires (tool_defs, the request's tool definitions) is dropped at once, and
// when it carries numbered copies of its own parameters (`content`, `content2`,
// ... `content468`: the model keeps enumerating parameter names and never
// reaches `path`) the note says so. Extra parameters alone are harmless and
// pass; so do calls of tools the request does not define.
//
// Also: a `write` of what a recent `write` just put into another file of the same
// directory is dropped (the copy under a new name changes nothing).
//
// Also: a `write` whose whole content (minus shebang, comments, blanks) is one command
// running a file the file the previous call `write`-wrote (`bash /tmp/a.sh` into /tmp/a1.sh) is dropped,
// with a note to call `bash` directly; Qwen otherwise chains a1.sh, a2.sh, ...
//
// A third one: a `bash` call that sleeps longer than SLEEP_MAX seconds
// (`sleep 600`, `sleep 5m`, `sleep 1h 30m`, `sleep infinity`; also inside `bash -c '...'`
// and in a script the conversation wrote with `write` and now runs) is dropped, with a
// note to poll for long waits instead: a blocked call holds the whole conversation
// (and usually the harness's own timeout) for nothing.

const STREAK = 3;        // same tool, same failure, in a row (variants of the call)
const ONE_SHOT = 1;      // the call is identical to a failed call just before it
const SIG_MAX = 200;     // characters of a failure that identify it

const SLEEP_MAX = 120;   // seconds a single `sleep` in a bash command may last

const FAILED = /(could not|couldn't|cannot|can't|failed|failure|error|not found|no such|invalid|denied|refused|exited with code [1-9]|rc=[1-9])/i;

function head(s, n) {
	s = trim(s ?? '');
	s = replace(s, /[\r\n]+/g, ' ');

	return (length(s) > n) ? substr(s, 0, n) + '...' : s;
}

// The failure with its echoed payload removed: the first paragraph, capped.
function sig(s) {
	s = trim(s ?? '');

	let k = index(s, '\n\n');

	if (k >= 0)
		s = substr(s, 0, k);

	return (length(s) > SIG_MAX) ? substr(s, 0, SIG_MAX) : s;
}

// Parameter names that are another parameter's name plus a number.
function numbered_copies(arguments) {
	let a = null;

	try {
		a = json(arguments);
	}
	catch (e) {
		return [];
	}

	if (type(a) != 'object')
		return [];

	let seen = {}, out = [];

	for (let k in keys(a))
		seen[k] = true;

	for (let k in keys(a)) {
		let m = match(k, /^(.*[^0-9])([0-9]+)$/);

		if (m && seen[m[1]])
			push(out, k);
	}

	return out;
}

// Required parameters of the called tool that the call does not carry.
function missing_params(call, defs) {
	let a = null;

	try {
		a = json(call.arguments);
	}
	catch (e) {
		return null;
	}

	if (type(a) != 'object' || type(defs) != 'array')
		return null;

	for (let d in defs) {
		let f = d.function ?? d;

		if (f.name != call.name)
			continue;

		let req = f.parameters?.required;

		if (type(req) != 'array')
			return [];

		return filter(req, (k) => a[k] == null);
	}

	return null;
}

let dup = numbered_copies(call.arguments);
let miss = missing_params(call, tool_defs);

if (miss != null && length(miss) > 0) {
	let why = `is missing the required parameter(s) \`${join('`, `', miss)}\``;
	let extra = length(dup) ? ` It carried ${length(dup)} parameters named like copies of another one ` +
		`(\`${join('`, `', slice(dup, 0, 3))}\`${length(dup) > 3 ? ', ...' : ''}).` : '';

	return `\n[call guard] That \`${call.name}\` call was not sent: it ${why}.${extra} ` +
		`Send the tool's own parameters, each exactly once, with their real names.\n`;
}

// Longest `sleep` (seconds) in a shell command line; 0 when there is none.
// `sleep` in command position (after the start, `;`, `&`, `|`, `(`, `{`, a backtick,
// a newline or a keyword such as `do`), GNU arguments: numbers with an optional
// s/m/h/d suffix, summed; `infinity`. A `sleep` inside a quoted string after `echo`
// is not told apart from a command.
const SLEEP_RE = regexp('(^|[;&|({`\n]|[[:space:]](do|then|else|elif|time|exec|nohup|sudo|xargs|!))' +
	'[[:space:]]*sleep[[:space:]]+([^;&|)\n]*)');
const DUR_RE = regexp('^([0-9]+\\.?[0-9]*|\\.[0-9]+)([smhd]?)$');
const UNIT = { s: 1, m: 60, h: 3600, d: 86400, '': 1 };

function longest_sleep(cmd) {
	let longest = 0, m;

	while (type(cmd) == 'string' && (m = match(cmd, SLEEP_RE))) {
		let at = index(cmd, m[0]);
		let total = 0, any = false;

		for (let w in split(trim(m[3]), /[[:space:]]+/)) {
			let d = match(w, DUR_RE);

			if (w == 'infinity' || w == 'inf') {
				total = 1e18;
				any = true;
			}
			else if (d) {
				total += (+d[1]) * UNIT[d[2]];
				any = true;
			}
			else
				break;
		}

		if (any && total > longest)
			longest = total;

		cmd = substr(cmd, at + length(m[0]));
	}

	return longest;
}

// The sleeps of a command line: its own, those inside `sh -c '...'`, and those of
// the scripts it runs that the conversation itself wrote (`bash /tmp/wait.sh`,
// the content of the newest `write` to that path). The model hides a `sleep 420`
// in a script and runs the script; the command line alone shows nothing.
function command_sleep(cmd) {
	let longest = longest_sleep(cmd), from = null;

	if (type(cmd) != 'string')
		return [0, null];

	// bash -c '...' / sh -c "..."
	let m = match(cmd, /(^|[[:space:]])(ba|z|da)?sh[[:space:]]+-[a-z]*c[[:space:]]+['"](.*)$/s);

	if (m) {
		let t = longest_sleep(m[3]);

		if (t > longest)
			longest = t;
	}

	if (longest > SLEEP_MAX)
		return [longest, null];

	for (let w in split(cmd, /[[:space:];&|()<>'"`]+/)) {
		if (!match(w, /\.sh$/))
			continue;

		w = replace(w, /^\.\//, '');

		for (let i = length(history) - 1; i >= 0; i--) {
			let h = history[i], a = null;

			if (h.name != 'write')
				continue;

			try {
				a = json(h.arguments);
			}
			catch (e) {
				continue;
			}

			if (type(a?.path) != 'string' || (a.path != w && substr(a.path, -(length(w) + 1)) != '/' + w))
				continue;

			let t = longest_sleep(a.content);

			if (t > longest) {
				longest = t;
				from = a.path;
			}

			break;      // (the newest version of the script is the one that runs)
		}
	}

	return [longest, from];
}

if (call.name == 'bash') {
	let a = null;

	try {
		a = json(call.arguments);
	}
	catch (e) {
		a = null;
	}

	let r = (type(a) == 'object') ? command_sleep(a.command ?? a.cmd) : [0, null];
	let t = r[0];

	if (t > SLEEP_MAX) {
		let shown = (t >= 1e18) ? 'forever' : sprintf('%d s', t);
		let who = r[1] ? `the script ${r[1]} it runs sleeps for ${shown}` : `it sleeps for ${shown}`;

		return `\n[call guard] That \`bash\` call was not sent: ${who}, ` +
			`and a call that long blocks the conversation (and the harness's timeout may kill it). ` +
			`Do not wait with long sleeps. Poll instead: check the state now (log tail, process, ` +
			`exit status), sleep at most ${SLEEP_MAX} s or not at all, and come back with another ` +
			`call; a long wait is several short polls.\n`;
	}
}

// A `write` of content that one of the last REWRITE_WINDOW calls just wrote to
// another file of the same directory (`/tmp/poll.sh`, then `/tmp/poll2.sh`, the
// same script three seconds later): the second copy changes nothing, the model
// meant to run the first. The two-writes ban of --call-guard only stops the third.
const REWRITE_WINDOW = 6;
const REWRITE_MIN = 24;       // characters of content below which equal files are normal

function dir_of(p) {
	let m = match(p, /^(.*\/)?/);

	return m ? (m[0] ?? '') : '';
}

if (call.name == 'write') {
	let a = null;

	try {
		a = json(call.arguments);
	}
	catch (e) {
		a = null;
	}

	if (type(a) == 'object' && type(a.path) == 'string' && type(a.content) == 'string' &&
	    length(rtrim(a.content)) >= REWRITE_MIN) {
		let lo = length(history) - REWRITE_WINDOW;

		for (let i = length(history) - 1; i >= 0 && i >= lo; i--) {
			let h = history[i], b = null;

			if (h.name != 'write')
				continue;

			try {
				b = json(h.arguments);
			}
			catch (e) {
				continue;
			}

			if (type(b?.path) != 'string' || type(b?.content) != 'string' ||
			    rtrim(b.content) != rtrim(a.content) || dir_of(b.path) != dir_of(a.path))
				continue;

			// (the same path again is a harmless rewrite, it may even restore a file)
			if (b.path == a.path)
				continue;

			// (a write that failed did not create the file)
			if (h.result != null && match(h.result, FAILED))
				continue;

			return `\n[call guard] That \`write\` call was not sent: ${b.path} already holds exactly this ` +
				`content (written ${length(history) - i} call(s) ago). Writing it again under another ` +
				`name changes nothing. Use that file: run it, or read it, and go on from there.\n`;
		}
	}
}

// A `write` that only runs a file a recent `write` created: the model wrote
// /tmp/a.sh, meant to run it, and wrote /tmp/a1.sh holding `bash /tmp/a.sh`, then
// /tmp/a2.sh holding `bash /tmp/a1.sh`, ... (Qwen: the intent to run the script
// is carried out with the wrong tool). After dropping comments, shebang and blank
// lines the new file is one line, and its command is the file the immediately
// preceding call wrote. A multi-file write (a project's files) never matches: its
// contents are not a call of the file before it.
const RUNNERS = /^(exec|sudo|env|time|nohup|command|source|\.|(ba|z|da|k)?sh|-[A-Za-z-]+)$/;

// The one command a script consists of once its setup is taken away: comments, blank
// lines, and the `cd dir`, `set -e`, `export X=1` steps around it (`cd /x/y && bash a.sh`
// is the same wrapper as `bash /x/y/a.sh`). null when the script holds more than one
// command, or none.
const SETUP = /^(cd|pushd|popd|set|export|umask|true|:)$/;

function sole_command(content) {
	let cmd = null;

	for (let l in split(content, /\r?\n/)) {
		l = trim(l);

		if (l == '' || substr(l, 0, 1) == '#')
			continue;

		// (`a && b`, `a; b`, `(a)`: the parts of a line; a pipe or redirect stays with its command)
		for (let seg in split(l, /(&&|\|\||[;()])/)) {
			seg = trim(seg);

			if (seg == '' || match(split(seg, /[[:space:]]+/)[0], SETUP))
				continue;

			if (cmd != null)
				return null;

			cmd = seg;
		}
	}

	return cmd;
}

// The file such a command runs, with the shell around it taken off (`exec bash -x ./a.sh x`).
function runs_file(line) {
	for (let w in split(line, /[[:space:]]+/)) {
		if (match(w, RUNNERS))
			continue;

		w = replace(w, /^['"]|['";&|]+$|['"]$/g, '');

		return replace(w, /^\.\//, '');
	}

	return null;
}

if (call.name == 'write') {
	let a = null;

	try {
		a = json(call.arguments);
	}
	catch (e) {
		a = null;
	}

	let line = (type(a) == 'object' && type(a.content) == 'string') ? sole_command(a.content) : null;
	let target = line ? runs_file(line) : null;

	if (target) {
		let past = [ ...history, ...(calls ?? []) ];

		let h = length(past) ? past[length(past) - 1] : null, b = null;

		if (h?.name == 'write') {
			try {
				b = json(h.arguments);
			}
			catch (e) {
				b = null;
			}
		}

		if (type(b?.path) == 'string' && b.path != a.path &&
		    (b.path == target || substr(b.path, -(length(target) + 1)) == '/' + target) &&
		    !(h.result != null && match(h.result, FAILED))) {
			return `\n[call guard] That \`write\` call was not sent: ${a.path ?? 'the file'} would only contain ` +
				`\`${head(line, 80)}\`, and ${b.path} already exists. A file that just runs another file is not ` +
				`what you want: to run ${b.path}, call the \`bash\` tool with that command.\n`;
		}
	}
}

let n = 0, last = null, same_call = false;

for (let i = length(history) - 1; i >= 0; i--) {
	let h = history[i];

	if (h.name != call.name || h.result == null || !match(h.result, FAILED))
		break;

	if (last == null)
		last = sig(h.result);
	else if (sig(h.result) != last)
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
