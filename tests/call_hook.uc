// tests/call_hook.uc — the --call-hook machinery without a model: the history
// of a chat request's tool calls (toolhist.uc) and the loop-breaker policy
// (examples/call_policy_loop.uc) on the shapes of a real session: an agent
// resending a failing `edit` as five variants, each answered by the same error.
import { tool_history } from 'ampstein.toolhist';
import * as chat_qwen35 from 'ampstein.chat_qwen35';
import * as chat_glm5 from 'ampstein.chat_glm5';

let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

// --- history extraction
let msgs = [
	{role: 'user', content: 'go'},
	{role: 'assistant', content: null, tool_calls: [
		{id: 'a', type: 'function', function: {name: 'edit', arguments: '{"path": "x", "n": 1}'}},
		{id: 'b', type: 'function', function: {name: 'bash', arguments: {command: 'ls'}}}]},
	{role: 'tool', tool_call_id: 'a', content: 'Could not find the exact text'},
	{role: 'tool', tool_call_id: 'b', content: [{type: 'text', text: 'one'}, {type: 'text', text: 'two'}]},
	{role: 'assistant', content: 'x', tool_calls: [
		{id: 'c', type: 'function', function: {name: 'read', arguments: '{}'}}]}
];

let h = tool_history(msgs);

check('three calls in order', length(h) == 3 && h[0].name == 'edit' && h[1].name == 'bash' && h[2].name == 'read');
check('arguments are canonical JSON text', h[0].arguments == '{ "n": 1, "path": "x" }' ||
	h[0].arguments == '{ "path": "x", "n": 1 }', h[0].arguments);
check('object arguments are serialised', index(h[1].arguments, 'ls') >= 0, h[1].arguments);
check('result of a string tool message', h[0].result == 'Could not find the exact text');
check('result of a text-parts tool message', h[1].result == 'onetwo', h[1].result);
check('unanswered call has no result', h[2].result == null);
check('no messages', length(tool_history(null)) == 0 && length(tool_history([])) == 0);

// --- policy
let policy = loadfile(sourcepath(0, true) + '/../examples/call_policy_loop.uc');
const ERR = 'Could not find the exact text in /home/x/mb. The old text must match exactly.';


function failed_edits(n, variants) {
	let hh = [];

	for (let i = 0; i < n; i++)
		push(hh, {name: 'edit', arguments: sprintf('{"v": %d}', variants ? i % variants : 0), result: ERR});

	return hh;
}

function ask(call_, hist, ov, defs) {
	return call(policy, null, {call: call_, calls: [], history: hist, text: '', reasoning: '',
		tools: ['edit', 'bash'], tool_defs: defs ?? [], tokens: 10, overrides: ov ?? 0});
}

let c0 = {name: 'edit', arguments: '{"v": 0}'};

check('fresh conversation: let through', ask(c0, [], 0) === true);
check('a first failure is not a loop (new variant)', ask({name: 'edit', arguments: '{"v": 9}'}, failed_edits(1, 1), 0) === true);
check('identical resend of a failed call: dropped', type(ask(c0, failed_edits(1, 1), 0)) == 'string');
check('three failures, new variant: dropped',
	type(ask({name: 'edit', arguments: '{"v": 9}'}, failed_edits(3, 3), 0)) == 'string');
check('the five-variant cycle of the real session: dropped',
	type(ask({name: 'edit', arguments: '{"v": 2}'}, failed_edits(12, 5), 0)) == 'string');
check('another tool: let through', ask({name: 'bash', arguments: '{}'}, failed_edits(12, 5), 0) === true);

let ok_hist = [];

for (let i = 0; i < 6; i++)
	push(ok_hist, {name: 'edit', arguments: '{"v": 0}', result: 'Successfully replaced 1 block(s)'});

check('repeated successes are not a loop', ask(c0, ok_hist, 0) === true);

let mixed = failed_edits(2, 2);

push(mixed, {name: 'edit', arguments: '{"v": 5}', result: 'Successfully replaced 1 block(s)'});
check('a success in between resets the run', ask({name: 'edit', arguments: '{"v": 1}'}, mixed, 0) === true);

let a = ask(c0, failed_edits(4, 1), 0), b = ask(c0, failed_edits(4, 1), 1), c = ask(c0, failed_edits(4, 1), 2);

check('escalating notes differ', a != b && b != c && a != c);
check('the note names the tool and the error', index(a, 'edit') >= 0 && index(a, 'Could not find') >= 0, a);

// --- the pi `write` loop (2026-10-13): no `path`, `content2` ... copies, and a validation
// error that echoes the (different every time) payload
function echo_fail(i) {
	return {name: 'write', arguments: sprintf('{"content": "script %d"}', i),
		result: sprintf('Validation failed for tool "write":\n  - path: must have required properties path\n\n' +
			'Received arguments:\n{\n  "content": "script %d"\n}', i)};
}

let wh = [];

for (let i = 0; i < 3; i++)
	push(wh, echo_fail(i));

let wc = {name: 'write', arguments: '{"content": "script 9"}'};

check('echoed-payload failures: same failure, new variant: dropped', type(ask(wc, wh, 0)) == 'string');
check('echoed-payload failures: two are not yet a streak', ask(wc, slice(wh, 0, 2), 0) === true);

const DEFS = [{type: 'function', function: {name: 'write', parameters: {type: 'object',
	properties: {path: {type: 'string'}, content: {type: 'string'}}, required: ['path', 'content']}}}];
let numbered = {name: 'write', arguments: '{"content": "x", "content2": "", "content3": ""}'};
let with_path = {name: 'write', arguments: '{"path": "/p", "content": "x", "content2": ""}'};

check('missing required parameter: dropped without history', type(ask(numbered, [], 0, DEFS)) == 'string');
check('...the note names it and the copies', index(ask(numbered, [], 0, DEFS), '`path`') >= 0 &&
	index(ask(numbered, [], 0, DEFS), 'content2') >= 0);
check('extra numbered parameters with the required ones present: let through', ask(with_path, [], 0, DEFS) === true);
check('a tool the request does not define: let through', ask({name: 'nope', arguments: '{}'}, [], 0, DEFS) === true);
check('no definitions, three copies: let through', ask(numbered, [], 0) === true);

check('distinct parameters ending in digits are fine',
	ask({name: 'bash', arguments: '{"command": "ls", "line2": 1}'}, [], 0) === true);

// --- long sleeps in bash
function bash(cmd) { return {name: 'bash', arguments: sprintf('%J', {command: cmd})}; }

for (let c in ['sleep 600', 'sleep 5m', 'cd /x && make; sleep 130', 'sleep 1h 30m', 'sleep infinity',
		'while true; do tail -1 log; sleep 300; done', 'sleep 2.5m', 'x | y && sleep 0.5 && sleep 121'])
	check(`sleep dropped: ${c}`, type(ask(bash(c), [], 0)) == 'string');

for (let c in ['sleep 120', 'sleep 2m', 'sleep 30; tail log', 'sleep 1', 'sleep 0.5', 'echo "sleeping 600"',
		'sleepy 600', 'my-sleep 600', './sleep 600', 'grep sleep 600 notes.txt', 'ls', 'sleep $N'])
	check(`sleep let through: ${c}`, ask(bash(c), [], 0) === true);

check('the sleep note says to poll', index(ask(bash('sleep 600'), [], 0), 'Poll') >= 0);
check('another tool with a sleep is not touched', ask({name: 'write', arguments: '{"content": "sleep 600"}'}, [], 0) === true);

// --- the tool-result form of a dropped call (what the model reads instead)
let q = chat_qwen35.tool_feedback('NOTE', false);

check('qwen: ends the turn, answers as a tool message, opens the next turn',
	q == '<|im_end|>\n<|im_start|>user\n<tool_response>\nNOTE\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', q);
check('qwen: thinking opens a think block', substr(chat_qwen35.tool_feedback('N', true), -8) == '<think>\n');
check('glm: observation block and generation prompt',
	chat_glm5.tool_feedback('NOTE', true) == '<|observation|><tool_response>NOTE</tool_response><|assistant|><think>', chat_glm5.tool_feedback('NOTE', true));
check('glm: no thinking', substr(chat_glm5.tool_feedback('N', false), -23) == '<|assistant|><think></think>'
	|| substr(chat_glm5.tool_feedback('N', false), -28) == '<|assistant|><think></think>', chat_glm5.tool_feedback('N', false));

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('call_hook: ok\n');
