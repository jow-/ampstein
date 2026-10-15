// tests/prompt_hook.uc — the --prompt-hook example policy (examples/prompt_policy_compaction.uc)
// on a summarization request shaped like pi's: a session whose serialized
// conversation is mostly thinking.
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

let policy = loadfile(sourcepath(0, true) + '/../examples/prompt_policy_compaction.uc');

const CPT = 3;   // characters per token of the fake tokenizer

function rep(s, n) {
	let o = '';

	while (length(o) < n)
		o += s;

	return substr(o, 0, n);
}

function request(n_turns, thinking_chars, result_chars) {
	let parts = [];

	for (let i = 0; i < n_turns; i++) {
		push(parts, `[User]: step ${i}`);
		push(parts, '[Assistant thinking]: ' + rep('hmm ', thinking_chars));
		push(parts, '[Assistant]: doing it');
		push(parts, `[Assistant tool calls]: bash(command="ls ${i}")`);
		push(parts, '[Tool result]: ' + rep('line\n\n[x] ', result_chars));
	}

	return [
		{role: 'system', content: 'You are a context summarization assistant. Your task is to read a conversation.'},
		{role: 'user', content: [{type: 'text', text: '<conversation>\n' + join('\n\n', parts) +
			'\n</conversation>\n\nThe messages above are a conversation to summarize.'}]}
	];
}

function chars_of(msgs) {
	let n = 0;

	for (let m in msgs)
		n += length(type(m.content) == 'string' ? m.content : join('', map(m.content, (p) => p.text)));

	return n;
}

function ask(msgs, context, want, round) {
	let tokens = chars_of(msgs) / CPT;

	return call(policy, null, {messages: msgs, tools: [], tokens: tokens, max_tokens: want,
		context: context, limit: context - 1 - want, fits: tokens + want < context, round: round ?? 0});
}

// ordinary requests are left alone
check('not a summarization request', ask([{role: 'user', content: 'hello'}], 1000, 100) === null);
check('system only', ask([{role: 'system', content: 'x'}], 1000, 100) === null);

// a request that fits is left alone, so is a lookalike without the summarizer's system prompt
check('fits: no verdict', ask(request(2, 10, 10), 100000, 1000) === null);

let other = request(40, 4000, 300);

other[0].content = 'You are a helpful assistant.';
check('other system prompt', ask(other, 40000, 3000) === null);

// the update prompt, with a previous summary between the conversation and the prompt
let upd = request(40, 4000, 300);

upd[1].content[0].text = replace(upd[1].content[0].text, 'The messages above are a conversation to summarize.',
	'<previous-summary>\nold\n</previous-summary>\n\nThe messages above are NEW conversation messages to incorporate.');
check('update prompt with previous summary', type(ask(upd, 40000, 3000)) == 'array');

// thinking is what makes it too big: dropping it is enough
let req = request(40, 4000, 300);
let ctx = 40000, want = 3000;
let v = ask(req, ctx, want);

check('verdict is an array of messages', type(v) == 'array' && length(v) == 2);

let n = chars_of(v);

check('fits: ' + n / CPT + ' tokens + ' + want + ' <= ' + ctx, n / CPT + want < ctx);
check('no thinking left', index(v[1].content, 'Assistant thinking') < 0);
check('results untouched when the thinking was enough', index(v[1].content, 'more characters omitted') < 0);
check('system message kept', v[0].content == req[0].content);
check('prompt tail kept', index(v[1].content, 'The messages above are a conversation') > 0);
check('request not mutated', index(req[1].content[0].text, 'Assistant thinking') > 0);

// long tool results go next
req = request(40, 400, 6000);
v = ask(req, 30000, 2000);
n = chars_of(v);

check('tool results cut: ' + n / CPT + ' tokens', type(v) == 'array' && n / CPT + 2000 < 30000);
check('results show the cut', index(v[1].content, 'more characters omitted') > 0);

// still too big: the oldest entries go, the newest stay
req = request(400, 4000, 3000);
v = ask(req, 20000, 2000);
n = chars_of(v);

check('oldest entries out: ' + n / CPT + ' tokens', type(v) == 'array' && n / CPT + 2000 < 20000);
check('says so', index(v[1].content, 'entries of the conversation omitted') > 0);
check('the newest step is there', index(v[1].content, 'ls 399') > 0);
check('the oldest step is there', index(v[1].content, 'step 0') > 0);

// a retry (the estimate was off) squeezes harder
let v0 = ask(req, 20000, 2000, 0), v2 = ask(req, 20000, 2000, 2);

check('later rounds keep less', chars_of(v2) < chars_of(v0));

// nothing to gain: no verdict
check('too small a budget', ask(request(2, 10, 10), 100, 90) === null);

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('prompt_hook: ok\n');
