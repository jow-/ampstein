// tests/callguard.uc — the constrained tool-call guard (--call-guard) on token streams, no model.
import * as callguard from 'ampstein.callguard';

let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

const TOOLS = [
	{type: 'function', function: {name: 'write', parameters: {type: 'object',
		properties: {path: {type: 'string'}, content: {type: 'string'}}, required: ['path', 'content']}}},
	{type: 'function', function: {name: 'bash', parameters: {type: 'object',
		properties: {command: {type: 'string'}, timeout: {type: 'number'}}, required: ['command']}}},
	{type: 'function', function: {name: 'edit', parameters: {type: 'object',
		properties: {path: {type: 'string'}, edits: {type: 'array'}}, required: ['path', 'edits']}}}
];

// split a string into pieces of n characters, as a stand-in for tokens
function pieces(s, n) {
	let out = [];

	for (let i = 0; i < length(s); i += n)
		push(out, substr(s, i, n));

	return out;
}

// the first violating piece index of a stream, feeding each accepted piece; -1 when clean
function run(g, text, n) {
	let ps = pieces(text, n);

	for (let i = 0; i < length(ps); i++) {
		if (g.veto([], ps[i]))
			return i;

		g.feed(ps[i]);
	}

	return -1;
}

const HEAD = 'ok.\n<tool_call>\n<function=write>\n';
const GOOD = HEAD + '<parameter=path>\n/tmp/x\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>';

for (let n in [1, 2, 3, 7, 16])
	check(`clean call, ${n}-char pieces`, run(callguard.qwen(TOOLS, false), GOOD, n) == -1);

check('no tool definitions: no guard', callguard.qwen([], false) == null && callguard.qwen(null, false) == null);

// content before path: vetoed at the name
let g = callguard.qwen(TOOLS, false);
let ps = pieces(HEAD + '<parameter=content>', 1);
let bad = run(g, HEAD + '<parameter=content>', 1);

check('content before path: vetoed inside the name', bad >= 0 && bad < length(ps), bad);

// the real failure: content first (when allowed), then a numbered copy
g = callguard.qwen(TOOLS, false);
g.feed(HEAD + '<parameter=path>\n/x\n</parameter>\n<parameter=content>\nbody\n</parameter>\n');
check('content2 after content: vetoed', g.veto([], '<parameter=content2>') == true);
check('closing with everything given: allowed', g.veto([], '</function>') == false);

// closing without path
g = callguard.qwen(TOOLS, false);
g.feed(HEAD);
check('close before any parameter: vetoed', g.veto([], '</function>') == true);
check('close as the piece "</": vetoed', g.veto([], '</') == true);
check('"<" is not (could be <parameter=)', g.veto([], '<') == false);
check('path opener: allowed', g.veto([], '<parameter=path>') == false);
check('path opener in two pieces: allowed', g.veto(['<parameter='], 'path>') == false);
check('a name that is only a prefix of path: allowed', g.veto(['<parameter='], 'pa') == false);
check('a wrong name: vetoed', g.veto(['<parameter='], 'command>') == true);
check('numbered copy: vetoed', g.veto(['<parameter=', 'path'], '2>') == true);

// after path the bulk parameter opens
g = callguard.qwen(TOOLS, false);
g.feed(HEAD + '<parameter=path>\n/x\n</parameter>\n');
check('after path: content allowed', g.veto([], '<parameter=content>') == false);
check('after path: path again vetoed', g.veto([], '<parameter=path>') == true);
check('after path: close still vetoed (content missing)', g.veto([], '</function>') == true);

// optional parameters and non-bulk tools
g = callguard.qwen(TOOLS, false);
g.feed('<tool_call>\n<function=bash>\n');
check('bash: command allowed', g.veto([], '<parameter=command>') == false);
check('bash: unknown parameter vetoed', g.veto([], '<parameter=cmd>') == true);
g.feed('<parameter=command>\nls\n</parameter>\n');
check('bash: optional timeout allowed', g.veto([], '<parameter=timeout>') == false);
check('bash: close allowed', g.veto([], '</function>') == false);

// unknown tool name
g = callguard.qwen(TOOLS, false);
g.feed('<tool_call>\n<function=');
check('unknown tool name: vetoed', g.veto([], 'command>') == true);
check('known tool prefix: allowed', g.veto([], 'wr') == false);
check('tool name continuing wrongly: vetoed', g.veto(['wr'], 'ong') == true);

// values are free text, even if they look like tags
g = callguard.qwen(TOOLS, false);
g.feed(HEAD + '<parameter=path>\n/x\n</parameter>\n<parameter=content>\n');
check('value text is not constrained', g.veto([], '</function> <parameter=zzz>') == false);
check('value ends, then a bad opener in one piece: vetoed',
	g.veto([], 'x\n</parameter>\n<parameter=path>') == true);

// reasoning first; a tool_call mentioned in thought does not start the guard until </think>
g = callguard.qwen(TOOLS, true);
check('inside thinking, a quoted call is not checked', run(g, 'I will emit <tool_call>\n<function=nope> next', 1) == -1);
g.feed('</think>\n\n');
check('after thinking, the guard is live', g.veto([], '<tool_call>\n<function=nope>') == true);

// state between calls: a second call is checked from scratch
g = callguard.qwen(TOOLS, false);
g.feed(GOOD + '\n');
g.feed('<tool_call>\n<function=write>\n');
check('second call starts clean: close vetoed', g.veto([], '</function>') == true);

// a window of pending pieces counts (speculative drafts)
g = callguard.qwen(TOOLS, false);
g.feed('<tool_call>\n<function=write>\n');
check('draft window: prefix then a violating piece', g.veto(['<parameter=', 'path>', '\nx\n</parameter>\n', '<parameter=', 'content>', '\ny\n</parameter>\n'], '<parameter=path>') == true);
check('draft window: prefix then a legal piece', g.veto(['<parameter=', 'path>', '\nx\n</parameter>\n'], '<parameter=content>') == false);

// when the schema cannot be satisfied the guard does not trap the model
g = callguard.qwen([{type: 'function', function: {name: 't', parameters: {properties: {a: {type: 'string'}}, required: ['b']}}}], false);
g.feed('<tool_call>\n<function=t>\n<parameter=a>\n1\n</parameter>\n');
check('unsatisfiable required parameter: nothing vetoed', g.veto([], '</function>') == false);

// fix(): the deterministic continuation after a refused position
g = callguard.qwen(TOOLS, false);
g.feed(HEAD);
check('fix at the start of a write: path first', g.fix([]) == '\n<parameter=path>\n' || g.fix([]) == '<parameter=path>\n', sprintf('%J', g.fix([])));
check('fix mid-name: the rest of path', (g.feed('<parameter=pa'), g.fix([])) == 'th>\n', sprintf('%J', g.fix([])));
g = callguard.qwen(TOOLS, false);
g.feed(HEAD + '<parameter=path>\n/x\n</parameter>\n');
check('fix after path: content', g.fix([]) == '<parameter=content>\n', sprintf('%J', g.fix([])));
g.feed('<parameter=content>\nbody\n</parameter>\n');
check('fix after everything: close the call', g.fix([]) == '</function>\n</tool_call>', sprintf('%J', g.fix([])));
check('fix uses pending pieces', callguard.qwen(TOOLS, false).fix([]) == null);
g = callguard.qwen(TOOLS, false);
g.feed('<tool_call>\n<function=wr');
check('fix in a tool name', g.fix([]) == 'ite>\n', sprintf('%J', g.fix([])));

// an invalid name releases the guard instead of vetoing everything after it
g = callguard.qwen(TOOLS, false);
g.feed('<tool_call>\n<function=zzz');
check('invalid name: guard let go', g.veto([], 'anything at all') == false);

// loop detection by repetition: the tool name is banned in this response
function hcall(name, a) { return {name: name, arguments: sprintf('%J', a)}; }

let E1 = hcall('edit', {path: '/x', edits: [{oldText: 'a', newText: 'b'}]});
let E2 = hcall('edit', {path: '/x', edits: [{oldText: 'a', newText: 'b\n'}]});
let E3 = hcall('edit', {path: '/x', edits: [{oldText: 'c', newText: 'd'}]});
let B1 = hcall('bash', {command: 'ls'});

check('one edit: no ban', length(callguard.repeated([E1])) == 0);
check('the same edit twice: ban edit', join(',', callguard.repeated([B1, E1, E1])) == 'edit');
check('two different edits: no ban', length(callguard.repeated([E1, E3])) == 0);
check('A B A B cycling: ban edit', join(',', callguard.repeated([E1, E2, E1, E2])) == 'edit');
check('three different edits in a row: no ban', length(callguard.repeated([E1, E3, E2, hcall('edit', {path: '/y'})])) == 0);
check('bash polling repeated: no ban', length(callguard.repeated([B1, B1, B1])) == 0);
check('a bash between breaks the cycle', length(callguard.repeated([E1, E2, B1, E1])) == 0);
let W5 = hcall('write', {path: '/tmp/stat5.sh', content: '#!/bin/bash\npgrep x\necho 5\n'});
let W6 = hcall('write', {path: '/tmp/stat6.sh', content: '#!/bin/bash\npgrep x\necho 6\n'});
let W7 = hcall('write', {path: '/tmp/stat7.sh', content: '#!/bin/bash\npgrep x\necho 7\n'});
let W8 = hcall('write', {path: '/home/x/other.sh', content: 'a different script\n'});
check('same write to numbered files, three in a row: ban write', join(',', callguard.repeated([W5, W6, W7])) == 'write');
check('two numbered scripts that differ in a digit of their content: no ban yet',
	length(callguard.repeated([W5, W6])) == 0);
check('a different script in between: no ban', length(callguard.repeated([W5, B1, W8, B1, W6])) == 0);
let SH = '#!/bin/bash\ncd /home/x || exit 1\ngrep -n foo bar | head\necho done\n';
function wr(path, content) { return hcall('write', {path: path, content: content}); }

check('same script, numbered path: ban write after two',
	join(',', callguard.repeated([B1, wr('/tmp/pyval.sh', SH), wr('/tmp/pyval2.sh', SH)])) == 'write');
check('same script, prefix + digit', join(',', callguard.repeated([wr('/tmp/a.sh', SH), wr('/tmp/a2.sh', SH)])) == 'write');
check('same script, other extension', join(',', callguard.repeated([wr('/tmp/a.sh', SH), wr('/tmp/a.txt', SH)])) == 'write');
check('same script, one letter changed', join(',', callguard.repeated([wr('/tmp/chk_a.sh', SH), wr('/tmp/chk_b.sh', SH)])) == 'write');
check('same path, same content again', join(',', callguard.repeated([wr('/tmp/a.sh', SH), wr('/tmp/a.sh', SH + '\n')])) == 'write');
check('same path, content changed: a revision, no ban',
	length(callguard.repeated([wr('/tmp/a.sh', SH), wr('/tmp/a.sh', SH + '# x\n')])) == 0);
check('numbered path, content nearly the same: no ban',
	length(callguard.repeated([wr('/tmp/a1.sh', SH), wr('/tmp/a2.sh', replace(SH, 'foo', 'bar'))])) == 0);
check('different scripts, similar names: no ban',
	length(callguard.repeated([wr('/tmp/step1.sh', SH), wr('/tmp/step2.sh', 'completely\ndifferent\ncontent\nhere\n')])) == 0);
check('same script, word suffix: ban', join(',', callguard.repeated([wr('/tmp/chk.sh', SH), wr('/tmp/chk_again.sh', SH)])) == 'write');
check('same script, another name in the directory: ban',
	join(',', callguard.repeated([wr('/tmp/pyval.sh', SH), wr('/tmp/verify_python.sh', SH)])) == 'write');
check('same script, unrelated directories: no ban',
	length(callguard.repeated([wr('/tmp/a.sh', SH), wr('/home/x/build/Makefile.inc', SH)])) == 0);
check('empty files in one directory: no ban',
	length(callguard.repeated([wr('/p/__init__.py', ''), wr('/p/py.typed', '')])) == 0);
check('short identical content, different names: no ban',
	length(callguard.repeated([wr('/p/a.txt', 'x\n'), wr('/p/b.txt', 'x\n')])) == 0);
check('same name in another directory: no ban', length(callguard.repeated([wr('/tmp/a.sh', SH), wr('/var/a.sh', SH)])) == 0);
check('a bash in between: no ban', length(callguard.repeated([wr('/tmp/a.sh', SH), B1, wr('/tmp/a2.sh', SH)])) == 0);
check('writing three different modules: no ban', length(callguard.repeated([
	wr('/src/mod_a.uc', 'let a = 1;\nexport function fa() {}\n'),
	wr('/src/mod_b.uc', 'import * as a from \'a\';\nexport function fb() { return a.fa(); }\n'),
	wr('/src/mod_c.uc', 'export const C = 3;\n')])) == 0);
check('no history', length(callguard.repeated(null)) == 0 && length(callguard.repeated([])) == 0);

g = callguard.qwen(TOOLS, false, ['edit']);
g.feed('<tool_call>\n<function=');
check('banned name: vetoed', g.veto([], 'edit>') == true);
check('banned name prefix: vetoed', g.veto([], 'ed') == true);
check('other names: allowed', g.veto([], 'read>') == false || g.veto([], 'bash>') == false);
check('bash still allowed', g.veto([], 'bash>') == false);
check('a banned name with no forced guess (the engine redraws)', g.fix([]) == null);
g = callguard.qwen([TOOLS[2]], false, ['edit']);
g.feed('<tool_call>\n<function=');
check('the only tool is never banned', g.veto([], 'edit>') == false);

// the conversation as the model sees it: stray arguments of earlier calls are dropped
let TOOLDEFS = [{type: 'function', function: {name: 'write', parameters: {type: 'object',
	properties: {path: {type: 'string'}, content: {type: 'string'}}, required: ['path', 'content']}}}];
let conv = [
	{role: 'user', content: 'go'},
	{role: 'assistant', content: null, tool_calls: [
		{id: 'a', type: 'function', function: {name: 'write', arguments: '{"content": "x", "content2": "", "content3": "", "path": "/p"}'}},
		{id: 'b', type: 'function', function: {name: 'bash', arguments: '{"command": "ls", "zzz": 1}'}},
		{id: 'c', type: 'function', function: {name: 'write', arguments: {content: 'y', path: '/q', content2: ''}}}]},
	{role: 'tool', tool_call_id: 'a', content: 'ok'}
];
let cl = callguard.clean_messages(conv, TOOLDEFS);

check('clean: stray arguments of a declared tool dropped (JSON text)',
	json(cl[1].tool_calls[0].function.arguments).content2 == null && json(cl[1].tool_calls[0].function.arguments).path == '/p',
	cl[1].tool_calls[0].function.arguments);
check('clean: object arguments too', cl[1].tool_calls[2].function.arguments.content2 == null && cl[1].tool_calls[2].function.arguments.path == '/q');
check('clean: a tool without declared properties is left alone', cl[1].tool_calls[1].function.arguments == '{"command": "ls", "zzz": 1}');
check('clean: the input is not modified', index(conv[1].tool_calls[0].function.arguments, 'content2') >= 0 && conv[1].tool_calls[2].function.arguments.content2 == '');
check('clean: other messages shared', cl[0] === conv[0] && cl[2] === conv[2]);
check('clean: nothing to drop returns the same array',
	callguard.clean_messages([{role: 'user', content: 'x'}], TOOLDEFS) != null &&
	callguard.clean_messages(null, TOOLDEFS) == null && callguard.clean_messages(conv, null) === conv);

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('callguard: ok\n');
