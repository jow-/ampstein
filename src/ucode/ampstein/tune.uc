// ucode/tune.uc — live tunables (tuning sessions without a restart).
//
// AMP_TUNE=FILE names a JSON object of tunables, e.g. {"cpu_threads": 12, "swaps": 1, "cpu_pf_max": 512}. A
// tunable read through get() takes the file's value when the file sets it, else the caller's default; the file is
// checked at most every half second (one stat) and re-read when its mtime changes, so a get() in a hot loop costs a
// clock read. Without AMP_TUNE every get() returns its default. on(name, fn) calls fn(value) whenever the file
// changes the value (and once at registration when it is set): for tunables that live in native code.
//
// The tunables are documented where they are read (grep for `tune.get('`); a change is logged once.

import {readfile, stat} from 'fs';

const PATH = getenv('AMP_TUNE');

let vals = {}, mtime = -1, next = 0.0, hooks = {};

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

function reload() {
	let st = stat(PATH);

	if (!st || st.mtime == mtime)
		return;

	mtime = st.mtime;

	let v = null;

	try {
		v = json(readfile(PATH));
	}
	catch (e) {
		warn(`tune: ${PATH}: ${e.message}\n`);

		return;
	}

	if (type(v) != 'object') {
		warn(`tune: ${PATH}: expected a JSON object\n`);

		return;
	}

	for (let k in v)
		if (vals[k] !== v[k]) {
			warn(`tune: ${k} = ${v[k]}\n`);

			for (let fn in (hooks[k] || []))
				fn(v[k]);
		}

	vals = v;
}

function poll() {
	if (!PATH)
		return;

	let t = now();

	if (t < next)
		return;

	next = t + 0.5;
	reload();
}

export function get(name, dflt) {
	poll();

	return (vals[name] != null) ? vals[name] : dflt;
}

export function on(name, fn) {
	hooks[name] ??= [];
	push(hooks[name], fn);
	poll();

	if (vals[name] != null)
		fn(vals[name]);
}

export const active = PATH ? true : false;
