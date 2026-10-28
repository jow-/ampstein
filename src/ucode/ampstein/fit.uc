// ucode/fit.uc — `ampstein --survey --model DIR [options]`: find the launch line that fits the GPUs.
//
// GGUF Qwen3.8-Flash-Next only (the model whose routed experts are the filler of the VRAM). Everything but the experts has
// a size fixed by the options (context, KV mode, chunk, MTP): the weights of the backbone, lm_head and the draft head, the
// KV and GDN state, the graphs, the turn checkpoint and the warm-up's working set. So the survey
//
//   1. runs the server as a child process (--fit-probe: it stops after the warm-up) with the experts at their floor
//      (one per layer) and reads the free memory of every GPU,
//   2. fills every GPU with the experts of its layers up to that free memory minus the margin (--survey-margin),
//      a fraction per GPU (--expert-resident A,B),
//   3. checks the result with a second probe (the same load, now with the experts in place), and raises the margin when
//      the verification leaves less free than asked.
//
// When the option set as given does not fit (the child dies, or the experts would get less than --survey-min-resident
// of the experts), it steps down, cheapest loss first: --chunk, then --cache-mode (FP16 -> Q8), then --mtp, then --ctx.
// A child process per attempt: an out-of-memory in libtorch only ends that attempt. The result is a launch line on stdout,
// the progress goes to stderr.

import {popen, readfile, readlink, unlink, stat} from 'fs';

const MiB = 1048576;
const MARGIN_MIN = 256 * MiB;
const MAX_RAISES = 3;

// options the survey sets itself (value flags), and its own flags
const OWN = {'--survey': 0, '--fit-probe': 0, '--survey-margin': 1, '--survey-min-resident': 1};

function quote(s) {
	return "'" + replace('' + s, "'", "'\\''") + "'";
}

/* argv without the survey's own flags and without the flags named in `drop` (each takes a value) */
function strip(argv, drop) {
	let out = [];

	for (let i = 0; i < length(argv); i++) {
		let k = argv[i];

		if (k in OWN) {
			i += OWN[k];
			continue;
		}

		if (drop && k in drop) {
			i++;
			continue;
		}

		push(out, k);
	}

	return out;
}

function flags_of(over) {
	let out = [];

	for (let k in keys(over))
		push(out, k, '' + over[k]);

	return out;
}

/* One child: {gpus: [{free, total}], experts: [{layers, all, resident}]} or null (the child died / printed nothing) */
function probe(self, argv, over, tag) {
	let log = sprintf('%s/ampstein-survey-%d-%s.log', getenv('TMPDIR') ?? '/tmp', self.t0, tag);
	let cmd = join(' ', map([ self.interp, self.script, ...strip(argv, { ...over, '--expert-resident': 1 }),
		...flags_of(over), '--fit-probe' ], quote)) +
		' 2>' + quote(log);
	let t0 = time();
	let p = popen(cmd, 'r');
	let res = null;

	for (let line = p.read('line'); line != null; line = p.read('line')) {
		if (substr(line, 0, 8) == 'AMP_FIT ') {
			try {
				res = json(substr(line, 8));
			}
			catch (e) {
				res = null;
			}
		}
	}

	let rc = p.close();

	warn(sprintf('survey: probe %s %s in %ds (log %s)\n', tag, res ? 'ok' : `failed (exit ${rc})`, time() - t0, log));

	if (res && stat(log))
		unlink(log);

	return res;
}

/* the fraction of every GPU's experts that fits: [f0, f1, ...] (1.0 for a GPU without experts), or null */
function fractions(p, margin) {
	let out = [];

	for (let d = 0; d < length(p.gpus); d++) {
		let e = p.experts[d];

		if (!e) {
			push(out, 1.0);
			continue;
		}

		let f = (e.resident + (p.gpus[d].free - margin)) * 1.0 / e.all;

		out[d] = (f >= 1.0) ? 1.0 : (f < 0.01 ? 0.0 : int(f * 100) / 100.0);
	}

	return out;
}

function min_free(p) {
	let m = null;

	for (let g in p.gpus)
		if (m == null || g.free < m)
			m = g.free;

	return m;
}

function ladder(args) {
	let cur = {'--ctx': args.ctx, '--chunk': args.chunk, '--cache-mode': args.cache_mode ?? 'FP16', '--mtp': args.mtp},
	    out = [ { ...cur } ];

	while (cur['--chunk'] > 1024) {
		cur['--chunk'] = max(1024, int(cur['--chunk'] / 2));
		push(out, { ...cur });
	}

	if (cur['--cache-mode'] != 'Q8') {
		cur['--cache-mode'] = 'Q8';
		push(out, { ...cur });
	}

	while (cur['--mtp'] > 0) {
		cur['--mtp'] = int(cur['--mtp'] / 2);
		push(out, { ...cur });
	}

	while (cur['--ctx'] > 16384) {
		cur['--ctx'] = max(16384, int(cur['--ctx'] * 3 / 4 / 1024) * 1024);
		push(out, { ...cur });
	}

	return out;
}

function describe(rung) {
	return join(' ', map(sort(keys(rung)), (k) => k + ' ' + rung[k]));
}

function fmt(f) {
	return join(',', map(f, (x) => sprintf('%.2f', x)));
}

/* Place the experts for one set of options: {rung, fractions} when it fits, else {rung, fractions, why}. */
function fit(self, argv, args, rung, margin, min_res) {
	warn(`survey: trying ${describe(rung)}\n`);

	let floor = probe(self, argv, rung, 'floor');

	if (!floor)
		return {rung: rung, why: 'does not load / warm up even with one resident expert per layer'};

	if (!floor.experts)
		return {rung: rung, why: 'the model has no host-resident experts to place (--survey is for GGUF Qwen3.8-Flash-Next)'};

	let m = margin;

	for (let n = 0; n <= MAX_RAISES; n++) {
		let f = fractions(floor, m);
		let lo = min(...f);

		warn(sprintf('survey: margin %d MiB -> --expert-resident %s\n', m / MiB, fmt(f)));

		if (lo < min_res)
			return {rung: rung, fractions: f, why: sprintf('only %.2f of the experts fit (wanted %.2f)', lo, min_res)};

		let v = probe(self, argv, { ...rung, '--expert-resident': fmt(f) }, 'verify');

		if (v) {
			let left = min_free(v);

			warn(sprintf('survey: verified, %d MiB free on the fullest GPU\n', left / MiB));

			if (left >= margin / 2)
				return {rung: rung, fractions: f, free: left};

			m += (margin - left) + 256 * MiB;
		}
		else {
			m += 512 * MiB;
		}
	}

	return {rung: rung, why: 'the placement did not verify (the working set grew with the experts in place)'};
}

export function run(args, self) {
	self.interp = readlink('/proc/self/exe') ?? 'ucode';
	self.t0 = time();

	let argv = self.argv;
	let margin = (args.survey_margin ?? 1024) * MiB;
	let min_res = args.survey_min_resident ?? 0.3;

	if (margin < MARGIN_MIN)
		margin = MARGIN_MIN;

	let best = null;

	for (let rung in ladder(args)) {
		let r = fit(self, argv, args, rung, margin, min_res);

		if (!r.why) {
			let line = join(' ', map([ 'ampstein', ...strip(argv, { ...rung, '--expert-resident': 1 }), ...flags_of(rung),
				'--expert-resident', fmt(r.fractions) ], (x) => (match(x, /^[A-Za-z0-9_\/.,:=+-]+$/)) ? x : quote(x)));

			warn(sprintf('survey: fits, %d MiB free on the fullest GPU\n', r.free / MiB));
			print(line + '\n');

			return 0;
		}

		warn(`survey: ${describe(rung)}: ${r.why}\n`);

		if (r.fractions && (!best || min(...r.fractions) > min(...best.fractions)))
			best = r;
	}

	warn('survey: nothing fits' + (best ? sprintf('; the closest was %s with --expert-resident %s', describe(best.rung),
		fmt(best.fractions)) : '') + '\n');

	return 1;
}
