// examples/dsv41_spark.uc — acceptance of the DSpark drafter on a greedy continuation (teacher forced).
//   ucode ... examples/dsv41_spark.uc <model_dir> [new tokens] [prompt file] [gpus]
// Prefills the prompt, greedy-decodes N tokens (feeding the drafter's main rows on the way), then drafts
// from every position of the continuation and counts how many drafts match the tokens that really follow.
import * as dsv41 from 'ampstein.dsv41';
import * as spark from 'ampstein.dsv41_spark';
import { readfile } from 'fs';

let dir = ARGV[0], N = +(ARGV[1] || 60), pfile = ARGV[2], gpus = +(ARGV[3] || 6);
let tok = torch.tokenizer(dir);
let text = pfile ? readfile(pfile) :
	'<｜begin▁of▁sentence｜><｜User｜>Write a short story about a lighthouse keeper who discovers a message in a bottle. ' +
	'Then explain, step by step, how a binary search works and give a Python implementation.<｜Assistant｜></think>';
let ids = tok.encode(text, true);
let m = dsv41.load(dir, {devices: gpus, max_seq: length(ids) + N + 64, prefill_rows: 512});
let sp = spark.load(m, {devs: [ 2, 3, 4 ]});
let cache = m.new_cache(length(ids) + N + 64);

printf('prompt: %d tokens\n', length(ids));

let lg = null;

for (let p = 0; p < length(ids); p += 512) {
	let e = min(p + 512, length(ids));

	lg = m.forward_chunk(slice(ids, p, e), cache, 1, (taps, pos) => sp.update_rows(taps, pos));
}

let seq = [ lg.to('float32').argmax(-1).tolist()[0] ];       // seq[i] is the token at position length(ids) + i
let base = length(ids);

for (let i = 0; i < N + 5; i++) {
	lg = m.forward_chunk([ seq[i] ], cache, 1, (taps, pos) => sp.update_rows(taps, pos));
	push(seq, lg.to('float32').argmax(-1).tolist()[0]);
}

printf('continuation: %s\n', tok.decode(seq));

let hist = [ 0, 0, 0, 0, 0, 0 ], total = 0, sumconf = [ 0, 0, 0, 0, 0 ];
let graphs = !getenv('AMP_DSPARK_NO_GRAPH');

if (graphs)
	sp.build_graphs();

let bad = 0;

for (let i = N - 60 > 0 ? N - 60 : 0; i < N; i++) {
	let r = sp.draft_eager(seq[i], base + i);

	if (graphs && base + i >= 128) {
		let r2 = sp.draft_graph(seq[i], base + i);
		let dc = 0;

		for (let k = 0; k < 5; k++)
			dc = max(dc, r.conf[k] > r2.conf[k] ? r.conf[k] - r2.conf[k] : r2.conf[k] - r.conf[k]);

		if (join(',', r.drafts) != join(',', r2.drafts) || dc > 0.02) {
			bad++;
			printf('MISMATCH at %d: eager %J graph %J (conf diff %.3f)\n', base + i, r.drafts, r2.drafts, dc);
		}
	}
	let a = 0;

	while (a < 5 && r.drafts[a] == seq[i + 1 + a])
		a++;

	hist[a]++;
	total++;

	for (let k = 0; k < 5; k++)
		sumconf[k] += r.conf[k];

	if (i % 10 == 0)
		printf('pos %d: drafts %s | truth %s | conf %.2f %.2f %.2f %.2f %.2f\n', base + i, tok.decode(r.drafts),
			tok.decode(slice(seq, i + 1, i + 6)), r.conf[0], r.conf[1], r.conf[2], r.conf[3], r.conf[4]);
}

let mean = 0;

for (let a = 0; a <= 5; a++)
	mean += a * hist[a];

printf('accepted-prefix histogram (0..5): %J over %d positions, mean accepted %.2f\n', hist, total, mean / total);
printf('mean confidence per draft: %.2f %.2f %.2f %.2f %.2f\n', ...map(sumconf, (c) => c / total));

if (graphs)
	printf('graph vs eager draft: %d mismatching positions of %d\n', bad, total);

if (graphs) {
	function now() {
		let c = clock(true);

		return c[0] * 1000.0 + c[1] / 1e6;
	}

	let p = base + N - 1, t0 = now();

	for (let i = 0; i < 20; i++)
		sp.draft_graph(seq[N - 1], p);

	torch.synchronize();
	printf('draft_graph: %.1f ms/round\n', (now() - t0) / 20);

	t0 = now();

	for (let i = 0; i < 20; i++)
		sp.draft_eager(seq[N - 1], p);

	torch.synchronize();
	printf('draft_eager: %.1f ms/round\n', (now() - t0) / 20);
}

if (graphs) {
	function now2() {
		let c = clock(true);

		return c[0] * 1000.0 + c[1] / 1e6;
	}

	function tm(name, f) {
		f();
		torch.synchronize();

		let t0 = now2();

		for (let i = 0; i < 20; i++)
			f();

		torch.synchronize();
		printf('  %-10s %.2f ms\n', name, (now2() - t0) / 20);
	}

	for (let b = 0; b < 3; b++)
		tm('stage ' + b, () => sp.gs[b].replay());

	tm('chain', () => sp.gchain.replay());
	tm('lm_head', () => m.lm_head.forward(sp.gio.xn.cuda(sp.ghd)));
}
