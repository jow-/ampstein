// examples/gemma4_gen.uc — greedy generation with the Gemma 4 text model (eager decode), for bring-up.
//   AMP_GEMMA4_GPUS=n ucode ... examples/gemma4_gen.uc <model_dir> "<prompt>" [new_tokens] [max_seq]
import * as gemma4 from 'ampstein.gemma4';
import * as gemma4_tp from 'ampstein.gemma4_tp';
import { readfile } from 'fs';

let dir = ARGV[0], prompt = ARGV[1], n = +(ARGV[2] || 64), max_seq = +(ARGV[3] || 4096);
let tok = torch.tokenizer(dir);
let t0 = clock(true);
let m = gemma4.load(dir, {devices: +(getenv('AMP_GEMMA4_GPUS') || 1), prefill_rows: +(getenv('AMP_GEMMA4_CHUNK') || 1024)});

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

if (getenv('AMP_GEMMA4_TP')) {
	let ng = +(getenv('AMP_GEMMA4_GPUS') || 2), rk = [];

	for (let g = 0; g < ng; g++)
		push(rk, g);

	m = gemma4_tp.shard(m, {devices: rk, prefill_rows: +(getenv('AMP_GEMMA4_CHUNK') || 1024)});
}

printf('loaded in %.1f s\n', now() - (t0[0] + t0[1] / 1e9));

let cache = m.new_cache(max_seq);
if (getenv('AMP_GEMMA4_FILE'))
	prompt = readfile(getenv('AMP_GEMMA4_FILE'));

let text = getenv('AMP_GEMMA4_RAW') ? prompt : '<bos><|turn>user\n' + prompt + '<turn|>\n<|turn>model\n';
let ids = tok.encode(text, true);

printf('%d prompt tokens\n', length(ids));

if (getenv('AMP_GEMMA4_TWICE')) {
	// a warm-up pass (allocator, autotuning), then the timed one on a fresh cache
	let run0 = m.prefill_pipeline(cache, ids, {chunk: +(getenv('AMP_GEMMA4_CHUNK') || 1024)});

	while (run0.step(1)) {}

	torch.synchronize();
	cache = m.new_cache(max_seq);
}

let t1 = now();
let lg;

if (getenv('AMP_GEMMA4_PIPE')) {
	let run = m.prefill_pipeline(cache, ids, {chunk: +(getenv('AMP_GEMMA4_CHUNK') || 1024)});

	while (run.step(1)) {}

	printf('issued in %.2f s\n', now() - t1);
	lg = run.logits();
}
else
	lg = m.forward_chunk(ids, cache, 1);

torch.synchronize();
if (gemma4.HP)
	for (let k in sort(keys(gemma4.HP)))
		printf('  host %-8s %.2f s\n', k, gemma4.HP[k]);

printf('prefill %.2f s (%.0f tok/s)\n', now() - t1, length(ids) / (now() - t1));

let out = [];
let gr = getenv('AMP_GEMMA4_GRAPH') ? m.graphs(cache, [ 1 ]) : null;
let t2 = now();

for (let i = 0; i < n; i++) {
	let nx = lg.view([1, -1]).argmax(-1).tolist()[0];

	if ((nx == 1 || nx == 106) && !getenv('AMP_GEMMA4_NOSTOP'))
		break;

	push(out, nx);
	lg = gr ? gr.run([ nx ], cache.length).logits : m.forward_chunk([ nx ], cache, 1);
}

torch.synchronize();
if (gemma4.HP)
	for (let k in sort(keys(gemma4.HP)))
		printf('  host %-8s %.2f s\n', k, gemma4.HP[k]);

printf('decode %d tokens in %.2f s (%.1f tok/s)\n', length(out), now() - t2, length(out) / (now() - t2));
printf('%s\n', tok.decode(out));
