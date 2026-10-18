// examples/dsv41_nll.uc — teacher-forced negative log-likelihood of real text (files given on the command line) on the
// last <score> positions of a long context: the quality measure for KV cache quantization (AMP_DSV41_KV=fp16|q8|q8pool).
//   ucode ... examples/dsv41_nll.uc <model_dir> <tokens> <score> <file> [file ...]
import * as dsv41 from 'ampstein.dsv41';
import * as math from 'math';
import { readfile } from 'fs';

let dir = ARGV[0], ntok = +ARGV[1], score = +ARGV[2];
let tok = torch.tokenizer(dir);
let text = '';

for (let i = 3; i < length(ARGV); i++)
	text += readfile(ARGV[i]) + '\n\n';

let ids = tok.encode(text, true);

while (length(ids) < ntok + 1)
	push(ids, ...slice(ids, 0, ntok + 1 - length(ids)));

ids = slice(ids, 0, ntok + 1);

let m = dsv41.load(dir, {devices: 6, max_seq: ntok + 64, prefill_rows: 1024});
let cache = m.new_cache(ntok + 64);
let n = ntok - score;
let run = m.prefill_pipeline(cache, slice(ids, 0, n), {chunk: 1024});

while (run.step(1)) {}

// the scored positions n .. ntok - 1 predict ids[n + 1 .. ntok]
let nll = 0.0, hit = 0;
let pos = n;

for (let p = n; p < ntok; p += 32) {
	let e = min(p + 32, ntok);
	let lg = m.forward_chunk(slice(ids, p, e), cache, e - p).to('float32').cpu();   // (rows, vocab) for positions p .. e - 1
	let lse = lg.softmax(-1).log();
	let want = torch.from_array(slice(ids, p + 1, e + 1), 'int64');
	let lp = lse.gather(1, want.view([-1, 1])).view([-1]).tolist();
	let am = lg.argmax(-1).tolist();

	for (let i = 0; i < e - p; i++) {
		nll -= lp[i];

		if (am[i] == ids[p + 1 + i])
			hit++;
	}
}

printf('KV=%s: %d tokens of context, NLL over the last %d positions: %.4f (perplexity %.3f), top-1 match %d/%d\n',
	getenv('AMP_DSV41_KV') || 'fp16', ntok, score, nll / score, math.exp(nll / score), hit, score);
