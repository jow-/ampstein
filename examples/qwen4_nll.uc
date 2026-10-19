// examples/qwen4_nll.uc — teacher-forced negative log-likelihood of real text on the last <score> positions of a long
// context, scored one token at a time through the CUDA-graph decode step (the path that writes and reads the KV cache
// the way serving does): the quality measure for the Q8 KV cache (AMP_QWEN4_KV=q8).
//   ucode ... examples/qwen4_nll.uc <model_dir> <tokens> <score> <file> [file ...]
import * as qwen4_tp from 'ampstein.qwen4_tp';
import * as math from 'math';
import { readfile } from 'fs';

let dir = ARGV[0], ntok = +ARGV[1], score = +ARGV[2];
let stages = +(getenv('AMP_QWEN4_STAGES') || 1);
let tok = torch.tokenizer(dir);
let text = '';

for (let i = 3; i < length(ARGV); i++)
	text += readfile(ARGV[i]) + '\n\n';

let ids = tok.encode(text, false);

while (length(ids) < ntok + 1)
	push(ids, ...slice(ids, 0, ntok + 1 - length(ids)));

ids = slice(ids, 0, ntok + 1);

let m = qwen4_tp.load(dir, {stages: stages, prefill_rows: 1024, devices: +(getenv('AMP_QWEN4_GPUS') || 0) || null});
let cache = m.new_cache(ntok + 64);
let g = m.graphs(cache, [ 1 ]);

m.reset(cache);
cache.length = 0;

let n = ntok - score;
let run = m.prefill_pipeline(cache, slice(ids, 0, n), {chunk: 1024});

while (run.step(1)) {}

// the scored positions n .. ntok - 1 predict ids[n + 1 .. ntok]
let nll = 0.0, hit = 0;

for (let p = n; p < ntok; p++) {
	let lg = g.run([ ids[p] ], p).logits.view([1, -1]).to('float32').cpu();
	let lp = lg.softmax(-1).log().gather(1, torch.from_array([ ids[p + 1] ], 'int64').view([1, 1])).view([-1]).tolist()[0];

	nll -= lp;

	if (lg.argmax(-1).tolist()[0] == ids[p + 1])
		hit++;
}

printf('KV=%s: %d tokens of context, NLL over the last %d positions: %.4f (perplexity %.3f), top-1 match %d/%d\n',
	getenv('AMP_QWEN4_KV') || 'fp16', ntok, score, nll / score, math.exp(nll / score), hit, score);
