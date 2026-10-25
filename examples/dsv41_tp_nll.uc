// examples/dsv41_tp_nll.uc — teacher-forced negative log-likelihood of real text under dsv41_tp with host-resident
// experts: the quality cost of computing them on the CPU (int8 activations) against the GPU kernels reading the same
// experts from host memory. The scored positions run as decode-sized steps (16 rows: the fused path, where the CPU
// experts serve the misses). Prints the NLL, the top-1 match and the argmax ids (diff two runs for the flips).
//   AMP_NLL_RESIDENT=0.58,0.65,... AMP_NLL_PROFILE=profile.json [AMP_NLL_CPU=12] [AMP_NLL_IDS=out.json]
//   ucode ... examples/dsv41_tp_nll.uc <model_dir> <tokens> <score> <file> [file ...]
import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as math from 'math';
import { readfile, writefile } from 'fs';

let dir = ARGV[0], ntok = +ARGV[1], score = +ARGV[2];
let tok = torch.tokenizer(dir);
let text = '';

for (let i = 3; i < length(ARGV); i++)
	text += readfile(ARGV[i]) + '\n\n';

let ids = tok.encode(text, true);

while (length(ids) < ntok + 1)
	push(ids, ...slice(ids, 0, ntok + 1 - length(ids)));

ids = slice(ids, 0, ntok + 1);

let res = getenv('AMP_NLL_RESIDENT'), cpu = +(getenv('AMP_NLL_CPU') || 0);
let hx = res ? {frac: (index(res, ',') >= 0) ? map(split(res, ','), (x) => +x) : +res, profile: getenv('AMP_NLL_PROFILE'),
	swaps: 0, cpu: cpu} : null;
let m = dsv41_tp.load(dir, {devices: 6, stages: 3, max_seq: ntok + 64, prefill_rows: 1536, host_experts: hx});
let cache = m.new_cache(ntok + 64);
let n = ntok - score;
let run = m.prefill_pipeline(cache, slice(ids, 0, n), {chunk: 1536});

while (run.step(1)) {}

// the scored positions n .. ntok - 1 predict ids[n + 1 .. ntok]
let nll = 0.0, hit = 0, args = [];
const ROWS = 16;

for (let p = n; p < ntok; p += ROWS) {
	let e = min(p + ROWS, ntok);
	let lg = m.forward_chunk(slice(ids, p, e), cache, e - p).to('float32').cpu();   // (rows, vocab) for positions p .. e - 1
	let lse = lg.softmax(-1).log();
	let want = torch.from_array(slice(ids, p + 1, e + 1), 'int64');
	let lp = lse.gather(1, want.view([-1, 1])).view([-1]).tolist();
	let am = lg.argmax(-1).tolist();

	for (let i = 0; i < e - p; i++) {
		nll -= lp[i];
		push(args, am[i]);

		if (am[i] == ids[p + 1 + i])
			hit++;
	}
}

printf('host experts %s, CPU %d: %d tokens of context, NLL over the last %d positions: %.4f (perplexity %.3f), top-1 match %d/%d\n',
	res || 'none', cpu, ntok, score, nll / score, math.exp(nll / score), hit, score);

if (getenv('AMP_NLL_IDS'))
	writefile(getenv('AMP_NLL_IDS'), sprintf('%J', args));
