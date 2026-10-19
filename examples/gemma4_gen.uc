// examples/gemma4_gen.uc — greedy generation with the Gemma 4 text model (eager decode), for bring-up.
//   AMP_GEMMA4_GPUS=n ucode ... examples/gemma4_gen.uc <model_dir> "<prompt>" [new_tokens] [max_seq]
import * as gemma4 from 'ampstein.gemma4';
import { readfile } from 'fs';

let dir = ARGV[0], prompt = ARGV[1], n = +(ARGV[2] || 64), max_seq = +(ARGV[3] || 4096);
let tok = torch.tokenizer(dir);
let t0 = clock(true);
let m = gemma4.load(dir, {devices: +(getenv('AMP_GEMMA4_GPUS') || 1), prefill_rows: 1024});

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

printf('loaded in %.1f s\n', now() - (t0[0] + t0[1] / 1e9));

let cache = m.new_cache(max_seq);
if (getenv('AMP_GEMMA4_FILE'))
	prompt = readfile(getenv('AMP_GEMMA4_FILE'));

let text = getenv('AMP_GEMMA4_RAW') ? prompt : '<bos><|turn>user\n' + prompt + '<turn|>\n<|turn>model\n';
let ids = tok.encode(text, true);

printf('%d prompt tokens\n', length(ids));

let t1 = now();
let lg = m.forward_chunk(ids, cache, 1);

torch.synchronize();
printf('prefill %.2f s (%.0f tok/s)\n', now() - t1, length(ids) / (now() - t1));

let out = [];
let t2 = now();

for (let i = 0; i < n; i++) {
	let nx = lg.view([1, -1]).argmax(-1).tolist()[0];

	if (nx == 1 || nx == 106)
		break;

	push(out, nx);
	lg = m.forward_chunk([ nx ], cache, 1);
}

torch.synchronize();
printf('decode %d tokens in %.2f s (%.1f tok/s)\n', length(out), now() - t2, length(out) / (now() - t2));
printf('%s\n', tok.decode(out));
