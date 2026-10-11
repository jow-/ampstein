// examples/bench_decode.uc — qwen35 decode throughput: plain greedy vs
// MTP speculative decoding (model.generate_mtp, EAGLE-1 with the
// checkpoint's MTP head), on a real chat prompt rendered through the
// model's chat template (acceptance depends on content).
//
//   ucode -L build -L ucode examples/bench_decode.uc [tokens] [windows]
//
// Defaults: 256 generated tokens, windows "2,3,4". Reports decode-only
// tok/s (prefill time measured separately and subtracted) and checks
// that every MTP run reproduces the plain greedy output token for
// token. Model: $AMP_MODEL_QWEN35 or
// /data/models/Qwen3.8-27B-exl3-5.00bpw, layers 0-31 on cuda:0, 32-63
// on cuda:1.

let torch = require('ampstein.torch');
global.torch = torch;
import * as qwen35 from 'ampstein.qwen35';
import * as chat from 'ampstein.chat_qwen35';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const N = int(ARGV[0] || 256);
const WINDOWS = map(split(ARGV[1] || '2,3,4', ','), (w) => int(w));

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < 32) ? 0 : 1);

let m = qwen35.load(MODEL, { devices: devs, mtp: true });
let tok = torch.tokenizer(MODEL);

let r = chat.render([
	{ role: 'user', content: 'Write a short Python function that parses an ISO 8601 date ' +
		'string into a (year, month, day) tuple without using the datetime module, ' +
		'and explain how it handles invalid input.' }
], null, { enable_thinking: false });

let prompt = tok.encode(r.prefix + r.gen, true);

printf('prompt %d tokens, generating %d\n', length(prompt), N);

function timed(fn) {
	torch.synchronize();

	let a = now_ms();
	let out = fn();

	torch.synchronize();

	return { ms: now_ms() - a, out: out };
}

// warm everything (autotune, allocator) once
m.generate(prompt, 8, { temperature: 0 });
m.generate_mtp(prompt, 8, { window: 4 });

// prefill + first token cost, subtracted from every run
let pre = timed(() => m.generate(prompt, 1, { temperature: 0 }));

let plain = timed(() => m.generate(prompt, N, { temperature: 0 }));
let ref = slice(plain.out, length(prompt));
let base = (N - 1) * 1000 / (plain.ms - pre.ms);

printf('plain greedy : %6.1f tok/s  (%d tokens, %.0f ms decode)\n',
	base, length(ref), plain.ms - pre.ms);

for (let w in WINDOWS) {
	let res = timed(() => m.generate_mtp(prompt, N, { window: w }));
	let gen = slice(res.out, length(prompt));
	let same = 0;

	while (same < length(gen) && same < length(ref) && gen[same] == ref[same])
		same++;

	let rate = (length(gen) - 1) * 1000 / (res.ms - pre.ms);

	printf('MTP window %d: %6.1f tok/s  (%.2fx, %d tokens, %d/%d match plain greedy)\n',
		w, rate, rate / base, length(gen), same, length(ref));
}

print(tok.decode(slice(ref, 0, 60)), '...\n');
