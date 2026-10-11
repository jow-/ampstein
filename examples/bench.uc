// examples/bench.uc — Phase 4 baseline benchmark.
//
// Times the engine's prefill and greedy decode against the real model
// and prints the numbers the exit gate (<= 15% tokens/s gap vs
// exllamav3 on the same GPU) will be measured against.
//
// Usage:
//   ucode -L build -L ucode examples/bench.uc [model-dir] [prefill-len] [decode-tokens]
//
// Defaults: /data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw, 128, 64.
//
// Timing notes:
//   - clock(true) is ucode's core monotonic clock: [secs, nsecs].
//   - torch.synchronize() must bracket every timed region; CUDA work is
//     async, so unsynced wall-clock measures launch overhead.

let torch = require('ampstein.torch');
global.torch = torch;
import * as llama from 'ampstein.llama';
import * as sampler from 'ampstein.sampler';
import * as graphdec from 'ampstein.graphdec';

const MODEL = ARGV[0] || '/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';
const PREFILL = int(ARGV[1] || 128);
const DECODE = int(ARGV[2] || 64);

/* Monotonic nanoseconds. */
function now_ns() {
	let t = clock(true);

	return t[0] * 1e9 + t[1];
}

/* Synced wall-clock around fn(): returns fn's return value; prints
 * nothing. Caller keeps the timestamps. */
function time_ms(fn) {
	let a = now_ns();

	fn();
	torch.synchronize();

	return (now_ns() - a) / 1e6;
}

let model = llama.load(MODEL);

// deterministic token sequence: repeat a common word pattern so the
// input is realistic but fixed across runs
let ids = [];

for (let i = 0; i < PREFILL; i++)
	push(ids, 791 + (i % 16));   // arbitrary in-vocab ids

// --- prefill ---------------------------------------------------------------

let cache = model.new_cache(PREFILL + DECODE);
let ms = time_ms(() => model.forward_chunk(ids, cache));

printf('prefill: %d tokens in %.2f ms  ->  %.0f tok/s\n',
	PREFILL, ms, PREFILL * 1000.0 / ms);

// --- decode ----------------------------------------------------------------

let last = null;
let t0 = now_ns();

for (let n = 0; n < DECODE; n++) {
	let logits = model.forward_chunk([791], cache);   // (1, 1, V)
	last = logits;
}

torch.synchronize();
let decode_ms = (now_ns() - t0) / 1e6;

printf('decode: %d tokens in %.2f ms  ->  %.2f tok/s  (%.2f ms/token)\n',
	DECODE, decode_ms, DECODE * 1000.0 / decode_ms, decode_ms / DECODE);

// --- decode with sampling included ------------------------------------------

// second cache: prefill once, then decode through the sampler path —
// this is what a real generation loop costs, incl. the D2H logits copy
// and CPU-side sampling
let cache2 = model.new_cache(PREFILL + DECODE);

let gen = [];

for (let i = 0; i < PREFILL; i++)
	push(gen, ids[i]);

let logits = model.forward_chunk(ids, cache2);
let last2 = logits.narrow(1, PREFILL - 1, 1);

t0 = now_ns();

for (let n = 0; n < DECODE; n++) {
	let tok = sampler.sample(last2, { temperature: 0.8, top_k: 40 });

	push(gen, tok);
	last2 = model.forward_chunk([tok], cache2);
}

torch.synchronize();
let full_ms = (now_ns() - t0) / 1e6;

printf('decode+sample: %d tokens in %.2f ms  ->  %.2f tok/s\n',
	DECODE, full_ms, DECODE * 1000.0 / full_ms);

// --- graph-captured decode, device-side loop -------------------------------

// The captured path replays the whole step as ONE launch and chains
// steps ON DEVICE: the argmax of each step feeds the next step's
// embedding gather, positions advance in device memory, and the only
// host traffic is the token-log read at the end (see ucode/graphdec.uc).
// Same workload as the eager decode above.
let dec = graphdec.create(model, PREFILL + DECODE + 8);
let plog = dec.prefill(ids);
let first = plog.narrow(1, PREFILL - 1, 1)
	.view([1, -1]).topk(1, 1).indices.item();

torch.synchronize();
t0 = now_ns();

dec.start_at(first);
dec.run(DECODE);
let toks = dec.tokens();

torch.synchronize();
let g_ms = (now_ns() - t0) / 1e6;

printf('graph decode: %d tokens in %.2f ms  ->  %.2f tok/s  (%.2f ms/token)  [%d tokens]\n',
	DECODE, g_ms, DECODE * 1000.0 / g_ms, g_ms / DECODE, length(toks));