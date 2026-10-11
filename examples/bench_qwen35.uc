// examples/bench_qwen35.uc — Qwen3.5/3.8 benchmark (multi-GPU layer split).
//
// Same workload as examples/bench.uc: deterministic token pattern,
// prefill + greedy decode + decode-with-sampling, all timed with synced
// wall clock. The oracle counterpart is
// tests/oracle/bench_qwen35_exllamav3.py (same token pattern, same
// GPUs via autosplit).
//
// Usage:
//   ucode -L build -L ucode examples/bench_qwen35.uc [model-dir] [prefill] [decode] [layers-per-gpu]
//
// Defaults: /data/models/Qwen3.8-27B-exl3-5.00bpw, 128, 64, 32.

let torch = require('ampstein.torch');
global.torch = torch;
import * as qwen35 from 'ampstein.qwen35';
import * as sampler from 'ampstein.sampler';

const MODEL = ARGV[0] || '/data/models/Qwen3.8-27B-exl3-5.00bpw';
const PREFILL = int(ARGV[1] || 128);
const DECODE = int(ARGV[2] || 64);
const PER_GPU = int(ARGV[3] || 32);

function now_ns() {
	let t = clock(true);

	return t[0] * 1e9 + t[1];
}

function time_ms(fn) {
	let a = now_ns();

	fn();
	torch.synchronize();

	return (now_ns() - a) / 1e6;
}

/* layer -> device map: first PER_GPU layers on cuda:0, rest on cuda:1 */
let devs = [];

for (let i = 0; i < 64; i++)
	push(devs, (i < PER_GPU) ? 0 : 1);

let model = qwen35.load(MODEL, { devices: devs });

let ids = [];

for (let i = 0; i < PREFILL; i++)
	push(ids, 791 + (i % 16));

// --- prefill ---------------------------------------------------------------

let cache = model.new_cache(PREFILL + DECODE);
let ms = time_ms(() => model.forward_chunk(ids, cache));

printf('prefill: %d tokens in %.2f ms  ->  %.0f tok/s\n',
	PREFILL, ms, PREFILL * 1000.0 / ms);

// --- decode (greedy, eager) ------------------------------------------------

let last = null;
let t0 = now_ns();

for (let n = 0; n < DECODE; n++) {
	let logits = model.forward_chunk([791], cache);
	last = logits;
}

torch.synchronize();
let decode_ms = (now_ns() - t0) / 1e6;

printf('decode: %d tokens in %.2f ms  ->  %.2f tok/s  (%.2f ms/token)\n',
	DECODE, decode_ms, DECODE * 1000.0 / decode_ms, decode_ms / DECODE);

// --- decode with sampling included -----------------------------------------

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