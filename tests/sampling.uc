// Phase 3 test: ucode/sampler.uc — greedy, temperature, top-k, seeds.
//
// Pure CPU-side checks on small hand-built logits tensors (no model
// needed); the sampler must behave identically on GPU logits.
//
// Checks:
//   1. greedy (temperature 0) returns the argmax
//   2. top_k = 1 always returns the argmax, whatever the temperature
//   3. seeded runs are reproducible; different seeds diverge
//   4. sampling frequencies match softmax(logits) probabilities
//   5. low temperature sharpens towards the argmax
//   6. top_k restricts the outcome set (tail token never sampled)
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/sampling.uc

import * as sampler from 'ampstein.sampler';
import * as math from 'math';

function abs(x) {
	return x < 0 ? -x : x;
}

// --- 1. greedy = argmax ------------------------------------------------
let logits = torch.from_array([0.1, 2.5, -1.0, 0.3]);

assert(sampler.sample(logits, {}) == 1, 'greedy picks argmax');
assert(sampler.sample(logits, { temperature: 0 }) == 1,
	'temperature 0 picks argmax');
assert(sampler.sample(logits.view([1, 1, 4]), {}) == 1,
	'3-D logits flatten to (vocab,)');

// --- 2. top_k = 1 is deterministic -------------------------------------
for (let t = 0.1; t <= 2.0; t += 0.5)
	assert(sampler.sample(logits, { temperature: t, top_k: 1 }) == 1,
		`top_k=1 picks argmax at T=${t}`);

// --- 3. seeded runs reproduce -------------------------------------------
// Seeding happens once per sequence (exactly what llama.generate does
// with opts.seed); a per-call seed would repeat one draw forever.
math.srand(42);
let a = [];

for (let i = 0; i < 16; i++)
	push(a, sampler.sample(logits, { temperature: 1.5 }));

math.srand(42);
let b = [];

for (let i = 0; i < 16; i++)
	push(b, sampler.sample(logits, { temperature: 1.5 }));

math.srand(7);
let c = [];

for (let i = 0; i < 16; i++)
	push(c, sampler.sample(logits, { temperature: 1.5 }));

let same = true;

for (let i = 0; i < 16; i++)
	same = same && (a[i] == b[i]);

assert(same, 'same seed reproduces the same sequence');

let diff = false;

for (let i = 0; i < 16; i++)
	diff = diff || (a[i] != c[i]);

assert(diff, 'different seeds diverge');

// --- 4. frequencies match softmax probabilities -------------------------
// logits ln([0.5, 0.3, 0.2]) -> softmax recovers the ratios exactly
let probs_logits = torch.from_array([
	math.log(0.5), math.log(0.3), math.log(0.2)
]);

let N = 6000;
let counts = [0, 0, 0];

math.srand(1234);

for (let i = 0; i < N; i++)
	counts[sampler.sample(probs_logits, { temperature: 1.0 })]++;

// binomial 5-sigma bands (sigma ~ sqrt(p(1-p)/N) ~ 0.0065)
let expected = [0.5, 0.3, 0.2];

for (let i = 0; i < 3; i++) {
	let f = counts[i] / (N * 1.0);

	assert(abs(f - expected[i]) < 0.04,
		`token ${i}: freq ${f} vs p ${expected[i]} (counts ${counts})`);
}

// --- 5. low temperature sharpens ----------------------------------------
// logits [1.0, 0.9, 0.8] at T=0.05: gaps of 2 and 4 nats ->
// p(argmax) = 1/(1 + e^-2 + e^-4) ~= 0.867, i.e. ~433/500
let soft = torch.from_array([1.0, 0.9, 0.8]);

let picks0 = 0;

math.srand(99);

for (let i = 0; i < 500; i++)
	picks0 += (sampler.sample(soft, { temperature: 0.05 }) == 0);

assert(picks0 >= 400 && picks0 <= 460,
	`T=0.05 picks argmax ${picks0}/500 times (expected ~433)`);

// --- 6. top_k excludes the tail -----------------------------------------
let wide = torch.from_array([5.0, 4.9, 0.0, -1.0, -2.0, -3.0]);

let tail = 0;

math.srand(5);

for (let i = 0; i < 500; i++)
	if (sampler.sample(wide, { temperature: 1.0, top_k: 2 }) > 1)
		tail++;

assert(tail == 0, 'top_k=2 never samples outside the top 2');

printf('sampling: greedy/topk/seed/frequency checks green '
	+ `(freqs ${counts[0]/(N*1.0)}, ${counts[1]/(N*1.0)}, ${counts[2]/(N*1.0)})\n`);