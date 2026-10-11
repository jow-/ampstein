// Test: speculative-sampling acceptance rule (engine.spec_step) — for a
// one-hot draft x and target distribution p, the emitted token must be
// distributed exactly as p, whatever x is (including a draft outside
// p's support). Monte Carlo, 3-sigma binomial bands. No GPU work.
//
//   ucode -L <build-dir> -L <src>/ucode tests/spec_sampling.uc

import * as engine from 'ampstein.engine';
import * as math from 'math';

math.srand(1234);

const N = 60000;
let d = { ids: [ 11, 22, 33, 44 ], probs: [ 0.5, 0.25, 0.15, 0.10 ] };

for (let x in [ 11, 33, 44, 99 ]) {
	let hits = {};
	let acc = 0;

	for (let i = 0; i < N; i++) {
		let r = engine.spec_step(d, x);

		hits[r.token] = (hits[r.token] || 0) + 1;

		if (r.accepted)
			acc++;
	}

	for (let i = 0; i < length(d.ids); i++) {
		let p = d.probs[i];
		let f = (hits[d.ids[i]] || 0) * 1.0 / N;
		let sigma = math.sqrt(p * (1 - p) / N);

		assert(math.abs(f - p) < 3 * sigma + 1e-9,
			`draft ${x}: token ${d.ids[i]} frequency ${f} vs p ${p}`);
	}

	assert(!hits[99], `draft 99 (outside the support) must never be emitted`);

	let pa = (index(d.ids, x) >= 0) ? d.probs[index(d.ids, x)] : 0;

	assert(math.abs(acc * 1.0 / N - pa) < 3 * math.sqrt(pa * (1 - pa) / N) + 1e-9,
		`draft ${x}: acceptance ${acc * 1.0 / N} vs p(x) ${pa}`);

	printf('draft %d: acceptance %.4f (p(x) %.2f), emitted distribution matches p\n',
		x, acc * 1.0 / N, pa);
}

print('spec_sampling: PASS\n');
