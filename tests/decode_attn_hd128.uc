// Synthetic decode_attn test at head_dim = 128 (dpl = 4) — the kernel
// branch the 1B oracle model never exercises (it has hd = 64), while
// Llama-3.1-8B and most other targets use 128 (a testing
// gap). No model needed: random-ish fp16 tensors, the reference is
// rope + KV write + SDPA over [0, pos] built from plain tensor ops.
//
// Checks, per config (rep 1 / 4 / 8) and position (split boundaries,
// empty splits, the last slot):
//   - decode_attn matches the SDPA reference (max |diff| bound)
//   - 100 repeated launches at a fixed position are BIT-identical —
//     the phase-1/phase-2 __syncthreads() race check
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/decode_attn_hd128.uc
//
// -D HD_OVERRIDE=64 runs the same sweep at head_dim 64 (dpl 2).

import * as llama from 'ampstein.llama';
import * as math from 'math';

const HD = int(HD_OVERRIDE != null ? HD_OVERRIDE : 128);
const MAX_SEQ = 256;
const NS = 8;

/* Deterministic LCG in [-1, 1) — ucode doubles hold s * 16807 exactly
 * (well below 2^53). */
function rand_arr(n, seed) {
	let a = [];
	let s = seed;

	for (let i = 0; i < n; i++) {
		s = (s * 16807) % 2147483647;
		push(a, s / 1073741824.0 - 1.0);
	}

	return a;
}

function cuda_f16(arr, shape) {
	return torch.from_array(arr).to('float16').cuda().view(shape);
}

/* Build one config's tensors and run the position sweep. */
function run_config(nh, kvh, seed) {
	let rep = nh / kvh;
	let wsrow = 4 + HD;

	let q = cuda_f16(rand_arr(nh * HD, seed), [nh, HD]);
	let k_new = cuda_f16(rand_arr(kvh * HD, seed + 1), [kvh, HD]);
	let v_new = cuda_f16(rand_arr(kvh * HD, seed + 2), [kvh, HD]);
	let k_cache = torch.zeros([kvh, MAX_SEQ, HD], 'float16', 'cuda');
	let v_cache = torch.zeros([kvh, MAX_SEQ, HD], 'float16', 'cuda');

	/* RoPE tables: arbitrary values in [-1, 1) — the kernel treats
	 * cos/sin as plain multipliers, the reference applies the same
	 * ones, so real cosine values are not required */
	let cos = cuda_f16(rand_arr(MAX_SEQ * HD, seed + 3), [MAX_SEQ, HD]);
	let sin = cuda_f16(rand_arr(MAX_SEQ * HD, seed + 4), [MAX_SEQ, HD]);

	let ws = torch.zeros([kvh, NS, rep, wsrow], 'float32', 'cuda');
	let out = torch.zeros([nh, HD], 'float16', 'cuda');
	let pos = torch.zeros([1], 'int64', 'cuda');
	let scale = 1.0 / math.sqrt(HD);

	/* Reference: rope q and k, write the roped k / v at slot p, then
	 * plain SDPA over [0, p] with GQA. */
	function reference(p) {
		let cp = cos.narrow(0, p, 1);
		let sp = sin.narrow(0, p, 1);

		let qr = llama.rope_apply(q.view([1, nh, 1, HD]), cp, sp);
		let kr = llama.rope_apply(k_new.view([1, kvh, 1, HD]), cp, sp);

		let kc = torch.zeros([kvh, MAX_SEQ, HD], 'float16', 'cuda');
		let vc = torch.zeros([kvh, MAX_SEQ, HD], 'float16', 'cuda');

		kc.copy_(k_cache);
		vc.copy_(v_cache);
		kc.narrow(1, p, 1).copy_(kr.view([kvh, 1, HD]));
		vc.narrow(1, p, 1).copy_(v_new.view([kvh, 1, HD]));

		let o = torch.sdpa(qr,
			kc.unsqueeze(0).narrow(2, 0, p + 1),
			vc.unsqueeze(0).narrow(2, 0, p + 1),
			false, null, true);

		return o.view([nh, HD]);
	}

	function kernel(p) {
		pos.copy_(torch.from_array([p], 'int64'));
		torch.decode_attn(q, k_new, v_new, k_cache, v_cache,
			cos, sin, ws, out, pos, scale);
	}

	/* split boundaries, an empty-split case and the last slot */
	let positions = [0, 1, 7, 8, 9, 127, MAX_SEQ - 1];
	let ncheck = 0;

	for (let i = 0; i < length(positions); i++) {
		let p = positions[i];

		kernel(p);

		let d = out.sub(reference(p)).abs().max().item();

		assert(d < 0.02,
			`nh=${nh} kvh=${kvh} pos=${p}: max |diff| ${d} >= 0.02`);

		/* determinism: 100 more launches must be bit-identical
		 * (a missing barrier between the KV write and the
		 * attention reads shows up as occasional bit flips) */
		let first = torch.zeros([nh, HD], 'float16', 'cuda');

		first.copy_(out);

		for (let r = 0; r < 100; r++) {
			kernel(p);

			let dd = out.sub(first).abs().max().item();

			assert(dd == 0,
				`nh=${nh} kvh=${kvh} pos=${p}: nondeterministic ` +
				`output on repeat ${r} (diff ${dd})`);
		}

		ncheck++;
	}

	printf('decode_attn hd=%d: nh=%d kvh=%d rep=%d — %d positions, ' +
		'reference + 100x determinism OK\n', HD, nh, kvh, rep, ncheck);

	return ncheck;
}

let total = 0;

total += run_config(8, 2, 12345);    /* rep 4 — the 1B model's shape */
total += run_config(8, 8, 54321);    /* rep 1 — no GQA */
total += run_config(16, 2, 98765);   /* rep 8 */

printf('decode_attn_hd128: %d configs green\n', total);