// tests/attn_decode_s.uc — torch.attn_decode (split-KV decode attention,
// S query rows) against a plain-tensor-op SDPA reference at the shapes
// the MTP verify path uses (Qwen3.8: E = 256, Hq = 24, Hk = 4, rep = 6)
// and at head_dim 128. Both kernel variants are exercised: the shuffle
// kernel (mma = 0) and the tensor-core mma kernel (mma = 1). No model
// needed.
//
// Checks, per (S, position):
//   - max |diff| vs the SDPA reference over the causal window
//   - determinism across repeated launches at a fixed position
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/attn_decode_s.uc

import * as math from 'math';

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

/* Deterministic LCG in [-1, 1). */
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

/* One (S, pos) case at head_dim E, nh/kvh heads, cache capacity CS,
 * kernel variant mma (0 = shuffle, 1 = tensor-core): build q/k/v, run
 * the kernel, compare against an SDPA reference over the causal window
 * [0, len - S + s]. */
function run_case(nh, kvh, E, S, pos, CS, seed, mma) {
	let rep = nh / kvh;
	let len = pos + 1;

	let q = cuda_f16(rand_arr(S * nh * E, seed), [S, nh, E]);
	let k_cache = torch.zeros([kvh, CS, E], 'float16', 'cuda');
	let v_cache = torch.zeros([kvh, CS, E], 'float16', 'cuda');

	// fill the used window with deterministic keys/values
	let kn = cuda_f16(rand_arr(len * kvh * E, seed + 1), [len, kvh, E]);
	let vn = cuda_f16(rand_arr(len * kvh * E, seed + 2), [len, kvh, E]);

	k_cache.view([kvh, CS, E]).narrow(1, 0, len)
		.copy_(kn.transpose(0, 1));
	v_cache.view([kvh, CS, E]).narrow(1, 0, len)
		.copy_(vn.transpose(0, 1));

	let len_t = torch.from_array([len], 'int32', 'cuda');

	let o = torch.attn_decode(q, k_cache, v_cache, len_t,
		1.0 / math.sqrt(E), 0, mma);

	// reference: per query row s, SDPA over keys [0, len - S + s]
	let scale = 1.0 / math.sqrt(E);
	let worst = 0;

	for (let s = 0; s < S; s++) {
		let kv = len - S + s + 1;
		let qs = q.narrow(0, s, 1).view([1, nh, 1, E]);       // (1, H, 1, E)
		let ks = k_cache.narrow(1, 0, kv).unsqueeze(0);        // (1, Hk, kv, E)
		let vs = v_cache.narrow(1, 0, kv).unsqueeze(0);

		// GQA: repeat kv heads rep times (expand + reshape, no copy)
		let ksr = ks.unsqueeze(2).expand([1, kvh, rep, kv, E])
			.reshape([1, nh, kv, E]);
		let vsr = vs.unsqueeze(2).expand([1, kvh, rep, kv, E])
			.reshape([1, nh, kv, E]);

		let ref = torch.sdpa(qs, ksr, vsr, false, null, scale);
		let d = o.narrow(0, s, 1).view([1, nh, 1, E]).sub(ref)
			.abs().max().item();

		if (d > worst)
			worst = d;
	}

	return worst;
}

/* Run a battery of (pos, S) cases for both kernel variants and check
 * each against the SDPA reference. */
function battery(name, nh, kvh, E, CS, positions, Smax, seedbase) {
	for (let mma = 0; mma <= 1; mma++) {
		let worst = 0;

		for (let i = 0; i < length(positions); i++) {
			let pos = positions[i];

			for (let S = 1; S <= Smax; S++) {
				if (pos + 1 < S)
					continue;

				let d = run_case(nh, kvh, E, S, pos, CS,
					seedbase + i * 10 + S, mma);

				if (d > worst)
					worst = d;
			}
		}

		check(name + (mma ? ' [mma]' : ' [shuffle]')
			+ ': max |diff| <= 0.05', worst <= 0.05, worst);
	}
}

/* --- 1. Qwen3.8 verify shapes (E = 256, rep = 6) --------------------------- */

battery('E=256', 24, 4, 256, 4096, [ 0, 1, 63, 1024, 4095 ], 3, 100);

/* --- 2. head_dim 128 (llama-family shapes) --------------------------------- */

battery('E=128', 32, 8, 128, 2048, [ 0, 511, 2047 ], 5, 200);

/* --- 3. determinism: 100 launches at a fixed position, both variants ------ */

{
	const NH = 24, KVH = 4, E = 256, S = 3, CS = 1024, POS = 777;
	let q = cuda_f16(rand_arr(S * NH * E, 300), [S, NH, E]);
	let k_cache = torch.zeros([KVH, CS, E], 'float16', 'cuda');
	let v_cache = torch.zeros([KVH, CS, E], 'float16', 'cuda');
	let kn = cuda_f16(rand_arr((POS + 1) * KVH * E, 301), [POS + 1, KVH, E]);
	let vn = cuda_f16(rand_arr((POS + 1) * KVH * E, 302), [POS + 1, KVH, E]);

	k_cache.view([KVH, CS, E]).narrow(1, 0, POS + 1)
		.copy_(kn.transpose(0, 1));
	v_cache.view([KVH, CS, E]).narrow(1, 0, POS + 1)
		.copy_(vn.transpose(0, 1));

	let len_t = torch.from_array([POS + 1], 'int32', 'cuda');

	for (let mma = 0; mma <= 1; mma++) {
		let o0 = torch.attn_decode(q, k_cache, v_cache, len_t,
			1.0 / math.sqrt(E), 0, mma);
		let stable = true;

		for (let i = 0; i < 100; i++) {
			let o = torch.attn_decode(q, k_cache, v_cache, len_t,
				1.0 / math.sqrt(E), 0, mma);

			if (o.sub(o0).abs().max().item() != 0)
				stable = false;
		}

		check('determinism' + (mma ? ' [mma]' : ' [shuffle]')
			+ ': 100 launches bit-identical', stable);
	}
}

/* --- 4. throughput: the long-context verify shape -------------------------- */
/* 180k keys, S = 3 (the MTP verify at a 210k context): the case that
 * motivated the mma kernel. K/V are generated on the device (a 184M-
 * element host-side array takes minutes and syncs the pageable H2D,
 * L58). Benchmarks both variants; the mma result must also stay close
 * to the shuffle result. */

{
	const NH = 24, KVH = 4, E = 256, S = 3, CS = 262144, KEYS = 180000;
	let q = torch.rand([S, NH, E], 'float16', 'cuda');
	let k_cache = torch.zeros([KVH, CS, E], 'float16', 'cuda');
	let v_cache = torch.zeros([KVH, CS, E], 'float16', 'cuda');

	k_cache.narrow(1, 0, KEYS)
		.copy_(torch.rand([KVH, KEYS, E], 'float16', 'cuda'));
	v_cache.narrow(1, 0, KEYS)
		.copy_(torch.rand([KVH, KEYS, E], 'float16', 'cuda'));

	let len_t = torch.from_array([KEYS], 'int32', 'cuda');
	let scale = 1.0 / math.sqrt(E);

	for (let mma = 0; mma <= 1; mma++) {
		// warmup
		torch.attn_decode(q, k_cache, v_cache, len_t, scale, 0, mma);
		torch.synchronize();

		let t0 = now_ms(), K = 20;

		for (let i = 0; i < K; i++)
			torch.attn_decode(q, k_cache, v_cache, len_t, scale, 0, mma);

		torch.synchronize();

		printf('throughput: %d keys, S=%d, E=%d, %s: %.3f ms/launch\n',
			KEYS, S, E, (mma ? 'mma' : 'shuffle'), (now_ms() - t0) / K);
	}

	// the variants must agree (different summation order, same math)
	let d = torch.attn_decode(q, k_cache, v_cache, len_t, scale, 0, 0)
		.sub(torch.attn_decode(q, k_cache, v_cache, len_t, scale, 0, 1))
		.abs().max().item();

	check('180k keys: |shuffle - mma| <= 0.05', d <= 0.05, d);
}

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('all green\n');
