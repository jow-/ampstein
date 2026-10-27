// Test: torch.gdn_scan (fused gated delta rule, src/amp_kernels.cu) vs the
// reference recurrence ucode/qwen35.uc delta_rule_scan (per-token ATen
// ops, the exllamav3 torch reference math). Synthetic data, no model.
//
// Covers GQA grouping (Hv = 3 * Hk), a non-zero initial state, both
// supported k head dims, and chunk splitting (one 37-token scan == a
// 20-token scan followed by a 17-token scan from the carried state).
//
//   ucode -L <build-dir> -L <src>/ucode tests/gdn_scan.uc

import * as qwen35 from 'ampstein.qwen35';
import * as math from 'math';

/* deterministic pseudo-random values in [-1, 1) */
let seed = 12345;

function rnd() {
	seed = (seed * 1103515245 + 12345) % 2147483648;

	return (seed / 1073741824.0) - 1.0;
}

function tensor(shape, lo, hi) {
	let n = 1;

	for (let i = 0; i < length(shape); i++)
		n *= shape[i];

	let vals = [];

	for (let i = 0; i < n; i++)
		push(vals, lo + (rnd() + 1.0) * 0.5 * (hi - lo));

	return torch.from_array(vals).view(shape).cuda(0);
}

function maxdiff(a, b) {
	return a.sub(b).abs().max().item();
}

function check(S, Hk, Hv, dk, dv) {
	let q = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let k = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let v = tensor([S, Hv, dv], -1, 1);
	let beta = tensor([S, Hv], 0.05, 0.95);
	let gexp = tensor([S, Hv], 0.80, 1.0);
	let st0 = tensor([Hv, dk, dv], -0.2, 0.2);
	let scale = 1.0 / math.sqrt(dk);
	let rep = Hv / Hk;

	// reference (repeats heads itself)
	let st_ref = st0.add(0.0);
	let out_ref = qwen35.delta_rule_scan(qwen35.repeat_heads(q, rep),
		qwen35.repeat_heads(k, rep), v, beta, gexp, st_ref, scale, 0);

	// fused, one launch
	let st_a = st0.add(0.0);
	let out_a = torch.gdn_scan(q, k, v, beta, gexp, st_a, scale);

	let d_out = maxdiff(out_a, out_ref);
	let d_st = maxdiff(st_a, st_ref);

	assert(d_out < 1e-4, `dk=${dk}: out maxdiff ${d_out}`);
	assert(d_st < 1e-4, `dk=${dk}: state maxdiff ${d_st}`);

	// fused, split into two chunks carrying the state
	let s1 = 20;
	let st_b = st0.add(0.0);
	let o1 = torch.gdn_scan(q.narrow(0, 0, s1).contiguous(), k.narrow(0, 0, s1).contiguous(),
		v.narrow(0, 0, s1).contiguous(), beta.narrow(0, 0, s1).contiguous(),
		gexp.narrow(0, 0, s1).contiguous(), st_b, scale);
	let o2 = torch.gdn_scan(q.narrow(0, s1, S - s1).contiguous(),
		k.narrow(0, s1, S - s1).contiguous(), v.narrow(0, s1, S - s1).contiguous(),
		beta.narrow(0, s1, S - s1).contiguous(), gexp.narrow(0, s1, S - s1).contiguous(),
		st_b, scale);
	let d_split = maxdiff(torch.cat([o1, o2], 0), out_a);
	let d_split_st = maxdiff(st_b, st_a);

	assert(d_split < 1e-6 && d_split_st < 1e-6,
		`dk=${dk}: chunked vs single scan differ (${d_split}, ${d_split_st})`);

	printf('gdn_scan dk=%d dv=%d Hk=%d Hv=%d S=%d: out %.2e, state %.2e, split %.2e\n',
		dk, dv, Hk, Hv, S, d_out, d_st, d_split);
}

check(37, 4, 12, 128, 128);
check(37, 2, 6, 64, 128);

/* states_out (speculative verify): the state after each of the first n
 * tokens equals the state of a scan over only those tokens */
function check_states_out(S, n, Hk, Hv, dk, dv) {
	let q = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let k = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let v = tensor([S, Hv, dv], -1, 1);
	let beta = tensor([S, Hv], 0.05, 0.95);
	let gexp = tensor([S, Hv], 0.80, 1.0);
	let st0 = tensor([Hv, dk, dv], -0.2, 0.2);
	let scale = 1.0 / math.sqrt(dk);
	let so = torch.zeros([n, Hv, dk, dv], 'float32', 'cuda:0');
	let st_a = st0.add(0.0);
	let out_a = torch.gdn_scan(q, k, v, beta, gexp, st_a, scale, so);

	for (let t = 0; t < n; t++) {
		let st_t = st0.add(0.0);
		let m = t + 1;

		torch.gdn_scan(q.narrow(0, 0, m).contiguous(), k.narrow(0, 0, m).contiguous(),
			v.narrow(0, 0, m).contiguous(), beta.narrow(0, 0, m).contiguous(),
			gexp.narrow(0, 0, m).contiguous(), st_t, scale);

		let d = maxdiff(so.narrow(0, t, 1).view([Hv, dk, dv]), st_t);

		assert(d == 0, `states_out[${t}] differs from a ${m}-token scan (${d})`);
	}

	// the final state and outputs are unaffected
	let st_b = st0.add(0.0);
	let out_b = torch.gdn_scan(q, k, v, beta, gexp, st_b, scale);

	assert(maxdiff(out_a, out_b) == 0 && maxdiff(st_a, st_b) == 0,
		'states_out changed the scan result');

	printf('gdn_scan states_out S=%d n=%d dk=%d: exact\n', S, n, dk);
}

check_states_out(3, 2, 4, 12, 128, 128);
check_states_out(5, 4, 2, 6, 64, 128);

/* chunked (64-token block, WY/UT form) kernel vs the same reference:
 * a single partial block, several blocks with a partial tail, and a
 * split into two calls carrying the state (block boundaries move) */
function check_chunk(S, Hk, Hv, split) {
	let dk = 128, dv = 128;
	let q = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let k = qwen35.l2norm(tensor([S, Hk, dk], -1, 1));
	let v = tensor([S, Hv, dv], -1, 1);
	let beta = tensor([S, Hv], 0.05, 0.95);
	let gexp = tensor([S, Hv], 0.80, 1.0);
	let st0 = tensor([Hv, dk, dv], -0.2, 0.2);
	let scale = 1.0 / math.sqrt(dk);
	let rep = Hv / Hk;

	let st_ref = st0.add(0.0);
	let out_ref = qwen35.delta_rule_scan(qwen35.repeat_heads(q, rep),
		qwen35.repeat_heads(k, rep), v, beta, gexp, st_ref, scale, 0);

	let st_c = st0.add(0.0), out_c;

	if (split) {
		let n2 = S - split;
		let o1 = torch.gdn_chunk(q.narrow(0, 0, split).contiguous(),
			k.narrow(0, 0, split).contiguous(), v.narrow(0, 0, split).contiguous(),
			beta.narrow(0, 0, split).contiguous(), gexp.narrow(0, 0, split).contiguous(),
			st_c, scale);
		let o2 = torch.gdn_chunk(q.narrow(0, split, n2).contiguous(),
			k.narrow(0, split, n2).contiguous(), v.narrow(0, split, n2).contiguous(),
			beta.narrow(0, split, n2).contiguous(), gexp.narrow(0, split, n2).contiguous(),
			st_c, scale);

		out_c = torch.cat([o1, o2], 0);
	}
	else {
		out_c = torch.gdn_chunk(q, k, v, beta, gexp, st_c, scale);
	}

	let d_out = maxdiff(out_c, out_ref);
	let d_st = maxdiff(st_c, st_ref);

	assert(d_out < 1e-4, `chunk S=${S}: out maxdiff ${d_out}`);
	assert(d_st < 1e-4, `chunk S=${S}: state maxdiff ${d_st}`);

	printf('gdn_chunk Hk=%d Hv=%d S=%d%s: out %.2e, state %.2e\n',
		Hk, Hv, S, split ? ` (split at ${split})` : '', d_out, d_st);
}

// (the 64-token-block WY/UT kernel is CUDA only: the ROCm backend raises "not available" and runs the sequential scan only)
let have_chunk = true;

try {
	torch.gdn_chunk(torch.zeros([1, 1, 128], 'float32', 'cuda:0'), torch.zeros([1, 1, 128], 'float32', 'cuda:0'),
		torch.zeros([1, 1, 128], 'float32', 'cuda:0'), torch.zeros([1, 1], 'float32', 'cuda:0'), torch.zeros([1, 1], 'float32', 'cuda:0'),
		torch.zeros([1, 128, 128], 'float32', 'cuda:0'), 1.0);
}
catch (e) {
	if (index(e.message, 'ROCm') >= 0)
		have_chunk = false;
}

if (have_chunk) {
	check_chunk(37, 4, 12, 0);
	check_chunk(200, 4, 12, 0);
	check_chunk(200, 2, 6, 130);
}
else {
	print('gdn_chunk: not available on this backend, skipped\n');
}

// bad input is rejected, not executed
let bad = false;

try {
	torch.gdn_scan(tensor([4, 2, 96], -1, 1), tensor([4, 2, 96], -1, 1),
		tensor([4, 2, 64], -1, 1), tensor([4, 2], 0, 1), tensor([4, 2], 0, 1),
		tensor([2, 96, 64], 0, 0), 1.0);
}
catch (e) {
	bad = true;
}

assert(bad, 'unsupported dk=96 must raise');

print('gdn_scan: PASS\n');
