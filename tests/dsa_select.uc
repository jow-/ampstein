// tests/dsa_select.uc — torch.dsa_select (the decode-time DSA pool selection
// of GLM-5.3's sparse MLA layers) against a host reference of its exact
// semantics: the top kp pools by score (ties to the lower pool index), their
// member tokens in ascending pool order, pools the query cannot see as
// padding, then the incomplete tail pool; the device-side position.
import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

function run_case(T, cap, kp, P, pos, levels, seed) {
	math.srand(seed);

	// scores: a few distinct fp16 values (ties everywhere), pools the row
	// cannot see (p * P + P - 1 > pos + r) masked like amp_idx_score does
	let flat = [];
	let vals = [];

	for (let r = 0; r < T; r++) {
		for (let p = 0; p < cap; p++) {
			let v = (p * P + P - 1 > pos + r) ? -65504.0 : (levels > 0 ?
				math.floor(math.rand() % levels) / 4.0 - 1.0 : math.rand() / 2147483648.0 * 8.0 - 4.0);

			push(flat, v);
		}
	}

	let sc = torch.from_array(flat, 'float32').to('float16').cuda(DEV).view([T, cap]);
	let host = sc.to('float32').cpu().tolist();
	let pt = torch.from_array([ pos ], 'int64', 'cuda:' + DEV);
	let out = torch.dsa_select(sc, pt, P, kp);
	let gi = out[0].cpu().tolist(), gp = out[1].to('float32').cpu().tolist();
	let K = kp * P + P - 1;
	let bad = null;

	for (let r = 0; r < T && !bad; r++) {
		let order = [];

		for (let p = 0; p < cap; p++)
			push(order, p);

		let row = host[r] ?? host;      // (T, cap) or flattened

		if (type(row) != 'array')
			row = slice(host, r * cap, (r + 1) * cap);

		order = sort(order, (a, b) => (row[b] - row[a]) || (a - b));

		let sel = sort(slice(order, 0, kp), (a, b) => a - b);
		let want_i = [], want_p = [];

		for (let p in sel) {
			let dead = (row[p] <= -60000.0);

			for (let j = 0; j < P; j++) {
				push(want_i, dead ? 0 : p * P + j);
				push(want_p, dead ? -65504.0 : 0.0);
			}
		}

		let qpos = pos + r;
		let start = int((qpos + 1) / P) * P;

		for (let j = 0; j < P - 1; j++) {
			push(want_i, (start + j <= qpos) ? start + j : 0);
			push(want_p, (start + j <= qpos) ? 0.0 : -65504.0);
		}

		let ri = gi[r], rp = gp[r];

		if (type(ri) != 'array') {
			ri = slice(gi, r * K, (r + 1) * K);
			rp = slice(gp, r * K, (r + 1) * K);
		}

		for (let k = 0; k < K; k++)
			if (ri[k] != want_i[k] || rp[k] != want_p[k]) {
				bad = `row ${r} slot ${k}: got (${ri[k]}, ${rp[k]}) want (${want_i[k]}, ${want_p[k]})`;
				break;
			}
	}

	return bad;
}

let cases = [
	// name, T, cap, kp, P, pos, levels
	[ 'distinct scores, all visible', 4, 4100, 64, 4, 20000, 0 ],
	[ 'heavy ties', 3, 3000, 100, 4, 12000, 6 ],
	[ 'fewer visible pools than kp (padding)', 3, 1000, 100, 4, 40, 0 ],
	[ 'kp == cap', 2, 700, 700, 4, 5000, 0 ],
	[ 'P = 128 (HCA-sized), odd sizes', 2, 333, 17, 128, 20000, 5 ],
	[ 'GLM decode shape (65537 pools, kp 512)', 2, 65537, 512, 4, 120000, 40 ]
];

for (let c in cases) {
	let bad = run_case(c[1], c[2], c[3], c[4], c[5], c[6], 1234 + c[2]);

	check(c[0], bad == null, bad);
}

// determinism: repeated launches are bit-identical
{
	let sc = torch.rand([4, 8192], 'float16', 'cuda:' + DEV);
	let pt = torch.from_array([ 9000 ], 'int64', 'cuda:' + DEV);
	let a = torch.dsa_select(sc, pt, 4, 256), same = true;

	for (let i = 0; i < 50; i++) {
		let b = torch.dsa_select(sc, pt, 4, 256);

		if (b[0].sub(a[0]).abs().max().item() != 0 || b[1].sub(a[1]).abs().max().item() != 0)
			same = false;
	}

	check('50 launches bit-identical', same);
}

function raises(f) {
	try { f(); } catch (e) { return true; }

	return false;
}

let sc = torch.rand([2, 100], 'float16', 'cuda:' + DEV);
let pt = torch.from_array([ 5 ], 'int64', 'cuda:' + DEV);

check('kp above the pool count is rejected', raises(() => torch.dsa_select(sc, pt, 4, 200)));
check('scores must be fp16', raises(() => torch.dsa_select(sc.to('float32'), pt, 4, 10)));
check('pos must be int64', raises(() => torch.dsa_select(sc, pt.to('int32'), 4, 10)));
check('P must be positive', raises(() => torch.dsa_select(sc, pt, 0, 10)));

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('dsa_select: ok\n');
