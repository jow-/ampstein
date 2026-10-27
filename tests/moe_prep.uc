// tests/moe_prep.uc — torch.moe_prep (token-sorted MoE dispatch in one kernel): the permutation is a stable sort by
// local expert (foreign picks last), with consistent counts, starts, inverse, tokens and weights.

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

function check(T, K, E, e0, NE) {
	let ids = [], ws = [];

	for (let a = 0; a < T * K; a++) {
		push(ids, (a * 37 + int(a / K) * 11) % NE);
		push(ws, 0.01 * ((a % 17) + 1));
	}

	let r = torch.moe_prep(torch.from_array(ids, 'int64', dv).view([T, K]), torch.from_array(ws, 'float32', dv).view([T, K]), E, e0);
	let fe = r[0].tolist(), order = r[1].tolist(), tok = r[2].tolist(), wts = r[3].to('float32').tolist();
	let count = r[4].tolist(), start = r[5].tolist(), inv = r[6].tolist(), kind = r[7].tolist();
	let A = T * K, cnt = [], seen = {};

	for (let b = 0; b <= E; b++)
		push(cnt, 0);

	for (let a = 0; a < A; a++) {
		let loc = ids[a] - e0, b = (loc >= 0 && loc < E) ? loc : E;

		assert(fe[a] == b, `fe[${a}]`);
		cnt[b]++;
	}

	let acc = 0;

	for (let b = 0; b <= E; b++) {
		assert(count[b] == cnt[b], `count[${b}]`);
		assert(start[b] == acc, `start[${b}]`);
		acc += cnt[b];

		if (b < E)
			assert(kind[b] == (cnt[b] > 0 ? 1 : 0), `kind[${b}]`);
	}

	let prev = -1, prev_a = -1;

	for (let j = 0; j < A; j++) {
		let a = order[j];

		assert(!seen[a], 'permutation');
		seen[a] = true;
		assert(fe[a] >= prev, 'sorted by bucket');
		assert(fe[a] != prev || a > prev_a, 'stable within a bucket');
		assert(inv[a] == j, 'inverse');
		assert(tok[j] == int(a / K), 'token');
		let dw = wts[j] - ws[a];

		assert(dw < 1e-3 && dw > -1e-3, 'weight');
		prev = fe[a];
		prev_a = a;
	}

	printf('ok T=%d K=%d E=%d e0=%d (experts %d)\n', T, K, E, e0, NE);
}

check(5, 6, 384, 0, 384);
check(40, 6, 192, 192, 384);
check(300, 6, 192, 0, 384);
check(128, 3, 64, 64, 128);
check(4096, 10, 512, 0, 512);         // a prefill chunk: the warp-segmented sort
check(1000, 10, 256, 100, 512);
check(33, 10, 512, 0, 512);
printf('moe_prep: all ok\n');
