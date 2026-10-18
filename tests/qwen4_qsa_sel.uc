// tests/qwen4_qsa_sel.uc — the decode token selection of the Qwen3.8-Flash-Next QSA layers: the DSA kernel chain
// (qsa_idx_dev: idx_score + dsa_select, live pools only) against the composite ATen chain (qsa_select_t), on random
// pooled keys and index queries at several positions (few complete blocks, the budget boundary, deep), for one and
// three consecutive rows. The token sets must agree except for scores within fp16 rounding of the threshold.
import * as qwen4 from 'ampstein.qwen4';
import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;
const cfg = {indexer_heads: 4, indexer_head_dim: 128, indexer_compress_ratio: 4, indexer_budget: 2048};
const CAP = 20000;             // pooled capacity (blocks)

function noise(rows, cols, sd) {
	let a = [];

	for (let r = 0; r < rows; r++) {
		let row = [];

		for (let c = 0; c < cols; c++)
			push(row, math.rand() / 2147483647.0 * 2.0 * sd - sd);

		push(a, row);
	}

	return torch.from_array(a, 'float32').to('float16').cuda(DEV);
}

let fails = 0;
let pooled = noise(CAP, 128, 1.0);

for (let S in [ 1, 3 ]) {
	for (let pos in [ 5, 700, 2051 * 1, 2300, 9000, 40000, 79000 ]) {
		let cl = {dev: DEV, pooled: pooled};
		let qi = noise(S * 4, 128, 1.0).view([S, 4, 128]);
		let posv = [];

		for (let r = 0; r < S; r++)
			push(posv, pos + r);

		let st = {pos: torch.from_array(posv, 'int64', dv)};
		let got = qwen4.qsa_idx_dev(cfg, cl, qi, st, S).tolist();

		// reference: the composite chain on the same inputs
		let qpos = st.pos.view([S, 1]);
		let nbq = qpos.add(torch.full([1], 1, 'int64', dv)).to('float32').mul(0.25).to('int64');
		let tail = nbq.mul(torch.full([1], 4, 'int64', dv)).add(torch.arange(4, 'int64', dv).view([1, 4]));
		let sel = qwen4.qsa_select_t(cfg, cl, qi, qpos, nbq, tail, CAP, S);
		let rt = sel.tok.tolist(), rk = sel.ok.tolist();

		// fp32 scores of every complete block, to judge the disagreements
		let s = torch.matmul(qi.to('float32').view([S * 4, 128]), cl.pooled.to('float32').t());
		let sc = s.clamp(0.0, null).view([S, 4, CAP]).sum(1).mul(1.0 / math.sqrt(128.0)).tolist();

		for (let r = 0; r < S; r++) {
			let a = {}, b = {}, na = 0, nb = 0;

			for (let t in got[r])
				if (t >= 0) {
					a[t] = 1;
					na++;
				}

			for (let k = 0; k < length(rt[r]); k++)
				if (rk[r][k] > 0.5) {
					b[rt[r][k]] = 1;
					nb++;
				}

			// the sets of tokens: blocks (and tail) in one but not the other
			let diff = 0, worst = 0;
			let nbl = int((pos + r + 1) / 4);
			let sorted = sort(slice(sc[r], 0, nbl), (x, y) => y - x);
			let kth = (nbl >= 512) ? sorted[511] : -1e30;

			for (let t in keys(a))
				if (!b[t]) {
					diff++;

					let blk = int(+t / 4);

					if (blk < nbl)
						worst = max(worst, math.abs(sc[r][blk] - kth));
				}

			for (let t in keys(b))
				if (!a[t]) {
					diff++;

					let blk = int(+t / 4);

					if (blk < nbl)
						worst = max(worst, math.abs(sc[r][blk] - kth));
				}

			// a mismatch is allowed for blocks whose score is within fp16 rounding of the k-th
			let ok = na == nb && worst < 0.05 && diff <= 24;

			printf('S=%d pos=%d row %d: %d vs %d tokens, %d differ (worst score gap %.4f) %s\n', S, pos, r, na, nb,
				diff, worst, ok ? 'ok' : 'FAIL');

			if (!ok)
				fails++;
		}
	}
}

exit(fails ? 1 : 0);
