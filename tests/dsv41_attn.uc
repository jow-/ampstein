// tests/dsv41_attn.uc — the DeepSeek-V4.1 attention layers (compressors, indexers, candidate blocks, shared
// pools) against DeepSeek's reference run on the CPU (tests/oracle/make_dsv41_oracle.py): a chunked
// prefill (chunks of odd length, so the half-built pairs of the ratio-2 compressor cross chunk
// boundaries) and single-token decode steps.
//   AMP_DSV41_ORACLE=<fixture>  AMP_DSV41_ATTN_W=<attn weights safetensors>  AMP_DSV41_CFG=<dir with config.json>

import * as dsv41 from 'ampstein.dsv41';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const FIX = getenv('AMP_DSV41_ORACLE') || '/data/models/dsv41_attn_oracle.safetensors';
const WF = getenv('AMP_DSV41_ATTN_W') || '/data/models/attn_l2_3_20_21_24.safetensors';
const CFGDIR = getenv('AMP_DSV41_CFG') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(FIX) || !stat(WF) || !stat(CFGDIR + '/config.json')) {
	warn('skip: oracle fixture, attention weights or config missing\n');
	exit(125);
}

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const DEV2 = +(getenv('AMP_TEST_DEV2') || DEV);          // the consumers run on this device
const LAYERS = [ 2, 3, 20, 21, 24 ];
const ND = +(getenv('AMP_DSV41_ND') || 4);          // decode steps at the end of the fixture

let cfg = dsv41.load_config(CFGDIR);

cfg.candidate_blocks = +(getenv('AMP_DSV41_CAND_BLOCKS') || 96);
cfg.candidate_block_size = +(getenv('AMP_DSV41_CAND_BS') || 8);

let i = rindex(WF, '/');
let ws = weights.open(substr(WF, 0, i), substr(WF, i + 1));
let fx = torch.st(FIX);
let A = {}, cache = {pools: {}, layers: {}};
let total = 0;

for (let l in LAYERS) {
	let dev = (l == 3 || l == 21) ? DEV2 : DEV;

	A[l] = dsv41.build_attn(ws, cfg, l, dev);
	cache.layers[l] = dsv41.new_layer_cache(cfg, A[l], 2048);

	if (A[l].kv_source)
		cache.pools[l] = {owner: cache.layers[l], mirrors: []};
}

let X = {}, Y = {}, got = {};
let CHUNKS = null;

for (let l in LAYERS) {
	X[l] = fx.tensor(`x_L${l}`).to('float16');
	Y[l] = fx.tensor(`y_L${l}`);
	got[l] = [];
	total = X[l].shape()[0];
}

CHUNKS = [ 301, 250, total - ND - 551 ];

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

let cand_got = null;

function run(pos0, n) {
	let ctx = {topk: null, cand: null};

	for (let l in LAYERS) {
		let x = X[l].narrow(0, pos0, n).contiguous().cuda(A[l].dev);
		let y = dsv41.attn_forward(cfg, A[l], ctx, cache, cache.layers[l], x, pos0);

		push(got[l], y.to('float32').cpu());

		if (l == 20 && pos0 == 0 && ctx.cand)
			cand_got = ctx.cand.src.cpu();
	}
}

function pass(tag) {
let pos = 0;

for (let c in CHUNKS) {
	run(pos, c);
	pos += c;
}

let nd = total - pos;

// decode steps through the device-position path (what the CUDA graphs capture), capacities of a bucket
const DECPATH = (getenv('AMP_DSV41_DECPATH') != 0);
let sts = {};

function run_dec(pos) {
	let ctx = {topk: null, cand: null};
	let caps = [ 0, 2048, 1024 ];

	for (let l in LAYERS) {
		let dev = A[l].dev;

		if (!sts[dev])
			sts[dev] = dsv41.new_dec_dev(cfg, dev);
	}

	for (let d in keys(sts))
		dsv41.dec_set_pos(cfg, sts[d], pos);

	for (let l in LAYERS) {
		let x = X[l].narrow(0, pos, 1).contiguous().cuda(A[l].dev);
		let y = dsv41.attn_decode(cfg, A[l], ctx, cache, cache.layers[l], x, sts[A[l].dev], caps);

		push(got[l], y.to('float32').cpu());

		if (A[l].kv_source) {
			let r = A[l].ratio;

			dsv41.set_pool_n(cache, l, int((pos + 1) / r));
			dsv41.mirror_entry(cache, l, int(pos / r));
		}
	}
}

for (let k = 0; k < nd; k++) {
	if (DECPATH)
		run_dec(pos);
	else
		run(pos, 1);

	pos++;
}

// candidate blocks of layer 20 (first chunk) against the reference mask
if (fx.list && index(fx.list(), 'cand_prefill') >= 0 && cand_got) {
	let ref = fx.tensor('cand_prefill').to('int32');       // (T0, S) 0/1
	let rows = CHUNKS[0], bs = cfg.candidate_block_size;
	let S = ref.shape()[1];
	let nb = int((S + bs - 1) / bs);
	let bad = 0;
	let cl = cand_got.tolist();
	let rl = ref.narrow(0, 0, rows).tolist();

	for (let r = 0; r < rows; r++) {
		let mine = {};

		for (let b in cl[r])
			if (b >= 0)
				mine[b] = true;

		for (let b = 0; b < nb; b++) {
			// a block is in the reference mask iff any row of it is
			let inref = false;

			for (let c = b * bs; c < b * bs + bs && c < S; c++)
				if (rl[r][c])
					inref = true;

			if ((mine[b] ? true : false) != inref)
				bad++;
		}
	}

	printf('candidate blocks: %d disagreements over %d rows\n', bad, rows);
}

for (let l in LAYERS) {
	let g = torch.cat(got[l], 0);
	let pre = rel_rms(g.narrow(0, 0, pos - nd), Y[l].narrow(0, 0, pos - nd));
	let dec = rel_rms(g.narrow(0, pos - nd, nd), Y[l].narrow(0, pos - nd, nd));

	printf('layer %d (ratio %d): prefill rel rms %.2e, decode %.2e\n', l, cfg.ratios[l], pre, dec);
	// ratio-1 layers pick 512 of ~1000 rows by indexer score: a rounding-level change of the scores (fp16 here)
	// swaps borderline picks and moves the output by ~1e-2 even in the reference itself
	let tol = (cfg.ratios[l] == 1) ? 0.06 : 0.005;

	assert(pre < tol && dec < tol, `layer ${l} differs from the reference`);
}

}

pass('first');

// the same caches again after a reset (a new conversation): the mirrors must not keep the old rows
for (let l in LAYERS) {
	got[l] = [];

	if (A[l].kv_source)
		dsv41.set_pool_n(cache, l, 0);
}

cand_got = null;
pass('after a reset');

printf('dsv41_attn: all ok\n');
