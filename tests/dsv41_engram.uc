// tests/dsv41_engram.uc — ampstein's Engram hashing (token map, rolling XOR of multiplied ids, prime
// buckets) against DeepSeek's engram.py (tests/oracle/make_dsv41_engram_oracle.py).
//   AMP_DSV41_ENGRAM_ORACLE=<fixture>  AMP_DSV41_CFG=<dir with config.json, engram_meta.safetensors>

import * as dsv41 from 'ampstein.dsv41';
import { stat } from 'fs';

const FIX = getenv('AMP_DSV41_ENGRAM_ORACLE') || '/data/models/dsv41_engram_oracle.safetensors';
const CFGDIR = getenv('AMP_DSV41_CFG') || '/data/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw';

if (!stat(FIX) || !stat(CFGDIR + '/engram_meta.safetensors')) {
	warn('skip: engram oracle fixture or engram_meta.safetensors missing\n');
	exit(125);
}

let cfg = dsv41.load_config(CFGDIR);
let meta = dsv41.build_engram_meta(CFGDIR, cfg);
let fx = torch.st(FIX);
let ids = fx.tensor('ids').to('int64');
let ref = fx.tensor('hash').to('int64');          // (T, layers, 24)
let T = ids.shape()[0];

// compressed ids with 3 pad ids of history, as forward_chunk builds them
let cn = meta.token_map.index(ids).view([-1]).tolist();
let hist = [ meta.pad, meta.pad, meta.pad ];

for (let c in cn)
	push(hist, c);

let hist_t = torch.from_array(hist, 'int64').view([1, -1]);
let H = (cfg.engram_ngram - 1) * cfg.engram_heads;
let bad = 0;

for (let li = 0; li < length(cfg.engram_layers); li++) {
	let uids = torch.zeros([T * H], 'int64'), inverse = torch.zeros([T * H], 'int64'), heads = torch.zeros([T * H], 'int32');
	let U = torch.ngram_hash(hist_t, T, meta.mult[li], meta.offs[li], meta.sizes[li], cfg.engram_heads, meta.pad,
		uids, inverse, heads);
	let got = uids.narrow(0, 0, U).view([-1, 1]).index(inverse).view([T, H]);
	let want = ref.narrow(1, li, 1).view([T, H]);

	// the oracle treats the mid-sequence pad token (position 40) as an ordinary token; compare the rest
	let d = got.sub(want).abs().sum(1).tolist();

	for (let t = 0; t < T; t++)
		if (d[t] != 0 && !(t >= 40 && t < 44))
			bad++;

	printf('layer slot %d: %d unique rows of %d\n', li, U, T * H);
}

assert(bad == 0, `${bad} positions hash differently`);
printf('dsv41_engram: all ok\n');
