// tests/mrope.uc — multimodal rope positions and embedding overlay.
// Fixture: tests/fixtures/mrope_oracle.safetensors (exllamav3's
// RoPE.get_mrope_freqs on a prompt with two images; make_mrope_oracle.py).
//
// qwen35.rope_rows must reproduce the oracle's per-row cos/sin whether
// the prompt is processed as one chunk or split anywhere (inside an
// image, across boundaries, text-only tail via the table), and
// mm_off must give the post-image text offset. embed_rows must put the
// vision embeddings on exactly the image rows.

import * as qwen35 from 'ampstein.qwen35';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';
const FIX = `${sourcepath(0, true)}/fixtures/mrope_oracle.safetensors`;

if (!stat(MODEL_DIR) || !stat(FIX)) {
	warn('skip: model dir or mrope fixture missing\n');
	exit(125);
}

let cfg = qwen35.load_config(MODEL_DIR);
let ref = torch.st(FIX);
let meta = ref.tensor('meta').tolist();
let s1 = meta[0], n1 = meta[1], s2 = meta[2], n2 = meta[3], nxt = meta[4], S = meta[5];
let want_cos = ref.tensor('cos').cuda(0), want_sin = ref.tensor('sin').cuda(0);

assert(cfg.mrope_section[0] == 11 && cfg.vision_merge == 2, 'config');

let rc = qwen35.rope_cache(cfg, 256, 0);
let cl = { dev: 0, cos: rc.cos, sin: rc.sin };

let mm = qwen35.mm_build(cfg, [
	{ start: s1, grid: [1, 14, 20], emb: null },
	{ start: s2, grid: [1, 8, 6], emb: null }
]);

assert(qwen35.mm_off(mm, S) + S == nxt, 'next text position');
assert(qwen35.mm_off(mm, 0) == 0 && qwen35.mm_off(mm, s1) == 0, 'offset before image');

function check(chunks) {
	let pos = 0, worst = 0.0;

	for (let len in chunks) {
		let r = qwen35.rope_rows(cfg, mm, cl, pos, len);
		let dc = r.cos.sub(want_cos.narrow(0, pos, len)).abs().max().item();
		let ds = r.sin.sub(want_sin.narrow(0, pos, len)).abs().max().item();

		if (dc > worst) worst = dc;
		if (ds > worst) worst = ds;

		pos += len;
	}

	assert(pos == S, 'chunks cover the prompt');
	assert(worst < 1e-4, `rope rows vs oracle, chunks ${chunks}: max diff ${worst}`);

	return worst;
}

printf('whole prompt: %g\n', check([S]));
printf('split inside images: %g\n', check([20, 40, 12, 20, 10]));
printf('1-token chunks around the boundaries: %g\n',
	check([s1 - 1, 1, 1, n1 - 2, 1, 1, S - (s1 + n1 + 1)]));

/* text-only region uses the shifted table rows (no custom build) */
let tail = qwen35.rope_rows(cfg, mm, cl, S - 4, 4);
assert(tail.cos.sub(want_cos.narrow(0, S - 4, 4)).abs().max().item() < 1e-4,
	'tail rows from the table');

/* no images: plain table rows */
let plain = qwen35.rope_rows(cfg, null, cl, 3, 5);
assert(plain.cos.sub(rc.cos.narrow(0, 3, 5)).abs().max().item() == 0, 'no-mm rows');

/* embed_rows: overlay lands on the image rows only */
let model = { cfg: cfg, devs: [0],
	embed_tokens: torch.zeros([250000, 16], 'float16', 'cuda:0') };
let emb1 = torch.full([n1, 16], 7.0, 'float16', 'cuda:0');
let emb2 = torch.full([n2, 16], 9.0, 'float16', 'cuda:0');
let mm2 = qwen35.mm_build(cfg, [
	{ start: s1, grid: [1, 14, 20], emb: emb1 },
	{ start: s2, grid: [1, 8, 6], emb: emb2 }
]);
let ids = [];
for (let i = 0; i < S; i++)
	push(ids, (i >= s1 && i < s1 + n1) || (i >= s2 && i < s2 + n2) ?
		qwen35.MARK_BASE + 5 : 3);

let h = qwen35.embed_rows(model, mm2, slice(ids, 10, 90), 10).to('float32')
	.sum(2).view([-1]).tolist();      // row sums: 0 text, 7*16 / 9*16 image

for (let i = 0; i < 80; i++) {
	let p = 10 + i;
	let want = (p >= s1 && p < s1 + n1) ? 7.0 * 16 :
		(p >= s2 && p < s2 + n2) ? 9.0 * 16 : 0.0;

	assert(h[i] == want, `overlay row ${p}: ${h[i]} != ${want}`);
}

assert(qwen35.mm_marker('00000ab-rest') > qwen35.MARK_BASE - 1, 'marker range');

print('mrope: ok\n');
