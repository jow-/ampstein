// tests/vision_preprocess.uc — native image preprocessing vs exllamav3.
// Fixtures (tests/oracle/make_vision_oracle.py): vision_test.png and
// vision_oracle.safetensors (patches/grid computed by exllamav3 with
// PIL). The decode + alpha compositing + smart resize + bicubic +
// normalize + patch-order flatten must match the PIL reference
// bit-for-bit in fp16.
//
// No GPU needed. Model-free apart from the fixtures (skips if absent).

import { stat, readfile } from 'fs';

const DIR = `${sourcepath(0, true)}/fixtures`;

if (!stat(`${DIR}/vision_oracle.safetensors`) || !stat(`${DIR}/vision_test.png`)) {
	warn('skip: vision fixtures missing (run tests/oracle/make_vision_oracle.py)\n');
	exit(125);
}

const OPTS = {
	patch_size: 16, merge_size: 2, temporal_patch_size: 2,
	min_pixels: 65536, max_pixels: 16777216,
	image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5],
};

let ref = torch.st(`${DIR}/vision_oracle.safetensors`);
let want = ref.tensor('patches');
let wgrid = ref.tensor('grid').tolist();

let png = readfile(`${DIR}/vision_test.png`);
let r = torch.image_prepare(png, OPTS);

assert(r.width == 301 && r.height == 223, 'source size');
assert(r.new_width == 320 && r.new_height == 224, 'smart-resized size');
assert(r.grid[0] == wgrid[0] && r.grid[1] == wgrid[1] && r.grid[2] == wgrid[2],
	'grid');
assert(r.patches.dtype() == 'float16', 'dtype');
assert(r.patches.shape()[0] == want.shape()[0] &&
       r.patches.shape()[1] == want.shape()[1], 'shape');

let diff = r.patches.to('float32').sub(want.to('float32')).abs();
let mx = diff.max().item();

printf('patches max |diff| vs PIL oracle: %g\n', mx);
assert(mx == 0, 'patches bit-exact vs exllamav3/PIL');

/* error paths */
function raises(f, what) {
	let ok = false;
	try { f(); } catch (e) { ok = true; }
	assert(ok, what);
}

raises(() => torch.image_prepare('not an image', OPTS), 'garbage data');
raises(() => torch.image_prepare(png, { ...OPTS, patch_size: 0 }), 'bad geometry');
raises(() => torch.image_prepare(png, { patch_size: 16 }), 'missing options');
raises(() => torch.image_prepare(png, { ...OPTS, image_mean: [0.5] }), 'bad mean');
raises(() => torch.image_prepare(substr(png, 0, 100), OPTS), 'truncated png');

/* max_pixels cap shrinks the grid (and keeps merge alignment) */
let small = torch.image_prepare(png, { ...OPTS, min_pixels: 1024, max_pixels: 16384 });
assert(small.grid[1] % 2 == 0 && small.grid[2] % 2 == 0, 'merge aligned');
assert(small.new_width * small.new_height <= 16384, 'max_pixels honoured');

print('vision_preprocess: ok\n');
