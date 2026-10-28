// ggml dequantization kernels (torch.gguf_dequant, src/torch/amp_gguf.cu) against the gguf package's reference
// (tests/oracle/make_gguf_dequant.py): float32 output is bit-identical, float16 / bfloat16 equal the rounded reference.

const FIXTURE = `${sourcepath(0, true)}/fixtures/gguf_small.gguf`;
const REF = `${sourcepath(0, true)}/fixtures/gguf_small_ref.safetensors`;

import { stat } from 'fs';

if (!stat(FIXTURE) || !stat(REF)) {
	warn('skip: tests/fixtures/gguf_small*.* missing (tests/oracle/make_gguf_dequant.py)\n');
	exit(125);
}

if (torch.device_count() < 1) {
	warn('skip: needs a GPU\n');
	exit(125);
}

let f = torch.gguf(FIXTURE);
let r = torch.st(REF);

function maxdiff(a, b) {
	return a.to('float32').sub(b.to('float32')).abs().max().item();
}

for (let name in [ 'q4_0', 'q8_0', 'iq4_nl', 'q5_k', 'iq2_xxs', 'iq2_xs', 'iq1_m', 'iq1_s',
	'q2_0', 'q2_k', 'q4_k', 'q3_k', 'q6_k', 'iq2_s', 'iq3_xxs', 'iq3_s', 'iq4_xs', 'stack' ]) {
	let ty = f.info(name).dtype;
	let ref = r.tensor(name).cuda();
	let raw = f.tensor(name).cuda();

	let o32 = torch.gguf_dequant(raw, ty, 'float32');

	assert(join('x', o32.shape()) == join('x', ref.shape()), `${name}: shape ${o32.shape()} vs ${ref.shape()}`);

	let d32 = maxdiff(o32, ref);
	let d16 = maxdiff(torch.gguf_dequant(raw, ty, 'float16'), ref.to('float16'));
	let db = maxdiff(torch.gguf_dequant(raw, ty), ref.to('bfloat16'));       // (the default: bfloat16)

	printf('%-8s %-8s max |diff|  f32 %g  f16 %g  bf16 %g\n', name, ty, d32, d16, db);
	assert(d32 == 0, `${name}: float32 differs from the reference`);
	assert(d16 == 0 && db == 0, `${name}: float16 / bfloat16 differ from the rounded reference`);
}

// matrix-vector products: gguf_gemv (dense, 1..8 rows) and gguf_moe_gemv (a stack of experts, one per slot) against
// x @ dequantized^T in fp32
function rel_err(a, b) {
	let d = a.to('float32').sub(b.to('float32')).abs().max().item();

	return d / (b.to('float32').abs().max().item() + 1e-9);
}

for (let name in [ 'q4_0', 'q8_0', 'iq4_nl', 'q5_k', 'iq2_xxs', 'iq2_xs', 'iq1_m', 'iq1_s',
	'q2_0', 'q2_k', 'q4_k', 'q3_k', 'q6_k', 'iq2_s', 'iq3_xxs', 'iq3_s', 'iq4_xs' ]) {
	let ty = f.info(name).dtype;
	let raw = f.tensor(name).cuda();
	let ref = r.tensor(name).cuda();                 // (24, 768) float32
	let worst = 0;

	for (let M in [ 1, 3, 8 ]) {
		let x = torch.rand([M, 768], 'float32', 'cuda').sub(0.5).to('float16');
		let y = torch.gguf_gemv(x, raw, ty);
		let e = rel_err(y, torch.matmul(x.to('float32'), ref.t()));

		assert(join('x', y.shape()) == `${M}x24` && y.dtype() == 'float16', `${name}: gemv result ${y.shape()}`);
		worst = max(worst, e);
	}

	// (these types run in int8 against a quantized activation row: ~1 % of the largest output)
	assert(worst < 2e-2, `${name}: gemv differs from the matmul by ${worst}`);
	printf('%-8s %-8s gemv rel err %g\n', name, ty, worst);
}

// grouped gemv (the attention's wo_a): G groups of rows, group g seeing columns [g * K, (g + 1) * K) of x, in one launch; the
// result equals the per-group launches
for (let name in [ 'q5_k', 'iq2_xxs', 'iq1_m', 'iq1_s', 'iq3_s', 'q8_0' ]) {
	let ty = f.info(name).dtype, raw = f.tensor(name).cuda();                 // (24, 768)
	let G = 3, R = 8, K = 768;
	let x = torch.rand([2, G * K], 'float32', 'cuda').sub(0.5).to('float16');
	let y = torch.gguf_gemv_grouped(x, raw, ty, G);
	let parts = [];

	for (let g = 0; g < G; g++)
		push(parts, torch.gguf_gemv(x.narrow(1, g * K, K).contiguous(), raw.narrow(0, g * R, R), ty));

	let e = rel_err(y, torch.cat(parts, 1));

	assert(join('x', y.shape()) == `2x${G * R}`, `${name}: grouped gemv shape ${y.shape()}`);
	assert(e < 1e-6, `${name}: grouped gemv differs from the per-group launches (${e})`);
}

{
	let raw = f.tensor('stack').cuda(), ref = r.tensor('stack').cuda();        // (5, 6, 132) bytes, (5, 6, 512) float
	let x = torch.rand([2, 512], 'float32', 'cuda').sub(0.5).to('float16');
	let ids = torch.from_array([ 3, -1, 0, 4, 3 ], 'int64', 'cuda');
	let y = torch.gguf_moe_gemv(x, ids, raw, 'IQ2_XXS', 3);                   // slot s reads x row s / 3
	let want = torch.zeros([5, 6], 'float32', 'cuda');
	let xf = x.to('float32');

	assert(join('x', y.shape()) == '5x6' && y.dtype() == 'float32', 'moe gemv result');

	for (let s = 0; s < 5; s++) {
		let e = [ 3, -1, 0, 4, 3 ][s];

		if (e >= 0)
			want.narrow(0, s, 1).copy_(torch.matmul(xf.narrow(0, int(s / 3), 1), ref.narrow(0, e, 1).view([6, 512]).t()));
	}

	let e = rel_err(y, want);

	printf('moe gemv rel err %g\n', e);
	assert(e < 2e-2, 'moe gemv differs from the matmul');          // (IQ2_XXS: int8 activations)
	assert(y.narrow(0, 1, 1).abs().max().item() == 0, 'an expert id of -1 contributes nothing');

	// the same grouped gemv through a pointer table (resident / mapped cold stacks)
	let ptrs = [];
	let rb = raw.shape()[2], nrows = raw.shape()[1];

	for (let e = 0; e < raw.shape()[0]; e++)
		push(ptrs, raw.data_ptr() + e * nrows * rb);

	let pt = torch.from_array(ptrs, 'int64', 'cuda');
	let ypt = torch.gguf_moe_gemv_pt(x, ids, pt, 'IQ2_XXS', 3, nrows);
	let ep = rel_err(ypt, y);

	printf('moe gemv pointer-table rel err %g\n', ep);
	assert(ep < 1e-6, 'pointer-table moe gemv differs from the stack moe gemv');
}

// fused gate / up / SwiGLU (dense and experts) and the per-slot output scale, against the separate launches
{
	let lim = 10.0;
	let act = (g, u) => g.clamp(null, lim).mul(g.clamp(null, lim).sigmoid()).mul(u.clamp(-lim, lim));

	for (let name in [ 'q5_k', 'iq2_xxs', 'iq2_xs', 'iq1_m', 'iq1_s', 'iq3_s', 'q8_0' ]) {
		let ty = f.info(name).dtype, wg = f.tensor(name).cuda(), wu = f.tensor('q8_0').cuda();
		let up = (name == 'q8_0') ? f.tensor('q5_k').cuda() : f.tensor(name).narrow(0, 0, 24).cuda();     // (a second matrix of the same type)
		let x = torch.rand([3, 768], 'float32', 'cuda').sub(0.5).to('float16');                    // outputs far beyond the clamp (10) but inside fp16
		let h = torch.gguf_gate_up(x, wg, wg.narrow(0, 0, 24).contiguous(), ty, lim);
		let g = torch.gguf_gemv(x, wg, ty).to('float32'), u = torch.gguf_gemv(x, wg.narrow(0, 0, 24).contiguous(), ty).to('float32');
		let e = rel_err(h, act(g, u).to('float16'));

		assert(join('x', h.shape()) == '3x24' && h.dtype() == 'float16', `${name}: gate_up result ${h.shape()}`);
		assert(e < 5e-3, `${name}: fused gate/up differs from the separate launches (${e})`);
	}

	// experts: the stack of 5 x 6 rows x IQ2_XXS (two copies as gate and up), ids with a -1, a scale per slot
	let st = f.tensor('stack').cuda();
	let x = torch.rand([2, 512], 'float32', 'cuda').sub(0.5).to('float16');
	let ids = torch.from_array([ 3, -1, 0, 4, 3 ], 'int64', 'cuda');
	let h = torch.gguf_moe_gate_up(x, ids, st, st, 'IQ2_XXS', 3, lim);
	let g = torch.gguf_moe_gemv(x, ids, st, 'IQ2_XXS', 3);
	let e = rel_err(h, act(g, g).to('float16'));

	assert(join('x', h.shape()) == '5x6' && e < 5e-3, `moe gate_up differs (${e})`);
	assert(h.narrow(0, 1, 1).abs().max().item() == 0, 'moe gate_up: an expert id of -1 gives zeros');

	let sc = torch.from_array([ 0.5, 2.0, 1.0, -1.0, 3.0 ], 'float32', 'cuda');
	let hh = torch.rand([5, 512], 'float32', 'cuda').sub(0.5).to('float16');
	let plain = torch.gguf_moe_gemv(hh, ids, st, 'IQ2_XXS', 1), scaled = torch.gguf_moe_gemv(hh, ids, st, 'IQ2_XXS', 1, sc);

	assert(rel_err(scaled, plain.mul(sc.view([5, 1]))) < 1e-6, 'moe gemv: the per-slot scale');
}

// embedding rows by id: the same values as the dequantized table, any id shape, an id outside the table gives row 0
for (let name in [ 'q4_0', 'q5_k', 'q8_0', 'iq2_xxs', 'iq1_m', 'iq1_s', 'iq4_nl',
	'q2_0', 'q2_k', 'q4_k', 'q3_k', 'q6_k', 'iq2_s', 'iq3_xxs', 'iq3_s', 'iq4_xs' ]) {
	let ty = f.info(name).dtype, raw = f.tensor(name).cuda();
	let full = torch.gguf_dequant(raw, ty, 'float16');
	let ids = torch.from_array([ [ 3, 0 ], [ 23, 99 ] ], 'int64', 'cuda');
	let got = torch.gguf_embed(raw, ty, ids);

	assert(join('x', got.shape()) == '2x2x768' && got.dtype() == 'float16', `${name}: embed shape ${got.shape()}`);
	assert(rel_err(got.narrow(0, 0, 1).view([2, 768]), torch.cat([ full.narrow(0, 3, 1), full.narrow(0, 0, 1) ], 0)) == 0, `${name}: embed rows 3, 0`);
	assert(rel_err(got.narrow(0, 1, 1).view([2, 768]), torch.cat([ full.narrow(0, 23, 1), full.narrow(0, 0, 1) ], 0)) == 0, `${name}: embed row 23 / out of range`);
}

// errors
let raised = (fn) => { try { fn(); } catch (e) { return true; } return false; };
let raw = f.tensor('q8_0').cuda();

assert(raised(() => torch.gguf_dequant(raw, 'Q2_K')), 'unsupported type raises');
assert(raised(() => torch.gguf_dequant(raw, 'Q8_0', 'int8')), 'bad dtype raises');
assert(raised(() => torch.gguf_dequant(f.tensor('q8_0'), 'Q8_0')), 'a CPU tensor raises');
assert(raised(() => torch.gguf_dequant(f.tensor('plain_f32').cuda(), 'Q8_0')), 'a non-uint8 tensor raises');
assert(raised(() => torch.gguf_dequant(raw.narrow(1, 0, 100), 'Q8_0')), 'a row that is no whole number of blocks raises');
assert(raised(() => torch.gguf_gemv(torch.zeros([9, 768], 'float16', 'cuda'), raw, 'Q8_0')), 'gemv with 9 rows raises');
assert(raised(() => torch.gguf_gemv_grouped(torch.zeros([1, 1536], 'float16', 'cuda'), raw, 'IQ4_NL', 2)), 'grouped gemv of IQ4_NL raises');
assert(raised(() => torch.gguf_gemv_grouped(torch.zeros([1, 1000], 'float16', 'cuda'), raw, 'Q8_0', 3)), 'grouped gemv with a ragged x raises');
assert(raised(() => torch.gguf_gemv(torch.zeros([1, 800], 'float16', 'cuda'), raw, 'Q8_0')), 'gemv with a K that does not match raises');
assert(raised(() => torch.gguf_gemv(torch.zeros([1, 768], 'float32', 'cuda'), raw, 'Q8_0')), 'gemv with float32 x raises');
assert(raised(() => torch.gguf_moe_gemv(torch.zeros([1, 768], 'float16', 'cuda'), torch.zeros([2], 'int32', 'cuda'), raw.view([1, 24, -1]), 'Q8_0', 1)), 'moe gemv with int32 ids raises');

// the picks of a verify step grouped by expert (gguf_moe_group): the grouped kernels give exactly the values of one slot per pick, for repeated,
// negative and out-of-shard picks, with and without a shard offset
{
	let raw = f.tensor('stack').cuda();                                         // (5, 6, 132) bytes: 5 experts, 6 rows, K = 512
	let T = 3, Kt = 3;
	let x = torch.rand([T, 512], 'float32', 'cuda').sub(0.5).to('float16');
	let pick = [ 3, 0, 4,  3, 1, -1,  0, 3, 7 ];                                // (7: past the shard)
	let ids = torch.from_array(pick, 'int64', 'cuda');
	let grp = torch.gguf_moe_group(ids);
	let h = torch.rand([9, 512], 'float32', 'cuda').sub(0.5).to('float16');
	let scale = torch.rand([9], 'float32', 'cuda');

	assert(join('x', grp.shape()) == '9x9' && grp.dtype() == 'int32', 'moe_group shape');

	for (let e0 in [ 0, 1 ]) {
		let a = torch.gguf_moe_gemv(x, ids, raw, 'IQ2_XXS', Kt, null, e0), b = torch.gguf_moe_gemv(x, ids, raw, 'IQ2_XXS', Kt, null, e0, grp, T);
		let c = torch.gguf_moe_gemv(h, ids, raw, 'IQ2_XXS', 1, scale, e0), d = torch.gguf_moe_gemv(h, ids, raw, 'IQ2_XXS', 1, scale, e0, grp, T);
		let u = torch.gguf_moe_gate_up(x, ids, raw, raw, 'IQ2_XXS', Kt, 7.0, e0), v = torch.gguf_moe_gate_up(x, ids, raw, raw, 'IQ2_XXS', Kt, 7.0, e0, grp, T);

		assert(rel_err(a, b) == 0 && rel_err(c, d) == 0 && rel_err(u, v) == 0, `grouped expert kernels differ from one slot per pick (e0 ${e0})`);
	}

	printf('moe_group: grouped == per-pick (exact)\n');
}

// the dense gemv of ONE row quantizes its activation inside the kernel, of two rows by a kernel of its own: the same bits
for (let name in [ 'q5_k', 'iq2_xxs', 'iq1_m', 'q8_0' ]) {
	let ty = f.info(name).dtype, raw = f.tensor(name).cuda();
	let K = 768;
	let x1 = torch.rand([1, K], 'float32', 'cuda').sub(0.5).to('float16');
	let y1 = torch.gguf_gemv(x1, raw, ty), y2 = torch.gguf_gemv(torch.cat([x1, x1], 0), raw, ty);

	assert(rel_err(y1, y2.narrow(0, 0, 1)) == 0, `${name}: the in-kernel activation quantization differs from the separate kernel`);
}

print('gguf_dequant: all ok\n');
