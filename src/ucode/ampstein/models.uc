// ucode/models.uc — model-level helpers on top of the torch module.
//
// Linear layers are routed by weight class, decided per key from what
// the weight set actually carries (see weights.uc):
//
//   - EXL3 quantized:  <key>.trellis (+ .suh/.svh, optional .bias)
//   - plain fp16/bf16: <key>.weight (+ optional .bias)
//
// LinearEXL3 derives everything from the trellis tensor (the format
// spec, see docs/LESSONS.md L2):
//
//   K            = trellis.shape[-1] / 16   (bitrate; half-integer K
//                                           requires the mul1 codebook)
//   in_features  = trellis.shape[0] * 16
//   out_features = trellis.shape[1] * 16
//
// forward(x) computes x @ dequant(trellis) via the vendored exl3_gemm
// kernel, with suh/svh Hadamard scales applied (required — the upstream
// kernel crashes without them, see docs/LESSONS.md L16).
//
// Usage:
//   import * as models from 'ampstein.models.uc';
//   let ws = weights.open('/path/to/model-dir');
//   let layer = models.linear(ws, 'model.layers.0.self_attn.q_proj');
//   let y = layer.forward(x);

/* Rows above which an EXL3 linear dequantizes its weight to fp16 and runs
 * one cuBLAS gemm (torch.exl3_gemm_recon) instead of the on-the-fly
 * trellis-decode gemm (torch.exl3_gemm), which redoes the decode for
 * every row tile — exllamav3's AUTO_RECONSTRUCT_THRESHOLD. Prefill
 * chunks take the recon path, decode steps the decode path. The recon
 * path handles at most RECON_MAX_N output features (the reconstruct
 * slice limit); wider weights (lm_head) always use exl3_gemm. */
const RECON_ROWS = 144;
const RECON_MAX_N = 32768;

/* Device index normalization: null/undefined -> 0. */
function dev_of(d) {
	return (d == null) ? 0 : d;
}

/* EXL3-quantized linear. dev: device index (default 0). The codebook
 * variant is detected from the checkpoint: a `.mul1` marker tensor
 * selects the mul1 codebook (cb=2), a `.mcg` marker the mcg codebook
 * (cb=1) — required by checkpoints quantized with "codebook": "mul1"
 * or "mcg" (e.g. Qwen3.8-27B exl3). Mutually exclusive. */
export function LinearEXL3(ws, key, dev) {
	let s = ws.info(key + '.trellis').shape;

	// half-integer bitrates (e.g. 2.5 bpw) carry 16*K + 8 uint16 per tile
	let rem = s[2] % 16;
	let mul1 = ws.has(key + '.mul1') ? true : false;
	let mcg = ws.has(key + '.mcg') ? true : false;

	if (mul1 && mcg)
		die(`LinearEXL3: ${key}: both .mul1 and .mcg marker tensors present`);

	if (rem != 0 && !mul1)
		die(`LinearEXL3: ${key}: half-integer bitrate (tile width ${s[2]}) requires mul1 support, not yet implemented`);

	let trellis = ws.gpu(key + '.trellis', dev);
	let suh = ws.gpu(key + '.suh', dev);
	let svh = ws.gpu(key + '.svh', dev);
	let bias = ws.has(key + '.bias') ? ws.gpu16(key + '.bias', dev) : null;

	/* NB: 16.0 forces float division — ucode's `/` on two integers is
	 * integer division, which would truncate a half-integer K (tile
	 * width 88 -> K 5 instead of 5.5). */
	let K = s[2] / 16.0;

	return {
		trellis: trellis,
		suh: suh,
		svh: svh,
		bias: bias,
		K: K,
		mul1: mul1,
		mcg: mcg,
		dev: dev_of(dev),
		in_features: s[0] * 16,
		out_features: s[1] * 16,

		forward: function (x) {
			let m = x.shape()[0];
			let out;

			if (m > RECON_ROWS && this.out_features <= RECON_MAX_N) {
				out = torch.exl3_gemm_recon(x.contiguous(), this.trellis,
					this.suh, this.svh, this.K, this.mul1, this.mcg);
			}
			else {
				// exl3_gemm writes every element: no zero fill
				out = torch.empty([m, this.out_features], 'float16',
					'cuda:' + this.dev);
				torch.exl3_gemm(x, this.trellis, out, this.suh, this.svh,
					this.mul1, this.mcg);
			}

			if (this.bias)
				out = out.add(this.bias);

			return out;
		}
	};
}

/* Sequence-aware EXL3 variant: accepts 2-D (m, k) or 3-D (batch, seq,
 * k) input; 3-D is flattened for the kernel and reshaped back. */
export function LinearEXL3Seq(ws, key, dev) {
	let base = LinearEXL3(ws, key, dev);

	return {
		trellis: base.trellis,
		suh: base.suh,
		svh: base.svh,
		bias: base.bias,
		K: base.K,
		mul1: base.mul1,
		mcg: base.mcg,
		dev: base.dev,
		in_features: base.in_features,
		out_features: base.out_features,

		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x :
				x.view([-1, this.in_features]);

			let m = flat.shape()[0];
			let out;

			if (m > RECON_ROWS && this.out_features <= RECON_MAX_N) {
				out = torch.exl3_gemm_recon(flat.contiguous(), this.trellis,
					this.suh, this.svh, this.K, this.mul1, this.mcg);
			}
			else {
				// exl3_gemm writes every element: no zero fill
				out = torch.empty([m, this.out_features], 'float16',
					'cuda:' + this.dev);
				torch.exl3_gemm(flat, this.trellis, out, this.suh, this.svh,
					this.mul1, this.mcg);
			}

			if (this.bias)
				out = out.add(this.bias);

			return (length(s) == 2) ? out :
				out.view([s[0], s[1], this.out_features]);
		}
	};
}

/* Plain fp16/bf16 linear: y = x @ W.T (+ bias). W is (out, in); cast
 * to fp16 at load so activations stay fp16 end-to-end. Accepts 2-D and
 * 3-D input (3-D is flattened and reshaped back). */
export function LinearFP16(ws, key, dev) {
	let w = ws.gpu16(key + '.weight', dev);
	let bias = ws.has(key + '.bias') ? ws.gpu16(key + '.bias', dev) : null;

	return {
		weight: w,
		bias: bias,
		in_features: w.shape()[1],
		out_features: w.shape()[0],

		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x :
				x.view([-1, this.in_features]);

			let y = torch.matmul(flat, this.weight.t());

			if (this.bias)
				y = y.add(this.bias);

			return (length(s) == 2) ? y :
				y.view([s[0], s[1], this.out_features]);
		}
	};
}

/* Class routing: build the right linear implementation for `key`.
 * seq: when true, EXL3 linears accept 3-D input (plain linears always
 * do). Returns null when the key carries no recognizable weight. */
export function linear(ws, key, seq, dev) {
	if (ws.has(key + '.trellis'))
		return seq ? LinearEXL3Seq(ws, key, dev) : LinearEXL3(ws, key, dev);

	if (ws.has(key + '.weight'))
		return LinearFP16(ws, key, dev);

	return null;
}

/* Construct every linear of a transformer layer. Names follow the HF
 * llama layout; keys without a trellis AND without a weight are left
 * out (e.g. absent biases). */
export function layer_linears(ws, layer_key, seq) {
	let names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
	             'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
	let out = {};

	for (let i = 0; i < length(names); i++) {
		let l = linear(ws, layer_key + '.' + names[i], seq);

		if (l)
			out[names[i]] = l;
	}

	return out;
}

/* --- batched decode linears (single cooperative launch) ------------------ */

/* Pointer-table bundle over EXL3 linears that SHARE one input: the
 * vendored exl3_mgemm kernel runs all of them as ONE cooperative
 * launch (measured on the 1B model: q+k+v 31.8 -> 15.9 us, gate+up
 * 63.7 -> 40.4 us vs separate gemms).
 *
 * Two modes:
 *
 *   multi(linears, out)     equal output widths; out is the stacked
 *                           (num, m, n) output buffer written directly
 *
 *   sliced(linears, outs)   unequal widths (Q/K/V); every matrix is
 *                           cut into equal-width column slices scheduled
 *                           as independent groups; outputs are written
 *                           through a per-slice pointer table into the
 *                           caller's buffers (one per linear)
 *
 * Both write into buffers supplied at construction — the addresses are
 * baked into the pointer tables, so buffers must be persistent (they
 * are: decode buffers live for the decoder's lifetime).
 *
 * Numerics: mgemm is a different kernel with a different accumulation
 * order — outputs agree with the separate-gemm path to fp16 noise
 * (rel-rms ~1e-6 measured), not bit-exactly. This is the same tradeoff
 * exllamav3 makes: its own attention runs Q/K/V through sliced mgemm. */

function gcd(a, b) {
	while (b > 0) {
		let t = a % b;
		a = b;
		b = t;
	}

	return a;
}

function ptr_table(linears, what, dev) {
	let vals = [];

	for (let i = 0; i < length(linears); i++)
		push(vals, linears[i][what].data_ptr());

	return torch.ptr_array(vals, dev);
}

/* Bundles fuse several gemms into ONE kernel launch, which applies one
 * codebook and one bitrate — every matrix must agree (exllamav3 guards
 * the same fusion with a K-equality check, see attn.py/multilinear.py;
 * mixed-bitrate checkpoints like the Qwen3.8 quants assign K per
 * tensor). */
function bundle_check(linears) {
	let K = linears[0].K;
	let mul1 = linears[0].mul1;
	let mcg = linears[0].mcg;
	let kin = linears[0].in_features;

	for (let i = 1; i < length(linears); i++) {
		let w = linears[i];

		if (w.K != K || w.mul1 != mul1 || w.mcg != mcg ||
		    w.in_features != kin)
			die(`models.bundle: fused linears must share K, ` +
			    `in_features and codebook (matrix ${i}: K=${w.K} ` +
			    `mul1=${w.mul1} mcg=${w.mcg} in=${w.in_features} ` +
			    `vs K=${K} mul1=${mul1} mcg=${mcg} in=${kin})`);
	}
}

/* Equal-shape bundle. out: (num, m, n) fp16 CUDA buffer. run(x) writes
 * all outputs stacked into out; x is (1, m, k) 3-D. */
export function multi(linears, out, dev) {
	bundle_check(linears);

	let dv = (dev == null) ? 'cuda' : 'cuda:' + dev;

	let B = ptr_table(linears, 'trellis', dev);
	let suh = ptr_table(linears, 'suh', dev);
	let svh = ptr_table(linears, 'svh', dev);
	let num = length(linears);
	let k_in = linears[0].in_features;
	let m = out.shape()[1];
	let Ahad = torch.zeros([num, m, k_in], 'float16', dv);
	let K = linears[0].K;
	let mul1 = linears[0].mul1;
	let mcg = linears[0].mcg;

	return {
		out: out,

		run: function (x) {
			torch.mgemm(x, B, out, suh, Ahad, svh, K, mcg, mul1);
		}
	};
}

/* Sliced bundle for unequal widths. outs: array of output buffers, one
 * per linear, each (1, m, out_features) fp16 CUDA. Slice width = gcd of
 * the widths (must be a multiple of 128 — the kernel's column-block
 * granularity). run(x) writes every slice in place into its buffer. */
export function sliced(linears, outs, dev) {
	bundle_check(linears);

	let dv = (dev == null) ? 'cuda' : 'cuda:' + dev;
	let di = (dev == null) ? 0 : dev;

	let width = linears[0].out_features;

	for (let i = 1; i < length(linears); i++)
		width = gcd(width, linears[i].out_features);

	if (width % 128 != 0 || width < 128)
		die(`models.sliced: unsuitable slice width ${width}`);

	let slices = [];

	for (let i = 0; i < length(linears); i++) {
		let w = linears[i];
		let tshape = w.trellis.shape();
		let twidth = tshape[2];                  // 16*K uint16 per tile
		let trellis_row = tshape[1] * twidth;    // uint16 per row-block

		for (let n0 = 0; n0 < w.out_features; n0 += width)
			push(slices, {
				t: w.trellis.data_ptr() + (n0 / 16) * twidth * 2,
				svh: w.svh.data_ptr() + n0 * 2,
				out: i,
				n0: n0,
				stride: w.out_features,
				src: i
			});
	}

	let ns = length(slices);
	let m = outs[0].shape()[1];
	let B = torch.ptr_array(map(slices, function (s) {return s.t;}), dev);
	let svhP = torch.ptr_array(map(slices, function (s) {return s.svh;}), dev);
	let suhP = ptr_table(linears, 'suh', dev);
	let cP = torch.ptr_array(map(slices, function (s) {
		return outs[s.out].data_ptr() + s.n0 * 2;
	}), dev);
	let sizeN = torch.from_array(map(slices, function (s) {return width;}), 'int32').cuda(di);
	let nStride = torch.from_array(map(slices, function (s) {return s.stride;}), 'int32').cuda(di);
	let hadSrc = torch.from_array(map(slices, function (s) {return s.src;}), 'int32').cuda(di);
	let carrier = torch.zeros([ns, m, width], 'float16', dv);
	let Ahad = torch.zeros([length(linears), m, linears[0].in_features], 'float16', dv);
	let K = linears[0].K;
	let mul1 = linears[0].mul1;
	let mcg = linears[0].mcg;
	let opts = {
		size_n_list: sizeN,
		c_ptrs: cP,
		n_stride_list: nStride,
		had_src_list: hadSrc,
		num_had_src: length(linears)
	};

	return {
		slices: ns,
		width: width,

		run: function (x) {
			torch.mgemm(x, B, carrier, suhP, Ahad, svhP, K, mcg, mul1, opts);
		}
	};
}
/* Column sub-linear of an EXL3 linear: the output columns of `ranges`
 * ([start, length] pairs, multiples of 256 — the trellis tiles are 16
 * columns, the output Hadamard works in 128-column blocks, the int8 GEMV
 * wants 256), concatenated. Computes exactly those columns of the full
 * product (suh is per input, svh per output column, trellis tiles slice
 * along their column dimension). Returns a LinearEXL3Seq-like object
 * plus map(i): sub-column index -> original column. */
export function sub_columns(lin, ranges) {
	let ts = [], ss = [], offs = [], n = 0;

	for (let i = 0; i < length(ranges); i++) {
		let a = ranges[i][0], len = ranges[i][1];

		if (a % 256 || len % 256 || a + len > lin.out_features)
			die(`models.sub_columns: range [${a}, ${len}] not 256-aligned/in bounds`);

		push(ts, lin.trellis.narrow(1, a / 16, len / 16));
		push(ss, lin.svh.narrow(0, a, len));
		push(offs, [ n, a, len ]);
		n += len;
	}

	let sub = {
		trellis: torch.cat(ts, 1).contiguous(),
		suh: lin.suh,
		svh: torch.cat(ss, 0).contiguous(),
		bias: null,
		K: lin.K,
		mul1: lin.mul1,
		mcg: lin.mcg,
		dev: lin.dev,
		in_features: lin.in_features,
		out_features: n,

		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x : x.view([-1, this.in_features]);
			let out = torch.empty([flat.shape()[0], this.out_features],
				'float16', 'cuda:' + this.dev);

			torch.exl3_gemm(flat, this.trellis, out, this.suh, this.svh,
				this.mul1, this.mcg);

			return (length(s) == 2) ? out : out.view([s[0], s[1], this.out_features]);
		},

		map: function (i) {
			for (let k = 0; k < length(offs); k++)
				if (i < offs[k][0] + offs[k][2])
					return offs[k][1] + (i - offs[k][0]);

			return -1;
		}
	};

	return sub;
}

/* EXL3 linear from explicit parts (tensor-parallel shards): p = {
 * trellis, suh, svh, K, mul1, mcg, dev}. Same forward as LinearEXL3Seq
 * (2-D or 3-D input; reconstruct + cuBLAS above RECON_ROWS rows). */
export function LinearEXL3Parts(p) {
	let s = p.trellis.shape();

	return {
		trellis: p.trellis, suh: p.suh, svh: p.svh, bias: null,
		K: p.K, mul1: p.mul1, mcg: p.mcg, dev: p.dev,
		in_features: s[0] * 16,
		out_features: s[1] * 16,

		forward: function (x) {
			let sh = x.shape();
			let flat = (length(sh) == 2) ? x : x.view([-1, this.in_features]);
			let m = flat.shape()[0];
			let out;

			if (m > RECON_ROWS && this.out_features <= RECON_MAX_N) {
				out = torch.exl3_gemm_recon(flat.contiguous(), this.trellis,
					this.suh, this.svh, this.K, this.mul1, this.mcg);
			}
			else {
				out = torch.empty([m, this.out_features], 'float16',
					'cuda:' + this.dev);
				torch.exl3_gemm(flat, this.trellis, out, this.suh, this.svh,
					this.mul1, this.mcg);
			}

			return (length(sh) == 2) ? out :
				out.view([sh[0], sh[1], this.out_features]);
		}
	};
}

/* Tensor-parallel shards of an EXL3 linear, placed on device `dev`:
 *
 *   cols(lin, ranges, dev)  output columns ([start, length] pairs,
 *                           256-aligned), concatenated
 *   rows(lin, k0, klen, dev) input rows [k0, k0 + klen) (128-aligned:
 *                           the input Hadamard works in 128-blocks) —
 *                           a PARTIAL product: the shards' outputs sum
 *                           to the full result (the output Hadamard and
 *                           svh are linear in the accumulated sum)
 */
export function cols(lin, ranges, dev) {
	let sub = sub_columns(lin, ranges);

	return LinearEXL3Parts({
		trellis: sub.trellis.cuda(dev), suh: sub.suh.cuda(dev),
		svh: sub.svh.cuda(dev), K: lin.K, mul1: lin.mul1, mcg: lin.mcg,
		dev: dev
	});
}

export function rows(lin, k0, klen, dev) {
	if (k0 % 128 || klen % 128 || k0 + klen > lin.in_features)
		die(`models.rows: range [${k0}, ${klen}] not 128-aligned/in bounds`);

	return LinearEXL3Parts({
		trellis: lin.trellis.narrow(0, k0 / 16, klen / 16).contiguous().cuda(dev),
		suh: lin.suh.narrow(0, k0, klen).contiguous().cuda(dev),
		svh: lin.svh.cuda(dev), K: lin.K, mul1: lin.mul1, mcg: lin.mcg,
		dev: dev
	});
}

/* fp16 linear (y = x W^T) restricted to output rows [n0, n0 + len) of
 * W, on device `dev`. */
export function fp16_cols(lin, n0, len, dev) {
	let w = lin.weight.narrow(0, n0, len).contiguous().cuda(dev);

	return {
		weight: w, bias: null, in_features: w.shape()[1], out_features: len,

		forward: function (x) {
			let s = x.shape();
			let flat = (length(s) == 2) ? x : x.view([-1, this.in_features]);
			let y = torch.matmul(flat, this.weight.t());

			return (length(s) == 2) ? y : y.view([s[0], s[1], this.out_features]);
		}
	};
}
