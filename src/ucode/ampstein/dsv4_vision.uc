// ucode/dsv4_vision.uc — DeepSeek-V4-Flash-Vision-Exp vision tower.
//
//   preprocess   aspect clamp / min-pixel upscale / token-budget resize, stretch or gray pad
//                to the patch grid, normalize to [-1, 1], raster patches (588 = 3 * 14 * 14)
//   patch_embed  flat matmul + bias (fp32 residual stream, no learned positions)
//   32 blocks    h += wo(attn(norm1(h)));  h += w2(silu(gate(norm2 h)) * up(norm2 h))
//                RMSNorm eps 1e-6; q/k/v/wo EXL3 with bias; NEOX 2-D rotary over the full
//                head (h frequencies in the first half of the angles, w in the second);
//                full (non-causal) attention; no qk norm
//   norm         RMSNorm -> fp16
//   aligner      zero-pad the grid to multiples of 3, unfold 3x3 (channel-major) into 9216-d
//                vectors, w1 (+bias) -> exact GELU -> w2 (+bias): one 4096-d row per LLM cell
//
// The text model sees an image as a BLOCK of rows (reference image_processor): IMAGE_START,
// the aligner rows interleaved in column pairs (every four consecutive rows cover a 2x2 cell
// of the LLM grid = one CSA compressor group), IMAGE_NEW_LINE after each grid row, pad rows,
// IMAGE_END; the marker rows are learned vectors (image_start / image_pad / image_newline /
// image_end at the checkpoint root). `block(out, nh, nw, start_pos)` assembles it.
//
//   let T = dsv4_vision.load(dir, {device: 0});
//   let r = T.embed(bytes);                   // {out (n, 4096) fp16, nh, nw, ...}
//   let b = T.block(r, start_pos);            // {rows (len, 4096) fp16, types, lead}

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import {readfile} from 'fs';

export const IMAGE_START = 0, IMAGE_PAD = 1, IMAGE = 2, IMAGE_NEW_LINE = 3, IMAGE_END = 4;
export const COMPRESS_PAD_TO = 4;

export function load_config(model_dir) {
	let c = json(readfile(`${model_dir}/config.json`));

	if (!(c.vision_n_layers > 0))
		die(`dsv4_vision: ${model_dir} has no vision tower`);

	let v = {
		n_layers: c.vision_n_layers,
		dim: c.vision_dim || 1024,
		n_heads: c.vision_n_heads || 16,
		inter_dim: c.vision_inter_dim || 2816,
		patch_size: c.vision_patch_size || 14,
		rope_theta: c.vision_rope_theta || 10000.0,
		ds: c.vision_downsample_ratio || 3,
		max_n_token: c.vision_max_n_token || 384,
		min_pixels: c.vision_min_pixels || 147456,
		max_wh_ratio: c.vision_max_wh_ratio || 8,
		eps: 1e-6,
		hidden: c.hidden_size
	};

	v.head_dim = v.dim / v.n_heads;

	return v;
}

/* ---- resize schedule (reference image_processor.py, verbatim) ---- */

function ceil_div(a, b) {
	return int((a + b - 1) / b);
}

export function grid_tokens(bh, bw, p, ds) {
	let nh = ceil_div(int(bh / p), ds);
	let nw = ceil_div(int(bw / p), ds);
	let nt = nh * (nw + 1) + 2;

	if (nh % 2 == 1)
		nt += nw + 1;

	nt += (int((nh + 1) / 2) * (nw + 1)) % 2 * 2;

	return [ nh, nw, nt ];
}

function solve_resize_ratio(height, width, p, ds, max_n) {
	let r = height * 1.0 / width;
	let mwf = math.sqrt((max_n - 2) / r + 0.25) - 0.5;
	let mhf = mwf * r;
	let bw, bh;

	if (mwf < 1.0) {
		let mw = 1;
		let mh = int((max_n - 2) / (mw + 1));

		if (mh % 2 == 1)
			mh--;

		bw = mw * p * ds;
		bh = mh * p * ds;
	}
	else if (mhf < 2.0) {
		let mh = 2;
		let mw = int((max_n - 2) / mh) - 1;

		if (mw <= 1)
			die('dsv4_vision: unsupported aspect ratio');

		bw = mw * p * ds;
		bh = mh * p * ds;
	}
	else {
		let mw = int(mwf), mh = int(mhf);

		if (mh % 2 == 1)
			mh--;

		let beta = min(mw * p * ds * 1.0 / width, mh * p * ds * 1.0 / height);

		bw = int(width * beta / p) * p;
		bh = int(height * beta / p) * p;
	}

	let g = grid_tokens(bh, bw, p, ds);

	return [ g[0], g[1], bh, bw, g[2] ];
}

export function safe_resize(height, width, bh, bw, p, ds, max_n) {
	max_n -= COMPRESS_PAD_TO - 1;

	let g = grid_tokens(bh, bw, p, ds);
	let nh = g[0], nw = g[1], nt = g[2];
	let budget = max_n;

	while (nt > max_n) {
		let s = solve_resize_ratio(height, width, p, ds, budget);

		nh = s[0]; nw = s[1]; bh = s[2]; bw = s[3]; nt = s[4];
		budget--;
	}

	return [ nh, nw, bh, bw ];
}

/* geometry of an image of w x h pixels: {nvh, nvw (ViT grid), nh, nw (LLM grid), bw, bh, stretch} */
export function geometry(v, w, h) {
	let p = v.patch_size;
	let width = w, height = h;

	if (v.max_wh_ratio && width > height * v.max_wh_ratio)
		width = height * v.max_wh_ratio;

	if (width * height > 0 && width * height < v.min_pixels) {
		let ratio = math.sqrt(v.min_pixels * 1.0 / (width * height));

		width = int(width * ratio);
		height = int(height * ratio);
	}

	let bw = ceil_div(width, p) * p, bh = ceil_div(height, p) * p;
	let s = safe_resize(height, width, bh, bw, p, v.ds, v.max_n_token);

	return {
		nh: s[0], nw: s[1], bh: s[2], bw: s[3],
		nvh: int(s[2] / p), nvw: int(s[3] / p),
		stretch: (v.max_wh_ratio && w >= v.max_wh_ratio * h) ? true : false
	};
}

/* N-layout token types of the block and the aligner-row order of its IMAGE slots
 * (reference build_image_block); start_pos = position of the block's first row. */
export function build_image_block(nh, nw, start_pos) {
	let cp = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO;
	let pad_h = nh % 2;
	let rows = nh + pad_h;
	let row_len = nw + 1;
	let pad_last = (int(rows / 2) * row_len) % 2 * 2;
	let types = [], idx = [];

	// grid rows, newline-terminated, then the pad row of an odd count
	for (let r = 0; r < rows; r++)
		for (let c = 0; c < row_len; c++) {
			if (r >= nh) {
				push(types, IMAGE_PAD);
				push(idx, -1);
			}
			else if (c == nw) {
				push(types, IMAGE_NEW_LINE);
				push(idx, -1);
			}
			else {
				push(types, IMAGE);
				push(idx, r * nw + c);
			}
		}

	// pairs of rows interleaved column by column: (rows/2, 2, row_len) -> (rows/2, row_len, 2)
	let ot = [], oi = [];

	for (let a = 0; a < int(rows / 2); a++)
		for (let c = 0; c < row_len; c++)
			for (let k = 0; k < 2; k++) {
				let src = (a * 2 + k) * row_len + c;

				push(ot, types[src]);
				push(oi, idx[src]);
			}

	let bt = [], bi = [];

	for (let i = 0; i < cp; i++) {
		push(bt, IMAGE_PAD);
		push(bi, -1);
	}

	push(bt, IMAGE_START);
	push(bi, -1);

	for (let i = 0; i < length(ot); i++) {
		push(bt, ot[i]);
		push(bi, oi[i]);
	}

	for (let i = 0; i < pad_last; i++) {
		push(bt, IMAGE_PAD);
		push(bi, -1);
	}

	push(bt, IMAGE_END);
	push(bi, -1);

	return {types: bt, src: bi, lead: cp};
}

/* NEOX rotary on (N, H, hd) fp16 with cos/sin (N, 1, hd/2) fp32 */
function rope_neox(x, cos, sin) {
	let half = x.shape()[2] / 2;
	let x1 = x.narrow(2, 0, half).to('float32');
	let x2 = x.narrow(2, half, half).to('float32');

	return torch.cat([ x1.mul(cos).sub(x2.mul(sin)), x2.mul(cos).add(x1.mul(sin)) ], 2).to('float16');
}

/* 2-D angles (N, hd/2): [h angles | w angles] per patch in raster order */
function rope_angles(v, nvh, nvw, dev) {
	let dv = 'cuda:' + dev;
	let dim = v.head_dim / 2;
	let inv = [];

	for (let i = 0; i < dim; i += 2)
		push(inv, 1.0 / math.pow(v.rope_theta, i * 1.0 / dim));

	let mx = (nvh > nvw) ? nvh : nvw;
	let tab = torch.arange(mx, 'float32', dv).view([mx, 1]).mul(torch.from_array(inv, 'float32', dv).view([1, -1]));
	let rows = torch.arange(nvh, 'int64', dv).view([nvh, 1]).expand([nvh, nvw]).contiguous().view([nvh * nvw]);
	let cols = torch.arange(nvw, 'int64', dv).view([1, nvw]).expand([nvh, nvw]).contiguous().view([nvh * nvw]);

	return torch.cat([ tab.index(rows), tab.index(cols) ], 1);
}

function build_block(ws, i, dev) {
	let k = `vision.blocks.${i}.`;

	return {
		n1: ws.gpu16(k + 'norm1.weight', dev),
		n2: ws.gpu16(k + 'norm2.weight', dev),
		q: models.LinearEXL3(ws, k + 'attn.q_proj', dev),
		k: models.LinearEXL3(ws, k + 'attn.k_proj', dev),
		v: models.LinearEXL3(ws, k + 'attn.v_proj', dev),
		o: models.LinearEXL3(ws, k + 'attn.wo', dev),
		gate: models.LinearEXL3(ws, k + 'mlp.gate_proj', dev),
		up: models.LinearEXL3(ws, k + 'mlp.up_proj', dev),
		down: models.LinearEXL3(ws, k + 'mlp.w2', dev)
	};
}

function block_forward(v, B, h, cos, sin) {
	let N = h.shape()[0], nh = v.n_heads, hd = v.head_dim;
	let r = glm5.rms_norm_f32(h, B.n1, v.eps);

	let q = rope_neox(B.q.forward(r).view([N, nh, hd]), cos, sin);
	let k = rope_neox(B.k.forward(r).view([N, nh, hd]), cos, sin);
	let val = B.v.forward(r).view([N, nh, hd]);

	let a = torch.sdpa(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
		val.transpose(0, 1).unsqueeze(0), false);

	a = a.squeeze(0).transpose(0, 1).reshape([N, nh * hd]);
	h = h.add(B.o.forward(a).to('float32'));

	r = glm5.rms_norm_f32(h, B.n2, v.eps);

	return h.add(B.down.forward(torch.silu_mul(B.gate.forward(r), B.up.forward(r))).to('float32'));
}

/* (nvh * nvw, C) raster grid fp16 -> (ceil(nvh/r) * ceil(nvw/r), C * r * r), channel-major
 * within each r x r cell (F.unfold order), zero padding bottom / right */
function unshuffle(x, nvh, nvw, r, dev) {
	let C = x.shape()[1];
	let g = x.view([nvh, nvw, C]);
	let ph = (r - nvh % r) % r, pw = (r - nvw % r) % r;

	if (pw)
		g = torch.cat([ g, torch.zeros([nvh, pw, C], 'float16', 'cuda:' + dev) ], 1);

	if (ph)
		g = torch.cat([ g, torch.zeros([ph, nvw + pw, C], 'float16', 'cuda:' + dev) ], 0);

	let H = nvh + ph, W = nvw + pw;

	// (bh, kh, bw, kw, c) -> (bh, bw, c, kh, kw)
	g = g.contiguous().view([int(H / r), r, int(W / r), r, C]);
	g = g.transpose(1, 2).contiguous();          // (bh, bw, kh, kw, c)
	g = g.transpose(2, 4).contiguous();          // (bh, bw, c, kw, kh)
	g = g.transpose(3, 4).contiguous();          // (bh, bw, c, kh, kw)

	return g.reshape([int(H / r) * int(W / r), C * r * r]);
}

export function load(model_dir, opts) {
	opts = opts || {};

	let v = load_config(model_dir);
	let dev = opts.device || 0;
	let ws = opts.ws || weights.open(model_dir);

	ws.upload(function (key) {
		if (index(key, 'vision.') == 0 || index(key, 'aligner.') == 0 ||
		    key == 'image_start' || key == 'image_pad' || key == 'image_newline' || key == 'image_end') {
			if (index(key, '.attn.wqkv.') >= 0 || index(key, '.mlp.w1.weight') >= 0)
				return null;                // fp16 fused originals; the EXL3 split tensors serve

			return dev;
		}

		return null;
	});

	let blocks = [];

	for (let i = 0; i < v.n_layers; i++)
		push(blocks, build_block(ws, i, dev));

	let pe_cols = 3 * v.patch_size * v.patch_size;
	let markers = torch.stack([
		ws.gpu16('image_start', dev), ws.gpu16('image_pad', dev),
		ws.gpu16('image_newline', dev), ws.gpu16('image_end', dev)
	], 0);

	let T = {
		cfg: v, dev: dev, blocks: blocks, markers: markers,
		patch_w: ws.gpu16('vision.patch_embed.proj.weight', dev).reshape([v.dim, pe_cols]).t().contiguous(),
		patch_b: ws.gpu16('vision.patch_embed.proj.bias', dev).to('float32'),
		norm: ws.gpu16('vision.norm.weight', dev),
		up: models.LinearEXL3(ws, 'aligner.w1', dev),
		down: models.LinearEXL3(ws, 'aligner.w2', dev)
	};

	/* encoded image bytes -> {patches (CPU fp16), nvh, nvw, nh, nw, width, height} */
	T.preprocess = function (bytes) {
		let info = torch.image_info(bytes);
		let g = geometry(this.cfg, info.width, info.height);
		let r = torch.image_prepare(bytes, {
			patch_size: this.cfg.patch_size, merge_size: 1, temporal_patch_size: 1,
			target_w: g.bw, target_h: g.bh, pad: !g.stretch,
			image_mean: [ 0.5, 0.5, 0.5 ], image_std: [ 0.5, 0.5, 0.5 ]
		});

		return {patches: r.patches, nvh: g.nvh, nvw: g.nvw, nh: g.nh, nw: g.nw,
			width: info.width, height: info.height};
	};

	/* patches (N, 588) fp16 -> aligner rows (nh * nw, hidden) fp16 on this.dev;
	 * trace: optional array receiving the residual after block 0 and the last block */
	T.forward = function (patches, nvh, nvw, trace) {
		let c = this.cfg;
		let N = nvh * nvw;

		if (patches.shape()[0] != N)
			die('dsv4_vision: patch count does not match the grid');

		let h = torch.matmul(patches.cuda(this.dev), this.patch_w).to('float32').add(this.patch_b);
		let ang = rope_angles(c, nvh, nvw, this.dev);
		let cos = ang.cos().unsqueeze(1), sin = ang.sin().unsqueeze(1);

		for (let i = 0; i < length(this.blocks); i++) {
			h = block_forward(c, this.blocks[i], h, cos, sin);

			if (trace && i == 0)
				push(trace, h.to('float16'));
		}

		if (trace)
			push(trace, h.to('float16'));

		let x = glm5.rms_norm_f32(h, this.norm, c.eps);          // fp16
		let u = unshuffle(x, nvh, nvw, c.ds, this.dev);

		if (trace)
			push(trace, u);

		return this.down.forward(this.up.forward(u).gelu());
	};

	T.embed = function (bytes) {
		let p = this.preprocess(bytes);
		let out = this.forward(p.patches, p.nvh, p.nvw);

		if (out.shape()[0] != p.nh * p.nw)
			die('dsv4_vision: aligner rows do not match the LLM grid');

		p.out = out;

		return p;
	};

	/* the embedding rows of the whole block for an embed() result at prompt position
	 * start_pos: {rows (len, hidden) fp16 on this.dev, types, lead (leading pad rows)} */
	T.block = function (r, start_pos) {
		let b = build_image_block(r.nh, r.nw, start_pos);
		let table = torch.cat([ this.markers, r.out ], 0);
		let sel = [];

		for (let i = 0; i < length(b.types); i++) {
			let t = b.types[i];

			push(sel, t == IMAGE ? 4 + b.src[i] :
				t == IMAGE_START ? 0 : t == IMAGE_PAD ? 1 : t == IMAGE_NEW_LINE ? 2 : 3);
		}

		b.rows = table.index(torch.from_array(sel, 'int64', 'cuda:' + this.dev));

		return b;
	};

	return T;
}
