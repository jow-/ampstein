// ucode/qwen35_vision.uc — Qwen3.5/3.8 vision tower (ViT + patch merger).
//
// Turns an encoded image into LLM-width embeddings:
//
//   bytes --image_prepare--> patches (N, 3*tp*ps*ps) fp16, grid (1, gh, gw)
//   patches --patch_embed (a Conv3d with stride == kernel == a matmul)
//           + bilinearly interpolated learned position embedding
//           --27 x [LayerNorm, MHA with 2-D RoPE (non-causal),
//                   LayerNorm, GELU-tanh MLP]
//           --merger (LayerNorm, 2x2 spatial merge, MLP)--> (N/4, 5120)
//
// The residual stream is fp32 (as in exllamav3's reference graph); the
// EXL3 linears run fp16. Patches arrive in merge-block order, so the
// merger is a plain reshape of consecutive rows.
//
// Weights: model.visual.* from the checkpoint (EXL3 q/k/v/proj/fc1/fc2
// at "vision_bits", fp16 norms/embeddings; the stray fused fp16
// attn.qkv tensors are ignored when the split EXL3 projections exist).
//
//   import * as vision from 'ampstein.qwen35_vision';
//   let v = vision.load(model_dir, {device: 0});
//   let r = v.embed(readfile('cat.png'));     // {emb, grid, n_tokens, ...}

import * as math from 'math';
import * as weights from './weights.uc';
import * as models from './models.uc';
import * as ggufw from './ggufw.uc';
import {readfile} from 'fs';

const PREFIX = 'model.visual.';

/* Vision geometry + preprocessing settings from config.json and
 * preprocessor_config.json. */
export function load_config(model_dir) {
	let cj = json(readfile(`${model_dir}/config.json`));
	let pj = json(readfile(`${model_dir}/preprocessor_config.json`));
	let v = cj.vision_config;

	if (!v)
		die(`qwen35_vision: ${model_dir}/config.json has no vision_config`);

	if (v.hidden_act != 'gelu_pytorch_tanh')
		die(`qwen35_vision: unsupported hidden_act ${v.hidden_act}`);

	let cfg = {
		depth: v.depth,
		hidden_size: v.hidden_size,
		intermediate_size: v.intermediate_size,
		num_heads: v.num_heads,
		num_position_embeddings: v.num_position_embeddings,
		out_hidden_size: v.out_hidden_size,
		patch_size: v.patch_size,
		merge_size: v.spatial_merge_size,
		temporal_patch_size: v.temporal_patch_size,
		deepstack: v.deepstack_visual_indexes || [],
		rope_theta: 10000.0,
		eps: 1e-6,
		image_token_id: cj.image_token_id,
		vision_start_id: cj.vision_start_token_id,
		vision_end_id: cj.vision_end_token_id,
		min_pixels: pj.size.shortest_edge,
		max_pixels: pj.size.longest_edge,
		image_mean: pj.image_mean,
		image_std: pj.image_std
	};

	cfg.head_dim = cfg.hidden_size / cfg.num_heads;

	if (length(cfg.deepstack) > 0)
		die('qwen35_vision: deepstack visual embeddings are not supported');

	if (cfg.head_dim * cfg.num_heads != cfg.hidden_size || cfg.head_dim % 2)
		die('qwen35_vision: hidden_size must split into even-width heads');

	/* the learned position table is a square grid */
	cfg.pos_grid = int(math.sqrt(cfg.num_position_embeddings) + 0.5);

	if (cfg.pos_grid * cfg.pos_grid != cfg.num_position_embeddings)
		die('qwen35_vision: num_position_embeddings is not a square');

	return cfg;
}

/* Number of LLM embedding rows an image with patch grid (t, h, w)
 * occupies: patches / merge^2. */
export function n_tokens(cfg, grid) {
	let m = cfg.merge_size;

	return grid[0] * (grid[1] / m) * (grid[2] / m);
}

/* Inverse frequencies of the 2-D rotary embedding: head_dim/4 values,
 * theta^(-2i / (head_dim/2)). */
function rope_inv_freq(cfg) {
	let dim = cfg.head_dim / 2;
	let f = [];

	for (let i = 0; i < dim; i += 2)
		push(f, 1.0 / math.pow(cfg.rope_theta, (i * 1.0) / dim));

	return f;
}

/* (gh, gw) index grid -> merge-block order, flattened: the row order
 * the patches are delivered in. */
function merge_order(t, gh, gw, m) {
	return t.reshape([gh / m, m, gw / m, m]).transpose(1, 2)
		.reshape([gh * gw]);
}

/* Bilinear interpolation of the learned (pos_grid x pos_grid) position
 * table onto the (gh, gw) patch grid, in merge-block order:
 * returns (gh*gw, hidden) fp32. */
function pos_embed(cfg, table, gh, gw, dev) {
	let n = cfg.pos_grid;
	let m = cfg.merge_size;

	function axis(g) {
		let lo = [], hi = [], d = [];

		for (let i = 0; i < g; i++) {
			let p = (g > 1) ? ((n - 1) * 1.0 * i) / (g - 1) : 0.0;
			let f = int(p);

			push(lo, f);
			push(hi, (f + 1 < n) ? f + 1 : n - 1);
			push(d, p - f);
		}

		/* *_n: the same indices pre-scaled by the table row stride (an
		 * int64 tensor times a scalar would promote to float) */
		return {lo: lo, hi: hi, d: d,
			lo_n: map(lo, (x) => x * n), hi_n: map(hi, (x) => x * n)};
	}

	let ah = axis(gh), aw = axis(gw);
	let dv = 'cuda:' + dev;

	function col(a, k) {     // (g, 1)
		return torch.from_array(a, k == 'd' ? 'float32' : 'int64', dv)
			.view([length(a), 1]);
	}

	function row(a, k) {     // (1, g)
		return torch.from_array(a, k == 'd' ? 'float32' : 'int64', dv)
			.view([1, length(a)]);
	}

	let hl = col(ah.lo_n), hh = col(ah.hi_n);
	let wl = row(aw.lo), wh = row(aw.hi);
	let dh = col(ah.d, 'd'), dw = row(aw.d, 'd');
	let dh1 = dh.mul(-1.0).add(1.0), dw1 = dw.mul(-1.0).add(1.0);

	let corners = [
		[ hl.add(wl), dh1.mul(dw1) ],
		[ hl.add(wh), dh1.mul(dw) ],
		[ hh.add(wl), dh.mul(dw1) ],
		[ hh.add(wh), dh.mul(dw) ]
	];

	let acc = null;

	for (let c in corners) {
		let idx = c[0].reshape([gh * gw]);
		let w = c[1].reshape([gh * gw, 1]);
		let e = table.index(idx).to('float32').mul(w);

		acc = acc ? acc.add(e) : e;
	}

	let hid = cfg.hidden_size;

	return acc.reshape([gh / m, m, gw / m, m, hid]).transpose(1, 2)
		.reshape([gh * gw, hid]);
}

/* 2-D rotary angles (N, head_dim/2) fp32 in merge-block order: the
 * first half of the channels follows the row index, the second the
 * column index. */
export function rope_angles(cfg, gh, gw, dev) {
	let dv = 'cuda:' + dev;
	let m = cfg.merge_size;
	let mx = (gh > gw) ? gh : gw;
	let inv = torch.from_array(rope_inv_freq(cfg), 'float32', dv)
		.view([1, -1]);
	let seq = torch.arange(mx, 'float32', dv).view([mx, 1]);
	let tab = seq.mul(inv);                          // (mx, hd/4)

	let rows = torch.arange(gh, 'int64', dv).view([gh, 1]).expand([gh, gw]);
	let cols = torch.arange(gw, 'int64', dv).view([1, gw]).expand([gh, gw]);
	let hpos = merge_order(rows.contiguous(), gh, gw, m);
	let wpos = merge_order(cols.contiguous(), gh, gw, m);

	return torch.cat([ tab.index(hpos), tab.index(wpos) ], 1);
}

/* NEOX rotary on (N, H, hd) fp16 with cos/sin (N, 1, hd/2) fp32. */
export function rope_neox(x, cos, sin) {
	let half = x.shape()[2] / 2;
	let x1 = x.narrow(2, 0, half).to('float32');
	let x2 = x.narrow(2, half, half).to('float32');
	let o1 = x1.mul(cos).sub(x2.mul(sin));
	let o2 = x2.mul(cos).add(x1.mul(sin));

	return torch.cat([ o1, o2 ], 2).to('float16');
}

function build_block(ws, i, dev) {
	let k = `${PREFIX}blocks.${i}.`;

	return {
		n1w: ws.gpu16(k + 'norm1.weight', dev).to('float32'),
		n1b: ws.gpu16(k + 'norm1.bias', dev).to('float32'),
		n2w: ws.gpu16(k + 'norm2.weight', dev).to('float32'),
		n2b: ws.gpu16(k + 'norm2.bias', dev).to('float32'),
		q: models.linear(ws, k + 'attn.q_proj', false, dev),
		k: models.linear(ws, k + 'attn.k_proj', false, dev),
		v: models.linear(ws, k + 'attn.v_proj', false, dev),
		o: models.linear(ws, k + 'attn.proj', false, dev),
		fc1: models.linear(ws, k + 'mlp.linear_fc1', false, dev),
		fc2: models.linear(ws, k + 'mlp.linear_fc2', false, dev)
	};
}

/* One transformer block on the fp32 residual stream h (N, hidden). */
function block_forward(cfg, B, h, cos, sin) {
	let N = h.shape()[0];
	let nh = cfg.num_heads, hd = cfg.head_dim;

	let r = torch.layer_norm(h, B.n1w, B.n1b, cfg.eps).to('float16');

	let q = rope_neox(B.q.forward(r).view([N, nh, hd]), cos, sin);
	let k = rope_neox(B.k.forward(r).view([N, nh, hd]), cos, sin);
	let v = B.v.forward(r).view([N, nh, hd]);

	// (N, H, hd) -> (1, H, N, hd); full (non-causal) attention
	let a = torch.sdpa(q.transpose(0, 1).unsqueeze(0),
		k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
		false);
	a = a.squeeze(0).transpose(0, 1).reshape([N, nh * hd]);

	h = h.add(B.o.forward(a).to('float32'));

	r = torch.layer_norm(h, B.n2w, B.n2b, cfg.eps).to('float16');

	let m = B.fc2.forward(B.fc1.forward(r).gelu_tanh());

	return h.add(m.to('float32'));
}

/* Load the tower onto device `opts.device` (default 0).
 * opts.max_pixels caps the preprocessed image area (default: the
 * checkpoint's preprocessor limit). */
export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let dev = opts.device || 0;
	// a GGUF checkpoint keeps the tower in its mmproj-*.gguf (ggufw.uc maps it to the HF names); EXL3 / safetensors packs have it in the shards
	let mm = ggufw.mmproj_file(model_dir);
	let ws = mm ? weights.open(model_dir, mm) : weights.open(model_dir);

	if (opts.max_pixels != null)
		cfg.max_pixels = opts.max_pixels;

	/* only the tower, and only the split EXL3 attention projections */
	ws.upload(function (key) {
		if (index(key, PREFIX) != 0 || index(key, '.attn.qkv.') >= 0)
			return null;

		return dev;
	});

	let blocks = [];

	for (let i = 0; i < cfg.depth; i++)
		push(blocks, build_block(ws, i, dev));

	let pe_w = ws.gpu16(PREFIX + 'patch_embed.proj.weight', dev);
	let pe_cols = cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size * 3;

	let T = {
		cfg: cfg,
		dev: dev,
		blocks: blocks,
		patch_w: pe_w.reshape([cfg.hidden_size, pe_cols]).t().contiguous(),
		patch_b: ws.gpu16(PREFIX + 'patch_embed.proj.bias', dev)
			.to('float32'),
		pos_table: ws.gpu16(PREFIX + 'pos_embed.weight', dev),
		mnw: ws.gpu16(PREFIX + 'merger.norm.weight', dev).to('float32'),
		mnb: ws.gpu16(PREFIX + 'merger.norm.bias', dev).to('float32'),
		fc1: models.linear(ws, PREFIX + 'merger.linear_fc1', false, dev),
		fc2: models.linear(ws, PREFIX + 'merger.linear_fc2', false, dev),

		/* Preprocess encoded image bytes: {patches (CPU fp16), grid,
		 * width, height, new_width, new_height}. */
		preprocess: function (bytes) {
			let c = this.cfg;

			return torch.image_prepare(bytes, {
				patch_size: c.patch_size,
				merge_size: c.merge_size,
				temporal_patch_size: c.temporal_patch_size,
				min_pixels: c.min_pixels,
				max_pixels: c.max_pixels,
				image_mean: c.image_mean,
				image_std: c.image_std
			});
		},

		/* Run the tower on preprocessed patches. patches: (N, cols)
		 * fp16 CPU/CUDA; grid [1, gh, gw]. trace: optional array that
		 * receives the residual stream after block 0 (fp16 copy).
		 * Returns (N / merge^2, out_hidden) fp16 on this.dev. */
		forward: function (patches, grid, trace) {
			let c = this.cfg;
			let gh = grid[1], gw = grid[2];
			let N = gh * gw;

			if (grid[0] != 1)
				die('qwen35_vision: only single images (t == 1) are supported');

			if (patches.shape()[0] != N)
				die('qwen35_vision: patch count does not match the grid');

			let x = patches.cuda(this.dev);
			let h = torch.matmul(x, this.patch_w).to('float32').add(this.patch_b);

			h = h.add(pos_embed(c, this.pos_table, gh, gw, this.dev));

			let ang = rope_angles(c, gh, gw, this.dev);
			let cos = ang.cos().unsqueeze(1), sin = ang.sin().unsqueeze(1);

			for (let i = 0; i < length(this.blocks); i++) {
				h = block_forward(c, this.blocks[i], h, cos, sin);

				if (trace && i == 0)
					push(trace, h.to('float16'));
			}

			let r = torch.layer_norm(h, this.mnw, this.mnb, c.eps)
				.to('float16')
				.reshape([N / (c.merge_size * c.merge_size),
					c.hidden_size * c.merge_size * c.merge_size]);

			return this.fc2.forward(this.fc1.forward(r).gelu_tanh());
		},

		/* bytes -> {emb, grid, n_tokens, width, height, new_width,
		 * new_height} */
		embed: function (bytes) {
			let p = this.preprocess(bytes);
			let emb = this.forward(p.patches, p.grid);

			return {
				emb: emb,
				grid: p.grid,
				n_tokens: emb.shape()[0],
				width: p.width,
				height: p.height,
				new_width: p.new_width,
				new_height: p.new_height
			};
		}
	};

	return T;
}
