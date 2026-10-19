// ucode/gemma4_vision.uc — Gemma 4 vision tower (Gemma4VisionModel + embed_vision) on one GPU.
//
//   preprocess  aspect-preserving resize to a multiple of 48 px within max_soft_tokens * 9 patches (bicubic, PIL-style),
//               pixels in [0, 1], 16 x 16 patches flattened (y, x, channel), raster order
//   patch embed input_proj(2 * (pixels - 0.5)) + x / y position tables (2, 10240, 1152) (summed)
//   27 blocks   h += post_attn_norm(attn(input_norm(h)));  h += post_ff_norm(mlp(pre_ff_norm(h)))
//                 attention: 16 heads of 72, q / k RMSNorm (weight), v RMSNorm (no weight), axial 2-D rope (the first 36
//                 dims rotate with the x position, the last 36 with y), scale 1, bidirectional over all patches;
//                 MLP: gelu_tanh(gate) * up, down. Residual stream fp32.
//   pooling     3 x 3 average over the patch grid, * sqrt(hidden), standardize ((x - std_bias) * std_scale)
//   embed_vision  unweighted RMSNorm, projection to the text width (2816)
// The output rows (n, 2816) replace the image placeholder tokens of the text model (not scaled by the embedding scale).
//
//   let T = gemma4_vision.load(dir, {device: 0});
//   let r = T.embed(png_bytes);        // {emb (n, 2816) fp16, n_tokens, width, height, new_width, new_height}

import * as weights from './weights.uc';
import * as glm5 from './glm5.uc';
import * as math from 'math';
import {readfile} from 'fs';

const PREFIX = 'model.vision_tower.';

export function load_config(model_dir) {
	let cj = json(readfile(`${model_dir}/config.json`));
	let pj = json(readfile(`${model_dir}/processor_config.json`));
	let v = cj.vision_config;

	if (!v || v.model_type != 'gemma4_vision')
		die(`gemma4_vision: ${model_dir} has no gemma4_vision config`);

	let ip = pj.image_processor;
	let cfg = {
		depth: v.num_hidden_layers,
		hidden_size: v.hidden_size,
		intermediate_size: v.intermediate_size,
		num_heads: v.num_attention_heads,
		head_dim: v.head_dim,
		patch_size: v.patch_size,
		pool: v.pooling_kernel_size,
		pos_size: v.position_embedding_size,
		eps: v.rms_norm_eps,
		rope_theta: v.rope_parameters.rope_theta,
		standardize: v.standardize,
		clipped: v.use_clipped_linears,
		text_eps: cj.text_config.rms_norm_eps,
		max_soft: ip.max_soft_tokens,
		boi: cj.boi_token_id,
		eoi: cj.eoi_token_id,
		image_token_id: cj.image_token_id
	};

	if (cfg.clipped)
		die('gemma4_vision: clipped linears are not supported');

	return cfg;
}

/* the size (w, h) of an image of w0 x h0 for at most `max_patches` patches: both sides a multiple of pool * patch */
export function target_size(cfg, w0, h0, max_patches) {
	let side = cfg.pool * cfg.patch_size;
	let factor = math.sqrt(max_patches * cfg.patch_size * cfg.patch_size / (w0 * h0 * 1.0));
	let th = int(factor * h0 / side) * side;
	let tw = int(factor * w0 / side) * side;
	let max_side = int(max_patches / (cfg.pool * cfg.pool)) * side;

	if (th == 0 && tw == 0)
		die('gemma4_vision: the image is too small to patch');

	if (th == 0) {
		th = side;
		tw = min(int(w0 / h0) * side, max_side);
	}
	else if (tw == 0) {
		tw = side;
		th = min(int(h0 / w0) * side, max_side);
	}

	return [ tw, th ];
}

function build_block(ws, i, dev) {
	let k = `${PREFIX}encoder.layers.${i}.`;
	let w = (key) => ws.gpu16(k + key + '.linear.weight', dev).t().contiguous();
	let n = (key) => ws.gpu(k + key + '.weight', dev).to('float32').contiguous();

	return {
		q: w('self_attn.q_proj'), k: w('self_attn.k_proj'), v: w('self_attn.v_proj'), o: w('self_attn.o_proj'),
		gate: w('mlp.gate_proj'), up: w('mlp.up_proj'), down: w('mlp.down_proj'),
		qn: n('self_attn.q_norm'), kn: n('self_attn.k_norm'),
		in_n: n('input_layernorm'), post_attn: n('post_attention_layernorm'),
		pre_ff: n('pre_feedforward_layernorm'), post_ff: n('post_feedforward_layernorm')
	};
}

/* axial 2-D rope on x (N, nh, hd) fp32: the first hd / 2 dims rotate (rotate-half) with the angles of the x position, the
 * rest with those of y; cx / sx, cy / sy (N, 1, hd / 4) */
function rope2d(x, cx, sx, cy, sy, hd) {
	let q = hd / 4;
	let parts = [];

	for (let p = 0; p < 2; p++) {
		let c = p ? cy : cx, s = p ? sy : sx;
		let a = x.narrow(2, p * 2 * q, q), b = x.narrow(2, p * 2 * q + q, q);

		push(parts, a.mul(c).sub(b.mul(s)), b.mul(c).add(a.mul(s)));
	}

	return torch.cat(parts, 2);
}

function block_forward(cfg, B, h, rp, ones_v) {
	let N = h.shape()[0], nh = cfg.num_heads, hd = cfg.head_dim, eps = cfg.eps;
	let r = torch.norm32(h, B.in_n, eps).to('float16');
	let q = torch.norm32(torch.matmul(r, B.q).view([N * nh, hd]), B.qn, eps).view([N, nh, hd]);
	let k = torch.norm32(torch.matmul(r, B.k).view([N * nh, hd]), B.kn, eps).view([N, nh, hd]);
	let v = torch.norm32(torch.matmul(r, B.v).view([N * nh, hd]), ones_v, eps).view([N, nh, hd]);

	q = rope2d(q, rp.cx, rp.sx, rp.cy, rp.sy, hd).to('float16');
	k = rope2d(k, rp.cx, rp.sx, rp.cy, rp.sy, hd).to('float16');

	let a = torch.sdpa(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
		v.to('float16').transpose(0, 1).unsqueeze(0), false, null, false, 1.0);

	a = a.squeeze(0).transpose(0, 1).reshape([N, nh * hd]).contiguous();
	h = torch.norm32(torch.matmul(a, B.o), B.post_attn, eps, h);

	r = torch.norm32(h, B.pre_ff, eps).to('float16');

	let m = torch.matmul(glm5.act_mul({act: 'gelu'}, torch.matmul(r, B.gate), torch.matmul(r, B.up)), B.down);

	return torch.norm32(m, B.post_ff, eps, h);
}

export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let dev = opts.device || 0;
	let ws = weights.open(model_dir);

	if (opts.max_soft != null)
		cfg.max_soft = opts.max_soft;

	ws.upload(function (key) {
		return (index(key, PREFIX) == 0 || index(key, 'model.embed_vision.') == 0) ? dev : null;
	});

	let blocks = [];

	for (let i = 0; i < cfg.depth; i++)
		push(blocks, build_block(ws, i, dev));

	let hd = cfg.head_dim, C = cfg.hidden_size;
	let inv = [];

	for (let i = 0; i < hd / 4; i++)
		push(inv, 1.0 / math.pow(cfg.rope_theta, 2.0 * i / (hd / 2)));

	let T = {
		cfg: cfg, dev: dev, blocks: blocks,
		in_w: ws.gpu16(PREFIX + 'patch_embedder.input_proj.weight', dev).t().contiguous(),
		pos_x: ws.gpu16(PREFIX + 'patch_embedder.position_embedding_table', dev).narrow(0, 0, 1).squeeze(0),
		pos_y: ws.gpu16(PREFIX + 'patch_embedder.position_embedding_table', dev).narrow(0, 1, 1).squeeze(0),
		std_bias: ws.gpu(PREFIX + 'std_bias', dev).to('float32').reshape([1, -1]),
		std_scale: ws.gpu(PREFIX + 'std_scale', dev).to('float32').reshape([1, -1]),
		proj: ws.gpu16('model.embed_vision.embedding_projection.weight', dev).t().contiguous(),
		ones_v: torch.ones([hd], 'float32', 'cuda:' + dev),
		ones_c: torch.ones([C], 'float32', 'cuda:' + dev),
		inv: torch.from_array(inv, 'float32', 'cuda:' + dev).view([1, hd / 4]),

		/* bytes -> {patches (CPU fp16, (N, 768)), gh, gw, width, height, new_width, new_height} */
		preprocess: function (bytes) {
			let c = this.cfg;
			let info = torch.image_info(bytes);
			let sz = target_size(c, info.width, info.height, c.max_soft * c.pool * c.pool);
			let p = torch.image_prepare(bytes, {
				patch_size: c.patch_size, merge_size: 1, temporal_patch_size: 1,
				target_w: sz[0], target_h: sz[1], pad: false, hwc: true,
				image_mean: [ 0, 0, 0 ], image_std: [ 1, 1, 1 ]
			});

			return {
				patches: p.patches, gh: p.grid[1], gw: p.grid[2],
				width: p.width, height: p.height, new_width: p.new_width, new_height: p.new_height
			};
		},

		/* the tower on patches (N, 768) fp16 of a gh x gw grid -> (N / 9, 2816) fp16 on this.dev */
		forward: function (patches, gh, gw) {
			let c = this.cfg;
			let dv = 'cuda:' + this.dev;
			let N = gh * gw;
			let k = c.pool;

			if (patches.shape()[0] != N || gh % k || gw % k)
				die('gemma4_vision: patch grid does not match the patches / pooling window');

			let xs = [], ys = [];

			for (let y = 0; y < gh; y++)
				for (let x = 0; x < gw; x++) {
					push(xs, x);
					push(ys, y);
				}

			let xi = torch.from_array(xs, 'int64', dv), yi = torch.from_array(ys, 'int64', dv);
			let px = patches.cuda(this.dev).sub(0.5).mul(2.0);
			let h = torch.matmul(px, this.in_w).to('float32')
				.add(this.pos_x.index(xi).to('float32')).add(this.pos_y.index(yi).to('float32'));

			let fx = xi.to('float32').view([N, 1]).mul(this.inv), fy = yi.to('float32').view([N, 1]).mul(this.inv);
			let rp = {
				cx: fx.cos().unsqueeze(1).contiguous(), sx: fx.sin().unsqueeze(1).contiguous(),
				cy: fy.cos().unsqueeze(1).contiguous(), sy: fy.sin().unsqueeze(1).contiguous()
			};

			for (let B in this.blocks)
				h = block_forward(c, B, h, rp, this.ones_v);

			// k x k average pooling of the raster grid, * sqrt(hidden), standardize
			let n = N / (k * k);
			let pooled = h.view([gh / k, k, gw / k, k, c.hidden_size]).mean(3).mean(1).reshape([n, c.hidden_size]);

			pooled = pooled.mul(math.sqrt(c.hidden_size * 1.0));

			if (c.standardize)
				pooled = pooled.sub(this.std_bias).mul(this.std_scale);

			return torch.matmul(torch.norm32(pooled.contiguous(), this.ones_c, c.eps).to('float16'), this.proj);
		},

		embed: function (bytes) {
			let p = this.preprocess(bytes);
			let emb = this.forward(p.patches, p.gh, p.gw);

			return {
				emb: emb, grid: [ 1, p.gh / this.cfg.pool, p.gw / this.cfg.pool ], n_tokens: emb.shape()[0],
				width: p.width, height: p.height, new_width: p.new_width, new_height: p.new_height
			};
		}
	};

	return T;
}
