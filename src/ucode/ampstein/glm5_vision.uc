// ucode/glm5_vision.uc — GLM-5.3-Flash vision tower (glm5_next_vision: the
// GLM4V structure with GLM5.3 deltas) on one GPU.
//
//   patch_embed   Conv3d(3, 1024, (2, 14, 14)) as a flat matmul + bias
//                 (fp32 residual stream; no learned position embedding —
//                 positions are 2-D rotary only)
//   24 blocks     h += proj(attn(norm1(h)));  h += down(act(norm2(h)))
//                   norm: RMSNorm; attention: q/k/v (EXL3, biased), per-head
//                   RMSNorm on q and k, NEOX 2-D rotary, full (non-causal)
//                   attention; MLP: gate/up with clamp (swiglu_limit),
//                   silu(gate) * up, down (all biased)
//   post_layernorm (RMSNorm) -> downsample (Conv2d 2x2 stride 2 over the
//                 merge blocks, 1024 -> 4096) -> merger: proj, LayerNorm
//                 (eps 1e-6), exact GELU, gated MLP (10240), down
//
// The output rows (grid_h * grid_w / 4, 4096) replace the <|image|> rows in
// the text model; the text side is NoPE, so no position bookkeeping.
//
// Usage:
//   let T = glm5_vision.load(dir, {device: 0});
//   let r = T.embed(png_bytes);     // {emb (n, 4096) fp16, grid, n_tokens, ...}

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import {rope_angles, rope_neox} from './qwen35_vision.uc';
import {readfile} from 'fs';

const PREFIX = 'model.visual.';

export function load_config(model_dir) {
	let cj = json(readfile(`${model_dir}/config.json`));
	let pj = json(readfile(`${model_dir}/processor_config.json`));
	let v = cj.vision_config;

	if (!v || v.model_type != 'glm5_next_vision')
		die(`glm5_vision: ${model_dir} has no glm5_next_vision config`);

	let ip = pj.image_processor;
	let cfg = {
		depth: v.depth,
		hidden_size: v.hidden_size,
		intermediate_size: v.intermediate_size,
		num_heads: v.num_heads,
		out_hidden_size: v.out_hidden_size,
		merger_size: v.projection_intermediate_size,
		patch_size: v.patch_size,
		merge_size: v.spatial_merge_size,
		temporal_patch_size: v.temporal_patch_size,
		act_limit: v.swiglu_limit,
		eps: v.rms_norm_eps,
		rope_theta: 10000.0,
		image_token_id: cj.image_token_id,
		start_id: cj.image_start_token_id,
		end_id: cj.image_end_token_id,
		min_tokens: ip.min_image_tokens,
		max_tokens: ip.max_image_tokens,
		image_mean: ip.image_mean,
		image_std: ip.image_std
	};

	cfg.head_dim = cfg.hidden_size / cfg.num_heads;

	if (cfg.head_dim * cfg.num_heads != cfg.hidden_size || cfg.head_dim % 4)
		die('glm5_vision: hidden_size must split into heads of a multiple of 4');

	return cfg;
}

/* Number of LLM embedding rows of an image with patch grid [t, h, w]. */
export function n_tokens(cfg, grid) {
	let m = cfg.merge_size;

	return grid[0] * (grid[1] / m) * (grid[2] / m);
}

function clamp_act(cfg, g, u) {
	let lim = cfg.act_limit;

	if (lim && lim != 0) {
		g = g.clamp(null, lim);
		u = u.clamp(-lim, lim);
	}

	return torch.silu_mul(g, u);
}

function build_block(ws, i, dev) {
	let k = `${PREFIX}blocks.${i}.`;

	return {
		n1: ws.gpu16(k + 'norm1.weight', dev),
		n2: ws.gpu16(k + 'norm2.weight', dev),
		qn: ws.gpu16(k + 'attn.q_norm.weight', dev),
		kn: ws.gpu16(k + 'attn.k_norm.weight', dev),
		q: models.LinearEXL3(ws, k + 'attn.q_proj', dev),
		k: models.LinearEXL3(ws, k + 'attn.k_proj', dev),
		v: models.LinearEXL3(ws, k + 'attn.v_proj', dev),
		o: models.LinearEXL3(ws, k + 'attn.proj', dev),
		gate: models.LinearEXL3(ws, k + 'mlp.gate_proj', dev),
		up: models.LinearEXL3(ws, k + 'mlp.up_proj', dev),
		down: models.LinearEXL3(ws, k + 'mlp.down_proj', dev)
	};
}

/* One block on the fp32 residual stream h (N, hidden). */
function block_forward(cfg, B, h, cos, sin) {
	let N = h.shape()[0];
	let nh = cfg.num_heads, hd = cfg.head_dim;
	let eps = cfg.eps;

	let r = glm5.rms_norm_f32(h, B.n1, eps);

	let q = torch.rms_norm(B.q.forward(r).view([N, nh, hd]), B.qn, eps);
	let k = torch.rms_norm(B.k.forward(r).view([N, nh, hd]), B.kn, eps);
	let v = B.v.forward(r).view([N, nh, hd]);

	q = rope_neox(q, cos, sin);
	k = rope_neox(k, cos, sin);

	// (N, H, hd) -> (1, H, N, hd); full (non-causal) attention
	let a = torch.sdpa(q.transpose(0, 1).unsqueeze(0),
		k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), false);

	a = a.squeeze(0).transpose(0, 1).reshape([N, nh * hd]);

	h = h.add(B.o.forward(a).to('float32'));

	r = glm5.rms_norm_f32(h, B.n2, eps);

	let m = B.down.forward(clamp_act(cfg, B.gate.forward(r), B.up.forward(r)));

	return h.add(m.to('float32'));
}

/* Load the tower onto device `opts.device` (default 0). opts.max_tokens
 * caps the image token budget (default: the checkpoint's 8000). */
export function load(model_dir, opts) {
	opts = opts || {};

	let cfg = load_config(model_dir);
	let dev = opts.device || 0;
	let ws = weights.open(model_dir);

	if (opts.max_tokens != null)
		cfg.max_tokens = opts.max_tokens;

	/* the tower, and only the quantized split attention projections */
	ws.upload(function (key) {
		if (index(key, PREFIX) != 0 || index(key, '.attn.qkv.') >= 0)
			return null;

		return dev;
	});

	let blocks = [];

	for (let i = 0; i < cfg.depth; i++)
		push(blocks, build_block(ws, i, dev));

	let pe_cols = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
	let C = cfg.hidden_size;

	let T = {
		cfg: cfg,
		dev: dev,
		blocks: blocks,
		patch_w: ws.gpu16(PREFIX + 'patch_embed.proj.weight', dev)
			.reshape([C, pe_cols]).t().contiguous(),
		patch_b: ws.gpu16(PREFIX + 'patch_embed.proj.bias', dev).to('float32'),
		post_norm: ws.gpu16(PREFIX + 'post_layernorm.weight', dev),
		// Conv2d (4096, 1024, 2, 2) as a matmul over (c, i, j) rows
		down_w: ws.gpu16(PREFIX + 'downsample.weight', dev)
			.reshape([cfg.out_hidden_size, C * cfg.merge_size * cfg.merge_size]).t().contiguous(),
		down_b: ws.gpu16(PREFIX + 'downsample.bias', dev),
		m_proj: models.LinearEXL3(ws, PREFIX + 'merger.proj', dev),
		m_gate: models.LinearEXL3(ws, PREFIX + 'merger.gate_proj', dev),
		m_up: models.LinearEXL3(ws, PREFIX + 'merger.up_proj', dev),
		m_down: models.LinearEXL3(ws, PREFIX + 'merger.down_proj', dev),
		m_nw: ws.gpu16(PREFIX + 'merger.post_projection_norm.weight', dev).to('float32'),
		m_nb: ws.gpu16(PREFIX + 'merger.post_projection_norm.bias', dev).to('float32'),

		/* Preprocess encoded image bytes: {patches (CPU fp16), grid,
		 * width, height, new_width, new_height}. */
		preprocess: function (bytes) {
			let c = this.cfg;

			return torch.image_prepare(bytes, {
				patch_size: c.patch_size,
				merge_size: c.merge_size,
				temporal_patch_size: c.temporal_patch_size,
				min_tokens: c.min_tokens,
				max_tokens: c.max_tokens,
				image_mean: c.image_mean,
				image_std: c.image_std
			});
		},

		/* Run the tower on preprocessed patches. patches: (N, cols) fp16;
		 * grid [1, gh, gw]. trace: optional array receiving the residual
		 * after block 0 (fp16). Returns (N / merge^2, out_hidden) fp16 on
		 * this.dev. */
		forward: function (patches, grid, trace, all) {
			let c = this.cfg;
			let gh = grid[1], gw = grid[2];
			let N = gh * gw;
			let m2 = c.merge_size * c.merge_size;

			if (grid[0] != 1)
				die('glm5_vision: only single images (t == 1) are supported');

			if (patches.shape()[0] != N)
				die('glm5_vision: patch count does not match the grid');

			let x = patches.cuda(this.dev);
			let h = torch.matmul(x, this.patch_w).to('float32').add(this.patch_b);

			let ang = rope_angles(c, gh, gw, this.dev);
			let cos = ang.cos().unsqueeze(1), sin = ang.sin().unsqueeze(1);

			for (let i = 0; i < length(this.blocks); i++) {
				h = block_forward(c, this.blocks[i], h, cos, sin);

				if (trace && (i == 0 || all))
					push(trace, h.to('float16'));
			}

			if (trace)
				push(trace, h.to('float16'));        // [1]: after the last block

			// post norm, then the 2x2 merge conv: rows 4n..4n+3 form one block
			let r = glm5.rms_norm_f32(h, this.post_norm, c.eps);
			let n = N / m2;
			let d = torch.matmul(r.view([n, m2, c.hidden_size]).transpose(1, 2)
				.reshape([n, c.hidden_size * m2]).contiguous(), this.down_w)
				.add(this.down_b);

			if (trace)
				push(trace, d);                      // [2]: after the downsample

			// merger
			let y = this.m_proj.forward(d);

			y = torch.layer_norm(y.to('float32'), this.m_nw, this.m_nb, 1e-6)
				.to('float16').gelu();

			return this.m_down.forward(clamp_act(c, this.m_gate.forward(y), this.m_up.forward(y)));
		},

		/* bytes -> {emb, grid, n_tokens, width, height, new_width, new_height} */
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
