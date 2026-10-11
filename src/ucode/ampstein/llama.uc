// ucode/llama.uc — llama-family model definition on top of torch + models.
//
// Config parsing (config.json, with head_dim fallback and rope-type
// checks), the RoPE cache (incl. llama3 scaling), RMSNorm, the attention
// block (GQA via SDPA) and the SwiGLU MLP — the full per-layer forward
// graph, in pure ucode. Both paths live here: the eager prefill/
// chunked-prefill forward (forward / forward_chunk) over a KV cache
// (kvcache.uc), and greedy/temperature generation (generate).
//
// Usage:
//   import * as llama from 'ampstein.llama';
//   let model = llama.load('/path/to/model-dir');
//   let logits = model.forward([791, 6864, 315, 9822, 374]);

import * as models from './models.uc';
import * as weights from './weights.uc';
import * as kvcache from './kvcache.uc';
import * as sampler from './sampler.uc';
import * as math from 'math';
import {readfile} from 'fs';

const PI = 3.14159265358979;

/* Parse config.json from a model dir. */
export function load_config(model_dir) {
	return json(readfile(model_dir + '/config.json'));
}

/* Fill in config defaults and reject unsupported values. Returns cfg.
 *
 *   - head_dim defaults to hidden_size / num_attention_heads
 *     (Llama-3.1-8B's config.json omits it)
 *   - rope_scaling.rope_type: 'llama3' is handled, 'default'/absent
 *     means plain RoPE; anything else (linear, dynamic, yarn, su)
 *     dies — silently ignoring it would mis-scale every position
 */
export function normalize_config(cfg) {
	if (cfg.head_dim == null) {
		if (cfg.hidden_size == null || cfg.num_attention_heads == null)
			die('llama.load: config has no head_dim (and no hidden_size/num_attention_heads to derive it)');

		cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;

		if (cfg.head_dim != int(cfg.head_dim))
			die('llama.load: hidden_size not divisible by num_attention_heads');
	}

	let scaling = cfg.rope_scaling;
	let rt = scaling ? (scaling.rope_type != null ?
		scaling.rope_type : scaling.type) : null;

	if (rt != null && rt != 'llama3' && rt != 'default')
		die(`llama.load: unsupported rope_scaling.rope_type '${rt}' ` +
		    `(supported: llama3, default/none)`);

	return cfg;
}

/* Build the inverse-frequency vector for RoPE, with llama3 scaling when
 * config.rope_scaling.rope_type == 'llama3'. dim is head_dim; returns a
 * float32 tensor of shape (dim/2,). */
export function rope_inv_freq(cfg) {
	let dim = cfg.head_dim;
	let theta = cfg.rope_theta;
	let freqs = [];

	let scaling = cfg.rope_scaling;
	let is_llama3 = scaling && scaling.rope_type == 'llama3';

	let factor = is_llama3 ? scaling.factor : 1.0;
	let low_freq_factor = is_llama3 ? scaling.low_freq_factor : 0.0;
	let high_freq_factor = is_llama3 ? scaling.high_freq_factor : 0.0;
	let old_context = is_llama3 ? scaling.original_max_position_embeddings : 0.0;
	let low_freq_wavelen = is_llama3 ? old_context / low_freq_factor : 0.0;
	let high_freq_wavelen = is_llama3 ? old_context / high_freq_factor : 0.0;

	for (let i = 0; i < dim; i += 2) {
		let freq = 1.0 / math.pow(theta, (i * 1.0) / dim);

		if (is_llama3) {
			let wavelen = 2 * PI / freq;

			if (wavelen < high_freq_wavelen)
				push(freqs, freq);
			else if (wavelen > low_freq_wavelen)
				push(freqs, freq / factor);
			else {
				let smooth = (old_context / wavelen - low_freq_factor) /
					(high_freq_factor - low_freq_factor);
				push(freqs, (1.0 - smooth) * freq / factor + smooth * freq);
			}
		} else
			push(freqs, freq);
	}

	return torch.from_array(freqs);
}

/* Precompute the RoPE cos/sin tables for positions 0..max_seq-1.
 * Returns {cos, sin}, each (max_seq, head_dim) fp16 on GPU. */
export function rope_cache(cfg, max_seq) {
	let inv = rope_inv_freq(cfg);                    // (hd/2,) f32
	let hd = cfg.head_dim;

	let positions = [];

	for (let i = 0; i < max_seq; i++)
		push(positions, i * 1.0);

	let t = torch.from_array(positions).view([max_seq, 1]);
	let freqs = torch.matmul(t, inv.unsqueeze(0));   // (max_seq, hd/2)
	let emb = torch.cat([freqs, freqs], 1);          // (max_seq, hd)

	return {
		cos: emb.cos().cuda().to('float16'),
		sin: emb.sin().cuda().to('float16')
	};
}

/* Apply RoPE (returns new tensor): x is (batch, heads, seq, head_dim);
 * cos/sin are (seq, head_dim) and broadcast over the batch/head dims.
 * rotate_half convention:
 *   x_rot = cat(-x2, x1);  out = x*cos + x_rot*sin
 *
 * Fused single-kernel path (torch.rope, src/amp_kernels.cu), fp32
 * rotation math; non-contiguous transposed views are made contiguous
 * inside the binding. */
export function rope_apply(x, cos, sin) {
	return torch.rope(x, cos, sin);
}

/* RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.
 *
 * Fused single-kernel path (torch.rms_norm, src/amp_kernels.cu): fp32
 * variance accumulation, fp16 I/O — the numerics exllamav3's
 * RMSNorm.forward_torch specifies.
 *
 * The variance path MUST run in fp32 (all-fp16 math destroys tiny
 * embeddings — x^2 goes subnormal and mean(x^2) + eps rounds
 * catastrophically; symptom: the BOS row, values ~3e-3, produced
 * garbage logits while every normal-magnitude token matched the
 * oracle). The fused kernel accumulates in fp32; the old composite
 * ATen path (x.float() ... mean ... rsqrt) is kept below as
 * documentation of the spec it must reproduce. */
const RMS_FUSED = true;

export function rmsnorm(x, weight, eps) {
	if (RMS_FUSED)
		return torch.rms_norm(x, weight, eps);

	let f = x.to('float32');
	let ms = f.mul(f).mean(-1, true).add(eps).rsqrt();

	return f.mul(ms).to('float16').mul(weight);
}

/* GQA: expand each KV head `rep` times so kv heads match query heads.
 * x: (batch, kv_heads, seq, head_dim) -> (batch, n_heads, seq, head_dim) */
export function repeat_kv(x, rep, n_heads) {
	let s = x.shape();

	return x.unsqueeze(2)
		.expand([s[0], s[1], rep, s[2], s[3]])
		.reshape([s[0], n_heads, s[2], s[3]]);
}

/* Additive attention mask for a chunk of S queries at cache position
 * pos with total visible length pos+S: (S, pos+S) fp16 CUDA, 0 where
 * position j is visible to query i (j <= pos + i), -65504 (fp16 min)
 * where masked. Only needed for chunked prefill with history
 * (S > 1, pos > 0); costs O(S * (pos+S)) ucode work, so keep chunks
 * reasonable. */
export function chunk_mask(S, pos) {
	let len = pos + S;
	let rows = [];

	for (let i = 0; i < S; i++) {
		let row = [];

		for (let j = 0; j < len; j++)
			push(row, (j <= pos + i) ? 0.0 : -65504.0);

		push(rows, row);
	}

	return torch.from_array(rows).to('float16').cuda();
}

/* Attention block for one layer. x: (batch, seq, hidden); cos/sin are
 * (seq, head_dim) rows for THIS chunk's positions.
 *
 * With a cache: k/v are roped, written into cache slots [pos, pos+S)
 * and attention runs over slots [0, pos+S). Causal masking is handled
 * as: empty cache + multi-token chunk -> fused is_causal; history +
 * multi-token chunk -> additive chunk_mask; single-token decode -> no
 * mask needed.
 *
 * Without a cache (cache == null): stateless full-prefix attention.
 * Returns (batch, seq, hidden). */
export function attention(self, layer, x, cos, sin, cache, i, pos) {
	let cfg = self.cfg;
	let B = x.shape()[0], S = x.shape()[1];
	let nh = cfg.num_attention_heads;
	let kvh = cfg.num_key_value_heads;
	let hd = cfg.head_dim;
	let rep = nh / kvh;

	let q = layer.linears['self_attn.q_proj'].forward(x);
	let k = layer.linears['self_attn.k_proj'].forward(x);
	let v = layer.linears['self_attn.v_proj'].forward(x);

	q = q.view([B, S, nh, hd]).transpose(1, 2);
	k = k.view([B, S, kvh, hd]).transpose(1, 2);
	v = v.view([B, S, kvh, hd]).transpose(1, 2);

	q = rope_apply(q, cos, sin);
	k = rope_apply(k, cos, sin);

	let K, V, causal, mask = null;

	if (cache) {
		cache.write(i, k, v, pos);
		let len = pos + S;

		K = repeat_kv(cache.keys(i, len), rep, nh);
		V = repeat_kv(cache.values(i, len), rep, nh);
		causal = (S > 1 && pos == 0);
		mask = (S > 1 && pos > 0) ? chunk_mask(S, pos) : null;
	} else {
		K = repeat_kv(k, rep, nh);
		V = repeat_kv(v, rep, nh);
		causal = true;
	}

	let attn = mask ? torch.sdpa(q, K, V, false, mask) :
		torch.sdpa(q, K, V, causal);   // (B, nh, S, hd)

	attn = attn.transpose(1, 2).reshape([B, S, nh * hd]);

	return layer.linears['self_attn.o_proj'].forward(attn);
}

/* SwiGLU MLP block for one layer. x: (batch, seq, hidden). */
export function mlp(self, layer, x) {
	let gate = layer.linears['mlp.gate_proj'].forward(x);
	let up = layer.linears['mlp.up_proj'].forward(x);

	return layer.linears['mlp.down_proj'].forward(gate.silu().mul(up));
}

/* One transformer layer. cache/pos may be null/0 for the stateless
 * path (see attention). */
export function layer_forward(self, i, h, cos, sin, cache, pos) {
	let L = self.layers[i];
	let eps = self.cfg.rms_norm_eps;

	let r = rmsnorm(h, L.input_layernorm, eps);
	h = h.add(attention(self, L, r, cos, sin, cache, i, pos));

	r = rmsnorm(h, L.post_attention_layernorm, eps);
	h = h.add(mlp(self, L, r));

	return h;
}

/* Build the full model: config + per-layer linears + norms + embed.
 *
 * lm_head routing: a quantized lm_head (lm_head.trellis present) wins
 * over tie_word_embeddings — exllamav3 quantizes the head separately
 * even when the HF config says tied. Fall back to the fp16 embed (tied)
 * or a plain fp16 lm_head.weight matmul. */
export function load(model_dir) {
	let cfg = normalize_config(load_config(model_dir));
	let ws = weights.open(model_dir);

	// load-time fast path: one batched pinned-async transfer for the
	// whole model; without it every tensor sync-copies on first touch
	ws.upload();

	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++)
		push(layers, {
			linears: models.layer_linears(ws, `model.layers.${i}`, true),
			input_layernorm: ws.gpu16(`model.layers.${i}.input_layernorm.weight`),
			post_attention_layernorm: ws.gpu16(`model.layers.${i}.post_attention_layernorm.weight`)
		});

	// embeddings: plain fp16/bf16 only for now — a quantized embed table
	// would need dequant-at-load (exllamav3 does the same internally)
	if (ws.has('model.embed_tokens.trellis'))
		die('llama.load: quantized embed_tokens not supported yet');

	let embed_tokens = ws.gpu16('model.embed_tokens.weight');

	// lm_head routing: a quantized head wins over tie_word_embeddings —
	// exllamav3 quantizes the head separately even when the HF config
	// says tied; then a plain lm_head.weight; then the tied embed
	let head;
	if (ws.has('lm_head.trellis'))
		head = models.LinearEXL3Seq(ws, 'lm_head');
	else if (ws.has('lm_head.weight'))
		head = models.LinearFP16(ws, 'lm_head');
	else if (cfg.tie_word_embeddings)
		head = {
			out_features: cfg.vocab_size,
			forward: function (h) {
				return torch.matmul(h, embed_tokens.t());
			}
		};
	else
		die('llama.load: no lm_head weight and embeddings are not tied');

	return {
		cfg: cfg,
		layers: layers,
		embed_tokens: embed_tokens,
		norm: ws.gpu16('model.norm.weight'),
		lm_head: head,

		/* A KV cache sized for max_seq positions, pre-filled with the
		 * model's RoPE cos/sin tables for those positions. */
		new_cache: function (max_seq) {
			let c = kvcache.create(this.cfg, max_seq);
			let rc = rope_cache(this.cfg, max_seq);

			c.cos = rc.cos;
			c.sin = rc.sin;

			return c;
		},

		/* Run a chunk of token_ids (array of ints) appending to cache
		 * at the cache's current length. Returns logits for the whole
		 * chunk: (1, S, vocab) fp16 on GPU.
		 *
		 * `trace` (optional array): when provided, the hidden state
		 * after each decoder layer (pre final-norm) is appended to it —
		 * the per-layer comparison hook used by the Phase-4 harness. */
		forward_chunk: function (token_ids, cache, trace) {
			let S = length(token_ids);
			let pos = cache.length;

			let ids = torch.from_array(token_ids, 'int64').view([1, S]).cuda();
			let cos = cache.cos.narrow(0, pos, S);
			let sin = cache.sin.narrow(0, pos, S);

			let h = this.embed_tokens.index(ids);   // (1, S, hidden)

			for (let i = 0; i < this.cfg.num_hidden_layers; i++) {
				h = layer_forward(this, i, h, cos, sin, cache, pos);

				if (trace)
					push(trace, h);
				}

			h = rmsnorm(h, this.norm, this.cfg.rms_norm_eps);

			return this.lm_head.forward(h);
		},

		/* Stateless prefill: fresh cache, discarded afterwards.
		 * Returns logits (1, seq, vocab) fp16 on GPU. `trace` behaves
		 * as in forward_chunk. */
		forward: function (token_ids, trace) {
			let cache = this.new_cache(length(token_ids));
			let logits = this.forward_chunk(token_ids, cache, trace);

			return logits;
		},

		/* Generation: prompt is an array of token ids; returns a NEW
		 * array prompt ++ max_new generated ids.
		 *
		 * opts (all optional, see sampler.uc):
		 *   temperature  0 = greedy argmax (default), else softmax temp
		 *   top_k        keep the k highest logits (0 = off)
		 *   seed         math.srand seed, drawn ONCE before generating */
		generate: function (prompt, max_new, opts) {
			opts = opts || {};

			if (opts.seed != null)
				math.srand(opts.seed);

			let cache = this.new_cache(length(prompt) + max_new);
			let out = [];

			for (let i = 0; i < length(prompt); i++)
				push(out, prompt[i]);

			let logits = this.forward_chunk(prompt, cache);   // (1, S, V)
			let last = logits.narrow(1, length(prompt) - 1, 1);

			for (let n = 0; n < max_new; n++) {
				let tok = sampler.sample(last, opts);

				push(out, tok);

				if (n + 1 >= max_new)
					break;

				last = this.forward_chunk([tok], cache);   // (1, 1, V)
			}

			return out;
		}
	};
}