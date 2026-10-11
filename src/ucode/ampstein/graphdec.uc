// ucode/graphdec.uc — graph-captured decode with a DEVICE-SIDE step
// loop for the llama family.
//
// Three layers of optimization stack up here:
//
//   1. CUDA graph capture: one decode step = ONE launch (the whole
//      kernel sequence is recorded; replay re-executes it).
//   2. Device-side loop: the token id, the position and the emitted-
//      token log live in persistent device buffers. The captured step
//      reads the CURRENT values, computes the next token (topk) and
//      advances the position — so N decode steps are N back-to-back
//      replays with ZERO host work and ZERO device-to-host traffic
//      until the very end. The old per-step .item() sync is gone.
//   3. Batched linears: q/k/v run as ONE sliced mgemm launch and
//      gate/up as ONE stacked mgemm launch (see models.uc); the
//      residual add + RMSNorm pair is one fused kernel
//      (torch.rms_norm_res) and SwiGLU activation is one fused kernel
//      (torch.silu_mul).
//
// Capture constraints shape the code (see docs/LESSONS.md L38/L39):
//   - SHAPES are baked in: only S=1 steps are captured; prefill stays
//     on the eager path (llama.forward_chunk).
//   - ADDRESSES are baked in: every tensor the graph reads or writes
//     lives in the persistent state object `st` for the decoder's
//     whole lifetime. The GC collected closure-held tensors in an
//     earlier version and replays read garbage — keep everything in
//     ONE object referenced by the returned closures.
//   - VALUES are read at replay: the device-side position/token-id
//     buffers drive the embedding gather, the KV write slot, the
//     attention span and the argmax_advance state update — all
//     re-read device memory on every replay.
//   - No host syncs inside capture: no .item()/.tolist()/D2H copies.
//
// Usage:
//   import * as graphdec from 'ampstein.graphdec';
//   let dec = graphdec.create(model, max_seq);
//   dec.prefill(prompt_ids);     // eager, once; returns prefill logits
//   dec.run(n);                  // n captured steps, one replay call
//   let toks = dec.tokens();     // the n generated token ids
//   let logits = dec.logits();   // final-step logits (1, 1, vocab)

import * as llama from './llama.uc';
import * as models from './models.uc';
import * as kvcache from './kvcache.uc';
import * as math from 'math';

/* Build the decoder for a loaded llama model (ucode/llama.uc load()).
 * max_seq: KV capacity. Runs one eager warmup step (autotune, SDPA
 * backend pick, cuBLAS init, exl3 lock buffers — everything that must
 * not happen inside capture), then captures the step. */
export function create(model, max_seq, opts) {
	opts = opts || {};
	let cfg = model.cfg;
	let nh = cfg.num_attention_heads;
	let kvh = cfg.num_key_value_heads;
	let hd = cfg.head_dim;
	let nl = cfg.num_hidden_layers;
	let hidden = cfg.hidden_size;
	let inter = cfg.intermediate_size;
	let eps = cfg.rms_norm_eps;

	let st = {
		/* device-side loop state */
		ids: torch.zeros([1, 1], 'int64', 'cuda'),
		pos: torch.zeros([1], 'int64', 'cuda'),
		log: torch.zeros([max_seq, 1], 'int64', 'cuda'),

		/* residual stream + first-layer zero residual */
		h: torch.zeros([1, 1, hidden], 'float16', 'cuda'),
		zerores: torch.zeros([1, 1, hidden], 'float16', 'cuda'),

		/* the captured output */
		logits: null
	};

	/* per-layer decode buffers + batched-linear bundles */
	st.qbuf = [];
	st.kbuf = [];
	st.vbuf = [];
	st.guout = [];
	st.oout = [];
	st.dout = [];
	st.qkv = [];
	st.gu = [];

	for (let i = 0; i < nl; i++) {
		let L = model.layers[i];
		let q = L.linears['self_attn.q_proj'];
		let k = L.linears['self_attn.k_proj'];
		let v = L.linears['self_attn.v_proj'];
		let gate = L.linears['mlp.gate_proj'];
		let up = L.linears['mlp.up_proj'];
		let down = L.linears['mlp.down_proj'];

		let qbuf = torch.zeros([1, 1, nh * hd], 'float16', 'cuda');
		let kbuf = torch.zeros([1, 1, kvh * hd], 'float16', 'cuda');
		let vbuf = torch.zeros([1, 1, kvh * hd], 'float16', 'cuda');
		let guout = torch.zeros([2, 1, inter], 'float16', 'cuda');
		let oout = torch.zeros([1, 1, hidden], 'float16', 'cuda');
		let dout = torch.zeros([1, 1, hidden], 'float16', 'cuda');

		push(st.qbuf, qbuf);
		push(st.kbuf, kbuf);
		push(st.vbuf, vbuf);
		push(st.guout, guout);
		push(st.oout, oout);
		push(st.dout, dout);

		/* q+k+v: unequal widths -> sliced mode, one launch */
		push(st.qkv, models.sliced([q, k, v], [qbuf, kbuf, vbuf]));

		/* gate+up: equal widths -> stacked mode, one launch */
		push(st.gu, models.multi([gate, up], guout));
	}

	/* RoPE tables for ALL positions (the eager prefill path reads
	 * them too) */
	let rc = llama.rope_cache(cfg, max_seq);
	st.cos = rc.cos;   // (max_seq, hd) fp16 CUDA
	st.sin = rc.sin;

	let cache = kvcache.create(cfg, max_seq);
	cache.cos = st.cos;
	cache.sin = st.sin;

	/* persistent output buffers for the single-weight gemms */
	st.attn = [];
	st.act = [];

	for (let i = 0; i < nl; i++) {
		push(st.attn, torch.zeros([1, nh * hd], 'float16', 'cuda'));
		push(st.act, torch.zeros([1, inter], 'float16', 'cuda'));
	}

	st.normed = torch.zeros([1, hidden], 'float16', 'cuda');
	st.logits_buf = torch.zeros([1, cfg.vocab_size], 'float16', 'cuda');

	/* fused decode-attention workspace: (kvh, NS, rep, wsrow) fp32;
	 * wsrow = 4 + hd padded to a multiple of 4 (float4 lanes) */
	let NS = 8;
	let wsrow = 4 + hd + ((4 - hd % 4) % 4);
	st.ws = torch.zeros([kvh, NS, nh / kvh, wsrow], 'float32', 'cuda');
	st.attn_scale = 1.0 / math.sqrt(hd);

	/* One decode step over the CURRENT state values. Runs eagerly
	 * (warmup) and inside the capture; never syncs with the host.
	 *
	 * On entry: st.ids = token at position st.pos.
	 * On exit:  st.ids = next token, st.pos advanced, the token at
	 *           the old position recorded in st.log. */
	function step_ops() {
	let h = model.embed_tokens.index(st.ids);   // (1, 1, hidden)

		for (let i = 0; i < nl; i++) {
			let L = model.layers[i];

			/* residual add + norm, fused: h += prev residual
			 * contribution (zero for the first), r = norm(h).
			 * BOTH outputs matter: rn[0] is the updated residual
			 * stream (the MLP contribution accumulates HERE),
			 * rn[1] feeds the projections. */
			let res = (i == 0) ? st.zerores : st.dout[i - 1];
			let rn = torch.rms_norm_res(h, res,
				L.input_layernorm, eps);
			h = rn[0];
			let r = rn[1];                       // (1, 1, hidden)

			/* q+k+v in ONE sliced mgemm launch */
			st.qkv[i].run(r);

			/* fused decode attention: ropes q and k, writes the
			 * KV pair at the device-computed slot, attends over
			 * [0, pos] with GQA — two kernels total */
			if (!opts.skip_attn)
				torch.decode_attn(st.qbuf[i].view([nh, hd]),
					st.kbuf[i].view([kvh, hd]),
					st.vbuf[i].view([kvh, hd]),
					cache.k[i].view([kvh, max_seq, hd]),
					cache.v[i].view([kvh, max_seq, hd]),
					st.cos, st.sin, st.ws,
					st.attn[i], st.pos, st.attn_scale);

			/* o_proj into the persistent buffer */
			let o = L.linears['self_attn.o_proj'];

			torch.exl3_gemm(st.attn[i], o.trellis,
				st.oout[i].view([1, hidden]),
				o.suh, o.svh, o.mul1, o.mcg);

			/* h += o_out; r = norm(h) */
			rn = torch.rms_norm_res(h, st.oout[i],
				L.post_attention_layernorm, eps);
			h = rn[0];
			r = rn[1];

			/* gate+up in ONE stacked mgemm launch */
			st.gu[i].run(r);

			/* fused SwiGLU activation */
			let a = torch.silu_mul(st.guout[i].narrow(0, 0, 1),
				st.guout[i].narrow(0, 1, 1));

			/* down_proj into the persistent buffer; it becomes
			 * the next layer's residual contribution */
			let dn = L.linears['mlp.down_proj'];

			torch.exl3_gemm(a.view([1, inter]),
				dn.trellis,
				st.dout[i].view([1, hidden]),
				dn.suh, dn.svh, dn.mul1, dn.mcg);
		}

		/* final norm: h += last down output, norm */
		let fin = torch.rms_norm_res(h, st.dout[nl - 1],
			model.norm, eps);

		if (opts.skip_head) {
			st.logits = fin[1].view([1, 1, hidden]);
			return;
		}

		/* lm_head into the persistent logits buffer */
		torch.exl3_gemm(fin[1].view([1, hidden]),
			model.lm_head.trellis, st.logits_buf,
			model.lm_head.suh, model.lm_head.svh,
			model.lm_head.mul1, model.lm_head.mcg);

		/* next token + position advance, all on device, ONE
		 * kernel: argmax over the vocab, log[pos] = ids (the
		 * token just processed), ids = argmax, pos += 1 */
		torch.argmax_advance(st.logits_buf, st.ids, st.pos,
			st.log);

		st.logits = st.logits_buf.view([1, 1, cfg.vocab_size]);
	}

	/* Eager prefill of prompt_ids through the varshape path; the
	 * captured steps then append from cache.length on. Returns the
	 * prefill logits (1, S, vocab) — position S-1 scores the first
	 * generated token. */
	function prefill(ids) {
		return model.forward_chunk(ids, cache);
	}

	/* warmup: one real step at slot 0 (executes; slot 0 is rewritten
	 * by any later prefill, and masked slots never win attention) */
	st.ids.copy_(torch.from_array([0], 'int64'));
	st.pos.copy_(torch.from_array([0], 'int64'));
	step_ops();
	torch.synchronize();

	/* capture the same op sequence (kernels are recorded, not run) */
	let g = torch.graph();
	g.capture_begin();
	step_ops();
	g.capture_end();
	torch.synchronize();

	return {
		cache: cache,
		graph: g,
		state: st,

		/* Number of valid positions in the cache. */
		length: function () {
			return cache.length;
		},

		prefill: prefill,

		/* Run `steps` captured decode steps. ONE host call: the
		 * steps chain on-device (each step's argmax feeds the
		 * next step's embedding gather). The first step processes
		 * the token left in st.ids by start_at(). */
		run: function (steps) {
			if (cache.length + steps > max_seq)
				die('graphdec: cache full');

			g.replay(steps);
			cache.length += steps;
			st.gen_len = (st.gen_len || 0) + steps;
		},

		/* Seed the device-side loop: `tok` is the token at
		 * position cache.length (e.g. the argmax of the prefill
		 * logits). Host->device, once per generation. Tokens
		 * generated by run() are read back with tokens(). */
		start_at: function (tok) {
			st.gen_from = cache.length;
			st.gen_len = 0;
			st.ids.copy_(torch.from_array([tok], 'int64'));
			st.pos.copy_(torch.from_array([cache.length], 'int64'));
		},

		/* The tokens recorded by the captured steps (defaults to
		 * exactly the range produced since the last start_at() —
		 * NOT the whole log buffer, whose tail is stale). One D2H
		 * sync. */
		tokens: function (from, to) {
			let t = st.log.cpu().tolist();
			let out = [];

			for (let i = (from != null ? from : st.gen_from);
					i < (to != null ? to :
					(st.gen_from || 0) + (st.gen_len || 0)); i++)
				push(out, t[i][0]);

			return out;
		},

		/* Logits of the most recent step (1, 1, vocab) fp16 CUDA —
		 * REUSED every run() call; copy out before advancing. */
		logits: function () {
			return st.logits;
		},

		/* Greedy next token of the most recent step (host sync). */
		next_token: function () {
			return st.logits_buf.topk(1, 1).indices.item();
		}
	};
}