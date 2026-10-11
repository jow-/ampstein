// ucode/qwen35_graph.uc — CUDA-graph-captured qwen35 forward steps for
// decode (plain S=1 steps, MTP verify windows and replays).
//
// One step of S tokens at the cache's current position runs as one
// replayed graph per device stage (the layer split's runs of layers on
// one GPU), joined by an asynchronous peer copy of the hidden state:
// the host enqueues a handful of operations instead of ~1600 kernel
// launches per token.
//
// Graph-safety rules (docs/LESSONS.md L38/L39): shapes and addresses are baked
// in at capture, values are read at replay. So:
//   - token ids, the position and everything derived from it (the KV
//     slots pos..pos+S-1, the attention key count, the RoPE rows) live
//     in persistent device buffers, refreshed before each replay;
//   - attention reads the full-size KV cache with a device-side key
//     count (qwen35.attn_forward_dev: index_copy_ + flash_attn_kvcache);
//   - GDN layers update their states in place (as in eager mode);
//   - every tensor a graph touches stays referenced from `st`.
//
// Usage:
//   import * as qgraph from 'ampstein.qwen35_graph';
//   let G = qgraph.create(model, cache, [1, 3]);   // capture step sizes
//   let r = G.run(ids, pos);    // ids: array of S token ids at pos
//   r.logits                    // (1, S, V) fp16 (buffer reused per run)
//   r.hidden                    // (1, S, hidden) post-final-norm hidden
//
// create() runs one eager warmup step per size (autotune, allocator,
// cuBLAS state) and CLOBBERS the cache contents doing so: create the
// graphs before any real prefill, then reset the cache state.

import * as qwen35 from './qwen35.uc';

/* Device stages: maximal runs of consecutive layers on one device. */
function stages_of(model) {
	let out = [];

	for (let i = 0; i < model.cfg.num_hidden_layers; i++) {
		let d = model.layers[i].dev;

		if (!length(out) || out[length(out) - 1].dev != d)
			push(out, {dev: d, first: i, end: i + 1});
		else
			out[length(out) - 1].end = i + 1;
	}

	return out;
}

/* Build the persistent buffers and captured graphs for step size S. */
function build(model, cache, S) {
	let cfg = model.cfg;
	let stages = stages_of(model);
	let max_seq = cache.max_seq;
	let H = cfg.hidden_size;
	let st = {S: S, stages: []};

	st.ids = torch.zeros([1, S], 'int64', 'cuda:' + stages[0].dev);

	for (let k = 0; k < length(stages); k++) {
		let sg = stages[k];
		let dev = 'cuda:' + sg.dev;
		let s = {
			dev: sg.dev, first: sg.first, end: sg.end,
			pos: torch.zeros([1], 'int64', dev),
			rpos: torch.zeros([1], 'int64', dev),   // pos + MRoPE offset
			ramp: torch.arange(S, 'int64', dev),
			cu_q: torch.from_array([0, S], 'int32', dev),
			cu_k: torch.from_array([0, max_seq], 'int32', dev),
			h_in: (k > 0) ? torch.zeros([1, S, H], 'float16', dev) : null,
			h_out: null, graph: null
		};

		// the stage's RoPE tables (one per device, shared by its
		// full-attention layers)
		for (let i = sg.first; i < sg.end; i++) {
			if (cache.layers[i].cos) {
				s.cos_tab = cache.layers[i].cos;
				s.sin_tab = cache.layers[i].sin;
				break;
			}
		}

		push(st.stages, s);
	}

	st.logits = null;
	st.hidden = null;

	/* the stage body: what each graph records */
	function body(k) {
		let s = st.stages[k];
		let h = (k == 0) ? model.embed_tokens.index(st.ids) : s.h_in;

		// device-side positions for this step
		s.pos_idx = s.ramp.add(s.pos);
		s.seqused = s.pos_idx.narrow(0, S - 1, 1).add(1).to('int32');

		// RoPE rows follow the text position, which trails the KV slot
		// by a constant after an image (multimodal prompts)
		if (s.cos_tab) {
			s.rope_idx = s.ramp.add(s.rpos);
			s.cos = s.cos_tab.index(s.rope_idx);
			s.sin = s.sin_tab.index(s.rope_idx);
		}

		for (let i = s.first; i < s.end; i++)
			h = qwen35.layer_forward(cfg, model.layers[i], cache.layers[i],
				h, null, null, null, s);

		if (k < length(st.stages) - 1) {
			s.h_out = h;
			return;
		}

		let hn = qwen35.rmsnorm_bias1(h, model.norm, cfg.rms_norm_eps);

		st.hidden = hn;
		st.logits = model.lm_head.forward(hn);
	}

	// eager warmup (autotune, cuBLAS handles, allocator) at position 0
	for (let k = 0; k < length(st.stages); k++) {
		st.stages[k].pos.fill_(0);
		st.stages[k].rpos.fill_(0);

		if (k > 0)
			st.stages[k].h_in.copy_(st.stages[k - 1].h_out);

		body(k);
	}

	torch.synchronize();

	// capture: one graph per stage, on its device
	for (let k = 0; k < length(st.stages); k++) {
		let s = st.stages[k];
		let g = torch.graph();

		g.capture_begin(s.dev);
		body(k);
		g.capture_end();
		s.graph = g;
	}

	torch.synchronize();

	return st;
}

export function create(model, cache, sizes) {
	let by_size = {};

	for (let i = 0; i < length(sizes); i++)
		by_size[sizes[i]] = build(model, cache, sizes[i]);

	return {
		sizes: sizes,

		has: function (S) {
			return by_size[S] != null;
		},

		/* Run ids (array of S token ids) at position pos (the cache's
		 * current length). Advances cache.length by S. */
		run: function (ids, pos) {
			let S = length(ids);
			let st = by_size[S];

			if (!st)
				die(`qwen35_graph: no graph captured for step size ${S}`);

			st.ids.copy_(torch.from_array(ids, 'int64', 'cuda:' + st.stages[0].dev).view([1, S]));

			for (let k = 0; k < length(st.stages); k++) {
				let s = st.stages[k];

				s.pos.fill_(pos);
				s.rpos.fill_(pos + qwen35.mm_off(cache.mm, pos));

				if (k > 0)
					s.h_in.copy_(st.stages[k - 1].h_out);

				s.graph.replay();
			}

			cache.length = pos + S;
			cache.hidden = st.hidden;

			return {logits: st.logits, hidden: st.hidden};
		},
	};
}
