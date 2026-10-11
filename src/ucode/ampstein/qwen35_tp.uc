// ucode/qwen35_tp.uc — tensor-parallel qwen35 over N GPUs.
//
// Every layer runs on ALL ranks, each on its share of it; the residual
// stream h is replicated (bit-identical on every rank) and the ranks'
// partial sublayer outputs are summed by an all-reduce. Decode reads
// 1/N of the weights per GPU per token, all GPUs at once — the layer
// split reads all of them, one GPU at a time.
//
// Sharding (rank r gets share w[r] of every split dimension, rounded to
// the alignment each needs — MLP and vocabulary columns: 256; GDN: k
// heads in pairs, so the q/k (128 per head) and v (3 x 128 per head)
// column ranges stay 256-aligned; attention: whole KV-head groups):
//   GDN        q|k|v by k-head groups (their v heads follow):
//              in_proj_qkv / conv taps by channel, in_proj_z / a / b,
//              dt_bias, A_log by v head; out_proj by ROWS (partial sum)
//   attention  q/gate (interleaved per head) and k/v by head; o_proj by
//              rows. With more ranks than KV heads some ranks hold none
//              and contribute a zero partial.
//   MLP        gate/up by columns, down by rows
//   lm_head    by columns (vocabulary ranges in rank order), logits
//              gathered on rank 0's device
//   norms      replicated; the embedding stays on rank 0
// Row-split EXL3 products are exact in sum (the output Hadamard and svh
// are linear in the accumulated product). The all-reduce
// (torch.p2p_comm) sums in fp32 in rank order on every rank, so h stays
// bit-identical everywhere.
//
// Usage:
//   let m = qwen35.load(dir, {devices: layer split over the same GPUs});
//   let tp = qwen35_tp.shard(m, {devices: [0, 1, 2]});  // consumes m's layers
//   let cache = tp.new_cache(max_seq);
//   let r = tp.forward(ids, cache);    // {logits (1,S,V) on rank 0, hidden}

import * as qwen35 from './qwen35.uc';
import * as models from './models.uc';

/* Cut `total` (a multiple of `unit`) into per-rank [start, length]
 * ranges proportional to weights w (largest remainder; a rank may get
 * nothing when there are fewer units than ranks). */
function split_n(total, unit, w) {
	let n = total / unit, G = length(w), sum = 0;

	for (let x in w)
		sum += x;

	let cnt = [], rem = [], used = 0;

	for (let g = 0; g < G; g++) {
		let exact = n * w[g] / sum;
		let c = int(exact);

		push(cnt, c);
		push(rem, [ exact - c, g ]);
		used += c;
	}

	rem = sort(rem, (a, b) => (b[0] - a[0]) || (a[1] - b[1]));

	for (let k = 0; used < n; k++, used++)
		cnt[rem[k % G][1]]++;

	let out = [], at = 0;

	for (let g = 0; g < G; g++) {
		push(out, [ at * unit, cnt[g] * unit ]);
		at += cnt[g];
	}

	return out;
}

function plan_of(cfg, w, V) {
	return {
		weights: w,
		mlp: split_n(cfg.intermediate_size, 256, w),
		gdn_k: split_n(cfg.ln_num_k_heads, 2, w),     // k heads [first, count]
		attn_kv: split_n(cfg.num_kv_heads, 1, w),     // kv heads [first, count]
		vocab: split_n(V, 256, w)
	};
}

/* per-rank config: this rank's share of the heads */
function shard_cfg(cfg, plan, g) {
	let c = {...cfg};
	let hk = plan.gdn_k[g][1], kv = plan.attn_kv[g][1];
	let rep = cfg.ln_num_v_heads / cfg.ln_num_k_heads;

	c.ln_num_k_heads = hk;
	c.ln_num_v_heads = hk * rep;
	c.ln_k_dim = c.ln_num_k_heads * cfg.ln_k_head_dim;
	c.ln_v_dim = c.ln_num_v_heads * cfg.ln_v_head_dim;
	c.fdim_qkv = 2 * c.ln_k_dim + c.ln_v_dim;
	c.num_kv_heads = kv;
	c.num_q_heads = kv * (cfg.num_q_heads / cfg.num_kv_heads);

	return c;
}

/* rank g's shard of layer L, placed on device dev */
function shard_layer(cfg, plan, L, g, dev) {
	let mi = plan.mlp[g];
	let S = {
		type: L.type, dev: dev,
		input_layernorm: L.input_layernorm.cuda(dev),
		post_attention_layernorm: L.post_attention_layernorm.cuda(dev),
		gate_proj: models.cols(L.gate_proj, [ mi ], dev),
		up_proj: models.cols(L.up_proj, [ mi ], dev),
		down_proj: models.rows(L.down_proj, mi[0], mi[1], dev)
	};

	if (L.type == 'linear_attention') {
		let rep = cfg.ln_num_v_heads / cfg.ln_num_k_heads;
		let kd = cfg.ln_num_k_heads * cfg.ln_k_head_dim;   // full q (= k) width
		let k0 = plan.gdn_k[g][0] * cfg.ln_k_head_dim;
		let kq = plan.gdn_k[g][1] * cfg.ln_k_head_dim;
		let v0 = plan.gdn_k[g][0] * rep * cfg.ln_v_head_dim;
		let vh = plan.gdn_k[g][1] * rep * cfg.ln_v_head_dim;
		let h0 = plan.gdn_k[g][0] * rep;
		let hv = plan.gdn_k[g][1] * rep;

		if (!kq)
			die('qwen35_tp: a rank without GDN heads is not supported (too many ranks)');

		S.in_proj_qkv = models.cols(L.in_proj_qkv,
			[ [ k0, kq ], [ kd + k0, kq ], [ 2 * kd + v0, vh ] ], dev);
		S.in_proj_z = models.cols(L.in_proj_z, [ [ v0, vh ] ], dev);
		S.in_proj_b = L.in_proj_b.trellis ?
			models.cols(L.in_proj_b, [ [ h0, hv ] ], dev) :
			models.fp16_cols(L.in_proj_b, h0, hv, dev);
		S.in_proj_a = L.in_proj_a.trellis ?
			models.cols(L.in_proj_a, [ [ h0, hv ] ], dev) :
			models.fp16_cols(L.in_proj_a, h0, hv, dev);
		S.out_proj = models.rows(L.out_proj, v0, vh, dev);

		S.conv_w = torch.cat([ L.conv_w.narrow(0, k0, kq),
			L.conv_w.narrow(0, kd + k0, kq),
			L.conv_w.narrow(0, 2 * kd + v0, vh) ], 0).contiguous().cuda(dev);
		S.dt_bias = L.dt_bias.narrow(0, h0, hv).contiguous().cuda(dev);
		S.neg_exp_a_log = L.neg_exp_a_log.narrow(0, h0, hv).contiguous().cuda(dev);
		S.norm_w = L.norm_w.cuda(dev);
	}
	else {
		let hd = cfg.head_dim;
		let qpk = cfg.num_q_heads / cfg.num_kv_heads;
		let kv0 = plan.attn_kv[g][0], kvn = plan.attn_kv[g][1];

		S.no_attn = (kvn == 0);

		if (!S.no_attn) {
			// q_proj packs q and gate interleaved per q head (2 * hd each)
			S.q_proj = models.cols(L.q_proj, [ [ kv0 * qpk * 2 * hd, kvn * qpk * 2 * hd ] ], dev);
			S.k_proj = models.cols(L.k_proj, [ [ kv0 * hd, kvn * hd ] ], dev);
			S.v_proj = models.cols(L.v_proj, [ [ kv0 * hd, kvn * hd ] ], dev);
			S.o_proj = models.rows(L.o_proj, kv0 * qpk * hd, kvn * qpk * hd, dev);
			S.q_norm_w = L.q_norm_w.cuda(dev);
			S.k_norm_w = L.k_norm_w.cuda(dev);
		}
	}

	return S;
}

/* largest all-reduce payload in rows through the P2P kernel: decode
 * and MTP windows; prefill chunks take the copy path */
const P2P_ROWS = 16;

/* all-reduce p (one partial per rank) -> per-rank sums, bit-identical
 * across ranks. Small payloads: torch.p2p_comm (direct peer stores +
 * flags, no cross-device stream dependency — those cost 25-90 us of
 * idle per all-reduce inside a graph, docs/LESSONS.md L67). Large ones: peer
 * copies of every partial to every rank, summed in fp32 in rank order
 * (the same rounding as the kernel). */
function allreduce(tp, p) {
	let G = length(p);

	if (p[0].numel() <= tp.comm_cap)
		return map(p, (x, g) => tp.comm.allreduce(x, g));

	let out = [];

	for (let g = 0; g < G; g++) {
		let dev = tp.devs[g];
		let acc = p[0].cuda(dev).to('float32');

		for (let r = 1; r < G; r++)
			acc = acc.add(p[r].cuda(dev).to('float32'));

		push(out, acc.to('float16'));
	}

	return out;
}

/* opts.devices: CUDA devices, one rank each (default [0, 1]);
 * opts.weights: per-rank share weights (default equal); opts.f0 (two
 * ranks): rank 0's share. */
export function shard(model, opts) {
	opts = opts || {};

	let cfg = model.cfg;
	let V = model.lm_head.out_features;
	let devs = opts.devices || [ 0, 1 ];
	let G = length(devs);
	let w = opts.weights;

	if (!w && opts.f0 != null && G == 2)
		w = [ opts.f0, 1 - opts.f0 ];

	if (!w) {
		w = [];

		for (let g = 0; g < G; g++)
			push(w, 1);
	}

	let plan = plan_of(cfg, w, V);
	let cg = map(devs, (d, g) => shard_cfg(cfg, plan, g));
	let layers = [];

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let L = model.layers[i];

		push(layers, map(devs, (d, g) => shard_layer(cfg, plan, L, g, d)));

		// drop the full layer: its device memory goes back to the
		// caching allocator for the shards still to come
		model.layers[i] = null;

		if (i % 8 == 7)
			gc();
	}

	gc();

	let heads = map(devs, (d, g) => models.cols(model.lm_head, [ plan.vocab[g] ], d));
	let cap = P2P_ROWS * cfg.hidden_size;

	let tp = {
		cfg: cfg, cfg_g: cg, plan: plan, layers: layers,
		comm: torch.p2p_comm(devs, cap), comm_cap: cap,
		embed_tokens: model.embed_tokens,
		norm: map(devs, (d) => model.norm.cuda(d)),
		lm_heads: heads,
		// the full head stays on its device for the MTP draft (and the
		// MTP block itself is not sharded)
		lm_head: model.lm_head,
		mtp: model.mtp,
		devs: devs,
		G: G,
		tp: true,

		// device of the post-norm hidden handed out (the MTP block's)
		hdev: model.mtp ? model.mtp.dev : devs[0],

		/* GDN state holders of layer i: one per rank */
		states: function (cache, i) {
			return cache.layers[i];
		},

		new_cache: function (max_seq) {
			let cache = {max_seq: max_seq, length: 0, layers: []};
			let rope = map(devs, (d) => qwen35.rope_cache(cfg, max_seq, d));

			for (let i = 0; i < cfg.num_hidden_layers; i++) {
				let row = [];

				for (let g = 0; g < G; g++) {
					let dev = 'cuda:' + devs[g];
					let cl = {type: cfg.layer_types[i], dev: devs[g]};

					if (cl.type == 'linear_attention') {
						cl.conv_state = torch.zeros([1, cg[g].fdim_qkv, cfg.conv_kernel - 1],
							'float16', dev);
						cl.rec_state = torch.zeros([cg[g].ln_num_v_heads,
							cfg.ln_k_head_dim, cfg.ln_v_head_dim], 'float32', dev);
					}
					else if (cg[g].num_kv_heads > 0) {
						cl.k_cache = torch.zeros([1, cg[g].num_kv_heads, max_seq, cfg.head_dim],
							'float16', dev);
						cl.v_cache = torch.zeros([1, cg[g].num_kv_heads, max_seq, cfg.head_dim],
							'float16', dev);
					}

					// RoPE tables on every rank (also the graph state
					// takes them from any attention layer)
					if (cl.type != 'linear_attention') {
						cl.cos = rope[g].cos;
						cl.sin = rope[g].sin;
					}

					push(row, cl);
				}

				push(cache.layers, row);
			}

			if (this.mtp) {
				let dc = qwen35.new_mtp_cache(cfg, max_seq, this.mtp.dev);
				let rc = qwen35.rope_cache(cfg, max_seq, this.mtp.dev);

				dc.cos = rc.cos;
				dc.sin = rc.sin;
				cache.mtp = dc;
			}

			return cache;
		},

		/* One layer on all ranks. h: replicated per-rank hidden; returns
		 * the new list. st (optional): per-rank device-position state
		 * for graph capture (see qwen35_graph), else host pos. */
		layer: function (i, h, cache, pos, st) {
			let eps = cfg.rms_norm_eps;
			let L = this.layers[i];
			let cls = cache.layers[i];
			let p = [], r = [];
			let S = h[0].shape()[1];

			for (let g = 0; g < G; g++)
				push(r, qwen35.rmsnorm_bias1(h[g], L[g].input_layernorm, eps));

			for (let g = 0; g < G; g++) {
				if (L[g].type == 'linear_attention')
					push(p, qwen35.gdn_forward(cg[g], L[g], r[g], cls[g]));
				else if (L[g].no_attn)
					push(p, torch.zeros([1, S, cfg.hidden_size], 'float16', 'cuda:' + devs[g]));
				else if (st)
					push(p, qwen35.attn_forward_dev(cg[g], L[g], r[g], cls[g], st[g]));
				else {
					let rr = qwen35.rope_rows(cfg, cache.mm, cls[g], pos, S);

					push(p, qwen35.attn_forward(cg[g], L[g], r[g], rr.cos, rr.sin,
						cls[g], pos));
				}
			}

			let s = allreduce(this, p);

			for (let g = 0; g < G; g++) {
				let hr = torch.rms_norm_res(h[g], s[g], L[g].post_attention_layernorm, eps);

				h[g] = hr[0];
				r[g] = hr[1];
			}

			for (let g = 0; g < G; g++)
				p[g] = qwen35.mlp_forward(L[g], r[g]);

			s = allreduce(this, p);

			return map(h, (x, g) => x.add(s[g]));
		},

		/* Logits gathered on rank 0 for the last `rows` positions (null:
		 * all) and the post-norm hidden (on hdev) of the same rows — or
		 * of all rows with hidden_all. */
		head: function (h, rows, hidden_all) {
			let eps = cfg.rms_norm_eps;
			let S = h[0].shape()[1];
			let cut = rows != null && rows < S;
			let hn = [], ls = [];
			let hidden = null;

			for (let g = 0; g < G; g++) {
				let x = (cut && !hidden_all) ? h[g].narrow(1, S - rows, rows).contiguous() : h[g];
				let n = qwen35.rmsnorm_bias1(x, this.norm[g], eps);
				let hl = (cut && hidden_all) ? n.narrow(1, S - rows, rows).contiguous() : n;
				let l = this.lm_heads[g].forward(hl);

				push(ls, (g == 0) ? l : l.cuda(devs[0]));

				if (devs[g] == this.hdev)
					hidden = n;
			}

			return {logits: torch.cat(ls, -1), hidden: hidden};
		},

		/* Greedy draft token from the MTP hidden D (1, 1, H): every rank
		 * scores its vocabulary range with its lm_head shard and reports
		 * (max, local argmax); the first rank holding the overall max
		 * wins. Ranges are in rank order and argmax takes the first
		 * maximum, so this equals the argmax over the full head — at
		 * 1/G of the head's weight reads (2.3 ms -> 0.7 ms per draft
		 * step on 6 A5000s). */
		draft_argmax: function (D) {
			let vals = [], idxs = [];

			for (let g = 0; g < G; g++) {
				let x = D.cuda(devs[g]);
				let l = this.lm_heads[g].forward(x).view([-1]);
				let am = l.argmax(-1);

				push(vals, l.max().view([1]).to('float32').cuda(devs[0]));
				push(idxs, am.view([1]).cuda(devs[0]));
			}

			let v = torch.cat(vals, 0).tolist(), ix = torch.cat(idxs, 0).tolist();
			let best = 0;

			for (let g = 1; g < G; g++)
				if (v[g] > v[best])
					best = g;

			return this.plan.vocab[best][0] + ix[best];
		},

		/* Zero the recurrent states and rewind (KV need no clearing). */
		reset: function (cache) {
			for (let i = 0; i < cfg.num_hidden_layers; i++)
				for (let g = 0; g < G; g++) {
					let cl = cache.layers[i][g];

					if (cl.rec_state) {
						cl.rec_state.fill_(0);
						cl.conv_state.fill_(0);
					}
				}

			cache.length = 0;
		},

		/* CUDA-graph steps of the given sizes (see qwen35_graph for the
		 * rules): ONE multi-GPU graph per size, captured on rank 0's
		 * device with the other ranks as side devices; the all-reduces
		 * need no cross-device edges. Clobbers the cache (eager warmup):
		 * call before any prefill, then reset(cache). Returns {has(S),
		 * run(ids, pos) -> {logits (1,S,V) on rank 0, hidden}}. */
		graphs: function (cache, sizes) {
			let tp = this;
			let by_size = {};

			for (let n = 0; n < length(sizes); n++) {
				let S = sizes[n];
				let st = {S: S, ids: torch.zeros([1, S], 'int64', 'cuda:' + devs[0]),
				           dev: [], out: null, bar: []};

				for (let g = 0; g < G; g++) {
					let dev = 'cuda:' + devs[g];
					let cl0 = null;

					for (let i = 0; i < cfg.num_hidden_layers && !cl0; i++)
						if (cache.layers[i][g].cos)
							cl0 = cache.layers[i][g];

					push(st.dev, {
						pos: torch.zeros([1], 'int64', dev),
						rpos: torch.zeros([1], 'int64', dev),   // pos + MRoPE offset
						ramp: torch.arange(S, 'int64', dev),
						cu_q: torch.from_array([0, S], 'int32', dev),
						cu_k: torch.from_array([0, cache.max_seq], 'int32', dev),
						cos_tab: cl0.cos, sin_tab: cl0.sin
					});

					// stream barriers rank 0 <-> rank g (a cross-device
					// copy orders both current streams): the graph
					// launches on rank 0's stream only, so eager work on
					// the other devices before/after a replay must be
					// ordered explicitly
					push(st.bar, torch.zeros([1], 'float32', dev));
				}

				let body = function () {
					// the other ranks take the position from rank 0
					// inside the graph: run() writes rank-0 inputs only
					for (let g = 1; g < G; g++) {
						st.dev[g].pos.copy_(st.dev[0].pos);
						st.dev[g].rpos.copy_(st.dev[0].rpos);
					}

					for (let g = 0; g < G; g++) {
						let d = st.dev[g];

						d.pos_idx = d.ramp.add(d.pos);
						d.seqused = d.pos_idx.narrow(0, S - 1, 1).add(1).to('int32');
						d.rope_idx = d.ramp.add(d.rpos);
						d.cos = d.cos_tab.index(d.rope_idx);
						d.sin = d.sin_tab.index(d.rope_idx);
					}

					let h0 = tp.embed_tokens.index(st.ids);
					let h = map(devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));

					for (let i = 0; i < cfg.num_hidden_layers; i++)
						h = tp.layer(i, h, cache, null, st.dev);

					st.out = tp.head(h, null);
				};

				st.dev[0].pos.fill_(0);
				st.dev[0].rpos.fill_(0);
				body();
				torch.synchronize();

				st.graph = torch.graph();
				st.graph.capture_begin(devs[0], slice(devs, 1));
				body();
				st.graph.capture_end();
				torch.synchronize();

				by_size[S] = st;
			}

			let barrier = function (st) {
				for (let g = 1; g < G; g++)
					st.bar[g].copy_(st.bar[0]);
			};

			return {
				has: function (S) {
					return by_size[S] != null;
				},

				run: function (ids, pos) {
					let S = length(ids);
					let st = by_size[S];

					if (!st)
						die(`qwen35_tp: no graph captured for step size ${S}`);

					st.ids.copy_(torch.from_array(ids, 'int64', 'cuda:' + devs[0]).view([1, S]));
					st.dev[0].pos.fill_(pos);
					st.dev[0].rpos.fill_(pos + qwen35.mm_off(cache.mm, pos));

					barrier(st);     // rank 0 waits for prior work on the others
					st.graph.replay();
					barrier(st);     // the others wait for the replay
					cache.length = pos + S;
					cache.hidden = st.out.hidden;

					return st.out;
				}
			};
		},

		/* Forward ids at cache.length (eager). Returns {logits, hidden}
		 * for the last `rows` positions (default: all); on_layer(li)
		 * (optional) runs after each layer. */
		forward: function (ids, cache, rows, on_layer, hidden_all) {
			let S = length(ids);
			let pos = cache.length;
			let h0 = qwen35.embed_rows(this, cache.mm, ids, pos);
			let h = map(devs, (d, g) => (g == 0) ? h0 : h0.cuda(d));

			for (let i = 0; i < cfg.num_hidden_layers; i++) {
				h = this.layer(i, h, cache, pos, null);

				if (on_layer)
					on_layer(i);
			}

			cache.length = pos + S;

			let r = this.head(h, rows, hidden_all);

			cache.hidden = r.hidden;

			return r;
		},

		/* engine interface (as qwen35's model.forward_chunk): logits of
		 * all rows, post-norm hidden in cache.hidden */
		forward_chunk: function (ids, cache) {
			return this.forward(ids, cache, null).logits;
		},

		/* engine interface (as qwen35.prefill_pipeline): chunked prefill
		 * of ids at cache.length, one chunk per step round. opts: chunk,
		 * cuts (extra chunk boundaries), on_layer(chunk_end, li), hidden
		 * (collect all rows' post-norm hidden, on hdev). All ranks work
		 * on every chunk, so there is no wavefront to pipeline. */
		prefill_pipeline: function (cache, ids, opts) {
			let tp = this;
			let chunk = opts.chunk || 1024;
			let start = cache.length, total = length(ids);
			let bounds = [];

			for (let p = start + chunk; p < start + total; p += chunk)
				push(bounds, p);

			for (let c in (opts.cuts || []))
				if (c > start && c < start + total && index(bounds, c) < 0)
					push(bounds, c);

			bounds = sort(bounds, (a, b) => a - b);
			push(bounds, start + total);

			let next = 0, done = 0, logits = null, parts = [];

			return {
				step: function (rounds) {
					for (let r = 0; r < (rounds || 1) && next < length(bounds); r++) {
						let end = bounds[next++];
						let a = cache.length;
						let last = next == length(bounds);
						let res = tp.forward(slice(ids, a - start, end - start), cache, 1,
							opts.on_layer ? (li) => opts.on_layer(end, li) : null,
							opts.hidden);

						if (opts.hidden)
							push(parts, res.hidden);

						if (last)
							logits = res.logits;

						done = end - start;
					}

					return next < length(bounds);
				},

				done_tokens: function () {
					return done;
				},

				logits: function () {
					return logits;
				},

				hidden: function () {
					if (!length(parts))
						return null;

					return (length(parts) == 1) ? parts[0] : torch.cat(parts, 1);
				}
			};
		}
	};

	return tp;
}
