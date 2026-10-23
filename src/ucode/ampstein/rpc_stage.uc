// ucode/rpc_stage.uc — a pipeline stage that lives on another host (docs/RPC.md, phase 4).
//
// The worker owns the FIRST stages of the pipeline (the embedding and the layers up to its
// share); the coordinator owns the rest and the head. A forward pass over n tokens is
//
//   coordinator --forward {cache, pos, n} + ids (int64)--> worker
//   worker: embed, its layers, the streams of its last stage
//   worker --h {n} + the residual streams (float32 [n, hc, hidden], one replica)--> coordinator
//   coordinator: copies the replica to every rank of its first stage and runs the rest
//
// The caches of the worker's layers live on the worker, one set per cache id; the coordinator
// only holds the id. The messages (meta.t):
//
//   new_cache {max_seq}      -> ok {cache}
//   free {cache}             -> ok
//   forward {cache, pos, n, graph?} + ids -> h {n} + streams    (graph: replay the captured decode graph)
//   graphs {cache, sizes}    -> ok            (capture the decode graphs of these step sizes; clobbers the
//                                              cache like glm5_tp.graphs: follow with reset)
//   reset {cache}            -> ok            (zero the recurrent states, length 0)
//   state_get {cache, layer}        -> state + bytes   (the recurrent state of one layer, all its ranks)
//   state_put {cache, layer} + bytes -> ok             (set it)
//   state_zero {cache, layer}       -> ok
// (the engine's snapshots of recurrent state, restores and resets reach the worker's layers this way)
//   stats {}                 -> ok {...}      (what the worker holds, for logs)
//   bye {}                   (the worker's serve loop returns 'bye')
//
// A failing command answers `error {message}` and the coordinator side raises it; the
// connection stays up. The model the worker passes is the partial glm5_tp model of its stages
// (own_layers, new_cache, embed, layer, handoff, stage_of); the unit tests pass a mock with the
// same few methods.

import * as rpc from './rpc.uc';

/* ---- worker ---------------------------------------------------------------------------------- */

/* worker(conn, model): { step(frame) -> null | 'bye', serve() -> 'bye' | (raises on a lost connection),
 *                        caches } */
export function worker(c, model) {
	let W = { caches: {}, graphs: {}, next: 1, forwards: 0, tokens: 0, replays: 0 };

	function ok(extra) {
		c.send({ t: 'ok', ...(extra ?? {}) });
	}

	function cache_of(meta) {
		let cache = W.caches[meta.cache];

		if (!cache)
			die(`no cache ${meta.cache}`);

		return cache;
	}

	let cmd = {
		new_cache: function (f) {
			let id = W.next++;

			W.caches[id] = model.new_cache(f.meta.max_seq);
			ok({ cache: id });
		},

		graphs: function (f) {
			let cache = cache_of(f.meta);

			W.graphs[f.meta.cache] = model.graphs(cache, f.meta.sizes);
			ok();
		},

		state_get: function (f) {
			c.send({ t: 'state', layer: f.meta.layer }, model.state_pack(cache_of(f.meta), f.meta.layer));
		},

		state_put: function (f) {
			if (!f.tensor)
				die('state_put: no bytes');

			model.state_unpack(cache_of(f.meta), f.meta.layer, f.tensor);
			ok();
		},

		state_zero: function (f) {
			model.state_zero(cache_of(f.meta), f.meta.layer);
			ok();
		},

		reset: function (f) {
			model.reset_cache(cache_of(f.meta));
			ok();
		},

		free: function (f) {
			cache_of(f.meta);
			delete W.caches[f.meta.cache];
			delete W.graphs[f.meta.cache];
			gc();
			ok();
		},

		forward: function (f) {
			let m = f.meta;
			let cache = cache_of(m);

			if (type(m.n) != 'int' || m.n < 1 || !f.tensor || f.tensor.numel() != m.n * 8)
				die('forward: expected n token ids');

			let ids = f.tensor.view_dtype('int64').tolist();       // the frame's buffer is reused: take the ids now

			if (m.graph) {
				let G = W.graphs[m.cache];

				if (!G || !G.has(m.n))
					die(`forward: no graph for ${m.n} token(s) on cache ${m.cache}`);

				let host = G.run(ids, m.pos).h.cpu().contiguous();
				let sh = host.shape();

				W.forwards++;
				W.replays++;
				W.tokens += m.n;
				c.send({ t: 'h', n: m.n, pos: m.pos, hc: sh[1], hidden: sh[2] }, host);

				return;
			}

			let h = model.embed(ids, null, m.pos);
			let cur = model.stage_of[model.own_layers[0]];

			for (let i = model.own_layers[0]; i < model.own_layers[1]; i++) {
				if (model.stage_of[i] != cur) {
					h = model.handoff(h, cur, model.stage_of[i]);
					cur = model.stage_of[i];
				}

				h = model.layer(i, h, cache, m.pos);
			}

			cache.length = m.pos + m.n;

			// every replica of the streams is identical after the stage's reductions: send one
			let host = h[0].cpu().contiguous();
			let sh = host.shape();

			W.forwards++;
			W.tokens += m.n;
			c.send({ t: 'h', n: m.n, pos: m.pos, hc: sh[1], hidden: sh[2] }, host);
		},

		stats: function (f) {
			ok({ caches: length(keys(W.caches)), forwards: W.forwards, tokens: W.tokens, replays: W.replays });
		}
	};

	W.step = function (f) {
		let t = f.meta.t;

		if (t == 'bye')
			return 'bye';

		let fn = cmd[t];

		try {
			if (!fn)
				die(`unknown command ${t}`);

			fn(f);
		}
		catch (e) {
			warn(`rpc_stage: ${t}: ${e}\n`);
			c.send({ t: 'error', message: `${t}: ${e}` });
		}

		return null;
	};

	W.serve = function () {
		for (;;) {
			let f = c.recv(-1);

			if (W.step(f) == 'bye')
				return 'bye';
		}
	};

	return W;
}

/* ---- coordinator ----------------------------------------------------------------------------- */

/* client(conn): the remote stages as seen from the coordinator. Every call that answers is
 * split in a send and a receive, so a pipelined prefill can have several in flight; the
 * plain forms do both. */
export function client(c) {
	let K = { conn: c, inflight: 0 };

	function answer(want, timeout_ms) {
		// (tests run the worker in this process: K.pump lets it take the request first)
		if (K.pump)
			K.pump();

		let f = c.recv(timeout_ms ?? 120000);

		if (!f)
			die(`rpc_stage: no ${want} within ${timeout_ms ?? 120000} ms`);

		if (f.meta.t == 'error')
			die(`rpc_stage: worker: ${f.meta.message}`);

		if (f.meta.t != want)
			die(`rpc_stage: expected ${want}, got ${f.meta.t}`);

		return f;
	}

	K.new_cache = function (max_seq) {
		c.send({ t: 'new_cache', max_seq: max_seq });

		return answer('ok').meta.cache;
	};

	K.free = function (cache) {
		c.send({ t: 'free', cache: cache });
		answer('ok');
	};

	K.stats = function () {
		c.send({ t: 'stats' });

		return answer('ok').meta;
	};

	/* forward_begin(cache, pos, ids): queue the request (ids: an array of token ids) */
	K.forward_begin = function (cache, pos, ids, graph) {
		c.send({ t: 'forward', cache: cache, pos: pos, n: length(ids), graph: graph ? true : false }, torch.from_array(ids, 'int64'));
		K.inflight++;
	};

	/* the worker captures its decode graphs for these step sizes (slow: seconds to a minute) */
	K.graphs = function (cache, sizes) {
		c.send({ t: 'graphs', cache: cache, sizes: sizes });
		answer('ok', 1800000);
	};

	/* the packed recurrent state of one worker layer: bytes in the connection's reusable buffer,
	 * valid until the next frame with a tensor — unpack at once */
	K.state_get = function (cache, layer) {
		c.send({ t: 'state_get', cache: cache, layer: layer });

		return answer('state').tensor;
	};

	K.state_put = function (cache, layer, bytes) {
		c.send({ t: 'state_put', cache: cache, layer: layer }, bytes);
		answer('ok');
	};

	K.state_zero = function (cache, layer) {
		c.send({ t: 'state_zero', cache: cache, layer: layer });
		answer('ok');
	};

	K.reset = function (cache) {
		c.send({ t: 'reset', cache: cache });
		answer('ok');
	};

	/* forward_end(timeout_ms): the oldest request's streams: {n, pos, h} with h a float32 host
	 * tensor [n, hc, hidden] in the connection's reusable buffer, valid until the next frame
	 * with a tensor arrives: copy it to the device at once */
	K.forward_end = function (timeout_ms) {
		let f = answer('h', timeout_ms);

		K.inflight--;

		return { n: f.meta.n, pos: f.meta.pos, h: f.tensor.view_dtype('float32').view([ f.meta.n, f.meta.hc, f.meta.hidden ]) };
	};

	K.forward = function (cache, pos, ids, graph) {
		K.forward_begin(cache, pos, ids, graph);

		return K.forward_end();
	};

	K.bye = function () {
		c.send({ t: 'bye' });
	};

	return K;
}
