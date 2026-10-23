// ucode/rpc_coord.uc — the coordinator's side of a GLM pipeline split over hosts (docs/RPC.md).
//
// attach({listen, model, gpus, stages, max_seq, kv_bits, prefill_rows, workers, wait}) listens for
// the worker(s) to dial in, hands each the first stages of the pipeline (as many as its GPUs make
// up, `gpus / stages` GPUs per stage like here), waits until it has loaded them, and returns the
// extra options for glm5_tp.load of the remaining stages on this host:
//
//   let ex = rpc_coord.attach({listen: '192.168.41.2:7700', model: dir, gpus: 6, stages: 3, ...});
//   let m = glm5_tp.load(dir, {devices: 6, ...ex.load});     // stages / own / stage_end / remote
//
// `gpus` and `stages` are this host's; the totals are larger by the worker's stages. Plain TCP, no
// TLS: bind to the address of a VPN / overlay or a private link.

import * as rpc from './rpc.uc';
import * as stage from './rpc_stage.uc';
import * as glm5_tp from './glm5_tp.uc';

function now_ms() {
	let t = clock(true);

	return t[0] * 1000 + int(t[1] / 1000000);
}

export function attach(o) {
	let m = match(o.listen ?? '', /^(.*):([0-9]+)$/);

	if (!m)
		die('rpc: --rpc-listen expects ADDR:PORT');

	if ((o.workers ?? 1) != 1)
		die('rpc: one worker is supported so far');

	if (o.gpus % o.stages)
		die(`rpc: ${o.gpus} GPUs do not split into ${o.stages} stages`);

	let Gs = o.gpus / o.stages;
	let assign = null;
	let plan_opts = { max_seq: o.max_seq, kv_bits: o.kv_bits, prefill_rows: o.prefill_rows, mtp: false, plan_only: true };

	let L = rpc.listen({
		bind: m[1], port: int(m[2]), manifest: rpc.manifest(o.model),

		// the assignment: the worker takes the first stages its GPUs make up; the plan of the whole
		// pipeline is made here and handed over, so that both sides cut the layers the same way
		welcome: function (hello) {
			let n = length(hello.gpus ?? []);

			if (n < Gs || n % Gs)
				die(`the worker has ${n} GPUs, stages are ${Gs} GPUs wide`);

			let ws = n / Gs, total = o.stages + ws;
			let devs = [];

			for (let i = 0; i < total * Gs; i++)
				push(devs, i);

			let plan = glm5_tp.load(o.model, { ...plan_opts, devices: devs, stages: total });

			warn(`rpc: ${ws} stage(s) for ${hello.name}, ${total} in total, stages start at layers ${join(' ', plan.stage_end)}\n`);

			assign = { stages: total, own: [ 0, ws ], stage_end: plan.stage_end, max_seq: o.max_seq,
				kv_bits: o.kv_bits, prefill_rows: o.prefill_rows };

			return assign;
		}
	});

	warn(`rpc: waiting for a worker on ${m[1]}:${L.port}\n`);

	let w = L.wait(1, o.wait ?? 3600)[0];
	let ready = w.conn.recv(3600000);

	if (!ready || ready.meta.t != 'ready')
		die('rpc: the worker did not report ready');

	warn(`rpc: worker ${w.name} ready with layers ${ready.meta.layers[0]}..${ready.meta.layers[1] - 1}\n`);

	let K = stage.client(w.conn);

	// the worker dropped (its host, the process or the link): wait until a worker dials in again — the
	// same assignment is sent in the welcome — and has loaded its stages; returns a client on the new
	// connection. The old worker's caches are gone with its connection.
	function recover(old) {
		try {
			old.conn.close();
		}
		catch (e) {
		}

		let wait = o.rejoin_wait ?? 300;
		let deadline = now_ms() + wait * 1000;

		for (;;) {
			let left = deadline - now_ms();

			if (left <= 0)
				die(`rpc: the worker did not come back within ${wait} s`);

			let w2 = L.accept_one(left > 1000 ? 1000 : left);

			if (!w2)
				continue;

			// (a connection from a worker that has died since — it lay in the backlog — fails here: keep waiting)
			let ready = null;

			try {
				ready = w2.conn.recv(deadline - now_ms() > 0 ? deadline - now_ms() : 1000);
			}
			catch (e) {
				warn(`rpc: a worker connection failed before it was ready (${e}); waiting for another\n`);
			}

			if (!ready || ready.meta.t != 'ready') {
				w2.conn.close();
				continue;
			}

			warn(`rpc: worker ${w2.name} is back with layers ${ready.meta.layers[0]}..${ready.meta.layers[1] - 1}\n`);

			return stage.client(w2.conn);
		}
	}

	return {
		listener: L, worker: w, client: K, assign: assign, recover: recover,
		load: { stages: assign.stages, own: [ assign.own[1], assign.stages ], stage_end: assign.stage_end, remote: K,
			recover: recover }
	};
}
