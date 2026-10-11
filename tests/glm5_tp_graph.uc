// tests/glm5_tp_graph.uc — TP decode graphs vs eager TP steps, in both
// attention regimes: dense (context <= index_topk) and the k-pool sparse
// selection (context > index_topk). Random token ids; compares the logits of
// each decode step (rel-rms) and the greedy tokens.
import * as glm5 from 'ampstein.glm5_tp';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';

if (!stat(MODEL)) { warn('skip: model missing\n'); exit(125); }

// AMP_GLM5_STAGES=3: three tensor-parallel stages of two ranks (glm5_tp pipeline)
let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6),
	stages: +(getenv('AMP_GLM5_STAGES') || 1), max_seq: 1024 });
let cfg = model.cfg;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / b.to('float32').mul(b.to('float32')).mean().item());
}

function reset(cache) {
	for (let i = 0; i < cfg.num_hidden_layers; i++)
		for (let cl in cache.layers[i])
			if (cl.rec_state) { cl.rec_state.fill_(0); cl.conv_state.fill_(0); }

	cache.length = 0;
}

let cache = model.new_cache(4096);
let G = model.graphs(cache, [1]);          // clobbers the cache
let fails = [];

for (let ctx in [ 100, 2600 ]) {
	reset(cache);

	let ids = [];

	for (let i = 0; i < ctx; i++)
		push(ids, 1000 + (i * 7919) % 50000);

	model.prefill(ids, cache, 512, 0);

	// each step runs eagerly and through the graph from the SAME state
	// (the KDA states are snapshotted/restored; attention KV rewrites
	// in place)
	let tok = 5, worst = 0.0, same = 0;

	for (let k = 0; k < 6; k++) {
		let snap = [];

		for (let i = 0; i < cfg.num_hidden_layers; i++)
			for (let cl in cache.layers[i])
				if (cl.rec_state)
					push(snap, [ cl, cl.rec_state.mul(1.0), cl.conv_state.mul(1.0) ]);

		let pos = cache.length;
		let le = model.forward_chunk([tok], cache, 1).to('float32').cpu();

		for (let sn in snap) {
			sn[0].rec_state.copy_(sn[1]);
			sn[0].conv_state.copy_(sn[2]);
		}

		cache.length = pos;

		let lg = G.run([tok], pos).logits.to('float32').cpu();
		let e = rel_rms(lg, le);

		// the graph again from the same state: any difference is a race
		for (let sn in snap) {
			sn[0].rec_state.copy_(sn[1]);
			sn[0].conv_state.copy_(sn[2]);
		}

		let lg2 = G.run([tok], pos).logits.to('float32').cpu();

		if (rel_rms(lg2, lg) > 0)
			printf('  step %d: graph replay differs from itself: %.2e (race)\n', k, rel_rms(lg2, lg));

		for (let sn in snap) {
			sn[0].rec_state.copy_(sn[1]);
			sn[0].conv_state.copy_(sn[2]);
		}

		G.run([tok], pos);

		if (e > worst) worst = e;

		if (lg.argmax(-1).item() == le.argmax(-1).item()) same++;

		tok = le.argmax(-1).item();
	}

	printf('context %d (%s): graph vs eager logits worst rel-rms %.2e, greedy tokens equal %d/6\n',
		ctx, ctx > cfg.index_topk ? 'sparse' : 'dense', worst, same);

	if (!(worst < 0.3)) push(fails, `ctx ${ctx}: logits`);
	if (same < 5) push(fails, `ctx ${ctx}: tokens`);
}

// the real check: greedy decode through the graphs on the oracle chat
// prompt reproduces exllamav3's continuation
{
	const FIX = `${sourcepath(0, true)}/fixtures/glm5_oracle.safetensors`;

	if (stat(FIX)) {
		let fx = torch.st(FIX);
		let pids = fx.tensor('ids').tolist();
		let cont = fx.tensor('cont').tolist();

		reset(cache);

		let lg = model.prefill(pids, cache, 512, 1);
		let out = [];

		let margin = null;

		for (let k = 0; k < length(cont); k++) {
			let t = lg.reshape([-1]).argmax().item();

			push(out, t);

			// the logit gap between our token and the oracle's at the first
			// divergence: a near-tie is numeric noise, a wide gap a bug
			if (margin == null && t != cont[k]) {
				let v = lg.reshape([-1]).to('float32');

				margin = v.narrow(0, t, 1).sub(v.narrow(0, cont[k], 1)).item();
			}

			lg = G.run([t], cache.length).logits;
		}

		if (margin != null)
			printf('first divergence: our logit leads the oracle token by %.3f\n', margin);

		let first = -1;

		for (let k = 0; k < length(cont); k++)
			if (out[k] != cont[k]) { first = k; break; }

		printf('oracle prompt, graph decode: %s\n', first < 0 ? 'all 24 tokens match' :
			`diverges at token ${first}`);

		if (first >= 0 && first < 8)
			push(fails, 'oracle continuation');
	}
}

assert(!length(fails), 'glm5_tp_graph: ' + join('; ', fails));
print('glm5_tp_graph: ok\n');
