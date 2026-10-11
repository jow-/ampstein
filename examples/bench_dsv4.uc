// examples/bench_dsv4.uc — DeepSeek-V4 prefill / decode throughput.
// Env: DSV4_LAYERS (0 = all), DSV4_DEVICES, BENCH_N (prompt tokens, default 4096),
// BENCH_CHUNK (default 1024), BENCH_DEC (decode steps, default 32), AMP_MODEL_DSV4.

let torch = require('ampstein.torch');
global.torch = torch;
import * as dsv4 from 'ampstein.dsv4';
import * as dsv4_tp from 'ampstein.dsv4_tp';
import * as glm5 from 'ampstein.glm5';

const MODEL = getenv('AMP_MODEL_DSV4') || '/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw';
const NL = +(getenv('AMP_DSV4_LAYERS') || 0);
const NDEV = +(getenv('AMP_DSV4_DEVICES') || 6);
const N = +(getenv('AMP_BENCH_N') || 4096);
const CH = +(getenv('AMP_BENCH_CHUNK') || 1024);
const DEC = +(getenv('AMP_BENCH_DEC') || 32);

function secs(c0) {
	let c1 = clock(true);

	return (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1e9;
}

const TP = +(getenv('AMP_DSV4_TP') || 0);         // pipeline stages of TP-2 pairs (0: layer split)
let opts = NL ? { layers: NL, devices: NDEV } : { devices: NDEV };
let model = TP ? dsv4_tp.load(MODEL, { ...opts, stages: TP, prefill_rows: CH }) : dsv4.load(MODEL, { ...opts, prefill_rows: CH });
let ids = [ 0 ];
let x = 12345;

for (let i = 1; i < N; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

for (let rep = 0; rep < 2; rep++) {
	let cache = model.new_cache(N + 3 * DEC + 16);
	let t0 = clock(true);
	let run = model.prefill_pipeline(cache, ids, { chunk: CH });

	while (run.step(1)) {}

	let lg = run.logits();

	torch.synchronize();

	let dt = secs(t0);

	printf('prefill %d tokens, chunk %d: %.2f s = %.0f tok/s\n', N, CH, dt, N / dt);

	if (dsv4.HP) {
		printf('host: phase A %.2f s, phase B (incl. count readback waits) %.2f s, of %.2f s\n', dsv4.HP.a, dsv4.HP.b, dt);
		printf('moe_finish: count readback wait %.2f s over %d calls, host work after it %.2f s, %d heavy experts\n', glm5.HPG.wait, glm5.HPG.n, glm5.HPG.after, glm5.HPG.heavy);
		printf('  of phase A: attention (incl. qkv/compressor/indexer/o-proj) %.2f s\n', dsv4.HP.attn);
		printf('  attention sections: q/kv proj %.2f  compressor %.2f  core (mask/matmul/softmax/indexer) %.2f  o-proj %.2f\n', dsv4.HP.s_proj, dsv4.HP.s_comp, dsv4.HP.s_core, dsv4.HP.s_oproj);
		dsv4.HP.s_proj = dsv4.HP.s_comp = dsv4.HP.s_core = dsv4.HP.s_oproj = 0;
		dsv4.HP.attn = 0;
		dsv4.HP.a = 0;
		dsv4.HP.b = 0;
		glm5.HPG.wait = glm5.HPG.after = glm5.HPG.n = glm5.HPG.heavy = 0;
	}

	if (rep == 1) {
		let tok = lg.view([-1]).argmax(-1).item();
		let t1 = clock(true);

		for (let i = 0; i < (getenv('AMP_BENCH_NOEAGER') ? 0 : DEC); i++)
			tok = model.forward_chunk([ tok ], cache).view([-1]).argmax(-1).item();

		let dd = secs(t1);

		printf('decode (eager) %d tokens: %.2f s = %.1f tok/s (%.1f ms/token)\n', DEC, dd, DEC / dd, dd * 1000 / DEC);

		let VS = +(getenv('AMP_BENCH_VERIFY') || 0);
		let graph = model.graphs(cache, VS ? [ 1, VS ] : [ 1 ]);

		torch.synchronize();

		for (let r2 = 0; r2 < 2; r2++) {
			let t2 = clock(true);

			for (let i = 0; i < DEC; i++)
				tok = graph.run([ tok ], cache.length).logits.view([-1]).argmax(-1).item();

			let d2 = secs(t2);

			printf('decode (graphs) %d tokens: %.2f s = %.1f tok/s (%.1f ms/token)\n', DEC, d2, DEC / d2, d2 * 1000 / DEC);
		}

		if (VS) {
			let vids = [];

			for (let i = 0; i < VS; i++)
				push(vids, 1000 + i);

			for (let r2 = 0; r2 < 2; r2++) {
				let t3 = clock(true);

				for (let i = 0; i < DEC; i++) {
					graph.run(vids, cache.length);
					cache.length -= VS - 1;              // position walk only; content is irrelevant here
					graph.run([ tok ], cache.length);
					torch.synchronize();
				}

				let d3 = secs(t3);

				printf('verify S=%d + step: %.1f ms per pair\n', VS, d3 * 1000 / DEC);
			}
		}
	}
}
