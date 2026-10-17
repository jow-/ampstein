// tests/moe_mixed.uc — the fused decode MoE (torch.moe_coop) with a different bit width per expert
// matrix (DeepSeek-V4.1 SAGE packs mix K = 1..6 inside a layer) against the composite path of
// per-expert EXL3 gemms. The fixture is 24 real experts of layer 3, see tests/oracle/README (range
// fetch from the pack): AMP_MOE_FIX=<dir>/moe_l3_e24.safetensors.

import * as math from 'math';
import * as weights from 'ampstein.weights';
import * as models from 'ampstein.models';
import * as glm5 from 'ampstein.glm5';
import * as dsv41 from 'ampstein.dsv41';
import { stat } from 'fs';

const FIX = getenv('AMP_MOE_FIX') || '/data/models/moe_l3_e24.safetensors';

if (!stat(FIX)) {
	warn('skip: fixture missing: ' + FIX + '\n');
	exit(125);
}

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;
const NE = 24, H = 5120, TOPK = 6;
const cfg = {hidden_size: H, num_experts_per_tok: TOPK, swiglu_limit: 10.0};

let i = rindex(FIX, '/');
let ws = weights.open(substr(FIX, 0, i), substr(FIX, i + 1));
let experts = [];

for (let e = 0; e < NE; e++) {
	let k = `layers.3.ffn.experts.${e}`;

	push(experts, {
		gate: models.linear(ws, k + '.w1', false, DEV),
		up: models.linear(ws, k + '.w3', false, DEV),
		down: models.linear(ws, k + '.w2', false, DEV)
	});
}

let ks = {};

for (let e in experts)
	for (let m in [ 'gate', 'up', 'down' ])
		ks[e[m].K] = true;

printf('bit widths present: %s\n', join(' ', sort(keys(ks))));

let M = {dev: DEV, experts: experts};
let coop = dsv41.build_coop(cfg, M);

assert(coop, 'build_coop');
M.coop = coop;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

function run(T, seed) {
	let xs = [], idl = [], wl = [];

	for (let t = 0; t < T; t++) {
		let used = {};

		for (let k = 0; k < TOPK; k++) {
			let e = (seed * 7 + t * 5 + k * 3 + (k * k)) % NE;

			while (used[e])
				e = (e + 1) % NE;

			used[e] = true;
			push(idl, e);
			push(wl, 0.3 + 0.1 * ((t + k) % 5));
		}
	}

	let y = torch.rand([T, H], 'float16', dv).sub(0.5).mul(2.0);
	let sel = torch.from_array(idl, 'int64', dv).view([T, TOPK]);
	let rw = torch.from_array(wl, 'float32', dv).view([T, TOPK]);

	torch.moe_coop(y.contiguous(), sel, rw.to('float16').contiguous(), coop, null);

	let got = coop.out.narrow(0, 0, T).add(0.0);

	// reference: every (token, expert) through its own gate / up / down
	let ref = torch.zeros([T, H], 'float32', dv);

	for (let t = 0; t < T; t++) {
		let yt = y.narrow(0, t, 1);

		for (let k = 0; k < TOPK; k++) {
			let ex = experts[idl[t * TOPK + k]];
			let d = ex.down.forward(glm5.act_mul(cfg, ex.gate.forward(yt), ex.up.forward(yt))).to('float32');

			ref.narrow(0, t, 1).add_(d.mul(wl[t * TOPK + k]));
		}
	}

	let e = rel_rms(got, ref);

	assert(e < 0.02, `T=${T}: rel rms ${e}`);
	printf('ok T=%d (rel %.2e)\n', T, e);
}

// the whole MoE of a block through glm5.moe_forward (fused decode kernel, token-sorted prefill kernel)
// against the composite path, with a random router over the 24 experts
function full(T) {
	let Mx = {
		type: 'moe', dev: DEV, experts: experts, router: 'sqrtsp', async_moe: true, vlon: false, shared: null,
		gate_t: torch.rand([H, NE], 'float32', dv).sub(0.5).mul(0.05),
		bias: torch.rand([1, NE], 'float32', dv).mul(0.1),
		coop: coop
	};
	let c2 = {hidden_size: H, num_experts_per_tok: TOPK, swiglu_limit: 10.0, n_routed_experts: NE,
		routed_scaling_factor: 1.5};
	let y = torch.rand([T, H], 'float16', dv).sub(0.5).mul(2.0);
	let got = glm5.moe_forward(c2, Mx, y);
	let ref = glm5.moe_forward_ref(c2, Mx, y);
	let e = rel_rms(got, ref);

	assert(e < 0.03, `full T=${T}: rel rms ${e}`);
	printf('ok full T=%d (rel %.2e)\n', T, e);
}

run(1, 1);
run(1, 5);
run(3, 2);
run(8, 3);
full(1);
full(5);
full(20);
full(100);
full(300);
printf('moe_mixed: all ok\n');
