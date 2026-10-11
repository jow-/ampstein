// tests/skinny.uc — torch.skinny (fp32 skinny product for 1..8 rows) vs
// torch.matmul on the shapes the GLM decode uses, plus bit-determinism of
// repeated calls and use inside a CUDA graph.
import * as math from 'math';

if (torch.device_count() < 1) { warn('skip: no GPU\n'); exit(125); }

function rel_rms(a, b) {
	let d = a.sub(b);

	return math.sqrt(d.mul(d).mean().item() / b.mul(b).mean().item());
}

let worst = 0.0;

for (let sh in [ [4096, 128], [128, 8192], [4096, 12], [4096, 10], [4096, 288], [128, 1536], [8192, 100], [512, 4] ]) {
	for (let M = 1; M <= 8; M++) {
		let x = torch.arange(M * sh[0], 'float32', 'cuda:0').mul(0.37).sin().view([M, sh[0]]);
		let w = torch.arange(sh[0] * sh[1], 'float32', 'cuda:0').mul(0.011).cos()
			.view([sh[0], sh[1]]).mul(0.05);
		let ref = torch.matmul(x, w);
		let got = torch.skinny(x, w);
		let e = rel_rms(got, ref);

		if (e > worst) worst = e;

		assert(e < 1e-4, `skinny ${M}x${sh[0]}x${sh[1]}: rel-rms ${e}`);

		let again = torch.skinny(x, w);

		assert(got.sub(again).abs().max().item() == 0, `skinny ${M}x${sh[0]}x${sh[1]}: not deterministic`);
	}
}

printf('skinny vs matmul: worst rel-rms %.2e over 64 shapes\n', worst);

// inside a graph, replayed
let x = torch.ones([3, 4096], 'float32', 'cuda:0');
let w = torch.ones([4096, 288], 'float32', 'cuda:0').mul(0.5);
let keep = null;

torch.skinny(x, w);
torch.synchronize();

let g = torch.graph();

g.capture_begin(0);
keep = torch.skinny(x, w);
g.capture_end();

for (let i = 0; i < 5; i++) {
	g.replay();
	torch.synchronize();
	assert(keep.sub(torch.matmul(x, w)).abs().max().item() == 0, 'graph replay value');
}

print('skinny: ok\n');
