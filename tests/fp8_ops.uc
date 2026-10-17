// tests/fp8_ops.uc — block-scaled FP8 weights (DeepSeek-V4.1): torch.fp8_dequant and
// torch.fp8_gemv against an fp32 composite reference built from the same bytes.

import * as math from 'math';

const DEV = +(getenv('AMP_TEST_DEV') || 0);
const dv = 'cuda:' + DEV;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

function check(N, K, M) {

	// random weights, quantized by the dtype conversion; scales are powers of two (e8m0)
	let w = torch.rand([N, K], 'float32', dv).sub(0.5).mul(12.0).to('float8_e4m3fn').contiguous();
	let sv = [];

	for (let i = 0; i < (N / 32) * (K / 32); i++)
		push(sv, math.pow(2.0, (i * 7) % 8 - 8));

	let s = torch.from_array(sv, 'float32', dv).view([int(N / 32), int(K / 32)]).to('float8_e8m0fnu').contiguous();

	let ref = w.to('float32').view([int(N / 32), 32, int(K / 32), 32])
		.mul(s.to('float32').view([int(N / 32), 1, int(K / 32), 1])).view([N, K]);

	let wd = torch.fp8_dequant(w, s);

	assert(rel_rms(wd, ref) < 1e-3, `dequant ${N}x${K}`);

	let x = torch.rand([M, K], 'float16', dv).sub(0.5).mul(4.0);
	let y = torch.fp8_gemv(x, w, s);
	let yr = torch.matmul(x.to('float32'), ref.t());

	let r = rel_rms(y, yr);

	assert(r < 2e-3, `gemv ${N}x${K} M=${M}: rel rms ${r}`);
	printf('ok %dx%d M=%d (rel %.2e)\n', N, K, M, r);
}

check(1280, 5120, 1);
check(32768, 1280, 1);
check(5120, 8192, 3);
check(2304, 5120, 8);
check(5120, 2304, 2);
printf('fp8_ops: all ok\n');
