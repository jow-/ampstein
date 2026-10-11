// tests/kv_q8.uc — Q8 latent-cache kernels (torch.kv_q8_put / kv_q8_get):
// quantization error against the group-absmax bound, range / index put and
// get agree, zero rows survive, out-of-range / wrong-type arguments raise.
import * as math from 'math';

const DEV = getenv('AMP_TEST_DEV') || 'cuda:0';
const DIM = 512, CAP = 300;

function raises(f) {
	try { f(); } catch (e) { return true; }

	return false;
}

let q = torch.zeros([CAP, DIM], 'int8', DEV);
let s = torch.zeros([CAP, DIM / 32], 'float16', DEV);

// values with a heavy-tailed group structure (a few outlier channels)
function noise(rows, lo, hi) {
	let a = [];

	for (let r = 0; r < rows; r++) {
		let row = [];

		for (let c = 0; c < DIM; c++)
			push(row, math.rand(lo, hi));

		push(a, row);
	}

	return torch.from_array(a, 'float32').to('float16').cuda(+substr(DEV, 5));
}

let x = noise(40, -1.0, 1.0);
let spike = torch.zeros([40, DIM], 'float16', DEV);

spike.narrow(1, 7, 1).fill_(6.0);
spike.narrow(1, 300, 1).fill_(-9.0);
x = x.add(spike);

// 1. put at a host start, get the range back
torch.kv_q8_put(x, q, s, 11);

let y = torch.kv_q8_get(q, s, 11, 40);

assert(y.shape()[0] == 40 && y.shape()[1] == DIM, 'range get shape');

let err = y.to('float32').sub(x.to('float32'));
let rel = math.sqrt(err.mul(err).mean().item() / x.to('float32').pow(2).mean().item());

printf('range round trip: rel-rms %.5f\n', rel);
assert(rel < 0.01, 'Q8 relative rms error stays below 1%');

// no value is further off than half a code step of the LARGEST group scale
// (absmax 9.x / 127 / 2), plus fp16 rounding slack
let amax = x.to('float32').abs().max().item();

assert(err.abs().max().item() <= amax / 254.0 * 1.05 + 0.002, 'error within half a code step');

// 2. put through an index tensor (the graph path), get through an index
let idx = torch.from_array([ 200, 3, 77, 150 ], 'int64', DEV);
let z = noise(4, 0.0, 1.0);

torch.kv_q8_put(z, q, s, idx);

let w = torch.kv_q8_get(q, s, idx);
let ez = w.to('float32').sub(z.to('float32')).abs().max().item();

assert(ez < 0.01, 'index put/get round trip');

// rows written earlier survive the scattered writes
let y2 = torch.kv_q8_get(q, s, 11, 40);

assert(y2.to('float32').sub(y.to('float32')).abs().max().item() == 0.0, 'untouched rows unchanged');

// 3. all-zero rows
torch.kv_q8_put(torch.zeros([2, DIM], 'float16', DEV), q, s, 250);

assert(torch.kv_q8_get(q, s, 250, 2).to('float32').abs().max().item() == 0.0, 'zero rows stay zero');

// 4. rejections
assert(raises(() => torch.kv_q8_put(x, q, s, CAP - 10)), 'put past the capacity raises');
assert(raises(() => torch.kv_q8_get(q, s, CAP - 1, 5)), 'get past the capacity raises');
assert(raises(() => torch.kv_q8_put(x, q, s, 1.5)), 'non-integer position raises');
assert(raises(() => torch.kv_q8_put(x.narrow(1, 0, 256), q, s, 0)), 'dim mismatch raises');
assert(raises(() => torch.kv_q8_get(q, s.to('float32'), 0, 1)), 'wrong scale dtype raises');
assert(raises(() => torch.kv_q8_put(x, q, s, torch.from_array([ 1, 2 ], 'int64', DEV))),
	'index count mismatch raises');

print('kv_q8: ok\n');
