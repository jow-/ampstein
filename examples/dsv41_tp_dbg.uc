// compare last-row logits of forward_chunk(n tokens) between the single-chain and the TP model for several n
import * as dsv41 from 'ampstein.dsv41';
import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as math from 'math';

let dir = ARGV[0], NL = +(ARGV[1] || 8);
let ns = [ 1, 4, 8, 9, 16, 40, 128 ];
let ids = [], x = 777;

for (let i = 0; i < 200; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return math.sqrt(d.mul(d).mean().item() / max(b.to('float32').mul(b.to('float32')).mean().item(), 1e-30));
}

function run(m) {
	let out = [];

	for (let n in ns) {
		let c = m.new_cache(256);

		push(out, m.forward_chunk(slice(ids, 0, n), c, 1).to('float32').cpu());
	}

	return out;
}

let a = run(dsv41.load(dir, {devices: 2, layers: NL, max_seq: 256, prefill_rows: 128}));

gc();

let b = run(dsv41_tp.load(dir, {devices: 4, stages: 2, layers: NL, max_seq: 256, prefill_rows: 128}));

for (let i = 0; i < length(ns); i++)
	printf('n=%d: rel rms %.2e\n', ns[i], rel_rms(b[i], a[i]));
