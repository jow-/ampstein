// examples/weight_sizes.uc — checkpoint bytes by tensor family (key with
// the layer index replaced), largest first. Used to review what a tensor-
// parallel layout shards and what it replicates.
import * as weights from 'ampstein.weights';

let dir = getenv('AMP_MODEL');

if (!dir) {
	warn('usage: AMP_MODEL=<model dir> ucode ... examples/weight_sizes.uc\n');
	exit(1);
}

let ws = weights.open(dir);
let fam = {};
let total = 0;

for (let key in ws.list()) {
	let k = replace(key, /\.layers\.[0-9]+\./, '.layers.N.');

	k = replace(k, /\.experts\.[0-9]+\./, '.experts.E.');
	k = replace(k, /\.(trellis|suh|svh|mcg|mul1|bias|weight)$/, '');

	let n = ws.info(key).nbytes;

	fam[k] = (fam[k] || 0) + n;
	total += n;
}

let rows = [];

for (let k in fam)
	push(rows, [ k, fam[k] ]);

rows = sort(rows, (a, b) => b[1] - a[1]);

printf('total %.2f GiB\n', total / 1073741824.0);

for (let i = 0; i < length(rows) && i < 40; i++)
	printf('%7.3f GiB  %5.1f%%  %s\n', rows[i][1] / 1073741824.0, 100.0 * rows[i][1] / total, rows[i][0]);
