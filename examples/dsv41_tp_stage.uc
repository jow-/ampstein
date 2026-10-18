// examples/dsv41_tp_stage.uc — time of ONE prefill chunk through each TP stage on its own (synchronized), against the
// wavefront cycle: is the pipeline limited by the stages or by how they overlap?
//   ucode ... examples/dsv41_tp_stage.uc <model_dir> [chunk] [warm chunks]
import * as dsv41_tp from 'ampstein.dsv41_tp';
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], chunk = +(ARGV[1] || 1024), warm = +(ARGV[2] || 6);
let m = dsv41_tp.load(dir, {devices: 6, stages: 3, max_seq: 32768, prefill_rows: chunk});
let cache = m.new_cache(32768);
let ids = [], x = 7;

for (let i = 0; i < chunk * (warm + 3); i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function now() {
	let c = clock(true);

	return c[0] + c[1] / 1e9;
}

// fill some context first (sequentially)
for (let k = 0; k < warm; k++)
	m.forward_chunk(slice(ids, k * chunk, (k + 1) * chunk), cache, 0);

torch.synchronize();

for (let rep = 0; rep < 3; rep++) {
	let c = m.chunk_begin(slice(ids, (warm + rep) * chunk, (warm + rep + 1) * chunk), cache);
	let line = '';

	for (let s = 0; s < m.NS; s++) {
		torch.synchronize();

		let t0 = now();

		m.chunk_stage(c, cache, s);

		let t1 = now();

		torch.synchronize();
		line += sprintf('  stage %d: issue %.0f ms, done %.0f ms', s, (t1 - t0) * 1000, (now() - t0) * 1000);
	}

	printf('chunk %d:%s\n', warm + rep, line);
}

if (getenv('AMP_DSV41_HPROF'))
	for (let k in sort(keys(dsv41.PROFILE)))
		printf('  hprof %-12s %.2f s\n', k, dsv41.PROFILE[k] / 1000);

if (m.hostprof)
	for (let k in sort(keys(m.hostprof)))
		printf('  host %-10s %.2f s\n', k, m.hostprof[k]);
