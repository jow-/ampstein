// examples/vram_prefill_trace.uc — per-GPU memory while a long prompt prefills (every `every` tokens): allocated tensors, the allocator's cache and the
// driver's free memory, to see what grows with the context.
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/vram_prefill_trace.uc MODEL_DIR [ctx=262144] [gpus=6] [drafter=1] [tokens=60000] [every=4096] [chunk=512]

import * as dsv41 from 'ampstein.dsv41';
import * as dsv41_spark from 'ampstein.dsv41_spark';

let dir = ARGV[0], CTX = +(ARGV[1] || 262144), NG = +(ARGV[2] || 6), SPARK = (ARGV[3] != '0'), NP = +(ARGV[4] || 60000);
let EVERY = +(ARGV[5] || 4096), CH = +(ARGV[6] || 512);
const MIB = 1048576.0;

let m = dsv41.load(dir, {devices: NG, max_seq: CTX, prefill_rows: CH, spark: SPARK});

if (SPARK && m.spark_devs)
	m.dspark = dsv41_spark.load(m, {devs: m.spark_devs});

let cache = m.new_cache(CTX);
let g = m.graphs(cache, SPARK ? [ 1, 2, 3, 4, 5 ] : [ 1 ]);            // (the engine captures every verify size 1 .. window + 1)

m.aux_reset(cache);
cache.length = 0;

let ids = [], x = 4321;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

function sample(tag) {
	let a = [], f = [], c = [];

	for (let d = 0; d < NG; d++) {
		let mi = torch.mem_info(d);

		push(a, sprintf('%5.0f', mi[0] / MIB));
		push(c, sprintf('%4.0f', (mi[1] - mi[0]) / MIB));
		push(f, sprintf('%5.0f', mi[2] / MIB));
	}

	printf('%-14s alloc %s | cache %s | free %s\n', tag, join(' ', a), join(' ', c), join(' ', f));
}

sample('start');

let last = 0;
let run = m.prefill_pipeline(cache, ids, {chunk: CH, on_aux: function (end) {
	if (end - last >= EVERY) {
		last = end;
		sample(`@${end}`);
	}
}});

while (run.step(1)) {}

torch.synchronize();
sample('done');
