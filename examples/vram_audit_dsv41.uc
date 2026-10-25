// examples/vram_audit_dsv41.uc — where the VRAM of a DeepSeek-V4.1 model goes, per GPU and phase: tensors (the allocator's allocated bytes),
// the allocator's cache (reserved - allocated, what torch.empty_cache() returns), non-PyTorch memory (the CUDA context, cuBLAS / kernel
// modules: total - free - reserved) and the driver's free memory, after: load, new_cache, the decode graphs, a prefill, torch.empty_cache().
//   ucode -L <build>/lib -L src/ucode -ltorch=ampstein.torch examples/vram_audit_dsv41.uc MODEL_DIR [ctx=262144] [gpus=6] [drafter=0] [prefill=4096]

import * as dsv41 from 'ampstein.dsv41';
import * as dsv41_spark from 'ampstein.dsv41_spark';

let dir = ARGV[0], CTX = +(ARGV[1] || 262144), NG = +(ARGV[2] || 6), SPARK = (ARGV[3] == '1'), NP = +(ARGV[4] || 4096);
const MIB = 1048576.0;
let prev = null;

function report(tag) {
	let line = [], tot_a = 0, tot_c = 0, tot_n = 0, tot_f = 0;

	for (let d = 0; d < NG; d++) {
		let m = torch.mem_info(d), alloc = m[0], res = m[1], free = m[2], total = m[3];
		let non = total - free - res;

		push(line, sprintf('cuda:%d alloc %6.0f cache %4.0f non-torch %4.0f free %5.0f', d, alloc / MIB, (res - alloc) / MIB, non / MIB, free / MIB));
		tot_a += alloc; tot_c += res - alloc; tot_n += non; tot_f += free;
	}

	printf('--- %s\n  %s\n  total GiB: tensors %.2f, allocator cache %.2f, non-torch %.2f, free %.2f\n', tag, join('\n  ', line), tot_a / MIB / 1024,
		tot_c / MIB / 1024, tot_n / MIB / 1024, tot_f / MIB / 1024);
}

report('before load');

let m = dsv41.load(dir, {devices: NG, max_seq: CTX, prefill_rows: 512, spark: SPARK});

report('after load (weights, embedding, engram buffers)');

if (SPARK && m.spark_devs) {
	m.dspark = dsv41_spark.load(m, {devs: m.spark_devs});
	report(`after the drafter (stages on ${join(',', m.spark_devs)})`);
}

let cache = m.new_cache(CTX);

report(`after new_cache(${CTX})`);

let g = m.graphs(cache, SPARK ? [ 1, 2, 3, 4, 5 ] : [ 1 ]);            // (the engine captures every verify size 1 .. window + 1)

report('after the decode graphs');
torch.empty_cache();
report('after the decode graphs + empty_cache (what is left is the graphs\' private pools and their tensors)');
m.aux_reset(cache);
cache.length = 0;

let ids = [], x = 4321;

for (let i = 0; i < NP; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let run = m.prefill_pipeline(cache, ids, {chunk: 512});

while (run.step(1)) {}

torch.synchronize();
report(`after a ${NP}-token prefill (chunks of 512), before any collection`);
gc('collect');
report('after a ucode garbage collection (gc)');
torch.empty_cache();
report('after torch.empty_cache()');
