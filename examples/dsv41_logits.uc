// examples/dsv41_logits.uc — logits of a fixed prompt for a given placement, saved for comparison.
//   ucode ... examples/dsv41_logits.uc <model_dir> <out.safetensors> <comma separated device per layer | gpus> [tokens]
import * as dsv41 from 'ampstein.dsv41';

let dir = ARGV[0], out = ARGV[1], spec = ARGV[2], n = +(ARGV[3] || 600);
let devs = (index(spec, ',') >= 0) ? map(split(spec, ','), (x) => +x) : +spec;
let m = dsv41.load(dir, {devices: devs, max_seq: 8192, prefill_rows: 512});
let ids = [ 0 ], x = 99;

for (let i = 1; i < n; i++) {
	x = (x * 1103515245 + 12345) % 2147483648;
	push(ids, 1000 + (x % 60000));
}

let cache = m.new_cache(8192);
let lg = m.prefill(ids, cache, 512, 1);
let g = m.graphs(cache, [ 1 ]);

printf('devices %J\n', m.devs);
torch.st_save(out, [ [ 'logits', lg.to('float32').cpu() ] ]);
