// tests/glm5_mtp_split.uc — GLM-5.3-Flash MTP draft head on the LAYER SPLIT
// (glm5.uc; glm5_mtp.uc is the tensor-parallel twin). Same protocol: prefill
// with the post-norm hidden of every row, seed the draft cache, step the
// target through the oracle continuation and let the draft head predict the
// next token from (token, previous hidden) — at least half must agree.
// Also: the CUDA-graph draft step must match the eager draft step, and a
// 3-row verify step (graph) must agree with three single-row steps.
import * as glm5 from 'ampstein.glm5';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_oracle.safetensors`;

if (!stat(MODEL) || !stat(FIX)) { warn('skip: model or fixture missing\n'); exit(125); }

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let cont = fx.tensor('cont').tolist();
let n = length(ids);
let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6) });
let cfg = model.cfg;

assert(model.mtp, 'the checkpoint has an MTP head');

let cache = model.new_cache(1024);
let G = model.graphs(cache, [ 1, 3 ]);          // clobbers the cache

for (let i = 0; i < cfg.num_hidden_layers; i++)
	if (cache.layers[i].rec_state) {
		cache.layers[i].rec_state.fill_(0);
		cache.layers[i].conv_state.fill_(0);
	}

cache.length = 0;

let run = model.prefill_pipeline(cache, ids, { chunk: 512, hidden: true });

while (run.step(1)) {}

let H_all = run.hidden();             // (1, n, hidden)
let dc = cache.mtp;

assert(H_all.shape()[1] == n, 'pipeline hands out one hidden row per prompt row');

// seed: position i pairs token i with the target hidden at i-1
let zero = torch.zeros([1, 1, cfg.hidden_size], 'float16', 'cuda:' + model.mtp.dev);
let seed = torch.cat([zero, H_all.narrow(1, 0, n - 1)], 1);

model.mtp_chunk(ids, seed.contiguous(), dc, 0, null);

let hit = 0, tot = 0, sample = [], gmiss = 0, gdiff = 0.0;
let H_last = H_all.narrow(1, n - 1, 1).contiguous();
let steps = length(cont) - 1;

// teacher-forced: the oracle's token (the trunk's own pick can flip on a near-tie)
for (let k = 0; k < steps; k++) {
	let t = cont[k];

	// draft: (token t at position n+k, hidden at n+k-1) -> token n+k+1
	let r = model.mtp_chunk([ t ], H_last, dc, n + k, true);
	let d = r.logits.reshape([ -1 ]).argmax().item();

	// the eager draft step (same inputs, same cache slot) must agree
	let mg = model.mtp_g;

	model.mtp_g = null;

	let re = model.mtp_chunk([ t ], H_last, dc, n + k, true);

	model.mtp_g = mg;

	let diff = r.logits.sub(re.logits).abs().max().item();

	if (diff > gdiff)
		gdiff = diff;

	if (re.logits.reshape([ -1 ]).argmax().item() != d)
		gmiss++;

	tot++;

	if (d == cont[k + 1])
		hit++;

	push(sample, d);

	let o = G.run([ t ], cache.length);

	H_last = o.hidden.add(0.0);
}

printf('draft head agrees with the true next token at %d/%d steps\n', hit, tot);
printf('drafted:  %s\n', join(',', sample));
printf('true next: %s\n', join(',', slice(cont, 1)));
printf('graph vs eager draft step: max |dlogit| %.4f, argmax mismatches %d\n', gdiff, gmiss);
assert(hit * 2 >= tot, 'the draft head predicts at least half of the next tokens');
assert(gmiss * 8 <= tot, 'the graph draft step matches the eager one');

// verify step: 3 rows through the size-3 graph vs three 1-row steps, each
// from a freshly prefilled cache
function fresh() {
	for (let i = 0; i < cfg.num_hidden_layers; i++)
		if (cache.layers[i].rec_state) {
			cache.layers[i].rec_state.fill_(0);
			cache.layers[i].conv_state.fill_(0);
		}

	cache.length = 0;

	let r = model.prefill_pipeline(cache, ids, { chunk: 512 });

	while (r.step(1)) {}
}

let toks = [ cont[0], cont[1], cont[2] ];

fresh();

let l3 = G.run(toks, n).logits.view([ 3, -1 ]);

fresh();

let worst = 0.0, same = 0;

for (let k = 0; k < 3; k++) {
	let l1 = G.run([ toks[k] ], n + k).logits.view([ -1 ]);
	let row = l3.narrow(0, k, 1).view([ -1 ]);
	let d = row.sub(l1).abs().max().item();

	if (d > worst)
		worst = d;

	if (row.argmax().item() == l1.argmax().item())
		same++;
}

printf('verify rows vs single steps: max |dlogit| %.4f, argmax equal %d/3\n', worst, same);
assert(same >= 2, 'the 3-row verify agrees with single-row steps');
assert(worst < 2.0, 'verify logits stay within fp16 noise of the single-row steps');
print('glm5_mtp_split: ok\n');
