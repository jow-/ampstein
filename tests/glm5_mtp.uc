// tests/glm5_mtp.uc — GLM-5.3-Flash MTP draft head (tensor-parallel). On the
// oracle chat prompt: prefill the target (collecting the post-norm hidden of
// every row), seed the draft cache, then step the target through the oracle
// continuation (graphs) and let the draft head predict the NEXT token at every
// step from (token, previous hidden): the draft must agree with the true
// continuation for most positions (an untrained/broken head scores ~0).
import * as glm5 from 'ampstein.glm5_tp';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_oracle.safetensors`;

if (!stat(MODEL) || !stat(FIX)) { warn('skip: model or fixture missing\n'); exit(125); }

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let cont = fx.tensor('cont').tolist();
let n = length(ids);
// AMP_GLM5_STAGES=3: three tensor-parallel stages of two ranks (glm5_tp pipeline)
let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6),
	stages: +(getenv('AMP_GLM5_STAGES') || 1), max_seq: 1024 });
let cfg = model.cfg;

assert(model.mtp, 'the checkpoint has an MTP head');

let cache = model.new_cache(1024);
let G = model.graphs(cache, [1]);          // clobbers the cache

for (let i = 0; i < cfg.num_hidden_layers; i++)
	for (let cl in cache.layers[i])
		if (cl.rec_state) { cl.rec_state.fill_(0); cl.conv_state.fill_(0); }

cache.length = 0;

let run = model.prefill_pipeline(cache, ids, { chunk: 512, hidden: true });

while (run.step(1)) {}

let H_all = run.hidden();             // (1, n, hidden)
let lg = run.logits();
let dc = cache.mtp;

// seed: position i pairs token i with the target hidden at i-1
let zero = torch.zeros([1, 1, cfg.hidden_size], 'float16', 'cuda:' + model.mtp.dev);
let seed = torch.cat([zero, H_all.narrow(1, 0, n - 1)], 1);

model.mtp_chunk(ids, seed.contiguous(), dc, 0, null);

let hit = 0, tot = 0, sample = [];
let H_last = H_all.narrow(1, n - 1, 1).contiguous();

for (let k = 0; k < length(cont) - 1; k++) {
	let t = cont[k];       // teacher-forced: the oracle's token (the trunk's own pick can flip on a near-tie)

	// draft: (token t at position n+k, hidden at n+k-1) -> token n+k+1
	let r = model.mtp_chunk([t], H_last, dc, n + k, true);
	let d = r.logits.reshape([-1]).argmax().item();

	tot++;

	if (d == cont[k + 1])
		hit++;

	push(sample, d);

	let o = G.run([t], cache.length);

	lg = o.logits;
	H_last = o.hidden.add(0.0);
}

printf('draft head agrees with the true next token at %d/%d steps\n', hit, tot);
printf('drafted:  %s\n', join(',', sample));
printf('true next: %s\n', join(',', slice(cont, 1)));
assert(hit * 2 >= tot, 'the draft head predicts at least half of the next tokens');
print('glm5_mtp: ok\n');
