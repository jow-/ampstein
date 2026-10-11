// tests/glm5_ppl.uc — GLM-5.3-Flash quality vs exllamav3 on real text
// beyond index_topk: mean NLL of the last 512 positions and the KL
// divergence to the reference distributions (fixture:
// make_glm5_ppl_oracle.py). Needs all six GPUs (~190 s load).
import * as glm5 from 'ampstein.glm5';
import * as math from 'math';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
const FIX = `${sourcepath(0, true)}/fixtures/glm5_ppl_oracle.safetensors`;
const ROWS = 512;

if (!stat(MODEL) || !stat(FIX)) { warn('skip: model or ppl fixture missing\n'); exit(125); }

let fx = torch.st(FIX);
let ids = fx.tensor('ids').tolist();
let T = length(ids);
// AMP_GLM5_KV_BITS=8: run with the Q8 latent cache
let model = glm5.load(MODEL, { devices: +(getenv('AMP_GLM5_GPUS') || 6),
	kv_bits: getenv('AMP_GLM5_KV_BITS') ? int(getenv('AMP_GLM5_KV_BITS')) : null });
let cache = model.new_cache(T + 8);
let t0 = time();
let lg = model.prefill(ids, cache, 600, ROWS);

torch.synchronize();
printf('prefill %d tokens: %.1fs\n', T, time() - t0);

let ours = lg.cpu().to('float32');
let ref = fx.tensor('logits').to('float32');
let po = ours.softmax(-1), pr = ref.softmax(-1);
let tgt = torch.from_array(slice(ids, T - ROWS + 1), 'int64').view([-1, 1]);
let nll_o = po.narrow(0, 0, ROWS - 1).gather(1, tgt).log().neg().view([-1]);
let nll_r = fx.tensor('nll');
let kl = pr.mul(pr.add(1e-30).log().sub(po.add(1e-30).log())).sum(-1);

printf('mean NLL ours %.5f  reference %.5f  (diff %+.5f)\n', nll_o.mean().item(),
	nll_r.mean().item(), nll_o.mean().item() - nll_r.mean().item());
printf('KLD(ref||ours) mean %.5f max %.5f; top-1 agreement %d/%d\n', kl.mean().item(),
	kl.max().item(), ours.argmax(-1).sub(ref.argmax(-1)).abs().le(0).sum().item() * 1, ROWS);

let d = nll_o.mean().item() - nll_r.mean().item();

assert(d < 0.02 && d > -0.02, 'mean NLL within 0.02 of the reference');
assert(kl.mean().item() < 0.03, 'KLD');
print('glm5_ppl: ok\n');
