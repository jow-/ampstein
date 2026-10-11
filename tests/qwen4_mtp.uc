// tests/qwen4_mtp.uc — Qwen3.8-Flash-Next MTP draft head, teacher-forced: the
// draft step is fed the target's TRUE pre-mixer stream stack and the true next
// token, and its argmax is compared with the token after that. exllamav3 ships
// the head without a reference ("the combine choice must be confirmed by the
// acceptance rate"), so this accuracy is the quality gate. Needs all six GPUs.
// Env: AMP_MODEL_QWEN4, QWEN4_FIX (a long-prompt oracle fixture; only 'ids' is used).

import * as qwen4 from 'ampstein.qwen4';
import { stat } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN4') ||
	'/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw';
const FIX = getenv('AMP_QWEN4_FIX') || `${sourcepath(0, true)}/fixtures/qwen4_long.safetensors`;

if (!stat(MODEL) || !stat(FIX)) {
	warn('skip: Qwen4 model dir or fixture missing\n');
	exit(125);
}

let ids = torch.st(FIX).tensor('ids').tolist();
let n = length(ids);
let model = qwen4.load(MODEL, { mtp: true });

if (!model.mtp) {
	warn('skip: checkpoint has no MTP head\n');
	exit(125);
}

let cfg = model.cfg;
let M = model.mtp;
let cache = model.new_cache(n + 8);

model.forward_chunk(ids, cache);

// draft inputs: token t_i with the stack of position i-1 (zeros before the start)
let stack = cache.stack.cuda(M.dev);
let zeros = torch.zeros([1, cfg.hc_mult, cfg.hidden_size], 'float32', 'cuda:' + M.dev);
let sin = torch.cat([ zeros, stack.narrow(0, 0, n - 1).to('float16').to('float32') ], 0);
let dc = qwen4.new_mtp_cache(cfg, n + 8, M.dev);
let out = qwen4.mtp_step(model, M, ids, sin, dc, 0, null);

// logits in slabs (a full (n, vocab) matrix does not fit), compare with the next token
let ok = 0, tot = 0, ok2 = 0, tot2 = 0;
let lo = 64;

for (let r0 = 0; r0 < n - 1; r0 += 256) {
	let R = min(256, n - 1 - r0);
	let lg = model.lm_head.forward(out.hidden.narrow(0, r0, R).view([1, R, cfg.hidden_size]));
	let am = lg.view([R, -1]).argmax(-1).tolist();

	for (let i = 0; i < R; i++) {
		let hit = (am[i] == ids[r0 + i + 1]);

		tot++;
		if (hit) ok++;

		if (r0 + i >= lo) {
			tot2++;
			if (hit) ok2++;
		}
	}
}

printf('MTP teacher-forced next-next-token accuracy: %.1f%% over %d positions (%.1f%% from position %d on)\n',
	100.0 * ok / tot, tot, 100.0 * ok2 / tot2, lo);

try { torch.synchronize(); } catch (e) {}

let good = ok2 * 1.0 / tot2 > 0.3;

print(good ? 'qwen4_mtp ok\n' : 'qwen4_mtp: accuracy below 30%\n');
exit(good ? 0 : 1);
