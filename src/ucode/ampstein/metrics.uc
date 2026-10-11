// ucode/metrics.uc — numerics for model comparison and evaluation.
//
// The Phase-4 harness primitives, ported from exllamav3's eval tooling
// (util/measures.py: compute_kl_div / compute_target_log_probs) and
// eval/model_diff.py's ppl loop. Everything runs in fp32 on whatever
// device the inputs live on; only the final scalars cross back to
// ucode numbers.
//
// Conventions follow exllamav3:
//   KLD here is KL(softmax(target) || softmax(input)) per row — the
//   "target" is the reference model, "input" the model under test.
//   Perplexity scores ids[1:] under logits for positions 0..S-1.

import * as math from 'math';

/* logsumexp along the last dim, keepdim. x: (R, V) float tensor.
 * Returns (R, 1). Computed as m + log(sum(exp(x - m))) with m the row
 * max — the stable two-pass form. */
export function logsumexp(x, dim) {
	let m = x.max(dim, true);

	// m + log(sum(exp(x - m))) — NOT log(m + sum(exp(x - m)))
	return m.add(x.sub(m).exp().sum(dim, true).log());
}

/* Per-row KL divergence between two logit tensors of identical shape.
 * a: (R, V) logits of the model under test; b: (R, V) reference
 * logits. Returns an (R,) fp32 tensor of KL(b || a).
 *
 * KL = sum_v p_b(v) * (log p_b(v) - log p_a(v))
 *    = sum_v p_b(v) * ((b_v - lse_b) - (a_v - lse_a)) */
export function kld_rows(a, b) {
	let af = a.to('float32');
	let bf = b.to('float32');

	let lse_a = logsumexp(af, -1);   // (R, 1)
	let lse_b = logsumexp(bf, -1);   // (R, 1)

	let logp_b = bf.sub(lse_b);
	let p_b = logp_b.exp();

	return p_b.mul(logp_b.sub(af.sub(lse_a))).sum(-1, false);
}

/* Perplexity of a token sequence under given logits.
 *
 * logits: (S, V) tensor covering positions 0..S-1; ids: ucode array of
 * S+1 token ids (the full sequence). Position i's logits score token
 * ids[i+1], so the first token (typically BOS) is context only.
 *
 * Returns {ppl, nll, count} — nll is the total negative log
 * likelihood over `count` scored tokens. */
export function ppl(logits, ids) {
	let S = length(ids) - 1;

	let lp = logits.to('float32');
	let logp = lp.sub(logsumexp(lp, -1));   // (S, V) log-softmax

	let tgt = [];

	for (let i = 1; i < length(ids); i++)
		push(tgt, ids[i]);

	let idx = torch.from_array(tgt, 'int64').view([S, 1]).cuda();
	let picked = logp.gather(-1, idx);      // (S, 1)

	let nll = -picked.sum().item();

	return {ppl: math.exp(nll / S), nll: nll, count: S};
}