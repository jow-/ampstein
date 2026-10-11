// ucode/dspark.uc — the DeepSeek-V4 DSpark block drafter (mtp.* in the checkpoint).
//
// Per round the drafter makes ONE forward over a block of dspark_block (5) tokens
// [seed, noise x 4] and emits 5 draft tokens:
//
//   target taps   stream mean after the target layers 40 / 41 / 42, fp16, per position
//   main row      main_norm(main_proj(cat(taps)))   (4096)        per position
//   main kv       per drafter block b: rope(kv_norm_b(wkv_b(main row)), position)   (512)
//   block b       plain V4 block (mHC, sliding attention, 256-expert MoE) whose queries
//                 (the 5 block tokens at positions p0 .. p0 + 4) attend NON-CAUSALLY over
//                 [the last <= 128 main kv rows ++ the block's own kv] with per-head sinks
//   state         hc_head + norm of the last block -> the TARGET's head -> logits (5, V)
//   draft         greedy chain with the markov bigram bias:
//                 out[0] = seed; logits[i] += markov_w2(markov_w1[out[i]]); out[i + 1] = argmax
//
// Embedding and head are the target's. let d = dspark.load(model, {dev});
//   d.update(taps, pos)     rows of positions pos .. pos + n - 1
//   d.draft(seed, p0)       -> [d1 .. d5]  (the main rows < p0 must be in)

import * as weights from './weights.uc';
import * as models from './models.uc';
import * as glm5 from './glm5.uc';
import * as dsv4 from './dsv4.uc';

/* Non-causal block attention: the B block queries (positions p0 + j) see the
 * last min(window, p0) main kv rows and the whole block, one softmax with sinks. */

