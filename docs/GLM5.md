# GLM-5.3-Flash (`Glm5NextForConditionalGeneration`)

Target checkpoint: `GLM-5.3-Flash-exl3-3.05bpw` (117 GB, 15 shards + `mtp.safetensors`
+ `kpool_aux.safetensors`), on the six-GPU host (6x RTX A5000/A5500, 24 GiB, sm_86,
PCIe 3.0 x4 each). Scope: text + MTP + vision.

Reference implementation: exllamav3 1.5.3 (the installed wheel — it is what
TabbyAPI serves with; an older source checkout may differ). Oracle scripts run
with the python environment that has that exllamav3 installed.

## 1. Architecture (from config.json + exllamav3 `glm5_next.py`)

45 layers, hidden 4096, vocab 154880, residual = **4 parallel fp32 streams**
(mHC hyper-connections, DeepSeek-V4 style) instead of one vector.

Per layer `i`:

```
streams x : (T, 4, 4096) fp32            (embedding broadcast into 4 copies)
attn site : (post, comb, c) = hc_attn.mix(x)      c = collapsed (T,4096)
            y = attn( input_layernorm(c) )
            x = post[:, :, None] * y[:, None, :] + comb^T @ x
ffn site  : same with hc_ffn, post_attention_layernorm, mlp
final     : mean over the 4 streams -> norm -> lm_head
```

* `hc_*` mix (fp32): `flat = unweighted_rmsnorm(x.flatten(1))`, `mix = flat @ fn^T`
  split into pre(4) / post(4) / comb(16); `pre = sigmoid(pre*s0 + b) + eps`,
  `post = 2*sigmoid(...)`, `comb = softmax(comb_logits, -1) + eps`, then 20
  Sinkhorn iterations (column / row normalisation with eps);
  `collapsed = sum_h pre[h] * x[h]`.
* Layer types (`layer_types`): 34 **KDA** (Kimi delta attention, linear) and
  11 **DSA** (NoPE MLA + sparse indexer) at layers 3,7,...,43.
* **KDA**: fused `qkv_proj` (EXL3) → depthwise causal conv (k=4) + silu over
  q|k|v → l2norm(q)*dk^-0.5, l2norm(k) → delta rule with a **per-key-channel**
  decay (state `S[dk,dv]` per head; 64 heads x 128): `S *= exp(g)[:,None];
  kv = k^T S; S += k (beta (v - kv))^T; o = q^T S`. `g = lower_bound *
  sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))` (lower_bound -5), `beta =
  sigmoid(b_proj x)`, output = `GatedRMSNorm(o, gate=g_b(g_a(x)))` (sigmoid
  gate) → `o_proj`.
* **DSA/MLA** (NoPE: `qk_rope_head_dim = 0`): `q = q_b(rms(q_a x))` (64 heads x
  256), latent `ckv = rms(kv_a x)` (512, cached), weights `kv_b` (fp16, split
  into W_UK / W_UV). Decode/absorbed form: `q_lat = q W_UK^T`, `s = q_lat ckv^T *
  256^-0.5`, `o_lat = softmax(s) ckv`, `o = o_lat W_UV` → `o_proj`. Context
  above `index_topk` (2048) selects keys with the **k-pool indexer**
  (`indexer.wq_b/wk/k_norm/weights_proj`, pool of 4 keys, always-select-tail);
  at <= 2048 tokens selection is all-inclusive so dense MLA is exact.
* **FFN**: layers 0-2 dense gated MLP (12288), layers 3-44 MoE: fp32 sigmoid
  router (`mlp.gate.weight`, `e_score_correction_bias`), top-8 of 288 on
  `score + bias`, weights = unbiased scores normalised to sum 1, x2.5, plus a
  shared expert (2048); silu with `swiglu_limit` = 10 clamp.
* **MTP** head (`mtp.safetensors`, DS3-style) and a **vision tower** (GLM4V
  structure with GLM5.3 deltas) — later milestones.

All big linears are EXL3 with the `mul1` codebook; routers, `b/f_a/f_b/g_a/g_b`,
indexer `wk/weights_proj`, `kv_b`, convs and norms are fp16/bf16 tensors.

## 2. Milestones (each gated by an oracle test against exllamav3)

| M | Content | Oracle |
|---|---|---|
| M1 (done) | Oracle capture script (`tests/oracle/make_glm5_oracle.py`): per-layer streams + sub-block outputs + logits | fixtures |
| M2 (done) | mHC ops (mix / apply / head), dense MLP layers 0-2 | layer 0-2 FFN |
| M3 (done) | KDA layer: conv + per-channel delta rule scan (+ gates) | layer 0 attn |
| M4 (done) | NoPE MLA dense path (prefill + absorbed decode), latent KV cache | layer 3 attn |
| M5 (done) | MoE (router, 288 EXL3 experts, shared expert) | layer 3+ FFN |
| M6 (done) | Whole model, layer split over 6 GPUs, prefill + decode, tokenizer/chat template | logits, greedy continuation |
| M7 (done) | k-pool DSA indexer (context > 2048) | long-context oracle |
| M8 (done) | Server/engine integration, CUDA graphs, performance | OpenAI client |
| M9 (done) | MTP head + speculative decoding | greedy equality |
| M10 (done) | Vision tower + preprocessing | embeddings + continuation |

## 3. Status

All milestones are implemented and oracle-checked; measured numbers are in
`OVERVIEW.md` section 5. Modules: `glm5.uc` (model, layer split), `glm5_tp.uc`
(tensor/expert parallel, MTP, graphs), `glm5_vision.uc`, `chat_glm5.uc`.
