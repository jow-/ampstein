# Gemma 4 (`Gemma4ForConditionalGeneration`): 26B-A4B

Checkpoint: `turboderp/gemma-4-26B-A4B-it-exl3` @ `6.10bpw` (23.6 GB: attention 8 bpw, dense MLP 7, experts 6, head 6).
References: transformers `modeling_gemma4.py`, exllamav3 `architecture/gemma4.py`. Text only for now.

## Architecture (as implemented in `gemma4.uc`)

- 30 layers, hidden 2816, vocab 262144, tied embeddings (the checkpoint ships a quantized `lm_head` too), logits soft-capped
  `30 * tanh(l / 30)`, embeddings scaled by 53 (bf16 rounding of sqrt(2816)).
- Attention: 16 q heads, **no 1/sqrt(d) scaling**, q / k norms with weight, v norm without; 25 sliding layers (window 1024,
  8 kv heads x 256, rope theta 1e4) and 5 global layers (2 kv heads x 512, K = V before the norms, rope theta 1e6 on a quarter of
  the dims: "proportional", the other angles are zero).
- Every layer: dense gated MLP (2112, gelu tanh) **and** 128 routed experts (top 8, 704, gelu) side by side:
  `h = post_ff(post_ff_1(mlp(pre_ff(x))) + post_ff_2(experts(pre_ff_2(x))))`, residual, `x *= layer_scalar`. The router sees the
  residual: rms(x) * scale / sqrt(hidden) -> softmax -> top 8 -> renormalised -> times `per_expert_scale[expert]`.
- Norms use their weight directly (not 1 + w). Residual stream fp32.

## Runtime

- `gemma4.uc` (layer split over `--gpus`, the default): contiguous layer runs per device. Sliding layers keep a ring of 1024 +
  chunk rows; prefill attends with fused sdpa under a window mask, decode with `qsa_attn` over the window's slots. Global layers
  attend in fp32 on slabs of queries (head dim 512 has no fused kernel).
- **Prefill** is a wavefront over the device runs (`prefill_pipeline`): every round each stage advances its own chunk, the layers
  of the stages are issued round-robin (a stage issued whole blocks the host on its launch queue and starves the other GPU).
  The MoE needs no host read (experts and `mcg` codebook: one tier of 16-row fused tiles). 3.6k tokens 1550 tok/s, 14k 2400-2580
  tok/s on two A5000.
- **Decode** replays CUDA graphs, one per device run and global-attention length bucket (power of two >= position + 1, captured
  lazily); the position is a device tensor (rope tables, ring slot, window slots, causal mask of the global layers). The post-norm
  residual blocks are one `norm32` kernel each. ~72 tok/s layer split, ~81-84 tok/s with `--tp`.
- `--tp` (`gemma4_tp.uc`): attention by KV-head groups (sliding 4+4, global 1+1), dense MLP by columns / rows over the 128-aligned
  padded width (2176 = 1152 + 1024), experts expert-parallel (64 + 64), lm_head by vocabulary ranges; two all-reduces per layer
  (attention; dense MLP and experts stacked). Decode is GPU-bound on ~1700 small kernels per token, so tp gains only ~15%;
  tp prefill is slower than the pipeline (1350-1700 tok/s).
- The RAM store / slots / hot reload are off (no `cache_export` yet).
- Tokenizer: SentencePiece-style BPE (`tokenizer.c` spm mode: spaces -> U+2581, merges over the whole run, `<0xXX>` byte
  fallback); identical to the HF reference on the test strings (`examples/tok_ids.uc`).
- Chat: `chat_gemma4.uc` + `chat_gemma4.jinja` (the checkpoint template with its two recursive macros unrolled to depth 4; the
  transpiler inlines macros). Thinking is off unless `enable_thinking` is set (the prompt then ends in the empty thought
  channel); tool calls `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>` are parsed to OpenAI tool calls, tool results go back
  inside the same model turn.

## Open

Fewer kernels per decode token (batched q/k/v and gate/up launches), a head-512 decode attention kernel (the global layers read
the whole cache in fp32), vision tower, store / hot reload, KV quantization of the 5 global layers.
