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

- `gemma4.uc`: layers split over `--gpus` (contiguous runs), prefill chunk by chunk, **eager decode** (no CUDA graphs, no
  speculative decoding, no TP yet). Sliding layers keep a ring of 1024 + chunk rows and attend with `qsa_attn` over the window's
  slots; global layers attend in fp32 on slabs of queries (head dim 512 has no fused kernel).
- Experts share one bit rate and the mcg codebook: decode uses the cooperative MoE kernel (gelu), prefill the composite
  per-expert path (the fused sorted path is `mul1`-only), which bounds prefill (~470 tok/s at 3.6k tokens on two A5000).
- The RAM store / slots / hot reload are off (no `cache_export` yet), `--tp` is ignored.
- Tokenizer: SentencePiece-style BPE (`tokenizer.c` spm mode: spaces -> U+2581, merges over the whole run, `<0xXX>` byte
  fallback); identical to the HF reference on the test strings (`examples/tok_ids.uc`).
- Chat: `chat_gemma4.uc` + `chat_gemma4.jinja` (the checkpoint template with its two recursive macros unrolled to depth 4; the
  transpiler inlines macros). Thinking is off unless `enable_thinking` is set (the prompt then ends in the empty thought
  channel); tool calls `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>` are parsed to OpenAI tool calls, tool results go back
  inside the same model turn.

## Measured (2 x RTX A5000, layer split, eager)

Needle at 1.8k depth (beyond the window) correct; prefill 470 tok/s (3.6k tokens), decode ~22 tok/s; chat, system prompts, tool
calls with a tool result round trip and thinking mode behave.

## Open

CUDA-graph decode, a fused mcg MoE prefill path, a head-512 decode attention kernel, TP / stage pipeline, vision tower,
store / hot reload, KV quantization of the 5 global layers.
