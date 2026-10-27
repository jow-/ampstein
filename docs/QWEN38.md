# Qwen3.8-27B from GGUF (`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`)

A 64-layer Qwen3.5-family hybrid (`general.architecture = qwen35`): every fourth layer is full attention (24 query / 4 KV heads, head
dim 256, gated), the other 48 are gated delta net layers (16 key / 48 value heads, dim 128), dense SwiGLU MLPs (17408), MRoPE, a vision
tower and a one-block MTP head. It is the same architecture as `qwen35.uc` already served from EXL3 packs; what is new is reading it from a
GGUF, on a ROCm GPU (`docs/ROCM.md`). No new model module: `qwen35.uc`, `qwen35_graph.uc`, `qwen35_vision.uc` and the MTP code run
unchanged apart from the points below.

```sh
M=/data/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp        # see "The directory"
ucode -L build-rocm/lib -L src/ucode ./ampstein --model $M --ctx 65536          # vision and MTP (window 3) on by default
```

## The checkpoint

`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB, 3.55 bits per weight with the head) assigns one ggml type **per tensor**: IQ3_S 144,
IQ3_XXS 78, IQ4_XS 96, Q4_K 39, IQ2_S 17, Q2_K 13, Q6_K 8 (the MTP block), IQ2_XXS 5, IQ2_XS 9, IQ1_M 1 tensors; the embedding is IQ2_S, the
head Q4_K, the small GDN projections (`ssm_alpha`, `ssm_beta`) BF16, norms and the GDN constants F32. The vision tower is a separate
`mmproj-Qwen3.8-27B-BF16.gguf` (0.9 GB, BF16), and the repository ships no `config.json` or tokenizer. Block 64 of the main file is the MTP
head. Measured: 12.1 GB of weights plus about 2.5 GB for the embedding table dequantized to fp16, so a 24 GB card keeps a long context free.

### The directory

```
Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, mmproj-Qwen3.8-27B-BF16.gguf        from the GGUF repository
config.json, tokenizer.json, tokenizer_config.json, generation_config.json, chat_template.jinja,
preprocessor_config.json, video_preprocessor_config.json, vocab.json, merges.txt                 from Qwen/Qwen3.8-27B
```

`weights.open` recognizes a directory with `.gguf` files and no `model.safetensors*` (`ggufw.uc`); `mmproj*.gguf` is not a backbone shard but
the vision tower's weight file. `scripts/hf-download-ranged.sh` fetches the files (one connection gets ~1 MiB/s from the Hub's CDN);
check the result against the SHA-256 the Hub lists.

## What a llama.cpp GGUF changes, and how it is undone (`ggufw.uc`)

llama.cpp's converter (`conversion/qwen.py`) does not store the HF tensors as they are. The adapter maps the names back and undoes
or accommodates each change, so that `qwen35.uc` reads the model as it reads an HF checkpoint:

| In the GGUF | Done here |
|---|---|
| tensor names (`blk.N.attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, `ffn_*`, ...) | mapped to `model.language_model.layers.N.linear_attn.in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj / A_log / dt_bias / conv1d / norm`, `mlp.*`, `self_attn.*` (checked against gguf-py's own tensor name map) |
| every `*norm.weight` except the GDN's own is stored as `1 + w` | `- 1` on load (`qwen35.uc` adds the 1 itself) |
| `ssm_a` is `-exp(A_log)` | `log(-a)` on load |
| the conv1d kernel is squeezed to (channels, taps) | un-squeezed |
| **the value heads of the GDN layers are reordered from grouped to tiled order** (HF: the 3 value heads of key head k together; GGUF: `[k0 k1 .. k15, k0 k1 ..]`), in `in_proj_qkv` (V rows), `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, the conv1d V channels and the **columns** of `out_proj` | **not undone**: permuting the columns of a block-quantized matrix needs a dequantization. The weights stay in the file's order and the scan maps value head `h` to key head `h % Hk` instead of `h / (Hv / Hk)` (`cfg.ln_tiled`, an optional last argument of `torch.gdn_scan`); everything else in the layer is per value head. `tests/gdn_scan_tiled.uc`: identical to the grouped scan on permuted data |
| the MTP head is block 64: `nextn.eh_proj / enorm / hnorm / shared_head_norm` | `mtp.fc / pre_fc_norm_embedding / pre_fc_norm_hidden / norm`; the block's tensors are `mtp.layers.0.*` |
| the vision tower: fused `attn_qkv`, the Conv3d patch embedding as two Conv2d tensors, `mm.0` / `mm.2` | rows split into q / k / v, the halves stacked on the temporal axis, merger names mapped |

## Quantization kernels (`amp_gguf.cu`)

The model needs ten ggml types; seven were new (IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, Q2_K, Q4_K, Q6_K). For every one: a
dequantization (`gguf_dequant`, embedding rows, the fp16-activation GEMV) written from ggml's `dequantize_row_*` and **bit-identical**
to the `gguf` Python package on random blocks (`tests/gguf_dequant.uc`, f32 / f16 / bf16 all 0 difference), and for every type but IQ4_NL an int8
`dp4a` decode GEMV (activations quantized per 32 as in llama.cpp's q8_1; relative error 0.4 - 0.8 % of the largest output). Lookup
tables (`iq3s_grid`, `iq3xxs_grid`, `iq2s_grid`) are appended verbatim to `vendor/ggml/ggml_tables.inc`. The MoE prefill kernels were not
extended (a dense model has no routed experts). Prefill of a dense matrix dequantizes a 16 MiB slice to fp16 and runs the BLAS gemm.

## Results

RX 7900 XTX (24 GB, 960 GB/s), greedy, one request at a time, 8k context, CUDA graphs on. See "Reproducing" for the prompts.

| | |
|---|---|
| load (12.1 GB from NVMe, page cache warm) | 6 s |
| prefill, 10.8k-token prompt (default 1024-token chunks) | 968 tok/s (a 69-token prompt: 220 tok/s, bound by dequantizing every weight once) |
| decode without MTP (`--mtp 0`) | 47 tok/s |
| decode with MTP (window 4, the ROCm default): short answer / code / prose | 96 / 95 / 70 tok/s, 37 - 81 % of the drafts accepted, 2.3 - 3.5 tokens per round (window 3: 81 / 89 / 68) |
| a 52059-token prompt (`--ctx 65536`) with a passcode hidden in the middle | retrieved exactly (measured before the tuning below: prefill 670 tok/s, decode 54 tok/s with MTP at that depth) |
| an image (224 x 224, red circle left, blue square right) | described correctly; the tower loads in under a second, a 137-token request answers in 4.5 s |

Reproducing: `curl .../v1/chat/completions` with `temperature 0` and the prompts "What is the capital of France? Answer in one short sentence." (a
correct one-sentence answer after a short reasoning), "Write a Python function that merges two sorted lists, with a docstring and tests.",
"Write a short story about a lighthouse keeper, about 200 words.", and for prefill 160 numbered "the quick brown fox" sentences. The server
logs one `chat:` line per request with the prefill / decode speeds and the MTP statistics.

## Context length: fp16 and Q8 KV cache

Only 16 of the 64 layers keep a KV cache (64 KiB per token in fp16); the 48 GDN layers hold a fixed state. `--cache-mode Q8` stores those
layers' keys and values as int8 codes plus an fp16 scale per 32 values (264 bytes per 256 values instead of 512: about half). Decode reads
it with `torch.attn_decode_q8` (dequantizes in the tile load, error ~0.5 % of the largest output, `tests/attn_decode_q8.uc`); prefill
quantizes each chunk into the cache, attends the first chunk to its exact keys, and later chunks to the cache dequantized one KV head at
a time (a transient fp16 copy of 1/4 of the prefix per layer). Same answers and MTP acceptance as fp16 in the checks below.

RX 7900 XTX (24 GB), loads with a warm-up (the limits below were found with `--mtp 3`; the default window 4 costs ~100 MB, see the safe maximum):

| `--ctx` | fp16 | Q8 |
|---|---|---|
| largest that loads | 122,880 (126,976: OOM) | 193,536 (196,608: OOM) |
| near-full needle (passcode hidden mid-prompt), vision off | 116,783 tokens retrieved | 180,247 tokens retrieved, 190,464 ctx |
| near-full needle, then an image, vision on (default) | not measured | **172,032 ctx: passes** (162,919 tokens; prefill 510 tok/s, decode 66 tok/s with MTP at that depth, against 95 near the start of a conversation). 180,224: needle passes, the following image request runs out of memory. 190,464: same |

**Safe maximum with Q8: `--ctx 172032`** (vision on, the defaults, MTP window 4; 176128 loads and answers the needle but fails the image request that follows). Above it the card holds the cache but not the vision tower's
activations after a very deep prefill; a text-only server (`--no-vision`) has been run to 190,464, at the price of being close to the limit.
Short-context quality with Q8: correct one-sentence answer, code generation, needle at 10.8k, MTP acceptance 69 - 85 % (fp16: 59 - 81 %).

Q8 limits: single chain only (no `--tp`, no `--dflash`); decode takes at most 5 rows per step (`--mtp` <= 4, already the ROCm limit; the WMMA decode kernel takes 32 query rows per kv head);
the MTP head's own cache stays fp16. The host-RAM cache store (`--cache-ram`, session parking) works with Q8: the engine parks the
`kq` / `ks` / `vq` / `vs` rows [0, n) of every attention layer as they are (lossless, no age tiers; ~0.12 MiB per token, 0.4 GiB for 3.2k
tokens against 0.5 GiB for fp16 with tiers off), and the hot-reload file holds them too. A 3.2k-token state parks in ~0.3 s and swaps
back in 10-30 ms on the 7900 XTX (`tests/engine_store_gguf*.uc`: restore == unparked continuation, token for token).

## Performance work on ROCm

Found by profiling (`rocprofv3 --kernel-trace`), in the order of what it paid:

- **The decode GEMV kernels are limited by vector ALU, not by memory, on RDNA3** (they were written for NVIDIA parts with far more integer
  throughput per byte). `__vsub4` (the sign trick of every i-quant decoder) was a per-byte loop of ~30 instructions; it is now the 6-op SWAR
  form (checked against the loop on 200M random pairs). IQ3_S takes its sign masks from an LDS table instead of computing them (the
  sign byte is raw, 256 entries), IQ4_XS looks up `kvalues_iq4nl` with two `v_perm_b32` and a select, and reads its 16 quant bytes as one
  128-bit load. Per call on the real model: IQ3_S 49.5 -> 35.6 us, IQ4_XS 45 -> 41 us; the GEMVs now read about 730 GB/s on average.
  Decode without MTP went from 39 to 45.5 tok/s.
- **Prefill GEMM shape**: `matmul_rows` (ggufq.uc) cut each dequantized matrix into slices of `8M / K` rows (1638 for K = 5120). hipBLASLt
  runs a 1024 x 5120 x 1536 product at 83 TFLOPS and 1638 columns at 33. Slices are now a multiple of 128 rows: prefill +15 - 20 %.
  `AMP_GGUF_SLICE` (elements) still sets the slice (a larger one is a little faster and costs VRAM); `--chunk` 2048 / 4096 is worth +5 %.
- **Q2_K** (13 tensors) has an int8 kernel now (it ran the fp16-activation one at ~270 GB/s: 100 -> 36 us on the gate / up shape). The scales and
  minimums of its 16-weight halves ride in the `mult` / `dlt` slots IQ1 uses.
- **Defaults on ROCm**: the dense GEMV quantizes its own activation row (`AMP_GGUF_QX=dense`, +1.3 %; `off` restores the separate kernel) and
  Qwen3.5 / 3.8 speculate with `--mtp 4` (the most the scalar decode attention verifies).
- **Gate / up / SwiGLU as one launch** (`ggufq.gate_up`, the `gguf_gate_up` kernel the DeepSeek / GLM paths use): 5 launches per layer become 2, ~190 fewer per
  token, +1 % decode. Small, because in graph mode the gap between kernels is ~1 - 2 us: the non-GEMV kernels are ~14 % of a token and the gaps at most ~8 %, so
  each further fusion (the residual add into the down GEMV's epilogue, the activation quantize into the norm, the delta-net helper kernels) is worth 1 - 2 %.
- **Butterfly shuffles** (`__shfl_xor_sync`, in every reduction) are DPP `row_xmask` and `v_permlanex16` instead of `ds_bpermute`
  (`amp_hip_compat.h`); the decode attention kernels are issue-bound, so this shows there.
- **Q8 decode attention** (`amp_attn_dec_kernel<..., Q8>`): the int8 codes are read as unsigned bytes (sign bit flipped, one
  `v_cvt_f32_ubyteN` each; the -128 is folded into the query sum and the V weight sum) and addressed from wave-uniform bases: 1488 -> 1090 us
  per layer at 180k keys (fp16: 911 us for twice the bytes), 2687 -> 2238 us for a 4-row verify.

What was tried and dropped: unrolling the GEMV item loop by 2 (more registers, slower), TunableOp / `TORCH_BLAS_PREFER_HIPBLASLT=0`
(no gain once the slices are aligned). Left: the gated delta rule scan is the remaining non-GEMM prefill cost (7 %, no chunked kernel on
ROCm), the prefill GEMMs read each weight as fp16 after a separate dequantization (a fused dequantize-GEMM would save ~15 %), a 4-row Q8 verify still costs twice the fp16 one per key.

## Long context

What slowed down with depth, and what was done (deep numbers: 159k-token prompt, Q8 KV, window 4):

- **Decode 28.7 -> 66 tok/s.** Two causes. (1) The scalar split-KV decode attention costs ~85 vector instructions per key and query row: for the 4-5 row verify of
  an MTP round, 2.2 - 2.6 ms per layer at 180k keys (36 ms per round). It is replaced by a kernel on the WMMA matrix units (`amp_attn_dec_wmma_kernel`,
  head dim 256, <= 32 query rows per kv head, fp16 and Q8 caches): a block takes one kv head and a split of its keys, all rep x S rows share every K / V tile
  (V stored transposed in LDS for the B fragments, the Q8 codes dequantized while staging, next tile's global loads in flight), `S = Q K^T` and `O += P V` are
  `v_wmma_f32_16x16x16_f16`, the softmax goes through LDS (which also transposes the WMMA C layout into the A layout). 0.69 ms per layer for a 5-row verify of
  180k keys (3.9x), 0.46 ms for one row: the Q8 cache is read at 850 GB/s. fp16 caches run at the same ~870 GB/s. (2) The MTP draft steps called ATen FlashAttention
  with ONE query row over the draft cache: 10 ms for 160k keys (24 programs for the whole GPU), 4 steps per round = 40 ms. Draft steps (S <= 5) now use the same decode kernel (0.8 ms).
- **Prefill attention** runs on a prefill variant of the WMMA kernel (32 rows = 5 positions x 6 heads per block, the whole causal prefix, normalized output, no
  split / combine), straight over the Q8 cache rows: no dequantized copy of the prefix per chunk, which also left room for 4k more context. It measures 40
  TFLOPS against 33 for ATen's FlashAttention (AOTriton), so prefill at depth improved by only ~1.5 %: the extra cost per token at depth (0.9 ms at an average depth
  of 80k) is attention at 33 TFLOPS, and a kernel at 40 does not change that much. A prefill attention at 60+ TFLOPS would (the WMMA kernel is issue-bound on its
  softmax and staging work, not on the matrix units).
- `AMP_ATTN_SCALAR=1` selects the scalar decode kernels, `AMP_NO_WMMA_PREFILL=1` the FlashAttention prefill, `AMP_WMMA_SPLITS=N` the key splits of the decode kernel (default 2 x CUs / kv heads).

Tests: `attn_decode_wmma` (against the scalar kernel, S = 1..5, tile and split boundaries), `attn_prefill_wmma` (against FlashAttention), `attn_decode_q8`.

## Known limits

- Decode reads the weights at ~730 GB/s of the card's 960 (see "Performance work").
- Head dims other than 256, and more than 32 query rows per kv head (a verify window above 5 at 6 heads per kv head), use the scalar decode kernel / ATen's SDPA.
- Prefill at depth is attention-bound at ~33 - 40 TFLOPS (see "Long context").
- `--tp` (the P2P all-reduce) is CUDA only; one GPU is what this was run with.
- `--mtp` is capped at 4 on ROCm (and 4 is the default) (a verify step has window + 1 rows; the scalar decode attention takes at most 5).
- Not compared against llama.cpp's logits, only against the `gguf` package's dequantization and by behaviour (coherent, correct answers,
  the vision example below, an MTP acceptance of 70 - 75 %, which only a correct head and a correct backbone reach).

## Tests

`gguf_dequant` (all fourteen types), `gguf_reader`, `gdn_scan`, `gdn_scan_tiled`.
