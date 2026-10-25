# GGUF checkpoints (DeepSeek-V4.1, `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`)

ampstein reads ggml-quantized GGUF files directly: the weights stay in their file format on the GPU and the kernels
dequantize in registers (decode) or per matrix (prefill). No llama.cpp, no conversion step. Supported today: the
single-chain DeepSeek-V4.1 model (`dsv41.uc`, layers split over the GPUs), with the DSpark drafter (taken from an EXL3 pack of the same
model); not yet `--tp` or the vision tower.

## The checkpoint

| File | Content |
|---|---|
| `...-GSQ-RCO-IQ2_XXS-0000i-of-00004.gguf` | backbone: attention, shared experts, routed experts, router, mHC, indexer, the Engram hash tables (token map, primes, offsets, multipliers), embedding and head. 138.2 GB = 128.7 GiB at 2.00 bpw |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (optional) | the two Engram lookup tables (layers 1 and 14), 384M rows x 256 at IQ4_NL (144 bytes per row), 110.6 GB, shared by all builds, read row by row from disk like the EXL3 packs' tables. ISTA's evaluation used them quantized like this |
| `model-00016-of-00017.safetensors`, `model-00017-of-00017.safetensors` (symlinks to a DeepSeek-V4.1-Flash-1017 EXL3 pack) | the same two tables at the original FP8 (+ e8m0 scales), 2 x 101.5 GB, already on disk next to the EXL3 packs: used instead of the IQ4_NL file when it is absent. Same tables: rows of both agree at cosine 0.997 (the IQ4_NL error), the hash tables are identical |
| `model-00002-of-00017.safetensors` (symlink to the same EXL3 pack) | the DSpark drafter (`mtp.*`, 3 stages, EXL3 + FP8, 3.45 GiB); the GGUF has none. `--mtp 0` runs without |
| `config.json`, `tokenizer.json`, `tokenizer_config.json` | copied from `deepseek-ai/DeepSeek-V4.1-Flash-1017` (a GGUF carries no config the engine reads) |

The builds (IQ1_M 120.7 GB, IQ2_XXS 138.2 GB, IQ2_XS 155.2 GB) assign one ggml type per tensor, so the routed experts of a
layer differ: gate / up / down are IQ1_M, IQ2_XS or IQ2_XXS per layer and matrix (a layer's 384 experts share a type);
the dense tensors are Q8_0 (`wq_a`, `wkv`) or Q5_K (everything else, embedding and head too); the compressors, indexer
projections, router and Engram `q` / `k` are bf16, norms and mHC parameters f32.

**Memory (jai, 6 x 23.5 GiB = 141 GiB):** IQ2_XXS (128.7 GiB) fits with a reduced context, IQ1_M (112.4 GiB) comfortably,
IQ2_XS (144.6 GiB) not at all.

## Running it (jai: 256k context + drafter, 0.0.0.0:8041)

**Model directory** (`/data/models/DSV4.1-Flash-1017-GSQ-RCO-IQ2_XXS`): symlinks to the four backbone shards (`/data/gguf/ista/...-0000i-of-00004.gguf`, 138 GB), to
shards 2, 16 and 17 of `/data/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw` (the drafter and the Engram tables), and copies of that repository's `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`. `weights.open` recognises a directory with `.gguf` files and no `model.safetensors*`.
The shards were fetched with many parallel ranged requests (one connection gets ~1 MiB/s from the Hub's CDN, `scripts/hf-download-ranged.sh`) and
checked against the SHA-256 of the Hub (all four match).

**Start** (`~/run-gguf.sh` on jai; `CTX`, `CHUNK`, `PORT`, `MTP=0` from the environment):

```sh
screen -dmS ampstein-gguf bash -c "~/run-gguf.sh > ~/ampstein-gguf.log 2>&1"
# which is
cd ~/ampstein-rpc && ucode -L $PWD/build/lib -L $PWD/src/ucode ./ampstein \
    --model /data/models/DSV4.1-Flash-1017-GSQ-RCO-IQ2_XXS --ctx 262144 --chunk 512 --host 0.0.0.0 --port 8041
```

Loading takes 85 - 130 s (138 GB from NVMe). `--tp` is refused for a GGUF directory.

**Memory.** 6 x 23.5 GiB = 141 GiB against 128.7 GiB of weights, so everything else has to be squeezed; what was done and what it saves, measured with
`examples/vram_audit_dsv41.uc` (tensors / allocator cache / non-torch memory / free, per GPU and phase) and `examples/vram_prefill_trace.uc`:

| | |
|---|---|
| layers are not contiguous per GPU | `rebalance` in `dsv41.uc` moves layers (and swaps pairs of them) off the GPUs that would keep less than 1.2 GiB free, counting weights, the kv pools and their per-GPU mirrors at the context length, the head (it follows the last layer), the embedding and the drafter's stages; each moved layer is two more hops of the activations (80 KB per token) |
| the drafter's three stages are placed jointly with the layers | on any GPU, even two stages on one; with 256k + drafter: GPUs 5, 2, 2 and GPU 2 gets six layers instead of seven |
| the embedding stays Q5_K (0.42 GiB, rows dequantized by id inside the graphs, `gguf_embed`) | -0.9 GiB against fp16 |
| **one decode-graph bucket** (`max_seq`) instead of four | **-2.8 GiB at 256k with the drafter**: each bucket kept its own ~0.5 GiB of graph memory (0.9 GiB with the verify sizes) and the pool is not shared across them. Decode is as fast at short and long contexts (29.2 - 31.5 tok/s at 47 tokens, 29.4 at 46k) |
| graphs only for the verify sizes 1, 3 and 5 rows | a size without a graph runs the next larger one and takes the extra rows back with `rollback` (bit-identical logits, `tests/dsv41_gguf_rows.uc`); capturing all of 2 - 5 cost 1.2 GiB more |
| a smaller tile in the prefill indexer (4M score elements instead of 32M) | its `topk` workspace was the context-dependent part of a prefill: 46k / 137k-token prompts ran out of memory, now they run at constant memory (`AMP_DSV41_IDX_SLAB`) |
| 16 MiB dequantization slices for the dense prefill matmuls (were 64 MiB) | `AMP_GGUF_SLICE` |

Result: free after load 10.5 GiB, after the drafter 6.9, after the kv cache for 256k 6.2, after the graphs ~2.8 (GPU 0: 0.4 GiB), and a 137k-token prompt
prefills without an out-of-memory error. It is tight by design: a configuration that does not fit shows up as `planned ... over its budget` at start and an OOM
in the warm-up; lower `CTX`, or `MTP=0` frees ~5 GiB.

**Speed** (jai, 6 x RTX A5000, greedy, steady state of a streamed answer; prefill = prompt tokens / time to the first token):

| | |
|---|---|
| prefill, 37k / 46k / 137k-token prompts (prompt skip active) | 1,892 / 1,313 / 1,487 tok/s (measured at different times: the 46k and 137k figures predate the deterministic prefill below) |
| prefill, 3.5k / 9k / 20k tokens (before the memory work: smaller tiles, one bucket) | 669 / 889 / 1,084 tok/s |
| generation without the drafter (`MTP=0`) | 37.7 tok/s in the server before the last kernel work; the pure graph-decode loop now runs 41 - 42 tok/s (24 ms per token; 22.5 ms without the Engram lookups, which are cold random reads from disk) |
| generation with the drafter: code / JSON / math / prose / chat (short prompts, wall clock) | 54 / 51 / 60 / 42.5 / 47 tok/s |
| decode kernel work since | two output rows per warp (weight loads of both rows in flight; `AMP_GGUF_RPW1=1` turns it off), activation-row-specialised kernels, IQ1_M decode with wide loads, a block of the IQ1_* / IQ2_XS kernels works through several rows per table fill, the shared expert on a forked stream during graph capture (`AMP_GGUF_NO_SHARED_PAR=1`: off; -1.1 ms), hop reduction, Engram rows of a later layer fetched while the earlier segments compute. Whole-model graph decode 22.5 ms per token (44 tok/s). The drafter's verify-cost table is the measured one (1 / 3 / 5 rows: 23.1 / 36.6 / 51.4 ms; 2 and 4 rows run the next captured graph), worth +10 - 20% on prose / JSON / chat. Tried and dropped: a conflict-free sign table for IQ2_XXS (slower: the kernel is limited by L1/TEX wavefronts, ALU and global requests together), grouping the picks of a verify step by expert (`AMP_GGUF_GROUP=1`, slower on random picks); a 2-byte-aligned vs 8-byte-aligned weight layout for IQ2_XXS / IQ2_XS (timing experiment, aligned 8-byte loads instead of four 2-byte ones): only -7 to -12% on those kernels (about 0.4 ms per token), not worth a repack of every consumer |
| the EXL3 1.62bpw pack that ran here before, single chain | 26 tok/s without drafter |

**Quality** (`examples/dsv41_nll.uc`, teacher-forced NLL of the same 14k tokens of text, last 2048 positions, same code and tokenizer): IQ2_XXS GGUF **2.354**
(perplexity 10.5, top-1 1024 / 2048) against the EXL3 1.62bpw pack 2.751 (15.7, 919 / 2048).

## Tensor parallel (`--tp --stages 3`: three stages of two GPUs)

`dsv41_tp.uc` runs GGUF checkpoints too: a stage is a pair of GPUs that both run every layer of the stage's layer range, each on its half of the
weights, and the partial sublayer outputs are summed by an all-reduce (a decode token then reads half of a stage's weights per GPU, but the
three stages still run one after the other). What is sharded: the attention by query heads / o-groups (`wq_b` rows, `wo_a` groups, `wo_b` columns),
the routed experts by expert range (`gq.e0`; the MoE kernels take the first expert of the shard and treat a pick outside it as none, so a rank uploads
only its slice of the stacked tensor), the shared expert, and the head by vocabulary rows. `wq_a`, `wkv`, the compressors, the indexer, the router, the
Engram and the embedding are replicated (about 1.3 GiB over the whole model). Column cuts of a quantized matrix fall on block boundaries.

```
CTX=65536 CHUNK=128 MTP=0 ~/run-gguf.sh --tp --stages 3
```

| | single chain (6 stages) | `--tp --stages 3` |
|---|---|---|
| context | 256k, with the drafter | **64k** (128-token chunks), no drafter |
| decode, no drafter (server, warmed) | 37.7 tok/s | **46.2 tok/s** |
| prefill | 1,300 - 1,500 tok/s | 780 tok/s (37k-token prompt, chunk 128) |

The limit is memory: the weights are 128.7 GiB of the 141 GiB, a contiguous stage split cannot be balanced better than 14 / 13 / 13 layers (the later layers
are bigger: 2.9 GiB at the start, 3.2 - 3.8 GiB at the end), which leaves 0.3 - 0.5 GiB free on the last stage's GPUs after the graphs. 16k with a 256-token chunk
and 64k with a 128-token chunk load and serve; 64k with a 512-token chunk does not warm up. The drafter (3.45 GiB) does not fit alongside. The smaller IQ1_M mix
(112 GiB) would leave room for both. `tests/dsv41_tp.uc` checks the layout against the single chain (`AMP_MODEL_DSV41=<gguf dir>`).

## Reproducibility and host overheads (found with the profile of a speculative round)

- **Prefill is bit-reproducible.** The two grouped MoE prefill kernels added each token's six expert outputs with `atomicAdd` in arbitrary order; the fp32 sum
  rounds differently in every run, the depth of the network amplified it, and the same prompt at temperature 0 produced different text after every server
  start. They now write each assignment's weighted output to its own scratch row and `torch.moe_gather` sums a token's rows in a fixed order (the way the EXL3
  path does it). No measurable prefill cost. `AMP_MOE_ATOMIC=1` restores the atomic sum. `examples/det_check_dsv41.uc` runs the same prompt twice and
  compares the logits bit for bit (graph steps of 1 and 3 rows included).
- **Verify-step prelude 2.3 -> 0.9 ms.** The position metadata of the six segments was rebuilt and uploaded six times through pageable memory; it is
  built once into page-locked memory (a ring of 8) and copied with non-blocking `copy_(src, true)`. The drafter's per-round inputs likewise. The Engram rows of
  a verify step are started for both layers up front and waited for just before the segment that holds them.
- **GPU idle time in a speculative round** (`nsys`, the live server): 18.4% of the wall clock before, 14.4% after. What remains: the three drafter stages and
  the eager `update_rows`, and the hops between GPUs (about 45 us each).
- `AMP_GC_INTERVAL` (default 50000): ucode's collector runs every 1000 allocations by default and costs about 1 ms per pass with the context arrays alive
  (50k-token prompts); decode rounds allocate enough for several passes. `AMP_DSPARK_PROF=1` prints the engine's per-round host timing, `AMP_DSV41_ROWPROF=1`
  the verify prelude.

## Code

| Part | Where |
|---|---|
| reader: `torch.gguf(path)` presents a GGUF as a safetensors-style header (ggml type names as dtypes, row-major shapes, the key/value metadata as `f.meta()` JSON). Everything built on `torch.st` works on it: zero-copy views, `h2d_file`, `read_into`, row `gather` | `src/torch/gguf.c`, `st.c` (`st_dtype_info`) |
| dequantization `torch.gguf_dequant(w, type, dtype)`, matrix-vector `torch.gguf_gemv` (dense, <= 8 rows), `torch.gguf_moe_gemv` (a stack of experts, one per slot). Types: Q8_0, IQ4_NL, Q5_K, IQ2_XXS, IQ2_XS, IQ1_M. Written from ggml's `dequantize_row_*`, bit-identical; lookup tables in `vendor/ggml` | `src/torch/amp_gguf.cu` |
| weight adapter: HF checkpoint names (`layers.N.attn.wq_a.weight`) -> GGUF names (`blk.N.attn_q_a.weight`), stacked experts as `layers.N.ffn.exps.w1/w3/w2`, Engram hash tables from the GGUF's own `engram_*` tensors | `ggufw.uc`, selected by `weights.open` for a directory with `.gguf` files |
| layers: quantized linears, grouped `wo_a`, the routed experts (decode: one GEMV launch per matrix over the (token, pick) slots; prefill: tokens sorted by expert, each active expert's three matrices dequantized to fp16 and run through cuBLAS) | `ggufq.uc`, hooks in `dsv41.uc` (`lin8`, `build_moe`, Engram) and `glm5.moe_begin` |

## Tests

| Test | Against |
|---|---|
| `gguf_reader` | fixture written by the `gguf` package: metadata of every value type, inventory, shapes, raw rows, malformed / truncated files |
| `gguf_dequant` | the `gguf` package's numpy dequantization (`tests/oracle/make_gguf_dequant.py`): float32 bit-identical, float16 / bfloat16 equal the rounded values; GEMV and the experts GEMV against `x @ W^T` |

## Kernel notes

- **Dequantization** (`gguf_dequant`) is bit-identical to ggml; it feeds prefill (cuBLAS on fp16 slices of a matrix) and the
  embedding.
- **Decode matrix-vector products** (`gguf_gemv`, `gguf_moe_gemv`) run in int8 for every type but IQ4_NL: the activation rows
  are quantized once per call (fp32 scale per 32 values, ggml's q8_1), the weights are decoded to signed bytes and multiplied
  with `dp4a` (Q5_K and IQ1_M add a `sum(x)` term for their minimum / +-1/8 offset). About 2 instructions per weight where the
  first version (fp16 activations, per-element decode) needed 12. RTX A5000 (768 GB/s), `examples/bench_gguf.uc`:

  | | us | GB/s |
  |---|---|---|
  | Q5_K 32768 x 1280 (`wq_b`), 1 row | 67 | 430 |
  | Q5_K 5120 x 8192 (`wo_b`) | 54 | 534 |
  | Q5_K 129280 x 5120 (head) | 655 | 695 |
  | IQ2_XXS gate / up of 6 experts | 43 | 428 |
  | IQ2_XXS down of 6 experts | 56 | 326 |
  | IQ2_XS gate / up, down | 46, 61 | 442, 338 |
  | IQ1_M gate / up, down | 50, 80 | 311, 194 |

  The scalar fp16-activation kernels (`AMP_GGUF_FP16=1`, exact dequantized weights x fp16 activations) are 3 - 8x slower for
  the i-quants. The activation rounding is ~0.3 - 0.6 % of an output (`tests/gguf_dequant.uc`).
- **Fusions in decode** (as llama.cpp's MMVQ does): the attention's `wo_a` (8 groups) is one launch (`gguf_gemv_grouped`); gate, up and
  the clamped SwiGLU of the routed experts and of the shared expert are one launch (`gguf_moe_gate_up`, `gguf_gate_up`), the routing
  weight is applied by the down projection (`gguf_moe_gemv(..., scale)`). 8 layers on 2 GPUs: 8.9 -> 6.4 - 7.5 ms per token
  (the machine was busy with a download, run-to-run spread of 20 %).
- **Prefill of the routed experts** (`gguf_moe_prefill`): tokens sorted by expert (`torch.moe_prep`), then two launches over (weight-row
  tile, expert, token tile of 32): stage A gate + up + SwiGLU -> `h`, `h` quantized, stage B down -> `out[token] += weight * result`.
  Default: int8 tensor cores (`mma.sync m16n8k16 s8`), the weights decoded from the quantized bytes straight into the A fragments
  (the weight rows are the M operand), activations int8 with an fp32 scale per 32, the int32 result converted and scaled per k16 (IQ2_XS /
  IQ1_M change their sub-scale every 16 weights; IQ1_M's +-1/8 offsets are folded in as `8 * grid +- 1`). `AMP_GGUF_PF_WMMA=1`: the fp16
  version (weights dequantized to shared memory tiles, WMMA); `AMP_GGUF_NO_PF=1`: the original per-expert dequantize + cuBLAS loop. 8 layers,
  1024 tokens: 102 ms per layer-chunk (cuBLAS loop) -> 85 (WMMA) -> 59 (int8); both kernels match the exact reference to 5e-4 (WMMA) /
  1e-2 (int8, the activation rounding), `examples/gguf_expert_check.uc`.
- llama.cpp (as of 2026-10-25) has DeepSeek-V4.1 (`deepseek41`, #31086) with its own converter (Engram tables inside the GGUF
  as one BF16 `per_layer_token_embd`), which is not the layout of the ISTA files (separate Engram GGUF): they do not load there.
  Its newer MMVQ for low-bit types (#30517, two rows per block for IQ1_M and friends) is the same idea as the 8 rows per block here.

## Validation against the EXL3 / FP8 pack of the same model (jai)

| Check | Result |
|---|---|
| `examples/gguf_check_dense.uc`: dequantized dense matrices (5 layers: `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, shared experts) against the pack's block-scaled FP8 | cosine 0.99999 (Q8_0), 0.9990 - 0.9994 (Q5_K) |
| `examples/gguf_check_plain.uc`: 133 plain tensors (router, bias, sinks, norms, mHC, compressors, indexer, Engram q / k) of 8 layers | bit-identical |
| `examples/gguf_expert_check.uc`: single experts, GGUF against the EXL3 pack, same input (layers 0, 7) | cosine 0.26 - 0.69 (two ~1.5 bpw quantizations of one expert, random input), a different expert 0.00 - 0.02; the GGUF kernels against their own dequantized matrices 1.0000; prefill path against decode path 6e-4 |
| Engram: IQ4_NL rows of the GGUF against the pack's FP8 rows (7 rows of both tables) | cosine 0.997; hash tables identical |
| `examples/gguf_vs_exl3.uc`: 4 layers, 96 tokens | hidden-state cosine 0.94, logits 0.92 |
