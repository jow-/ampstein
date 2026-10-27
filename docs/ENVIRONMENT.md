# Environment variables

Everything in Ampstein that is steered by an environment variable, grouped by
what it affects. **All project variables carry the `AMP_` prefix.** The vendored `EXL3_*` and the
PyTorch allocator variables keep their names. Paths are relative to the repo root.

## Conventions

* **"Set" means set.** Most switches are tested as *truthy / not null*: ucode treats
  any non-empty string as true, so `FOO=0` **enables** `FOO`. To turn a switch off,
  `unset FOO` (or `env -u FOO`). The exceptions are called out per variable:
  `== '1'` (only the literal `1`), `!= '0'` (on unless literally `0`), numbers.
* Variables are read **once, at module load** (ucode `const`) or at model
  load / graph-build time. Changing them needs a server restart.
* "Debug / instrumentation" variables cost throughput and are meant for
  reproduction runs, never for production servers.
* Variables marked **(opt-out)** disable an optimisation that is on by default;
  they exist to A/B correctness or isolate a bug.

Quick index of the three diagnostic variables added while chasing the
`4444…` repetition bug (details in [Diagnostics](#diagnostics-nan--non-finite-hunting)):

| variable | model | one-line effect |
|---|---|---|
| `AMP_ENGINE_PROBE` | any | after every request, scan all persistent cache tensors for NaN/inf and log `scan: …` |
| `AMP_DSV4_PROBE` | DeepSeek-V4 | log fp32-stream / fp16-collapse peak magnitudes per request (`probe: …`) |
| `AMP_GDN_TRACE` | Qwen3.8-Flash-Next (qwen4) | per-op non-finite tracing of GDN, HC mix and QSA attention during prefill |

---

## Diagnostics: NaN / non-finite hunting

These three were added together. All are **no-ops when unset**. They are what
located the `qsa_attn` empty-split bug (see [LESSONS.md](../../LESSONS.md) /
the kernel fix in `src/torch/amp_kernels.cu`).

### `AMP_ENGINE_PROBE`

| | |
|---|---|
| Read at | `ampstein` / `core.uc` (per request, in the request-summary logger) |
| Applies to | every model; implemented by `engine.scan()` in `src/ucode/ampstein/engine.uc` |
| Truthiness | set (any value, including `0`) |
| Cost | ~270 ms per request at `--ctx 262144` on 6 GPUs (up to ~470 ms with the draft cache); a device sync per cache tensor |

After each request the server walks `cache.layers`, `cache.mtp` and the PLE conv
tail and checks each tensor for non-finite values (`x - x` is NaN iff `x` is
inf/NaN, then `max().item()`). Bad tensors are bisected for the first bad row.
One line is logged just before the `chat: prompt …` summary:

```
scan: clean (275 ms)
scan: layers[8][0].rec_state: first-bad-row 0/24; layers[11][1].k_cache: first-bad-row 125675/262144; … (343 ms)
```

How to read it:

* `layers[i][g]` is layer `i`, tensor-parallel rank `g` (`mtp.layers[0][g]` is the
  MTP draft block). `first-bad-row N/total` is the first non-finite row of the tensor
  (`k_cache` / `v_cache` / `raw_k` rows are token positions; `pooled` rows are
  4-token blocks; `rec_state` is not row-indexed, `0/…` just means "bad").
* The **lowest layer** with a bad `rec_state` / first bad KV row is the one the
  NaN was born in or downstream of; the first bad KV row is the token position.
* `spec_states` / `spec_conv[*]` are the speculative-verify rollback buffers; they
  go bad as a *consequence* once decoding runs on a poisoned state.
* A `clean` scan after a request that produced garbage means the corruption was
  in transient buffers (workspace, graph scratch), not in the persistent cache.

### `AMP_DSV4_PROBE`

| | |
|---|---|
| Read at | `src/ucode/ampstein/dsv4.uc` (`PROBE`), `ampstein` / `core.uc` |
| Applies to | DeepSeek-V4 path only (`hc_collapse`); prints zeros on other models |
| Truthiness | set (any value) |
| Cost | two device-side `max` reductions per hyper-connection collapse; no host sync until the per-request report. Graph-capture safe. |

Tracks, on the device, the peak magnitude of the fp32 residual streams and of the
fp16 HC collapse **before** the ±65504 saturation, and resets them at each report:

```
probe: stream-peak 73697.3 collapse-peak 14040
probe: stream-peak nan collapse-peak 70012 NONFINITE-OR-OVERFLOW
```

* `stream-peak` of ~7.3e4 is normal: that is the BOS row's massive activation, held
  in fp32 (a prompt without BOS peaks around 1.3e3).
* `collapse-peak` is what gets narrowed to fp16. Healthy runs sit at ~1.4e4,
  i.e. >4× below 65504. `NONFINITE-OR-OVERFLOW` means the unsaturated collapse
  exceeded 65503 or was NaN — the DSV4 fp16-overflow failure class (LESSONS L105d).
* Not applicable to Qwen: on that model the line prints `stream-peak 0 collapse-peak 0`.

### `AMP_GDN_TRACE`

| | |
|---|---|
| Read at | `src/ucode/ampstein/qwen4.uc` (`AMP_GDN_TRACE`), `src/ucode/ampstein/qwen4_tp.uc` (`TRACE`) |
| Applies to | Qwen3.8-Flash-Next (qwen4) only |
| Truthiness | set (any value) |
| Cost | **large** — many `.item()` syncs per layer. Prefill drops from ~2100 to ~1300 tok/s at 125k context. Decode is untouched. |

Only active for prefill-sized steps (`S >= 8`) because CUDA-graph-captured step
sizes (1 and the MTP window, ≤ 4) must not synchronise. Do not combine with an
`--mtp` window ≥ 8. Three kinds of output:

| log prefix | where | meaning |
|---|---|---|
| `gdn-trace: call N S=… non-finite: x,qkv,…` | `qwen4.gdn_forward` | after each GDN layer call, which intermediates (`x` input, raw `qkv`, gates `g0/g1`, conv output `q3/k3/v3`, `core`, `rec`/`conv` state) are non-finite. If `x` is already listed the NaN came from *before* this layer. First 40 lines. |
| `trace: layer L attn\|mlp S=… stream-max … mixed-max …[ MIXED-NONFINITE]` | `qwen4_tp.row_a` | peak of the fp32 streams and of the fp16 mixed sublayer input at each HC-mix site; printed only when a peak is > 20000 or non-finite. First 400 lines. |
| `qsa-trace: dev … pos … S … slab r0 … R … nb … \| out bad \| in: q … qi … k_cache … v_cache … pooled …` | `qwen4.qsa_attend` | a QSA attention slab produced a non-finite output; lists which inputs were non-finite, so `out bad` with every input `ok` points at the kernel itself. `R` is the slab's row count. |

Typical workflow (this is how the empty-split bug was found): run with
`AMP_ENGINE_PROBE=1` to learn *which layer and token position*, then
`AMP_GDN_TRACE=1` to learn *which op*, then read the slab line.

---

## Server / engine (`ampstein`, `core.uc`, `engine.uc`)

| variable | default | effect |
|---|---|---|
| `AMP_GLM5_NO_GRAPHS` | unset | GLM-5 and Qwen4 only: disable CUDA-graph decode (graphs are on by default). Server flag equivalent for graphs is internal (`graphs: false`). Use to bisect graph-capture issues. |
| `AMP_VIS_EMB` | unset | Debug: path to a safetensors file with an `emb` tensor that **replaces the vision tower output**, for A/B of the text side. Never set in service. |
| `AMP_MTP_TIMING` | unset | Engine: per-phase MTP-round timings, synchronising at phase boundaries, printed per request. Skews throughput. |
| `AMP_TAIL_GRAPH_MAX` | `12` | Engine: a prefill of at most this many new tokens (a short continuation of the live state or a snapshot) replays the captured decode / verify step graphs instead of the eager layer loop (`graph_runner`, L135). `0` disables. Not used with DFlash / DSpark, image prompts, or models with state outside the layer caches (DeepSeek-V4, Qwen3.8-Flash-Next). |
| `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF` | engine sets `expandable_segments:True` | `src/torch/amp_torch.cpp` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at module init **unless `PYTORCH_ALLOC_CONF` is set**; an explicit value wins. The launch scripts export it too. Fragmentation-free growth matters at 256k context. |

## Native kernels / comms (`src/torch/amp_torch.cpp`)

| variable | default | effect |
|---|---|---|
| `AMP_COMM_BLOCK_DIV` | `16384` | P2P all-reduce: one CUDA block per ~`N` values per chunk (clamped to `AMP_COMM_MAXB`). Lower = more blocks. Must be identical on all ranks of a run. |
| `AMP_GEMM_F16ACC` | on | EXL3 prefill GEMM (`recon_mm`) runs cuBLAS with `CUBLAS_COMPUTE_16F`, i.e. **fp16 accumulation** (~2× tensor-core rate on GeForce Ampere, +8.5% prefill). Set to exactly `0` to fall back to ATen `mm` with fp32 accumulation — the first thing to try if activations may overflow fp16 partial sums (LESSONS L61). Only the literal `0` disables. |
| `AMP_RECON_FUSED_ROWS` | `2048` | Weight-reconstruction strategy by row count: at `rows >= N` the Hadamard-folded weight is reconstructed once (`reconstruct_had_slice`) and multiplied; below it the activation is Hadamard-rotated and the plain weight reconstructed (`had_r_128` → `reconstruct` → mm → `had_r_128`). Numeric. |

## DeepSeek-V4 (`src/ucode/ampstein/dsv4.uc`, `src/ucode/ampstein/dsv4_tp.uc`, `src/ucode/ampstein/dspark_tp.uc`)

| variable | default | effect |
|---|---|---|
| `AMP_DSV4_NO_FUSED` **(opt-out)** | unset | Disable all fused-op builders (`build_fans`, `build_gu`) and use the composite path. Correctness A/B. |
| `AMP_DSV4_NO_FAN` **(opt-out)** | unset | Disable only the fused fan-out / shared-expert gate+up projections. |
| `AMP_DSV4_OGRAPH` | unset | Opt-in: CUDA-graph the attention output projection. |
| `AMP_DSV4_NO_SIDE_HO` **(opt-out)** | unset | Stage hand-off copies stay on the compute stream instead of a side stream of the source device (the side-stream path is +8% layer-split prefill). |
| `AMP_DSV4_PF32` **(opt-out)** | unset | Turn off the fused fp16 tensor-core prefill attention / device-twin compressor state and use the fp32 path. Slower; precision reference. |
| `AMP_DSV4_SYNC_MOE` **(opt-out)** | unset | Run the MoE synchronously instead of the async begin/finish split. |
| `AMP_MOE_ATOMIC` **(opt-out)** | unset | Sum the routed experts' outputs with atomic adds in arrival order (the kernel's original mode) instead of the default fixed-order slot gather (`torch.moe_fused` scratch + `torch.moe_gather`). The atomic sum rounds differently run to run, which the depth of the network amplifies to differences of ~20 at the massive-activation rows and flips near-tie tokens; the deterministic default costs ~1% of the prefill speed (129k tokens: 1531 -> 1514 tok/s). Shared by DeepSeek-V4, GLM-5 and Qwen (`glm5.moe_begin`). |
| `AMP_DSV4_NO_PAR` **(opt-out)** | unset | TP decode graphs: do not fork the shared expert onto side streams during capture. |
| `AMP_DSV4_IDX_COMPOSITE` **(opt-out)** | unset | Indexer scoring through the ATen composite instead of the fused kernel. The composite materialises `(R, Hi, ne)` fp32 = **2 GiB at 256k** — do not use at long context. |
| `AMP_DSV4_KV_RING` | `8192` | Raw-KV ring size in rows (the 256k-context fix). `0` keeps the full-length KV (OOMs at 256k on 24 GB cards). Snapshots save the window rows. |
| `AMP_DSV4_HOSTPROF` | unset | Enable host-side profile sections (shared with `glm5.uc`); counters are dumped by the bench scripts. |
| `AMP_DSPARK_NO_FP8KV` **(opt-out)** | unset | Disable e4m3 fake-quantisation of the DSpark drafter's KV rows (the drafter was trained with it; disabling reduces draft acceptance). |
| `AMP_DSPARK_CONF` | `0.3` | DSpark drafter confidence threshold (float): a draft is kept while its confidence is `>=` this value (up to the window). Overridden by `opts.threshold`. Raise for fewer, safer drafts. |
| `AMP_DSV4_PROBE` | unset | See [Diagnostics](#dsv4_probe). |

## Qwen3.8-Flash-Next (`src/ucode/ampstein/qwen4.uc`, `src/ucode/ampstein/qwen4_tp.uc`, `src/ucode/ampstein/qwen35.uc`)

| variable | default | effect |
|---|---|---|
| `AMP_QWEN4_NO_FUSED_MIX` | off (only `1` enables) | Use the composite HC mixer instead of the fused `torch.gr_mix` kernel. |
| `AMP_QWEN4_QSA_REF` | off (only `1` enables) | Sparse attention via the composite ATen reference (`qsa_gather_attend_ref`) instead of the fused `torch.qsa_attn` kernel. **Useful to confirm or bypass kernel bugs** (it avoids the split-K workspace entirely). |
| `AMP_QWEN4_MTP_TAP` | on (only `0` disables) | MTP input: per-stream stack (reference default) vs averaged. |
| `AMP_QWEN4_NO_MTP_GRAPH` | unset | Do not CUDA-graph the MTP draft step. |
| `AMP_GGUF_QX` | unset (CUDA: off; ROCm: `dense`) | `dense`, `experts`, `all` or `off`: the int8 GEMV blocks quantize their own activation row instead of a separate `gguf_quant_x` launch. |
| `AMP_ATTN_SCALAR` / `AMP_NO_WMMA_PREFILL` / `AMP_WMMA_SPLITS` | unset (ROCm) | `AMP_ATTN_SCALAR=1`: the scalar decode attention instead of the WMMA kernel; `AMP_NO_WMMA_PREFILL=1`: ATen FlashAttention for prefill; `AMP_WMMA_SPLITS=N`: key splits per kv head of the WMMA decode kernel. |
| `AMP_QWEN4_KV` | unset | `q8`: the QSA K / V caches as int8 + scales without the server flag (benchmarks, tests; `--cache-mode Q8` sets it). |
| `AMP_QWEN4_NO_PAR` | unset | TP decode graphs: no forked stream for the shared expert. |
| `AMP_QWEN4_NO_DSEL` | unset | Decode token selection through the composite ATen chain instead of `idx_score` + `dsa_select`. |
| `AMP_QWEN4_PLE_AHEAD` | `3` | TP prefill: chunks ahead of the wavefront whose PLE rows are read from disk. |
| `AMP_MTP_NO_CHAIN` | unset | Engine: draft step by step with a host read after each (instead of the device-side chain, TP Qwen4). |
| `AMP_GDN_CHUNK` | off (only `1` enables) | Block-parallel GDN scan from chunk length 64 up (`qwen35.uc`). Experimental, opt-in. |
| `AMP_GDN_NO_ABG` **(opt-out)** | unset | Qwen3.5/3.8: compute the fp16 `in_proj_a` / `in_proj_b` products and the gates as separate launches for decode / verify rows instead of `torch.gdn_ab_gates`. |
| `AMP_NO_MLP_BUNDLE` **(opt-out)** | unset | Qwen3.5/3.8: run gate and up as two GEMMs for decode / verify rows instead of one `mgemm` launch. |
| `AMP_TP_PF_ROWS` | `--chunk` | Qwen3.5/3.8 TP: rows of the second P2P communicator used for prefill all-reduces (payloads above it take the peer-copy path). `0` disables it. ~`(2 * ranks + 2) * rows * hidden * 2` bytes per rank. |
| `AMP_DFLASH_TRACE` | unset | `--dflash`: print the absmax / mean of the drafter's tensors per stage (every round, synchronising; debugging only). |
| `AMP_GDN_SCAN_CFG` | `0` (default shape) | GDN scan kernel shape, for experiments: `0` = register-blocked (`11`), `10`-`13` = other register-blocked shapes, `VT*100+RS` (e.g. `3204`, `6401`) = the per-column form. Read once at module load. |
| `AMP_GDN_TRACE` | unset | See [Diagnostics](#gdn_trace). |

## Gemma 4 (`src/ucode/ampstein/gemma4.uc`, `src/ucode/ampstein/gemma4_tp.uc`, `src/ucode/ampstein/gemma4_vision.uc`)

| variable | default | effect |
|---|---|---|
| `AMP_GEMMA4_NO_GRAPHS` **(opt-out)** | unset | Eager decode instead of the bucketed CUDA graphs. |
| `AMP_GEMMA4_NO_SDPA` **(opt-out)** | unset | Sliding-layer prefill through the gathered `qsa_attn` kernel instead of fused sdpa with a window mask (also disables the bidirectional image mask). |
| `AMP_GEMMA4_SYNC_MOE` **(opt-out)** | unset | MoE with the host read of the per-expert counts instead of the readback-free fused path. |
| `AMP_GEMMA4_HP`, `AMP_GEMMA4_HPSYNC` | unset | Host seconds per phase of a layer (`gemma4.HP`, printed by `examples/gemma4_gen.uc`); with `HPSYNC` every phase ends in a device synchronisation (so it is the GPU time). |
| `AMP_GEMMA4_GPUS`, `AMP_GEMMA4_TP`, `AMP_GEMMA4_PIPE`, `AMP_GEMMA4_GRAPH`, `AMP_GEMMA4_TWICE`, `AMP_GEMMA4_CHUNK`, `AMP_GEMMA4_FILE`, `AMP_GEMMA4_RAW`, `AMP_GEMMA4_NOSTOP` | `examples/gemma4_gen.uc` | Bring-up script: GPU count, tensor parallel, wavefront prefill, graph decode, a warm-up pass before the timed one, chunk, prompt file, raw prompt, ignore the end-of-turn token. |
| `AMP_MODEL_GEMMA4` | `tests/gemma4_tokenizer.uc`, `tests/engine_store_gemma4.uc` | Model directory (`/data/models/gemma-4-26B-A4B-it-exl3-6.10bpw`). `AMP_GEMMA4_GPUS` / `AMP_GEMMA4_TP=1` choose the layout of the store test. |

## GLM-5 (`src/ucode/ampstein/glm5.uc`, `src/ucode/ampstein/glm5_tp.uc`)

| variable | default | effect |
|---|---|---|
| `AMP_GLM5_NO_SKINNY` **(opt-out)** | unset | Use cuBLAS for ≤ 8-row matmuls instead of the `skinny` kernel. |
| `AMP_GLM5_NO_IDX_KERNEL` **(opt-out)** | unset | Composite ATen indexer scoring instead of the fused 32-head × 128-dim kernel. |
| `AMP_GLM5_NO_COOP` **(opt-out)** | unset | Disable the cooperative MoE path. |
| `AMP_GLM5_NO_SORTED` **(opt-out)** | unset | Disable the sorted-routing MoE path (when `coop` is on). |
| `AMP_GLM5_NO_MTP_GRAPH` **(opt-out)** | unset | Do not graph the MTP step. |
| `AMP_GLM5_NO_PAR` **(opt-out)** | unset | Do not fork side streams during graph capture. |
| `AMP_GLM5_NO_SHARED_POOL` **(opt-out)** | unset | Each graph gets its own memory pool instead of sharing one (more VRAM). |
| `AMP_GLM5_TP_COPY_AR` | unset | Use the copy-based all-reduce for prefill instead of a second P2P communicator (smaller VRAM, slower). |
| `AMP_GLM5_TP_OVERLAP` | unset | Experimental double-buffered prefill (measured slower; LESSONS.md L94). |
| `AMP_GLM5_TP_FAKE_AR` | unset | **Wrong numerics.** Skip the all-reduce entirely, timing experiments only. |

## Vendored exllamav3 extension (`vendor/exl3/exllamav3_ext/**`)

These belong to the vendored kernel library; semantics are as read from the
call sites (`file:line` is the read). Names are stable upstream interfaces.

| variable | read at | effect |
|---|---|---|
| `EXL3_TP_NO_FP16_WIRE` | `parallel/all_reduce_cpu.cu:565` | Disable the fp16 wire format of the CPU TP all-reduce (non-`0` value). |
| `EXL3_TP_TRACE_WIRE` | `parallel/all_reduce_cpu.cu:570` | Trace the all-reduce wire format (non-`0`). |
| `EXL3_TP_REDUCE_THREADS` | `parallel/all_reduce_cpu_avx512.cpp:283` | Number of CPU reduce threads (int, `0` = auto). |
| `EXL3_MOE_HANDOFF_PROF` | `cpu/moe_handoff.cu:271` | MoE GPU↔CPU hand-off profiling output. |
| `EXL3_MOE_POOL_PRIME` | `cpu/moe_handoff.cu:277` | Prime the MoE worker pool at startup. |
| `EXL3_MOE_CPU_MAX_ISA` | `cpu/moe_mul1.cpp:1537` | Cap the CPU MoE kernel ISA level. |
| `EXL3_MOE_CPU_PIN` | `cpu/moe_mul1.cpp:1793` | CPU affinity for the MoE workers. |
| `EXL3_MOE_CPU_SMALL_WORKERS` | `cpu/moe_mul1.cpp:2557` | Worker count for small CPU MoE batches. |
| `EXL3_NGRAM_GATHER_PROF` | `ngram_gather_win.cpp:61` | Profile the n-gram table gather. |
| `EXL3_INT8_GEMV`, `EXL3_INT8_GEMV_MAX_K` | `quant/exl3_gemv_int8.cu:27,48` | Enable / bound the int8 GEMV path. |
| `EXL3_MOE_COOP_KSPLIT`, `EXL3_MOE_COOP_WIDE`, `EXL3_MOE_COOP_DBG` | `quant/exl3_moe_coop.cu:81,94,122` | Cooperative MoE tuning / debug. |
| `EXL3_MOE_TILE_N` | `quant/exl3_moe.cu:30` | MoE tile width override. |
| `EXL3_GEMV`, `EXL3_GEMV_SMEM` | `quant/exl3_gemv.cu:31,39` | GEMV path selection / shared-memory size. |
| `EXL3_HGEMM_F16ACC` | `hgemm_f16acc.cu:400` | fp16-accumulate hgemm switch. |
| `EXL3_DSA_TOPK_SPLIT` | `dsa_topk.cu:713` | Split top-k (on unless `0`). |
| `EXL3_QT_OPTIMIZED` | `quant/quantize.cu:96` | Quantiser-side optimised path (offline tools). |
| `EXL3_BCM_DEBUG`, `EXL3_STLOADER_DEBUG` | `libtorch/mla_attention.cpp:330`, `stloader.cpp:384` | Debug logging. |
| `EXLLAMAV3_TUNE_CACHE` | `quant/coop_autotune.cu:77` | Override the autotune cache file path (else `XDG_CACHE_HOME`/`HOME`). |

## Models, devices and paths (benches and tests)

None of these affect the server (which takes `--model`); they select inputs for
`examples/bench_*.uc` and `tests/*.uc`.

| variable | used by | effect |
|---|---|---|
| `AMP_MODEL_QWEN35`, `_QWEN4`, `_GLM5`, `_DSV4`, `_1B` | benches, tests | Model directory for the matching family (defaults point to `/data/models/…`). |
| `AMP_TEST_DEV` | `tests/kv_q8.uc`, `tests/idx_score.uc` | CUDA device for the test. |
| `AMP_TP_DEVICES`, `AMP_TP_F0` | `tests/qwen35_tp.uc` | TP device list (`0,1`) / first-rank fraction. |
| `AMP_GLM5_GPUS`, `AMP_GLM5_STAGES`, `AMP_GLM5_DEV`, `AMP_GLM5_KV_BITS`, `AMP_GLM5_FULL`, `AMP_GLM5_MOE_LAYER`, `AMP_GLM5_VIS_DEV` | `tests/glm5_*.uc` | GPU count, pipeline stages, device, KV bit width, chunk, full-model flag, layer, profile length. |
| `AMP_DSV4_LAYERS`, `AMP_DSV4_DEVICES`, `AMP_DSV4_TP`, `AMP_DSV4_TP_DEVICES`, `AMP_DSV4_TP_STAGES`, `AMP_DSV4_CHUNK`, `AMP_DSV4_CONT`, `AMP_DSV4_NOTRACE`, `AMP_DSV4_FIX`, `AMP_DSV4_VFIX` | `bench_dsv4.uc`, `tests/dsv4_*.uc` | Layers to load, device count, TP layout, chunk size, oracle fixture / layer selection. |
| `AMP_BENCH_N`, `AMP_BENCH_CHUNK`, `AMP_BENCH_DEC`, `AMP_BENCH_NOEAGER`, `AMP_BENCH_VERIFY` | `examples/bench_dsv4.uc` | Prompt length, chunk, decode steps, skip eager, verify-window size. |
| `AMP_QWEN4_FIX`, `AMP_QWEN4_LAYERS`, `AMP_QWEN4_TP_DEVS`, `AMP_QWEN4_TP_STAGES`, `AMP_QWEN4_TRACE`, `AMP_QWEN4_GRAPH`, `AMP_QWEN4_PIPE`, `AMP_QWEN4_CHUNK`, `AMP_QWEN4_DEPTH`, `AMP_QWEN4_DEPTHS`, `AMP_QWEN4_DEPTH_FIX`, `AMP_QWEN4_BUDGET`, `AMP_QWEN4_STEP`, `AMP_QWEN4_EAGER`, `AMP_QWEN4_PROBE_FIX`, `AMP_QWEN4_VIS_FIX`, `AMP_QWEN4_VIS_IMG`, `AMP_QWEN4_VIS_DEV` | `tests/qwen4_*.uc` | Oracle fixture, layer count, TP layout, trace / graph toggles (`0` disables), wavefront prefill, depth sweep, indexer budget. |
| `AMP_VERIFY_S`, `AMP_VERIFY_N`, `AMP_TP_N`, `AMP_TP_STEPS`, `AMP_STORE_GPUS`, `AMP_STORE_Q_WORDS`, `AMP_QSA_DEV`, `AMP_SLICE_GEN`, `AMP_SLICE_TOK`, `AMP_STORE_DEVICES`, `AMP_STORE_STAGES` | `tests/*.uc` | Sizes / steps / device for the respective test. |
| `CONT`, `DRY` | `tests/oracle/*.py` | Oracle generators: continuation length / dry run. |

---

## Notes

* The attention-kernel fix that the diagnostics located is **not** a variable:
  zero-filled accumulators for empty splits in `amp_qsa_attn_kernel` and
  `amp_attn_dec_mma_kernel` (`src/torch/amp_kernels.cu`).

## DeepSeek-V4.1

| variable | effect |
|---|---|
| `AMP_DSV41_RING` | raw window rows kept per layer (rollback span), default 8192 |
| `AMP_DSV41_NO_COOP` | do not build the fused mixed-width MoE state (composite experts; slow) |
| `AMP_DSV41_NO_ENGRAM` | Engram contributes zeros (tests without the 200 GB tables) |
| `AMP_DSV41_PROF` | accumulate the time of the phases (synchronizes between them; `dsv41.PROFILE`) |
| `AMP_DSV41_HOSTPROF` | host time per layer of a prefill chunk (`dsv41.HOSTPROF`) |
| `AMP_DSV41_ROWLOOP` | multi-row (verify) decode: attention row by row instead of batched (reference path) |
| `AMP_DSV41_SKIP` | profiling: `attn` skips the attention of multi-row steps (wrong outputs) |
| `AMP_DSPARK_PROF` | print the drafter round phases (draft / verify / update, synchronizes) every 50 rounds |
| `AMP_DSPARK_NO_GRAPH` | draft eagerly instead of through CUDA graphs |
| `AMP_DSPARK_NO_PREFILL_TAPS` | do not feed the drafter during prefill (it then needs a plain step per window) |
| `AMP_DSPARK_COSTS` | ms of a 1..6-row verify step for the choice of the draft length (default `36.8,45.2,54.2,61.3,67.1,73.0`) |
| `AMP_DSPARK_DRAFT_MS` | drafter time added to every round's cost estimate, default 8 |
| `AMP_GGUF_RPW1` | unset | GGUF kernels: one output row per warp instead of two (the experts and single-row dense products default to two). |
| `AMP_GGUF_NO_SHARED_PAR` | unset | GGUF checkpoints: the shared expert runs on the compute stream instead of a forked side stream during graph capture. |
| `AMP_GGUF_GROUP` | unset | GGUF checkpoints: verify steps (2 - 8 rows) group the picks by expert (`torch.gguf_moe_group`) so an expert is read once for all its rows; slower on random picks, so off. |
| `AMP_DSPARK_COSTS` | per model | the drafter's verify cost table, ms for 1 .. 6 rows (GGUF default `23.1,36.6,36.6,51.4,51.4,66.0`). |
| `AMP_MOE_ATOMIC` (GGUF prefill too) | unset | the grouped MoE prefill kernels add the experts' outputs atomically (arbitrary order, not reproducible) instead of the fixed-order gather. |
| `AMP_GC_INTERVAL` | `50000` | allocations between ucode garbage collections in the server (ucode's default is 1000). |
| `AMP_DSV41_ROWPROF` | unset | prints the host time of the verify-step prelude (position metadata, token hashing, Engram start, embedding). |
| `AMP_ENGINE_PROF` | unset | `step_fwd` prints the host enqueue time against the step period. |
| `AMP_GGUF_RESERVE_MIB` | `1200` | GGUF checkpoints, single-chain DeepSeek-V4.1: the memory the layer plan keeps free on every GPU after weights, kv pools, mirrors and drafter (graphs ~0.35 GiB, a prefill ~0.35 GiB, margin). Above what the six GPUs can give, the start warns `planned ... over its budget`. |
| `AMP_DSV41_NO_REBALANCE` | unset | GGUF: keep the contiguous layer plan of `plan_devices` (no layers moved between GPUs, the drafter stages where it put them). |
| `AMP_DSV41_GRAPH_BUCKETS` | all (GGUF: `1`) | Decode graphs: capture only the n largest context buckets (4k, 16k, 64k, ... `max_seq`); each costs ~0.5 GiB of graph memory over the GPUs and the pool is not shared. |
| `AMP_DSV41_ROW_SIZES` | all (GGUF: `3,5`) | Verify (multi-row) graph sizes to capture, plus the largest requested; a size without a graph runs the next larger one and takes the extra rows back. |
| `AMP_DSV41_IDX_SLAB` | `33554432` (GGUF: `4194304`) | Score elements of one tile of the prefill indexer (rows x pool entries); its `topk` needs a few times that as workspace. |
| `AMP_NO_GRAPHS` | unset | Run the decode steps eagerly instead of through captured CUDA graphs (any model; the GLM / Qwen3.8-Flash-Next specific `AMP_GLM5_NO_GRAPHS` still works). |
| `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | `0` (ROCm builds) | HIP runtime: graph packet capture leaks and crashes a captured step after ~10000 launched nodes; `torch.so` sets it to `0` unless the environment says otherwise (`docs/ROCM.md`). |
| `AMP_GGUF_SLICE` | `8388608` | GGUF: elements of one dequantized fp16 slice of a dense prefill matmul (16 MiB). |
| `AMP_GGUF_EMBED_FP16` | unset | GGUF: dequantize the embedding table to fp16 on the GPU (1.3 GiB) instead of keeping Q5_K and dequantizing rows by id (0.42 GiB). |
| `AMP_GGUF_FP16` | unset | GGUF checkpoints: the matrix-vector kernels with fp16 activations and exactly dequantized weights instead of the int8 `dp4a` kernels (about 3 - 8x slower; also switches the grouped / fused launches off). |
| `AMP_GGUF_NO_GROUPED` **(opt-out)** | unset | GGUF: `wo_a` as one launch per group instead of one launch. |
| `AMP_GGUF_NO_FUSED` **(opt-out)** | unset | GGUF: gate, up and SwiGLU of the experts / the shared expert as separate launches. |
| `AMP_GGUF_NO_PF` **(opt-out)** | unset | GGUF: prefill of the routed experts through per-expert dequantization + cuBLAS instead of the grouped tensor-core kernel. |
| `AMP_GGUF_PF_WMMA` | unset | GGUF: the grouped prefill kernel on fp16 WMMA (weights dequantized into shared memory) instead of int8 `mma.sync`. |
| `AMP_DSV41_PROMPT_SKIP` **(opt-out)** | unset | DeepSeek-V4.1 `--tp` prefill: `0` runs every layer for every prompt position (default: the prompt skip, `docs/DSV41.md`). Also the tunable `prompt_skip`. |
| `AMP_DSV41_KV` | compressed pools (rows + indexer keys): `q8` (default: int8 codes + a fp16 scale per 32), `q8pool` (rows only), `fp16` |
| `AMP_V41_NO_TC` | use the CUDA-core attention kernel instead of the tensor-core one for prefill chunks |
| `AMP_DEBUG_PROMPT` | log the rendered prompt text of every chat request (all architectures) |
