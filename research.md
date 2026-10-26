# CPU / host-memory optimization survey

Survey of open PRs in llama.cpp, Strata LLM (`Niko1221/Strata`) and vLLM that
are relevant to Ampstein's Qwen4 GGUF path: 256k context, layer-split `qwen4`,
resident experts in VRAM, cold experts mapped/pinned in host RAM, PLE n-gram
table streamed from disk, and optional MTP.

Most of these are not CPU instruction kernels. For Ampstein the useful ones are
host-memory placement, PCIe/DMA staging, page-cache management, graph
scheduling, CPU affinity, and quantized CPU kernels. The GPU kernels remain
CUDA; CPU fallbacks are only a comparison point.

## llama.cpp

| PR | Status | Technique | Relevance |
|---|---|---|---|
| [#29963 ggml: pipeline parallelism with MoE experts in host RAM](https://github.com/ggml-org/llama.cpp/pull/29963) | open | Keeps pipeline parallelism when MoE experts are in host RAM. Each GPU uploads its own layers' experts; consecutive ubatches overlap across devices. Uploads each graph input once per device, async on the split stream. Keeps graph topology constant and reserves the worst-case graph after a smaller graph. Uses one host-weight copy per device, reuses it between splits, and runs host-weight ops on the device of their own layer. For large batches it uploads the whole expert tensor instead of waiting for MoE ids. | Directly relevant to layer-split Qwen4 with cold host experts. Could improve PP by overlapping ubatches and avoiding per-device duplicate host copies. The "upload whole expert tensor for large batches" rule is a practical alternative to per-token host reads. |
| [#27986 ggml-cpu: add mirror NUMA strategy](https://github.com/ggml-org/llama.cpp/pull/27986) | draft | Replicates CPU weights on every NUMA node so CPU GEMM reads are node-local. Explains that first-touch NUMA and CUDA pinned host memory are mutually exclusive: pinned pages are faulted by the loader and cannot migrate. A pinned primary plus per-node replicas gives both DMA-speed prefill and NUMA-local decode. Measured pinned distribute: 125.1 prefill / 11.5 decode; pinned mirror: 128.6 / 18.2. | Very relevant if `j7` or another host has multiple NUMA nodes. On a single-node host the mirror is unnecessary, but the pinned-vs-first-touch trade-off is the key lesson. |
| [#25294 llama: stream MoE routed experts from disk](https://github.com/ggml-org/llama.cpp/pull/25294) | open | Routed experts stay on disk; each streamed layer keeps a small device-side cache of expert slots filled on demand. A CPU id-remap op runs after router top-k. Missing experts load through a pread thread pool. Eviction uses decaying route hotness with an LRU tiebreak. Options include `--moe-stream`, cache budget, IO threads, and `O_DIRECT` reads with a buffered fallback. Prefill waves handle ubatches touching more experts than the cache holds. A pinned host mirror variant serves misses at PCIe speed. | Relevant to disk-backed cold experts and to the PLE table. The device-side slot cache, route-hotness eviction, pread pool, and wave prefill are all applicable patterns. `O_DIRECT` is useful when the page cache is not wanted. |
| [#30971 llama: add adaptive single-graph MoE offload](https://github.com/ggml-org/llama.cpp/pull/30971) | draft | Five categories: miscellaneous fusions; merging the graph via waiting kernels so CPU-offloaded experts do not split the graph into disconnected per-layer graphs; timing CPU-GPU data copies; optimizing expert placement; CPU affinity/core utilization. | This is the closest match to Ampstein's decode overhead. Current host cold experts can cause graph splits and launch/sync overhead. The draft is explicitly about beating Strata on decode with host offload. |
| [#30448 ggml-cpu: faster AVX2 dot products for IQ3_S and IQ3_XXS](https://github.com/ggml-org/llama.cpp/pull/30448) | open | Replaces table lookups with register sign computation for IQ3_XXS; builds the IQ3_S grid index in scalar registers instead of SIMD+union; unrolls inner loops. Measured ~23% faster IQ3_S and ~15% faster IQ3_XXS single dot; MoE decode op ~24% / ~5%. | CPU quant kernel technique. Not needed for Ampstein's CUDA expert path, but relevant if a CPU fallback or CPU pool is added. |
| [#26348 ggml-cpu: x86 VNNI Q2_0 dot product](https://github.com/ggml-org/llama.cpp/pull/26348) | open | AVX-VNNI / AVX-512 VNNI implementation of `ggml_vec_dot_q2_0_q8_0`. Measured ~3.0-3.6x PP/TG on Bonsai Q2_0 GGUFs. Notes FMA contraction changes perplexity slightly versus generic. | CPU quant kernel for Q2_0. Relevant only if Q2_0 experts are computed on CPU; Ampstein already uses CUDA. |
| [#27032 CPU Optimizations - Prefill, Tokenization, Token Generation](https://github.com/ggml-org/llama.cpp/pull/27032) | draft | Fuses `GET_ROWS` embedding lookup with positional `ADD`; caches repeated token IDs; uses `__builtin_prefetch` for next embedding row; reduces decode memory bandwidth and buffer allocations. | Relevant to CPU-side embedding/PLE and tokenization overhead. The repeated-token cache and prefetch ideas apply to n-gram/PLE row gathering. |
| [#30723 common: store ngram cache parts as sorted vectors](https://github.com/ggml-org/llama.cpp/pull/30723) | open | Replaces per-ngram inner unordered maps with sorted token/count vectors and branchless binary search. ~1.9x faster drafting and peak cache memory 3558 MB -> 2019 MB for a 541 MB static cache. | Relevant to n-gram lookup and speculative decoding. Compact sorted vectors reduce host RAM and branch mispredictions. |
| [#30196 sampling: read context outputs with one synchronization](https://github.com/ggml-org/llama.cpp/pull/30196) | open | Adds `llama_get_outputs(ctx)` to synchronize, reorder, and prepare batch outputs once; sampling and speculative verification reuse the view. ~6.9% decode throughput in the measured SPEED-Bench case. | Relevant to host/GPU sync overhead in decode and MTP verification. One sync per batch instead of per scalar getter. |
| [#29796 cuda: skip redundant stream synchronization](https://github.com/ggml-org/llama.cpp/pull/29796) | draft | Tracks whether a CUDA stream has pending work and skips `cudaStreamSynchronize` when idle. Measured decode +52.55%, `cudaStreamSynchronize` calls 36,199 -> 12,370. | Very relevant to Ampstein's nsys finding that launch/host-sync overhead dominates. A stream-use flag can avoid redundant waits. |
| [#23762 vulkan: fix UMA performance by preferring cached host memory](https://github.com/ggml-org/llama.cpp/pull/23762) | draft | Prefers cached host memory over write-combining memory for UMA buffers; sets memory-property flags from what the driver actually allocated. | Relevant to host-mapped/UMA memory: write-combining is fast to write but slow to read. Ampstein's mapped cold experts should prefer cached, read-friendly host memory. |
| [#22022 llama-mmap: add MADV_HUGEPAGE hint for THP](https://github.com/ggml-org/llama.cpp/pull/22022) | open | Calls `madvise(MADV_HUGEPAGE)` on read-only mmap model weights. For a 4-5 GB map, ~1M 4 KB pages -> ~2K 2 MB pages, reducing TLB pressure and minor faults. Advisory; skipped for NUMA allocations. Complements HugeTLB. | Relevant to the PLE table mmap and any large host-mapped region. Fewer pages means fewer GPU/CPU TLB misses on random row reads. |
| [#24156 model-loader: reclaim mmap source pages after CPU_REPACK](https://github.com/ggml-org/llama.cpp/pull/24156) | open | After CPU_REPACK copies tensors out of mmap, `madvise(MADV_DONTNEED)` drops dormant file-backed source pages. RSS savings: 13.0 GiB (-42%) on Qwen3-30B-A3B, 5.4 GiB (-34%) on DeepSeek-V2-Lite. Safe because the VMA re-faults identical bytes; skips tensors still read in place. | Relevant to avoiding a second/third copy of model bytes. Ampstein already drops raw dense tensors after permutation; this is the same idea for mmap sources and page cache. |
| [#20596 ggml-cpu: improve `--n-cpu-moe` TG performance](https://github.com/ggml-org/llama.cpp/pull/20596) | open | Special fused gated-activation path for `n_tokens == 1` when experts are on CPU; avoids row mapping; each thread quantizes its activation, removing a barrier. Barrier cost ~30% -> ~7%. Reviewer notes cache-line spacing to avoid false sharing. | Relevant if Ampstein ever computes cold experts on CPU for single-token decode. For GPU host reads, less direct, but the barrier/false-sharing lesson applies to CPU pools. |
| [#28785 ggml-cpu: skip threadpool for graphs with no CPU work](https://github.com/ggml-org/llama.cpp/pull/28785) | open | Avoids creating/waking CPU worker threads when the graph only has views/NOPs; keeps calling thread priority/affinity in sync. Measured TG 88.08 -> 107.09 tok/s on Apple M3. | Relevant to mixed GPU/CPU graphs: do not pay threadpool wake-up when a graph has no CPU ops. |
| [#27333 llama-mmap: fix NUMA path hint inconsistency](https://github.com/ggml-org/llama.cpp/pull/27333) | open | Fixes inconsistent NUMA path hint in mmap. | Relevant to NUMA-aware mmap placement. |
| [#16882 Disable NUMA-specific chunking for high-core-count HPC systems](https://github.com/ggml-org/llama.cpp/pull/16882) | open | Avoids NUMA-specific chunking where it hurts high-core-count systems. | Relevant to CPU thread/chunk tuning on large hosts. |
| [#16548 Implement and use CUDA graph plans](https://github.com/ggml-org/llama.cpp/pull/16548) | open | CUDA graph plans for ggml graphs. | Relevant to graph capture/replay and launch overhead. |
| [#16000 `--numa mirror`: mirror model weights to every NUMA node](https://github.com/ggml-org/llama.cpp/pull/16000) | open | Earlier NUMA mirror PR. | Same idea as #27986; useful context. |
| [#29420 llama-cli: fix broken numa flag](https://github.com/ggml-org/llama.cpp/pull/29420) | open | NUMA CLI flag fix. | Minor; useful if testing NUMA modes. |
| [#29301 llama-bench: fix default thread count on NUMA systems](https://github.com/ggml-org/llama.cpp/pull/29301) | open | Correct default thread count on NUMA systems. | Relevant to benchmark comparability. |
| [#18698 Improving inference speed for the repack buffer type on NUMA architectures](https://github.com/ggml-org/llama.cpp/pull/18698) | open | Repack buffer performance on NUMA. | Relevant if using CPU_REPACK host weights. |
| [#23358 ggml-cpu: Optimized Arm NEON q1_0 dot](https://github.com/ggml-org/llama.cpp/pull/23358) | open | ARM NEON q1_0 dot. | Not relevant on x86 `j7`; useful only for ARM hosts. |

Related discussion: [#12289 GGML_NUMA_MIRROR](https://github.com/ggml-org/llama.cpp/discussions/12289) argues for replicating the model per NUMA node so each node uses local memory bandwidth. The `roblee04/numa_llamacpp` project is also a NUMA buffer-type experiment.

## Strata LLM (`Niko1221/Strata`)

Strata is especially relevant because it targets Qwen3.8-Flash-Next on consumer
hardware with dynamic expert offload between VRAM, RAM, and NVMe.

| PR | Status | Technique | Relevance |
|---|---|---|---|
| [#4016 dsv41: `--numa-experts bands`](https://github.com/Niko1221/Strata/pull/4016) | open | On a 4-NUMA-node host, scatters each 128-row band of every expert matrix across nodes by `(band + expert) % nodes`; one copy of the model in RAM, every node's cores and local memory serve. Per-node pools dispatch tickets by band. Boot-time byte check. Measured 30.2 -> 44.8 tok/s (+48%) on ThreadRipper. | Directly relevant to multi-NUMA hosts. On single-node `j7`, the band-scattering idea still informs how to partition expert rows across worker pools. |
| [#4179 Qwen3.6-35B-A3B on a 4 GB Maxwell card](https://github.com/Niko1221/Strata/pull/4179) | draft | Many host/VRAM techniques: `STRATA_Q36_KV=q8|k8v4` with the first part in VRAM and the rest in mapped pinned host memory, read once per layer through a VRAM bounce buffer; staging slots/shared buffers placed in expert-slot memory; `STRATA_Q36_LEND_HEAD`; `STRATA_Q36_WINDOW` layer-by-layer prompt windows with experts staged once per window; `STRATA_Q36_DENSE_Q8S`; `STRATA_Q36_DRAFT_P`; `STRATA_Q36_MTP_LAST`; `STRATA_Q36_REBAL_*`; CPU pool multi-token i-quant downs. | Very relevant. Host-spilled KV with a bounce buffer, staging buffers in expert slots, prompt windows, and MTP-last-row are all patterns Ampstein can adapt. |
| [#3906 qwen35 hybrid mode: expert placement opt-ins, `--async-swaps`, `--lend-staging`](https://github.com/Niko1221/Strata/pull/3906) | open | Opt-in expert profile/save, decode swaps, `--hybrid-zero-copy` mapped pinned memory, `--expert-prefetch P` fills slots from next layer's router, async slot swaps off the engine stream, and lending prompt-staging buffers as extra expert slots while decoding. Full set +17.3% at 4K / +13.0% at 40K. Notes timing-dependent outputs. | Very relevant to Ampstein's expert placement and host cold experts. Prefetching next-layer hot experts and async swaps are plausible next steps. |
| [#3907 qwen35: CPU expert pool warm-up `--pool-warm`](https://github.com/Niko1221/Strata/pull/3907) | open | Uses idle CPU pool workers between layers to load predicted experts' row slices into their own caches. A warm call of one expert ~9 us vs ~24 us. Seqlock for warm sets; requires pinned workers and no NUMA split. `--prefetch-copies 0` avoids competing PCIe copies. Phase counters. Measured decode +6.4% at 4K / +4.6% at 40K. | Relevant if a CPU pool computes cold experts. For Ampstein's GPU-reads-host path, the prefetch/warm idea could be adapted as GPU slot prefetch rather than CPU worker warm-up. |
| [#3910 qwen35: hybrid mode sizes CPU half by rows that reach it](https://github.com/Niko1221/Strata/pull/3910) | open | Sizes CPU-half device buffers by `--hybrid-rows` instead of whole prompt chunk (8 MiB instead of 128 MiB at prefill 2048), returns VRAM to expert slots. Low-water reserve 768 -> 256 MiB. Measured +1.1% / +1.2% decode and 5 extra resident experts per layer. | Relevant to chunk/scratch sizing and VRAM reserve. Ampstein's chunk 4096 may benefit from sizing scratch by actual rows and reducing reserve. |
| [#4039 smaller prompt scratch (`STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION`)](https://github.com/Niko1221/Strata/pull/4039) | open | Two opt-in buffer reductions: group scratch holds one expert group's rows instead of whole chunk; hyper-connection scratch is a fourth shared region. With both, `--prefill auto` chunk grows 3840 -> 5888 and prompt +18-24%, outputs byte-identical. | Very relevant to Ampstein PP. Smaller group/HC scratch could allow larger chunks than 4096, reducing PCIe crossings for streamed experts. |
| [#4258 `--drop-file-cache`](https://github.com/Niko1221/Strata/pull/4258) | open | After startup reads, `madvise(MADV_DONTNEED)` over mapped views plus `posix_fadvise(POSIX_FADV_DONTNEED)` on files releases the model files' page cache (a third copy). Pair is needed: one call alone releases nothing. Measured 103.68 GiB advised, Cached 77.2 -> 11.2 GiB. Refused if experts still read from files. | Relevant to Ampstein's host RAM/page cache. For the PLE table, page cache is the serving tier, so do not drop it; for fully host-copied experts, drop the file cache after load. |
| [#4224 opt-in parked cache deduplication](https://github.com/Niko1221/Strata/pull/4224) | open | Content-addressed immutable parked conversation storage; snapshots share payloads, copy-on-write on divergence; budget/RAM admission uses physical shared bytes plus metadata. Measured 33.7 GB logical vs 25.3 GB physical, ~25% saved. | Relevant to conversation/session cache memory. Dedup shared prefixes rather than storing full copies. |
| [#4031 conversation cache under RAM pressure](https://github.com/Niko1221/Strata/pull/4031) | draft | Evicts smallest parked conversation rather than oldest under slot pressure; admission gate records measured empty-cache ceiling; cache shrinks slots/budget when a snapshot cannot fit. | Relevant to host RAM pressure and cache policy. Avoid eviction loops that destroy cache without admitting anything. |
| [#4002 parallelize staged conversation cache copies](https://github.com/Niko1221/Strata/pull/4002) | draft | Persistent per-stage workers for capture, parked restore, direct slot copy, host copy; reusable host K/V shuttles. Notes slot<->pool bandwidth ~5 GB/s, PCIe probe 26.8 GB/s. | Relevant to host-GPU copy bandwidth and layer-split staging. Parallel per-stage copies can reduce TTFT for cached requests. |
| [#4049 replicas: one shared expert arena](https://github.com/Niko1221/Strata/pull/4049) | draft | Each replica gets its own server/port, pinned to its own cores, sharing one expert arena file. | Relevant to multi-replica CPU affinity and shared host expert memory. |
| [#4068 KV cache: exact-cell window for q4lm8v](https://github.com/Niko1221/Strata/pull/4068) | draft | 4-bit Lloyd K/V, recent S=256 cells exact, first R=64 sink cells exact, side pool. 5.29 bit/cell, 1016 MiB at 128K vs int8 1584 MiB; KL better than int8. Names follow-ups: per-channel K, lower-bit V, parity test. | Relevant to 256k KV memory. A windowed 4-bit KV could free VRAM for experts, but accuracy needs KL/needle checks. |
| [#3919 KV: adaptive resident ring](https://github.com/Niko1221/Strata/pull/3919) | open | `STRATA_KV_RING_ADAPT=1`: configured `--kv-resident` is a cap; shallow runs keep the live floor, deep runs grow toward the cap. Growth threshold 2x cap. Measured +23.9% decode at 128K, no-op at 64K, avoids 64K regression. | Relevant to sizing KV by actual run depth instead of max context. Ampstein currently allocates max-seq KV; adaptive/paged KV could free VRAM for shallow requests. |
| [#3872 2x RTX 3060 no-AVX2: pool-workers and k8v4 trade-off](https://github.com/Niko1221/Strata/pull/3872) | open | `--pool-workers 11` rejected: -12.7% chat decode, +45.9% TTFT. `k8v4` vs `q4_0`: +7-8% decode but -6-11% prompt read and higher TTFT. | Relevant to CPU pool worker tuning and KV format trade-off: decode-vs-read is workload-dependent. |
| [#3871 kq_cpu_bench: i-quant pairs and Q2_0 CPU pool path](https://github.com/Niko1221/Strata/pull/3871) | open | Fixes i-quant bench crashes; times Q2_0 on Strata's own rows (pool path) rather than ggml-cpu's Q2_0 dot. Q2_0 pool path 0.272 ms vs 0.742 ms. | Relevant to CPU pool quant kernels: benchmark the actual engine path, not generic ggml dot. |
| [#3930 MULTI_GPU: when a layer split pays on a slow CPU](https://github.com/Niko1221/Strata/pull/3930) | open | Docs note: layer split pays when the CPU pool is the bottleneck. Measured dual-socket Xeon pair: decode 35.2 -> 51.0 tok/s, window 70.06 -> 37.96 ms because CPU pool 9.52 -> 0.79 ms. | Relevant to deciding layer split vs single GPU on slow CPU hosts. |
| [#3929 2x RTX 2080 Ti on a slow dual-socket CPU](https://github.com/Niko1221/Strata/pull/3929) | open | Benchmark of layer split vs one GPU, batch, and vLLM on a slow dual-socket CPU. | Relevant data for CPU pool bottleneck. |
| [#4228 dsv41: read Engram tables stored as ggml Q5_K](https://github.com/Niko1221/Strata/pull/4228) | draft | Adds Q5_K as an Engram row format, with a new `q5k_value` dequant kernel and bit-exact tests against ggml `dequantize_row_q5_K`. | Relevant to streaming large lookup tables in arbitrary GGUF row formats. Ampstein's PLE table is IQ4_NL; same row-format/dequant pattern. |
| [#4226 persistent disk conversation caching](https://github.com/Niko1221/Strata/pull/4226) | open | Compact snapshots persisted to disk. | Relevant to session cache persistence, less directly to decode CPU. |

## vLLM

vLLM's CPU-label PRs cover CPU quantized MoE, KV cache, and host offload. The
Qwen4Exp PLE PRs are especially close to Ampstein.

| PR | Status | Technique | Relevance |
|---|---|---|---|
| [#62847 Qwen4Exp: stage CPU-offloaded PLE rows on host when UVA unavailable](https://github.com/vllm-project/vllm/pull/62847) | open | When `cpu_offload` is on and UVA is unavailable, gathers each step's PLE rows from the host table on CPU and copies them to device before forward. The forward reads persistent device buffers, so FULL/PIECEWISE CUDA graphs are kept. Related: pinned host memory under Confidential Computing reports `is_pinned() == False`; UVA views go stale. | Directly analogous to Ampstein's PLE table. Key idea: stage rows into fixed device buffers before the graph so CUDA graphs remain capturable. |
| [#59958 Qwen4Exp: PLE disk offload](https://github.com/vllm-project/vllm/pull/59958) | open | PLE n-gram table stays in checkpoint files; each step reads only its rows on host. No GPU or pinned host copy; recently read rows sit in OS page cache. Rows read in checkpoint format and decoded on GPU with existing PLE kernel. Forward reads fixed device buffers, so CUDA graphs replay unchanged. Measured -27.0 GiB weights, decode within 3.5%, TTFT +11%. Loader releases zero-copy safetensors mmap views and opens files read-only. | Very close to Ampstein's current PLE streaming. The fixed-device-buffer staging and page-cache/reclaim behavior are the main lessons. |
| [#62359 Perf DSv4.1: share Engram host tables as host-NUMA cuMem](https://github.com/vllm-project/vllm/pull/62359) | draft | Allocates shared Engram table as host-NUMA `cuMem`, mapped by GPU with 2M pages instead of 64K pages, reducing GPU TLB misses on random row reads. Rank 0 creates on its GPU's NUMA node and broadcasts FABRIC handles; falls back to `/dev/shm`. Allocation in 2 GiB chunks with 5% reserve; `__GFP_RETRY_MAYFAIL | __GFP_THISNODE`; `mbind(MPOL_LOCAL)`; cgroup-aware available memory; sort rows only for small pages. | Very relevant to cold expert/PLE host tables. 2M pages and host-NUMA cuMem can reduce GPU TLB misses for random row reads. |
| [#54371 Qwen4Exp: UVA PLE offload and Engram tensor parallelism](https://github.com/vllm-project/vllm/pull/54371) | merged | Pinned CPU memory per local PLE shard; GPU reads required rows directly through UVA on a dedicated CUDA stream; `start_prefetch()`/`forward()`; device vs pinned-host backends; ETP sharding. | Relevant architecture: pinned host + UVA direct reads + dedicated stream + prefetch. Ampstein's mapped host experts are similar, but could add prefetch and a dedicated stream. |
| [#54517 Fuse Qwen4Exp PLE kernels](https://github.com/vllm-project/vllm/pull/54517) | merged | Fuses n-gram ID generation, dilated short-conv, and PLE gate; merges key/value projections; avoids temporary intermediates. N-gram ID generation 31-40x faster than eager. | Relevant to PLE CPU/GPU overhead. Fusing ngram ID generation + conv + gate reduces host/launch overhead and memory. |
| [#53899 Support PLE-Offload for Qwen3.8-Flash-Next](https://github.com/vllm-project/vllm/pull/53899) | closed/paused | Dedicated CPU offload worker transfers selected PLE rows to GPU workers. | Context for worker-based row transfer; #54371 supersedes with UVA. |
| [#54129 Disk-backed mmap PLE table](https://github.com/vllm-project/vllm/pull/54129) | closed | `VLLM_PLE_MMAP`: read required rows from safetensors via read-only mmap; gather during input prep into stable GPU buffers; FULL cudagraph works. Workers, chunk size, prewarm/readahead, pinned H2D, serial threshold. | Very relevant. The worker/chunk/prewarm/readahead/pinned-H2D parameters are concrete tuning knobs for PLE row streaming. |
| [#63424 CPU: share FP8 MoE projections and enable native MXFP4 W4A8](https://github.com/vllm-project/vllm/pull/63424) | open | Shares CPU FP8/MXFP4 MoE projection execution; automatically uses MXFP4 W4A8 on AMX-FP8 CPUs; quantizes down-projection activations; panel cache; BF16 for tiny experts; falls back to W4A16 without AMX. Measured GPT-OSS-20B batch +37%, Qwen3-30B +4.1%. | Relevant only if CPU MoE computation is used. For Ampstein, GPU kernels are preferred; this is a comparison for CPU fallback. |
| [#63711 CPU Zen: Laguna MoE via grouped W8A16 (DA8W8)](https://github.com/vllm-project/vllm/pull/63711) | open | Extends Zen CPU WNA16 path to grouped W8A16, handling mixed grouped int4/int8 experts and ungrouped sigmoid routing. | Relevant to mixed quant expert formats on CPU. Ampstein's expert types are IQ2_XS/IQ3_XXS; not W8A16. |
| [#61346 CPU: BF16 fallback for MXFP4 MoE without AMX](https://github.com/vllm-project/vllm/pull/61346) | open | Decodes packed E2M1 weights and E8M0 scales once at load to BF16, then uses CPU grouped GEMM; ~4x storage; fallback when no native AMX. | Relevant to CPU fallback design: unpack quant once at load, accept larger host memory, preserve output. |
| [#61168 CPU: UltraQuant 4-bit KV cache](https://github.com/vllm-project/vllm/pull/61168) | open | Runs ultraquant_4bit KV cache on CPU; shares format modules with GPU path; reference tests; GSM8K 71.19% vs fp8 73.16%. | Relevant to 4-bit KV on CPU/host spill. Accuracy trade-off must be checked. |
| [#62022 SVE Paged Attention FP32/FP16/BF16](https://github.com/vllm-project/vllm/pull/62022) | open | ARM SVE256 attention kernels; BF16 BFMMLA; +5.7% FP32, +2.7% FP16, +13.7% BF16 vs NEON on Graviton3. | Not relevant on x86 `j7`; relevant only for ARM hosts. |
| [#62440 AArch64 top-k and top-p sampling kernels](https://github.com/vllm-project/vllm/pull/62440) | open | AArch64 sampling kernels. | ARM-specific; not relevant on x86. |
| [#62933 KV Offload: mirror CPU offload region slots across nodes](https://github.com/vllm-project/vllm/pull/62933) | open | Multi-node TP workers mirror their CPU region slots to head node via NIXL RDMA; `mirror_source: auto|cpu|gpu`; GPU→RDMA→remote CPU 1.33/1.45 GB/s vs GPU→local CPU→RDMA 11.0 GB/s. | Relevant to multi-node host KV staging. Shows staging through local CPU is much faster than direct GPU RDMA in that setup. |
| [#63779 KV Offload/Tiering: let a tier fail a promotion and defer releasing chunks](https://github.com/vllm-project/vllm/pull/63779) | open | Tier promotion failure handling; defer chunk release. | Relevant to tiered host/disk KV cache design. |
| [#63551 KV Offload: protect HITs from promotion eviction](https://github.com/vllm-project/vllm/pull/63551) | open | Prevents promotion eviction from invalidating reported HITs. | Relevant to cache correctness under tier promotion. |
| [#63524 KV Offload: report only written blocks from fs tier store jobs](https://github.com/vllm-project/vllm/pull/63524) | open | Reports written blocks only. | Relevant to async filesystem tier store jobs. |
| [#63441 KV Events: batch BlockRemoved events per allocation](https://github.com/vllm-project/vllm/pull/63441) | open | Batches block-removed events. | Relevant to event overhead in tiered cache. |
| [#63808 Mamba-aware LRU prefix-cache eviction](https://github.com/vllm-project/vllm/pull/63808) | open | Eviction policy for hybrid models. | Relevant to cache eviction policy for hybrid attention models. |
| [#63366 Bugfix CPU: default Triton to CPU backend](https://github.com/vllm-project/vllm/pull/63366) | open | Default Triton CPU backend. | Minor CPU backend selection. |
| [#63738 Bugfix CPU: encoder-only attention bidirectional in CPU_ATTN](https://github.com/vllm-project/vllm/pull/63738) | open | CPU attention correctness. | Less relevant to decode optimization. |
| [#63508 Bugfix CPU: validate attention head size at CPU backend selection](https://github.com/vllm-project/vllm/pull/63508) | open | CPU backend validation. | Less relevant. |

## Priority experiments for Ampstein

1. **Pipeline parallelism with host experts** (llama.cpp #29963). The current
   layer-split runs sequentially. Each GPU could upload its own layers' host
   experts and overlap consecutive ubatches. Keep graph topology constant and
   upload the whole expert tensor for large batches instead of waiting on MoE
   ids.
2. **Single-graph MoE offload** (llama.cpp #30971). Host cold experts can split
   the graph. Merge graph splits via waiting kernels, fuse ops, keep a fixed
   topology, and reduce host-GPU copy timing.
3. **Fixed PLE staging buffers for CUDA graphs** (vLLM #62847, #59958, #54129).
   Gather PLE rows during input prep into stable device buffers, then let the
   captured graph read only those buffers. Add worker/chunk tuning,
   prewarm/readahead, pinned H2D, and a serial threshold.
4. **Huge pages / host-NUMA cuMem** (llama.cpp #22022, vLLM #62359). Use 2 MB
   pages for large host tables and mapped cold experts to reduce GPU TLB misses
   on random row reads. `MADV_HUGEPAGE` is cheap and advisory.
5. **Page-cache reclaim** (llama.cpp #24156, Strata #4258). After host copies
   are complete, drop the model files' page cache with `madvise(MADV_DONTNEED)`
   plus `posix_fadvise(POSIX_FADV_DONTNEED)`. Do not drop the PLE page cache if
   the PLE table is still streamed from disk.
6. **NUMA mirror / expert bands** (llama.cpp #27986, Strata #4016). On a
   multi-NUMA host, keep a pinned primary for DMA and per-node replicas for
   local decode. On a single-node host, skip this.
7. **Smaller prompt scratch and larger chunks** (Strata #4039, #3910). Size
   group/HC scratch by actual rows rather than whole chunk, and reduce VRAM
   reserve. This could allow chunk 8192 where chunk 4096 currently fits.
8. **Prefetch/async expert placement** (Strata #3906, #3907). Prefetch the next
   layer's predicted hot cold experts into free slots; async slot swaps off the
   engine stream; lend staging buffers as expert slots.
9. **Adaptive/windowed KV** (Strata #3919, #4068, vLLM #61168). Size KV by run
   depth, or use exact-cell 4-bit KV windows. This can free VRAM for experts at
   shallow contexts, but needs KL/needle checks at 256k.
10. **Sync elimination** (llama.cpp #29796, #30196). Track stream use and skip
    redundant `cudaStreamSynchronize`; use one output view/sync per batch.
11. **CPU pool tuning** (llama.cpp #20596, #28785, Strata #3907, #3872). Skip
    threadpool when no CPU ops; fuse single-token gated activation; avoid
    false sharing; warm predicted experts; do not blindly raise pool workers.
12. **CPU quant kernels** (llama.cpp #30448, #26348, Strata #3871, vLLM #63424,
    #63711, #61346, #61168). AVX2/VNNI, grouped W8A16, MXFP4 W4A8, 4-bit KV.
    These are mainly for CPU fallback or CPU pools, not the CUDA expert path.

## Already implemented in Ampstein

- `MADV_HUGEPAGE` hints on the GGUF/safetensors mmap and the PLE gather mapping.
- `AMP_QWEN4_DROP_FILE_CACHE=1` drops the model files' clean page cache after
  load (`madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)`), keeping
  the PLE table file because it is still streamed.
- PLE rows are copied into the fixed device buffer `it.emb` before graph replay,
  so the captured graph reads only persistent buffers.
- `prefill_pipeline` already does wavefront layer-split prefill across stages.
- Cold experts use mapped pinned host memory plus device pointer tables; this is
  already a zero-copy host-read path.

## Caveats

- Several PRs are drafts or hardware-specific. Treat the measured gains as
  directional, not guarantees.
- Some techniques change output (async swaps, prefetch timing, FMA contraction).
  Ampstein's invariant is oracle/bit-exactness; any timing-dependent prefetch or
  async swap must be checked against the EXL3 oracle.
- CPU kernels are not a substitute for the CUDA dequant/gemv path. Use them only
  for CPU fallback or CPU pool work.
- `j7` has two RTX 3090s at PCIe x8 and one host NUMA domain; NUMA mirror and
  multi-node techniques are less relevant there than single-node host staging,
  page cache, and graph scheduling.
