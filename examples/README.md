# Examples

Small runnable ucode scripts. Run them with the module on the search path, e.g.

```sh
ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/<script>.uc [-- args]
```

| Script | What |
|---|---|
| `version.uc` | the module loads and reports its libtorch version |
| `mlp.uc` | a 2-layer MLP forward pass on the GPU, in pure ucode |
| `chat.uc` | interactive terminal chat on a llama-family EXL3 model |
| `bench.uc` | prefill and greedy decode throughput of the llama path (eager vs CUDA graph) |
| `weight_sizes.uc` | checkpoint bytes by tensor family |
| `bench_qwen35.uc` | Qwen3.5 / 3.8 benchmark over a multi-GPU layer split |
| `bench_prefill.uc` | sequential versus pipelined prefill |
| `bench_decode.uc` | plain greedy versus MTP speculative decoding |
| `perf_server.uc` | the server's decode path outside the server (layer split or `--tp`) |
| `bench_allreduce.uc` | latency of the P2P all-reduce inside a CUDA graph |
| `bench_glm5.uc`, `bench_prefill_glm5.uc`, `mem_glm5.uc` | GLM-5.3-Flash decode latency, prefill throughput, per-GPU memory |
