# ucode-level tests, run with the ucode interpreter against the freshly built
# torch.so. Model-backed tests skip (exit code 125) when the model or its
# oracle fixture (tests/oracle/make_*.py -> tests/fixtures/) is absent; they
# run serially so they do not compete for VRAM.

get_filename_component(UCODE_BIN_DIR "${UCODE_INTERPRETER}" DIRECTORY)

function(amp_test name)
    cmake_parse_arguments(T "" "SCRIPT;ENV" "" ${ARGN})
    if(NOT T_SCRIPT)
        set(T_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/${name})
    elseif(NOT IS_ABSOLUTE "${T_SCRIPT}")
        set(T_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/${T_SCRIPT})
    endif()
    add_test(NAME ${name}
        COMMAND ${UCODE_INTERPRETER}
            -L ${CMAKE_BINARY_DIR}/lib
            -L ${CMAKE_SOURCE_DIR}/src/ucode
            -L ${UCODE_BIN_DIR}
            -ltorch=ampstein.torch
            ${T_SCRIPT}.uc)
    set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE SKIP_RETURN_CODE 125)
    if(T_ENV)
        set_tests_properties(${name} PROPERTIES ENVIRONMENT "${T_ENV}")
    endif()
endfunction()

# --- Foundation: tensors, safetensors reader / writer
# tensor_basics: create / move / read back; no model
amp_test(tensor_basics)
# tensor_ops: elementwise, reductions, views; no model
amp_test(tensor_ops)
# st_reader: mmap safetensors reader against a python-written fixture
amp_test(st_reader)
# st_save: safetensors writer round-tripped through torch.st
amp_test(st_save)
# bindings_errors: every binding raises (instead of returning null) on wrong argument types / counts
amp_test(bindings_errors)

# --- EXL3 kernels and the llama-family model
# exl3: exl3_gemm vs a torch reference, synthetic tensors
amp_test(exl3)
# exl3_real: one real layer of a quantized model vs the exllamav3 oracle
amp_test(exl3_real)
# exl3_recon: reconstruct + cuBLAS prefill gemm vs the trellis-decode gemm
amp_test(exl3_recon)
# exl3_stack: 16-layer stack vs exllamav3, bit-exact
amp_test(exl3_stack)
# weightmap: weight-map routing and linear class selection
amp_test(weightmap)
# tokenizer: native BPE tokenizer vs the HF tokenizers oracle
amp_test(tokenizer)
# sampling: sampler semantics: greedy / temperature / top-k / seeds
amp_test(sampling)
# llama_forward: full prefill forward vs the exllamav3 logits oracle
amp_test(llama_forward)
# llama_decode: KV-cache decode path vs the same oracle
amp_test(llama_decode)
# chat_prompt: chat-template prompt (BOS + atomized specials)
amp_test(chat_prompt)
# kld: per-layer KLD / rel-RMS / perplexity harness vs exllamav3
amp_test(kld)
# graphdec: CUDA-graph captured decode vs the oracle
amp_test(graphdec)
# graphdec_loop: multi-step device loop is token-exact vs eager greedy
amp_test(graphdec_loop)
# loadperf: batched pinned-async weight upload (bit-exactness + timing)
amp_test(loadperf)
# decode_attn_hd128: decode attention at head_dim 128 vs SDPA, split boundaries + determinism
amp_test(decode_attn_hd128)
# mlp_gpu: 2-layer MLP forward on the GPU (examples/mlp.uc)
amp_test(mlp_gpu SCRIPT ${CMAKE_SOURCE_DIR}/examples/mlp)

# --- Jinja -> utpl transpiler
# taint_values: runtime taint values: cells, mixed strings, scope frames
amp_test(taint_values)
# tv_sink: template output sink: merging, taint mapping, nested capture
amp_test(tv_sink)
# utpl: taint-aware transpilation over an embedded template battery
amp_test(utpl)
# utpl_template: real checkpoint chat templates, byte-exact vs the jinja2 oracle
amp_test(utpl_template)

# --- Qwen3.5 / 3.8 hybrid (Gated DeltaNet + attention)
# gdn_scan: fused gated delta rule kernel vs the per-token recurrence
amp_test(gdn_scan)
# gdn_fused: fused GDN layer kernels vs the composite ATen reference
amp_test(gdn_fused)
# qwen35_layers: per-layer oracle (GDN + full attention)
amp_test(qwen35_layers)
# qwen35_full: full model, multi-GPU layer split, generation vs the oracle
amp_test(qwen35_full)
# qwen35_pipeline: pipelined chunked prefill == sequential prefill (bit-exact)
amp_test(qwen35_pipeline)
# qwen35_graph: CUDA-graph decode steps vs eager, two GPUs
amp_test(qwen35_graph)
# attn_decode_s: split-KV decode attention, shuffle + tensor-core variants vs SDPA
amp_test(attn_decode_s)
# qwen35_attn_dev: decode attention wiring selects the mma kernel
amp_test(qwen35_attn_dev)

# --- Server and engine
# httpd: HTTP head / body readers driven with sliced input
amp_test(httpd)
# chat_qwen35: chat template port vs jinja2 + streaming output parser
amp_test(chat_qwen35)
# chat_jinja: transpiled renderer: the server's prompt path, byte-exact vs jinja2
amp_test(chat_jinja)
# sampling_dist: engine.dist (tensor-op sampling) vs the reference implementation
amp_test(sampling_dist)

# --- MTP speculative decoding and the async engine
# exl3_gemv_rows: EXL3 small-m products (verify rows) vs the dequantized reference
amp_test(exl3_gemv_rows)
# spec_sampling: speculative-sampling acceptance rule: emitted token ~ target p
amp_test(spec_sampling)
# qwen35_mtp: MTP speculative decoding == plain greedy, incl. rejections + state rollback
amp_test(qwen35_mtp)
# notifier: completion notifier + copy_async: uloop integration, AND semantics, pinned D2H
amp_test(notifier)
# engine_async: async sampling reproduces dist() / argmax exactly
amp_test(engine_async)

# --- Tensor parallelism
# p2p_comm: N-way P2P all-reduce, 2..visible GPUs, both modes
amp_test(p2p_comm)
# attn_decode_tp: TP decode attention scenario, cross-device CUDA graph
amp_test(attn_decode_tp)
# qwen35_tp: tensor-parallel qwen35 (eager + multi-GPU graph) vs the layer split
amp_test(qwen35_tp)

# --- Vision
# vision_preprocess: native image preprocessing vs the PIL oracle
amp_test(vision_preprocess)
# vision_tower: ViT + merger vs exllamav3's tower
amp_test(vision_tower)
# mrope: MRoPE positions / cos-sin rows / embedding overlay
amp_test(mrope)
# qwen35_vision: image prompt through the 27B: continuation, chunking invariance, graph decode
amp_test(qwen35_vision)

# --- GLM-5.3-Flash and the engine's cache store
# kv_q8: Q8 latent cache rows put / get
amp_test(kv_q8)
# idx_score: fused DSA indexer scoring kernel
amp_test(idx_score)
# skinny: deterministic split-K fp32 product, 1..8 rows
amp_test(skinny)
# glm5_chat: GLM chat template renderer + output parser (pure ucode)
amp_test(glm5_chat)
# glm5_blocks: mHC dense blocks vs the oracle
amp_test(glm5_blocks)
# glm5_kda: KDA layer vs the oracle
amp_test(glm5_kda)
# glm5_mla: MLA layer vs the oracle
amp_test(glm5_mla)
# glm5_moe: MoE layer vs the oracle
amp_test(glm5_moe)
# glm5_full: full model vs the oracle
amp_test(glm5_full)
# glm5_long: long context: indexer selection + sparse attention
amp_test(glm5_long)
# glm5_ppl: perplexity / KLD on real text beyond index_topk
amp_test(glm5_ppl)
# glm5_mtp_split: MTP head on the layer split
amp_test(glm5_mtp_split)
# glm5_tp: tensor / expert parallel
amp_test(glm5_tp)
# glm5_tp_graph: TP decode graphs
amp_test(glm5_tp_graph)
# glm5_tp_pipe: pipeline of TP stages
amp_test(glm5_tp_pipe)
# glm5_mtp: MTP draft head under TP
amp_test(glm5_mtp)
# glm5_vision: GLM vision tower vs the oracle
amp_test(glm5_vision)
# engine_inject: force() hook (reasoning budget): forced continuation == extended prompt
amp_test(engine_inject)
# engine_store: host-RAM prompt-cache store: restore == fresh prefill
amp_test(engine_store)
# engine_store_quant: tiered KV quantization of stored state at long context
amp_test(engine_store_quant)
# the TP tests again as a pipeline of three 2-rank stages
amp_test(glm5_tp_stages SCRIPT glm5_tp ENV AMP_GLM5_STAGES=3)
amp_test(glm5_tp_graph_stages SCRIPT glm5_tp_graph ENV AMP_GLM5_STAGES=3)
amp_test(glm5_mtp_stages SCRIPT glm5_mtp ENV AMP_GLM5_STAGES=3)
