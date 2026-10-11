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
