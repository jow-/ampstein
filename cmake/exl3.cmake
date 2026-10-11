# Vendored EXL3 kernels (from exllamav3, MIT — see vendor/exl3/LICENSE.exllamav3 and VERSION).
#
# One object library with the native entry points torch.so binds: exl3_gemm /
# gemv / reconstruct, the Hadamard transforms and the per-model fused kernels
# (GDN, mHC, MoE, PLE, ...). Only the translation units the bindings reach are
# compiled.

set(EXL3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/exl3/exllamav3_ext)

# --- EXL3 quantized GEMM / GEMV, reconstruct, Hadamard
set(EXL3_KERNEL_SRCS
    ${EXL3_DIR}/quant/exl3_gemm.cu
    ${EXL3_DIR}/quant/exl3_gemv.cu
    ${EXL3_DIR}/quant/exl3_gemv_int8.cu
    ${EXL3_DIR}/quant/exl3_kernel_map.cu
    ${EXL3_DIR}/quant/exl3_devctx.cu
    ${EXL3_DIR}/quant/coop_autotune.cu
    ${EXL3_DIR}/quant/reconstruct.cu
    ${EXL3_DIR}/quant/hadamard.cu
)

# Codebook kernel instances: one TU per (bitrate K, codebook cb)
foreach(k RANGE 1 8)
    foreach(cb RANGE 0 2)
        list(APPEND EXL3_KERNEL_SRCS
            ${EXL3_DIR}/quant/comp_units/exl3_comp_unit_${k}_cb${cb}.cu)
    endforeach()
endforeach()

foreach(h RANGE 1 3)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_comp_unit_h${h}.cu)
endforeach()

# int8 GEMV kernel instances (mul1-codebook tensors at m == 1). The int8
# path is ON by default in exl3_gemv_int8.cu (EXL3_INT8_GEMV, default 2),
# so a mul1 checkpoint hitting an m == 1 gemm resolves these symbols —
# leaving them out of the link is a latent crash, not a dead path.
foreach(k RANGE 1 8)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_gemv_int8_inst_coop_k${k}.cu)
endforeach()
foreach(h RANGE 1 3)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_gemv_int8_inst_coop_h${h}.cu)
endforeach()
foreach(k RANGE 1 6)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_gemv_int8_inst_sq_k${k}.cu)
endforeach()
foreach(h RANGE 1 3)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_gemv_int8_inst_sq_h${h}.cu)
endforeach()

# --- mixture of experts: fused token-sorted MoE (prefill / small batches) and
# the fused decode-shaped MoE (two launches per layer), instances per K
file(GLOB EXL3_MOE_INSTS ${EXL3_DIR}/quant/comp_units/exl3_moe_inst_*.cu)
list(APPEND EXL3_KERNEL_SRCS
    ${EXL3_DIR}/quant/exl3_moe.cu
    ${EXL3_DIR}/quant/exl3_moe_coop.cu
    ${EXL3_MOE_INSTS})

foreach(k RANGE 1 8)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_moe_coop_inst_k${k}.cu)
endforeach()
foreach(h RANGE 1 3)
    list(APPEND EXL3_KERNEL_SRCS
        ${EXL3_DIR}/quant/comp_units/exl3_moe_coop_inst_h${h}.cu)
endforeach()

# --- mHC mixing (GLM-5.3)
list(APPEND EXL3_KERNEL_SRCS ${EXL3_DIR}/hc_mix.cu)

# --- Qwen3.8-Flash-Next: n-gram embedding, per-layer embeddings, norms
list(APPEND EXL3_KERNEL_SRCS
    ${EXL3_DIR}/ngram.cu
    ${EXL3_DIR}/ple.cu
    ${EXL3_DIR}/norm.cu)

add_library(exl3_kernels OBJECT ${EXL3_KERNEL_SRCS})

target_include_directories(exl3_kernels PRIVATE
    ${EXL3_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Flags mirror exllamav3's util/cuda_flags.py (minus compress-mode)
target_compile_options(exl3_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math
    -Xcudafe="--diag_suppress=177"
    -Xcudafe="--diag_suppress=20012">
)

target_compile_definitions(exl3_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:TORCH_EXTENSION_NAME=exl3_kernels>
)