# Libtorch distribution: fetched from the PyTorch wheel index / PyPI, cached.
#
# libtorch has no packaged C++ release we can rely on for the ABI our module
# was developed against, so we take the pip distribution: the `torch` wheel
# (libtorch*.so, headers, TorchConfig.cmake) plus, on cuda, the NVIDIA runtime
# wheels it links (cudnn, cublas, nccl, ...). Extracted side by side in ONE
# root with the same layout as site-packages:
#
#     <root>/torch/{lib,include,share/cmake}
#     <root>/nvidia/<pkg>/lib        (cuda only)
#
# which is exactly what the cuda wheels' own RPATHs ($ORIGIN/../../nvidia/*/lib)
# expect, so the installed copy works unchanged. The rocm wheel has no such
# split-package layout — its ROCm runtime libs (rocblas, hipblaslt, MIOpen,
# ...) are bundled straight into torch/lib, so AMPSTEIN_TORCH_NVIDIA_LIBDIRS
# is simply empty on that backend and every nvidia-specific step downstream
# (CMakeLists.txt's install loop) is a no-op.
#
# Which manifest is used depends on AMPSTEIN_GPU_BACKEND (set in
# CMakeLists.txt before this file is included).
#
# Caching: everything lives OUTSIDE the build tree (default
# ${XDG_CACHE_HOME:-~/.cache}/ampstein/libtorch, override with
# -DAMPSTEIN_CACHE_DIR= or $AMPSTEIN_CACHE_DIR), keyed by version, so
# `rm -rf build` and extra build dirs never re-download. Each wheel is
# downloaded to a .part file, hash-verified, then renamed; a stamp file marks a
# completed extraction, so a configure with a warm cache does no work at all
# (no hashing, no network). Interrupted runs resume at the first missing wheel.
#
# To use an existing distribution (e.g. a venv's site-packages) instead:
#   -DAMPSTEIN_TORCH_DIST_DIR=/path/containing/torch-and-nvidia
#
# Output variables: AMPSTEIN_TORCH_DIST_ROOT, AMPSTEIN_TORCH_DIST_NAME
# (install dir name, torch-dist-<version>), AMPSTEIN_TORCH_NVIDIA_LIBDIRS.

if(NOT DEFINED AMPSTEIN_TORCH_VERSION)
    if(AMPSTEIN_GPU_BACKEND STREQUAL "rocm")
        set(AMPSTEIN_TORCH_VERSION "2.16.0.dev20261006+rocm7.14")
    else()
        set(AMPSTEIN_TORCH_VERSION "2.11.0+cu128")
    endif()
endif()
set(AMPSTEIN_TORCH_VERSION "${AMPSTEIN_TORCH_VERSION}" CACHE STRING "libtorch (pip torch) version")
set(AMPSTEIN_TORCH_DIST_DIR "" CACHE PATH
    "Use this existing torch distribution root (contains torch/ and nvidia/) instead of downloading")

if(AMPSTEIN_TORCH_VERSION STREQUAL "2.16.0.dev20261006+rocm7.14")
    # torch's native libraries are python-version independent; cp313 is just a
    # carrier. No nvidia_*-style split packages: the wheel bundles its own
    # ROCm runtime libs directly under torch/lib. No STABLE release wheel
    # matches this host's ROCm 7.1.0 (the newest stable is rocm7.0, torch
    # 2.10.0); this is a nightly build tagged for rocm7.1-series specifically.
    # Nightly builds roll off PyTorch's CDN after a while and don't publish a
    # stable URL/hash the way release wheels do -- this one was downloaded
    # and hashed by hand on 2026-10-26; if the cache is ever cleared and the
    # URL below 404s, pick a current one from
    # https://download.pytorch.org/whl/nightly/rocm7.14/torch/ and update
    # both the version string above and the entry below together.
    set(_AMP_WHEELS
        "torch-2.16.0.dev20261006%2Brocm7.14-cp313-cp313-manylinux_2_28_x86_64.whl|https://download.pytorch.org/whl/nightly/rocm7.14/torch-2.16.0.dev20261006%2Brocm7.14-cp313-cp313-manylinux_2_28_x86_64.whl|6d55b718c9e2ac48e324e54665b13f2a66642e40448e8befcc52d16871316ec3"
    )
elseif(AMPSTEIN_TORCH_VERSION STREQUAL "2.11.0+cu128")
    # torch's native libraries are python-version independent; cp312 is just a
    # carrier. Hashes come from the index page / PyPI JSON API.
    set(_AMP_WHEELS
        "torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl|https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl|d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997"
        "nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl|https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl|8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142"
        "nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/f8/02/2adcaa145158bf1a8295d83591d22e4103dbfd821bcaf6f3f53151ca4ffa/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|ea0cb07ebda26bb9b29ba82cda34849e73c166c18162d3913575b0c9db9a6182"
        "nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl|https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl|a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994"
        "nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90"
        "nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl|https://files.pythonhosted.org/packages/c5/41/65225d42fba06fb3dd3972485ea258e7dd07a40d6e01c95da6766ad87354/nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl|ac6ad90a075bb33a94f2b4cf4622eac13dd4dc65cf6dd9c7572a318516a36625"
        "nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74"
        "nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/bb/fe/1bcba1dfbfb8d01be8d93f07bfc502c93fa23afa6fd5ab3fc7c1df71038a/nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|1d069003be650e131b21c932ec3d8969c1715379251f8d23a1860554b1cb24fc"
        "nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl|https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl|b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9"
        "nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl|https://files.pythonhosted.org/packages/85/48/9a13d2975803e8cf2777d5ed57b87a0b6ca2cc795f9a4f59796a910bfb80/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl|4376c11ad263152bd50ea295c05370360776f8c3427b30991df774f9fb26c450"
        "nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/c2/f5/e1854cb2f2bcd4280c44736c93550cc300ff4b8c95ebe370d0aa7d2b473d/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|1ec05d76bbbd8b61b06a80e1eaf8cf4959c3d4ce8e711b65ebd0443bb0ebb13b"
        "nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl|https://files.pythonhosted.org/packages/56/79/12978b96bd44274fe38b5dde5cfb660b1d114f70a65ef962bcbbed99b549/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl|f1bb701d6b930d5a7cea44c19ceb973311500847f81b634d802b7b539dc55623"
        "nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl|https://files.pythonhosted.org/packages/4a/4e/44dbb46b3d1b0ec61afda8e84837870f2f9ace33c564317d59b70bc19d3e/nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl|485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab"
        "nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl|https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl|81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88"
        "nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/b5/09/6ea3ea725f82e1e76684f0708bbedd871fc96da89945adeba65c3835a64c/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|042f2500f24c021db8a06c5eec2539027d57460e1c1a762055a6554f72c369bd"
        "nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|https://files.pythonhosted.org/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f"
    )
else()
    message(FATAL_ERROR "No wheel manifest for AMPSTEIN_TORCH_VERSION=${AMPSTEIN_TORCH_VERSION}; "
        "add one to cmake/Libtorch.cmake or pass -DAMPSTEIN_TORCH_DIST_DIR=<existing distribution>.")
endif()

set(AMPSTEIN_TORCH_DIST_NAME "torch-dist-${AMPSTEIN_TORCH_VERSION}")

if(AMPSTEIN_TORCH_DIST_DIR)
    get_filename_component(AMPSTEIN_TORCH_DIST_ROOT "${AMPSTEIN_TORCH_DIST_DIR}" ABSOLUTE)
    if(NOT EXISTS "${AMPSTEIN_TORCH_DIST_ROOT}/torch/share/cmake/Torch/TorchConfig.cmake")
        message(FATAL_ERROR "AMPSTEIN_TORCH_DIST_DIR=${AMPSTEIN_TORCH_DIST_DIR}: no torch/share/cmake/Torch/TorchConfig.cmake")
    endif()
else()
    if(NOT AMPSTEIN_CACHE_DIR)
        if(DEFINED ENV{AMPSTEIN_CACHE_DIR})
            set(AMPSTEIN_CACHE_DIR "$ENV{AMPSTEIN_CACHE_DIR}")
        elseif(DEFINED ENV{XDG_CACHE_HOME})
            set(AMPSTEIN_CACHE_DIR "$ENV{XDG_CACHE_HOME}/ampstein")
        else()
            set(AMPSTEIN_CACHE_DIR "$ENV{HOME}/.cache/ampstein")
        endif()
    endif()
    set(AMPSTEIN_CACHE_DIR "${AMPSTEIN_CACHE_DIR}" CACHE PATH "Download/extraction cache (shared by all build trees)")

    set(_dl_dir "${AMPSTEIN_CACHE_DIR}/downloads")
    set(AMPSTEIN_TORCH_DIST_ROOT "${AMPSTEIN_CACHE_DIR}/libtorch/${AMPSTEIN_TORCH_DIST_NAME}")
    set(_stamp "${AMPSTEIN_TORCH_DIST_ROOT}/.ampstein-complete")

    if(NOT EXISTS "${_stamp}")
        file(MAKE_DIRECTORY "${_dl_dir}")
        set(_part_root "${AMPSTEIN_TORCH_DIST_ROOT}.part")
        file(REMOVE_RECURSE "${_part_root}")
        file(MAKE_DIRECTORY "${_part_root}")

        foreach(_w IN LISTS _AMP_WHEELS)
            string(REPLACE "|" ";" _f "${_w}")
            list(GET _f 0 _name)
            list(GET _f 1 _url)
            list(GET _f 2 _sha)
            string(REPLACE "%2B" "+" _name "${_name}")
            set(_file "${_dl_dir}/${_name}")

            # A wheel is only ever renamed into place after its hash matched,
            # so existence == verified; no re-hash, no re-download.
            if(NOT EXISTS "${_file}")
                message(STATUS "ampstein: downloading ${_name}")
                file(DOWNLOAD "${_url}" "${_file}.part"
                    EXPECTED_HASH SHA256=${_sha}
                    SHOW_PROGRESS STATUS _st TLS_VERIFY ON)
                list(GET _st 0 _rc)
                if(NOT _rc EQUAL 0)
                    file(REMOVE "${_file}.part")
                    message(FATAL_ERROR "ampstein: download of ${_url} failed: ${_st}")
                endif()
                file(RENAME "${_file}.part" "${_file}")
            endif()

            message(STATUS "ampstein: extracting ${_name}")
            file(ARCHIVE_EXTRACT INPUT "${_file}" DESTINATION "${_part_root}")
        endforeach()

        file(REMOVE_RECURSE "${AMPSTEIN_TORCH_DIST_ROOT}")
        file(RENAME "${_part_root}" "${AMPSTEIN_TORCH_DIST_ROOT}")
        file(WRITE "${_stamp}" "${AMPSTEIN_TORCH_VERSION}\n")
    endif()
endif()

file(GLOB AMPSTEIN_TORCH_NVIDIA_LIBDIRS "${AMPSTEIN_TORCH_DIST_ROOT}/nvidia/*/lib")
message(STATUS "ampstein: libtorch ${AMPSTEIN_TORCH_VERSION} at ${AMPSTEIN_TORCH_DIST_ROOT}")

set(Torch_DIR "${AMPSTEIN_TORCH_DIST_ROOT}/torch/share/cmake/Torch" CACHE PATH "" FORCE)
