cmake_minimum_required(VERSION 3.16)
project(Ampstein C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- GPU backend selection ------------------------------------------------
# cuda (default) or rocm. Auto-detected from what's on the host when not
# set explicitly: an nvcc turns up -> cuda; else a hipcc / /opt/rocm* tree
# -> rocm. Everything downstream (libtorch wheel manifest, which kernel
# object libraries get built, EXL3 inclusion) branches on this one variable.
if(NOT DEFINED AMPSTEIN_GPU_BACKEND)
    find_program(_amp_probe_nvcc nvcc
        HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDAToolkit_ROOT /usr/local/cuda
        PATH_SUFFIXES bin)
    if(_amp_probe_nvcc)
        set(AMPSTEIN_GPU_BACKEND "cuda")
    else()
        find_program(_amp_probe_hipcc hipcc
            HINTS ENV ROCM_PATH ENV HIP_PATH /opt/rocm
            PATH_SUFFIXES bin)
        if(_amp_probe_hipcc)
            set(AMPSTEIN_GPU_BACKEND "rocm")
        else()
            set(AMPSTEIN_GPU_BACKEND "cuda") # neither found: keep the historical nvcc-required error below
        endif()
    endif()
endif()
set(AMPSTEIN_GPU_BACKEND "${AMPSTEIN_GPU_BACKEND}" CACHE STRING "GPU backend: cuda or rocm")
set_property(CACHE AMPSTEIN_GPU_BACKEND PROPERTY STRINGS cuda rocm)
if(NOT AMPSTEIN_GPU_BACKEND STREQUAL "cuda" AND NOT AMPSTEIN_GPU_BACKEND STREQUAL "rocm")
    message(FATAL_ERROR "AMPSTEIN_GPU_BACKEND must be 'cuda' or 'rocm' (got '${AMPSTEIN_GPU_BACKEND}').")
endif()
message(STATUS "ampstein: GPU backend ${AMPSTEIN_GPU_BACKEND}")

if(AMPSTEIN_GPU_BACKEND STREQUAL "cuda")
    # --- CUDA toolchain discovery ----------------------------------------
    # nvcc is usually NOT on PATH (CMake then finds the toolkit directory but no
    # compiler and dies in Caffe2's enable_language(CUDA)), so locate it here,
    # before find_package(Torch) enables CUDA: -DCMAKE_CUDA_COMPILER, $CUDACXX,
    # $CUDA_HOME / $CUDA_PATH / $CUDAToolkit_ROOT, /usr/local/cuda, then the
    # newest /usr/local/cuda-*.
    if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        file(GLOB _cuda_versioned LIST_DIRECTORIES true /usr/local/cuda-*)
        list(SORT _cuda_versioned ORDER DESCENDING)
        find_program(AMPSTEIN_NVCC nvcc
            HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDAToolkit_ROOT
                  /usr/local/cuda ${_cuda_versioned}
            PATH_SUFFIXES bin
            DOC "CUDA compiler (auto-detected)")
        if(NOT AMPSTEIN_NVCC)
            message(FATAL_ERROR "nvcc not found. Install the CUDA toolkit or pass "
                "-DCMAKE_CUDA_COMPILER=/path/to/nvcc (or set CUDA_HOME).")
        endif()
        set(CMAKE_CUDA_COMPILER "${AMPSTEIN_NVCC}" CACHE FILEPATH "CUDA compiler")
        message(STATUS "ampstein: nvcc ${AMPSTEIN_NVCC}")
    endif()
    # GPU architecture(s): default to the installed GPU(s); override with
    # -DCMAKE_CUDA_ARCHITECTURES=86 (e.g. for a build host without a GPU).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
else()
    # --- ROCm/HIP toolchain discovery ------------------------------------
    # Mirrors the CUDA block above, but CMake's HIP language wants the Clang
    # driver directly (amdclang++), not the hipcc wrapper script — passing it
    # the wrapper fails CMakeDetermineHIPCompiler.cmake's sanity check.
    # $ROCM_PATH / $HIP_PATH, /opt/rocm, then the newest /opt/rocm-*.
    if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX})
        file(GLOB _rocm_versioned LIST_DIRECTORIES true /opt/rocm-*)
        list(SORT _rocm_versioned ORDER DESCENDING)
        find_program(AMPSTEIN_HIPCC amdclang++
            HINTS ENV ROCM_PATH ENV HIP_PATH /opt/rocm ${_rocm_versioned}
            PATH_SUFFIXES bin
            DOC "HIP compiler (auto-detected)")
        if(NOT AMPSTEIN_HIPCC)
            message(FATAL_ERROR "amdclang++ not found. Install ROCm or pass "
                "-DCMAKE_HIP_COMPILER=/path/to/amdclang++ (or set ROCM_PATH).")
        endif()
        set(CMAKE_HIP_COMPILER "${AMPSTEIN_HIPCC}" CACHE FILEPATH "HIP compiler")
        message(STATUS "ampstein: HIP compiler ${AMPSTEIN_HIPCC}")
    endif()
    # The ROCm tree's own include/ is only ever added via -idirafter by
    # clang's HIP driver (after /usr/include), so a distro-packaged
    # libamdhip64-dev's /usr/include/hip (if installed) silently shadows it
    # with a mismatched header set. Resolve this build's actual ROCm root
    # from the compiler path and add its include/ with -I (ranks ahead of
    # -idirafter) so our own kernels see one consistent header set. Done
    # unconditionally (not nested in the if() above) so it still runs on a
    # reconfigure where CMAKE_HIP_COMPILER is already cached.
    get_filename_component(_amp_rocm_bin "${CMAKE_HIP_COMPILER}" DIRECTORY)
    get_filename_component(AMPSTEIN_ROCM_ROOT "${_amp_rocm_bin}" DIRECTORY)
    message(STATUS "ampstein: ROCm root ${AMPSTEIN_ROCM_ROOT}")
    # GPU architecture(s): no portable "native" detector for HIP across CMake
    # versions, so default to this project's known target and let
    # -DCMAKE_HIP_ARCHITECTURES=gfxNNNN override for a different card.
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT DEFINED ENV{HIPARCHS})
        set(CMAKE_HIP_ARCHITECTURES "gfx1100")
    endif()
endif()

# --- ucode: hard requirement -------------------------------------------
# The torch.so module links against ucode's public headers (include/ucode/*.h)
# and is dlopen()ed by the VM at runtime. With ucode as a hard project
# requirement, the headers AND the interpreter are both mandatory: the
# interpreter is what runs the ucode layer and the tests.
set(UCODE_INCLUDE_DIR "" CACHE PATH "Path to ucode source tree (contains include/ucode)")
if(DEFINED UCODE_INCLUDE_DIR AND NOT UCODE_INCLUDE_DIR)
    unset(UCODE_INCLUDE_DIR CACHE)
endif()
if(NOT UCODE_INCLUDE_DIR)
    find_path(UCODE_INCLUDE_DIR NAMES include/ucode/module.h
        HINTS "$ENV{HOME}/devel/ucode.git"
        DOC "ucode source tree (contains include/ucode)")
endif()
if(NOT UCODE_INCLUDE_DIR)
    message(FATAL_ERROR "ucode source tree not found. Set -DUCODE_INCLUDE_DIR=<dir containing include/ucode>.")
endif()
include_directories(${UCODE_INCLUDE_DIR}/include)

# Interpreter used to run the test suite; required.
if(DEFINED UCODE_INTERPRETER AND NOT UCODE_INTERPRETER)
    unset(UCODE_INTERPRETER CACHE)
endif()
find_program(UCODE_INTERPRETER NAMES ucode)
if(NOT UCODE_INTERPRETER)
    message(FATAL_ERROR "ucode interpreter not found on PATH. Set -DUCODE_INTERPRETER=<path to ucode>.")
endif()

# --- torch.so (hard requirement, no enablement flag) --------------------
# libtorch + its runtime libs come from the pip wheels, downloaded once into
# a user-level cache (cmake/Libtorch.cmake) and shared by all build trees.
# cuda and rocm wheels are both "torch" with a different local version tag;
# Libtorch.cmake picks the manifest matching AMPSTEIN_GPU_BACKEND.
include(cmake/Libtorch.cmake)

if(AMPSTEIN_GPU_BACKEND STREQUAL "cuda")
    # Link the host toolkit's cudart statically (torch::cudart -> CUDA::cudart_static):
    # the module then has no DT_NEEDED on the build host's libcudart.so.<major>, so an
    # installed prefix is self-contained (libtorch brings its own libcudart.so.12).
    set(CAFFE2_STATIC_LINK_CUDA ON)
    find_package(Torch REQUIRED)
    enable_language(CUDA)
    # vendor/exl3: raw NVPTX inline asm and nvcuda::wmma, CUDA-only. Not built
    # on rocm (this model's GGUF path uses none of it; see docs/GGUF.md /
    # AGENTS.md note on the ROCm backend's scope).
    include(cmake/exl3.cmake)
    set(AMPSTEIN_KERNEL_LANG CUDA)
else()
    # Caffe2Config.cmake's LoadHIP.cmake (pulled in by find_package(Torch))
    # reads this before CMAKE_HIP_ARCHITECTURES is even consulted.
    if(NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
        set(ENV{PYTORCH_ROCM_ARCH} "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    find_package(Torch REQUIRED)
    enable_language(HIP)
    set(AMPSTEIN_KERNEL_LANG HIP)
endif()

# Project-owned fused kernels (RMSNorm, RoPE, GGUF dequant/decode) — NOT vendored code.
# Built as HIP translation units on a rocm configuration (set_source_files_properties
# below); as plain .cu/nvcc otherwise.
add_library(amp_kernels OBJECT src/torch/amp_kernels.cu src/torch/amp_dsv4.cu src/torch/amp_gguf.cu)
target_compile_options(amp_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>)
if(AMPSTEIN_GPU_BACKEND STREQUAL "rocm")
    set_source_files_properties(
        src/torch/amp_kernels.cu src/torch/amp_dsv4.cu src/torch/amp_gguf.cu
        PROPERTIES LANGUAGE HIP)
    target_compile_definitions(amp_kernels PRIVATE AMPSTEIN_ROCM=1)
    # target_include_directories(... BEFORE ...) does not make it into the
    # Unix Makefiles generator's HIP_INCLUDES for an OBJECT library on this
    # CMake (3.31) — a -I compile option does, so take that route instead.
    # (amp_hip_compat.h avoids hip_bf16.h / hip_fp8.h altogether — see its
    # own comment — specifically because a distro-packaged libamdhip64-dev,
    # if installed alongside this ROCm tree, shadows pieces of those two
    # with an older, incompatible header one #include layer down; this -I
    # is only for hip_runtime.h / hip_fp16.h, which aren't affected.)
    target_compile_options(amp_kernels PRIVATE "-I${AMPSTEIN_ROCM_ROOT}/include")
endif()

# NB: target must not be named "torch" — find_package(Torch) imports an
# INTERFACE target with that exact name.
add_library(ucode_torch MODULE
    src/torch/amp_torch.cpp
    src/torch/st.c
    src/torch/gguf.c
    src/torch/tokenizer.c
    src/torch/image.c
    src/torch/amp_tokenizer.cpp
    vendor/exl3/exllamav3_ext/cpu/moe_mul1.cpp
)
# The CPU expert kernels pick their instruction set at run time (per-function target attributes), so no -m flags
# here, but they need an optimizer: the module itself is built without one by default
set_source_files_properties(vendor/exl3/exllamav3_ext/cpu/moe_mul1.cpp PROPERTIES COMPILE_OPTIONS "-O3")
# Build tree mirrors the install layout (<build>/lib/ampstein/torch.so) so the
# tests can use `-L <build>/lib -ltorch=ampstein.torch`.
#
# RPATH: the installed module finds libtorch & friends at
# $ORIGIN/<torch-dist-VER>/torch/lib (the wheels' own RPATHs then locate the
# NVIDIA libs below ../../nvidia), making the install prefix relocatable. The
# build-tree copy points at the cache instead (absolute).
set_target_properties(ucode_torch PROPERTIES
    OUTPUT_NAME torch
    PREFIX ""
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/ampstein"
    BUILD_WITH_INSTALL_RPATH FALSE
    SKIP_BUILD_RPATH FALSE
    BUILD_RPATH "${AMPSTEIN_TORCH_DIST_ROOT}/torch/lib"
    INSTALL_RPATH "\$ORIGIN/${AMPSTEIN_TORCH_DIST_NAME}/torch/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)
set(_amp_exl3_lib)
if(AMPSTEIN_GPU_BACKEND STREQUAL "cuda")
    set(_amp_exl3_lib exl3_kernels)
endif()
target_link_libraries(ucode_torch PRIVATE ${TORCH_LIBRARIES} json-c ${_amp_exl3_lib} amp_kernels pcre2-8)
target_include_directories(ucode_torch PRIVATE src/torch vendor)
target_compile_definitions(ucode_torch PRIVATE AMP_MODULE_TORCH_BUILD)
if(AMPSTEIN_GPU_BACKEND STREQUAL "rocm")
    target_compile_definitions(ucode_torch PRIVATE AMPSTEIN_ROCM=1)
    # c10/cuda/CUDAMacros.h: "We have not yet modified the AMD HIP build to
    # generate this file" (c10/cuda/impl/cuda_cmake_macros.h, a configure_file
    # output the ROCm wheel doesn't ship) -- PyTorch's own documented escape
    # hatch for exactly this.
    target_compile_definitions(ucode_torch PRIVATE C10_CUDA_NO_CMAKE_CONFIGURE_FILE=1)
    # libtorch's ROCm wheel ships c10/cuda, ATen/cuda etc. unmodified from
    # the CUDA build (hipify only rewrites what goes into the wheel's .so,
    # not the installed header tree) -- they still say
    # `#include <cuda_runtime_api.h>` and expect a real CUDA toolkit to
    # provide it. src/torch/rocm_cuda_compat/ stands in; BEFORE so it wins
    # over anything else named cuda_runtime_api.h on the system.
    target_include_directories(ucode_torch BEFORE PRIVATE src/torch/rocm_cuda_compat)
endif()

# --- Tests --------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
# Version reported by torch.version()
set(AMP_VERSION "0.0.1" CACHE STRING "Ampstein version")
target_compile_definitions(ucode_torch PRIVATE AMP_VERSION="${AMP_VERSION}")

# --- install -------------------------------------------------------------
#
#   <prefix>/bin/ampstein                          loader (relocatable)
#   <prefix>/lib/ampstein/torch.so                 native module
#   <prefix>/lib/ampstein/torch-dist-<ver>/        libtorch distribution root:
#       torch/lib/libtorch*.so, libc10*.so         (site-packages layout, so the
#       nvidia/<pkg>/lib/libcudnn*.so, ...          wheels' RPATHs keep working)
#   <prefix>/share/ampstein/*.uc                   ucode engine layer
#
# The loader derives everything from its own location; nothing is baked in.
include(GNUInstallDirs)

# `cmake --install` has no build step of its own; make the install script run
# one first, so `cmake --install <dir>` behaves like `make install` (a no-op
# when everything is already up to date, which is also the case when invoked
# via the `install` target, which depends on `all`).
install(CODE "
    message(STATUS \"ampstein: building before install\")
    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" --build \"${CMAKE_BINARY_DIR}\" --parallel
        RESULT_VARIABLE _amp_build_rc)
    if(NOT _amp_build_rc EQUAL 0)
        message(FATAL_ERROR \"ampstein: build failed, nothing installed\")
    endif()")

install(PROGRAMS ampstein DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ucode_torch LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/ampstein)
install(DIRECTORY src/ucode/ampstein DESTINATION ${CMAKE_INSTALL_DATADIR}
    FILES_MATCHING PATTERN "*.uc" PATTERN "*.jinja")

# The runtime libraries only: shared objects (and their SONAME symlinks), no
# python glue (libtorch_python.so, __init__.py), headers, static / bitcode
# device libraries or cmake configs. With -DAMPSTEIN_INSTALL_TORCH_DIST=OFF a
# symlink to the cache is installed instead — for dev staging; not relocatable
# across machines.
option(AMPSTEIN_INSTALL_TORCH_DIST "Copy the libtorch distribution into the prefix" ON)
set(_torch_dst "${CMAKE_INSTALL_LIBDIR}/ampstein/${AMPSTEIN_TORCH_DIST_NAME}")
if(AMPSTEIN_INSTALL_TORCH_DIST)
    install(DIRECTORY "${AMPSTEIN_TORCH_DIST_ROOT}/torch/lib"
        DESTINATION ${_torch_dst}/torch
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "libtorch_python.so" EXCLUDE
        PATTERN "libshm" EXCLUDE PATTERN "libshm_windows" EXCLUDE
        PATTERN "*.so" PATTERN "*.so.*")
    foreach(_d IN LISTS AMPSTEIN_TORCH_NVIDIA_LIBDIRS)
        get_filename_component(_pkg "${_d}" DIRECTORY)
        get_filename_component(_pkg "${_pkg}" NAME)
        install(DIRECTORY "${_d}" DESTINATION ${_torch_dst}/nvidia/${_pkg}
            USE_SOURCE_PERMISSIONS
            FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
    endforeach()
else()
    install(CODE "
        set(_l \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_torch_dst}\")
        file(REMOVE_RECURSE \"\${_l}\")
        file(CREATE_LINK \"${AMPSTEIN_TORCH_DIST_ROOT}\" \"\${_l}\" SYMBOLIC)")
endif()
