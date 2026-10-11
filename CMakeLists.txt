cmake_minimum_required(VERSION 3.16)
project(Ampstein C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- CUDA toolchain discovery --------------------------------------------
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
# libtorch + NVIDIA runtime libs come from the pip wheels, downloaded once into
# a user-level cache (cmake/Libtorch.cmake) and shared by all build trees.
include(cmake/Libtorch.cmake)
# Link the host toolkit's cudart statically (torch::cudart -> CUDA::cudart_static):
# the module then has no DT_NEEDED on the build host's libcudart.so.<major>, so an
# installed prefix is self-contained (libtorch brings its own libcudart.so.12).
set(CAFFE2_STATIC_LINK_CUDA ON)
find_package(Torch REQUIRED)
enable_language(CUDA)
include(cmake/exl3.cmake)

# Project-owned fused kernels (RMSNorm, RoPE) — NOT vendored code.
add_library(amp_kernels OBJECT src/torch/amp_kernels.cu)
target_compile_options(amp_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# NB: target must not be named "torch" — find_package(Torch) imports an
# INTERFACE target with that exact name.
add_library(ucode_torch MODULE
    src/torch/amp_torch.cpp
    src/torch/st.c
    src/torch/tokenizer.c
    src/torch/image.c
    src/torch/amp_tokenizer.cpp
)
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
target_link_libraries(ucode_torch PRIVATE ${TORCH_LIBRARIES} json-c exl3_kernels amp_kernels pcre2-8)
target_include_directories(ucode_torch PRIVATE src/torch vendor)
target_compile_definitions(ucode_torch PRIVATE AMP_MODULE_TORCH_BUILD)

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
    FILES_MATCHING PATTERN "*.uc")

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
