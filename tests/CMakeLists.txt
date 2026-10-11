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
