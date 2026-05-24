cmake_minimum_required(VERSION 3.17)

project(gpu_comm VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINIGPU_BUILD_PYTHON "Build Python bindings" ON)
option(MINIGPU_BUILD_TESTS "Build Mini-GPU runtime tests" ON)
option(MINIGPU_BUILD_TORCH "Build PyTorch PrivateUse1 backend stubs" ON)
option(MINIGPU_BUILD_KERNELS "Add a target to rebuild Mini-GPU kernel artifacts" ON)

find_package(yaml-cpp REQUIRED)

if(MINIGPU_BUILD_PYTHON OR MINIGPU_BUILD_TORCH OR MINIGPU_BUILD_KERNELS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

if(MINIGPU_BUILD_PYTHON OR MINIGPU_BUILD_TORCH)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
    find_package(pybind11 REQUIRED)
endif()

if(MINIGPU_BUILD_TORCH)
    find_package(Torch REQUIRED)
    separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/runtime/src/generated/kernel_id_map.hpp
           ${CMAKE_CURRENT_SOURCE_DIR}/runtime/src/generated/kernel_hashes.hpp
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/runtime/scripts/generate_kernel_metadata.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernels/kernels.yaml
    COMMENT "Generating kernel metadata..."
)

add_library(minigpu_runtime
    runtime/src/minigpu_runtime.cpp
    runtime/src/minigpu_kernels.cpp
)

set_target_properties(minigpu_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(minigpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)

target_compile_definitions(minigpu_runtime
    PRIVATE
        MINIGPU_DEFAULT_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernels"
)

target_link_libraries(minigpu_runtime
    PUBLIC
        yaml-cpp::yaml-cpp
)

if(MINIGPU_BUILD_KERNELS)
    file(GLOB MINIGPU_KERNEL_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernels/src/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernels/include/*.h
    )

    add_custom_target(minigpu_kernels
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/build_kernel_artifacts.py
            --kernel-root ${CMAKE_CURRENT_SOURCE_DIR}/runtime/kernels
        DEPENDS
            ${MINIGPU_KERNEL_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/build_kernel_artifacts.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/minigpucc.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/get_cuda_ast.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/emit_llvm_ir.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/llvm_to_ir.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/ir_to_isa.py
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler/isa_to_bin.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Rebuilding registered Mini-GPU kernel artifacts"
    )
endif()

add_library(comm
    gpu_comm/src/com.c
)

set_target_properties(comm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(comm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm/src
)

add_library(minigpu_runtime_gpu_comm
    runtime/src/transports/gpu_comm_transport.cpp
)

set_target_properties(minigpu_runtime_gpu_comm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(minigpu_runtime_gpu_comm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm/src
)

target_link_libraries(minigpu_runtime_gpu_comm
    PUBLIC
        minigpu_runtime
        comm
)

if(MINIGPU_BUILD_PYTHON)
    pybind11_add_module(_gpu_comm
        gpu_comm/bindings/python_bindings.cpp
    )

    target_include_directories(_gpu_comm
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm/src
    )

    target_link_libraries(_gpu_comm
        PRIVATE
            comm
    )

    set_target_properties(_gpu_comm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm
    )
endif()

if(MINIGPU_BUILD_TORCH)
    pybind11_add_module(minigpu_torch
        torch_ext/src/backend/state.cpp
        torch_ext/src/backend/transport.cpp
        torch_ext/src/bindings/module.cpp
        torch_ext/src/ops/elementwise.cpp
        torch_ext/src/ops/math.cpp
        torch_ext/src/ops/tensor.cpp
        torch_ext/src/registrations/aten.cpp
        torch_ext/src/registrations/custom.cpp
    )

    target_include_directories(minigpu_torch
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext/include
    )

    target_link_libraries(minigpu_torch
        PRIVATE
            minigpu_runtime_gpu_comm
            ${TORCH_LIBRARIES}
    )

    target_compile_options(minigpu_torch
        PRIVATE
            ${TORCH_CXX_FLAGS_LIST}
    )

    set_target_properties(minigpu_torch PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/torch_mini_gpu
    )
endif()
