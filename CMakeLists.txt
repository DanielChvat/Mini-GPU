cmake_minimum_required(VERSION 3.17)

project(gpu_comm VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINIGPU_BUILD_PYTHON "Build Python bindings" ON)
option(MINIGPU_BUILD_TESTS "Build Mini-GPU runtime tests" ON)
option(MINIGPU_BUILD_TORCH "Build PyTorch PrivateUse1 backend stubs" ON)

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

add_library(minigpu_runtime
    runtime/src/minigpu_runtime.cpp
)

set_target_properties(minigpu_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(minigpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)

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
    pybind11_add_module(_C
        torch_ext/src/minigpu_torch.cpp
    )

    target_include_directories(_C
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext/include
    )

    target_link_libraries(_C
        PRIVATE
            minigpu_runtime
            ${TORCH_LIBRARIES}
    )

    target_compile_options(_C
        PRIVATE
            ${TORCH_CXX_FLAGS_LIST}
    )

    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/torch_mini_gpu
    )
endif()

if(MINIGPU_BUILD_TESTS)
    enable_testing()

    add_executable(minigpu_runtime_smoke
        test/runtime/runtime_smoke.cpp
    )

    target_link_libraries(minigpu_runtime_smoke
        PRIVATE
            minigpu_runtime
    )

    add_test(NAME minigpu_runtime_smoke COMMAND minigpu_runtime_smoke)
endif()
