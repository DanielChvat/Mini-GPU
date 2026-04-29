cmake_minimum_required(VERSION 3.17)

project(uart_gpu_comm VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(uart
    gpu_comm/src/com.c
)

target_include_directories(uart
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm/src
)

pybind11_add_module(_gpu_comm
    gpu_comm/bindings/python_bindings.cpp
)

target_include_directories(_gpu_comm
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm/src
)

target_link_libraries(_gpu_comm
    PRIVATE
        uart
)

set_target_properties(_gpu_comm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gpu_comm
)