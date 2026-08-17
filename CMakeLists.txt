cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "PyTorch CUDA architectures")
endif()

project(cuda_nvfp4_decoder_attention LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

set(
    CUDA_PRIMITIVES_PYTORCH_CUDA_VERSION
    ""
    CACHE STRING
    "CUDA version used to build the installed PyTorch"
)
if(
    CUDA_PRIMITIVES_PYTORCH_CUDA_VERSION
    AND NOT CUDA_PRIMITIVES_PYTORCH_CUDA_VERSION VERSION_EQUAL CUDAToolkit_VERSION
)
    message(WARNING
        "PyTorch was built with CUDA ${CUDA_PRIMITIVES_PYTORCH_CUDA_VERSION}, "
        "while this extension uses CUDA Toolkit ${CUDAToolkit_VERSION}. "
        "This is a minor-version difference and must be validated on the target GPU."
    )
endif()

add_library(
    cuda_primitives
    SHARED
    src/bindings.cpp
    src/nvfp4.cu
    src/rmsnorm.cu
    src/rope.cu
    src/w4a16.cu
    src/w4a16_grouped.cu
)

target_include_directories(
    cuda_primitives
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(
    cuda_primitives
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(cuda_primitives PRIVATE ${TORCH_CXX_FLAGS_LIST})

set_target_properties(
    cuda_primitives
    PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        CUDA_ARCHITECTURES 89
        CUDA_RUNTIME_LIBRARY Shared
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        POSITION_INDEPENDENT_CODE ON
        PREFIX ""
)

message(STATUS "CUDA primitives build type: ${CMAKE_BUILD_TYPE}")
get_target_property(
    CUDA_PRIMITIVES_TARGET_ARCHITECTURES
    cuda_primitives
    CUDA_ARCHITECTURES
)
message(STATUS "CUDA primitives target architectures: ${CUDA_PRIMITIVES_TARGET_ARCHITECTURES}")
message(STATUS "PyTorch prefix: ${TORCH_INSTALL_PREFIX}")
