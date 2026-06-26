cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpufl_tutorial_02 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gpufl_tutorial_02 src/multiple_streams.cu)
add_executable(gpufl_tutorial_02_single_stream src/single_stream.cu)
add_executable(gpufl_tutorial_02_explicit_streams src/explicit_streams.cu)
add_executable(gpufl_tutorial_02_occupancy src/occupancy_example.cu)
add_executable(gpufl_tutorial_02_block_style src/block_style_example.cu)

foreach(target
        gpufl_tutorial_02
        gpufl_tutorial_02_single_stream
        gpufl_tutorial_02_explicit_streams
        gpufl_tutorial_02_occupancy
        gpufl_tutorial_02_block_style)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
endforeach()
