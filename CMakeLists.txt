cmake_minimum_required(VERSION 3.22.1)
project(cuda_practice CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

add_executable(cuda_practice main.cu
        kernel.cuh
        blob.cuh
        utils.cuh
)

#target_link_libraries(cuda_practice cudnn cublas)
