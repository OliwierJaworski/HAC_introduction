cmake_minimum_required(VERSION 3.10.0)
project(C_DeviceProperties VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(C_DeviceProperties main.cu)

target_link_libraries(C_DeviceProperties PRIVATE CUDA::cudart)

set_target_properties(C_DeviceProperties PROPERTIES CUDA_SEPARABLE_COMPILATION ON)