cmake_minimum_required(VERSION 3.5.0)
project(Convolution VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(COMMON_LIBS INTERFACE)

target_link_libraries(COMMON_LIBS INTERFACE CUDA::cudart)

include_directories(COMMON_LIBS INTERFACE "extensions/stb/")

set(SUB_DIRS 
    "${CMAKE_CURRENT_SOURCE_DIR}/Convolution"
    "${CMAKE_CURRENT_SOURCE_DIR}/MaxMinPooling")
foreach(DIR IN LISTS SUB_DIRS)
    if (EXISTS "${DIR}" AND EXISTS "${DIR}/CMakeLists.txt")
        add_subdirectory(${DIR})
    else()
        message(WARNING "my_add_subdirectory: Can't find ${DIR}/CMakeLists.txt")
    endif()
endforeach()

set_target_properties(COMMON_LIBS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


