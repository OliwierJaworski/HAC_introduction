cmake_minimum_required(VERSION 3.5.0)
project(MaxMinPool VERSION 0.1.0 LANGUAGES C CXX CUDA)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

add_executable(MaxMinPool  main.cpp
                                ${SRC_FILES}
                                ${CU_SRC_FILES}
                                ${HEADER_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} 
    COMMON_LIBS)

