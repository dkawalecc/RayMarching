cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(supported_languages CXX CUDA)
set(CMAKE_CUDA_COMPILER_FORCED True)

if(NOT CMAKE_MAXIMUM_RECURSION_DEPTH)
  set(CMAKE_MAXIMUM_RECURSION_DEPTH 200)
endif()

project(simple-triangle ${supported_languages})
project(shaded-triangle ${supported_languages})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

message(STATUS "GLFW3 Directory: $ENV{glfw3_DIR}")
if ("$ENV{glfw3_DIR}" STREQUAL "")
  message(FATAL_ERROR "No GLFW3 installation found (glfw3_DIR env variable empty)")
endif()
add_subdirectory("$ENV{glfw3_DIR}" glfw-build)

add_executable(simple-triangle
            simple-triangle.cu
            # GL
            ../deps/glad/src/gl.c
            )

add_executable(shaded-triangle
            shaded-triangle.cu
            # GL
            ../deps/glad/src/gl.c
)

target_include_directories(simple-triangle PUBLIC
                          "${PROJECT_SOURCE_DIR}/../deps/glad/include"
                          )

target_include_directories(shaded-triangle PUBLIC
                          "${PROJECT_SOURCE_DIR}/../deps/glad/include"
                          )

target_link_libraries(simple-triangle
                      glfw 
                      )

target_link_libraries(shaded-triangle
                      glfw 
                      )