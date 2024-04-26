cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(supported_languages CXX CUDA)
set(CMAKE_CUDA_COMPILER_FORCED True)

if(NOT CMAKE_MAXIMUM_RECURSION_DEPTH)
  set(CMAKE_MAXIMUM_RECURSION_DEPTH 200)
endif()

project(UniRayMarching ${supported_languages})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package( OpenCV REQUIRED )

if ("$ENV{glfw3_DIR}" STREQUAL "")
  message(STATUS "GLFW3 Directory: $ENV{glfw3_DIR}")
  message(FATAL_ERROR "No GLFW3 installation found (glfw3_DIR env variable empty)")
endif()
 
if ("$ENV{lua_DIR}" STREQUAL "")
  message(STATUS "LUA Directory: $ENV{lua_DIR}")
  message(FATAL_ERROR "No LUA installation found (lua_DIR env variable empty)")
endif()


message(STATUS "GLFW3 Directory: $ENV{glfw3_DIR}")
message(STATUS "LUA Directory: $ENV{lua_DIR}")
add_subdirectory("$ENV{glfw3_DIR}" glfw-build)
add_subdirectory("$ENV{lua_DIR}" lua-build)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(UniRayMarching
            src/cudaTexture.cu
            src/main.cu
            src/displayRect.cpp
            # GL
            deps/glad/src/gl.c
            )

target_compile_definitions(UniRayMarching PRIVATE COMMAND_LINE_ARGS_STRINGS="${ARGV}")

target_include_directories(UniRayMarching PUBLIC
                          "${PROJECT_SOURCE_DIR}/deps/glad/include"
                          )

target_link_libraries(UniRayMarching
                      glfw 
                      lua_static
                      ${OpenCV_LIBS}
                      )