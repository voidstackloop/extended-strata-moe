cmake_minimum_required(VERSION 3.14)
project(routing_trace CXX)

# llama.cpp with the strata-moe patch: inside this repo (scripts/setup.sh), else beside it; override with -DLLAMA_DIR=...
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/CMakeLists.txt")
    set(LLAMA_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp")
else()
    set(LLAMA_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../../../llama.cpp")
endif()
set(LLAMA_DIR "${LLAMA_DIR_DEFAULT}" CACHE PATH "llama.cpp source")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)   # static: read_profile links an internal llama symbol
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(routing_trace routing_trace.cpp)
target_link_libraries(routing_trace PRIVATE llama ggml)
target_compile_features(routing_trace PRIVATE cxx_std_17)

add_executable(read_profile read_profile.cpp)
target_include_directories(read_profile PRIVATE ${LLAMA_DIR}/src)
target_link_libraries(read_profile PRIVATE llama ggml)
target_compile_features(read_profile PRIVATE cxx_std_17)
