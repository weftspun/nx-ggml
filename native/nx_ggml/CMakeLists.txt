cmake_minimum_required(VERSION 3.14)
project(nx_ggml_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 1: CPU-only. GGML_VULKAN is turned on in Phase 7 once op coverage
# is proven correct on CPU.
option(NX_GGML_VULKAN "nx_ggml: enable ggml's Vulkan compute backend" OFF)

set(GGML_VULKAN ${NX_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

# Throwaway Phase 1 harness: proves ggml's CPU compute backend builds and
# computes correctly on this machine, before any NIF/Elixir code exists.
# Not part of the eventual NIF build.
add_executable(test_ggml_cpu test_ggml_cpu.cpp)
target_link_libraries(test_ggml_cpu PRIVATE ggml)
