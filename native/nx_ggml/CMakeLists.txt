cmake_minimum_required(VERSION 3.14)
project(nx_ggml_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 1: CPU-only. GGML_VULKAN is turned on in Phase 7 once op coverage
# is proven correct on CPU.
option(NX_GGML_VULKAN "nx_ggml: enable ggml's Vulkan compute backend" OFF)

# Everything (ggml + the NIF) is linked into a single shared library so
# there's exactly one binary for the BEAM to load, with no sibling ggml
# DLLs to locate at runtime.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
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

# --- NIF (Phase 2+) -----------------------------------------------------
#
# ERTS_INCLUDE_DIR and FINE_INCLUDE_DIR are forwarded from mix.exs/Makefile
# (elixir_make sets the former automatically; make_env sets the latter via
# Fine.include_dir()). NX_GGML_PRIV_DIR is where elixir_make expects the
# compiled NIF to land (MIX_APP_PATH/priv); default it for standalone
# `cmake` invocations done outside of `mix compile`.
if(NOT DEFINED NX_GGML_PRIV_DIR)
    set(NX_GGML_PRIV_DIR "${CMAKE_BINARY_DIR}/priv")
endif()

add_library(nx_ggml_nif MODULE nx_ggml_nif.cpp)
set_target_properties(nx_ggml_nif PROPERTIES
    OUTPUT_NAME "nx_ggml"
    PREFIX "lib"
    RUNTIME_OUTPUT_DIRECTORY "${NX_GGML_PRIV_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${NX_GGML_PRIV_DIR}"
)
target_include_directories(nx_ggml_nif PRIVATE "${ERTS_INCLUDE_DIR}" "${FINE_INCLUDE_DIR}")
# ggml is linked statically (BUILD_SHARED_LIBS=OFF above), so the NIF
# shared library is fully self-contained.
target_link_libraries(nx_ggml_nif PRIVATE ggml)

if(NOT MSVC)
    target_compile_options(nx_ggml_nif PRIVATE -fvisibility=hidden)
endif()
