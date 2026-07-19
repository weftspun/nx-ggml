# Invoked by elixir_make (see mix.exs) during `mix compile`.
#
# Shells out to CMake to build native/nx_ggml: ggml (vendored via git
# subtree, linked statically) plus the Fine-based NIF, landing the result
# directly in $(MIX_APP_PATH)/priv as libnx_ggml.{so,dll,dylib}.
#
# elixir_make provides MIX_APP_PATH and ERTS_INCLUDE_DIR; mix.exs's
# make_env provides FINE_INCLUDE_DIR (Fine.include_dir()).

CMAKE ?= cmake
NX_GGML_CMAKE_GENERATOR ?= Ninja
NX_GGML_VULKAN ?= OFF

PRIV_DIR := $(MIX_APP_PATH)/priv
BUILD_DIR := native/nx_ggml/build

.PHONY: all
all:
	@mkdir -p "$(PRIV_DIR)"
ifeq ($(OS),Windows_NT)
	@# Real MSVC (cl/link) is required on Windows, not Clang: a minimal
	@# repro proved Clang/lld-link does not run the C++ global static
	@# initializers Fine's NIF registration depends on for a MODULE-type
	@# DLL (load_nif then silently binds zero functions). Implemented as a
	@# PowerShell script (not a .bat via `cmd /c`) because invoking cmd.exe
	@# from this MSYS/git-bash Makefile shell mangles path-like arguments;
	@# PowerShell has no such mangling. See native/nx_ggml/build_msvc.ps1.
	powershell -NoProfile -ExecutionPolicy Bypass -File native/nx_ggml/build_msvc.ps1 \
		-BuildDir "$(BUILD_DIR)" \
		-ErtsIncludeDir "$(ERTS_INCLUDE_DIR)" \
		-FineIncludeDir "$(FINE_INCLUDE_DIR)" \
		-PrivDir "$(PRIV_DIR)" \
		-NxGgmlVulkan "$(NX_GGML_VULKAN)"
else
	$(CMAKE) -S native/nx_ggml -B "$(BUILD_DIR)" -G "$(NX_GGML_CMAKE_GENERATOR)" \
		-DCMAKE_BUILD_TYPE=Release \
		-DERTS_INCLUDE_DIR="$(ERTS_INCLUDE_DIR)" \
		-DFINE_INCLUDE_DIR="$(FINE_INCLUDE_DIR)" \
		-DNX_GGML_PRIV_DIR="$(PRIV_DIR)" \
		-DNX_GGML_VULKAN="$(NX_GGML_VULKAN)"
	$(CMAKE) --build "$(BUILD_DIR)" --target nx_ggml_nif --config Release
endif
