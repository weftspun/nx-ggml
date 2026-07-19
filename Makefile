# Invoked by elixir_make (see mix.exs) during `mix compile`.
#
# Phase 0: no native code yet, so this is a no-op that only makes sure
# priv/ exists. From Phase 1 onward this will shell out to CMake to build
# native/nx_ggml (which links ggml's CPU/Vulkan backends) into a NIF
# shared library copied into priv/.

PRIV_DIR ?= priv

.PHONY: all
all:
	@echo "nx_ggml: no native build yet (Phase 0 bootstrap)"
	@mkdir -p $(PRIV_DIR)
