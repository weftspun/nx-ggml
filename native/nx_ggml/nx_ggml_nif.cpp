// Phase 2 NIF skeleton, built on Fine (https://github.com/elixir-nx/fine)
// instead of hand-rolled erl_nif.h boilerplate.
//
// This wires up: a persistent CPU ggml_backend_t created once at NIF load
// (backend_registry), and one real smoke-test NIF (cpu_add_f32) that
// exercises the exact ggml-backend call sequence (context -> graph ->
// ggml_backend_alloc_ctx_tensors -> ggml_backend_tensor_set/get ->
// ggml_backend_graph_compute) that Phase 3's expr_lowering/graph_cache will
// generalize into a full Nx.Defn.Expr -> ggml_cgraph lowering with proper
// shape/dtype-signature caching. For now each call builds and tears down
// its own tiny context, which is correct but not yet cached.

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fine.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

ggml_backend_t g_cpu_backend = nullptr;

ggml_backend_t cpu_backend() {
  if (g_cpu_backend == nullptr) {
    throw std::runtime_error("nx_ggml: CPU backend not initialized");
  }
  return g_cpu_backend;
}

} // namespace

// Elementwise f32 add over raw binaries, computed via ggml's CPU backend.
// Both binaries must have the same size and be a whole number of f32
// elements (this is a smoke test, not the general dtype/shape-checked
// lowering that lands in Phase 3).
fine::Term cpu_add_f32(ErlNifEnv *env, std::string_view a, std::string_view b) {
  if (a.size() != b.size() || a.size() % sizeof(float) != 0) {
    throw std::invalid_argument(
        "cpu_add_f32: inputs must be equal-length f32 binaries");
  }

  const int64_t n = static_cast<int64_t>(a.size() / sizeof(float));

  struct ggml_init_params params = {
      /*.mem_size   =*/ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  struct ggml_context *ctx = ggml_init(params);
  if (!ctx) {
    throw std::runtime_error("cpu_add_f32: ggml_init failed");
  }

  struct ggml_tensor *ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  struct ggml_tensor *tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  struct ggml_tensor *sum = ggml_add(ctx, ta, tb);

  struct ggml_cgraph *graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, sum);

  struct ggml_backend_buffer *buffer =
      ggml_backend_alloc_ctx_tensors(ctx, cpu_backend());
  if (!buffer) {
    ggml_free(ctx);
    throw std::runtime_error("cpu_add_f32: ggml_backend_alloc_ctx_tensors failed");
  }

  ggml_backend_tensor_set(ta, a.data(), 0, a.size());
  ggml_backend_tensor_set(tb, b.data(), 0, b.size());

  enum ggml_status status = ggml_backend_graph_compute(cpu_backend(), graph);
  if (status != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    throw std::runtime_error("cpu_add_f32: ggml_backend_graph_compute failed");
  }

  std::string result(a.size(), '\0');
  ggml_backend_tensor_get(sum, result.data(), 0, result.size());

  ggml_backend_buffer_free(buffer);
  ggml_free(ctx);

  return fine::make_new_binary(env, result.data(), result.size());
}

FINE_NIF(cpu_add_f32, 0);

namespace {

auto load_registration =
    fine::Registration::register_load([](ErlNifEnv *, void **, fine::Term) {
      g_cpu_backend = ggml_backend_cpu_init();
      if (g_cpu_backend == nullptr) {
        throw std::runtime_error("nx_ggml: ggml_backend_cpu_init failed");
      }
    });

auto unload_registration =
    fine::Registration::register_unload([](ErlNifEnv *, void *) noexcept {
      if (g_cpu_backend != nullptr) {
        ggml_backend_free(g_cpu_backend);
        g_cpu_backend = nullptr;
      }
    });

} // namespace

FINE_INIT("Elixir.NxGgml.Nif");
