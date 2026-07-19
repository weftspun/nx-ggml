// NIF entrypoint, built on Fine (https://github.com/elixir-nx/fine) instead
// of hand-rolled erl_nif.h boilerplate.
//
// This file owns the persistent backend registry (CPU always; Vulkan too
// when the native build was configured with NX_GGML_VULKAN=ON, i.e.
// NX_GGML_HAVE_VULKAN is defined -- see CMakeLists.txt), exposed to other
// translation units via nx_ggml_backend_for(device), plus one smoke-test
// NIF (cpu_add_f32) kept from Phase 2. The real Nx.Defn.Expr ->
// ggml_cgraph lowering lives in graph_builder.cpp (Phase 3+).

#include <stdexcept>
#include <string>
#include <string_view>

#include <fine.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef NX_GGML_HAVE_VULKAN
#include "ggml-vulkan.h"
#endif

namespace {
ggml_backend_t g_cpu_backend = nullptr;
#ifdef NX_GGML_HAVE_VULKAN
ggml_backend_t g_vulkan_backend = nullptr;
#endif
} // namespace

ggml_backend_t nx_ggml_cpu_backend() {
  if (g_cpu_backend == nullptr) {
    throw std::runtime_error("nx_ggml: CPU backend not initialized");
  }
  return g_cpu_backend;
}

ggml_backend_t nx_ggml_backend_for(const std::string &device) {
  if (device == "cpu") {
    return nx_ggml_cpu_backend();
  }

  if (device == "vulkan") {
#ifdef NX_GGML_HAVE_VULKAN
    if (g_vulkan_backend == nullptr) {
      throw std::runtime_error(
          "nx_ggml: Vulkan backend not initialized (no Vulkan-capable device found?)");
    }
    return g_vulkan_backend;
#else
    throw std::invalid_argument(
        "nx_ggml: this build was compiled without Vulkan support "
        "(rebuild with NX_GGML_VULKAN=ON)");
#endif
  }

  throw std::invalid_argument("nx_ggml: unknown device '" + device + "'");
}

// Elementwise f32 add over raw binaries, computed via ggml's CPU backend.
// Both binaries must have the same size and be a whole number of f32
// elements. Kept as a standalone smoke test from Phase 2; the general
// Expr-driven path is GraphBuilder/CompiledGraph in graph_builder.cpp.
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
      ggml_backend_alloc_ctx_tensors(ctx, nx_ggml_cpu_backend());
  if (!buffer) {
    ggml_free(ctx);
    throw std::runtime_error("cpu_add_f32: ggml_backend_alloc_ctx_tensors failed");
  }

  ggml_backend_tensor_set(ta, a.data(), 0, a.size());
  ggml_backend_tensor_set(tb, b.data(), 0, b.size());

  enum ggml_status status = ggml_backend_graph_compute(nx_ggml_cpu_backend(), graph);
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

// Returns true if the "vulkan" device is actually usable (compiled in and
// a backend was successfully initialized), so Elixir tests can skip
// gracefully on machines without a Vulkan-capable GPU instead of failing.
bool nx_ggml_vulkan_available(ErlNifEnv *) {
#ifdef NX_GGML_HAVE_VULKAN
  return g_vulkan_backend != nullptr;
#else
  return false;
#endif
}
FINE_NIF(nx_ggml_vulkan_available, 0);

namespace {

auto load_registration =
    fine::Registration::register_load([](ErlNifEnv *, void **, fine::Term) {
      g_cpu_backend = ggml_backend_cpu_init();
      if (g_cpu_backend == nullptr) {
        throw std::runtime_error("nx_ggml: ggml_backend_cpu_init failed");
      }

#ifdef NX_GGML_HAVE_VULKAN
      // Best-effort: a machine without a Vulkan-capable GPU simply leaves
      // g_vulkan_backend null; requesting device: :vulkan then raises a
      // clear error (nx_ggml_backend_for) rather than crashing NIF load.
      g_vulkan_backend = ggml_backend_vk_init(0);
#endif
    });

auto unload_registration =
    fine::Registration::register_unload([](ErlNifEnv *, void *) noexcept {
      if (g_cpu_backend != nullptr) {
        ggml_backend_free(g_cpu_backend);
        g_cpu_backend = nullptr;
      }
#ifdef NX_GGML_HAVE_VULKAN
      if (g_vulkan_backend != nullptr) {
        ggml_backend_free(g_vulkan_backend);
        g_vulkan_backend = nullptr;
      }
#endif
    });

} // namespace

FINE_INIT("Elixir.NxGgml.Nif");
