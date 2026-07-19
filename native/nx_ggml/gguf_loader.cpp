// One-shot GGUF weight loading, separate from GraphBuilder/CompiledGraph:
// this is a "read weights off disk into an Elixir binary" utility, not part
// of the compute-graph machinery. Backed directly by ggml's own gguf.h API
// (gguf_init_from_file with no_alloc=false populates a real ggml_context
// with tensor data already loaded), so there's no hand-rolled GGUF parser
// to get wrong.

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fine.hpp>

#include "ggml.h"
#include "gguf.h"

namespace {

// Reverses ggml's ne[] (fastest-varying-first) into Nx's row-major shape
// (last axis innermost) -- the same convention graph_builder.cpp's
// to_ggml_ne uses, just in the opposite direction.
std::vector<int64_t> from_ggml_ne(const ggml_tensor *t) {
  int n = ggml_n_dims(t);
  std::vector<int64_t> shape(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    shape[static_cast<size_t>(n - 1 - i)] = t->ne[i];
  }
  return shape;
}

// gguf_init_from_file(no_alloc=false) loads *every* tensor's data for the
// whole file, not just the one being asked for -- fine for a handful of
// calls, but real per-layer validation scripts call this dozens of times
// per file (e.g. 21 tensors x 30 blocks against one 5GB GGUF), which would
// otherwise re-read and re-allocate the entire file from disk on every
// single call. Cached by path for the life of the process instead: these
// are read-only reference/weight files that don't change during a run, so
// keeping the parsed context alive and never freeing it is a correct
// trade of (bounded, model-file-sized) memory for avoiding that redundant
// I/O -- appropriate for this NIF's actual use (validation scripts loading
// a handful of distinct files), not a general-purpose cache eviction
// policy a long-lived server would need.
struct LoadedGguf {
  gguf_context *gguf;
  ggml_context *ctx;
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, LoadedGguf> g_cache;

const ggml_tensor *cached_tensor(const std::string &path, const std::string &tensor_name) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);

  auto it = g_cache.find(path);
  if (it == g_cache.end()) {
    ggml_context *ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/false,
        /*.ctx      =*/&ctx,
    };

    struct gguf_context *gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
      throw std::runtime_error("nx_ggml: failed to open GGUF file '" + path + "'");
    }

    it = g_cache.emplace(path, LoadedGguf{gguf, ctx}).first;
  }

  return it->second.ctx ? ggml_get_tensor(it->second.ctx, tensor_name.c_str()) : nullptr;
}

} // namespace

// Reads a single f32 tensor by name out of a GGUF file. Raises if the file
// can't be opened, the tensor doesn't exist, or it isn't f32 (this is a
// deliberately narrow utility for loading models already converted with
// --ftype 0 / f32, not a general dtype-converting GGUF reader).
fine::Term nx_ggml_gguf_read_f32(ErlNifEnv *env, std::string path, std::string tensor_name) {
  const ggml_tensor *t = cached_tensor(path, tensor_name);
  if (!t) {
    throw std::runtime_error("nx_ggml: tensor '" + tensor_name + "' not found in '" + path + "'");
  }

  if (t->type != GGML_TYPE_F32) {
    throw std::runtime_error("nx_ggml: tensor '" + tensor_name +
                              "' is not f32 (convert with --ftype 0)");
  }

  std::vector<int64_t> shape = from_ggml_ne(t);
  std::string data(static_cast<const char *>(t->data), ggml_nbytes(t));

  return fine::encode(env, std::make_tuple(shape, fine::Term(fine::make_new_binary(
                                                       env, data.data(), data.size()))));
}
FINE_NIF(nx_ggml_gguf_read_f32, 0);
