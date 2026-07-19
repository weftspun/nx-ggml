// One-shot GGUF weight loading, separate from GraphBuilder/CompiledGraph:
// this is a "read weights off disk into an Elixir binary" utility, not part
// of the compute-graph machinery. Backed directly by ggml's own gguf.h API
// (gguf_init_from_file with no_alloc=false populates a real ggml_context
// with tensor data already loaded), so there's no hand-rolled GGUF parser
// to get wrong.

#include <stdexcept>
#include <string>
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

} // namespace

// Reads a single f32 tensor by name out of a GGUF file. Raises if the file
// can't be opened, the tensor doesn't exist, or it isn't f32 (this is a
// deliberately narrow utility for loading models already converted with
// --ftype 0 / f32, not a general dtype-converting GGUF reader).
fine::Term nx_ggml_gguf_read_f32(ErlNifEnv *env, std::string path, std::string tensor_name) {
  ggml_context *ctx = nullptr;
  struct gguf_init_params params = {
      /*.no_alloc =*/false,
      /*.ctx      =*/&ctx,
  };

  struct gguf_context *gguf = gguf_init_from_file(path.c_str(), params);
  if (!gguf) {
    throw std::runtime_error("nx_ggml: failed to open GGUF file '" + path + "'");
  }

  struct ggml_tensor *t = ctx ? ggml_get_tensor(ctx, tensor_name.c_str()) : nullptr;
  if (!t) {
    gguf_free(gguf);
    if (ctx) ggml_free(ctx);
    throw std::runtime_error("nx_ggml: tensor '" + tensor_name + "' not found in '" + path + "'");
  }

  if (t->type != GGML_TYPE_F32) {
    gguf_free(gguf);
    ggml_free(ctx);
    throw std::runtime_error("nx_ggml: tensor '" + tensor_name +
                              "' is not f32 (convert with --ftype 0)");
  }

  std::vector<int64_t> shape = from_ggml_ne(t);
  std::string data(static_cast<const char *>(t->data), ggml_nbytes(t));

  gguf_free(gguf);
  ggml_free(ctx);

  return fine::encode(env, std::make_tuple(shape, fine::Term(fine::make_new_binary(
                                                       env, data.data(), data.size()))));
}
FINE_NIF(nx_ggml_gguf_read_f32, 0);
