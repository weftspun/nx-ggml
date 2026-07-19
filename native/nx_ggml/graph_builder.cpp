#include "graph_builder.h"

#include <stdexcept>

#include "ggml-alloc.h"

namespace {

// ggml's ne[] is fastest-varying-first (ne[0] innermost), the reverse of
// Nx's row-major shape (last axis innermost). Reversing here keeps flat
// memory layout consistent between Nx binaries and ggml tensors for every
// op that follows (matters once reshape/broadcast/matmul land in later
// phases; for the pure elementwise ops in Phase 3 it doesn't yet affect
// correctness, but getting the convention right now avoids rework).
std::vector<int64_t> to_ggml_ne(const std::vector<int64_t> &shape) {
  std::vector<int64_t> ne(GGML_MAX_DIMS, 1);
  int64_t n = static_cast<int64_t>(shape.size());
  if (n > GGML_MAX_DIMS) {
    throw std::invalid_argument("nx_ggml: tensor rank exceeds ggml's GGML_MAX_DIMS (4)");
  }
  for (int64_t i = 0; i < n; i++) {
    ne[i] = shape[static_cast<size_t>(n - 1 - i)];
  }
  return ne;
}

int64_t element_count(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

} // namespace

GraphBuilder::GraphBuilder() {
  struct ggml_init_params params = {
      /*.mem_size   =*/ggml_tensor_overhead() * 1024 + ggml_graph_overhead(),
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/true,
  };
  ctx_ = ggml_init(params);
  if (!ctx_) {
    throw std::runtime_error("nx_ggml: ggml_init failed while building a graph");
  }
}

GraphBuilder::~GraphBuilder() {
  if (ctx_) {
    ggml_free(ctx_);
  }
}

ggml_tensor *GraphBuilder::new_leaf_f32(const std::vector<int64_t> &shape) {
  auto ne = to_ggml_ne(shape);
  return ggml_new_tensor(ctx_, GGML_TYPE_F32, GGML_MAX_DIMS, ne.data());
}

int64_t GraphBuilder::push(ggml_tensor *tensor) {
  nodes_.push_back(tensor);
  return static_cast<int64_t>(nodes_.size() - 1);
}

ggml_tensor *GraphBuilder::node(int64_t index) const {
  if (index < 0 || static_cast<size_t>(index) >= nodes_.size()) {
    throw std::out_of_range("nx_ggml: invalid graph node index");
  }
  return nodes_[static_cast<size_t>(index)];
}

int64_t GraphBuilder::add_param(const std::vector<int64_t> &shape) {
  ggml_tensor *t = new_leaf_f32(shape);
  params_.push_back(t);
  return push(t);
}

int64_t GraphBuilder::add_constant_f32(const std::vector<int64_t> &shape,
                                        const std::string &data) {
  int64_t n = element_count(shape);
  if (static_cast<int64_t>(data.size()) != n * static_cast<int64_t>(sizeof(float))) {
    throw std::invalid_argument(
        "nx_ggml: constant binary size does not match shape (expected f32 elements)");
  }
  ggml_tensor *t = new_leaf_f32(shape);
  pending_constants_.emplace_back(t, data);
  return push(t);
}

int64_t GraphBuilder::add_add(int64_t a, int64_t b) {
  ggml_tensor *result = ggml_add(ctx_, node(a), node(b));
  return push(result);
}

CompiledGraph::CompiledGraph(GraphBuilder &builder, int64_t output_index) {
  output_ = builder.node(output_index);

  ggml_cgraph *graph = ggml_new_graph(builder.ctx_);
  ggml_build_forward_expand(graph, output_);

  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(builder.ctx_, nx_ggml_cpu_backend());
  if (!buffer) {
    throw std::runtime_error("nx_ggml: ggml_backend_alloc_ctx_tensors failed");
  }

  for (auto &pending : builder.pending_constants_) {
    ggml_backend_tensor_set(pending.first, pending.second.data(), 0,
                             pending.second.size());
  }

  // Take ownership of the builder's context/graph; GraphBuilder's
  // destructor becomes a no-op once ctx_ is nulled out here.
  ctx_ = builder.ctx_;
  buffer_ = buffer;
  graph_ = graph;
  params_ = builder.params_;
  builder.ctx_ = nullptr;
}

CompiledGraph::~CompiledGraph() {
  if (buffer_) {
    ggml_backend_buffer_free(buffer_);
  }
  if (ctx_) {
    ggml_free(ctx_);
  }
}

std::string CompiledGraph::run(const std::vector<std::string> &inputs) {
  if (inputs.size() != params_.size()) {
    throw std::invalid_argument(
        "nx_ggml: wrong number of input binaries for compiled graph");
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    ggml_backend_tensor_set(params_[i], inputs[i].data(), 0, inputs[i].size());
  }

  enum ggml_status status = ggml_backend_graph_compute(nx_ggml_cpu_backend(), graph_);
  if (status != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("nx_ggml: ggml_backend_graph_compute failed");
  }

  std::string result(ggml_nbytes(output_), '\0');
  ggml_backend_tensor_get(output_, result.data(), 0, result.size());
  return result;
}

FINE_RESOURCE(GraphBuilder);
FINE_RESOURCE(CompiledGraph);

fine::ResourcePtr<GraphBuilder> nx_ggml_builder_new(ErlNifEnv *) {
  return fine::make_resource<GraphBuilder>();
}
FINE_NIF(nx_ggml_builder_new, 0);

int64_t nx_ggml_builder_add_param(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                   std::vector<int64_t> shape) {
  return builder->add_param(shape);
}
FINE_NIF(nx_ggml_builder_add_param, 0);

int64_t nx_ggml_builder_add_constant_f32(ErlNifEnv *,
                                          fine::ResourcePtr<GraphBuilder> builder,
                                          std::vector<int64_t> shape,
                                          std::string data) {
  return builder->add_constant_f32(shape, data);
}
FINE_NIF(nx_ggml_builder_add_constant_f32, 0);

int64_t nx_ggml_builder_add_add(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                 int64_t a, int64_t b) {
  return builder->add_add(a, b);
}
FINE_NIF(nx_ggml_builder_add_add, 0);

fine::ResourcePtr<CompiledGraph>
nx_ggml_builder_finalize(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                          int64_t output_index) {
  return fine::make_resource<CompiledGraph>(*builder, output_index);
}
FINE_NIF(nx_ggml_builder_finalize, 0);

fine::Term nx_ggml_compiled_run(ErlNifEnv *env, fine::ResourcePtr<CompiledGraph> compiled,
                                 std::vector<std::string> inputs) {
  std::string result = compiled->run(inputs);
  return fine::make_new_binary(env, result.data(), result.size());
}
FINE_NIF(nx_ggml_compiled_run, 0);
