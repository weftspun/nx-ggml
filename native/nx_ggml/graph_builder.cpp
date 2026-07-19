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

int64_t GraphBuilder::add_binary(const std::string &op, int64_t a, int64_t b) {
  ggml_tensor *ta = node(a);
  ggml_tensor *tb = node(b);
  ggml_tensor *result;

  if (op == "add") {
    result = ggml_add(ctx_, ta, tb);
  } else if (op == "subtract") {
    result = ggml_sub(ctx_, ta, tb);
  } else if (op == "multiply") {
    result = ggml_mul(ctx_, ta, tb);
  } else if (op == "divide") {
    result = ggml_div(ctx_, ta, tb);
  } else {
    throw std::invalid_argument("nx_ggml: unknown binary op '" + op + "'");
  }

  return push(result);
}

int64_t GraphBuilder::add_unary(const std::string &op, int64_t a) {
  ggml_tensor *ta = node(a);
  ggml_tensor *result;

  if (op == "negate") {
    result = ggml_neg(ctx_, ta);
  } else if (op == "abs") {
    result = ggml_abs(ctx_, ta);
  } else if (op == "sign") {
    result = ggml_sgn(ctx_, ta);
  } else if (op == "sqrt") {
    result = ggml_sqrt(ctx_, ta);
  } else if (op == "exp") {
    result = ggml_exp(ctx_, ta);
  } else if (op == "log") {
    result = ggml_log(ctx_, ta);
  } else if (op == "sigmoid") {
    result = ggml_sigmoid(ctx_, ta);
  } else if (op == "tanh") {
    result = ggml_tanh(ctx_, ta);
  } else if (op == "sin") {
    result = ggml_sin(ctx_, ta);
  } else if (op == "cos") {
    result = ggml_cos(ctx_, ta);
  } else {
    throw std::invalid_argument("nx_ggml: unknown unary op '" + op + "'");
  }

  return push(result);
}

int64_t GraphBuilder::add_broadcast(int64_t a, const std::vector<int64_t> &shape) {
  // ggml_repeat's second argument is only inspected for its shape/type, so
  // a never-allocated, never-fed template leaf is sufficient.
  ggml_tensor *result = ggml_repeat(ctx_, node(a), new_leaf_f32(shape));
  return push(result);
}

int64_t GraphBuilder::add_reshape(int64_t a, const std::vector<int64_t> &shape) {
  auto ne = to_ggml_ne(shape);
  ggml_tensor *result = ggml_reshape_4d(ctx_, node(a), ne[0], ne[1], ne[2], ne[3]);
  return push(result);
}

int64_t GraphBuilder::add_transpose(int64_t a, const std::vector<int64_t> &axes) {
  int r = static_cast<int>(axes.size());
  if (r > GGML_MAX_DIMS) {
    throw std::invalid_argument("nx_ggml: transpose rank exceeds ggml's GGML_MAX_DIMS (4)");
  }

  // Nx's axes are in row-major axis numbering (axis 0 = outermost); ggml's
  // ne[]/permute dims are the reverse (dim 0 = innermost). `axes[i] = j`
  // (output Nx-axis i takes input Nx-axis j) becomes, in ggml-dim terms,
  // "input ggml-dim (r-1-j) maps to output ggml-dim (r-1-i)". Dims beyond
  // rank r (padding up to GGML_MAX_DIMS) are left as the identity mapping.
  int permute_args[GGML_MAX_DIMS] = {0, 1, 2, 3};
  for (int i = 0; i < r; i++) {
    int input_dim = r - 1 - static_cast<int>(axes[static_cast<size_t>(i)]);
    int output_dim = r - 1 - i;
    permute_args[input_dim] = output_dim;
  }

  ggml_tensor *permuted = ggml_permute(ctx_, node(a), permute_args[0], permute_args[1],
                                       permute_args[2], permute_args[3]);
  // ggml_permute returns a (typically non-contiguous) view; materialize it
  // so downstream ops (and the final ggml_backend_tensor_get) see a plain
  // contiguous buffer.
  ggml_tensor *result = ggml_cont(ctx_, permuted);
  return push(result);
}

int64_t GraphBuilder::add_matmul_2d(int64_t a, int64_t b) {
  // a is Nx shape (m,k) -> ggml ne=[k,m] (to_ggml_ne reverses); this is
  // already exactly the shape ggml_mul_mat wants for its *second* argument.
  // b is Nx shape (k,n) -> ggml ne=[n,k]; ggml_mul_mat requires both
  // operands to share ne0 (the contraction dim, k), so b must be
  // transposed first: transpose(b) is Nx shape (n,k) -> ggml ne=[k,n].
  //
  // ggml_mul_mat(ctx, X, Y) requires X.ne0 == Y.ne0 and produces a result
  // with ne0 = X.ne1, ne1 = Y.ne1. Calling ggml_mul_mat(ctx, transpose(b),
  // a) therefore gives X.ne1 = n, Y.ne1 = m -> result ne=[n,m] -> Nx shape
  // (reversed) = (m,n), which is the standard matmul result shape. This
  // was derived from ggml's ne[]/mul_mat convention and confirmed against
  // Nx.BinaryBackend with a hand-picked asymmetric matrix (see
  // shape_ops_test.exs) rather than trusted from memory alone -- exactly
  // the row-major/ggml-ne[]-convention risk flagged for matmul.
  ggml_tensor *b_t = ggml_cont(ctx_, ggml_transpose(ctx_, node(b)));
  ggml_tensor *result = ggml_mul_mat(ctx_, b_t, node(a));
  return push(result);
}

int64_t GraphBuilder::add_sum_all(int64_t a) {
  ggml_tensor *result = ggml_sum(ctx_, node(a));
  return push(result);
}

int64_t GraphBuilder::add_clamp(int64_t a, float min, float max) {
  ggml_tensor *result = ggml_clamp(ctx_, node(a), min, max);
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

int64_t nx_ggml_builder_add_binary(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                    std::string op, int64_t a, int64_t b) {
  return builder->add_binary(op, a, b);
}
FINE_NIF(nx_ggml_builder_add_binary, 0);

int64_t nx_ggml_builder_add_unary(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                   std::string op, int64_t a) {
  return builder->add_unary(op, a);
}
FINE_NIF(nx_ggml_builder_add_unary, 0);

int64_t nx_ggml_builder_add_broadcast(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                       int64_t a, std::vector<int64_t> shape) {
  return builder->add_broadcast(a, shape);
}
FINE_NIF(nx_ggml_builder_add_broadcast, 0);

int64_t nx_ggml_builder_add_reshape(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                     int64_t a, std::vector<int64_t> shape) {
  return builder->add_reshape(a, shape);
}
FINE_NIF(nx_ggml_builder_add_reshape, 0);

int64_t nx_ggml_builder_add_transpose(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                       int64_t a, std::vector<int64_t> axes) {
  return builder->add_transpose(a, axes);
}
FINE_NIF(nx_ggml_builder_add_transpose, 0);

int64_t nx_ggml_builder_add_matmul_2d(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                      int64_t a, int64_t b) {
  return builder->add_matmul_2d(a, b);
}
FINE_NIF(nx_ggml_builder_add_matmul_2d, 0);

int64_t nx_ggml_builder_add_sum_all(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                     int64_t a) {
  return builder->add_sum_all(a);
}
FINE_NIF(nx_ggml_builder_add_sum_all, 0);

int64_t nx_ggml_builder_add_clamp(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                   int64_t a, double min, double max) {
  return builder->add_clamp(a, static_cast<float>(min), static_cast<float>(max));
}
FINE_NIF(nx_ggml_builder_add_clamp, 0);

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
