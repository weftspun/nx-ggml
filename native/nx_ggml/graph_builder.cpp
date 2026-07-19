#include "graph_builder.h"

#include <cmath>
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

GraphBuilder::GraphBuilder(const std::string &device) {
  backend_ = nx_ggml_backend_for(device);

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

ggml_tensor *GraphBuilder::new_leaf_i32(const std::vector<int64_t> &shape) {
  auto ne = to_ggml_ne(shape);
  return ggml_new_tensor(ctx_, GGML_TYPE_I32, GGML_MAX_DIMS, ne.data());
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

int64_t GraphBuilder::add_param_i32(const std::vector<int64_t> &shape) {
  ggml_tensor *t = new_leaf_i32(shape);
  params_.push_back(t);
  return push(t);
}

int64_t GraphBuilder::add_constant_i32(const std::vector<int64_t> &shape,
                                        const std::string &data) {
  int64_t n = element_count(shape);
  if (static_cast<int64_t>(data.size()) != n * static_cast<int64_t>(sizeof(int32_t))) {
    throw std::invalid_argument(
        "nx_ggml: constant binary size does not match shape (expected i32 elements)");
  }
  ggml_tensor *t = new_leaf_i32(shape);
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

namespace {
void erf_custom_op(ggml_tensor *dst, const ggml_tensor *a, int ith, int nth, void *) {
  // Single-threaded (GraphBuilder::add_erf passes n_tasks=1), but written
  // to tolerate nth>1 defensively: only ith==0 does anything.
  if (ith != 0) return;
  int64_t n = ggml_nelements(a);
  const float *src = static_cast<const float *>(a->data);
  float *out = static_cast<float *>(dst->data);
  for (int64_t i = 0; i < n; i++) {
    out[i] = erff(src[i]);
  }
  (void)nth;
}
} // namespace

int64_t GraphBuilder::add_erf(int64_t a) {
  ggml_tensor *result = ggml_map_custom1(ctx_, node(a), erf_custom_op, /*n_tasks=*/1, nullptr);
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

int64_t GraphBuilder::add_matmul(int64_t a, int64_t b) {
  // a is Nx shape (...batch,m,k) -> ggml ne=[k,m,...batch] (to_ggml_ne
  // reverses); this is already exactly the shape ggml_mul_mat wants for
  // its *second* argument. b is Nx shape (...batch,k,n) -> ggml
  // ne=[n,k,...batch]; ggml_mul_mat requires both operands to share ne0
  // (the contraction dim, k), so b must be transposed first:
  // ggml_transpose only swaps ne0/ne1 (leaving batch dims in ne2/ne3
  // alone), giving Nx shape (...batch,n,k) -> ggml ne=[k,n,...batch].
  //
  // ggml_mul_mat(ctx, X, Y) requires X.ne0 == Y.ne0 and produces a result
  // with ne0 = X.ne1, ne1 = Y.ne1, ne2/ne3 = Y.ne2/ne3 (X's batch dims
  // must only evenly divide Y's -- trivially true here since they're
  // equal). Calling ggml_mul_mat(ctx, transpose(b), a) therefore gives
  // X.ne1 = n, Y.ne1 = m -> result ne=[n,m,...batch] -> Nx shape
  // (reversed) = (...batch,m,n), the standard (batched) matmul result
  // shape. This was derived from ggml's ne[]/mul_mat convention and
  // confirmed against Nx.BinaryBackend with a hand-picked asymmetric
  // matrix (see linalg_test.exs) rather than trusted from memory alone --
  // exactly the row-major/ggml-ne[]-convention risk flagged for matmul.
  ggml_tensor *b_t = ggml_cont(ctx_, ggml_transpose(ctx_, node(b)));
  ggml_tensor *result = ggml_mul_mat(ctx_, b_t, node(a));
  return push(result);
}

int64_t GraphBuilder::add_sum_all(int64_t a) {
  ggml_tensor *result = ggml_sum(ctx_, node(a));
  return push(result);
}

int64_t GraphBuilder::add_sum_last_axis(int64_t a) {
  ggml_tensor *result = ggml_sum_rows(ctx_, node(a));
  return push(result);
}

int64_t GraphBuilder::add_reduce_max_last_axis(int64_t a, int64_t last_axis_size) {
  // ggml has no dedicated row-max reduction; a global 1-D max-pool over
  // the whole ne0 extent (kernel = stride = last_axis_size, no padding)
  // is exactly a max reduction over ne0, collapsing it to size 1 -- the
  // same shape contract as add_sum_last_axis.
  int k0 = static_cast<int>(last_axis_size);
  ggml_tensor *result = ggml_pool_1d(ctx_, node(a), GGML_OP_POOL_MAX, k0, k0, 0);
  return push(result);
}

int64_t GraphBuilder::add_clamp(int64_t a, float min, float max) {
  ggml_tensor *result = ggml_clamp(ctx_, node(a), min, max);
  return push(result);
}

int64_t GraphBuilder::add_concat(int64_t a, int64_t b, int64_t axis, int64_t rank) {
  // Same Nx-axis -> ggml-dim reversal as add_transpose: Nx axis `axis`
  // (0 = outermost) is ggml dim `rank - 1 - axis` (0 = innermost).
  int dim = static_cast<int>(rank) - 1 - static_cast<int>(axis);
  ggml_tensor *result = ggml_concat(ctx_, node(a), node(b), dim);
  return push(result);
}

int64_t GraphBuilder::add_get_rows(int64_t a, int64_t b) {
  // a: Nx shape (vocab, dim) -> ggml ne=[dim,vocab,1,1]. b: Nx shape (n,)
  // -> ggml ne=[n,1,1,1] (i32). ggml_get_rows requires a.ne2==b.ne1,
  // a.ne3==b.ne2, b.ne3==1 -- all satisfied here since both are
  // effectively rank-2/rank-1 with trailing 1-padding. Result ne =
  // [a.ne0, b.ne0, b.ne1, b.ne2] = [dim,n,1,1] -> Nx shape (1,1,n,dim);
  // the caller reshapes down to the expected (n,dim).
  ggml_tensor *result = ggml_get_rows(ctx_, node(a), node(b));
  return push(result);
}

int64_t GraphBuilder::add_conv2d(int64_t kernel, int64_t input, int64_t s0, int64_t s1,
                                  int64_t p0, int64_t p1, int64_t d0, int64_t d1) {
  ggml_tensor *result =
      ggml_conv_2d(ctx_, node(kernel), node(input), static_cast<int>(s0), static_cast<int>(s1),
                   static_cast<int>(p0), static_cast<int>(p1), static_cast<int>(d0),
                   static_cast<int>(d1));
  return push(result);
}

CompiledGraph::CompiledGraph(GraphBuilder &builder, int64_t output_index) {
  output_ = builder.node(output_index);

  // Inputs/outputs must be marked before allocation so ggml_gallocr knows
  // not to recycle their buffer space as scratch for some other tensor --
  // params_ already exist before compute starts (real inputs set in run())
  // and pending_constants_ are read throughout the graph's lifetime, so
  // both need GGML_TENSOR_FLAG_INPUT; the final output must survive past
  // the last op that touches it, hence GGML_TENSOR_FLAG_OUTPUT.
  for (ggml_tensor *param : builder.params_) {
    ggml_set_input(param);
  }
  for (auto &pending : builder.pending_constants_) {
    ggml_set_input(pending.first);
  }
  ggml_set_output(output_);

  ggml_cgraph *graph = ggml_new_graph(builder.ctx_);
  ggml_build_forward_expand(graph, output_);

  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(builder.backend_));
  if (!ggml_gallocr_alloc_graph(galloc, graph)) {
    ggml_gallocr_free(galloc);
    throw std::runtime_error("nx_ggml: ggml_gallocr_alloc_graph failed");
  }

  for (auto &pending : builder.pending_constants_) {
    ggml_backend_tensor_set(pending.first, pending.second.data(), 0,
                             pending.second.size());
  }

  // Take ownership of the builder's context/graph/backend; GraphBuilder's
  // destructor becomes a no-op once ctx_ is nulled out here.
  ctx_ = builder.ctx_;
  backend_ = builder.backend_;
  galloc_ = galloc;
  graph_ = graph;
  params_ = builder.params_;
  builder.ctx_ = nullptr;
}

CompiledGraph::~CompiledGraph() {
  if (galloc_) {
    ggml_gallocr_free(galloc_);
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

  enum ggml_status status = ggml_backend_graph_compute(backend_, graph_);
  if (status != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("nx_ggml: ggml_backend_graph_compute failed");
  }

  std::string result(ggml_nbytes(output_), '\0');
  ggml_backend_tensor_get(output_, result.data(), 0, result.size());
  return result;
}

FINE_RESOURCE(GraphBuilder);
FINE_RESOURCE(CompiledGraph);

fine::ResourcePtr<GraphBuilder> nx_ggml_builder_new(ErlNifEnv *, std::string device) {
  return fine::make_resource<GraphBuilder>(device);
}
FINE_NIF(nx_ggml_builder_new, 0);

int64_t nx_ggml_builder_add_param(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                   std::vector<int64_t> shape) {
  return builder->add_param(shape);
}
FINE_NIF(nx_ggml_builder_add_param, 0);

int64_t nx_ggml_builder_add_param_i32(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                       std::vector<int64_t> shape) {
  return builder->add_param_i32(shape);
}
FINE_NIF(nx_ggml_builder_add_param_i32, 0);

int64_t nx_ggml_builder_add_constant_i32(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                          std::vector<int64_t> shape, std::string data) {
  return builder->add_constant_i32(shape, data);
}
FINE_NIF(nx_ggml_builder_add_constant_i32, 0);

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

int64_t nx_ggml_builder_add_erf(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder, int64_t a) {
  return builder->add_erf(a);
}
FINE_NIF(nx_ggml_builder_add_erf, 0);

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

int64_t nx_ggml_builder_add_matmul(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                    int64_t a, int64_t b) {
  return builder->add_matmul(a, b);
}
FINE_NIF(nx_ggml_builder_add_matmul, 0);

int64_t nx_ggml_builder_add_sum_all(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                     int64_t a) {
  return builder->add_sum_all(a);
}
FINE_NIF(nx_ggml_builder_add_sum_all, 0);

int64_t nx_ggml_builder_add_sum_last_axis(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                           int64_t a) {
  return builder->add_sum_last_axis(a);
}
FINE_NIF(nx_ggml_builder_add_sum_last_axis, 0);

int64_t nx_ggml_builder_add_reduce_max_last_axis(ErlNifEnv *,
                                                  fine::ResourcePtr<GraphBuilder> builder,
                                                  int64_t a, int64_t last_axis_size) {
  return builder->add_reduce_max_last_axis(a, last_axis_size);
}
FINE_NIF(nx_ggml_builder_add_reduce_max_last_axis, 0);

int64_t nx_ggml_builder_add_clamp(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                   int64_t a, double min, double max) {
  return builder->add_clamp(a, static_cast<float>(min), static_cast<float>(max));
}
FINE_NIF(nx_ggml_builder_add_clamp, 0);

int64_t nx_ggml_builder_add_concat(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                    int64_t a, int64_t b, int64_t axis, int64_t rank) {
  return builder->add_concat(a, b, axis, rank);
}
FINE_NIF(nx_ggml_builder_add_concat, 0);

int64_t nx_ggml_builder_add_get_rows(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                      int64_t a, int64_t b) {
  return builder->add_get_rows(a, b);
}
FINE_NIF(nx_ggml_builder_add_get_rows, 0);

int64_t nx_ggml_builder_add_conv2d(ErlNifEnv *, fine::ResourcePtr<GraphBuilder> builder,
                                    int64_t kernel, int64_t input, int64_t s0, int64_t s1,
                                    int64_t p0, int64_t p1, int64_t d0, int64_t d1) {
  return builder->add_conv2d(kernel, input, s0, s1, p0, p1, d0, d1);
}
FINE_NIF(nx_ggml_builder_add_conv2d, 0);

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
