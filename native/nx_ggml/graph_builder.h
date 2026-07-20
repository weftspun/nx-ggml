// Phase 3: the real Nx.Defn.Expr -> ggml_cgraph lowering target.
//
// GraphBuilder accumulates ggml tensor/op nodes into one ggml_context while
// NxGgml.ExprLowering (Elixir) walks a traced Nx.Defn.Expr tree. Nodes are
// addressed by a plain int64_t index into GraphBuilder's own node list
// rather than by a separate Fine resource per node, so Elixir only ever
// holds one native reference (the builder, then the compiled graph).
//
// CompiledGraph is the cached, executable result (what
// NxGgml.GraphCache stores): a finished ggml_cgraph with allocated
// backend storage, ready to have its parameter tensors' data replaced and
// re-run on every subsequent call with the same shape/dtype signature.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <fine.hpp>
#include <ggml-backend.h>
#include <ggml.h>

ggml_backend_t nx_ggml_cpu_backend();
ggml_backend_t nx_ggml_backend_for(const std::string &device);

class GraphBuilder {
public:
  explicit GraphBuilder(const std::string &device);
  ~GraphBuilder();

  int64_t add_param(const std::vector<int64_t> &shape);
  int64_t add_constant_f32(const std::vector<int64_t> &shape,
                            const std::string &data);
  // i32 leaves are only ever used for gather/get_rows index tensors (all
  // compute ops require f32 -- see NxGgml.ExprLowering's check_dtype!);
  // callers must already have converted Nx's index tensor to {:s, 32}.
  int64_t add_param_i32(const std::vector<int64_t> &shape);
  int64_t add_constant_i32(const std::vector<int64_t> &shape, const std::string &data);
  int64_t add_binary(const std::string &op, int64_t a, int64_t b);
  int64_t add_unary(const std::string &op, int64_t a);
  // ggml has no built-in erf op (only the fused gelu_erf activation), so
  // this wraps the C standard library's erff() as a custom CPU callback
  // via ggml_map_custom1 -- see graph_builder.cpp for why this is CPU-only
  // (custom map ops aren't portable to the Vulkan backend).
  int64_t add_erf(int64_t a);
  // Broadcasts `a` up to `shape` (ggml_repeat); a no-op shape-wise if `a`
  // already has that shape. Used to make every binary op's operands match
  // the node's output shape *before* calling the ggml op, which sidesteps
  // ggml_add/sub/mul/div's `ggml_can_repeat(b, a)` requirement entirely
  // (no need to reason about which operand ggml wants first).
  int64_t add_broadcast(int64_t a, const std::vector<int64_t> &shape);
  int64_t add_reshape(int64_t a, const std::vector<int64_t> &shape);
  // `axes[i] = j` means output axis i takes input axis j's size (Nx's
  // row-major axis numbering) -- see the .cpp for the ne[]-reversal
  // mapping into ggml_permute's argument convention.
  int64_t add_transpose(int64_t a, const std::vector<int64_t> &axes);
  // a is Nx shape (...batch, m, k), b is Nx shape (...batch, k, n), result
  // is Nx shape (...batch, m, n) -- batch dims (if any) are handled for
  // free by ggml_mul_mat's own ne2/ne3 ("t1 broadcastable to t0") ,
  // handling, since Nx's leading axes already land in ggml's ne2/ne3 slots
  // via the same reversal to_ggml_ne always applies. See the .cpp for the
  // ggml_mul_mat operand-order derivation (the ne[]-convention risk area
  // for matmul).
  int64_t add_matmul(int64_t a, int64_t b);
  // Full reduction to a 0-d (scalar) tensor.
  int64_t add_sum_all(int64_t a);
  // Reduces only the last (fastest-varying / ne0) axis, keeping it as a
  // trailing size-1 dim (ggml_sum_rows) -- the caller reshapes to the
  // desired final shape (dropping the axis entirely, i.e. keep_axes:
  // false, is just add_reshape to that shape; the sum itself doesn't need
  // to know keep_axes).
  int64_t add_sum_last_axis(int64_t a);
  // Reduces only the last axis to its max, same shape contract as
  // add_sum_last_axis. ggml has no dedicated row-max reduction op, so this
  // is implemented via ggml_pool_1d(GGML_OP_POOL_MAX) with a kernel
  // spanning the whole last axis (`last_axis_size`, the caller already
  // knows this from the input tensor's Nx shape) -- a global 1-D max-pool
  // is exactly a max reduction over ne0.
  int64_t add_reduce_max_last_axis(int64_t a, int64_t last_axis_size);
  int64_t add_clamp(int64_t a, float min, float max);
  // Concatenates a and b along Nx axis `axis` (reversed to ggml's dim
  // convention internally, same as add_transpose).
  int64_t add_concat(int64_t a, int64_t b, int64_t axis, int64_t rank);
  // Embedding-lookup-style gather: selects whole rows of a 2-D table `a`
  // (Nx shape (vocab, dim)) by a 1-D or 2-D integer index tensor `b`
  // (ggml_get_rows), matching Nx.gather's common single-axis-0,
  // whole-row-selection case.
  int64_t add_get_rows(int64_t a, int64_t b);
  // kernel is Nx shape (OC,IC,KH,KW), input is Nx shape (N,IC,IH,IW),
  // result is Nx shape (N,OC,OH,OW) -- ggml_conv_2d's a/b/result layout
  // comments already match Nx/PyTorch's standard conv convention exactly
  // once reversed by to_ggml_ne, so (unlike matmul/transpose) this needed
  // no operand-order derivation at all.
  int64_t add_conv2d(int64_t kernel, int64_t input, int64_t s0, int64_t s1, int64_t p0,
                      int64_t p1, int64_t d0, int64_t d1);

private:
  friend class CompiledGraph;

  ggml_tensor *node(int64_t index) const;
  int64_t push(ggml_tensor *tensor);
  ggml_tensor *new_leaf_f32(const std::vector<int64_t> &shape);
  ggml_tensor *new_leaf_i32(const std::vector<int64_t> &shape);

  ggml_context *ctx_;
  ggml_backend_t backend_;
  std::vector<ggml_tensor *> nodes_;
  std::vector<ggml_tensor *> params_;
  std::vector<std::pair<ggml_tensor *, std::string>> pending_constants_;
};

class CompiledGraph {
public:
  CompiledGraph(GraphBuilder &builder, int64_t output_index);
  ~CompiledGraph();

  std::string run(const std::vector<std::string> &inputs);

private:
  // No buffer_/allocator field: tensor storage comes from a shared,
  // process-lifetime BufferPool (graph_builder.cpp) keyed by backend, not
  // an allocation owned by this instance -- see the constructor.
  ggml_context *ctx_;
  ggml_backend_t backend_;
  ggml_cgraph *graph_;
  std::vector<ggml_tensor *> params_;
  ggml_tensor *output_;
};
