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

class GraphBuilder {
public:
  GraphBuilder();
  ~GraphBuilder();

  int64_t add_param(const std::vector<int64_t> &shape);
  int64_t add_constant_f32(const std::vector<int64_t> &shape,
                            const std::string &data);
  int64_t add_binary(const std::string &op, int64_t a, int64_t b);
  int64_t add_unary(const std::string &op, int64_t a);
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
  // Standard 2-D matmul: a is Nx shape (m,k), b is Nx shape (k,n), result
  // is Nx shape (m,n) -- see the .cpp for the ggml_mul_mat operand-order
  // derivation (this is the ne[]-convention risk area for matmul).
  int64_t add_matmul_2d(int64_t a, int64_t b);
  // Full reduction to a 0-d (scalar) tensor.
  int64_t add_sum_all(int64_t a);
  int64_t add_clamp(int64_t a, float min, float max);

private:
  friend class CompiledGraph;

  ggml_tensor *node(int64_t index) const;
  int64_t push(ggml_tensor *tensor);
  ggml_tensor *new_leaf_f32(const std::vector<int64_t> &shape);

  ggml_context *ctx_;
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
  ggml_context *ctx_;
  ggml_backend_buffer_t buffer_;
  ggml_cgraph *graph_;
  std::vector<ggml_tensor *> params_;
  ggml_tensor *output_;
};
