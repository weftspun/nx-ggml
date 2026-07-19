defmodule NxGgml.AttentionTest do
  @moduledoc """
  Synthetic-shape conformance test for the multi-head self-attention
  composition validated against real DINOv3 weights in
  `scratch_dino_attention.exs` -- batched matmul (Q@K^T, attn@V) + the
  reduce_max/subtract/exp/sum/divide softmax composition + the
  reshape+matmul "rotate_half" trick used for RoPE, all exercised together
  at a small scale so this runs without any downloaded weights.
  """

  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  # Fixed (head_dim x head_dim) matrix implementing y = concat(-x2, x1) for
  # x = concat(x1, x2) via a matmul, instead of a slice+concat -- see
  # scratch_dino_attention.exs for the full derivation.
  defn rotate_half(x, rot_mat) do
    {n, h, d} = Nx.shape(x)
    x |> Nx.reshape({n * h, d}) |> Nx.dot(rot_mat) |> Nx.reshape({n, h, d})
  end

  defn mini_attention(q, k, v, rot_mat, scale) do
    q = rotate_half(q, rot_mat)
    k = rotate_half(k, rot_mat)

    qp = Nx.transpose(q, axes: [1, 0, 2])
    kp = Nx.transpose(k, axes: [1, 0, 2])
    vp = Nx.transpose(v, axes: [1, 0, 2])
    kt = Nx.transpose(kp, axes: [0, 2, 1])

    scores = Nx.dot(qp, [2], [0], kt, [1], [0]) * scale
    shifted = scores - Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp = Nx.exp(shifted)
    attn = exp / Nx.sum(exp, axes: [-1], keep_axes: true)

    out = Nx.dot(attn, [2], [0], vp, [1], [0])
    Nx.transpose(out, axes: [1, 0, 2])
  end

  test "multi-head attention composition (rope-style rotate_half + batched softmax attention) matches Nx.Defn.Evaluator" do
    n = 5
    h = 2
    d = 4
    half = div(d, 2)

    rot_mat =
      for i <- 0..(d - 1) do
        for j <- 0..(d - 1) do
          cond do
            j < half and i == half + j -> -1.0
            j >= half and i == j - half -> 1.0
            true -> 0.0
          end
        end
      end
      |> Nx.tensor(type: :f32)

    q = Nx.iota({n, h, d}, type: :f32) |> Nx.divide(10.0)
    k = Nx.iota({n, h, d}, type: :f32) |> Nx.divide(7.0) |> Nx.subtract(1.0)
    v = Nx.iota({n, h, d}, type: :f32) |> Nx.divide(3.0)
    scale = 1.0 / :math.sqrt(d)

    result = assert_matches_evaluator(&mini_attention/5, [q, k, v, rot_mat, scale], close: true)
    assert Nx.shape(result) == {n, h, d}
  end
end
