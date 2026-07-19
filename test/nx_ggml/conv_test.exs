defmodule NxGgml.ConvTest do
  @moduledoc """
  `Nx.conv` lowering (`ggml_conv_2d`), added while validating nx-ggml against
  a real trellis2cpp/DINOv3 model stage (patch embedding is a "patchify"
  conv: stride == kernel size, no padding) -- see `scratch_dino_patch_embed.exs`
  at the project root for the full real-weights validation script.
  """

  use ExUnit.Case, async: false

  import Nx.Defn
  import NxGgml.Conformance

  defn conv_valid(x, w), do: Nx.conv(x, w, strides: [1, 1])
  defn conv_strided(x, w), do: Nx.conv(x, w, strides: [2, 2])
  defn patchify(x, w), do: Nx.conv(x, w, strides: [4, 4])

  test "conv2d with stride 1 (no padding) matches Nx.Defn.Evaluator" do
    x = Nx.iota({1, 1, 4, 4}, type: :f32)
    w = Nx.iota({1, 1, 2, 2}, type: :f32)
    result = assert_matches_evaluator(&conv_valid/2, [x, w])
    assert Nx.shape(result) == {1, 1, 3, 3}
  end

  test "conv2d with stride 2 matches Nx.Defn.Evaluator" do
    x = Nx.iota({1, 1, 4, 4}, type: :f32)
    w = Nx.iota({1, 1, 2, 2}, type: :f32)
    result = assert_matches_evaluator(&conv_strided/2, [x, w])
    assert Nx.shape(result) == {1, 1, 2, 2}
  end

  test "patchify conv (stride == kernel size, multi-channel, multi-filter) matches Nx.Defn.Evaluator" do
    # 2 input channels, 3 output filters, 8x8 image patchified into 2x2
    # (kernel 4x4, stride 4) -- the same "stride == kernel size, no padding"
    # shape as a ViT patch embedding, just much smaller.
    x = Nx.iota({1, 2, 8, 8}, type: :f32) |> Nx.divide(64.0)
    w = Nx.iota({3, 2, 4, 4}, type: :f32) |> Nx.divide(32.0)
    result = assert_matches_evaluator(&patchify/2, [x, w], close: true)
    assert Nx.shape(result) == {1, 3, 2, 2}
  end
end
