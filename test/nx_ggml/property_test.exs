defmodule NxGgml.PropertyTest do
  @moduledoc """
  Phase 6: property-based conformance sweep. Rather than a handful of
  fixed examples, generates random shapes/values and checks NxGgml.Compiler
  against Nx.Defn.Evaluator for every currently-lowered op family — the
  systematic-matrix idea from the project plan's Phase 6, scoped to the
  dtype/op subset actually implemented (f32; see README for the full gap
  list) rather than a synthetic ranks-0-4/f16/s32 sweep that would mostly
  just be exercising the (already-tested) Evaluator fallback path.
  """

  use ExUnit.Case, async: false
  use ExUnitProperties

  import Nx.Defn
  import NxGgml.Conformance

  defn add_(x, y), do: x + y
  defn sub_(x, y), do: x - y
  defn mul_(x, y), do: x * y
  defn negate_(x), do: -x
  defn sigmoid_(x), do: Nx.sigmoid(x)
  defn reshape_flat(x), do: Nx.reshape(x, {Nx.size(x)})

  # Small dims keep tensors tiny (property tests run many cases) while
  # still covering rank 1-3 shapes.
  defp dim, do: StreamData.integer(1..5)

  defp shape(rank) do
    List.duplicate(dim(), rank)
    |> StreamData.fixed_list()
    |> StreamData.map(&List.to_tuple/1)
  end

  defp tensor_of_shape(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(1, &(&1 * &2))
    |> then(fn n -> StreamData.list_of(float_gen(), length: n) end)
    |> StreamData.map(&Nx.tensor(&1, type: :f32))
    |> StreamData.map(&Nx.reshape(&1, shape))
  end

  defp float_gen do
    StreamData.float(min: -100.0, max: 100.0)
  end

  defp same_shape_pair(rank) do
    StreamData.bind(shape(rank), fn shape ->
      StreamData.bind(tensor_of_shape(shape), fn a ->
        StreamData.bind(tensor_of_shape(shape), fn b ->
          StreamData.constant({a, b})
        end)
      end)
    end)
  end

  property "add matches Nx.Defn.Evaluator for random same-shape tensors" do
    check all(rank <- StreamData.integer(1..3), {a, b} <- same_shape_pair(rank), max_runs: 25) do
      assert_matches_evaluator(&add_/2, [a, b], close: true)
    end
  end

  property "subtract matches Nx.Defn.Evaluator for random same-shape tensors" do
    check all(rank <- StreamData.integer(1..3), {a, b} <- same_shape_pair(rank), max_runs: 25) do
      assert_matches_evaluator(&sub_/2, [a, b], close: true)
    end
  end

  property "multiply matches Nx.Defn.Evaluator for random same-shape tensors" do
    check all(rank <- StreamData.integer(1..3), {a, b} <- same_shape_pair(rank), max_runs: 25) do
      assert_matches_evaluator(&mul_/2, [a, b], close: true)
    end
  end

  property "negate matches Nx.Defn.Evaluator for random tensors" do
    check all(
            rank <- StreamData.integer(1..3),
            a <- StreamData.bind(shape(rank), &tensor_of_shape/1),
            max_runs: 25
          ) do
      assert_matches_evaluator(&negate_/1, [a], close: true)
    end
  end

  property "sigmoid matches Nx.Defn.Evaluator for random tensors" do
    check all(
            rank <- StreamData.integer(1..3),
            a <- StreamData.bind(shape(rank), &tensor_of_shape/1),
            max_runs: 25
          ) do
      assert_matches_evaluator(&sigmoid_/1, [a], close: true)
    end
  end

  property "reshape to a flat vector preserves element order for random tensors" do
    check all(
            rank <- StreamData.integer(1..3),
            a <- StreamData.bind(shape(rank), &tensor_of_shape/1),
            max_runs: 25
          ) do
      assert_matches_evaluator(&reshape_flat/1, [a], close: true)
    end
  end
end
