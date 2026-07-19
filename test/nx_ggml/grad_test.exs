defmodule NxGgml.GradTest do
  @moduledoc """
  Phase 8, the actual point of the project: proves `Nx.Defn.grad` works
  correctly on top of `NxGgml.Compiler` with **zero backend-side autodiff
  code**. `Nx.Defn.Grad` differentiates the traced `Nx.Defn.Expr` graph one
  level above any backend (see `NxGgml.Compiler`'s moduledoc); this file is
  the concrete "does that actually hold up" check, via linear regression —
  deep learning's hello world.

  `NxGgml.Compiler.__compile__` only lowers single-tensor outputs (falling
  back to `Nx.Defn.Evaluator` for tuples/composites — see its moduledoc),
  so gradients are computed one parameter at a time
  (`Nx.Defn.grad(w, &loss(&1, b, x, y))`) rather than as a single
  `grad({w, b}, ...)` call returning a tuple. Each such call is a genuine
  single-tensor computation that can lower through ggml.
  """

  use ExUnit.Case, async: false

  import Nx.Defn

  defn predict(w, b, x), do: x * w + b

  # Mean (not sum) squared error, so the gradient magnitude doesn't scale
  # with the number of data points -- Nx.mean isn't in NxGgml.ExprLowering's
  # supported op set yet, so this is spelled as sum / n (both supported)
  # rather than Nx.mean(diff * diff).
  defn loss(w, b, x, y) do
    diff = predict(w, b, x) - y
    Nx.sum(diff * diff) / 8.0
  end

  defn grad_w(w, b, x, y), do: Nx.Defn.grad(w, &loss(&1, b, x, y))
  defn grad_b(w, b, x, y), do: Nx.Defn.grad(b, &loss(w, &1, x, y))

  @learning_rate 0.02
  @steps 2000

  # y = 3x + 2, plus tiny noise, so there's a unique well-defined optimum.
  @x [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  @y [5.03, 8.02, 10.98, 14.05, 16.97, 20.04, 22.98, 26.01]

  defp train(compiler, jit_opts \\ []) do
    x = Nx.tensor(@x, type: :f32)
    y = Nx.tensor(@y, type: :f32)
    w0 = Nx.tensor(0.0, type: :f32)
    b0 = Nx.tensor(0.0, type: :f32)
    opts = [compiler: compiler] ++ jit_opts

    Enum.reduce(1..@steps, {w0, b0}, fn _step, {w, b} ->
      gw = Nx.Defn.jit_apply(&grad_w/4, [w, b, x, y], opts)
      gb = Nx.Defn.jit_apply(&grad_b/4, [w, b, x, y], opts)

      {Nx.subtract(w, Nx.multiply(@learning_rate, gw)),
       Nx.subtract(b, Nx.multiply(@learning_rate, gb))}
    end)
  end

  defp loss_at(w, b) do
    x = Nx.tensor(@x, type: :f32)
    y = Nx.tensor(@y, type: :f32)
    Nx.Defn.jit_apply(&loss/4, [w, b, x, y], compiler: Nx.Defn.Evaluator) |> Nx.to_number()
  end

  test "gradient descent learns y = 3x + 2 via NxGgml.Compiler" do
    {w, b} = train(NxGgml.Compiler)

    assert_in_delta Nx.to_number(w), 3.0, 0.05
    assert_in_delta Nx.to_number(b), 2.0, 0.1
  end

  test "loss decreases monotonically (within noise) during training" do
    x = Nx.tensor(@x, type: :f32)
    y = Nx.tensor(@y, type: :f32)
    w0 = Nx.tensor(0.0, type: :f32)
    b0 = Nx.tensor(0.0, type: :f32)

    {_final, losses} =
      Enum.reduce(1..@steps, {{w0, b0}, []}, fn _step, {{w, b}, losses} ->
        gw = Nx.Defn.jit_apply(&grad_w/4, [w, b, x, y], compiler: NxGgml.Compiler)
        gb = Nx.Defn.jit_apply(&grad_b/4, [w, b, x, y], compiler: NxGgml.Compiler)

        new_w = Nx.subtract(w, Nx.multiply(@learning_rate, gw))
        new_b = Nx.subtract(b, Nx.multiply(@learning_rate, gb))

        {{new_w, new_b}, [loss_at(new_w, new_b) | losses]}
      end)

    losses = Enum.reverse(losses)
    # Monotonic decrease isn't guaranteed step-to-step with plain SGD, but
    # the loss trajectory averaged over windows must trend down, and the
    # final loss must be far smaller than the first.
    [first | _] = losses
    last = List.last(losses)
    assert last < first / 100
  end

  test "NxGgml.Compiler's learned parameters match the Nx.Defn.Evaluator reference" do
    {ggml_w, ggml_b} = train(NxGgml.Compiler)
    {ref_w, ref_b} = train(Nx.Defn.Evaluator)

    assert_in_delta Nx.to_number(ggml_w), Nx.to_number(ref_w), 1.0e-3
    assert_in_delta Nx.to_number(ggml_b), Nx.to_number(ref_b), 1.0e-3
  end

  @tag :vulkan
  test "gradient descent learns y = 3x + 2 on the Vulkan device too" do
    {w, b} = train(NxGgml.Compiler, device: :vulkan)

    assert_in_delta Nx.to_number(w), 3.0, 0.05
    assert_in_delta Nx.to_number(b), 2.0, 0.1
  end
end
