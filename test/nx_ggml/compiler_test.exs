defmodule NxGgml.CompilerTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  defn identity(x), do: x

  test "Nx.Defn.jit round-trips through NxGgml.Compiler" do
    result = Nx.Defn.jit(&identity/1, compiler: NxGgml.Compiler).(Nx.tensor([1, 2, 3]))

    assert Nx.to_flat_list(result) == [1, 2, 3]
  end
end
