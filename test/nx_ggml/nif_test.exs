defmodule NxGgml.NifTest do
  use ExUnit.Case, async: true

  test "cpu_add_f32 computes elementwise sum via ggml's CPU backend" do
    a = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
    b = <<10.0::float-32-native, 20.0::float-32-native, 30.0::float-32-native>>

    result = NxGgml.Nif.cpu_add_f32(a, b)

    assert result == <<11.0::float-32-native, 22.0::float-32-native, 33.0::float-32-native>>
  end

  test "cpu_add_f32 raises on mismatched binary sizes" do
    assert_raise ArgumentError, fn ->
      NxGgml.Nif.cpu_add_f32(<<1.0::float-32-native>>, <<1.0::float-32-native, 2.0::float-32-native>>)
    end
  end
end
