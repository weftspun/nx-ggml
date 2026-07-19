defmodule NxGgml.Nif do
  @moduledoc false

  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:nx_ggml), ~c"libnx_ggml")

    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, reason} -> raise "failed to load NxGgml NIF library, reason: #{inspect(reason)}"
    end
  end

  @doc false
  def cpu_add_f32(_a, _b) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_new(_device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_vulkan_available do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_param(_builder, _shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_constant_f32(_builder, _shape, _data) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_binary(_builder, _op, _a, _b) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_unary(_builder, _op, _a) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_broadcast(_builder, _a, _shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_reshape(_builder, _a, _shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_transpose(_builder, _a, _axes) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_matmul(_builder, _a, _b) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_sum_all(_builder, _a) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_sum_last_axis(_builder, _a) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_clamp(_builder, _a, _min, _max) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_add_concat(_builder, _a, _b, _axis, _rank) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_builder_finalize(_builder, _output_index) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc false
  def nx_ggml_compiled_run(_compiled, _inputs) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
