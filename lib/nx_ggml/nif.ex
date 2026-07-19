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
end
