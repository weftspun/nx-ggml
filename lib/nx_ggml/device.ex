defmodule NxGgml.Device do
  @moduledoc """
  The ggml compute device a compiled graph runs on: `:cpu` (always
  available) or `:vulkan` (only if the native build was configured with
  `NX_GGML_VULKAN=ON` and a Vulkan-capable GPU was found at NIF load).

  Select a device via the `:device` compiler option, e.g.
  `Nx.Defn.jit_apply(fun, args, compiler: NxGgml.Compiler, device: :vulkan)`.
  """

  @type t :: :cpu | :vulkan

  @doc "Extracts the `:device` compiler option (default `:cpu`) as the string the NIF expects."
  @spec from_opts(keyword) :: String.t()
  def from_opts(opts) do
    case Keyword.get(opts, :device, :cpu) do
      :cpu -> "cpu"
      :vulkan -> "vulkan"
      other -> raise ArgumentError, "unknown NxGgml device #{inspect(other)}, expected :cpu or :vulkan"
    end
  end

  @doc "Whether the `:vulkan` device is actually usable on this build/machine."
  @spec vulkan_available?() :: boolean
  def vulkan_available? do
    NxGgml.Nif.nx_ggml_vulkan_available()
  end
end
