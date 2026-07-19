# Vulkan tests (test/nx_ggml/vulkan_test.exs) are tagged `:vulkan` and
# excluded unless the native build was configured with NX_GGML_VULKAN=ON
# *and* a Vulkan-capable GPU was found at NIF load
# (NxGgml.Device.vulkan_available?/0) -- see that test file's moduledoc.
exclude = if NxGgml.Device.vulkan_available?(), do: [], else: [:vulkan]

ExUnit.start(exclude: exclude)
