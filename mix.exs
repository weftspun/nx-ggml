defmodule NxGgml.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_ggml,
      version: "0.1.0",
      elixir: "~> 1.20",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_makefile: "Makefile",
      make_executable: make_executable(),
      make_env: fn -> %{"FINE_INCLUDE_DIR" => Fine.include_dir()} end,
      deps: deps(),
      dialyzer: [plt_add_apps: [:nx, :mix]]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.7"},
      {:elixir_make, "~> 0.7", runtime: false},
      {:fine, "~> 0.1", runtime: false},
      {:stream_data, "~> 1.0", only: :test},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
    ]
  end

  # elixir_make defaults to nmake/make on Windows, neither of which is
  # installed on this dev machine (only llvm-mingw's mingw32-make). Allow
  # overriding via NX_GGML_MAKE for machines that do have a real make/nmake.
  defp make_executable do
    case System.get_env("NX_GGML_MAKE") do
      nil ->
        case :os.type() do
          {:win32, _} -> "mingw32-make"
          _ -> :default
        end

      exec ->
        exec
    end
  end
end
