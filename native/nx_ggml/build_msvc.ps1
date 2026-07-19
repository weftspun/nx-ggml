# Configures and builds native/nx_ggml using real MSVC (cl.exe/link.exe),
# not Clang. This is required, not cosmetic: a minimal repro proved that
# Clang/lld-link on this Windows toolchain does not run the C++ global
# static initializers that Fine's NIF registration relies on for a
# MODULE-type DLL (Registration::erl_nif_funcs stays empty, so
# :erlang.load_nif succeeds but binds zero functions and every NIF call
# falls through to its Elixir stub). Real MSVC does not have this problem
# and matches the toolchain Fine's own CI tests on Windows
# (windows-2022 + ilammy/msvc-dev-cmd, i.e. vcvarsall-initialized cl/link).
#
# Implemented in PowerShell (not a .bat invoked via `cmd /c` from the
# Makefile) because invoking cmd.exe from an MSYS/git-bash shell (which is
# what elixir_make's Makefile runs under via mingw32-make) silently mangles
# arguments that look like Unix paths (e.g. `/c` itself, or `c:/...`
# forward-slash paths), corrupting the command line. PowerShell has no such
# mangling.

param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$ErtsIncludeDir,
    [Parameter(Mandatory = $true)][string]$FineIncludeDir,
    [Parameter(Mandatory = $true)][string]$PrivDir,
    [string]$NxGgmlVulkan = "OFF"
)

$ErrorActionPreference = "Stop"

function Find-VsWhere {
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    throw "nx_ggml: could not locate vswhere.exe. Install Visual Studio Build Tools (Desktop development with C++) and retry."
}

function Import-VcVarsAll {
    $vswhere = Find-VsWhere
    $installPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installPath) {
        throw "nx_ggml: vswhere found no Visual Studio install with the VC.Tools.x86.x64 component."
    }

    $vcvarsall = Join-Path $installPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        throw "nx_ggml: vcvarsall.bat not found at $vcvarsall"
    }

    # cmd.exe here is invoked from PowerShell (not from an MSYS/git-bash
    # shell), so none of the argument-mangling that motivated this script
    # applies to this specific call.
    $envDump = & cmd /c "call `"$vcvarsall`" amd64 >nul 2>&1 && set"
    foreach ($line in $envDump) {
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    if (-not $env:DevEnvDir -and -not $env:VCToolsInstallDir) {
        throw "nx_ggml: vcvarsall.bat ran but did not appear to initialize the MSVC environment."
    }
}

Import-VcVarsAll

New-Item -ItemType Directory -Force -Path $PrivDir | Out-Null

& cmake -S native/nx_ggml -B $BuildDir -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=cl `
    -DCMAKE_CXX_COMPILER=cl `
    -DERTS_INCLUDE_DIR="$ErtsIncludeDir" `
    -DFINE_INCLUDE_DIR="$FineIncludeDir" `
    -DNX_GGML_PRIV_DIR="$PrivDir" `
    -DNX_GGML_VULKAN="$NxGgmlVulkan"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build $BuildDir --target nx_ggml_nif --config Release
exit $LASTEXITCODE
