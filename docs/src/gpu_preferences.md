# GPU & Custom AMReX Builds

By default, `AMReX.jl` links against `AMReX_jll`, which provides a prebuilt CPU + MPI `libamrex.so`.

## Switching to a Custom or GPU-Enabled AMReX Build

Using `Preferences.jl`, you can point `AMReX.jl` to any local installation of AMReX (for example, an AMReX library compiled with `-DAMReX_GPU_BACKEND=CUDA` or `-DAMReX_OMP=ON`):

```julia
using AMReX

# Point to a custom AMReX install prefix and rebuild libamrex_jl_shim.so:
AMReX.set_amrex_prefix!("/opt/amrex-cuda")
```

This writes a `LocalPreferences.toml` entry (`amrex_prefix`) in your active environment and invokes CMake to recompile `deps/lib/libamrex_jl_shim.so` against your custom `AMReXConfig.cmake`.

To revert to the default `AMReX_jll` artifact at any time:

```julia
AMReX.set_amrex_prefix!(nothing)
```

