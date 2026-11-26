# GPU & Custom AMReX Builds

By default, `AMReX.jl` links against `AMReX_jll`, which provides a prebuilt CPU + MPI `libamrex_3d`.

## Switching to a Custom or GPU-Enabled AMReX Build

Using `Preferences.jl` and `Scratch.jl`, you can point `AMReX.jl` to any local installation of AMReX (for example, an AMReX library compiled with `-DAMReX_GPU_BACKEND=CUDA` or `-DAMReX_OMP=ON`, or against a system OpenMPI installation):

```julia
using AMReX

# Point to a custom AMReX install prefix and compile the shim into Scratch.jl storage:
AMReX.set_amrex_prefix!("/opt/amrex-cuda")
```

This writes a `LocalPreferences.toml` entry (`amrex_prefix`) in your active environment and invokes the configured C++ compiler (`ENV["CXX"]`, defaulting to `g++`) via `deps/build.jl` to compile `libamrex_jl_shim` into a per-(Julia version, prefix, architecture) `Scratch.jl` directory, without writing into the package source tree or leaking `AMReX_jll` (`MPICH_jll`) RPATHs.

You may also point directly to a prebuilt shim shared library via `Preferences.set_preferences!(AMReX, "custom_shim_lib" => "/path/to/libamrex_jl_shim.so")`.

To revert to the default `AMReX_jll` artifact at any time:

```julia
AMReX.set_amrex_prefix!(nothing)
```
