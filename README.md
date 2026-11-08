# AMReX.jl

High-performance Julia bindings for the [AMReX](https://amrex-codes.github.io/amrex/) C++ framework for block-structured Adaptive Mesh Refinement (AMR), powered by [`CxxWrap.jl`](https://github.com/JuliaInterop/CxxWrap.jl) and [`AMReX_jll.jl`](https://github.com/JuliaBinaryWrappers/AMReX_jll.jl).

## Key Features

- **Zero-Copy 4D `OffsetArray` Views**: Access cell-centered, face-centered, edge-centered, and nodal `MultiFab`, `iMultiFab`, and `TagBoxArray` patches directly with 0-based AMReX index ranges `(rx, ry, rz, 1:ncomp)` and zero heap allocations in inner loops via `each_fab`.
- **Multi-Level `AmrCore` & Dynamic Error Tagging**: Manage hierarchical AMR grids with coarse-fine interpolation (`CellConservativeLinear`, `NodeBilinear`, `PiecewiseConstant`, `CellQuadratic`), physical boundary conditions (`BCRec`, `fill_domain_boundary!`), `FillPatchSingleLevel` / `FillPatchTwoLevels`, and dynamic regridding via `TagBoxArray` and Julia callbacks (`on_error_est!`, `tag_cells_by_value!`, `tag_cells_by_grad!`).
- **Flux Registers & Staggered Averaging**: Enforce conservation and divergence/curl constraints across coarse-fine interfaces with `YAFluxRegister`, `EdgeFluxRegister`, `average_down!`, `average_down_nodal!`, `average_down_faces!`, and `average_down_edges!`.
- **Multi-Level Linear Solvers (`MLMG`)**: Solve elliptic equations across single-level and multi-level AMR hierarchies using `MLPoisson`, `MLABecLaplacian`, and `MLNodeLaplacian` with configurable boundary conditions (`LinOpBCType`), bottom solvers (`BottomSolver`), gradients, and fluxes.
- **Pure-SoA `ParticleContainer`**: High-throughput Structure-of-Arrays (SoA) particle storage with 64-bit packed `idcpu` words (`pack_idcpu`, `unpack_id`, `unpack_cpu`), zero-copy 1D `Vector` views over particle tiles, and multi-level/multi-rank `redistribute!`.
- **Plotfiles, HDF5 & Checkpoint/Restart I/O**: Write and read arbitrary $N$-level native and HDF5 plotfiles (`write_plotfile`, `write_plotfile_hdf5`), `VisMF` field checkpoints (`vismf_write`, `vismf_read!`), and native or HDF5 particle checkpoints (`checkpoint`, `restart!`, `checkpoint_hdf5`, `restart_hdf5!`).
- **MPI Parallelism**: Works out of the box with `AMReX_jll` (MPICH + HDF5) for multi-process domain decomposition, halo exchange (`fill_boundary!`, `sum_boundary!`, `parallel_copy_from!`), global reductions (`parallel_reduce_*`), and cross-rank particle migration.

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/AMReX.jl")
```

On first load (or via `julia --project=AMReX.jl AMReX.jl/deps/build.jl`), `AMReX.jl` compiles its lightweight C++ shim library (`deps/lib/libamrex_jl_shim.so`) against `AMReX_jll` and `libcxxwrap_julia_jll`.

To switch between the default `AMReX_jll` binary and a custom local AMReX installation via `Preferences.jl`:

```julia
using AMReX
set_amrex_prefix!("/path/to/custom/amrex/install") # or `nothing` to restore AMReX_jll default
```

## Quick Start

```julia
using AMReX

amrex_init()

# 1. Define a 3D index domain [0:31, 0:31, 0:31] and physical box [0, 1]^3
dom  = Box(0, 0, 0, 31, 31, 31)
rbox = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
geom = Geometry(dom, rbox; is_periodic=(false, false, false))

# 2. Decompose into 16^3 patches and distribute across MPI ranks
ba = BoxArray(dom; max_grid_size=16)
dm = DistributionMapping(ba)

# 3. Allocate solution and RHS MultiFabs
sol = MultiFab(ba, dm, 1, 1)
rhs = MultiFab(ba, dm, 1, 0)
set_val!(sol, 0.0)

dx, dy, dz = cell_sizes(geom)
each_fab(rhs) do mfi, vbx, arr
    rx, ry, rz = bounds(vbx)
    @inbounds for k in rz, j in ry, i in rx
        x = (i + 0.5) * dx
        y = (j + 0.5) * dy
        z = (k + 0.5) * dz
        arr[i, j, k, 1] = -3π^2 * sin(π * x) * sin(π * y) * sin(π * z)
    end
end

# 4. Solve ∇²ϕ = rhs with homogeneous Dirichlet BCs using MLMG
poisson = MLPoisson(geom, ba, dm)
set_domain_bc!(poisson,
               (LODirichlet, LODirichlet, LODirichlet),
               (LODirichlet, LODirichlet, LODirichlet))
set_level_bc!(poisson, 0, nothing)

mlmg = MLMG(poisson; verbose=0)
solve!(mlmg, sol, rhs; reltol=1e-10)

# 5. Write plotfile
write_plotfile("plt00000", sol, geom; varnames=["phi"], time=0.0, step=0)

amrex_finalize()
```

## Running with MPI

You can launch multi-process MPI scripts using the MPICH `mpiexec` bundled with `AMReX_jll`:

```bash
julia --project=AMReX.jl -e '
using AMReX_jll
run(`$(AMReX_jll.MPICH_jll.mpiexec()) -n 2 $(Base.julia_cmd()) --project=AMReX.jl my_script.jl`)
'
```

## Running the Test Suite

Run the full test suite (including `Aqua.jl` code quality checks, single-rank unit tests, and the 2-process MPI subprocess suite):

```bash
julia --project=AMReX.jl -e 'using Pkg; Pkg.test()'
```

You can also run individual test suites via `test_args` or run the 2-rank MPI test directly:

```bash
# Run only the MLMG and MPI suites
julia --project=AMReX.jl -e 'using Pkg; Pkg.test(; test_args=["mlmg", "mpi"])'

# Run the 2-rank MPI test script directly
julia --project=AMReX.jl -e 'using AMReX_jll; run(`$(AMReX_jll.MPICH_jll.mpiexec()) -n 2 $(Base.julia_cmd()) --project=AMReX.jl AMReX.jl/test/test_mpi.jl`)'
```
