# Getting Started

## Installation

`AMReX.jl` uses `AMReX_jll` and `CxxWrap.jl` to compile a lightweight C++ shim (`libamrex_jl_shim.so`) automatically upon first load or package build.

```julia
using Pkg
Pkg.develop(path="/path/to/AMReX.jl")
```

## Runtime Lifecycle

AMReX manages an internal arena allocator and MPI communicator. Initialize AMReX before constructing any `MultiFab`, `AmrCore`, or `ParticleContainer`, and finalize it when your program exits:

```julia
using AMReX

amrex_init()            # Passes verbose=false by default
@assert amrex_initialized()

# ... run simulation ...

amrex_finalize()
```

!!! tip "Garbage Collection Before `amrex_finalize()`"
    `AMReX.jl` guards C++ destructors (`MFIter`, `ParIter`, `ParticleContainer`) against post-finalize teardown, and shares ownership of `AmrCore` with any child `ParticleContainer`. Calling `GC.gc()` prior to `amrex_finalize()` ensures deterministic cleanup of all C++ arena memory before MPI shuts down.

## Index Space & Physical Geometry

AMReX uses **0-based inclusive** integer indices `(i, j, k)` for cells and nodes:

- A cell-centered domain of $N_x \times N_y \times N_z$ cells spans `0:(Nx-1)` in each direction.
- A nodal (corner) box surrounding those cells spans `0:Nx` in each direction.

```julia
# Cell-centered box [0:63, 0:63, 0:0] (Fake-2D domain with 1 cell in z)
dom = Box(0, 0, 0, 63, 63, 0)

# Convert to nodal box [0:64, 0:64, 0:1]
nbx = surrounding_nodes(dom)

# Physical coordinates [0.0, 10.0] x [0.0, 10.0] x [0.0, 1.0]
rbox = RealBox(0.0, 0.0, 0.0, 10.0, 10.0, 1.0)
geom = Geometry(dom, rbox; is_periodic=(true, true, true))

dx, dy, dz = cell_sizes(geom)
inv_dx, inv_dy, inv_dz = inv_cell_sizes(geom)
```

## Domain Decomposition (`BoxArray` & `DistributionMapping`)

A `BoxArray` partitions the index domain into block-structured patches, and a `DistributionMapping` assigns each box to an MPI process via space-filling curve (SFC) or knapsack load balancing:

```julia
ba = BoxArray(dom; max_grid_size=(32, 32, 1))
dm = DistributionMapping(ba)

# Create a nodal BoxArray sharing the same DistributionMapping
nba = nodal_box_array(ba; dirs=(true, true, true))
```

## Plotfile Output

You can export single-level or multi-level `MultiFab` data to standard AMReX plotfiles (compatible with ParaView, VisIt, and `yt`):

```julia
mf = MultiFab(ba, dm, 3, 1)
set_val!(mf, 1.0)

write_plotfile("plt00000", mf, geom;
               varnames=["Bx", "By", "Bz"],
               time=0.0, step=0)
```

