```@raw html
---
layout: home

hero:
  name: "AMReX.jl"
  text: "Block-Structured AMR in Julia"
  tagline: "Zero-copy OffsetArray field views, pure-SoA particle containers, and multi-level adaptive mesh refinement powered by AMReX and CxxWrap.jl."
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: API Reference
      link: /api

features:
  - title: Zero-Copy 4D OffsetArray Views
    details: Access cell-centered and nodal MultiFab / iMultiFab FArrayBoxes directly with 0-based AMReX index ranges (rx, ry, rz, 1:ncomp) and zero heap allocations in inner loops.
  - title: Pure-SoA Particle Containers
    details: High-throughput Structure-of-Arrays (SoA) particle storage with 64-bit packed idcpu words, direct tile vector views, and multi-level MPI redistribution.
  - title: Block-Structured Multi-Level AMR & Linear Solvers
    details: Manage hierarchical meshes via AmrCore with dynamic TagBoxArray error tagging, physical boundary conditions (BCRec / PhysBCFunct), YAFluxRegister / EdgeFluxRegister refluxing, and MLMG multi-level multigrid solvers (MLPoisson, MLABecLaplacian, MLNodeLaplacian).
  - title: MPI Parallelism, HDF5 & Checkpoint/Restart I/O
    details: Ready out of the box with AMReX_jll (MPICH + HDF5) for N-level plotfiles, HDF5 plotfile/particle I/O, VisMF field checkpoints, and particle checkpoint/restart.
---
```

# Overview

**AMReX.jl** provides an ergonomic, zero-overhead Julia interface to the [AMReX](https://amrex-codes.github.io/amrex/) C++ framework for block-structured Adaptive Mesh Refinement (AMR). It is designed as the foundational mesh and particle layer for kinetic plasma and fluid solvers such as **FLEKS.jl**.

## Quick Example

```julia
using AMReX

AMReX.amrex_init()

# 1. Define a 3D index domain [0:31, 0:31, 0:31] and physical box [0, 1]^3
dom  = Box(0, 0, 0, 31, 31, 31)
rbox = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
geom = Geometry(dom, rbox; is_periodic=(true, true, true))

# 2. Decompose into 16^3 patches and distribute across MPI ranks
ba = BoxArray(dom; max_grid_size=16)
dm = DistributionMapping(ba)

# 3. Allocate a 3-component MultiFab with 1 ghost cell layer
mf = MultiFab(ba, dm, 3, 1)
set_val!(mf, 0.0)

# 4. Populate fields using zero-copy 4D OffsetArray views
dx, dy, dz = cell_sizes(geom)
each_fab(mf) do mfi, vbx, arr
    rx, ry, rz = bounds(vbx)
    @inbounds for k in rz, j in ry, i in rx
        x = (i + 0.5) * dx
        y = (j + 0.5) * dy
        arr[i, j, k, 1] = sin(2π * x) * cos(2π * y)
    end
end

# 5. Synchronize periodic and inter-patch ghost cells
fill_boundary!(mf, geom)

AMReX.amrex_finalize()
```

