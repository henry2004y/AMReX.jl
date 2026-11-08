# API Reference

## Runtime & Parallel Utilities

```@docs
AMReX
amrex_init
amrex_initialized
amrex_finalize
set_amrex_prefix!
mpi_rank
parallel_reduce_sum
```

## Index Space, Geometry & Domain Decomposition

```@docs
Box
bounds
RealBox
Geometry
cell_sizes
inv_cell_sizes
prob_lo_tuple
prob_hi_tuple
BoxArray
nodal_box_array
DistributionMapping
```

## `MultiFab`, `iMultiFab` & Zero-Copy Iterators

```@docs
MultiFab
iMultiFab
array4
iarray4
each_fab
n_grow
AMReX.copy!
plus!
scale!
saxpy!
lincomb!
```

## Multi-Level `AmrCore`

```@docs
InterpType
AmrCore
ref_ratio_tuple
add_tag_box!
geometry
dist_map
```

## Pure-SoA `ParticleContainer`

```@docs
ParticleContainer
pack_idcpu
unpack_id
unpack_cpu
is_valid_idcpu
make_invalid_idcpu
make_valid_idcpu
idcpu_vec
pos_vecs
real_vec
int_vec
tile_idcpu_vec
tile_pos_vecs
tile_real_vec
tile_int_vec
each_particle_tile
redistribute!
total_number_of_particles
number_of_particles_at_level
```

## Plotfile I/O

```@docs
write_plotfile
```
