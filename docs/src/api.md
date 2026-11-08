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

## Multi-Level `AmrCore`, Physical BCs & Dynamic Error Tagging

```@docs
InterpType
MathBCType
BCRec
fill_domain_boundary!
TagVal
TagBoxArray
tag_array4
AmrCore
ref_ratio_tuple
add_tag_box!
on_error_est!
tag_cells_by_value!
tag_cells_by_grad!
init_from_scratch!
regrid_levels!
geometry
dist_map
interp_from_coarse_to_fine!
fill_patch_single_level!
fill_patch_two_levels!
average_down!
average_down_nodal!
sum_fine_to_coarse_nodal!
```

## Flux Registers & Staggered Averaging

```@docs
YAFluxRegister
EdgeFluxRegister
reset!
crse_add!
fine_add!
reflux!
crse_has_work
fine_has_work
average_down_faces!
average_down_face!
average_down_edges!
```

## Multi-Level Linear Solvers (`MLMG`)

```@docs
LinOpBCType
BottomSolver
MLMGLocation
LPInfo
MLPoisson
MLABecLaplacian
MLNodeLaplacian
MLMG
set_domain_bc!
set_domain_bc_loc!
set_level_bc!
set_coarse_fine_bc!
set_scalars!
set_a_coeffs!
set_b_coeffs!
set_sigma!
comp_divergence!
update_velocity!
solve!
get_grad_solution!
get_fluxes!
comp_residual!
apply!
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

## Plotfile, HDF5 & `VisMF` Checkpoint/Restart I/O

```@docs
write_plotfile
write_plotfile_hdf5
checkpoint_hdf5
restart!
restart_hdf5!
```
