# Multi-Level AMR & Particles

## Managing Mesh Hierarchies with `AmrCore`

`AmrCore` manages a hierarchy of refined levels `0:finest_level(amr)`:

```julia
rbox = RealBox(0.0, 0.0, 0.0, 10.0, 10.0, 1.0)
amr = AmrCore(rbox, (32, 32, 1);
              max_level=1,
              is_periodic=(true, true, true),
              max_grid_size=(16, 16, 1),
              blocking_factor=(4, 4, 1))

# Tag a physical region [2.5, 7.5] x [2.5, 7.5] for refinement to level 1
add_tag_box!(amr, 2.5, 2.5, 0.0, 7.5, 7.5, 1.0; lev=0)
init_from_scratch!(amr, 0.0)

@assert finest_level(amr) == 1
```

### Coarse-Fine Field Synchronization

`AMReX.jl` wraps AMReX's multi-level prolongation and restriction operators:

- **`fill_patch_two_levels!`**: Fills a fine-level `MultiFab` (including its coarse-fine ghost boundary) by interpolating from the coarse level (`CellConservativeLinear`, `NodeBilinear`, `PiecewiseConstant`, or `CellQuadratic`) and copying from same-level boxes.
- **`average_down!`** / **`average_down_nodal!`**: Restricts fine-level cell-centered or nodal fields onto the underlying coarse level.
- **`sum_fine_to_coarse_nodal!`**: Accumulates nodal fine-level quantities (such as deposited moments or currents) onto shared coarse nodes.

## Pure-SoA `ParticleContainer`

`ParticleContainer` wraps `amrex::AmrParticleContainer` configured in **Pure Structure-of-Arrays (SoA)** layout (`AoS` has 0 extra components; positions `x, y, z`, 64-bit `idcpu`, `n_real` `Float64` attributes, and `n_int` `Int32` attributes are stored as contiguous 1D arrays per tile).

```julia
pc = ParticleContainer(amr; n_real=4, n_int=1)
reserve_and_resize_data!(pc)
```

### Direct Zero-Copy Tile Access

You can iterate over all active tiles at level `lev` with `each_particle_tile`:

```julia
each_particle_tile(pc, 0) do pti
    np = Int(num_particles(pti))
    px, py, pz = pos_vecs(pti)
    idcpu = idcpu_vec(pti)
    qp = real_vec(pti, 1)
    up = real_vec(pti, 2)
    vp = real_vec(pti, 3)
    wp = real_vec(pti, 4)

    @inbounds for p in 1:np
        if is_valid_idcpu(idcpu[p])
            px[p] += 0.01 * up[p]
            py[p] += 0.01 * vp[p]
            pz[p] += 0.01 * wp[p]
        end
    end
end

# Reassign particles to their new AMR levels, grids, and MPI ranks
redistribute!(pc)
```

### Bit-Packed 64-Bit `idcpu` Format

AMReX packs each particle's validity sign bit (bit 63), 39-bit particle ID (bits 24–62), and 24-bit originating MPI rank (bits 0–23) into a single `UInt64`:

- `w = pack_idcpu(id, cpu)`
- `id = unpack_id(w)`
- `cpu = unpack_cpu(w)`
- `is_valid_idcpu(w)`
- `w_invalid = make_invalid_idcpu(w)` — marks a particle for deletion during the next `redistribute!(pc)` call.

