module AMReX

using AMReX_jll
using CxxWrap
using Libdl
using OffsetArrays
using PrecompileTools: @setup_workload, @compile_workload
using Preferences

const _SHIM_LIB_PATH = abspath(joinpath(@__DIR__, "..", "deps", "lib", "libamrex_jl_shim.so"))

function get_shim_library_path()
    custom_lib = @load_preference("custom_shim_lib", nothing)
    if custom_lib !== nothing && isfile(custom_lib)
        return custom_lib
    end
    if !isfile(_SHIM_LIB_PATH)
        include(joinpath(@__DIR__, "..", "deps", "build.jl"))
        Base.invokelatest(build_cshim)
    end
    return _SHIM_LIB_PATH
end

"""
    set_amrex_prefix!(amrex_prefix::Union{Nothing, String})

Configure `AMReX.jl` via `Preferences.jl` to use either the default `AMReX_jll` CPU
binary (`nothing`) or a custom local AMReX installation (e.g., a CUDA-enabled `libamrex`
build), and recompile the C++ shim library.
"""
function set_amrex_prefix!(amrex_prefix::Union{Nothing, String})
    @set_preferences!("amrex_prefix" => amrex_prefix)
    include(joinpath(@__DIR__, "..", "deps", "build.jl"))
    Base.invokelatest(build_cshim; custom_amrex_prefix=amrex_prefix)
    @info "Updated AMReX prefix preference to $(something(amrex_prefix, "AMReX_jll default")). Restart Julia for changes to take effect."
end

import Base: length, size, contains

@wrapmodule(get_shim_library_path)

function __init__()
    # Ensure all JLL dependency shared libraries (MPICH, HDF5, etc.) are dlopen-able
    for libdir in AMReX_jll.LIBPATH_list
        if isdir(libdir)
            push!(Libdl.DL_LOAD_PATH, libdir)
        end
    end
    @initcxx
end

include("box_geometry.jl")
include("multifab.jl")
include("amrcore.jl")
include("particles.jl")
include("io.jl")

amrex_init() = amrex_init(false)

export amrex_init, amrex_initialized, amrex_finalize
export parallel_myproc, parallel_nprocs, parallel_ioprocessor, parallel_barrier
export parallel_reduce_sum_real, parallel_reduce_max_real, parallel_reduce_min_real
export parallel_reduce_sum_long, parallel_reduce_sum_real_vec!

export JlBox, JlBoxAllocated, Box, bounds, lo, hi, is_nodal, num_pts, is_empty, contains
export surrounding_nodes, enclosed_cells, grow, intersect
export JlRealBox, JlRealBoxAllocated, RealBox
export JlGeometry, JlGeometryAllocated, Geometry, cell_size, inv_cell_size, prob_lo, prob_hi, domain, prob_domain, is_periodic
export cell_sizes, inv_cell_sizes, prob_lo_tuple, prob_hi_tuple
export JlBoxArray, JlBoxArrayAllocated, BoxArray, max_size!, convert_to_nodal, nodal_box_array, get_box, contains_cell
export JlDistributionMapping, JlDistributionMappingAllocated, DistributionMapping, get_rank

export JlMultiFab, JlMultiFabAllocated, MultiFab, JliMultiFab, JliMultiFabAllocated, iMultiFab, JlMFIter, JlMFIterAllocated
export n_comp, n_grow, local_size, box_array, dmap
export set_val!, set_val_comp!, plus_val!, mult_val!, add_from!, copy_from!, parallel_copy_from!
export copy!, plus!, scale!, saxpy!, lincomb!
export fill_boundary!, fill_boundary_comp!, sum_boundary!, sum_boundary_comp!
export min_val, max_val, norm0, norm1, norm2, sum_val
export is_valid, next!, close!, index, grid_index, local_tile_index, valid_box, tile_box, fab_box
export fab_lo, fab_hi, ifab_lo, ifab_hi, array4, iarray4, each_fab

export JlAmrCore, JlAmrCoreAllocated, AmrCore, InterpType, CellConservativeLinear, NodeBilinear, PiecewiseConstant, CellQuadratic
export add_refine_box!, add_tag_box!, clear_refine_boxes!, init_from_scratch!, regrid_levels!
export finest_level, max_level, geom, geometry, dmap, dist_map, node_box_array, ref_ratio, ref_ratio_tuple
export mpi_rank, parallel_reduce_sum
export set_box_array!, set_dmap!, set_finest_level!
export interp_from_coarse_to_fine!, fill_patch_single_level!, fill_patch_two_levels!
export average_down!, average_down_nodal!, sum_fine_to_coarse_nodal!

export JlParticleContainer, JlParticleContainerAllocated, ParticleContainer, JlParIter, JlParIterAllocated
export pack_idcpu, unpack_id, unpack_cpu, is_valid_idcpu, make_invalid_idcpu, make_valid_idcpu
export next_particle_id, reserve_and_resize_data!, ensure_tile!, resize_tile!, tile_num_particles, num_particles
export idcpu_vec, pos_vecs, real_vec, int_vec
export tile_idcpu_vec, tile_pos_vecs, tile_real_vec, tile_int_vec
export each_particle_tile, redistribute!, total_number_of_particles, number_of_particles_at_level, clear_particles!
export write_plotfile, checkpoint, set_amrex_prefix!

@setup_workload begin
    @compile_workload begin
        w = pack_idcpu(Int64(42), 0)
        _ = unpack_id(w)
        _ = unpack_cpu(w)
        _ = is_valid_idcpu(w)
        w_inv = make_invalid_idcpu(w)
        _ = make_valid_idcpu(w_inv)
    end
end

end # module AMReX
