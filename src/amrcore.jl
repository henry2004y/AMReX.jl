# ============================================================================
# AmrCore & Multi-Level AMR Interpolation Utilities
# ============================================================================

@enum InterpType::Int32 begin
    CellConservativeLinear = 0
    NodeBilinear           = 1
    PiecewiseConstant      = 2
    CellQuadratic          = 3
end

"""
    AmrCore(rbox::JlRealBox, ncells::NTuple{3, Integer};
            max_level::Integer=0,
            coord::Integer=0,
            is_periodic::NTuple{3, Bool}=(true, true, true),
            max_grid_size::NTuple{3, Integer}=(32, 32, 32),
            blocking_factor::NTuple{3, Integer}=(4, 4, 4),
            grid_eff::Real=0.7)

Construct an AMReX `JlAmrCore` multi-level mesh hierarchy manager.
Automatically sets the z-direction blocking factor to 1 when `ncells[3] == 1` (Fake-1D/2D).
"""
function AmrCore(rbox::JlRealBox, ncells::NTuple{3, Integer};
                 max_level::Integer=0,
                 coord::Integer=0,
                 is_periodic::NTuple{3, Bool}=(true, true, true),
                 max_grid_size::NTuple{3, Integer}=(32, 32, 32),
                 blocking_factor::NTuple{3, Integer}=(4, 4, 4),
                 grid_eff::Real=0.7)
    nx, ny, nz = Int32.(ncells)
    mgx, mgy, mgz = Int32.(max_grid_size)
    bx = Int32(min(blocking_factor[1], ncells[1]))
    by = Int32(min(blocking_factor[2], ncells[2]))
    bz = Int32(ncells[3] == 1 ? 1 : min(blocking_factor[3], ncells[3]))
    return JlAmrCore(rbox, Int32(max_level),
                     nx, ny, nz, Int32(coord),
                     is_periodic[1], is_periodic[2], is_periodic[3],
                     mgx, mgy, mgz, bx, by, bz, Float64(grid_eff))
end

"""
    ref_ratio_tuple(amr::JlAmrCore, lev::Integer) -> NTuple{3, Int}

Return the 3D refinement ratio between `lev` and `lev + 1`.
"""
@inline function ref_ratio_tuple(amr::JlAmrCore, lev::Integer)
    return (
        Int(ref_ratio(amr, Int32(lev), Int32(0))),
        Int(ref_ratio(amr, Int32(lev), Int32(1))),
        Int(ref_ratio(amr, Int32(lev), Int32(2))),
    )
end

@inline add_tag_box!(amr::JlAmrCore, xlo::Real, ylo::Real, zlo::Real, xhi::Real, yhi::Real, zhi::Real) =
    add_refine_box!(amr, Float64(xlo), Float64(ylo), Float64(zlo), Float64(xhi), Float64(yhi), Float64(zhi))

@inline geometry(amr::JlAmrCore, lev::Integer=0) = geom(amr, Int32(lev))
@inline dist_map(amr::JlAmrCore, lev::Integer=0) = dmap(amr, Int32(lev))
@inline mpi_rank() = Int(parallel_myproc())
@inline grid_index(it) = Int(index(it))
@inline parallel_reduce_sum(x::Real) = parallel_reduce_sum_real(Float64(x))
