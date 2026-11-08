# ============================================================================
# Linear Solvers: LinOpBCType, LPInfo, MLPoisson, MLABecLaplacian,
# MLNodeLaplacian, and MLMG
# ============================================================================

"""
    LinOpBCType

AMReX linear operator boundary condition enum (`amrex::LinOpBCType` from `AMReX_LO_BCTYPES.H`).
"""
@enum LinOpBCType::Int32 begin
    LOInterior         = 0
    LODirichlet        = 101
    LONeumann          = 102
    LOReflectOdd       = 103
    LOMarshak          = 104
    LOSanchezPomraning = 105
    LOInflow           = 106
    LOInhomogNeumann   = 107
    LORobin            = 108
    LOSymmetry         = 109
    LOPeriodic         = 200
    LOBogus            = 1729
end

"""
    BottomSolver

AMReX `MLMG` bottom solver enum (`amrex::BottomSolver`).
"""
@enum BottomSolver::Int32 begin
    BottomDefault  = 0
    BottomSmoother = 1
    BottomBiCGStab = 2
    BottomCG       = 3
    BottomBiCGCG   = 4
    BottomCGBiCG   = 5
    BottomHypre    = 6
    BottomPETSc    = 7
end

"""
    MLMGLocation

Location enum (`amrex::MLLinOp::Location`) for `get_grad_solution!` and `get_fluxes!`.
"""
@enum MLMGLocation::Int32 begin
    LocFaceCenter   = 0
    LocFaceCentroid = 1
    LocCellCenter   = 2
    LocCellCentroid = 3
end

const _MLLinOpUnion = Union{JlMLPoisson, JlMLABecLaplacian, JlMLNodeLaplacian}

"""
    LPInfo(; max_coarsening_level=30, do_agglomeration=true, do_consolidation=true,
             agg_grid_size=-1, con_grid_size=-1, has_metric_term=true,
             hidden_direction=-1) -> JlLPInfoAllocated

Construct an `LPInfo` configuration struct for AMReX linear operators.
Set `hidden_direction=2` (0-based z-direction) when solving 2D problems on a 3D `nz=1` grid.
"""
function LPInfo(; max_coarsening_level::Integer=30,
                do_agglomeration::Bool=true,
                do_consolidation::Bool=true,
                agg_grid_size::Integer=-1,
                con_grid_size::Integer=-1,
                has_metric_term::Bool=true,
                hidden_direction::Integer=-1)
    return JlLPInfo(
        Int32(max_coarsening_level),
        do_agglomeration,
        do_consolidation,
        Int32(agg_grid_size),
        Int32(con_grid_size),
        has_metric_term,
        Int32(hidden_direction)
    )
end

"""
    MLPoisson(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
              info::JlLPInfo=LPInfo()) -> JlMLPoissonAllocated
    MLPoisson(geoms::AbstractVector{<:JlGeometry}, bas::AbstractVector{<:JlBoxArray},
              dms::AbstractVector{<:JlDistributionMapping};
              info::JlLPInfo=LPInfo()) -> JlMLPoissonAllocated
    MLPoisson(amr::JlAmrCore; info::JlLPInfo=LPInfo()) -> JlMLPoissonAllocated

Construct a cell-centered Poisson linear operator `∇² φ = rhs`.
"""
function MLPoisson(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
                   info::JlLPInfo=LPInfo())
    return JlMLPoisson(geom, ba, dm, info)
end

function MLPoisson(geoms::AbstractVector{<:JlGeometry},
                   bas::AbstractVector{<:JlBoxArray},
                   dms::AbstractVector{<:JlDistributionMapping};
                   info::JlLPInfo=LPInfo())
    length(geoms) == length(bas) == length(dms) ||
        throw(ArgumentError("geoms, bas, and dms must have the same number of levels"))
    op = JlMLPoisson()
    stage_clear!(op)
    for l in eachindex(geoms)
        stage_push_level!(op, geoms[l], bas[l], dms[l])
    end
    define_staged!(op, info)
    return op
end

function MLPoisson(amr::JlAmrCore; info::JlLPInfo=LPInfo())
    nlevs = Int(finest_level(amr)) + 1
    geoms = [geom(amr, Int32(l - 1)) for l in 1:nlevs]
    bas   = [box_array(amr, Int32(l - 1)) for l in 1:nlevs]
    dms   = [dmap(amr, Int32(l - 1)) for l in 1:nlevs]
    return MLPoisson(geoms, bas, dms; info=info)
end

"""
    MLABecLaplacian(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
                    info::JlLPInfo=LPInfo(), ncomp::Integer=1) -> JlMLABecLaplacianAllocated
    MLABecLaplacian(geoms::AbstractVector{<:JlGeometry}, bas::AbstractVector{<:JlBoxArray},
                    dms::AbstractVector{<:JlDistributionMapping};
                    info::JlLPInfo=LPInfo(), ncomp::Integer=1) -> JlMLABecLaplacianAllocated
    MLABecLaplacian(amr::JlAmrCore; info::JlLPInfo=LPInfo(), ncomp::Integer=1) -> JlMLABecLaplacianAllocated

Construct a canonical cell-centered Helmholtz / diffusion operator `(A α - B ∇ ⋅ (β ∇)) φ = rhs`.
"""
function MLABecLaplacian(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
                         info::JlLPInfo=LPInfo(), ncomp::Integer=1)
    return JlMLABecLaplacian(geom, ba, dm, info, Int32(ncomp))
end

function MLABecLaplacian(geoms::AbstractVector{<:JlGeometry},
                         bas::AbstractVector{<:JlBoxArray},
                         dms::AbstractVector{<:JlDistributionMapping};
                         info::JlLPInfo=LPInfo(), ncomp::Integer=1)
    length(geoms) == length(bas) == length(dms) ||
        throw(ArgumentError("geoms, bas, and dms must have the same number of levels"))
    op = JlMLABecLaplacian()
    stage_clear!(op)
    for l in eachindex(geoms)
        stage_push_level!(op, geoms[l], bas[l], dms[l])
    end
    define_staged!(op, info, Int32(ncomp))
    return op
end

function MLABecLaplacian(amr::JlAmrCore; info::JlLPInfo=LPInfo(), ncomp::Integer=1)
    nlevs = Int(finest_level(amr)) + 1
    geoms = [geom(amr, Int32(l - 1)) for l in 1:nlevs]
    bas   = [box_array(amr, Int32(l - 1)) for l in 1:nlevs]
    dms   = [dmap(amr, Int32(l - 1)) for l in 1:nlevs]
    return MLABecLaplacian(geoms, bas, dms; info=info, ncomp=ncomp)
end

"""
    MLNodeLaplacian(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
                    info::JlLPInfo=LPInfo(), const_sigma::Real=0.0) -> JlMLNodeLaplacianAllocated
    MLNodeLaplacian(geoms::AbstractVector{<:JlGeometry}, bas::AbstractVector{<:JlBoxArray},
                    dms::AbstractVector{<:JlDistributionMapping};
                    info::JlLPInfo=LPInfo(), const_sigma::Real=0.0) -> JlMLNodeLaplacianAllocated
    MLNodeLaplacian(amr::JlAmrCore; info::JlLPInfo=LPInfo(), const_sigma::Real=0.0) -> JlMLNodeLaplacianAllocated

Construct a nodal variable-coefficient Poisson operator `∇ ⋅ (σ ∇ φ) = rhs` where `φ` and `rhs`
are nodal and `σ` is cell-centered (`ba` must be cell-centered).
"""
function MLNodeLaplacian(geom::JlGeometry, ba::JlBoxArray, dm::JlDistributionMapping;
                         info::JlLPInfo=LPInfo(), const_sigma::Real=0.0)
    return JlMLNodeLaplacian(geom, ba, dm, info, Float64(const_sigma))
end

function MLNodeLaplacian(geoms::AbstractVector{<:JlGeometry},
                         bas::AbstractVector{<:JlBoxArray},
                         dms::AbstractVector{<:JlDistributionMapping};
                         info::JlLPInfo=LPInfo(), const_sigma::Real=0.0)
    length(geoms) == length(bas) == length(dms) ||
        throw(ArgumentError("geoms, bas, and dms must have the same number of levels"))
    op = JlMLNodeLaplacian()
    stage_clear!(op)
    for l in eachindex(geoms)
        stage_push_level!(op, geoms[l], bas[l], dms[l])
    end
    define_staged!(op, info, Float64(const_sigma))
    return op
end

function MLNodeLaplacian(amr::JlAmrCore; info::JlLPInfo=LPInfo(), const_sigma::Real=0.0)
    nlevs = Int(finest_level(amr)) + 1
    geoms = [geom(amr, Int32(l - 1)) for l in 1:nlevs]
    bas   = [box_array(amr, Int32(l - 1)) for l in 1:nlevs]
    dms   = [dmap(amr, Int32(l - 1)) for l in 1:nlevs]
    return MLNodeLaplacian(geoms, bas, dms; info=info, const_sigma=const_sigma)
end

"""
    set_domain_bc!(op, lo_bc::NTuple{3, Union{Integer, LinOpBCType}},
                       hi_bc::NTuple{3, Union{Integer, LinOpBCType}})

Set physical domain boundary conditions on `op` (`MLPoisson`, `MLABecLaplacian`, or `MLNodeLaplacian`).
"""
function set_domain_bc!(op::_MLLinOpUnion,
                        lo_bc::NTuple{3, Union{Integer, LinOpBCType}},
                        hi_bc::NTuple{3, Union{Integer, LinOpBCType}})
    set_domain_bc!(
        op,
        Int32(lo_bc[1]), Int32(lo_bc[2]), Int32(lo_bc[3]),
        Int32(hi_bc[1]), Int32(hi_bc[2]), Int32(hi_bc[3])
    )
    return op
end

"""
    set_domain_bc_loc!(op, lo_loc::NTuple{3, Real}, hi_loc::NTuple{3, Real})

Set boundary condition location offsets from the domain faces.
"""
function set_domain_bc_loc!(op::Union{JlMLPoisson, JlMLABecLaplacian},
                            lo_loc::NTuple{3, Real}, hi_loc::NTuple{3, Real})
    set_domain_bc_loc!(
        op,
        Float64(lo_loc[1]), Float64(lo_loc[2]), Float64(lo_loc[3]),
        Float64(hi_loc[1]), Float64(hi_loc[2]), Float64(hi_loc[3])
    )
    return op
end

"""
    set_level_bc!(op, amrlev::Integer, bcdata::Union{Nothing, JlMultiFab}=nothing;
                  robin_a::Union{Nothing, JlMultiFab}=nothing,
                  robin_b::Union{Nothing, JlMultiFab}=nothing,
                  robin_f::Union{Nothing, JlMultiFab}=nothing)

Set level boundary condition data for AMR level `amrlev` (0-based). Pass `nothing` for homogeneous BCs.
"""
function set_level_bc!(op::Union{JlMLPoisson, JlMLABecLaplacian},
                       amrlev::Integer,
                       bcdata::Union{Nothing, JlMultiFab}=nothing;
                       robin_a::Union{Nothing, JlMultiFab}=nothing,
                       robin_b::Union{Nothing, JlMultiFab}=nothing,
                       robin_f::Union{Nothing, JlMultiFab}=nothing)
    lev = Int32(amrlev)
    if bcdata === nothing
        set_level_bc_homog!(op, lev)
    elseif robin_a !== nothing && robin_b !== nothing && robin_f !== nothing
        set_level_bc_robin!(op, lev, bcdata, robin_a, robin_b, robin_f)
    else
        set_level_bc!(op, lev, bcdata)
    end
    return op
end

"""
    set_coarse_fine_bc!(op, crse::Union{Nothing, JlMultiFab},
                        ref_ratio::Union{Integer, NTuple{3, Integer}};
                        bc_type::Union{Integer, LinOpBCType}=LODirichlet)

Set coarse/fine boundary conditions for a fine-only solve where level 0 of the operator is embedded
inside a coarser AMR level `crse`.
"""
function set_coarse_fine_bc!(op::Union{JlMLPoisson, JlMLABecLaplacian},
                             crse::Union{Nothing, JlMultiFab},
                             ref_ratio::Union{Integer, NTuple{3, Integer}};
                             bc_type::Union{Integer, LinOpBCType}=LODirichlet)
    rr = ref_ratio isa Integer ? (ref_ratio, ref_ratio, ref_ratio) : ref_ratio
    if crse === nothing
        set_coarse_fine_bc_homog!(op, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]), Int32(bc_type))
    else
        set_coarse_fine_bc!(op, crse, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]), Int32(bc_type))
    end
    return op
end

"""
    set_a_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, alpha::Real)
    set_a_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, alpha::JlMultiFab)

Set the α coefficient on level `amrlev` of an `MLABecLaplacian` operator.
"""
set_a_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, alpha::Real) =
    (set_a_coeffs_scalar!(op, Int32(amrlev), Float64(alpha)); op)

"""
    set_b_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, beta::Real)
    set_b_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, beta::NTuple{3, JlMultiFab})

Set the face-centered β coefficients on level `amrlev` of an `MLABecLaplacian` operator.
"""
set_b_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, beta::Real) =
    (set_b_coeffs_scalar!(op, Int32(amrlev), Float64(beta)); op)
set_b_coeffs!(op::JlMLABecLaplacian, amrlev::Integer, beta::NTuple{3, JlMultiFab}) =
    (set_b_coeffs!(op, Int32(amrlev), beta[1], beta[2], beta[3]); op)

"""
    comp_divergence!(op::JlMLNodeLaplacian, rhs::JlMultiFab, vel::JlMultiFab)

Compute the nodal divergence `rhs = ∇ ⋅ vel` from cell-centered `vel`.
"""
comp_divergence!(op::JlMLNodeLaplacian, rhs::JlMultiFab, vel::JlMultiFab) =
    (comp_divergence_1!(op, rhs, vel); nothing)

"""
    update_velocity!(op::JlMLNodeLaplacian, vel::JlMultiFab, sol::JlMultiFab)

Project cell-centered velocity `vel = vel - σ ∇ sol` using nodal solution `sol`.
"""
update_velocity!(op::JlMLNodeLaplacian, vel::JlMultiFab, sol::JlMultiFab) =
    (update_velocity_1!(op, vel, sol); nothing)

"""
    MLMG(op; verbose=0, bottom_verbose=0, max_iter=100, max_fmg_iter=0,
         bottom_solver=BottomDefault) -> JlMLMGAllocated

Construct an AMReX Multi-Level Multi-Grid (`MLMG`) solver for `op` (`MLPoisson`,
`MLABecLaplacian`, or `MLNodeLaplacian`).
"""
function MLMG(op::_MLLinOpUnion;
              verbose::Integer=0,
              bottom_verbose::Integer=0,
              max_iter::Integer=100,
              max_fmg_iter::Integer=0,
              bottom_solver::BottomSolver=BottomDefault)
    solver = JlMLMG(op)
    set_verbose!(solver, Int32(verbose))
    set_bottom_verbose!(solver, Int32(bottom_verbose))
    set_max_iter!(solver, Int32(max_iter))
    set_max_fmg_iter!(solver, Int32(max_fmg_iter))
    set_bottom_solver!(solver, Int32(bottom_solver))
    return solver
end

"""
    solve!(mlmg::JlMLMG, sol::JlMultiFab, rhs::JlMultiFab;
           reltol::Real=1e-11, abstol::Real=0.0) -> Float64
    solve!(mlmg::JlMLMG, sols::AbstractVector{<:JlMultiFab}, rhss::AbstractVector{<:JlMultiFab};
           reltol::Real=1e-11, abstol::Real=0.0) -> Float64

Solve the linear system `L(sol) = rhs` to relative tolerance `reltol` and absolute tolerance `abstol`,
returning the final infinity-norm residual.
"""
function solve!(mlmg::JlMLMG, sol::JlMultiFab, rhs::JlMultiFab;
                reltol::Real=1e-11, abstol::Real=0.0)
    stage_clear!(mlmg)
    stage_push_sol_rhs!(mlmg, sol, rhs)
    return Float64(solve_staged!(mlmg, Float64(reltol), Float64(abstol)))
end

function solve!(mlmg::JlMLMG,
                sols::AbstractVector{<:JlMultiFab},
                rhss::AbstractVector{<:JlMultiFab};
                reltol::Real=1e-11, abstol::Real=0.0)
    length(sols) == length(rhss) ||
        throw(ArgumentError("sols and rhss must have the same number of levels"))
    stage_clear!(mlmg)
    for l in eachindex(sols)
        stage_push_sol_rhs!(mlmg, sols[l], rhss[l])
    end
    return Float64(solve_staged!(mlmg, Float64(reltol), Float64(abstol)))
end

"""
    get_grad_solution!(mlmg::JlMLMG, grad::NTuple{3, JlMultiFab};
                       loc::MLMGLocation=LocFaceCenter)
    get_grad_solution!(mlmg::JlMLMG, grads::AbstractVector{<:NTuple{3, JlMultiFab}};
                       loc::MLMGLocation=LocFaceCenter)

Compute the face-centered gradient `(∂φ/∂x, ∂φ/∂y, ∂φ/∂z)` after `solve!`.
"""
function get_grad_solution!(mlmg::JlMLMG, grad::NTuple{3, JlMultiFab};
                            loc::MLMGLocation=LocFaceCenter)
    stage_clear!(mlmg)
    stage_push_face_vec!(mlmg, grad[1], grad[2], grad[3])
    get_grad_solution_staged!(mlmg, Int32(loc))
    return nothing
end

function get_grad_solution!(mlmg::JlMLMG,
                            grads::AbstractVector{<:NTuple{3, JlMultiFab}};
                            loc::MLMGLocation=LocFaceCenter)
    stage_clear!(mlmg)
    for g in grads
        stage_push_face_vec!(mlmg, g[1], g[2], g[3])
    end
    get_grad_solution_staged!(mlmg, Int32(loc))
    return nothing
end

"""
    get_fluxes!(mlmg::JlMLMG, flux::NTuple{3, JlMultiFab}; loc::MLMGLocation=LocFaceCenter)
    get_fluxes!(mlmg::JlMLMG, fluxes::AbstractVector{<:NTuple{3, JlMultiFab}}; loc::MLMGLocation=LocFaceCenter)
    get_fluxes!(mlmg::JlMLMG, flux::JlMultiFab; loc::MLMGLocation=LocCellCenter)
    get_fluxes!(mlmg::JlMLMG, fluxes::AbstractVector{<:JlMultiFab}; loc::MLMGLocation=LocCellCenter)

Compute the diffusive fluxes (`-β ∇φ`) on faces or cell centers after `solve!`.
"""
function get_fluxes!(mlmg::JlMLMG, flux::NTuple{3, JlMultiFab};
                     loc::MLMGLocation=LocFaceCenter)
    stage_clear!(mlmg)
    stage_push_face_vec!(mlmg, flux[1], flux[2], flux[3])
    get_fluxes_face_staged!(mlmg, Int32(loc))
    return nothing
end

function get_fluxes!(mlmg::JlMLMG,
                     fluxes::AbstractVector{<:NTuple{3, JlMultiFab}};
                     loc::MLMGLocation=LocFaceCenter)
    stage_clear!(mlmg)
    for f in fluxes
        stage_push_face_vec!(mlmg, f[1], f[2], f[3])
    end
    get_fluxes_face_staged!(mlmg, Int32(loc))
    return nothing
end

function get_fluxes!(mlmg::JlMLMG, flux::JlMultiFab;
                     loc::MLMGLocation=LocCellCenter)
    stage_clear!(mlmg)
    stage_push_out!(mlmg, flux)
    get_fluxes_cc_staged!(mlmg, Int32(loc))
    return nothing
end

function get_fluxes!(mlmg::JlMLMG,
                     fluxes::AbstractVector{<:JlMultiFab};
                     loc::MLMGLocation=LocCellCenter)
    stage_clear!(mlmg)
    for f in fluxes
        stage_push_out!(mlmg, f)
    end
    get_fluxes_cc_staged!(mlmg, Int32(loc))
    return nothing
end

"""
    comp_residual!(mlmg::JlMLMG, res::JlMultiFab, sol::JlMultiFab, rhs::JlMultiFab)
    comp_residual!(mlmg::JlMLMG, ress::AbstractVector{<:JlMultiFab},
                   sols::AbstractVector{<:JlMultiFab}, rhss::AbstractVector{<:JlMultiFab})

Compute the residual `res = rhs - L(sol)`.
"""
function comp_residual!(mlmg::JlMLMG, res::JlMultiFab, sol::JlMultiFab, rhs::JlMultiFab)
    stage_clear!(mlmg)
    stage_push_out_sol_rhs!(mlmg, res, sol, rhs)
    comp_residual_staged!(mlmg)
    return nothing
end

function comp_residual!(mlmg::JlMLMG,
                        ress::AbstractVector{<:JlMultiFab},
                        sols::AbstractVector{<:JlMultiFab},
                        rhss::AbstractVector{<:JlMultiFab})
    length(ress) == length(sols) == length(rhss) ||
        throw(ArgumentError("ress, sols, and rhss must have the same number of levels"))
    stage_clear!(mlmg)
    for l in eachindex(ress)
        stage_push_out_sol_rhs!(mlmg, ress[l], sols[l], rhss[l])
    end
    comp_residual_staged!(mlmg)
    return nothing
end

"""
    apply!(mlmg::JlMLMG, out::JlMultiFab, in_mf::JlMultiFab)
    apply!(mlmg::JlMLMG, outs::AbstractVector{<:JlMultiFab}, ins::AbstractVector{<:JlMultiFab})

Evaluate the linear operator `out = L(in_mf)` without solving.
"""
function apply!(mlmg::JlMLMG, out::JlMultiFab, in_mf::JlMultiFab)
    stage_clear!(mlmg)
    stage_push_out_in!(mlmg, out, in_mf)
    apply_staged!(mlmg)
    return nothing
end

function apply!(mlmg::JlMLMG,
                outs::AbstractVector{<:JlMultiFab},
                ins::AbstractVector{<:JlMultiFab})
    length(outs) == length(ins) ||
        throw(ArgumentError("outs and ins must have the same number of levels"))
    stage_clear!(mlmg)
    for l in eachindex(outs)
        stage_push_out_in!(mlmg, outs[l], ins[l])
    end
    apply_staged!(mlmg)
    return nothing
end

