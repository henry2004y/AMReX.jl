# ============================================================================
# Flux Registers (YAFluxRegister & EdgeFluxRegister) and Staggered Averaging
# ============================================================================

"""
    YAFluxRegister(fba::JlBoxArray, cba::JlBoxArray,
                   fdm::JlDistributionMapping, cdm::JlDistributionMapping,
                   fgeom::JlGeometry, cgeom::JlGeometry,
                   ref_ratio::NTuple{3, Integer},
                   fine_lev::Integer, nvar::Integer) -> JlYAFluxRegisterAllocated

Construct an AMReX `YAFluxRegister` ("Yet-Another Flux Register") for conservative refluxing
of face-centered fluxes across coarse/fine level boundaries.
"""
function YAFluxRegister(fba::JlBoxArray, cba::JlBoxArray,
                        fdm::JlDistributionMapping, cdm::JlDistributionMapping,
                        fgeom::JlGeometry, cgeom::JlGeometry,
                        ref_ratio::NTuple{3, Integer},
                        fine_lev::Integer, nvar::Integer)
    return JlYAFluxRegister(
        fba, cba, fdm, cdm, fgeom, cgeom,
        Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]),
        Int32(fine_lev), Int32(nvar)
    )
end

"""
    YAFluxRegister(amr::JlAmrCore, fine_lev::Integer, nvar::Integer) -> JlYAFluxRegisterAllocated

Convenience constructor building a `YAFluxRegister` between `fine_lev` and `fine_lev - 1` of `amr`.
"""
function YAFluxRegister(amr::JlAmrCore, fine_lev::Integer, nvar::Integer)
    flev = Int32(fine_lev)
    clev = Int32(fine_lev - 1)
    return YAFluxRegister(
        box_array(amr, flev), box_array(amr, clev),
        dmap(amr, flev), dmap(amr, clev),
        geom(amr, flev), geom(amr, clev),
        ref_ratio_tuple(amr, clev),
        flev, nvar
    )
end

"""
    crse_add!(fr::JlYAFluxRegister, mfi::JlMFIter,
              flux::NTuple{3, JlMultiFab}, dx::NTuple{3, Real}, dt::Real;
              srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=1)

Accumulate coarse face-centered fluxes `(fx, fy, fz)` for the tile at `mfi` into `fr`.
"""
function crse_add!(fr::JlYAFluxRegister, mfi::JlMFIter,
                   flux::NTuple{3, JlMultiFab}, dx::NTuple{3, Real}, dt::Real;
                   srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=1)
    crse_add!(
        fr, mfi, flux[1], flux[2], flux[3],
        Float64(dx[1]), Float64(dx[2]), Float64(dx[3]), Float64(dt),
        Int32(srccomp), Int32(destcomp), Int32(numcomp)
    )
    return nothing
end

"""
    fine_add!(fr::JlYAFluxRegister, mfi::JlMFIter,
              flux::NTuple{3, JlMultiFab}, dx::NTuple{3, Real}, dt::Real;
              srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=1)

Accumulate fine face-centered fluxes `(fx, fy, fz)` for the tile at `mfi` into `fr`.
"""
function fine_add!(fr::JlYAFluxRegister, mfi::JlMFIter,
                   flux::NTuple{3, JlMultiFab}, dx::NTuple{3, Real}, dt::Real;
                   srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=1)
    fine_add!(
        fr, mfi, flux[1], flux[2], flux[3],
        Float64(dx[1]), Float64(dx[2]), Float64(dx[3]), Float64(dt),
        Int32(srccomp), Int32(destcomp), Int32(numcomp)
    )
    return nothing
end

"""
    reflux!(fr::JlYAFluxRegister, state::JlMultiFab;
            srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=n_comp(state))

Apply the accumulated coarse/fine flux mismatch in `fr` to the coarse cell-centered `state`.
"""
function reflux!(fr::JlYAFluxRegister, state::JlMultiFab;
                 srccomp::Integer=0, destcomp::Integer=0, numcomp::Integer=n_comp(state))
    reflux!(fr, state, Int32(srccomp), Int32(destcomp), Int32(numcomp))
    return nothing
end

"""
    EdgeFluxRegister(fba::JlBoxArray, cba::JlBoxArray,
                     fdm::JlDistributionMapping, cdm::JlDistributionMapping,
                     fgeom::JlGeometry, cgeom::JlGeometry,
                     nvar::Integer=1) -> JlEdgeFluxRegisterAllocated

Construct an AMReX `EdgeFluxRegister` for constrained-transport (`∂B/∂t + ∇ × E = 0`)
refluxing of edge-centered electric fields `E` onto face-centered magnetic fields `B`.
"""
function EdgeFluxRegister(fba::JlBoxArray, cba::JlBoxArray,
                          fdm::JlDistributionMapping, cdm::JlDistributionMapping,
                          fgeom::JlGeometry, cgeom::JlGeometry,
                          nvar::Integer=1)
    return JlEdgeFluxRegister(fba, cba, fdm, cdm, fgeom, cgeom, Int32(nvar))
end

"""
    EdgeFluxRegister(amr::JlAmrCore, fine_lev::Integer, nvar::Integer=1) -> JlEdgeFluxRegisterAllocated

Convenience constructor building an `EdgeFluxRegister` between `fine_lev` and `fine_lev - 1` of `amr`.
"""
function EdgeFluxRegister(amr::JlAmrCore, fine_lev::Integer, nvar::Integer=1)
    flev = Int32(fine_lev)
    clev = Int32(fine_lev - 1)
    return EdgeFluxRegister(
        box_array(amr, flev), box_array(amr, clev),
        dmap(amr, flev), dmap(amr, clev),
        geom(amr, flev), geom(amr, clev),
        nvar
    )
end

"""
    crse_add!(efr::JlEdgeFluxRegister, mfi::JlMFIter,
              E_crse::NTuple{3, JlMultiFab}, dt_crse::Real)

Accumulate coarse edge-centered electric fields `(Ex, Ey, Ez)` for tile `mfi` into `efr`.
"""
function crse_add!(efr::JlEdgeFluxRegister, mfi::JlMFIter,
                   E_crse::NTuple{3, JlMultiFab}, dt_crse::Real)
    crse_add!(efr, mfi, E_crse[1], E_crse[2], E_crse[3], Float64(dt_crse))
    return nothing
end

"""
    fine_add!(efr::JlEdgeFluxRegister, mfi::JlMFIter,
              E_fine::NTuple{3, JlMultiFab}, dt_fine::Real)

Accumulate fine edge-centered electric fields `(Ex, Ey, Ez)` for tile `mfi` into `efr`.
"""
function fine_add!(efr::JlEdgeFluxRegister, mfi::JlMFIter,
                   E_fine::NTuple{3, JlMultiFab}, dt_fine::Real)
    fine_add!(efr, mfi, E_fine[1], E_fine[2], E_fine[3], Float64(dt_fine))
    return nothing
end

"""
    reflux!(efr::JlEdgeFluxRegister, B_crse::NTuple{3, JlMultiFab})

Apply the accumulated edge-centered fluxes in `efr` to coarse face-centered magnetic fields `(Bx, By, Bz)`.
"""
function reflux!(efr::JlEdgeFluxRegister, B_crse::NTuple{3, JlMultiFab})
    reflux!(efr, B_crse[1], B_crse[2], B_crse[3])
    return nothing
end

"""
    average_down_faces!(fine::NTuple{3, JlMultiFab}, crse::NTuple{3, JlMultiFab},
                        ref_ratio::NTuple{3, Integer}, cgeom::JlGeometry)

Average fine face-centered `MultiFab`s `(fx_f, fy_f, fz_f)` onto coarse face-centered
`MultiFab`s `(fx_c, fy_c, fz_c)`.
"""
function average_down_faces!(fine::NTuple{3, JlMultiFab}, crse::NTuple{3, JlMultiFab},
                             ref_ratio::NTuple{3, Integer}, cgeom::JlGeometry)
    average_down_faces!(
        fine[1], fine[2], fine[3],
        crse[1], crse[2], crse[3],
        Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]),
        cgeom
    )
    return nothing
end

"""
    average_down_faces!(fine::JlMultiFab, crse::JlMultiFab,
                        ref_ratio::NTuple{3, Integer}, cgeom::JlGeometry)

Average a single face-centered `MultiFab` from `fine` to `crse`.
"""
function average_down_faces!(fine::JlMultiFab, crse::JlMultiFab,
                             ref_ratio::NTuple{3, Integer}, cgeom::JlGeometry)
    average_down_face!(
        fine, crse,
        Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]),
        cgeom
    )
    return nothing
end

"""
    average_down_edges!(fine::NTuple{3, JlMultiFab}, crse::NTuple{3, JlMultiFab},
                        ref_ratio::NTuple{3, Integer}, ngcrse::Integer=0)

Average fine edge-centered `MultiFab`s `(ex_f, ey_f, ez_f)` onto coarse edge-centered
`MultiFab`s `(ex_c, ey_c, ez_c)`.
"""
function average_down_edges!(fine::NTuple{3, JlMultiFab}, crse::NTuple{3, JlMultiFab},
                             ref_ratio::NTuple{3, Integer}, ngcrse::Integer=0)
    average_down_edges!(
        fine[1], fine[2], fine[3],
        crse[1], crse[2], crse[3],
        Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]),
        Int32(ngcrse)
    )
    return nothing
end

