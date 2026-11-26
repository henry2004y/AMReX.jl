# ============================================================================
# AmrCore, Dynamic Error Tagging, Physical BCs & Multi-Level AMR Utilities
# ============================================================================

"""
    InterpType

AMReX coarse-to-fine spatial interpolation schemes:
- `CellConservativeLinear = 0`: Conservative linear cell-centered interpolation (`amrex::cell_cons_interp`).
- `NodeBilinear = 1`: Bilinear/trilinear nodal interpolation (`amrex::node_bilinear_interp`).
- `PiecewiseConstant = 2`: Piecewise-constant cell-centered injection (`amrex::pc_interp`).
- `CellQuadratic = 3`: Quadratic cell-centered interpolation (`amrex::quadratic_interp`).
"""
@enum InterpType::Int32 begin
    CellConservativeLinear = 0
    NodeBilinear = 1
    PiecewiseConstant = 2
    CellQuadratic = 3
end

"""
    MathBCType

AMReX mathematical boundary condition types (`amrex::BCType`):
- `BCBogus = -666`: Uninitialized / bogus boundary condition.
- `BCReflectOdd = -1`: Odd reflection (`F(-n) = -F(n)`).
- `BCIntDir = 0`: Interior / periodic boundary (`amrex::BCType::int_dir`).
- `BCReflectEven = 1`: Even reflection / symmetry (`F(-n) = F(n)`).
- `BCFOExtrap = 2`: First-order (zero-gradient) extrapolation (`amrex::BCType::foextrap`).
- `BCExtDir = 3`: External Dirichlet boundary (`amrex::BCType::ext_dir`).
- `BCHOExtrap = 4`: Higher-order extrapolation to face (`amrex::BCType::hoextrap`).
- `BCHOExtrapCC = 5`: Linear extrapolation to cell center (`amrex::BCType::hoextrapcc`).
- `BCExtDirCC = 6`: External Dirichlet specified at cell center (`amrex::BCType::ext_dir_cc`).
"""
@enum MathBCType::Int32 begin
    BCBogus = -666
    BCReflectOdd = -1
    BCIntDir = 0
    BCReflectEven = 1
    BCFOExtrap = 2
    BCExtDir = 3
    BCHOExtrap = 4
    BCHOExtrapCC = 5
    BCExtDirCC = 6
end

"""
    BCRec(lox, loy, loz, hix, hiy, hiz)
    BCRec(; lo=(BCIntDir, BCIntDir, BCIntDir), hi=(BCIntDir, BCIntDir, BCIntDir))

Construct an AMReX `JlBCRec` mathematical boundary condition record for 3D domain faces.
"""
function BCRec(
        lox::Union{Integer, MathBCType},
        loy::Union{Integer, MathBCType},
        loz::Union{Integer, MathBCType},
        hix::Union{Integer, MathBCType},
        hiy::Union{Integer, MathBCType},
        hiz::Union{Integer, MathBCType}
    )
    return JlBCRec(
        Int32(lox), Int32(loy), Int32(loz),
        Int32(hix), Int32(hiy), Int32(hiz)
    )
end

function BCRec(;
        lo::NTuple{3, Union{Integer, MathBCType}} = (BCIntDir, BCIntDir, BCIntDir),
        hi::NTuple{3, Union{Integer, MathBCType}} = (BCIntDir, BCIntDir, BCIntDir)
    )
    return BCRec(lo[1], lo[2], lo[3], hi[1], hi[2], hi[3])
end

"""
    fill_domain_boundary!(mf::JlMultiFab, geom::JlGeometry, bc::JlBCRec;
                          time::Real=0.0,
                          val_lo::Union{Nothing, NTuple{3, Real}}=nothing,
                          val_hi::Union{Nothing, NTuple{3, Real}}=nothing)

Fill non-periodic physical domain ghost cells of `mf` according to `bc`.
Supports `BCFOExtrap`, `BCHOExtrap`, `BCHOExtrapCC`, `BCReflectEven`, `BCReflectOdd`,
and external Dirichlet (`BCExtDir`, `BCExtDirCC`) when `val_lo`/`val_hi` are provided.
"""
function fill_domain_boundary!(
        mf::JlMultiFab, geom::JlGeometry, bc::JlBCRec;
        time::Real = 0.0,
        val_lo::Union{Nothing, NTuple{3, Real}} = nothing,
        val_hi::Union{Nothing, NTuple{3, Real}} = nothing
    )
    if val_lo === nothing && val_hi === nothing
        fill_domain_boundary_val!(
            mf, geom, bc, Float64(time),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
    else
        vlo = something(val_lo, (0.0, 0.0, 0.0))
        vhi = something(val_hi, (0.0, 0.0, 0.0))
        fill_domain_boundary_val!(
            mf, geom, bc, Float64(time),
            Float64(vlo[1]), Float64(vlo[2]), Float64(vlo[3]),
            Float64(vhi[1]), Float64(vhi[2]), Float64(vhi[3])
        )
    end
    return mf
end

"""
    TagVal

AMReX cell tagging states (`amrex::TagBox::TagVal`):
- `TagClear = 0`: Do not refine cell (`amrex::TagBox::CLEAR`).
- `TagBuf = 1`: Buffer cell (`amrex::TagBox::BUF`).
- `TagSet = 2`: Tag cell for mesh refinement (`amrex::TagBox::SET`).
"""
@enum TagVal::Int8 begin
    TagClear = 0
    TagBuf = 1
    TagSet = 2
end

"""
    TagBoxArrayHandle

Non-owning `isbits` handle wrapping AMReX's native `amrex::TagBoxArray*` inside `ErrorEst`
callbacks, avoiding extra `TagBoxArray` allocations and scalar triple-loop copies.
"""
struct TagBoxArrayHandle
    ptr::Ptr{Cvoid}
end

"""
    TagBoxArray(ba::JlBoxArray, dm::JlDistributionMapping, nghost::Integer=0) -> JlTagBoxArrayAllocated

Construct a cell-centered `TagBoxArray` with the given `BoxArray` and `DistributionMapping`.
"""
TagBoxArray(ba::JlBoxArray, dm::JlDistributionMapping, nghost::Integer = 0) =
    JlTagBoxArray(ba, dm, Int32(nghost))

@inline function _acquire_mfiter(tb::JlTagBoxArray, tiling::Bool)::MFIterHandle
    return @ccall $(_FN_MFITER_ACQUIRE_TB[])(tb.cpp_object::Ptr{Cvoid}, UInt8(tiling)::UInt8)::MFIterHandle
end

@inline function _acquire_mfiter(tb::TagBoxArrayHandle, tiling::Bool)::MFIterHandle
    return @ccall $(_FN_MFITER_ACQUIRE_TB_RAW[])(tb.ptr::Ptr{Cvoid}, UInt8(tiling)::UInt8)::MFIterHandle
end

"""
    tag_array4(tb::Union{JlTagBoxArray, TagBoxArrayHandle}, mfi::Union{JlMFIter, MFIterHandle}) -> Array4View{Int8}

Return a zero-allocation, zero-copy 4D `Array4View{Int8}` view `(i, j, k, 1)` over the
local `TagBox` at `mfi`. Assign `Int8(TagSet)` (`2`) to mark cells for refinement and
`Int8(TagClear)` (`0`) to clear.
"""
@inline function tag_array4(tb::JlTagBoxArray, mfi::Union{JlMFIter, MFIterHandle})
    fv = @ccall $(_FN_TFAB_VIEW[])(tb.cpp_object::Ptr{Cvoid}, _raw_mfi(mfi)::Ptr{Cvoid})::_CFabView
    return _cfab_to_view(Int8, fv)
end

@inline function tag_array4(tb::TagBoxArrayHandle, mfi::Union{JlMFIter, MFIterHandle})
    fv = @ccall $(_FN_TFAB_VIEW_RAW[])(tb.ptr::Ptr{Cvoid}, _raw_mfi(mfi)::Ptr{Cvoid})::_CFabView
    return _cfab_to_view(Int8, fv)
end

"""
    each_fab(f::Function, tb::Union{JlTagBoxArray, TagBoxArrayHandle}; tiling::Bool=false)

Iterate over all local `TagBox`es of `tb` with zero heap allocations, calling
`f(mfi, valid_bx, tag_arr)` where `tag_arr` is the zero-copy `Array4View{Int8}` from `tag_array4(tb, mfi)`.
"""
function each_fab(f::F, tb::Union{JlTagBoxArray, TagBoxArrayHandle}; tiling::Bool = false) where {F}
    mfi = _acquire_mfiter(tb, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = tag_array4(tb, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        _release_mfiter(mfi)
    end
    return nothing
end

const _AMR_ERROR_CALLBACKS = Dict{UInt64, Function}()
const _ERROR_EST_TRAMPOLINE = Ref{Ptr{Cvoid}}(C_NULL)

function _amrex_error_est_trampoline(amr_key::UInt64, lev::Cint, tags_ptr::UInt64, time::Cdouble)::Cvoid
    cb = get(_AMR_ERROR_CALLBACKS, amr_key, nothing)
    if cb !== nothing
        cb(Int(lev), TagBoxArrayHandle(Ptr{Cvoid}(tags_ptr)), Float64(time))
    end
    return nothing
end

"""
    on_error_est!(cb::Union{Nothing, Function}, amr::JlAmrCore)

Register a dynamic Julia error-estimation callback `cb(lev::Int, tags, time::Float64)`
invoked directly by AMReX's C++ `ErrorEst` during `init_from_scratch!` and `regrid_levels!`.
Pass `nothing` to clear.
"""
function on_error_est!(cb::Union{Nothing, Function}, amr::JlAmrCore)
    key = UInt64(core_key(amr))
    if cb === nothing
        delete!(_AMR_ERROR_CALLBACKS, key)
        set_error_cb!(amr, UInt64(0))
    else
        _AMR_ERROR_CALLBACKS[key] = cb
        set_error_cb!(amr, UInt64(_ERROR_EST_TRAMPOLINE[]))
    end
    return amr
end

"""
    init_from_scratch!(amr::JlAmrCore, time::Real=0.0)

Initialize the AMR hierarchy from scratch at `time`, invoking any registered `on_error_est!`
callback and/or refine boxes and tag masks on each level.
"""
function init_from_scratch!(amr::JlAmrCore, time::Real = 0.0)
    _init_from_scratch!(amr, Float64(time))
    return nothing
end

"""
    regrid_levels!(amr::JlAmrCore, base_lev::Integer, time::Real=0.0)

Rebuild all AMR levels finer than `base_lev` at `time`, invoking any registered `on_error_est!`
callback and/or refine boxes and tag masks on levels `>= base_lev`.
"""
function regrid_levels!(amr::JlAmrCore, base_lev::Integer, time::Real = 0.0)
    _regrid_levels!(amr, Int32(base_lev), Float64(time))
    return nothing
end

"""
    tag_cells_by_value!(amr::JlAmrCore, lev::Integer, mf::JlMultiFab, threshold::Real;
                        comp::Integer=1, abs_val::Bool=false) -> JliMultiFab

Build and register a level-`lev` `iMultiFab` refinement mask on `amr` tagging all cells where
`mf[i, j, k, comp] >= threshold` (or `abs(mf[i, j, k, comp]) >= threshold` if `abs_val=true`).
"""
function tag_cells_by_value!(
        amr::JlAmrCore, lev::Integer, mf::JlMultiFab, threshold::Real;
        comp::Integer = 1, abs_val::Bool = false
    )
    mask = iMultiFab(box_array(mf), dmap(mf), 1, 0)
    set_val!(mask, Int32(0))
    thresh = Float64(threshold)
    c = Int(comp)
    each_fab(mf) do mfi, vbx, arr
        m_arr = iarray4(mask, mfi)
        rx, ry, rz = bounds(vbx)
        @inbounds for k in rz, j in ry, i in rx
            v = abs_val ? abs(arr[i, j, k, c]) : arr[i, j, k, c]
            if v >= thresh
                m_arr[i, j, k, 1] = Int32(1)
            end
        end
    end
    set_tag_mask!(amr, Int32(lev), mask)
    return mask
end

"""
    tag_cells_by_grad!(amr::JlAmrCore, lev::Integer, mf::JlMultiFab, grad_threshold::Real;
                       comp::Integer=1) -> JliMultiFab

Build and register a level-`lev` `iMultiFab` refinement mask on `amr` tagging cells where
the maximum neighbor difference `max(|Δ_x|, |Δ_y|, |Δ_z|) >= grad_threshold`.
"""
function tag_cells_by_grad!(
        amr::JlAmrCore, lev::Integer, mf::JlMultiFab, grad_threshold::Real;
        comp::Integer = 1
    )
    mask = iMultiFab(box_array(mf), dmap(mf), 1, 0)
    set_val!(mask, Int32(0))
    thresh = Float64(grad_threshold)
    c = Int(comp)
    each_fab(mf) do mfi, vbx, arr
        m_arr = iarray4(mask, mfi)
        rx, ry, rz = bounds(vbx)
        frx, fry, frz = axes(arr)[1:3]
        @inbounds for k in rz, j in ry, i in rx
            vc = arr[i, j, k, c]
            ip = min(i + 1, last(frx))
            im = max(i - 1, first(frx))
            jp = min(j + 1, last(fry))
            jm = max(j - 1, first(fry))
            kp = min(k + 1, last(frz))
            km = max(k - 1, first(frz))
            gx = max(abs(arr[ip, j, k, c] - vc), abs(vc - arr[im, j, k, c]))
            gy = max(abs(arr[i, jp, k, c] - vc), abs(vc - arr[i, jm, k, c]))
            gz = max(abs(arr[i, j, kp, c] - vc), abs(vc - arr[i, j, km, c]))
            if max(gx, gy, gz) >= thresh
                m_arr[i, j, k, 1] = Int32(1)
            end
        end
    end
    set_tag_mask!(amr, Int32(lev), mask)
    return mask
end

"""
    interp_from_coarse_to_fine!(coarse::JlMultiFab, fine::JlMultiFab,
                                icomp::Integer, ncomp::Integer,
                                ref_ratio::NTuple{3, Integer},
                                cgeom::JlGeometry, fgeom::JlGeometry,
                                interp::InterpType=CellConservativeLinear,
                                nghost::Integer=0;
                                bc::Union{Nothing, JlBCRec}=nothing,
                                val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                                val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))

Interpolate `ncomp` components starting at 0-based `icomp` from `coarse` to `fine`, optionally
applying physical boundary condition `bc`.
"""
function interp_from_coarse_to_fine!(
        coarse::JlMultiFab, fine::JlMultiFab,
        icomp::Integer, ncomp::Integer,
        ref_ratio::NTuple{3, Integer},
        cgeom::JlGeometry, fgeom::JlGeometry,
        interp::InterpType = CellConservativeLinear,
        nghost::Integer = 0;
        bc::Union{Nothing, JlBCRec} = nothing,
        val_lo::NTuple{3, Real} = (0.0, 0.0, 0.0),
        val_hi::NTuple{3, Real} = (0.0, 0.0, 0.0)
    )
    rx, ry, rz = Int32.(ref_ratio)
    if bc === nothing
        interp_from_coarse_to_fine!(
            coarse, fine, Int32(icomp), Int32(ncomp),
            rx, ry, rz, cgeom, fgeom, Int32(interp), Int32(nghost)
        )
    else
        interp_from_coarse_to_fine_bc!(
            coarse, fine, Int32(icomp), Int32(ncomp),
            rx, ry, rz, cgeom, fgeom, Int32(interp), Int32(nghost),
            bc,
            Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
            Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3])
        )
    end
    return fine
end

"""
    fill_patch_single_level!(dst::JlMultiFab, time::Real, src::JlMultiFab,
                             scomp::Integer, dcomp::Integer, ncomp::Integer,
                             geom::JlGeometry;
                             bc::Union{Nothing, JlBCRec}=nothing,
                             val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                             val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))

Fill `dst` (including ghost cells) from single-level `src` at `time`, optionally applying physical `bc`.
"""
function fill_patch_single_level!(
        dst::JlMultiFab, time::Real, src::JlMultiFab,
        scomp::Integer, dcomp::Integer, ncomp::Integer,
        geom::JlGeometry;
        bc::Union{Nothing, JlBCRec} = nothing,
        val_lo::NTuple{3, Real} = (0.0, 0.0, 0.0),
        val_hi::NTuple{3, Real} = (0.0, 0.0, 0.0)
    )
    if bc === nothing
        fill_patch_single_level!(dst, Float64(time), src, Int32(scomp), Int32(dcomp), Int32(ncomp), geom)
    else
        fill_patch_single_level_bc!(
            dst, Float64(time), src, Int32(scomp), Int32(dcomp), Int32(ncomp),
            geom, bc,
            Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
            Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3])
        )
    end
    return dst
end

"""
    fill_patch_two_levels!(dst::JlMultiFab, time::Real,
                           cmf::JlMultiFab, fmf::JlMultiFab,
                           scomp::Integer, dcomp::Integer, ncomp::Integer,
                           cgeom::JlGeometry, fgeom::JlGeometry,
                           ref_ratio::NTuple{3, Integer},
                           interp::InterpType=CellConservativeLinear;
                           bc::Union{Nothing, JlBCRec}=nothing,
                           val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                           val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))

Fill `dst` on the fine level by interpolating from `cmf` at coarse-fine boundaries and copying from `fmf`,
optionally applying physical `bc`.
"""
function fill_patch_two_levels!(
        dst::JlMultiFab, time::Real,
        cmf::JlMultiFab, fmf::JlMultiFab,
        scomp::Integer, dcomp::Integer, ncomp::Integer,
        cgeom::JlGeometry, fgeom::JlGeometry,
        ref_ratio::NTuple{3, Integer},
        interp::InterpType = CellConservativeLinear;
        bc::Union{Nothing, JlBCRec} = nothing,
        val_lo::NTuple{3, Real} = (0.0, 0.0, 0.0),
        val_hi::NTuple{3, Real} = (0.0, 0.0, 0.0)
    )
    rx, ry, rz = Int32.(ref_ratio)
    if bc === nothing
        fill_patch_two_levels!(
            dst, Float64(time), cmf, fmf,
            Int32(scomp), Int32(dcomp), Int32(ncomp),
            cgeom, fgeom, rx, ry, rz, Int32(interp)
        )
    else
        fill_patch_two_levels_bc!(
            dst, Float64(time), cmf, fmf,
            Int32(scomp), Int32(dcomp), Int32(ncomp),
            cgeom, fgeom, rx, ry, rz, Int32(interp), bc,
            Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
            Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3])
        )
    end
    return dst
end

"""
    fill_patch_two_levels!(dst::JlMultiFab, time::Real,
                           cmf_old::JlMultiFab, t_old::Real,
                           cmf_new::JlMultiFab, t_new::Real,
                           fmf::JlMultiFab, ft::Real,
                           scomp::Integer, dcomp::Integer, ncomp::Integer,
                           cgeom::JlGeometry, fgeom::JlGeometry,
                           ref_ratio::NTuple{3, Integer},
                           interp::InterpType=CellConservativeLinear;
                           bc::Union{Nothing, JlBCRec}=nothing,
                           val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                           val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))

Fill `dst` on the fine level at `time ∈ [t_old, t_new]` by linearly interpolating in time
between `cmf_old` (at `t_old`) and `cmf_new` (at `t_new`) on coarse–fine boundaries and copying
from `fmf` (at `ft`), enabling Berger–Colella temporal subcycling.
"""
function fill_patch_two_levels!(
        dst::JlMultiFab, time::Real,
        cmf_old::JlMultiFab, t_old::Real,
        cmf_new::JlMultiFab, t_new::Real,
        fmf::JlMultiFab, ft::Real,
        scomp::Integer, dcomp::Integer, ncomp::Integer,
        cgeom::JlGeometry, fgeom::JlGeometry,
        ref_ratio::NTuple{3, Integer},
        interp::InterpType = CellConservativeLinear;
        bc::Union{Nothing, JlBCRec} = nothing,
        val_lo::NTuple{3, Real} = (0.0, 0.0, 0.0),
        val_hi::NTuple{3, Real} = (0.0, 0.0, 0.0)
    )
    rx, ry, rz = Int32.(ref_ratio)
    if bc === nothing
        fill_patch_two_levels_time!(
            dst, Float64(time),
            cmf_old, Float64(t_old), cmf_new, Float64(t_new),
            fmf, Float64(ft),
            Int32(scomp), Int32(dcomp), Int32(ncomp),
            cgeom, fgeom, rx, ry, rz, Int32(interp)
        )
    else
        fill_patch_two_levels_time_bc!(
            dst, Float64(time),
            cmf_old, Float64(t_old), cmf_new, Float64(t_new),
            fmf, Float64(ft),
            Int32(scomp), Int32(dcomp), Int32(ncomp),
            cgeom, fgeom, rx, ry, rz, Int32(interp), bc,
            Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
            Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3])
        )
    end
    return dst
end

"""
    make_fine_mask(cba::JlBoxArray, cdm::JlDistributionMapping,
                   fba::JlBoxArray, ref_ratio::NTuple{3, Integer},
                   cgeom::JlGeometry;
                   crse_val::Integer=0, fine_val::Integer=1,
                   nghost::Union{Integer, NTuple{3, Integer}}=0) -> JliMultiFabAllocated
    make_fine_mask(amr::JlAmrCore, crse_lev::Integer;
                   crse_val::Integer=0, fine_val::Integer=1,
                   nghost::Union{Integer, NTuple{3, Integer}}=0) -> JliMultiFabAllocated

Build an MPI-safe `iMultiFab` on the coarse level marking cells covered by `fba` with `fine_val`
and uncovered coarse cells with `crse_val` (via `amrex::makeFineMask`).
"""
function make_fine_mask(
        cba::JlBoxArray, cdm::JlDistributionMapping,
        fba::JlBoxArray, ref_ratio::NTuple{3, Integer},
        cgeom::JlGeometry;
        crse_val::Integer = 0, fine_val::Integer = 1,
        nghost::Union{Integer, NTuple{3, Integer}} = 0
    )
    ng = nghost isa Integer ? (nghost, nghost, nghost) : nghost
    out = JliMultiFab()
    define_fine_mask!(
        out, cba, cdm,
        Int32(ng[1]), Int32(ng[2]), Int32(ng[3]),
        fba, Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]),
        cgeom, Int32(crse_val), Int32(fine_val)
    )
    return out
end

function make_fine_mask(
        amr::JlAmrCore, crse_lev::Integer;
        crse_val::Integer = 0, fine_val::Integer = 1,
        nghost::Union{Integer, NTuple{3, Integer}} = 0
    )
    clev = Int32(crse_lev)
    flev = Int32(crse_lev + 1)
    return make_fine_mask(
        box_array(amr, clev), dmap(amr, clev),
        box_array(amr, flev), ref_ratio_tuple(amr, crse_lev),
        geom(amr, clev);
        crse_val = crse_val, fine_val = fine_val, nghost = nghost
    )
end

"""
    coarsened_multifab(fba::JlBoxArray, fdm::JlDistributionMapping,
                       ref_ratio::NTuple{3, Integer},
                       ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}}=0) -> JlMultiFabAllocated
    coarsened_multifab(amr::JlAmrCore, fine_lev::Integer,
                       ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}}=0) -> JlMultiFabAllocated

Allocate a `MultiFab` on `coarsen(fba, ref_ratio)` sharing `fdm` (the fine level's
`DistributionMapping`). Pair with `parallel_copy!` for zero-Dict, MPI-safe custom
coarse–fine prolongation, mortar filling, and conservative restriction.
"""
function coarsened_multifab(
        fba::JlBoxArray, fdm::JlDistributionMapping,
        ref_ratio::NTuple{3, Integer},
        ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}} = 0
    )
    cba_on_fine = coarsen(fba, ref_ratio)
    return MultiFab(cba_on_fine, fdm, ncomp, nghost)
end

function coarsened_multifab(
        amr::JlAmrCore, fine_lev::Integer,
        ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}} = 0
    )
    flev = Int32(fine_lev)
    clev = Int(fine_lev) - 1
    return coarsened_multifab(
        box_array(amr, flev), dmap(amr, flev),
        ref_ratio_tuple(amr, clev), ncomp, nghost
    )
end

"""
    AmrCore(rbox::JlRealBox, ncells::NTuple{3, Integer};
            max_level::Integer=0,
            coord::Integer=0,
            is_periodic::NTuple{3, Bool}=(true, true, true),
            max_grid_size::NTuple{3, Integer}=(32, 32, 32),
            blocking_factor::NTuple{3, Integer}=(4, 4, 4),
            grid_eff::Real=0.7,
            ref_ratio::Union{Integer, NTuple{3, Integer}}=(2, 2, 2),
            n_error_buf::Integer=1)

Construct an AMReX `JlAmrCore` multi-level mesh hierarchy manager.
Automatically sets the y/z blocking factor and refinement ratio to 1 when `ncells[2] == 1`
or `ncells[3] == 1` (Fake-1D/2D).
"""
function AmrCore(
        rbox::JlRealBox, ncells::NTuple{3, Integer};
        max_level::Integer = 0,
        coord::Integer = 0,
        is_periodic::NTuple{3, Bool} = (true, true, true),
        max_grid_size::NTuple{3, Integer} = (32, 32, 32),
        blocking_factor::NTuple{3, Integer} = (4, 4, 4),
        grid_eff::Real = 0.7,
        ref_ratio::Union{Integer, NTuple{3, Integer}} = (2, 2, 2),
        n_error_buf::Integer = 1
    )
    nx, ny, nz = Int32.(ncells)
    mgx, mgy, mgz = Int32.(max_grid_size)
    bx = Int32(min(blocking_factor[1], ncells[1]))
    by = Int32(ncells[2] == 1 ? 1 : min(blocking_factor[2], ncells[2]))
    bz = Int32(ncells[3] == 1 ? 1 : min(blocking_factor[3], ncells[3]))
    rr = ref_ratio isa Integer ? (ref_ratio, ref_ratio, ref_ratio) : ref_ratio
    rrx = Int32(rr[1])
    rry = Int32(ncells[2] == 1 ? 1 : rr[2])
    rrz = Int32(ncells[3] == 1 ? 1 : rr[3])
    return JlAmrCore(
        rbox, Int32(max_level),
        nx, ny, nz, Int32(coord),
        is_periodic[1], is_periodic[2], is_periodic[3],
        mgx, mgy, mgz, bx, by, bz, Float64(grid_eff),
        rrx, rry, rrz, Int32(n_error_buf)
    )
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

"""
    add_tag_box!(amr::JlAmrCore, xlo, ylo, zlo, xhi, yhi, zhi; lev::Integer=0)

Register a physical bounding box `[xlo, xhi] × [ylo, yhi] × [zlo, zhi]` for mesh refinement tagging at level `lev`.
"""
@inline add_tag_box!(amr::JlAmrCore, xlo::Real, ylo::Real, zlo::Real, xhi::Real, yhi::Real, zhi::Real; lev::Integer = 0) =
    add_refine_box!(amr, Int32(lev), Float64(xlo), Float64(ylo), Float64(zlo), Float64(xhi), Float64(yhi), Float64(zhi))

"""
    geometry(amr::JlAmrCore, lev::Integer=0) -> JlGeometry

Return the `JlGeometry` object at AMR level `lev`.
"""
@inline geometry(amr::JlAmrCore, lev::Integer = 0) = geom(amr, Int32(lev))

"""
    dist_map(amr::JlAmrCore, lev::Integer=0) -> JlDistributionMapping

Return the `JlDistributionMapping` at AMR level `lev`.
"""
@inline dist_map(amr::JlAmrCore, lev::Integer = 0) = dmap(amr, Int32(lev))

"""
    mpi_rank() -> Int

Return the 0-based MPI rank of the calling process.
"""
@inline mpi_rank() = Int(parallel_myproc())

"""
    grid_index(it) -> Int

Return the 0-based box index of the current `JlMFIter`, `MFIterHandle`, `TileHandle`, `JlParIter`, or `ParIterHandle` tile.
"""
@inline grid_index(it) = Int(index(it))

"""
    parallel_reduce_sum(x::Real) -> Float64

Perform an MPI global sum reduction (`ParallelDescriptor::ReduceRealSum`) on `x`.
"""
@inline parallel_reduce_sum(x::Real) = parallel_reduce_sum_real(Float64(x))
