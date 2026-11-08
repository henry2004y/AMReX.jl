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
    NodeBilinear           = 1
    PiecewiseConstant      = 2
    CellQuadratic          = 3
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
    BCBogus       = -666
    BCReflectOdd  = -1
    BCIntDir      = 0
    BCReflectEven = 1
    BCFOExtrap    = 2
    BCExtDir      = 3
    BCHOExtrap    = 4
    BCHOExtrapCC  = 5
    BCExtDirCC    = 6
end

"""
    BCRec(lox, loy, loz, hix, hiy, hiz)
    BCRec(; lo=(BCIntDir, BCIntDir, BCIntDir), hi=(BCIntDir, BCIntDir, BCIntDir))

Construct an AMReX `JlBCRec` mathematical boundary condition record for 3D domain faces.
"""
function BCRec(lox::Union{Integer, MathBCType},
               loy::Union{Integer, MathBCType},
               loz::Union{Integer, MathBCType},
               hix::Union{Integer, MathBCType},
               hiy::Union{Integer, MathBCType},
               hiz::Union{Integer, MathBCType})
    return JlBCRec(Int32(lox), Int32(loy), Int32(loz),
                   Int32(hix), Int32(hiy), Int32(hiz))
end

function BCRec(; lo::NTuple{3, Union{Integer, MathBCType}}=(BCIntDir, BCIntDir, BCIntDir),
                 hi::NTuple{3, Union{Integer, MathBCType}}=(BCIntDir, BCIntDir, BCIntDir))
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
function fill_domain_boundary!(mf::JlMultiFab, geom::JlGeometry, bc::JlBCRec;
                               time::Real=0.0,
                               val_lo::Union{Nothing, NTuple{3, Real}}=nothing,
                               val_hi::Union{Nothing, NTuple{3, Real}}=nothing)
    if val_lo === nothing && val_hi === nothing
        fill_domain_boundary_val!(mf, geom, bc, Float64(time),
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else
        vlo = something(val_lo, (0.0, 0.0, 0.0))
        vhi = something(val_hi, (0.0, 0.0, 0.0))
        fill_domain_boundary_val!(mf, geom, bc, Float64(time),
                                  Float64(vlo[1]), Float64(vlo[2]), Float64(vlo[3]),
                                  Float64(vhi[1]), Float64(vhi[2]), Float64(vhi[3]))
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
    TagBuf   = 1
    TagSet   = 2
end

"""
    TagBoxArray(ba::JlBoxArray, dm::JlDistributionMapping, nghost::Integer=0) -> JlTagBoxArrayAllocated

Construct a cell-centered `TagBoxArray` with the given `BoxArray` and `DistributionMapping`.
"""
TagBoxArray(ba::JlBoxArray, dm::JlDistributionMapping, nghost::Integer=0) =
    JlTagBoxArray(ba, dm, Int32(nghost))

"""
    tag_array4(tb::JlTagBoxArray, mfi::JlMFIter) -> OffsetArray{Int8, 4, Array{Int8, 4}}

Return a zero-copy 4D `OffsetArray{Int8, 4}` view `(i, j, k, 1)` over the local `TagBox` at `mfi`.
Assign `Int8(TagSet)` (or `Int8(2)`) to mark cells for refinement and `Int8(TagClear)` (`0`) to clear.
"""
@inline function tag_array4(tb::JlTagBoxArray, mfi::JlMFIter)
    rx = Int(tfab_lo(mfi, tb, Int32(0))):Int(tfab_hi(mfi, tb, Int32(0)))
    ry = Int(tfab_lo(mfi, tb, Int32(1))):Int(tfab_hi(mfi, tb, Int32(1)))
    rz = Int(tfab_lo(mfi, tb, Int32(2))):Int(tfab_hi(mfi, tb, Int32(2)))
    nx, ny, nz = length(rx), length(ry), length(rz)
    ptr = tdata_ptr(mfi, tb).cpp_object
    raw = unsafe_wrap(Array, ptr, (nx, ny, nz, 1); own=false)
    return OffsetArray(raw, rx, ry, rz, 1:1)
end

"""
    each_fab(f::Function, tb::JlTagBoxArray; tiling::Bool=false)

Iterate over all local `TagBox`es of `tb`, calling `f(mfi, valid_bx, tag_arr)` where `tag_arr`
is the zero-copy `OffsetArray{Int8, 4}` from `tag_array4(tb, mfi)`.
"""
function each_fab(f::F, tb::JlTagBoxArray; tiling::Bool=false) where {F}
    mfi = JlMFIter(tb, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = tag_array4(tb, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        close!(mfi)
    end
    return nothing
end

const _AMR_ERROR_CALLBACKS = Dict{UInt, Function}()

"""
    on_error_est!(cb::Union{Nothing, Function}, amr::JlAmrCore)

Register a dynamic Julia error-estimation callback `cb(lev::Int, tags::JlTagBoxArray, time::Float64)`
invoked during `AmrCore` `init_from_scratch!` and `regrid_levels!`. Pass `nothing` to clear.
"""
function on_error_est!(cb::Union{Nothing, Function}, amr::JlAmrCore)
    key = UInt(amr.cpp_object)
    if cb === nothing
        delete!(_AMR_ERROR_CALLBACKS, key)
    else
        _AMR_ERROR_CALLBACKS[key] = cb
    end
    return amr
end

"""
    init_from_scratch!(amr::JlAmrCore, time::Real=0.0)

Initialize the AMR hierarchy from scratch at `time`, invoking any registered `on_error_est!`
callback and/or refine boxes and tag masks on each level.
"""
function init_from_scratch!(amr::JlAmrCore, time::Real=0.0)
    t = Float64(time)
    cb = get(_AMR_ERROR_CALLBACKS, UInt(amr.cpp_object), nothing)
    if cb === nothing
        _init_from_scratch!(amr, t)
    else
        ensure_base_grids!(amr)
        for lev in 0:(Int(max_level(amr)) - 1)
            lev > Int(finest_level(amr)) && break
            tags = TagBoxArray(box_array(amr, Int32(lev)), dmap(amr, Int32(lev)), 0)
            cb(lev, tags, t)
            set_tag_boxes!(amr, Int32(lev), tags)
            _regrid_levels!(amr, Int32(lev), t)
        end
    end
    return nothing
end

"""
    regrid_levels!(amr::JlAmrCore, base_lev::Integer, time::Real=0.0)

Rebuild all AMR levels finer than `base_lev` at `time`, invoking any registered `on_error_est!`
callback and/or refine boxes and tag masks on levels `>= base_lev`.
"""
function regrid_levels!(amr::JlAmrCore, base_lev::Integer, time::Real=0.0)
    t = Float64(time)
    lbase = Int(base_lev)
    cb = get(_AMR_ERROR_CALLBACKS, UInt(amr.cpp_object), nothing)
    if cb === nothing
        _regrid_levels!(amr, Int32(lbase), t)
    else
        max_tag_lev = min(Int(finest_level(amr)), Int(max_level(amr)) - 1)
        for lev in lbase:max_tag_lev
            tags = TagBoxArray(box_array(amr, Int32(lev)), dmap(amr, Int32(lev)), 0)
            cb(lev, tags, t)
            set_tag_boxes!(amr, Int32(lev), tags)
        end
        _regrid_levels!(amr, Int32(lbase), t)
    end
    return nothing
end

"""
    tag_cells_by_value!(amr::JlAmrCore, lev::Integer, mf::JlMultiFab, threshold::Real;
                        comp::Integer=1, abs_val::Bool=false) -> JliMultiFab

Build and register a level-`lev` `iMultiFab` refinement mask on `amr` tagging all cells where
`mf[i, j, k, comp] >= threshold` (or `abs(mf[i, j, k, comp]) >= threshold` if `abs_val=true`).
"""
function tag_cells_by_value!(amr::JlAmrCore, lev::Integer, mf::JlMultiFab, threshold::Real;
                             comp::Integer=1, abs_val::Bool=false)
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
function tag_cells_by_grad!(amr::JlAmrCore, lev::Integer, mf::JlMultiFab, grad_threshold::Real;
                            comp::Integer=1)
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
function interp_from_coarse_to_fine!(coarse::JlMultiFab, fine::JlMultiFab,
                                     icomp::Integer, ncomp::Integer,
                                     ref_ratio::NTuple{3, Integer},
                                     cgeom::JlGeometry, fgeom::JlGeometry,
                                     interp::InterpType=CellConservativeLinear,
                                     nghost::Integer=0;
                                     bc::Union{Nothing, JlBCRec}=nothing,
                                     val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                                     val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))
    rx, ry, rz = Int32.(ref_ratio)
    if bc === nothing
        interp_from_coarse_to_fine!(coarse, fine, Int32(icomp), Int32(ncomp),
                                    rx, ry, rz, cgeom, fgeom, Int32(interp), Int32(nghost))
    else
        interp_from_coarse_to_fine_bc!(coarse, fine, Int32(icomp), Int32(ncomp),
                                       rx, ry, rz, cgeom, fgeom, Int32(interp), Int32(nghost),
                                       bc,
                                       Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
                                       Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3]))
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
function fill_patch_single_level!(dst::JlMultiFab, time::Real, src::JlMultiFab,
                                  scomp::Integer, dcomp::Integer, ncomp::Integer,
                                  geom::JlGeometry;
                                  bc::Union{Nothing, JlBCRec}=nothing,
                                  val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                                  val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))
    if bc === nothing
        fill_patch_single_level!(dst, Float64(time), src, Int32(scomp), Int32(dcomp), Int32(ncomp), geom)
    else
        fill_patch_single_level_bc!(dst, Float64(time), src, Int32(scomp), Int32(dcomp), Int32(ncomp),
                                    geom, bc,
                                    Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
                                    Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3]))
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
function fill_patch_two_levels!(dst::JlMultiFab, time::Real,
                                cmf::JlMultiFab, fmf::JlMultiFab,
                                scomp::Integer, dcomp::Integer, ncomp::Integer,
                                cgeom::JlGeometry, fgeom::JlGeometry,
                                ref_ratio::NTuple{3, Integer},
                                interp::InterpType=CellConservativeLinear;
                                bc::Union{Nothing, JlBCRec}=nothing,
                                val_lo::NTuple{3, Real}=(0.0, 0.0, 0.0),
                                val_hi::NTuple{3, Real}=(0.0, 0.0, 0.0))
    rx, ry, rz = Int32.(ref_ratio)
    if bc === nothing
        fill_patch_two_levels!(dst, Float64(time), cmf, fmf,
                               Int32(scomp), Int32(dcomp), Int32(ncomp),
                               cgeom, fgeom, rx, ry, rz, Int32(interp))
    else
        fill_patch_two_levels_bc!(dst, Float64(time), cmf, fmf,
                                  Int32(scomp), Int32(dcomp), Int32(ncomp),
                                  cgeom, fgeom, rx, ry, rz, Int32(interp), bc,
                                  Float64(val_lo[1]), Float64(val_lo[2]), Float64(val_lo[3]),
                                  Float64(val_hi[1]), Float64(val_hi[2]), Float64(val_hi[3]))
    end
    return dst
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

"""
    add_tag_box!(amr::JlAmrCore, xlo, ylo, zlo, xhi, yhi, zhi; lev::Integer=0)

Register a physical bounding box `[xlo, xhi] × [ylo, yhi] × [zlo, zhi]` for mesh refinement tagging at level `lev`.
"""
@inline add_tag_box!(amr::JlAmrCore, xlo::Real, ylo::Real, zlo::Real, xhi::Real, yhi::Real, zhi::Real; lev::Integer=0) =
    add_refine_box!(amr, Int32(lev), Float64(xlo), Float64(ylo), Float64(zlo), Float64(xhi), Float64(yhi), Float64(zhi))

"""
    geometry(amr::JlAmrCore, lev::Integer=0) -> JlGeometry

Return the `JlGeometry` object at AMR level `lev`.
"""
@inline geometry(amr::JlAmrCore, lev::Integer=0) = geom(amr, Int32(lev))

"""
    dist_map(amr::JlAmrCore, lev::Integer=0) -> JlDistributionMapping

Return the `JlDistributionMapping` at AMR level `lev`.
"""
@inline dist_map(amr::JlAmrCore, lev::Integer=0) = dmap(amr, Int32(lev))

"""
    mpi_rank() -> Int

Return the 0-based MPI rank of the calling process.
"""
@inline mpi_rank() = Int(parallel_myproc())

"""
    grid_index(it) -> Int

Return the 0-based box index of the current `JlMFIter` or `JlParIter` tile.
"""
@inline grid_index(it) = Int(index(it))

"""
    parallel_reduce_sum(x::Real) -> Float64

Perform an MPI global sum reduction (`ParallelDescriptor::ReduceRealSum`) on `x`.
"""
@inline parallel_reduce_sum(x::Real) = parallel_reduce_sum_real(Float64(x))
