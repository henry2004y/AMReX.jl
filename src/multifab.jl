# ============================================================================
# MultiFab, iMultiFab, MFIter, and Zero-Copy OffsetArray Views
# ============================================================================

"""
    MultiFab(ba::JlBoxArray, dm::JlDistributionMapping, ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}})

Allocate an AMReX `JlMultiFab` (`Float64`) with `ncomp` components and `nghost` guard cells.
"""
function MultiFab(
        ba::JlBoxArray, dm::JlDistributionMapping,
        ncomp::Integer, nghost::Integer
    )
    return JlMultiFab(ba, dm, Int32(ncomp), Int32(nghost))
end

function MultiFab(
        ba::JlBoxArray, dm::JlDistributionMapping,
        ncomp::Integer, nghost::NTuple{3, Integer}
    )
    return JlMultiFab(
        ba, dm, Int32(ncomp),
        Int32(nghost[1]), Int32(nghost[2]), Int32(nghost[3])
    )
end

"""
    iMultiFab(ba::JlBoxArray, dm::JlDistributionMapping, ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}})

Allocate an AMReX `JliMultiFab` (`Int32`) with `ncomp` components and `nghost` guard cells.
"""
function iMultiFab(
        ba::JlBoxArray, dm::JlDistributionMapping,
        ncomp::Integer, nghost::Integer
    )
    return JliMultiFab(ba, dm, Int32(ncomp), Int32(nghost))
end

function iMultiFab(
        ba::JlBoxArray, dm::JlDistributionMapping,
        ncomp::Integer, nghost::NTuple{3, Integer}
    )
    return JliMultiFab(
        ba, dm, Int32(ncomp),
        Int32(nghost[1]), Int32(nghost[2]), Int32(nghost[3])
    )
end

"""
    array4(mf::JlMultiFab, mfi::JlMFIter) -> OffsetArray{Float64, 4, Array{Float64, 4}}

Return a zero-copy 4D `OffsetArray` view `(i, j, k, n)` over the local `FArrayBox`
at `mfi`, where `(i, j, k)` spans the full allocated `fab_box` (including ghost cells)
in 0-based AMReX coordinates and `n` spans `1:n_comp(mf)`.
"""
@inline function array4(mf::JlMultiFab, mfi::JlMFIter)
    info = Ref{NTuple{7, Int32}}()
    ptr = GC.@preserve info fab_info!(mfi, mf, Base.unsafe_convert(Ptr{Int32}, info)).cpp_object
    lo0, lo1, lo2, hi0, hi1, hi2, nc32 = info[]
    rx = Int(lo0):Int(hi0)
    ry = Int(lo1):Int(hi1)
    rz = Int(lo2):Int(hi2)
    nc = Int(nc32)
    raw = unsafe_wrap(Array, ptr, (length(rx), length(ry), length(rz), nc); own = false)
    return OffsetArray(raw, rx, ry, rz, 1:nc)
end

"""
    iarray4(imf::JliMultiFab, mfi::JlMFIter) -> OffsetArray{Int32, 4, Array{Int32, 4}}

Return a zero-copy 4D `OffsetArray` view `(i, j, k, n)` over the local `IArrayBox`
at `mfi`, where `(i, j, k)` spans `fab_box(mfi)` and `n` spans `1:n_comp(imf)`.
"""
@inline function iarray4(imf::JliMultiFab, mfi::JlMFIter)
    info = Ref{NTuple{7, Int32}}()
    ptr = GC.@preserve info ifab_info!(mfi, imf, Base.unsafe_convert(Ptr{Int32}, info)).cpp_object
    lo0, lo1, lo2, hi0, hi1, hi2, nc32 = info[]
    rx = Int(lo0):Int(hi0)
    ry = Int(lo1):Int(hi1)
    rz = Int(lo2):Int(hi2)
    nc = Int(nc32)
    raw = unsafe_wrap(Array, ptr, (length(rx), length(ry), length(rz), nc); own = false)
    return OffsetArray(raw, rx, ry, rz, 1:nc)
end

"""
    each_fab(f::Function, mf::JlMultiFab; tiling::Bool=false)

Iterate over all local FABs/tiles of `mf`, calling `f(mfi, valid_bx, arr)` where `arr`
is the zero-copy `OffsetArray` from `array4(mf, mfi)`.
"""
function each_fab(f::F, mf::JlMultiFab; tiling::Bool = false) where {F}
    mfi = JlMFIter(mf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = array4(mf, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        close!(mfi)
    end
    return nothing
end

function each_fab(f::F, imf::JliMultiFab; tiling::Bool = false) where {F}
    mfi = JlMFIter(imf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = iarray4(imf, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        close!(mfi)
    end
    return nothing
end

"""
    n_grow(mf::Union{JlMultiFab, JliMultiFab}) -> Int

Return the number of ghost cells along the first dimension (or `n_grow(mf, dir)` for 0-based `dir`).
"""
@inline n_grow(mf::JlMultiFab) = Int(n_grow(mf, Int32(0)))
@inline n_grow(imf::JliMultiFab) = Int(n_grow(imf, Int32(0)))

"""
    copy!(dst::JlMultiFab, src::JlMultiFab, scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)

Local in-place copy of `ncomp` components from `src` (starting at 0-based `scomp`) to `dst`
(starting at 0-based `dcomp`) including `nghost` ghost cells.
"""
@inline function copy!(
        dst::JlMultiFab, src::JlMultiFab,
        scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer
    )
    copy_from!(dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp), Int32(nghost))
    return dst
end

"""
    plus!(dst::JlMultiFab, src::JlMultiFab, scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)

Local in-place addition `dst += src` of `ncomp` components from 0-based `scomp` to `dcomp`
including `nghost` ghost cells.
"""
@inline function plus!(
        dst::JlMultiFab, src::JlMultiFab,
        scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer
    )
    add_from!(dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp), Int32(nghost))
    return dst
end

"""
    scale!(mf::JlMultiFab, val::Real)

Multiply all components and ghost cells of `mf` in-place by scalar `val`.
"""
@inline function scale!(mf::JlMultiFab, val::Real)
    mult_val!(mf, Float64(val), Int32(0), n_comp(mf), n_grow(mf, Int32(0)))
    return mf
end

"""
    saxpy!(dst::JlMultiFab, a::Real, src::JlMultiFab, scomp::Integer=0, dcomp::Integer=0, ncomp::Integer=Int(n_comp(dst)), nghost::Integer=0)

Compute `dst[dcomp + c] += a * src[scomp + c]` for `c in 1:ncomp` (0-based `scomp`, `dcomp`)
over the valid box plus `nghost` ghost cells.
"""
function saxpy!(
        dst::JlMultiFab, a::Real, src::JlMultiFab,
        scomp::Integer = 0, dcomp::Integer = 0,
        ncomp::Integer = Int(n_comp(dst)), nghost::Integer = 0
    )
    sc = Int(scomp)
    dc = Int(dcomp)
    nc = Int(ncomp)
    ng = Int(nghost)
    (0 <= sc && sc + nc <= Int(n_comp(src))) || throw(BoundsError("saxpy! scomp=$sc, ncomp=$nc out of bounds for src with $(n_comp(src)) components (0-based)"))
    (0 <= dc && dc + nc <= Int(n_comp(dst))) || throw(BoundsError("saxpy! dcomp=$dc, ncomp=$nc out of bounds for dst with $(n_comp(dst)) components (0-based)"))
    saxpy_from!(dst, Float64(a), src, Int32(sc), Int32(dc), Int32(nc), Int32(ng))
    return dst
end

"""
    lincomb!(dst::JlMultiFab, a::Real, x::JlMultiFab, xcomp::Integer, b::Real, y::JlMultiFab, ycomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)

Compute the linear combination `dst[dcomp + c] = a * x[xcomp + c] + b * y[ycomp + c]`
for `c in 1:ncomp` (0-based `xcomp`, `ycomp`, `dcomp`) over the valid box plus `nghost` ghost cells
via native `amrex::MultiFab::LinComb`.
"""
function lincomb!(
        dst::JlMultiFab,
        a::Real, x::JlMultiFab, xcomp::Integer,
        b::Real, y::JlMultiFab, ycomp::Integer,
        dcomp::Integer, ncomp::Integer, nghost::Integer
    )
    xc = Int(xcomp)
    yc = Int(ycomp)
    dc = Int(dcomp)
    nc = Int(ncomp)
    ng = Int(nghost)
    (0 <= xc && xc + nc <= Int(n_comp(x))) || throw(BoundsError("lincomb! xcomp=$xc, ncomp=$nc out of bounds for x with $(n_comp(x)) components (0-based)"))
    (0 <= yc && yc + nc <= Int(n_comp(y))) || throw(BoundsError("lincomb! ycomp=$yc, ncomp=$nc out of bounds for y with $(n_comp(y)) components (0-based)"))
    (0 <= dc && dc + nc <= Int(n_comp(dst))) || throw(BoundsError("lincomb! dcomp=$dc, ncomp=$nc out of bounds for dst with $(n_comp(dst)) components (0-based)"))
    lincomb_from!(dst, Float64(a), x, Int32(xc), Float64(b), y, Int32(yc), Int32(dc), Int32(nc), Int32(ng))
    return dst
end

"""
    dot_mf(x::JlMultiFab, y::JlMultiFab; comp1=0, comp2=0, ncomp=n_comp(x), nghost=0, local_only=false) -> Float64

Compute the inner product `∑ x[comp1 + c] * y[comp2 + c]` across all valid cells (plus `nghost`
ghost cells) via native `amrex::MultiFab::Dot` (MPI-global unless `local_only=true`).
"""
function dot_mf(
        x::JlMultiFab, y::JlMultiFab;
        comp1::Integer = 0, comp2::Integer = 0,
        ncomp::Integer = Int(n_comp(x)), nghost::Integer = 0,
        local_only::Bool = false
    )
    return Float64(dot_with(x, y, Int32(comp1), Int32(comp2), Int32(ncomp), Int32(nghost), local_only))
end

"""
    fill_boundary_nowait!(mf::JlMultiFab, geom::JlGeometry; icomp=nothing, ncomp=nothing)

Initiate a non-blocking MPI ghost-cell exchange (`FillBoundary_nowait`). Pair with
`fill_boundary_finish!(mf)` to overlap communication with interior computation.
"""
function fill_boundary_nowait!(
        mf::JlMultiFab, geom::JlGeometry;
        icomp::Union{Nothing, Integer} = nothing,
        ncomp::Union{Nothing, Integer} = nothing
    )
    sc = Int32(something(icomp, 0))
    nc = Int32(something(ncomp, Int(n_comp(mf)) - Int(sc)))
    fill_boundary_comp_nowait!(mf, sc, nc, geom)
    return mf
end

"""
    parallel_copy!(dst, src, geom; scomp=0, dcomp=0, ncomp=n_comp(dst), snghost=0, dnghost=0)

MPI-safe copy of `ncomp` components from `src` to `dst` across arbitrary `BoxArray` and
`DistributionMapping` layouts, supporting isotropic or anisotropic `(ngx, ngy, ngz)` ghost counts.
Works on both `JlMultiFab` and `JliMultiFab`.
"""
function parallel_copy!(
        dst::Union{JlMultiFab, JliMultiFab},
        src::Union{JlMultiFab, JliMultiFab},
        geom::JlGeometry;
        scomp::Integer = 0,
        dcomp::Integer = 0,
        ncomp::Integer = Int(n_comp(dst)),
        snghost::Union{Integer, NTuple{3, Integer}} = 0,
        dnghost::Union{Integer, NTuple{3, Integer}} = 0
    )
    if snghost isa Integer && dnghost isa Integer
        parallel_copy_from!(
            dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp),
            Int32(snghost), Int32(dnghost), geom
        )
    else
        sg = snghost isa Integer ? (snghost, snghost, snghost) : snghost
        dg = dnghost isa Integer ? (dnghost, dnghost, dnghost) : dnghost
        parallel_copy_from_aniso!(
            dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp),
            Int32(sg[1]), Int32(sg[2]), Int32(sg[3]),
            Int32(dg[1]), Int32(dg[2]), Int32(dg[3]),
            geom
        )
    end
    return dst
end
