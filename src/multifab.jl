# ============================================================================
# MultiFab, iMultiFab, MFIter, and Zero-Copy OffsetArray Views
# ============================================================================

"""
    MultiFab(ba::JlBoxArray, dm::JlDistributionMapping, ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}})

Allocate an AMReX `JlMultiFab` (`Float64`) with `ncomp` components and `nghost` guard cells.
"""
function MultiFab(ba::JlBoxArray, dm::JlDistributionMapping,
                  ncomp::Integer, nghost::Integer)
    return JlMultiFab(ba, dm, Int32(ncomp), Int32(nghost))
end

function MultiFab(ba::JlBoxArray, dm::JlDistributionMapping,
                  ncomp::Integer, nghost::NTuple{3, Integer})
    return JlMultiFab(ba, dm, Int32(ncomp),
                      Int32(nghost[1]), Int32(nghost[2]), Int32(nghost[3]))
end

"""
    iMultiFab(ba::JlBoxArray, dm::JlDistributionMapping, ncomp::Integer, nghost::Union{Integer, NTuple{3, Integer}})

Allocate an AMReX `JliMultiFab` (`Int32`) with `ncomp` components and `nghost` guard cells.
"""
function iMultiFab(ba::JlBoxArray, dm::JlDistributionMapping,
                   ncomp::Integer, nghost::Integer)
    return JliMultiFab(ba, dm, Int32(ncomp), Int32(nghost))
end

function iMultiFab(ba::JlBoxArray, dm::JlDistributionMapping,
                   ncomp::Integer, nghost::NTuple{3, Integer})
    return JliMultiFab(ba, dm, Int32(ncomp),
                       Int32(nghost[1]), Int32(nghost[2]), Int32(nghost[3]))
end

"""
    array4(mf::JlMultiFab, mfi::JlMFIter) -> OffsetArray{Float64, 4, Array{Float64, 4}}

Return a zero-copy 4D `OffsetArray` view `(i, j, k, n)` over the local `FArrayBox`
at `mfi`, where `(i, j, k)` spans the full allocated `fab_box` (including ghost cells)
in 0-based AMReX coordinates and `n` spans `1:n_comp(mf)`.
"""
@inline function array4(mf::JlMultiFab, mfi::JlMFIter)
    rx = Int(fab_lo(mfi, mf, Int32(0))):Int(fab_hi(mfi, mf, Int32(0)))
    ry = Int(fab_lo(mfi, mf, Int32(1))):Int(fab_hi(mfi, mf, Int32(1)))
    rz = Int(fab_lo(mfi, mf, Int32(2))):Int(fab_hi(mfi, mf, Int32(2)))
    nx, ny, nz = length(rx), length(ry), length(rz)
    nc = Int(n_comp(mf))
    ptr = data_ptr(mfi, mf).cpp_object
    raw = unsafe_wrap(Array, ptr, (nx, ny, nz, nc); own=false)
    return OffsetArray(raw, rx, ry, rz, 1:nc)
end

"""
    iarray4(imf::JliMultiFab, mfi::JlMFIter) -> OffsetArray{Int32, 4, Array{Int32, 4}}

Return a zero-copy 4D `OffsetArray` view `(i, j, k, n)` over the local `IArrayBox`
at `mfi`, where `(i, j, k)` spans `fab_box(mfi)` and `n` spans `1:n_comp(imf)`.
"""
@inline function iarray4(imf::JliMultiFab, mfi::JlMFIter)
    rx = Int(ifab_lo(mfi, imf, Int32(0))):Int(ifab_hi(mfi, imf, Int32(0)))
    ry = Int(ifab_lo(mfi, imf, Int32(1))):Int(ifab_hi(mfi, imf, Int32(1)))
    rz = Int(ifab_lo(mfi, imf, Int32(2))):Int(ifab_hi(mfi, imf, Int32(2)))
    nx, ny, nz = length(rx), length(ry), length(rz)
    nc = Int(n_comp(imf))
    ptr = idata_ptr(mfi, imf).cpp_object
    raw = unsafe_wrap(Array, ptr, (nx, ny, nz, nc); own=false)
    return OffsetArray(raw, rx, ry, rz, 1:nc)
end

"""
    each_fab(f::Function, mf::JlMultiFab; tiling::Bool=false)

Iterate over all local FABs/tiles of `mf`, calling `f(mfi, valid_bx, arr)` where `arr`
is the zero-copy `OffsetArray` from `array4(mf, mfi)`.
"""
function each_fab(f::F, mf::JlMultiFab; tiling::Bool=false) where {F}
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

function each_fab(f::F, imf::JliMultiFab; tiling::Bool=false) where {F}
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

@inline n_grow(mf::JlMultiFab) = Int(n_grow(mf, Int32(0)))
@inline n_grow(imf::JliMultiFab) = Int(n_grow(imf, Int32(0)))

@inline function copy!(dst::JlMultiFab, src::JlMultiFab,
                       scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)
    copy_from!(dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp), Int32(nghost))
    return dst
end

@inline function plus!(dst::JlMultiFab, src::JlMultiFab,
                       scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)
    add_from!(dst, src, Int32(scomp), Int32(dcomp), Int32(ncomp), Int32(nghost))
    return dst
end

@inline function scale!(mf::JlMultiFab, val::Real)
    mult_val!(mf, Float64(val), Int32(0), n_comp(mf), n_grow(mf, Int32(0)))
    return mf
end

function saxpy!(dst::JlMultiFab, a::Real, src::JlMultiFab,
                scomp::Integer, dcomp::Integer, ncomp::Integer, nghost::Integer)
    af = Float64(a)
    sc = Int(scomp)
    dc = Int(dcomp)
    nc = Int(ncomp)
    ng = Int(nghost)
    each_fab(dst) do mfi, vbx, d_arr
        s_arr = array4(src, mfi)
        rx0, ry0, rz0 = bounds(vbx)
        frx, fry, frz = axes(d_arr)[1:3]
        rx = max(first(frx), first(rx0) - ng):min(last(frx), last(rx0) + ng)
        ry = max(first(fry), first(ry0) - ng):min(last(fry), last(ry0) + ng)
        rz = max(first(frz), first(rz0) - ng):min(last(frz), last(rz0) + ng)
        @inbounds for c in 1:nc, k in rz, j in ry, i in rx
            d_arr[i, j, k, dc + c] += af * s_arr[i, j, k, sc + c]
        end
    end
    return dst
end

function lincomb!(dst::JlMultiFab,
                  a::Real, x::JlMultiFab, xcomp::Integer,
                  b::Real, y::JlMultiFab, ycomp::Integer,
                  dcomp::Integer, ncomp::Integer, nghost::Integer)
    af = Float64(a)
    bf = Float64(b)
    xc = Int(xcomp)
    yc = Int(ycomp)
    dc = Int(dcomp)
    nc = Int(ncomp)
    ng = Int(nghost)
    each_fab(dst) do mfi, vbx, d_arr
        x_arr = array4(x, mfi)
        y_arr = array4(y, mfi)
        rx0, ry0, rz0 = bounds(vbx)
        frx, fry, frz = axes(d_arr)[1:3]
        rx = max(first(frx), first(rx0) - ng):min(last(frx), last(rx0) + ng)
        ry = max(first(fry), first(ry0) - ng):min(last(fry), last(ry0) + ng)
        rz = max(first(frz), first(rz0) - ng):min(last(frz), last(rz0) + ng)
        @inbounds for c in 1:nc, k in rz, j in ry, i in rx
            d_arr[i, j, k, dc + c] = af * x_arr[i, j, k, xc + c] + bf * y_arr[i, j, k, yc + c]
        end
    end
    return dst
end
