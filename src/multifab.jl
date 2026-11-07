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
    fbox = fab_box(mfi)
    rx, ry, rz = bounds(fbox)
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
    fbox = fab_box(mfi)
    rx, ry, rz = bounds(fbox)
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
