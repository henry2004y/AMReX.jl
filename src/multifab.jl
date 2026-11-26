# ============================================================================
# MultiFab, iMultiFab, MFIterHandle, TileHandle, and Zero-Alloc Array4View
# ============================================================================

struct _CFabView
    ptr::Ptr{Cvoid}
    lo0::Int32
    lo1::Int32
    lo2::Int32
    hi0::Int32
    hi1::Int32
    hi2::Int32
    ncomp::Int32
    _pad::Int32
end

"""
    Array4View{T} <: AbstractArray{T, 4}

Immutable, `isbits`, non-owning zero-allocation 4D view over an AMReX `FArrayBox` /
`IArrayBox` / `TagBox` buffer with 0-based spatial coordinates `(i, j, k)` and 1-based
component index `n`.

**Lifetime contract**: `Array4View` holds a raw pointer into AMReX's C++ arena and must
never be stored across `regrid_levels!`, `redistribute!`, or `MultiFab` destruction.
"""
struct Array4View{T} <: AbstractArray{T, 4}
    ptr::Ptr{T}
    lo::NTuple{3, Int}
    dims::NTuple{4, Int}
end

@inline Base.IndexStyle(::Type{<:Array4View}) = IndexCartesian()
@inline Base.size(v::Array4View) = v.dims
@inline Base.axes(v::Array4View) = (
    v.lo[1]:(v.lo[1] + v.dims[1] - 1),
    v.lo[2]:(v.lo[2] + v.dims[2] - 1),
    v.lo[3]:(v.lo[3] + v.dims[3] - 1),
    Base.OneTo(v.dims[4]),
)
@inline Base.pointer(v::Array4View) = v.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, v::Array4View{T}) where {T} = v.ptr
@inline Base.strides(v::Array4View) = (1, v.dims[1], v.dims[1] * v.dims[2], v.dims[1] * v.dims[2] * v.dims[3])
@inline Base.elsize(::Type{Array4View{T}}) where {T} = sizeof(T)

@inline Base.@propagate_inbounds function Base.getindex(v::Array4View{T}, i::Int, j::Int, k::Int, n::Int) where {T}
    @boundscheck checkbounds(v, i, j, k, n)
    lin = (i - v.lo[1]) + v.dims[1] * ((j - v.lo[2]) + v.dims[2] * ((k - v.lo[3]) + v.dims[3] * (n - 1))) + 1
    return unsafe_load(v.ptr, lin)
end

@inline Base.@propagate_inbounds function Base.setindex!(v::Array4View{T}, val, i::Int, j::Int, k::Int, n::Int) where {T}
    @boundscheck checkbounds(v, i, j, k, n)
    lin = (i - v.lo[1]) + v.dims[1] * ((j - v.lo[2]) + v.dims[2] * ((k - v.lo[3]) + v.dims[3] * (n - 1))) + 1
    unsafe_store!(v.ptr, convert(T, val), lin)
    return val
end

@inline function _cfab_to_view(::Type{T}, fv::_CFabView) where {T}
    nx = Int(fv.hi0) - Int(fv.lo0) + 1
    ny = Int(fv.hi1) - Int(fv.lo1) + 1
    nz = Int(fv.hi2) - Int(fv.lo2) + 1
    nc = Int(fv.ncomp)
    return Array4View{T}(
        Ptr{T}(fv.ptr),
        (Int(fv.lo0), Int(fv.lo1), Int(fv.lo2)),
        (nx, ny, nz, nc),
    )
end

"""
    MFIterHandle

Zero-allocation `isbits` handle to a thread-local pooled `amrex::MFIter` used inside `each_fab`.
"""
struct MFIterHandle
    ptr::Ptr{Cvoid}
    slot::Ptr{Cvoid}
end

"""
    TileHandle

Zero-allocation `isbits` tile descriptor passed to worker closures in `each_fab_threaded`.
Supports `index(th)`, `grid_index(th)`, `local_tile_index(th)`, `valid_box(th)`, `array4(mf, th)`,
and `iarray4(imf, th)`.
"""
struct TileHandle
    gid::Int32
    tid::Int32
    vbx::IndexBox
end

@inline _raw_mfi(mfi::MFIterHandle)::Ptr{Cvoid} = mfi.ptr
@inline _raw_mfi(mfi::JlMFIter)::Ptr{Cvoid} =
    @ccall $(_FN_JLMFI_RAW[])(mfi.cpp_object::Ptr{Cvoid})::Ptr{Cvoid}

@inline is_valid(mfi::MFIterHandle)::Bool =
    (@ccall $(_FN_MFITER_IS_VALID[])(mfi.ptr::Ptr{Cvoid})::UInt8) != 0
@inline next!(mfi::MFIterHandle) =
    (@ccall $(_FN_MFITER_NEXT[])(mfi.ptr::Ptr{Cvoid})::Cvoid; nothing)
@inline index(mfi::MFIterHandle)::Int32 =
    @ccall $(_FN_MFITER_INDEX[])(mfi.ptr::Ptr{Cvoid})::Int32
@inline local_tile_index(mfi::MFIterHandle)::Int32 =
    @ccall $(_FN_MFITER_LOCAL_TILE_INDEX[])(mfi.ptr::Ptr{Cvoid})::Int32
@inline valid_box(mfi::MFIterHandle)::IndexBox =
    IndexBox(@ccall $(_FN_MFITER_VALID_BOX[])(mfi.ptr::Ptr{Cvoid})::_CBoxInfo)
@inline tile_box(mfi::MFIterHandle)::IndexBox =
    IndexBox(@ccall $(_FN_MFITER_TILE_BOX[])(mfi.ptr::Ptr{Cvoid})::_CBoxInfo)
@inline fab_box(mfi::MFIterHandle)::IndexBox =
    IndexBox(@ccall $(_FN_MFITER_FAB_BOX[])(mfi.ptr::Ptr{Cvoid})::_CBoxInfo)

@inline index(th::TileHandle)::Int32 = th.gid
@inline local_tile_index(th::TileHandle)::Int32 = th.tid
@inline valid_box(th::TileHandle)::IndexBox = th.vbx
@inline tile_box(th::TileHandle)::IndexBox = th.vbx

@inline function _acquire_mfiter(mf::JlMultiFab, tiling::Bool)::MFIterHandle
    return @ccall $(_FN_MFITER_ACQUIRE_MF[])(mf.cpp_object::Ptr{Cvoid}, UInt8(tiling)::UInt8)::MFIterHandle
end

@inline function _acquire_mfiter(imf::JliMultiFab, tiling::Bool)::MFIterHandle
    return @ccall $(_FN_MFITER_ACQUIRE_IMF[])(imf.cpp_object::Ptr{Cvoid}, UInt8(tiling)::UInt8)::MFIterHandle
end

@inline function _release_mfiter(mfi::MFIterHandle)
    @ccall $(_FN_MFITER_RELEASE[])(mfi.slot::Ptr{Cvoid})::Cvoid
    return nothing
end

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
    array4(mf::JlMultiFab, mfi::Union{JlMFIter, MFIterHandle, TileHandle}) -> Array4View{Float64}

Return a zero-allocation, zero-copy 4D `Array4View{Float64}` view `(i, j, k, n)` over the
local `FArrayBox` at `mfi`, where `(i, j, k)` spans the full allocated `fab_box`
(including ghost cells) in 0-based AMReX coordinates and `n` spans `1:n_comp(mf)`.
"""
@inline function array4(mf::JlMultiFab, mfi::Union{JlMFIter, MFIterHandle})
    fv = @ccall $(_FN_FAB_VIEW[])(mf.cpp_object::Ptr{Cvoid}, _raw_mfi(mfi)::Ptr{Cvoid})::_CFabView
    return _cfab_to_view(Float64, fv)
end

@inline function array4(mf::JlMultiFab, th::TileHandle)
    fv = @ccall $(_FN_FAB_VIEW_BY_GID[])(mf.cpp_object::Ptr{Cvoid}, th.gid::Int32)::_CFabView
    return _cfab_to_view(Float64, fv)
end

"""
    iarray4(imf::JliMultiFab, mfi::Union{JlMFIter, MFIterHandle, TileHandle}) -> Array4View{Int32}

Return a zero-allocation, zero-copy 4D `Array4View{Int32}` view `(i, j, k, n)` over the
local `IArrayBox` at `mfi`.
"""
@inline function iarray4(imf::JliMultiFab, mfi::Union{JlMFIter, MFIterHandle})
    fv = @ccall $(_FN_IFAB_VIEW[])(imf.cpp_object::Ptr{Cvoid}, _raw_mfi(mfi)::Ptr{Cvoid})::_CFabView
    return _cfab_to_view(Int32, fv)
end

@inline function iarray4(imf::JliMultiFab, th::TileHandle)
    fv = @ccall $(_FN_IFAB_VIEW_BY_GID[])(imf.cpp_object::Ptr{Cvoid}, th.gid::Int32)::_CFabView
    return _cfab_to_view(Int32, fv)
end

"""
    each_fab(f::Function, mf::Union{JlMultiFab, JliMultiFab}; tiling::Bool=false)

Iterate over all local FABs/tiles of `mf` with zero heap allocations, calling
`f(mfi, valid_bx, arr)` where `vbx` is an `IndexBox` and `arr` is an `Array4View`.
"""
function each_fab(f::F, mf::JlMultiFab; tiling::Bool = false) where {F}
    mfi = _acquire_mfiter(mf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = array4(mf, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        _release_mfiter(mfi)
    end
    return nothing
end

function each_fab(f::F, imf::JliMultiFab; tiling::Bool = false) where {F}
    mfi = _acquire_mfiter(imf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            arr = iarray4(imf, mfi)
            f(mfi, bx, arr)
            next!(mfi)
        end
    finally
        _release_mfiter(mfi)
    end
    return nothing
end

"""
    each_fab_threaded(f::Function, mf::Union{JlMultiFab, JliMultiFab}; tiling::Bool=true)

Pre-collect all local FAB/tile descriptors `(TileHandle, IndexBox, Array4View)` and
execute `f(th, vbx, arr)` across Julia threads (`Threads.@threads :static`). Falls back to
`each_fab` when `Threads.nthreads() == 1`.
"""
function each_fab_threaded(f::F, mf::JlMultiFab; tiling::Bool = true) where {F}
    if Threads.nthreads() <= 1
        return each_fab(f, mf; tiling = tiling)
    end
    nloc = Int(local_size(mf))
    tiles = Vector{Tuple{TileHandle, Array4View{Float64}}}()
    sizehint!(tiles, nloc)
    mfi = _acquire_mfiter(mf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            th = TileHandle(index(mfi), local_tile_index(mfi), bx)
            push!(tiles, (th, array4(mf, mfi)))
            next!(mfi)
        end
    finally
        _release_mfiter(mfi)
    end
    Threads.@threads :static for idx in eachindex(tiles)
        th, arr = @inbounds tiles[idx]
        f(th, th.vbx, arr)
    end
    return nothing
end

function each_fab_threaded(f::F, imf::JliMultiFab; tiling::Bool = true) where {F}
    if Threads.nthreads() <= 1
        return each_fab(f, imf; tiling = tiling)
    end
    nloc = Int(local_size(imf))
    tiles = Vector{Tuple{TileHandle, Array4View{Int32}}}()
    sizehint!(tiles, nloc)
    mfi = _acquire_mfiter(imf, tiling)
    try
        while is_valid(mfi)
            bx = tiling ? tile_box(mfi) : valid_box(mfi)
            th = TileHandle(index(mfi), local_tile_index(mfi), bx)
            push!(tiles, (th, iarray4(imf, mfi)))
            next!(mfi)
        end
    finally
        _release_mfiter(mfi)
    end
    Threads.@threads :static for idx in eachindex(tiles)
        th, arr = @inbounds tiles[idx]
        f(th, th.vbx, arr)
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
