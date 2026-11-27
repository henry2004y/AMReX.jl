const _ScratchPoolKey = Tuple{Int, Int, NTuple{3, Int}, NTuple{3, Bool}}

"""
    ScratchMultiFabPool()

Re-entrant level-scratch `AMReX.MultiFab` pool eliminating repeated C++ arena allocations on
temporary subcycling, flux, or elliptic buffers. Automatically invalidates cached buffers
when the AMR hierarchy's level count or `BoxArray` geometry changes after regridding.
"""
mutable struct ScratchMultiFabPool
    signature::Vector{UInt64}
    available::Dict{_ScratchPoolKey, Vector{JlMultiFabAllocated}}
    hits::Int
    misses::Int
end

ScratchMultiFabPool() = ScratchMultiFabPool(UInt64[], Dict{_ScratchPoolKey, Vector{JlMultiFabAllocated}}(), 0, 0)

function _grid_signature(amr::JlAmrCore)::Vector{UInt64}
    nlev = n_levels(amr)
    sig = Vector{UInt64}(undef, nlev + 1)
    sig[1] = UInt64(nlev)
    @inbounds for lev in 0:(nlev - 1)
        ba = box_array(amr, Int32(lev))
        nb = Int(size(ba))
        h = hash(nb)
        for b in 0:(nb - 1)
            bx = get_box(ba, Int32(b))
            h = hash(bounds(bx), h)
        end
        sig[lev + 2] = h
    end
    return sig
end

"""
    invalidate_scratch_pool!(pool::ScratchMultiFabPool)

Drop all cached `MultiFab`s from `pool` and reset its hierarchy signature.
"""
function invalidate_scratch_pool!(pool::ScratchMultiFabPool)
    empty!(pool.signature)
    empty!(pool.available)
    return pool
end

function _sync_pool_signature!(pool::ScratchMultiFabPool, amr::JlAmrCore)
    sig = _grid_signature(amr)
    if pool.signature != sig
        empty!(pool.available)
        resize!(pool.signature, length(sig))
        copyto!(pool.signature, sig)
    end
    return nothing
end

@inline _norm_nghost(ng::Integer) = (Int(ng), Int(ng), Int(ng))
@inline _norm_nghost(ng::NTuple{3, Integer}) = (Int(ng[1]), Int(ng[2]), Int(ng[3]))

"""
    checkout_scratch_mf!(pool::ScratchMultiFabPool, amr::JlAmrCore, lev::Integer,
                         ncomp::Integer, nghost = 0; dirs = (false, false, false),
                         init_val = nothing) -> JlMultiFabAllocated

Acquire a `MultiFab` matching `(lev, ncomp, nghost, dirs)` on `amr` from `pool`,
allocating a new one on cache miss. Optionally initialize entries to `init_val`.
"""
function checkout_scratch_mf!(
        pool::ScratchMultiFabPool,
        amr::JlAmrCore,
        lev::Integer,
        ncomp::Integer,
        nghost::Union{Integer, NTuple{3, Integer}} = 0;
        dirs::NTuple{3, Bool} = (false, false, false),
        init_val::Union{Nothing, Float64} = nothing,
    )::JlMultiFabAllocated
    _sync_pool_signature!(pool, amr)
    l = Int(lev)
    nc = Int(ncomp)
    ng3 = _norm_nghost(nghost)
    key = (l, nc, ng3, dirs)
    stack = get!(pool.available, key) do
        JlMultiFabAllocated[]
    end
    mf = if isempty(stack)
        pool.misses += 1
        l32 = Int32(l)
        ba = any(dirs) ? nodal_box_array(box_array(amr, l32); dirs = dirs) : box_array(amr, l32)
        MultiFab(ba, dmap(amr, l32), nc, ng3)
    else
        pool.hits += 1
        pop!(stack)
    end
    init_val === nothing || set_val!(mf, init_val)
    return mf
end

"""
    release_scratch_mf!(pool::ScratchMultiFabPool, mf::JlMultiFabAllocated,
                        lev::Integer, ncomp::Integer, nghost = 0; dirs = (false, false, false))

Return `mf` checked out via [`checkout_scratch_mf!`](@ref) back to `pool` for reuse.
"""
function release_scratch_mf!(
        pool::ScratchMultiFabPool,
        mf::JlMultiFabAllocated,
        lev::Integer,
        ncomp::Integer,
        nghost::Union{Integer, NTuple{3, Integer}} = 0;
        dirs::NTuple{3, Bool} = (false, false, false),
    )
    key = (Int(lev), Int(ncomp), _norm_nghost(nghost), dirs)
    stack = get!(pool.available, key) do
        JlMultiFabAllocated[]
    end
    push!(stack, mf)
    return nothing
end

"""
    with_scratch_mf!(f, pool::ScratchMultiFabPool, amr::JlAmrCore, lev::Integer,
                     ncomp::Integer, nghost = 0; dirs = (false, false, false), init_val = nothing)

Check out a scratch `MultiFab`, pass it to `f(mf)`, and unconditionally release it back into
`pool` inside a `try ... finally` block.
"""
function with_scratch_mf!(
        f,
        pool::ScratchMultiFabPool,
        amr::JlAmrCore,
        lev::Integer,
        ncomp::Integer,
        nghost::Union{Integer, NTuple{3, Integer}} = 0;
        dirs::NTuple{3, Bool} = (false, false, false),
        init_val::Union{Nothing, Float64} = nothing,
    )
    mf = checkout_scratch_mf!(pool, amr, lev, ncomp, nghost; dirs = dirs, init_val = init_val)
    try
        return f(mf)
    finally
        release_scratch_mf!(pool, mf, lev, ncomp, nghost; dirs = dirs)
    end
end
