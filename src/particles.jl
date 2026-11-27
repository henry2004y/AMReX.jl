# ============================================================================
# Pure-SoA ParticleContainer, ParIterHandle, PtrVector, & 64-bit idcpu
# ============================================================================

"""
    PtrVector{T} <: DenseVector{T}

Immutable, `isbits`, non-owning zero-allocation 1D vector view over a contiguous C++
particle SoA buffer.

**Lifetime contract**: `PtrVector` holds a raw pointer into AMReX's C++ arena and must
never be stored across `redistribute!`, `resize_tile!`, `regrid_levels!`, or container destruction.
"""
struct PtrVector{T} <: DenseVector{T}
    ptr::Ptr{T}
    len::Int
end

@inline Base.IndexStyle(::Type{<:PtrVector}) = IndexLinear()
@inline Base.size(v::PtrVector) = (v.len,)
@inline Base.pointer(v::PtrVector) = v.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, v::PtrVector{T}) where {T} = v.ptr
@inline Base.elsize(::Type{PtrVector{T}}) where {T} = sizeof(T)
@inline Base.strides(::PtrVector) = (1,)

@inline Base.@propagate_inbounds function Base.getindex(v::PtrVector{T}, i::Int) where {T}
    @boundscheck checkbounds(v, i)
    return unsafe_load(v.ptr, i)
end

@inline Base.@propagate_inbounds function Base.setindex!(v::PtrVector{T}, val, i::Int) where {T}
    @boundscheck checkbounds(v, i)
    unsafe_store!(v.ptr, convert(T, val), i)
    return val
end

struct _CSoATileInfo
    np::Int64
    idcpu::Ptr{UInt64}
    px::Ptr{Float64}
    py::Ptr{Float64}
    pz::Ptr{Float64}
    n_real::Int32
    n_int::Int32
end

"""
    ParIterHandle

Zero-allocation `isbits` handle to a thread-local pooled `PureSoAParIter` used inside `each_particle_tile`.
"""
struct ParIterHandle
    ptr::Ptr{Cvoid}
    slot::Ptr{Cvoid}
end

"""
    SoATileViews{NR, NI}

Immutable `isbits` bundle of all SoA particle tile pointers `(np, idcpu, px, py, pz, real, int)`
fetched in a single C++ tile lookup with zero heap allocations.
"""
struct SoATileViews{NR, NI}
    np::Int
    idcpu::PtrVector{UInt64}
    px::PtrVector{Float64}
    py::PtrVector{Float64}
    pz::PtrVector{Float64}
    real::NTuple{NR, PtrVector{Float64}}
    int::NTuple{NI, PtrVector{Int32}}
end

@inline _raw_pti(pti::ParIterHandle)::Ptr{Cvoid} = pti.ptr
@inline _raw_pti(pti::JlParIter)::Ptr{Cvoid} =
    @ccall $(_FN_JLPTI_RAW[])(pti.cpp_object::Ptr{Cvoid})::Ptr{Cvoid}

@inline is_valid(pti::ParIterHandle)::Bool =
    (@ccall $(_FN_PARITER_IS_VALID[])(pti.ptr::Ptr{Cvoid})::UInt8) != 0
@inline next!(pti::ParIterHandle) =
    (@ccall $(_FN_PARITER_NEXT[])(pti.ptr::Ptr{Cvoid})::Cvoid; nothing)
@inline index(pti::ParIterHandle)::Int32 =
    @ccall $(_FN_PARITER_INDEX[])(pti.ptr::Ptr{Cvoid})::Int32
@inline local_tile_index(pti::ParIterHandle)::Int32 =
    @ccall $(_FN_PARITER_LOCAL_TILE_INDEX[])(pti.ptr::Ptr{Cvoid})::Int32
@inline num_particles(pti::ParIterHandle)::Int64 =
    @ccall $(_FN_PARITER_NUM_PARTICLES[])(pti.ptr::Ptr{Cvoid})::Int64
@inline valid_box(pti::ParIterHandle)::IndexBox =
    IndexBox(@ccall $(_FN_PARITER_VALID_BOX[])(pti.ptr::Ptr{Cvoid})::_CBoxInfo)
@inline tile_box(pti::ParIterHandle)::IndexBox =
    IndexBox(@ccall $(_FN_PARITER_TILE_BOX[])(pti.ptr::Ptr{Cvoid})::_CBoxInfo)
@inline resize_tile!(pti::ParIterHandle, new_size::Integer) =
    (@ccall $(_FN_PARITER_RESIZE_TILE[])(pti.ptr::Ptr{Cvoid}, Int64(new_size)::Int64)::Cvoid; nothing)

@inline function _acquire_pariter(pc::JlParticleContainer, lev::Integer)::ParIterHandle
    return @ccall $(_FN_PARITER_ACQUIRE[])(pc.cpp_object::Ptr{Cvoid}, Int32(lev)::Int32)::ParIterHandle
end

@inline function _release_pariter(pti::ParIterHandle)
    @ccall $(_FN_PARITER_RELEASE[])(pti.slot::Ptr{Cvoid})::Cvoid
    return nothing
end

"""
    pack_idcpu(id::Int64, cpu::Integer) -> UInt64

Pack a 39-bit signed particle `id` (`>= 0` for valid particles) and 24-bit MPI `cpu` rank
into AMReX's 64-bit `idcpu` format (`AMReX_Particle.H`).
"""
@inline function pack_idcpu(id::Int64, cpu::Integer)::UInt64
    sign_bit = ifelse(id >= 0, UInt64(1), UInt64(0))
    abs_id = UInt64(abs(id)) & 0x0000007FFFFFFFFF
    cpu_bits = UInt64(cpu) & 0x0000000000FFFFFF
    return (sign_bit << 63) | (abs_id << 24) | cpu_bits
end

"""
    unpack_id(idcpu::UInt64) -> Int64

Extract the signed particle ID from an AMReX 64-bit `idcpu` word.
"""
@inline function unpack_id(idcpu::UInt64)::Int64
    sign_bit = idcpu >> 63
    val = Int64((idcpu >> 24) & 0x0000007FFFFFFFFF)
    return ifelse(sign_bit == 1, val, -val)
end

"""
    unpack_cpu(idcpu::UInt64) -> Int

Extract the 24-bit MPI rank (`cpu`) from an AMReX 64-bit `idcpu` word.
"""
@inline function unpack_cpu(idcpu::UInt64)::Int
    return Int(idcpu & 0x0000000000FFFFFF)
end

"""
    is_valid_idcpu(idcpu::UInt64) -> Bool

Return `true` if the particle's sign bit (bit 63) is set (`id >= 0`).
"""
@inline is_valid_idcpu(idcpu::UInt64)::Bool = (idcpu >> 63) == 1

"""
    make_invalid_idcpu(idcpu::UInt64) -> UInt64

Clear bit 63 of `idcpu` so AMReX's `Redistribute!` discards the particle.
"""
@inline make_invalid_idcpu(idcpu::UInt64)::UInt64 = idcpu & ~(UInt64(1) << 63)

"""
    make_valid_idcpu(idcpu::UInt64) -> UInt64

Set bit 63 of `idcpu` to mark the particle valid.
"""
@inline make_valid_idcpu(idcpu::UInt64)::UInt64 = idcpu | (UInt64(1) << 63)

"""
    ParticleContainer(amr::JlAmrCore; n_real::Integer=4, n_int::Integer=1)
    ParticleContainer(geom::JlGeometry, dm::JlDistributionMapping, ba::JlBoxArray; n_real::Integer=4, n_int::Integer=1)

Construct a Pure-SoA `JlParticleContainer` where `x, y, z, idcpu` plus `n_real` runtime
`Float64` components and `n_int` runtime `Int32` components are stored in contiguous 1D SoA arrays.
"""
function ParticleContainer(amr::JlAmrCore; n_real::Integer = 4, n_int::Integer = 1)
    return JlParticleContainer(amr, Int32(n_real), Int32(n_int))
end

function ParticleContainer(
        geom::JlGeometry, dm::JlDistributionMapping, ba::JlBoxArray;
        n_real::Integer = 4, n_int::Integer = 1
    )
    return JlParticleContainer(geom, dm, ba, Int32(n_real), Int32(n_int))
end

"""
    idcpu_vec(pti::Union{JlParIter, ParIterHandle}) -> PtrVector{UInt64}

Return a zero-allocation, zero-copy 1D `PtrVector{UInt64}` view of the packed `idcpu` array on `pti`.
"""
@inline function idcpu_vec(pti::Union{JlParIter, ParIterHandle})
    raw = _raw_pti(pti)
    np = Int(@ccall $(_FN_PARITER_NUM_PARTICLES[])(raw::Ptr{Cvoid})::Int64)
    ptr = @ccall $(_FN_PARITER_IDCPU_PTR[])(raw::Ptr{Cvoid})::Ptr{UInt64}
    return PtrVector{UInt64}(ptr, np)
end

"""
    pos_vecs(pti::Union{JlParIter, ParIterHandle}) -> NTuple{3, PtrVector{Float64}}

Return a tuple `(px, py, pz)` of zero-allocation 1D `PtrVector{Float64}` views for particle positions on `pti`.
"""
@inline function pos_vecs(pti::Union{JlParIter, ParIterHandle})
    raw = _raw_pti(pti)
    np = Int(@ccall $(_FN_PARITER_NUM_PARTICLES[])(raw::Ptr{Cvoid})::Int64)
    px = PtrVector{Float64}(@ccall($(_FN_PARITER_POS_PTR[])(raw::Ptr{Cvoid}, Int32(0)::Int32)::Ptr{Float64}), np)
    py = PtrVector{Float64}(@ccall($(_FN_PARITER_POS_PTR[])(raw::Ptr{Cvoid}, Int32(1)::Int32)::Ptr{Float64}), np)
    pz = PtrVector{Float64}(@ccall($(_FN_PARITER_POS_PTR[])(raw::Ptr{Cvoid}, Int32(2)::Int32)::Ptr{Float64}), np)
    return (px, py, pz)
end

"""
    real_vec(pti::Union{JlParIter, ParIterHandle}, comp::Integer) -> PtrVector{Float64}

Return a zero-allocation 1D `PtrVector{Float64}` view of runtime real SoA component `comp` (1-based) on `pti`.
"""
@inline function real_vec(pti::Union{JlParIter, ParIterHandle}, comp::Integer)
    raw = _raw_pti(pti)
    np = Int(@ccall $(_FN_PARITER_NUM_PARTICLES[])(raw::Ptr{Cvoid})::Int64)
    ptr = @ccall $(_FN_PARITER_REAL_PTR[])(raw::Ptr{Cvoid}, Int32(comp - 1)::Int32)::Ptr{Float64}
    return PtrVector{Float64}(ptr, np)
end

"""
    int_vec(pti::Union{JlParIter, ParIterHandle}, comp::Integer) -> PtrVector{Int32}

Return a zero-allocation 1D `PtrVector{Int32}` view of runtime integer SoA component `comp` (1-based) on `pti`.
"""
@inline function int_vec(pti::Union{JlParIter, ParIterHandle}, comp::Integer)
    raw = _raw_pti(pti)
    np = Int(@ccall $(_FN_PARITER_NUM_PARTICLES[])(raw::Ptr{Cvoid})::Int64)
    ptr = @ccall $(_FN_PARITER_INT_PTR[])(raw::Ptr{Cvoid}, Int32(comp - 1)::Int32)::Ptr{Int32}
    return PtrVector{Int32}(ptr, np)
end

"""
    soa_views(pti::Union{JlParIter, ParIterHandle}, ::Val{NR}, ::Val{NI}) -> SoATileViews{NR, NI}

Fetch particle count, `idcpu`, `(px, py, pz)`, `NR` real components, and `NI` integer components
in a single zero-allocation C++ call.
"""
@inline function soa_views(pti::Union{JlParIter, ParIterHandle}, ::Val{NR}, ::Val{NI}) where {NR, NI}
    raw = _raw_pti(pti)
    info = @ccall $(_FN_PARITER_SOA_INFO[])(
        raw::Ptr{Cvoid},
        Ptr{Ptr{Float64}}(C_NULL)::Ptr{Ptr{Float64}}, Int32(0)::Int32,
        Ptr{Ptr{Int32}}(C_NULL)::Ptr{Ptr{Int32}}, Int32(0)::Int32
    )::_CSoATileInfo
    np = Int(info.np)
    real_vecs = ntuple(
        c -> PtrVector{Float64}(@ccall($(_FN_PARITER_REAL_PTR[])(raw::Ptr{Cvoid}, Int32(c - 1)::Int32)::Ptr{Float64}), np),
        Val(NR)
    )
    int_vecs = ntuple(
        c -> PtrVector{Int32}(@ccall($(_FN_PARITER_INT_PTR[])(raw::Ptr{Cvoid}, Int32(c - 1)::Int32)::Ptr{Int32}), np),
        Val(NI)
    )
    return SoATileViews{NR, NI}(
        np,
        PtrVector{UInt64}(info.idcpu, np),
        PtrVector{Float64}(info.px, np),
        PtrVector{Float64}(info.py, np),
        PtrVector{Float64}(info.pz, np),
        real_vecs,
        int_vecs,
    )
end

"""
    tile_idcpu_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer) -> PtrVector{UInt64}

Return a zero-allocation 1D `PtrVector{UInt64}` view of the packed `idcpu` array on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_idcpu_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer)
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    np = Int(@ccall $(_FN_TILE_NUM_PARTICLES[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32)::Int64)
    ptr = @ccall $(_FN_TILE_IDCPU_PTR[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32)::Ptr{UInt64}
    return PtrVector{UInt64}(ptr, np)
end

"""
    tile_pos_vecs(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer) -> NTuple{3, PtrVector{Float64}}

Return `(px, py, pz)` zero-allocation 1D `PtrVector{Float64}` views on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_pos_vecs(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer)
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    info = @ccall $(_FN_TILE_SOA_INFO[])(
        pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32,
        Ptr{Ptr{Float64}}(C_NULL)::Ptr{Ptr{Float64}}, Int32(0)::Int32,
        Ptr{Ptr{Int32}}(C_NULL)::Ptr{Ptr{Int32}}, Int32(0)::Int32
    )::_CSoATileInfo
    np = Int(info.np)
    return (PtrVector{Float64}(info.px, np), PtrVector{Float64}(info.py, np), PtrVector{Float64}(info.pz, np))
end

"""
    tile_real_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer) -> PtrVector{Float64}

Return a zero-allocation 1D `PtrVector{Float64}` view of runtime real component `comp` (1-based) on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_real_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer)
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    np = Int(@ccall $(_FN_TILE_NUM_PARTICLES[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32)::Int64)
    ptr = @ccall $(_FN_TILE_REAL_PTR[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32, Int32(comp - 1)::Int32)::Ptr{Float64}
    return PtrVector{Float64}(ptr, np)
end

"""
    tile_int_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer) -> PtrVector{Int32}

Return a zero-allocation 1D `PtrVector{Int32}` view of runtime integer component `comp` (1-based) on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_int_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer)
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    np = Int(@ccall $(_FN_TILE_NUM_PARTICLES[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32)::Int64)
    ptr = @ccall $(_FN_TILE_INT_PTR[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32, Int32(comp - 1)::Int32)::Ptr{Int32}
    return PtrVector{Int32}(ptr, np)
end

"""
    tile_soa_views(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer,
                   ::Val{NR}, ::Val{NI}) -> SoATileViews{NR, NI}

Fetch all SoA pointers for tile `(lev, grid_id, tile_id)` in a single `DefineAndReturnParticleTile` lookup.
"""
@inline function tile_soa_views(
        pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer,
        ::Val{NR}, ::Val{NI}
    ) where {NR, NI}
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    info = @ccall $(_FN_TILE_SOA_INFO[])(
        pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32,
        Ptr{Ptr{Float64}}(C_NULL)::Ptr{Ptr{Float64}}, Int32(0)::Int32,
        Ptr{Ptr{Int32}}(C_NULL)::Ptr{Ptr{Int32}}, Int32(0)::Int32
    )::_CSoATileInfo
    np = Int(info.np)
    real_vecs = ntuple(
        c -> PtrVector{Float64}(@ccall($(_FN_TILE_REAL_PTR[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32, Int32(c - 1)::Int32)::Ptr{Float64}), np),
        Val(NR)
    )
    int_vecs = ntuple(
        c -> PtrVector{Int32}(@ccall($(_FN_TILE_INT_PTR[])(pc.cpp_object::Ptr{Cvoid}, l::Int32, g::Int32, t::Int32, Int32(c - 1)::Int32)::Ptr{Int32}), np),
        Val(NI)
    )
    return SoATileViews{NR, NI}(
        np,
        PtrVector{UInt64}(info.idcpu, np),
        PtrVector{Float64}(info.px, np),
        PtrVector{Float64}(info.py, np),
        PtrVector{Float64}(info.pz, np),
        real_vecs,
        int_vecs,
    )
end

"""
    each_particle_tile(f::Function, pc::JlParticleContainer, lev::Integer=0)

Iterate over all active particle tiles at level `lev` with zero heap allocations, calling `f(pti)`.
"""
function each_particle_tile(f::F, pc::JlParticleContainer, lev::Integer = 0) where {F}
    pti = _acquire_pariter(pc, lev)
    try
        while is_valid(pti)
            f(pti)
            next!(pti)
        end
    finally
        _release_pariter(pti)
    end
    return nothing
end

"""
    each_particle_tile_threaded(f::Function, pc::JlParticleContainer, lev::Integer=0,
                                ::Val{NR}=Val(4), ::Val{NI}=Val(1))

Collect all active particle tiles `(gid, tid, views::SoATileViews{NR, NI})` on level `lev`
and execute `f(gid, tid, views)` across Julia threads (`Threads.@threads :static`).
"""
function each_particle_tile_threaded(
        f::F, pc::JlParticleContainer, lev::Integer = 0,
        ::Val{NR} = Val(4), ::Val{NI} = Val(1)
    ) where {F, NR, NI}
    tiles = Vector{Tuple{Int32, Int32, SoATileViews{NR, NI}}}()
    pti = _acquire_pariter(pc, lev)
    try
        while is_valid(pti)
            push!(tiles, (index(pti), local_tile_index(pti), soa_views(pti, Val(NR), Val(NI))))
            next!(pti)
        end
    finally
        _release_pariter(pti)
    end
    if Threads.nthreads() <= 1
        for (gid, tid, views) in tiles
            f(gid, tid, views)
        end
    else
        Threads.@threads :static for idx in eachindex(tiles)
            gid, tid, views = @inbounds tiles[idx]
            f(gid, tid, views)
        end
    end
    return nothing
end

"""
    redistribute!(pc::JlParticleContainer; lev_min::Integer=0, lev_max::Integer=-1, ngrow::Integer=0)

Redistribute particles across AMR levels, grids, and MPI ranks according to their updated
coordinates, and remove any particles marked invalid via `make_invalid_idcpu`.
"""
@inline redistribute!(pc::JlParticleContainer; lev_min::Integer = 0, lev_max::Integer = -1, ngrow::Integer = 0) =
    redistribute!(pc, Int32(lev_min), Int32(lev_max), Int32(ngrow))

"""
    total_number_of_particles(pc::JlParticleContainer; only_valid::Bool=true, local_only::Bool=false) -> Int64

Return the total number of particles across all AMR levels in `pc`.
"""
@inline total_number_of_particles(pc::JlParticleContainer; only_valid::Bool = true, local_only::Bool = false) =
    total_number_of_particles(pc, only_valid, local_only)

"""
    number_of_particles_at_level(pc::JlParticleContainer, lev::Integer; only_valid::Bool=true, local_only::Bool=false) -> Int64

Return the number of particles at AMR level `lev` in `pc`.
"""
@inline number_of_particles_at_level(pc::JlParticleContainer, lev::Integer; only_valid::Bool = true, local_only::Bool = false) =
    number_of_particles_at_level(pc, Int32(lev), only_valid, local_only)
