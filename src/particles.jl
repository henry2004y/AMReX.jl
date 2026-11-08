# ============================================================================
# Pure-SoA ParticleContainer, ParIter, and 64-bit idcpu Bit Manipulation
# ============================================================================

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
function ParticleContainer(amr::JlAmrCore; n_real::Integer=4, n_int::Integer=1)
    return JlParticleContainer(amr, Int32(n_real), Int32(n_int))
end

function ParticleContainer(geom::JlGeometry, dm::JlDistributionMapping, ba::JlBoxArray;
                           n_real::Integer=4, n_int::Integer=1)
    return JlParticleContainer(geom, dm, ba, Int32(n_real), Int32(n_int))
end

"""
    idcpu_vec(pti::JlParIter) -> Vector{UInt64}

Return a zero-copy 1D `Vector{UInt64}` view of the packed `idcpu` array on `pti`.
"""
@inline function idcpu_vec(pti::JlParIter)
    np = Int(num_particles(pti))
    ptr = idcpu_ptr(pti).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    pos_vecs(pti::JlParIter) -> NTuple{3, Vector{Float64}}

Return a tuple `(px, py, pz)` of zero-copy 1D `Vector{Float64}` views for the particle
positions on `pti`.
"""
@inline function pos_vecs(pti::JlParIter)
    np = Int(num_particles(pti))
    px = unsafe_wrap(Array, pos_ptr(pti, Int32(0)).cpp_object, (np,); own=false)
    py = unsafe_wrap(Array, pos_ptr(pti, Int32(1)).cpp_object, (np,); own=false)
    pz = unsafe_wrap(Array, pos_ptr(pti, Int32(2)).cpp_object, (np,); own=false)
    return (px, py, pz)
end

"""
    real_vec(pti::JlParIter, comp::Integer) -> Vector{Float64}

Return a zero-copy 1D `Vector{Float64}` view of runtime real SoA component `comp` (1-based) on `pti`.
"""
@inline function real_vec(pti::JlParIter, comp::Integer)
    np = Int(num_particles(pti))
    ptr = real_ptr(pti, Int32(comp - 1)).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    int_vec(pti::JlParIter, comp::Integer) -> Vector{Int32}

Return a zero-copy 1D `Vector{Int32}` view of runtime integer SoA component `comp` (1-based) on `pti`.
"""
@inline function int_vec(pti::JlParIter, comp::Integer)
    np = Int(num_particles(pti))
    ptr = int_ptr(pti, Int32(comp - 1)).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    tile_idcpu_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer) -> Vector{UInt64}

Return a zero-copy 1D `Vector{UInt64}` view of the packed `idcpu` array on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_idcpu_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer)
    np = Int(tile_num_particles(pc, Int32(lev), Int32(grid_id), Int32(tile_id)))
    ptr = tile_idcpu_ptr(pc, Int32(lev), Int32(grid_id), Int32(tile_id)).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    tile_pos_vecs(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer) -> NTuple{3, Vector{Float64}}

Return `(px, py, pz)` zero-copy 1D `Vector{Float64}` views on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_pos_vecs(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer)
    np = Int(tile_num_particles(pc, Int32(lev), Int32(grid_id), Int32(tile_id)))
    l, g, t = Int32(lev), Int32(grid_id), Int32(tile_id)
    px = unsafe_wrap(Array, tile_pos_ptr(pc, l, g, t, Int32(0)).cpp_object, (np,); own=false)
    py = unsafe_wrap(Array, tile_pos_ptr(pc, l, g, t, Int32(1)).cpp_object, (np,); own=false)
    pz = unsafe_wrap(Array, tile_pos_ptr(pc, l, g, t, Int32(2)).cpp_object, (np,); own=false)
    return (px, py, pz)
end

"""
    tile_real_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer) -> Vector{Float64}

Return a zero-copy 1D `Vector{Float64}` view of runtime real component `comp` (1-based) on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_real_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer)
    np = Int(tile_num_particles(pc, Int32(lev), Int32(grid_id), Int32(tile_id)))
    ptr = tile_real_ptr(pc, Int32(lev), Int32(grid_id), Int32(tile_id), Int32(comp - 1)).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    tile_int_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer) -> Vector{Int32}

Return a zero-copy 1D `Vector{Int32}` view of runtime integer component `comp` (1-based) on tile `(lev, grid_id, tile_id)`.
"""
@inline function tile_int_vec(pc::JlParticleContainer, lev::Integer, grid_id::Integer, tile_id::Integer, comp::Integer)
    np = Int(tile_num_particles(pc, Int32(lev), Int32(grid_id), Int32(tile_id)))
    ptr = tile_int_ptr(pc, Int32(lev), Int32(grid_id), Int32(tile_id), Int32(comp - 1)).cpp_object
    return unsafe_wrap(Array, ptr, (np,); own=false)
end

"""
    each_particle_tile(f::Function, pc::JlParticleContainer, lev::Integer=0)

Iterate over all active particle tiles at level `lev`, calling `f(pti)`.
"""
function each_particle_tile(f::F, pc::JlParticleContainer, lev::Integer=0) where {F}
    pti = JlParIter(pc, Int32(lev))
    try
        while is_valid(pti)
            f(pti)
            next!(pti)
        end
    finally
        close!(pti)
    end
    return nothing
end

"""
    redistribute!(pc::JlParticleContainer; lev_min::Integer=0, lev_max::Integer=-1, ngrow::Integer=0)

Redistribute particles across AMR levels, grids, and MPI ranks according to their updated
coordinates, and remove any particles marked invalid via `make_invalid_idcpu`.
"""
@inline redistribute!(pc::JlParticleContainer; lev_min::Integer=0, lev_max::Integer=-1, ngrow::Integer=0) =
    redistribute!(pc, Int32(lev_min), Int32(lev_max), Int32(ngrow))

"""
    total_number_of_particles(pc::JlParticleContainer; only_valid::Bool=true, local_only::Bool=false) -> Int64

Return the total number of particles across all AMR levels in `pc`.
"""
@inline total_number_of_particles(pc::JlParticleContainer; only_valid::Bool=true, local_only::Bool=false) =
    total_number_of_particles(pc, only_valid, local_only)

"""
    number_of_particles_at_level(pc::JlParticleContainer, lev::Integer; only_valid::Bool=true, local_only::Bool=false) -> Int64

Return the number of particles at AMR level `lev` in `pc`.
"""
@inline number_of_particles_at_level(pc::JlParticleContainer, lev::Integer; only_valid::Bool=true, local_only::Bool=false) =
    number_of_particles_at_level(pc, Int32(lev), only_valid, local_only)
