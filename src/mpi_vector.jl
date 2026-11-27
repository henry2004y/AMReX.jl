"""
    MPIDistVector <: AbstractVector{Float64}

MPI-distributed `Float64` vector adapter for `LinearAlgebra` and `Krylov.jl` (via
`AMReXKrylovExt`). Elementwise operations execute on local rank DOFs while inner products
(`LinearAlgebra.dot`, `LinearAlgebra.norm`, `Krylov.kdot`, `Krylov.knorm`) accumulate across
AMReX's MPI communicator via `parallel_reduce_sum_real`.
"""
struct MPIDistVector <: AbstractVector{Float64}
    data::Vector{Float64}
end

MPIDistVector(::UndefInitializer, n::Integer) = MPIDistVector(Vector{Float64}(undef, n))

Base.size(v::MPIDistVector) = size(v.data)
Base.length(v::MPIDistVector) = length(v.data)
Base.IndexStyle(::Type{MPIDistVector}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(v::MPIDistVector, i::Int) = v.data[i]
Base.@propagate_inbounds Base.setindex!(v::MPIDistVector, x, i::Int) = (v.data[i] = x)
Base.similar(v::MPIDistVector, ::Type{Float64}, dims::Dims{1}) = MPIDistVector(Vector{Float64}(undef, dims[1]))
Base.similar(v::MPIDistVector, ::Type{T}, dims::Dims) where {T} = similar(v.data, T, dims)
Base.copy(v::MPIDistVector) = MPIDistVector(copy(v.data))
Base.copyto!(dst::MPIDistVector, src::MPIDistVector) = (copyto!(dst.data, src.data); dst)
Base.copyto!(dst::MPIDistVector, src::AbstractVector) = (copyto!(dst.data, src); dst)
Base.fill!(v::MPIDistVector, x) = (fill!(v.data, x); v)

Base.BroadcastStyle(::Type{MPIDistVector}) = Base.Broadcast.ArrayStyle{MPIDistVector}()
function Base.similar(
        bc::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{MPIDistVector}},
        ::Type{ElType},
    ) where {ElType}
    ax = axes(bc)
    if ElType === Float64 && length(ax) == 1
        return MPIDistVector(Vector{Float64}(undef, length(ax[1])))
    end
    return Array{ElType}(undef, length.(ax)...)
end

@inline function _mpi_sum(x::Float64)::Float64
    return parallel_nprocs() > 1 ? Float64(parallel_reduce_sum_real(x)) : x
end

LinearAlgebra.dot(a::MPIDistVector, b::MPIDistVector) = _mpi_sum(dot(a.data, b.data))
LinearAlgebra.norm(a::MPIDistVector) = sqrt(_mpi_sum(dot(a.data, a.data)))
