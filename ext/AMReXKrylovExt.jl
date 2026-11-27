module AMReXKrylovExt

using AMReX: MPIDistVector, _mpi_sum
using Krylov: Krylov

Krylov.kdot(n::Integer, x::MPIDistVector, y::MPIDistVector) = _mpi_sum(Krylov.kdot(n, x.data, y.data))
Krylov.knorm(n::Integer, x::MPIDistVector) = sqrt(_mpi_sum(Krylov.kdot(n, x.data, x.data)))
Krylov.kaxpy!(n::Integer, s::Float64, x::MPIDistVector, y::MPIDistVector) = (Krylov.kaxpy!(n, s, x.data, y.data); y)
Krylov.kaxpby!(n::Integer, s::Float64, x::MPIDistVector, t::Float64, y::MPIDistVector) =
    (Krylov.kaxpby!(n, s, x.data, t, y.data); y)
Krylov.kscal!(n::Integer, s::Float64, x::MPIDistVector) = (Krylov.kscal!(n, s, x.data); x)
Krylov.kcopy!(n::Integer, y::MPIDistVector, x::MPIDistVector) = (Krylov.kcopy!(n, y.data, x.data); y)
Krylov.kfill!(x::MPIDistVector, val::Float64) = (Krylov.kfill!(x.data, val); x)
Krylov.kdiv!(n::Integer, x::MPIDistVector, s::Float64) = (Krylov.kdiv!(n, x.data, s); x)

end # module AMReXKrylovExt
