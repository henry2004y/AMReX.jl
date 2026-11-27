"""
    AMReX

High-performance Julia bindings for the [AMReX](https://amrex-codes.github.io/amrex/)
block-structured Adaptive Mesh Refinement (AMR) framework, powered by `CxxWrap.jl`,
`AMReX_jll`, and zero-allocation C-ABI views.

Provides zero-allocation 4D `Array4View`s over `MultiFab`/`iMultiFab` `FArrayBox`es,
zero-allocation `PtrVector` views over pure Structure-of-Arrays (SoA) `ParticleContainer`
tiles, multi-threaded tile iterators, multi-level `AmrCore` mesh hierarchies, ABI-portable
MPI primitives, and `Preferences.jl`/`Scratch.jl`-configurable CPU/GPU backends.
"""
module AMReX

using AMReX_jll
using CxxWrap
using Libdl
using PrecompileTools: @setup_workload, @compile_workload
using Preferences
using Scratch

const _SHIM_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

# Cached C-ABI function pointers populated in __init__() for zero-allocation @ccall dispatch
const _FN_IS_GPU_BUILD = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_IS_DEVICE_ACCESSIBLE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_BOX_BOUNDS = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_JLMFI_RAW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_ACQUIRE_MF = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_ACQUIRE_IMF = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_ACQUIRE_TB = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_ACQUIRE_TB_RAW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_RELEASE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_IS_VALID = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_NEXT = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_INDEX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_LOCAL_TILE_INDEX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_VALID_BOX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_TILE_BOX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MFITER_FAB_BOX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FAB_VIEW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_IFAB_VIEW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TFAB_VIEW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TFAB_VIEW_RAW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FAB_VIEW_BY_GID = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_IFAB_VIEW_BY_GID = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_YAFR_CRSE_HAS_WORK = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_YAFR_FINE_HAS_WORK = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_YAFR_CRSE_ADD = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_YAFR_FINE_ADD = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_EFR_CRSE_ADD = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_EFR_FINE_ADD = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_JLPTI_RAW = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_ACQUIRE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_RELEASE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_IS_VALID = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_NEXT = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_INDEX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_LOCAL_TILE_INDEX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_NUM_PARTICLES = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_VALID_BOX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_TILE_BOX = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_RESIZE_TILE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_IDCPU_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_POS_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_REAL_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_INT_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_PARITER_SOA_INFO = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_SOA_INFO = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_NUM_PARTICLES = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_IDCPU_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_POS_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_REAL_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_TILE_INT_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_REQUEST_SIZEOF = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_IRECV_BYTES = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_ISEND_BYTES = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_WAITALL = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_ALLGATHER_BYTES = Ref{Ptr{Cvoid}}(C_NULL)

const _MPI_REQ_SIZE = Ref{Int}(0)

function _fnv1a64_update(h::UInt64, bytes::AbstractVector{UInt8})
    for b in bytes
        h = xor(h, UInt64(b))
        h *= 0x00000100000001b3
    end
    return h
end

function _shim_scratch_dir(amrex_prefix::Union{Nothing, String})
    h = 0xcbf29ce484222325
    amrex_dir = amrex_prefix !== nothing ? amrex_prefix : (AMReX_jll.is_available() ? AMReX_jll.artifact_dir : "")
    cxx = get(ENV, "CXX", "g++")
    cxxwrap_dir = CxxWrap.prefix_path()
    meta = "$(VERSION.major).$(VERSION.minor)|$(amrex_dir)|$(cxxwrap_dir)|$(cxx)|$(Sys.ARCH)|$(Sys.KERNEL)|"
    h = _fnv1a64_update(h, codeunits(meta))
    for relpath in ("deps/cshim/amrex_jl_shim.cpp", "src/build_cshim.jl")
        p = joinpath(@__DIR__, "..", relpath)
        isfile(p) && (h = _fnv1a64_update(h, read(p)))
    end
    key = "cshim-jl$(VERSION.major).$(VERSION.minor)-$(string(h; base = 16, pad = 16))"
    return @get_scratch!(key)
end

include("build_cshim.jl")

function get_shim_library_path()
    custom_lib = @load_preference("custom_shim_lib", nothing)
    if custom_lib !== nothing && isfile(custom_lib)
        return custom_lib
    end
    amrex_prefix = @load_preference("amrex_prefix", nothing)
    scratch_dir = _shim_scratch_dir(amrex_prefix)
    lib_path = joinpath(scratch_dir, "libamrex_jl_shim." * Libdl.dlext)
    if !isfile(lib_path)
        build_cshim(; custom_amrex_prefix = amrex_prefix, out_dir = scratch_dir)
    end
    return lib_path
end

"""
    set_amrex_prefix!(amrex_prefix::Union{Nothing, String})

Configure `AMReX.jl` via `Preferences.jl` to use either the default `AMReX_jll` CPU
binary (`nothing`) or a custom local AMReX installation (e.g., a CUDA-enabled `libamrex`
build), and recompile the C++ shim library into `Scratch.jl` storage.
"""
function set_amrex_prefix!(amrex_prefix::Union{Nothing, String})
    @set_preferences!("amrex_prefix" => amrex_prefix)
    scratch_dir = _shim_scratch_dir(amrex_prefix)
    build_cshim(; custom_amrex_prefix = amrex_prefix, out_dir = scratch_dir)
    return @info "Updated AMReX prefix preference to $(something(amrex_prefix, "AMReX_jll default")). Restart Julia for changes to take effect."
end

import Base: length, size, contains

@wrapmodule(get_shim_library_path)

function _resolve_c_symbols!(h::Ptr{Cvoid})
    _FN_IS_GPU_BUILD[] = Libdl.dlsym(h, :amrex_jl_is_gpu_build)
    _FN_IS_DEVICE_ACCESSIBLE[] = Libdl.dlsym(h, :amrex_jl_is_device_accessible)
    _FN_BOX_BOUNDS[] = Libdl.dlsym(h, :amrex_jl_box_bounds)
    _FN_JLMFI_RAW[] = Libdl.dlsym(h, :amrex_jl_jlmfi_raw)
    _FN_MFITER_ACQUIRE_MF[] = Libdl.dlsym(h, :amrex_jl_mfiter_acquire_mf)
    _FN_MFITER_ACQUIRE_IMF[] = Libdl.dlsym(h, :amrex_jl_mfiter_acquire_imf)
    _FN_MFITER_ACQUIRE_TB[] = Libdl.dlsym(h, :amrex_jl_mfiter_acquire_tb)
    _FN_MFITER_ACQUIRE_TB_RAW[] = Libdl.dlsym(h, :amrex_jl_mfiter_acquire_tb_raw)
    _FN_MFITER_RELEASE[] = Libdl.dlsym(h, :amrex_jl_mfiter_release)
    _FN_MFITER_IS_VALID[] = Libdl.dlsym(h, :amrex_jl_mfiter_is_valid)
    _FN_MFITER_NEXT[] = Libdl.dlsym(h, :amrex_jl_mfiter_next)
    _FN_MFITER_INDEX[] = Libdl.dlsym(h, :amrex_jl_mfiter_index)
    _FN_MFITER_LOCAL_TILE_INDEX[] = Libdl.dlsym(h, :amrex_jl_mfiter_local_tile_index)
    _FN_MFITER_VALID_BOX[] = Libdl.dlsym(h, :amrex_jl_mfiter_valid_box)
    _FN_MFITER_TILE_BOX[] = Libdl.dlsym(h, :amrex_jl_mfiter_tile_box)
    _FN_MFITER_FAB_BOX[] = Libdl.dlsym(h, :amrex_jl_mfiter_fab_box)
    _FN_FAB_VIEW[] = Libdl.dlsym(h, :amrex_jl_fab_view)
    _FN_IFAB_VIEW[] = Libdl.dlsym(h, :amrex_jl_ifab_view)
    _FN_TFAB_VIEW[] = Libdl.dlsym(h, :amrex_jl_tfab_view)
    _FN_TFAB_VIEW_RAW[] = Libdl.dlsym(h, :amrex_jl_tfab_view_raw)
    _FN_FAB_VIEW_BY_GID[] = Libdl.dlsym(h, :amrex_jl_fab_view_by_gid)
    _FN_IFAB_VIEW_BY_GID[] = Libdl.dlsym(h, :amrex_jl_ifab_view_by_gid)
    _FN_YAFR_CRSE_HAS_WORK[] = Libdl.dlsym(h, :amrex_jl_yafr_crse_has_work)
    _FN_YAFR_FINE_HAS_WORK[] = Libdl.dlsym(h, :amrex_jl_yafr_fine_has_work)
    _FN_YAFR_CRSE_ADD[] = Libdl.dlsym(h, :amrex_jl_yafr_crse_add)
    _FN_YAFR_FINE_ADD[] = Libdl.dlsym(h, :amrex_jl_yafr_fine_add)
    _FN_EFR_CRSE_ADD[] = Libdl.dlsym(h, :amrex_jl_efr_crse_add)
    _FN_EFR_FINE_ADD[] = Libdl.dlsym(h, :amrex_jl_efr_fine_add)
    _FN_JLPTI_RAW[] = Libdl.dlsym(h, :amrex_jl_jlpti_raw)
    _FN_PARITER_ACQUIRE[] = Libdl.dlsym(h, :amrex_jl_pariter_acquire)
    _FN_PARITER_RELEASE[] = Libdl.dlsym(h, :amrex_jl_pariter_release)
    _FN_PARITER_IS_VALID[] = Libdl.dlsym(h, :amrex_jl_pariter_is_valid)
    _FN_PARITER_NEXT[] = Libdl.dlsym(h, :amrex_jl_pariter_next)
    _FN_PARITER_INDEX[] = Libdl.dlsym(h, :amrex_jl_pariter_index)
    _FN_PARITER_LOCAL_TILE_INDEX[] = Libdl.dlsym(h, :amrex_jl_pariter_local_tile_index)
    _FN_PARITER_NUM_PARTICLES[] = Libdl.dlsym(h, :amrex_jl_pariter_num_particles)
    _FN_PARITER_VALID_BOX[] = Libdl.dlsym(h, :amrex_jl_pariter_valid_box)
    _FN_PARITER_TILE_BOX[] = Libdl.dlsym(h, :amrex_jl_pariter_tile_box)
    _FN_PARITER_RESIZE_TILE[] = Libdl.dlsym(h, :amrex_jl_pariter_resize_tile)
    _FN_PARITER_IDCPU_PTR[] = Libdl.dlsym(h, :amrex_jl_pariter_idcpu_ptr)
    _FN_PARITER_POS_PTR[] = Libdl.dlsym(h, :amrex_jl_pariter_pos_ptr)
    _FN_PARITER_REAL_PTR[] = Libdl.dlsym(h, :amrex_jl_pariter_real_ptr)
    _FN_PARITER_INT_PTR[] = Libdl.dlsym(h, :amrex_jl_pariter_int_ptr)
    _FN_PARITER_SOA_INFO[] = Libdl.dlsym(h, :amrex_jl_pariter_soa_info)
    _FN_TILE_SOA_INFO[] = Libdl.dlsym(h, :amrex_jl_tile_soa_info)
    _FN_TILE_NUM_PARTICLES[] = Libdl.dlsym(h, :amrex_jl_tile_num_particles)
    _FN_TILE_IDCPU_PTR[] = Libdl.dlsym(h, :amrex_jl_tile_idcpu_ptr)
    _FN_TILE_POS_PTR[] = Libdl.dlsym(h, :amrex_jl_tile_pos_ptr)
    _FN_TILE_REAL_PTR[] = Libdl.dlsym(h, :amrex_jl_tile_real_ptr)
    _FN_TILE_INT_PTR[] = Libdl.dlsym(h, :amrex_jl_tile_int_ptr)
    _FN_MPI_REQUEST_SIZEOF[] = Libdl.dlsym(h, :amrex_jl_mpi_request_sizeof)
    _FN_MPI_IRECV_BYTES[] = Libdl.dlsym(h, :amrex_jl_mpi_irecv_bytes)
    _FN_MPI_ISEND_BYTES[] = Libdl.dlsym(h, :amrex_jl_mpi_isend_bytes)
    _FN_MPI_WAITALL[] = Libdl.dlsym(h, :amrex_jl_mpi_waitall)
    _FN_MPI_ALLGATHER_BYTES[] = Libdl.dlsym(h, :amrex_jl_mpi_allgather_bytes)
    _MPI_REQ_SIZE[] = Int(@ccall $(_FN_MPI_REQUEST_SIZEOF[])()::Int32)
    return nothing
end

include("box_geometry.jl")
include("multifab.jl")
include("amrcore.jl")
include("flux_register.jl")
include("mlmg.jl")
include("particles.jl")
include("io.jl")

function __init__()
    # Only load AMReX_jll library directories when using the default JLL build
    if @load_preference("amrex_prefix", nothing) === nothing
        for libdir in AMReX_jll.LIBPATH_list
            if isdir(libdir)
                push!(Libdl.DL_LOAD_PATH, libdir)
            end
        end
    end
    @initcxx
    shim_path = get_shim_library_path()
    h = Libdl.dlopen(shim_path, Libdl.RTLD_LOCAL | Libdl.RTLD_LAZY)
    _SHIM_HANDLE[] = h
    _resolve_c_symbols!(h)
    _ERROR_EST_TRAMPOLINE[] = @cfunction(_amrex_error_est_trampoline, Cvoid, (UInt64, Cint, UInt64, Cdouble))
    return nothing
end

"""
    is_gpu_build() -> Bool

Return `true` if the active `AMReX` C++ shim was compiled against a GPU-enabled AMReX (`AMREX_USE_GPU`).
"""
@inline is_gpu_build()::Bool = (@ccall $(_FN_IS_GPU_BUILD[])()::UInt8) != 0

"""
    is_device_accessible(mf::JlMultiFab) -> Bool

Return `true` when `mf` resides in GPU/device-accessible AMReX memory under an active GPU launch region.
"""
@inline is_device_accessible(mf::JlMultiFab)::Bool =
    (@ccall $(_FN_IS_DEVICE_ACCESSIBLE[])(mf.cpp_object::Ptr{Cvoid})::UInt8) != 0

# Extension hooks implemented by AMReXCUDAExt when CUDA.jl is loaded
function device_array4 end
function sync_amrex_gpu_stream! end

# ABI-portable MPI point-to-point & collective helpers on AMReX's communicator
const PARALLEL_ANY_SOURCE = Int32(-1)

"""
    MPIRequestBuffer(capacity::Integer = 0)

Opaque buffer storing `MPI_Request` handles sized dynamically to the active MPI ABI's
`sizeof(MPI_Request)`. Reused across steps for zero-allocation non-blocking exchanges.
"""
mutable struct MPIRequestBuffer
    bytes::Vector{UInt8}
    req_size::Int
end

function MPIRequestBuffer(capacity::Integer = 0)
    rs = max(1, _MPI_REQ_SIZE[])
    return MPIRequestBuffer(Vector{UInt8}(undef, Int(capacity) * rs), rs)
end

"""
    ensure_requests!(buf::MPIRequestBuffer, n::Integer) -> MPIRequestBuffer

Resize `buf` if needed to hold at least `n` `MPI_Request` entries without shrinking.
"""
@inline function ensure_requests!(buf::MPIRequestBuffer, n::Integer)
    needed = Int(n) * buf.req_size
    if length(buf.bytes) < needed
        resize!(buf.bytes, needed)
    end
    return buf
end

"""
    parallel_irecv!(buf::AbstractArray{T}, src_rank::Integer, tag::Integer,
                    reqs::MPIRequestBuffer, req_idx::Integer; n::Integer = length(buf))

Post a non-blocking `MPI_Irecv` on AMReX's MPI communicator for `n` elements of `T` into `buf`
at 1-based request slot `req_idx`. Pass `PARALLEL_ANY_SOURCE` (`-1`) to accept any source rank.
"""
@inline function parallel_irecv!(
        buf::AbstractArray{T}, src_rank::Integer, tag::Integer,
        reqs::MPIRequestBuffer, req_idx::Integer;
        n::Integer = length(buf)
    ) where {T}
    isbitstype(T) || throw(ArgumentError("parallel_irecv! requires an isbits element type, got $T"))
    nbytes = Int64(n) * Int64(sizeof(T))
    GC.@preserve buf reqs begin
        @ccall $(_FN_MPI_IRECV_BYTES[])(
            pointer(buf)::Ptr{Cvoid}, nbytes::Int64, Int32(src_rank)::Int32, Int32(tag)::Int32,
            pointer(reqs.bytes)::Ptr{Cvoid}, Int32(req_idx - 1)::Int32
        )::Cvoid
    end
    return nothing
end

"""
    parallel_isend!(buf::AbstractArray{T}, dst_rank::Integer, tag::Integer,
                    reqs::MPIRequestBuffer, req_idx::Integer; n::Integer = length(buf))

Post a non-blocking `MPI_Isend` on AMReX's MPI communicator for `n` elements of `T` from `buf`
at 1-based request slot `req_idx`.
"""
@inline function parallel_isend!(
        buf::AbstractArray{T}, dst_rank::Integer, tag::Integer,
        reqs::MPIRequestBuffer, req_idx::Integer;
        n::Integer = length(buf)
    ) where {T}
    isbitstype(T) || throw(ArgumentError("parallel_isend! requires an isbits element type, got $T"))
    nbytes = Int64(n) * Int64(sizeof(T))
    GC.@preserve buf reqs begin
        @ccall $(_FN_MPI_ISEND_BYTES[])(
            pointer(buf)::Ptr{Cvoid}, nbytes::Int64, Int32(dst_rank)::Int32, Int32(tag)::Int32,
            pointer(reqs.bytes)::Ptr{Cvoid}, Int32(req_idx - 1)::Int32
        )::Cvoid
    end
    return nothing
end

"""
    parallel_waitall!(reqs::MPIRequestBuffer, count::Integer)

Complete the first `count` non-blocking requests in `reqs` via `MPI_Waitall`.
"""
@inline function parallel_waitall!(reqs::MPIRequestBuffer, count::Integer)
    c = Int32(count)
    c <= 0 && return nothing
    GC.@preserve reqs begin
        @ccall $(_FN_MPI_WAITALL[])(c::Int32, pointer(reqs.bytes)::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

"""
    parallel_allgather!(recvbuf::AbstractArray{T}, sendbuf::AbstractArray{T}; n_per_rank::Integer = length(sendbuf))

Gather `n_per_rank` elements of `isbits` type `T` from every rank in `AMReX`'s communicator into `recvbuf`.
"""
@inline function parallel_allgather!(
        recvbuf::AbstractArray{T}, sendbuf::AbstractArray{T};
        n_per_rank::Integer = length(sendbuf)
    ) where {T}
    isbitstype(T) || throw(ArgumentError("parallel_allgather! requires an isbits element type, got $T"))
    nbytes = Int64(n_per_rank) * Int64(sizeof(T))
    GC.@preserve sendbuf recvbuf begin
        @ccall $(_FN_MPI_ALLGATHER_BYTES[])(
            pointer(sendbuf)::Ptr{Cvoid}, pointer(recvbuf)::Ptr{Cvoid}, nbytes::Int64
        )::Cvoid
    end
    return recvbuf
end

"""
    amrex_init(verbose::Bool=false)

Initialize the AMReX runtime and MPI communicator if not already initialized.
"""
amrex_init() = amrex_init(false)

"""
    amrex_initialized() -> Bool

Return `true` if the AMReX runtime is currently initialized.
"""
function amrex_initialized end

"""
    amrex_finalize()

Finalize the AMReX runtime and release global AMReX arenas and resources.
"""
function amrex_finalize end

export amrex_init, amrex_initialized, amrex_finalize
export is_gpu_build, is_device_accessible, device_array4, sync_amrex_gpu_stream!
export parallel_myproc, parallel_nprocs, parallel_ioprocessor, parallel_barrier
export parallel_reduce_sum_real, parallel_reduce_max_real, parallel_reduce_min_real
export parallel_reduce_sum_long, parallel_reduce_sum_real_vec!
export PARALLEL_ANY_SOURCE, MPIRequestBuffer, ensure_requests!, parallel_irecv!, parallel_isend!, parallel_waitall!, parallel_allgather!

export JlBox, JlBoxAllocated, Box, IndexBox, bounds, lo, hi, is_nodal, num_pts, is_empty, contains
export surrounding_nodes, enclosed_cells, grow, coarsen, refine, shift, intersects, intersect
export JlRealBox, JlRealBoxAllocated, RealBox
export JlGeometry, JlGeometryAllocated, Geometry, cell_size, inv_cell_size, prob_lo, prob_hi, domain, prob_domain, is_periodic
export cell_sizes, inv_cell_sizes, prob_lo_tuple, prob_hi_tuple
export JlBoxArray, JlBoxArrayAllocated, BoxArray, max_size!, convert_to_nodal, nodal_box_array, get_box, contains_cell, coarsenable
export JlDistributionMapping, JlDistributionMappingAllocated, DistributionMapping, get_rank, make_knapsack_dmap, make_sfc_dmap

export JlMultiFab, JlMultiFabAllocated, MultiFab, JliMultiFab, JliMultiFabAllocated, iMultiFab
export JlMFIter, JlMFIterAllocated, MFIterHandle, TileHandle, Array4View
export n_comp, n_grow, local_size, box_array, dmap
export set_val!, set_val_comp!, plus_val!, mult_val!, add_from!, copy_from!, parallel_copy_from!, parallel_copy!
export copy!, plus!, scale!, saxpy!, lincomb!, dot_mf
export fill_boundary!, fill_boundary_comp!, fill_boundary_nowait!, fill_boundary_finish!, sum_boundary!, sum_boundary_comp!
export min_val, max_val, norm0, norm1, norm2, sum_val
export is_valid, next!, close!, index, grid_index, local_tile_index, valid_box, tile_box, fab_box
export fab_lo, fab_hi, ifab_lo, ifab_hi, tfab_lo, tfab_hi, array4, iarray4, tag_array4, each_fab, each_fab_threaded

export JlAmrCore, JlAmrCoreAllocated, AmrCore, InterpType, CellConservativeLinear, NodeBilinear, PiecewiseConstant, CellQuadratic
export MathBCType, BCBogus, BCReflectOdd, BCIntDir, BCReflectEven, BCFOExtrap, BCExtDir, BCHOExtrap, BCHOExtrapCC, BCExtDirCC
export JlBCRec, JlBCRecAllocated, BCRec, set_lo!, set_hi!, fill_domain_boundary!
export JlTagBoxArray, JlTagBoxArrayAllocated, TagBoxArray, TagBoxArrayHandle, TagVal, TagClear, TagBuf, TagSet
export add_refine_box!, add_tag_box!, clear_refine_boxes!, set_tag_mask!, set_tag_boxes!, clear_tag_masks!
export on_error_est!, tag_cells_by_value!, tag_cells_by_grad!, init_from_scratch!, regrid_levels!
export finest_level, max_level, geom, geometry, dmap, dist_map, node_box_array, ref_ratio, ref_ratio_tuple
export make_fine_mask, coarsened_multifab
export mpi_rank, parallel_reduce_sum
export set_box_array!, set_dmap!, set_finest_level!
export interp_from_coarse_to_fine!, fill_patch_single_level!, fill_patch_two_levels!
export average_down!, average_down_nodal!, average_down_faces!, average_down_face!, average_down_edges!, sum_fine_to_coarse_nodal!
export JlYAFluxRegister, JlYAFluxRegisterAllocated, YAFluxRegister
export JlEdgeFluxRegister, JlEdgeFluxRegisterAllocated, EdgeFluxRegister
export reset!, crse_add!, fine_add!, reflux!, crse_has_work, fine_has_work

export LinOpBCType, LOInterior, LODirichlet, LONeumann, LOReflectOdd, LOMarshak, LOSanchezPomraning
export LOInflow, LOInhomogNeumann, LORobin, LOSymmetry, LOPeriodic, LOBogus
export BottomSolver, BottomDefault, BottomSmoother, BottomBiCGStab, BottomCG, BottomBiCGCG, BottomCGBiCG, BottomHypre, BottomPETSc
export MLMGLocation, LocFaceCenter, LocFaceCentroid, LocCellCenter, LocCellCentroid
export JlLPInfo, JlLPInfoAllocated, LPInfo
export set_max_coarsening_level!, set_agglomeration!, set_consolidation!, set_agg_grid_size!, set_con_grid_size!, set_metric_term!, set_hidden_direction!
export JlMLPoisson, JlMLPoissonAllocated, MLPoisson
export JlMLABecLaplacian, JlMLABecLaplacianAllocated, MLABecLaplacian
export JlMLNodeLaplacian, JlMLNodeLaplacianAllocated, MLNodeLaplacian
export JlMLMG, JlMLMGAllocated, MLMG
export set_domain_bc!, set_domain_bc_loc!, set_level_bc!, set_level_bc_homog!, set_level_bc_robin!, set_coarse_fine_bc!, set_coarse_fine_bc_homog!
export set_max_order!, get_max_order, set_verbose!, set_enforce_singular_solvable!
export set_scalars!, set_a_coeffs!, set_a_coeffs_scalar!, set_b_coeffs!, set_b_coeffs_scalar!
export set_sigma!, set_gauss_seidel!, set_harmonic_average!, comp_divergence!, update_velocity!
export set_bottom_verbose!, set_max_iter!, set_max_fmg_iter!, set_fixed_iter!, set_pre_smooth!, set_post_smooth!, set_final_smooth!, set_bottom_smooth!
export set_bottom_solver!, set_bottom_max_iter!, set_bottom_tolerance!, set_bottom_tolerance_abs!, set_final_fill_bc!
export num_amr_levels, get_init_residual, get_final_residual, get_init_rhs
export solve!, get_grad_solution!, get_fluxes!, comp_residual!, apply!

export JlParticleContainer, JlParticleContainerAllocated, ParticleContainer
export JlParIter, JlParIterAllocated, ParIterHandle, PtrVector, SoATileViews
export pack_idcpu, unpack_id, unpack_cpu, is_valid_idcpu, make_invalid_idcpu, make_valid_idcpu
export next_particle_id, reserve_and_resize_data!, ensure_tile!, resize_tile!, tile_num_particles, num_particles
export idcpu_vec, pos_vecs, real_vec, int_vec, soa_views
export tile_idcpu_vec, tile_pos_vecs, tile_real_vec, tile_int_vec, tile_soa_views
export each_particle_tile, each_particle_tile_threaded, redistribute!, total_number_of_particles, number_of_particles_at_level, clear_particles!
export write_plotfile, write_plotfile_hdf5, checkpoint, checkpoint_hdf5, restart!, restart_hdf5!
export vismf_write, vismf_read!, vismf_exist, set_amrex_prefix!

@setup_workload begin
    @compile_workload begin
        w = pack_idcpu(Int64(42), 0)
        _ = unpack_id(w)
        _ = unpack_cpu(w)
        _ = is_valid_idcpu(w)
        w_inv = make_invalid_idcpu(w)
        _ = make_valid_idcpu(w_inv)

        buf = Vector{Float64}(undef, 8)
        fill!(buf, 1.0)
        GC.@preserve buf begin
            v4 = Array4View{Float64}(pointer(buf), (0, 0, 0), (2, 2, 2, 1))
            v4[0, 0, 0, 1] = 2.0
            _ = v4[0, 0, 0, 1]
            pv = PtrVector{Float64}(pointer(buf), 8)
            pv[1] = 3.0
            _ = pv[1]
        end
        ibx = IndexBox((0, 0, 0), (1, 1, 1), (false, false, false))
        _ = bounds(ibx)
    end
end

end # module AMReX
