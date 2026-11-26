module AMReXCUDAExt

using AMReX
using CUDA
using KernelAbstractions

"""
    AMReX.device_array4(mf::AMReX.JlMultiFab, mfi::Union{AMReX.JlMFIter, AMReX.MFIterHandle, AMReX.TileHandle})

Wrap the device-resident `FArrayBox` pointer of `mf` at `mfi` as a zero-copy `(CuDeviceArray, lo)`
pair when running against a GPU-enabled AMReX build (`is_gpu_build() == true`).
"""
@inline function AMReX.device_array4(
        mf::AMReX.JlMultiFab,
        mfi::Union{AMReX.JlMFIter, AMReX.MFIterHandle, AMReX.TileHandle}
    )
    AMReX.is_gpu_build() || throw(ErrorException("AMReX.device_array4 requires a GPU-enabled AMReX build (see AMReX.set_amrex_prefix!)"))
    v = AMReX.array4(mf, mfi)
    arr = unsafe_wrap(CuArray, CuPtr{Float64}(UInt(v.ptr)), v.dims; own = false)
    return (arr, v.lo)
end

"""
    AMReX.sync_amrex_gpu_stream!()

Synchronize the active CUDA stream with AMReX GPU work.
"""
@inline function AMReX.sync_amrex_gpu_stream!()
    CUDA.synchronize()
    return nothing
end

end # module AMReXCUDAExt
