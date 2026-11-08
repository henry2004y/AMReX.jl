# ============================================================================
# Plotfile, HDF5 & VisMF Checkpoint/Restart High-Level Wrappers
# ============================================================================

function _stage_multilevel_plotfile!(mfs::AbstractVector{<:JlMultiFab},
                                     geoms::AbstractVector{<:JlGeometry},
                                     step::Union{Integer, AbstractVector{<:Integer}},
                                     ref_ratio::Union{NTuple{3, Integer}, AbstractVector{<:NTuple{3, Integer}}})
    nlevs = length(mfs)
    length(geoms) == nlevs || throw(DimensionMismatch("length(mfs) == $nlevs != length(geoms) == $(length(geoms))"))
    plotfile_stage_clear!()
    for l in 1:nlevs
        st = step isa Integer ? Int32(step) : Int32(step[l])
        if l < nlevs
            rr = ref_ratio isa Tuple ? ref_ratio : ref_ratio[l]
            plotfile_stage_push_level!(mfs[l], geoms[l], st, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
        else
            plotfile_stage_push_level!(mfs[l], geoms[l], st, Int32(0), Int32(0), Int32(0))
        end
    end
    return nothing
end

"""
    write_plotfile(filename::AbstractString, mf::JlMultiFab, geom::JlGeometry;
                   varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0)

Write a single-level AMReX plotfile.
"""
function write_plotfile(filename::AbstractString, mf::JlMultiFab, geom::JlGeometry;
                        varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0)
    csv = join(varnames, ",")
    write_single_level_plotfile(String(filename), mf, csv, geom, Float64(time), Int32(step))
    return filename
end

"""
    write_plotfile(filename::AbstractString,
                   mfs::AbstractVector{<:JlMultiFab},
                   geoms::AbstractVector{<:JlGeometry};
                   varnames::Vector{String}=String[], time::Real=0.0,
                   step::Union{Integer, AbstractVector{<:Integer}}=0,
                   ref_ratio::Union{NTuple{3, Integer}, AbstractVector{<:NTuple{3, Integer}}}=(2, 2, 2))

Write an arbitrary `N`-level (`N >= 1`) AMReX plotfile into a single hierarchy directory `filename`
containing `Level_0` through `Level_{N-1}`.
"""
function write_plotfile(filename::AbstractString,
                        mfs::AbstractVector{<:JlMultiFab},
                        geoms::AbstractVector{<:JlGeometry};
                        varnames::Vector{String}=String[], time::Real=0.0,
                        step::Union{Integer, AbstractVector{<:Integer}}=0,
                        ref_ratio::Union{NTuple{3, Integer}, AbstractVector{<:NTuple{3, Integer}}}=(2, 2, 2))
    isempty(mfs) && throw(ArgumentError("mfs must not be empty"))
    csv = join(varnames, ",")
    if length(mfs) == 1
        st = step isa Integer ? Int32(step) : Int32(step[1])
        write_single_level_plotfile(String(filename), mfs[1], csv, geoms[1], Float64(time), st)
    else
        _stage_multilevel_plotfile!(mfs, geoms, step, ref_ratio)
        write_staged_multilevel_plotfile(String(filename), csv, Float64(time))
    end
    return filename
end

"""
    write_plotfile_hdf5(filename::AbstractString, mf::JlMultiFab, geom::JlGeometry;
                        varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0,
                        compression::AbstractString="None@0")

Write a single-level HDF5-backed AMReX plotfile (`WriteSingleLevelPlotfileHDF5`).
"""
function write_plotfile_hdf5(filename::AbstractString, mf::JlMultiFab, geom::JlGeometry;
                             varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0,
                             compression::AbstractString="None@0")
    csv = join(varnames, ",")
    write_single_level_plotfile_hdf5(String(filename), mf, csv, geom, Float64(time), Int32(step), String(compression))
    return filename
end

"""
    write_plotfile_hdf5(filename::AbstractString,
                        mfs::AbstractVector{<:JlMultiFab},
                        geoms::AbstractVector{<:JlGeometry};
                        varnames::Vector{String}=String[], time::Real=0.0,
                        step::Union{Integer, AbstractVector{<:Integer}}=0,
                        ref_ratio::Union{NTuple{3, Integer}, AbstractVector{<:NTuple{3, Integer}}}=(2, 2, 2),
                        compression::AbstractString="None@0")

Write an arbitrary `N`-level (`N >= 1`) HDF5-backed AMReX plotfile (`WriteMultiLevelPlotfileHDF5`).
"""
function write_plotfile_hdf5(filename::AbstractString,
                             mfs::AbstractVector{<:JlMultiFab},
                             geoms::AbstractVector{<:JlGeometry};
                             varnames::Vector{String}=String[], time::Real=0.0,
                             step::Union{Integer, AbstractVector{<:Integer}}=0,
                             ref_ratio::Union{NTuple{3, Integer}, AbstractVector{<:NTuple{3, Integer}}}=(2, 2, 2),
                             compression::AbstractString="None@0")
    isempty(mfs) && throw(ArgumentError("mfs must not be empty"))
    csv = join(varnames, ",")
    if length(mfs) == 1
        st = step isa Integer ? Int32(step) : Int32(step[1])
        write_single_level_plotfile_hdf5(String(filename), mfs[1], csv, geoms[1], Float64(time), st, String(compression))
    else
        _stage_multilevel_plotfile!(mfs, geoms, step, ref_ratio)
        write_staged_multilevel_plotfile_hdf5(String(filename), csv, Float64(time), String(compression))
    end
    return filename
end

"""
    write_plotfile_hdf5(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles";
                        compression::AbstractString="None@0")

Write particle plotfile data in HDF5 format (`ParticleContainer::WritePlotFileHDF5`).
"""
function write_plotfile_hdf5(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles";
                             compression::AbstractString="None@0")
    write_plotfile_hdf5(pc, String(dir), String(name), String(compression))
    return dir
end

"""
    checkpoint_hdf5(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles";
                    compression::AbstractString="None@0")

Write a particle checkpoint in HDF5 format (`ParticleContainer::CheckpointHDF5`).
"""
function checkpoint_hdf5(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles";
                         compression::AbstractString="None@0")
    checkpoint_hdf5(pc, String(dir), String(name), String(compression))
    return dir
end

write_plotfile(pc::JlParticleContainer, dir::AbstractString) = write_plotfile(pc, String(dir), "particles")
checkpoint(pc::JlParticleContainer, dir::AbstractString) = checkpoint(pc, String(dir), "particles")

"""
    restart!(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles")

Restart particles in `pc` from an AMReX particle checkpoint directory `joinpath(dir, name)`.
"""
restart!(pc::JlParticleContainer, dir::AbstractString) = restart!(pc, String(dir), "particles")

"""
    restart_hdf5!(pc::JlParticleContainer, dir::AbstractString, name::AbstractString="particles")

Restart particles in `pc` from an HDF5 particle checkpoint in `dir`.
"""
restart_hdf5!(pc::JlParticleContainer, dir::AbstractString) = restart_hdf5!(pc, String(dir), "particles")

