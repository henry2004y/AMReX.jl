# ============================================================================
# Plotfile & Checkpoint I/O High-Level Wrappers
# ============================================================================

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
    write_plotfile(filename::AbstractString, mfs::Vector{JlMultiFab}, geoms::Vector{JlGeometry};
                   varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0,
                   ref_ratio::NTuple{3, Integer}=(2, 2, 2))

Write a single-level or two-level AMReX plotfile.
"""
function write_plotfile(filename::AbstractString, mfs::Vector{JlMultiFab}, geoms::Vector{JlGeometry};
                        varnames::Vector{String}=String[], time::Real=0.0, step::Integer=0,
                        ref_ratio::NTuple{3, Integer}=(2, 2, 2))
    csv = join(varnames, ",")
    if length(mfs) == 1
        write_single_level_plotfile(String(filename), mfs[1], csv, geoms[1], Float64(time), Int32(step))
    elseif length(mfs) == 2
        write_two_level_plotfile(String(filename), mfs[1], mfs[2], csv,
                                 geoms[1], geoms[2], Float64(time), Int32(step),
                                 Int32(ref_ratio[1]), Int32(ref_ratio[2]), Int32(ref_ratio[3]))
    else
        for l in eachindex(mfs)
            write_single_level_plotfile("$(filename)_lev$(l-1)", mfs[l], csv, geoms[l], Float64(time), Int32(step))
        end
    end
    return filename
end
