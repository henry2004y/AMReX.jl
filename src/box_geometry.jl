# ============================================================================
# Box, RealBox, Geometry, BoxArray, DistributionMapping high-level API
# ============================================================================

"""
    Box(lox, loy, loz, hix, hiy, hiz; nodal=(false, false, false))

Construct a 3D `JlBox` with inclusive 0-based bounds `(lox:hix, loy:hiy, loz:hiz)`.
"""
function Box(lox::Integer, loy::Integer, loz::Integer,
             hix::Integer, hiy::Integer, hiz::Integer;
             nodal::NTuple{3, Bool}=(false, false, false))
    return JlBox(Int32(lox), Int32(loy), Int32(loz),
                 Int32(hix), Int32(hiy), Int32(hiz),
                 Int32(nodal[1]), Int32(nodal[2]), Int32(nodal[3]))
end

"""
    bounds(bx::JlBox) -> ((lox, hix), (loy, hiy), (loz, hiz))

Return the inclusive index ranges `(lox:hix, loy:hiy, loz:hiz)` of `bx`.
"""
@inline function bounds(bx::JlBox)
    return (Int(lo(bx, 0)):Int(hi(bx, 0)),
            Int(lo(bx, 1)):Int(hi(bx, 1)),
            Int(lo(bx, 2)):Int(hi(bx, 2)))
end

"""
    RealBox(xlo, ylo, zlo, xhi, yhi, zhi)

Construct a physical domain bounding box `[xlo, xhi] × [ylo, yhi] × [zlo, zhi]`.
"""
function RealBox(xlo::Real, ylo::Real, zlo::Real,
                 xhi::Real, yhi::Real, zhi::Real)
    return JlRealBox(Float64(xlo), Float64(ylo), Float64(zlo),
                     Float64(xhi), Float64(yhi), Float64(zhi))
end

"""
    Geometry(domain::JlBox, rbox::JlRealBox; coord::Integer=0, is_periodic=(true, true, true))

Construct an AMReX `JlGeometry` mapping index space `domain` to physical space `rbox`.
"""
function Geometry(dom::JlBox, rb::JlRealBox;
                  coord::Integer=0,
                  is_periodic::NTuple{3, Bool}=(true, true, true))
    return JlGeometry(dom, rb, Int32(coord),
                      is_periodic[1], is_periodic[2], is_periodic[3])
end

@inline cell_sizes(geom::JlGeometry) = (
    Float64(cell_size(geom, 0)),
    Float64(cell_size(geom, 1)),
    Float64(cell_size(geom, 2)),
)

@inline inv_cell_sizes(geom::JlGeometry) = (
    Float64(inv_cell_size(geom, 0)),
    Float64(inv_cell_size(geom, 1)),
    Float64(inv_cell_size(geom, 2)),
)

@inline prob_lo_tuple(geom::JlGeometry) = (
    Float64(prob_lo(geom, 0)),
    Float64(prob_lo(geom, 1)),
    Float64(prob_lo(geom, 2)),
)

@inline prob_hi_tuple(geom::JlGeometry) = (
    Float64(prob_hi(geom, 0)),
    Float64(prob_hi(geom, 1)),
    Float64(prob_hi(geom, 2)),
)

"""
    BoxArray(dom::JlBox; max_grid_size=nothing)

Construct a `JlBoxArray` from a domain box and optionally chop into sub-boxes of `max_grid_size`.
"""
function BoxArray(dom::JlBox; max_grid_size::Union{Nothing, Int, NTuple{3, Int}}=nothing)
    ba = JlBoxArray(dom)
    if max_grid_size isa Int
        max_size!(ba, Int32(max_grid_size), Int32(max_grid_size), Int32(max_grid_size))
    elseif max_grid_size isa NTuple{3, Int}
        max_size!(ba, Int32(max_grid_size[1]), Int32(max_grid_size[2]), Int32(max_grid_size[3]))
    end
    return ba
end

"""
    nodal_box_array(ba::JlBoxArray; dirs=(true, true, true))

Convert a cell-centered `BoxArray` to nodal along the specified directions.
"""
function nodal_box_array(ba::JlBoxArray; dirs::NTuple{3, Bool}=(true, true, true))
    return convert_to_nodal(ba, Int32(dirs[1]), Int32(dirs[2]), Int32(dirs[3]))
end

"""
    DistributionMapping(ba::JlBoxArray)

Construct an AMReX `JlDistributionMapping` for `ba`.
"""
DistributionMapping(ba::JlBoxArray) = JlDistributionMapping(ba)

