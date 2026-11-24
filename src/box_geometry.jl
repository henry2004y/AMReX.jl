# ============================================================================
# Box, RealBox, Geometry, BoxArray, DistributionMapping high-level API
# ============================================================================

"""
    Box(lox, loy, loz, hix, hiy, hiz; nodal=(false, false, false))

Construct a 3D `JlBox` with inclusive 0-based bounds `(lox:hix, loy:hiy, loz:hiz)`.
"""
function Box(
        lox::Integer, loy::Integer, loz::Integer,
        hix::Integer, hiy::Integer, hiz::Integer;
        nodal::NTuple{3, Bool} = (false, false, false)
    )
    return JlBox(
        Int32(lox), Int32(loy), Int32(loz),
        Int32(hix), Int32(hiy), Int32(hiz),
        Int32(nodal[1]), Int32(nodal[2]), Int32(nodal[3])
    )
end

"""
    bounds(bx::JlBox) -> ((lox, hix), (loy, hiy), (loz, hiz))

Return the inclusive index ranges `(lox:hix, loy:hiy, loz:hiz)` of `bx`.
"""
@inline function bounds(bx::JlBox)
    buf = Ref{NTuple{6, Int32}}()
    GC.@preserve buf fill_bounds!(bx, Base.unsafe_convert(Ptr{Int32}, buf))
    lo0, lo1, lo2, hi0, hi1, hi2 = buf[]
    return (
        Int(lo0):Int(hi0),
        Int(lo1):Int(hi1),
        Int(lo2):Int(hi2),
    )
end

"""
    coarsen(bx::JlBox, ref_ratio::Union{Integer, NTuple{3, Integer}}) -> JlBoxAllocated
    coarsen(ba::JlBoxArray, ref_ratio::Union{Integer, NTuple{3, Integer}}) -> JlBoxArrayAllocated

Coarsen `bx` or `ba` by `ref_ratio`.
"""
@inline function coarsen(bx::JlBox, rr::Integer)
    r = Int32(rr)
    return coarsen(bx, r, r, r)
end
@inline function coarsen(bx::JlBox, rr::NTuple{3, Integer})
    return coarsen(bx, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
end
@inline function coarsen(ba::JlBoxArray, rr::Integer)
    r = Int32(rr)
    return coarsen(ba, r, r, r)
end
@inline function coarsen(ba::JlBoxArray, rr::NTuple{3, Integer})
    return coarsen(ba, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
end

"""
    refine(bx::JlBox, ref_ratio::Union{Integer, NTuple{3, Integer}}) -> JlBoxAllocated
    refine(ba::JlBoxArray, ref_ratio::Union{Integer, NTuple{3, Integer}}) -> JlBoxArrayAllocated

Refine `bx` or `ba` by `ref_ratio`.
"""
@inline function refine(bx::JlBox, rr::Integer)
    r = Int32(rr)
    return refine(bx, r, r, r)
end
@inline function refine(bx::JlBox, rr::NTuple{3, Integer})
    return refine(bx, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
end
@inline function refine(ba::JlBoxArray, rr::Integer)
    r = Int32(rr)
    return refine(ba, r, r, r)
end
@inline function refine(ba::JlBoxArray, rr::NTuple{3, Integer})
    return refine(ba, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
end

"""
    grow(bx::JlBox, ng::NTuple{3, Integer}) -> JlBoxAllocated
    grow(ba::JlBoxArray, ng::Union{Integer, NTuple{3, Integer}}) -> JlBoxArrayAllocated

Grow `bx` or `ba` by isotropic or anisotropic ghost widths.
"""
@inline function grow(bx::JlBox, ng::NTuple{3, Integer})
    return grow_aniso(bx, Int32(ng[1]), Int32(ng[2]), Int32(ng[3]))
end
@inline function grow(ba::JlBoxArray, ng::Integer)
    n = Int32(ng)
    return grow(ba, n, n, n)
end
@inline function grow(ba::JlBoxArray, ng::NTuple{3, Integer})
    return grow(ba, Int32(ng[1]), Int32(ng[2]), Int32(ng[3]))
end

"""
    shift(bx::JlBox, offset::NTuple{3, Integer}) -> JlBoxAllocated

Shift `bx` by index offset `(sx, sy, sz)`.
"""
@inline function shift(bx::JlBox, offset::NTuple{3, Integer})
    return shift(bx, Int32(offset[1]), Int32(offset[2]), Int32(offset[3]))
end

"""
    surrounding_nodes(bx::JlBox, dir::Integer) -> JlBoxAllocated

Convert `bx` to nodal along 0-based direction `dir`.
"""
@inline surrounding_nodes(bx::JlBox, dir::Integer) = surrounding_nodes_dir(bx, Int32(dir))

"""
    enclosed_cells(bx::JlBox, dir::Integer) -> JlBoxAllocated

Convert `bx` to cell-centered along 0-based direction `dir`.
"""
@inline enclosed_cells(bx::JlBox, dir::Integer) = enclosed_cells_dir(bx, Int32(dir))

"""
    coarsenable(ba::JlBoxArray, ref_ratio::Union{Integer, NTuple{3, Integer}}) -> Bool

Return `true` if every box in `ba` is evenly coarsenable by `ref_ratio`.
"""
@inline function coarsenable(ba::JlBoxArray, rr::Integer)
    r = Int32(rr)
    return coarsenable(ba, r, r, r)
end
@inline function coarsenable(ba::JlBoxArray, rr::NTuple{3, Integer})
    return coarsenable(ba, Int32(rr[1]), Int32(rr[2]), Int32(rr[3]))
end

"""
    intersects(ba::JlBoxArray, bx::JlBox) -> Bool

Return `true` if any box in `ba` intersects `bx`.
"""
@inline intersects(ba::JlBoxArray, bx::JlBox) = intersects_box(ba, bx)

"""
    RealBox(xlo, ylo, zlo, xhi, yhi, zhi)

Construct a physical domain bounding box `[xlo, xhi] × [ylo, yhi] × [zlo, zhi]`.
"""
function RealBox(
        xlo::Real, ylo::Real, zlo::Real,
        xhi::Real, yhi::Real, zhi::Real
    )
    return JlRealBox(
        Float64(xlo), Float64(ylo), Float64(zlo),
        Float64(xhi), Float64(yhi), Float64(zhi)
    )
end

"""
    Geometry(domain::JlBox, rbox::JlRealBox; coord::Integer=0, is_periodic=(true, true, true))

Construct an AMReX `JlGeometry` mapping index space `domain` to physical space `rbox`.
"""
function Geometry(
        dom::JlBox, rb::JlRealBox;
        coord::Integer = 0,
        is_periodic::NTuple{3, Bool} = (true, true, true)
    )
    return JlGeometry(
        dom, rb, Int32(coord),
        is_periodic[1], is_periodic[2], is_periodic[3]
    )
end

"""
    cell_sizes(geom::JlGeometry) -> NTuple{3, Float64}

Return the physical cell widths `(dx, dy, dz)` for `geom`.
"""
@inline cell_sizes(geom::JlGeometry) = (
    Float64(cell_size(geom, 0)),
    Float64(cell_size(geom, 1)),
    Float64(cell_size(geom, 2)),
)

"""
    inv_cell_sizes(geom::JlGeometry) -> NTuple{3, Float64}

Return the inverse physical cell widths `(1/dx, 1/dy, 1/dz)` for `geom`.
"""
@inline inv_cell_sizes(geom::JlGeometry) = (
    Float64(inv_cell_size(geom, 0)),
    Float64(inv_cell_size(geom, 1)),
    Float64(inv_cell_size(geom, 2)),
)

"""
    prob_lo_tuple(geom::JlGeometry) -> NTuple{3, Float64}

Return the lower physical coordinates `(xlo, ylo, zlo)` of the problem domain.
"""
@inline prob_lo_tuple(geom::JlGeometry) = (
    Float64(prob_lo(geom, 0)),
    Float64(prob_lo(geom, 1)),
    Float64(prob_lo(geom, 2)),
)

"""
    prob_hi_tuple(geom::JlGeometry) -> NTuple{3, Float64}

Return the upper physical coordinates `(xhi, yhi, zhi)` of the problem domain.
"""
@inline prob_hi_tuple(geom::JlGeometry) = (
    Float64(prob_hi(geom, 0)),
    Float64(prob_hi(geom, 1)),
    Float64(prob_hi(geom, 2)),
)

"""
    BoxArray(dom::JlBox; max_grid_size=nothing)

Construct a `JlBoxArray` from a domain box and optionally chop into sub-boxes of `max_grid_size`.
"""
function BoxArray(dom::JlBox; max_grid_size::Union{Nothing, Int, NTuple{3, Int}} = nothing)
    ba = JlBoxArray(dom)
    if max_grid_size isa Int
        max_size!(ba, Int32(max_grid_size), Int32(max_grid_size), Int32(max_grid_size))
    elseif max_grid_size isa NTuple{3, Int}
        max_size!(ba, Int32(max_grid_size[1]), Int32(max_grid_size[2]), Int32(max_grid_size[3]))
    end
    return ba
end

function BoxArray(boxes::AbstractVector{<:JlBox})
    n = length(boxes)
    buf = Vector{Int32}(undef, 6 * n)
    tmp = Ref{NTuple{6, Int32}}()
    GC.@preserve tmp begin
        ptr = Base.unsafe_convert(Ptr{Int32}, tmp)
        for (idx, bx) in enumerate(boxes)
            fill_bounds!(bx, ptr)
            b = tmp[]
            base = 6 * (idx - 1)
            @inbounds for d in 1:6
                buf[base + d] = b[d]
            end
        end
    end
    return GC.@preserve buf make_box_array_from_bounds(pointer(buf), Int64(n))
end

"""
    nodal_box_array(ba::JlBoxArray; dirs=(true, true, true))

Convert a cell-centered `BoxArray` to nodal along the specified directions.
"""
function nodal_box_array(ba::JlBoxArray; dirs::NTuple{3, Bool} = (true, true, true))
    return convert_to_nodal(ba, Int32(dirs[1]), Int32(dirs[2]), Int32(dirs[3]))
end

"""
    DistributionMapping(ba::JlBoxArray)

Construct an AMReX `JlDistributionMapping` for `ba`.
"""
DistributionMapping(ba::JlBoxArray) = JlDistributionMapping(ba)

"""
    make_knapsack_dmap(costs::AbstractVector{<:Real}) -> JlDistributionMappingAllocated

Build a load-balanced `DistributionMapping` using AMReX's Knapsack algorithm
with per-box computational weights `costs`.
"""
function make_knapsack_dmap(costs::AbstractVector{<:Real})
    c = Float64.(costs)
    return GC.@preserve c make_knapsack_dmap_raw(pointer(c), Int64(length(c)))
end

"""
    make_sfc_dmap(costs::AbstractVector{<:Real}, ba::JlBoxArray; sort::Bool=true) -> JlDistributionMappingAllocated

Build a locality-preserving load-balanced `DistributionMapping` using AMReX's
Space-Filling Curve (SFC) algorithm with per-box weights `costs`.
"""
function make_sfc_dmap(costs::AbstractVector{<:Real}, ba::JlBoxArray; sort::Bool = true)
    length(costs) == Int(size(ba)) ||
        throw(ArgumentError("length(costs) ($(length(costs))) must match size(ba) ($(size(ba)))"))
    c = Float64.(costs)
    return GC.@preserve c make_sfc_dmap_raw(pointer(c), Int64(length(c)), ba, sort)
end
