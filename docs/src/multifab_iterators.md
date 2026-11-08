# MultiFab & Zero-Copy Iterators

## Allocating `MultiFab` and `iMultiFab`

- `MultiFab(ba, dm, ncomp, nghost)` allocates double-precision (`Float64`) field data.
- `iMultiFab(ba, dm, ncomp, nghost)` allocates 32-bit integer (`Int32`) mask or status data.

`nghost` can be either a uniform integer or an anisotropic tuple `(ngx, ngy, ngz)`:

```julia
mf_cell = MultiFab(ba, dm, 3, (2, 2, 1))
imf_status = iMultiFab(ba, dm, 1, 2)
```

## Zero-Copy 4D `OffsetArray` Views

Inside an `each_fab` block, `AMReX.jl` wraps the raw C++ `FArrayBox` pointer in a 4D `OffsetArray{Float64, 4, Array{Float64, 4}}` without copying memory:

- Spatial axes `axes(arr, 1)`, `axes(arr, 2)`, `axes(arr, 3)` use **0-based inclusive AMReX indices** (including ghost layers).
- Component axis `axes(arr, 4)` uses **1-based Julia indices** `1:n_comp(mf)`.

```julia
each_fab(mf_cell; tiling=false) do mfi, vbx, arr
    rx, ry, rz = bounds(vbx) # valid interior box ranges
    @inbounds for k in rz, j in ry, i in rx
        arr[i, j, k, 1] = Float64(i)
        arr[i, j, k, 2] = Float64(j)
        arr[i, j, k, 3] = Float64(k)
    end
end
```

### Simultaneous Multi-Field Iteration

When a stencil reads from one `MultiFab` and writes to another that shares the same `BoxArray` and `DistributionMapping`, iterate over the destination `MultiFab` and call `array4(src, mfi)` using the same `JlMFIter`:

```julia
each_fab(mf_dst) do mfi, vbx, dst
    src = array4(mf_src, mfi)
    rx, ry, rz = bounds(vbx)
    @inbounds for k in rz, j in ry, i in rx
        dst[i, j, k, 1] = 0.5 * (src[i + 1, j, k, 1] - src[i - 1, j, k, 1])
    end
end
```

## Ghost-Cell Synchronization

- **`fill_boundary!(mf, geom)`**: Exchanges interior valid values into ghost layers of neighboring boxes and across periodic domain boundaries. For nodal `MultiFab`s, shared boundary nodes are synchronized consistently.
- **`sum_boundary!(mf, geom)`**: Accumulates values deposited into ghost cells/nodes (e.g., PIC charge and current density deposition) back into the owning valid cells/nodes.

```julia
# After particle-to-grid deposition into ghost nodes:
sum_boundary!(J_node, geom)
fill_boundary!(J_node, geom)
```

## Built-in Linear Algebra & Reductions

`AMReX.jl` exposes both C++ AMReX reductions and allocation-free linear combinations:

| Operation | Description |
| :--- | :--- |
| `set_val!(mf, val)` | Set all components and ghost cells to `val` |
| `scale!(mf, a)` | In-place scaling `mf *= a` |
| `copy!(dst, src, sc, dc, nc, ng)` | Local FAB-to-FAB component copy |
| `plus!(dst, src, sc, dc, nc, ng)` | Local FAB-to-FAB component addition |
| `saxpy!(dst, a, src, sc, dc, nc, ng)` | In-place `dst += a * src` |
| `lincomb!(dst, a, x, xc, b, y, yc, dc, nc, ng)` | In-place `dst = a * x + b * y` |
| `min_val(mf, comp)`, `max_val(mf, comp)` | Global min/max of 0-based component `comp` |
| `norm0(mf, comp)`, `norm1(mf, comp)`, `norm2(mf, comp)` | $L_\infty$, $L_1$, and $L_2$ norms |
| `sum_val(mf, comp)` | Global sum over valid domain |

