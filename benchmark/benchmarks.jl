using BenchmarkTools
using AMReX

if !amrex_initialized()
    amrex_init(false)
end

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# 1. 64-bit idcpu bit-packing & unpacking
# ----------------------------------------------------------------------------
SUITE["idcpu"] = BenchmarkGroup(["bitmask", "particles"])
let
    w = pack_idcpu(Int64(1234567), 3)
    SUITE["idcpu"]["pack_idcpu"] = @benchmarkable pack_idcpu($(Int64(1234567)), $(Int32(3)))
    SUITE["idcpu"]["unpack_id"]  = @benchmarkable unpack_id($w)
    SUITE["idcpu"]["unpack_cpu"] = @benchmarkable unpack_cpu($w)
    SUITE["idcpu"]["is_valid"]   = @benchmarkable is_valid_idcpu($w)
end

# ----------------------------------------------------------------------------
# 2. MultiFab zero-copy array4, linear algebra & halo exchange
# ----------------------------------------------------------------------------
SUITE["multifab"] = BenchmarkGroup(["field", "halo"])
let
    bx = Box(0, 0, 0, 31, 31, 7)
    rb = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    geom = Geometry(bx, rb; is_periodic=(true, true, true))
    ba = BoxArray(bx; max_grid_size=(16, 16, 8))
    dm = DistributionMapping(ba)

    mf_a = MultiFab(ba, dm, 3, 2)
    mf_b = MultiFab(ba, dm, 3, 2)
    mf_c = MultiFab(ba, dm, 3, 2)
    set_val!(mf_a, 1.25)
    set_val!(mf_b, 2.50)
    set_val!(mf_c, 0.0)

    SUITE["multifab"]["each_fab_array4"] = @benchmarkable begin
        each_fab($mf_a) do mfi, vbx, arr
            rx, ry, rz = bounds(vbx)
            @inbounds for k in rz, j in ry, i in rx
                arr[i, j, k, 1] = 1.0
            end
        end
    end
    SUITE["multifab"]["saxpy!"] = @benchmarkable saxpy!($mf_c, 0.5, $mf_a, 0, 0, 3, 0)
    SUITE["multifab"]["lincomb!"] = @benchmarkable lincomb!($mf_c, 0.5, $mf_a, 0, 1.5, $mf_b, 0, 0, 3, 0)
    SUITE["multifab"]["fill_boundary!"] = @benchmarkable fill_boundary!($mf_a, $geom)
    SUITE["multifab"]["sum_boundary!"] = @benchmarkable sum_boundary!($mf_a, $geom)
end

# ----------------------------------------------------------------------------
# 3. Pure-SoA ParticleContainer tile traversal
# ----------------------------------------------------------------------------
SUITE["particles"] = BenchmarkGroup(["soa", "particles"])
let
    rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
    amr = AmrCore(rb, (16, 16, 1); max_level=0, max_grid_size=(8, 8, 1))
    init_from_scratch!(amr, 0.0)
    pc = ParticleContainer(amr; n_real=4, n_int=1)
    mf_dummy = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 0)
    my_cpu = parallel_myproc()

    each_fab(mf_dummy) do mfi, vbx, _
        gid = index(mfi)
        tid = local_tile_index(mfi)
        rx, ry, rz = bounds(vbx)
        np = length(rx) * length(ry) * length(rz) * 4
        resize_tile!(pc, Int32(0), gid, tid, Int64(np))
        idcpus = tile_idcpu_vec(pc, 0, gid, tid)
        px, py, pz = tile_pos_vecs(pc, 0, gid, tid)
        w = tile_real_vec(pc, 0, gid, tid, 1)
        ux = tile_real_vec(pc, 0, gid, tid, 2)
        for p in 1:np
            idcpus[p] = pack_idcpu(Int64(p), my_cpu)
            px[p] = 0.5
            py[p] = 0.5
            pz[p] = 0.5
            w[p] = 1.0
            ux[p] = 0.1
        end
    end

    SUITE["particles"]["soa_tile_scan"] = @benchmarkable begin
        s = 0.0
        each_particle_tile($pc, 0) do pti
            np = Int(num_particles(pti))
            px, py, _ = pos_vecs(pti)
            ux = real_vec(pti, 2)
            @inbounds @simd for p in 1:np
                s += px[p] + py[p] + ux[p]
            end
        end
        s
    end
end

