@testset "Pure-SoA ParticleContainer, 64-bit idcpu, Redistribute! & Plotfiles" begin
    let
        @test @inferred(pack_idcpu(Int64(10), 0)) isa UInt64
        @test @inferred(unpack_id(pack_idcpu(Int64(10), 0))) == 10
        @test @inferred(unpack_cpu(pack_idcpu(Int64(10), 2))) == 2

        rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
        amr = AmrCore(rb, (8, 8, 1); max_level=0, max_grid_size=(4, 4, 1))
        init_from_scratch!(amr, 0.0)

        pc = ParticleContainer(amr; n_real=4, n_int=1)
        my_cpu = parallel_myproc()

        mf_dummy = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 0)
        g0 = geom(amr, Int32(0))
        dx, dy, dz = cell_sizes(g0)

        each_fab(mf_dummy) do mfi, vbx, _
            gid = index(mfi)
            tid = local_tile_index(mfi)
            rx, ry, rz = bounds(vbx)
            n_new = length(rx) * length(ry) * length(rz)
            resize_tile!(pc, Int32(0), gid, tid, Int64(n_new))

            idcpus = tile_idcpu_vec(pc, 0, gid, tid)
            px, py, pz = tile_pos_vecs(pc, 0, gid, tid)
            w  = tile_real_vec(pc, 0, gid, tid, 1)
            ux = tile_real_vec(pc, 0, gid, tid, 2)
            sup = tile_int_vec(pc, 0, gid, tid, 1)

            idx = 1
            for k in rz, j in ry, i in rx
                pid = next_particle_id()
                idcpus[idx] = pack_idcpu(pid, my_cpu)
                px[idx] = (i + 0.5) * dx + 4.5
                py[idx] = (j + 0.5) * dy
                pz[idx] = (k + 0.5) * dz
                w[idx]  = 1.25
                ux[idx] = 2.0
                sup[idx] = Int32(7)
                idx += 1
            end
        end

        redistribute!(pc, Int32(0), Int32(0), Int32(0))
        @test total_number_of_particles(pc, true, false) == 64

        removed = Ref(0)
        each_particle_tile(pc, 0) do pti
            if removed[] == 0 && num_particles(pti) >= 4
                idcpus = idcpu_vec(pti)
                for p in 1:4
                    @test is_valid_idcpu(idcpus[p])
                    idcpus[p] = make_invalid_idcpu(idcpus[p])
                    @test !is_valid_idcpu(idcpus[p])
                end
                removed[] = 4
            end
        end
        redistribute!(pc, Int32(0), Int32(0), Int32(0))
        @test total_number_of_particles(pc, true, false) == 60

        mktempdir() do tmpdir
            plt_dir = joinpath(tmpdir, "plt00000")
            write_plotfile(plt_dir, mf_dummy, g0; varnames=["rho"], time=0.0, step=0)
            @test isfile(joinpath(plt_dir, "Header"))
            write_plotfile(pc, plt_dir, "particles")
            @test isdir(joinpath(plt_dir, "particles"))
        end
    end
end
