@testset "Pure-SoA ParticleContainer, 64-bit idcpu, Redistribute! & Plotfiles" begin
    let
        @test @inferred(pack_idcpu(Int64(10), 0)) isa UInt64
        @test @inferred(unpack_id(pack_idcpu(Int64(10), 0))) == 10
        @test @inferred(unpack_cpu(pack_idcpu(Int64(10), 2))) == 2

        rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
        amr = AmrCore(rb, (8, 8, 1); max_level = 0, max_grid_size = (4, 4, 1))
        init_from_scratch!(amr, 0.0)

        pc = ParticleContainer(amr; n_real = 4, n_int = 1)
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
            w = tile_real_vec(pc, 0, gid, tid, 1)
            ux = tile_real_vec(pc, 0, gid, tid, 2)
            sup = tile_int_vec(pc, 0, gid, tid, 1)

            idx = 1
            for k in rz, j in ry, i in rx
                pid = next_particle_id()
                idcpus[idx] = pack_idcpu(pid, my_cpu)
                px[idx] = (i + 0.5) * dx + 4.5
                py[idx] = (j + 0.5) * dy
                pz[idx] = (k + 0.5) * dz
                w[idx] = 1.25
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
            write_plotfile(plt_dir, mf_dummy, g0; varnames = ["rho"], time = 0.0, step = 0)
            @test isfile(joinpath(plt_dir, "Header"))
            write_plotfile(pc, plt_dir, "particles")
            @test isdir(joinpath(plt_dir, "particles"))

            # Native ParticleContainer checkpoint + restart! roundtrip
            chk_dir = joinpath(tmpdir, "chk00000")
            checkpoint(pc, chk_dir, "particles")
            @test isfile(joinpath(chk_dir, "particles", "Header"))

            pc_restored = ParticleContainer(amr; n_real = 4, n_int = 1)
            @test total_number_of_particles(pc_restored) == 0
            restart!(pc_restored, chk_dir, "particles")
            @test total_number_of_particles(pc_restored) == 60
            each_particle_tile(pc_restored, 0) do pti
                np = num_particles(pti)
                if np > 0
                    w_vec = real_vec(pti, 1)
                    ux_vec = real_vec(pti, 2)
                    sup_vec = int_vec(pti, 1)
                    idcpus = idcpu_vec(pti)
                    @test all(isapprox.(w_vec, 1.25; atol = 1.0e-12))
                    @test all(isapprox.(ux_vec, 2.0; atol = 1.0e-12))
                    @test all(==(Int32(7)), sup_vec)
                    @test all(is_valid_idcpu, idcpus)
                end
            end

            # HDF5 ParticleContainer plotfile & checkpoint/restart roundtrip
            plt_h5_dir = joinpath(tmpdir, "plt_h5_00000")
            write_plotfile_hdf5(plt_h5_dir, mf_dummy, g0; varnames = ["rho"], time = 0.0, step = 0)
            @test isfile(joinpath(plt_h5_dir) * ".h5") || isdir(plt_h5_dir)
            write_plotfile_hdf5(pc, plt_h5_dir, "particles")
            @test isfile(joinpath(plt_h5_dir, "particles", "particles.h5"))

            chk_h5_dir = joinpath(tmpdir, "chk_h5_00000")
            checkpoint_hdf5(pc, chk_h5_dir, "particles")
            @test isfile(joinpath(chk_h5_dir, "particles", "particles.h5"))

            pc_h5_restored = ParticleContainer(amr; n_real = 4, n_int = 1)
            restart_hdf5!(pc_h5_restored, chk_h5_dir, "particles")
            @test total_number_of_particles(pc_h5_restored) == 60
            each_particle_tile(pc_h5_restored, 0) do pti
                np = num_particles(pti)
                if np > 0
                    w_vec = real_vec(pti, 1)
                    ux_vec = real_vec(pti, 2)
                    sup_vec = int_vec(pti, 1)
                    idcpus = idcpu_vec(pti)
                    @test all(isapprox.(w_vec, 1.25; atol = 1.0e-12))
                    @test all(isapprox.(ux_vec, 2.0; atol = 1.0e-12))
                    @test all(==(Int32(7)), sup_vec)
                    @test all(is_valid_idcpu, idcpus)
                end
            end

            # VisMF MultiFab checkpoint write/read roundtrip
            mf_src = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 2, 0)
            set_val_comp!(mf_src, 3.14, 0, 1, 0)
            set_val_comp!(mf_src, -2.71, 1, 1, 0)
            vismf_path = joinpath(tmpdir, "vismf_chk", "Level_0", "Cell")
            nbytes = vismf_write(mf_src, vismf_path)
            @test nbytes > 0
            @test vismf_exist(vismf_path)

            mf_dst = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 2, 0)
            set_val!(mf_dst, 0.0)
            vismf_read!(mf_dst, vismf_path)
            @test min_val(mf_dst, 0, 0) ≈ 3.14
            @test max_val(mf_dst, 0, 0) ≈ 3.14
            @test min_val(mf_dst, 1, 0) ≈ -2.71
            @test max_val(mf_dst, 1, 0) ≈ -2.71

            # 3-level N-level plotfile (native and HDF5)
            amr3 = AmrCore(rb, (8, 8, 8); max_level = 2, max_grid_size = (8, 8, 8), blocking_factor = (2, 2, 2))
            add_refine_box!(amr3, Int32(0), 2.0, 2.0, 0.2, 6.0, 6.0, 0.8)
            add_refine_box!(amr3, Int32(1), 3.0, 3.0, 0.3, 5.0, 5.0, 0.7)
            init_from_scratch!(amr3, 0.0)
            @test finest_level(amr3) == 2

            mfs3 = [MultiFab(box_array(amr3, Int32(l)), dmap(amr3, Int32(l)), 2, 0) for l in 0:2]
            geoms3 = [geom(amr3, Int32(l)) for l in 0:2]
            for l in 1:3
                set_val!(mfs3[l], Float64(l))
            end
            plt3_dir = joinpath(tmpdir, "plt_3lev")
            write_plotfile(plt3_dir, mfs3, geoms3; varnames = ["u", "v"], time = 1.5, step = [1, 2, 4], ref_ratio = (2, 2, 2))
            @test isfile(joinpath(plt3_dir, "Header"))
            @test isdir(joinpath(plt3_dir, "Level_0"))
            @test isdir(joinpath(plt3_dir, "Level_1"))
            @test isdir(joinpath(plt3_dir, "Level_2"))

            plt3_h5 = joinpath(tmpdir, "plt_3lev_h5")
            write_plotfile_hdf5(plt3_h5, mfs3, geoms3; varnames = ["u", "v"], time = 1.5, step = 1, ref_ratio = (2, 2, 2))
            @test isfile(plt3_h5 * ".h5") || isdir(plt3_h5)
        end
    end
end
