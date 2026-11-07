using Test
using AMReX

amrex_init(false)

try
    @testset "AMReX.jl Comprehensive Test Suite" begin
        @testset "Lifecycle & Parallel Reductions" begin
            @test amrex_initialized()
            @test parallel_nprocs() >= 1
            @test parallel_myproc() >= 0
            @test parallel_reduce_sum_real(2.5) ≈ 2.5 * parallel_nprocs()
            v = [1.0, 2.0, 3.0]
            parallel_reduce_sum_real_vec!(v)
            @test v ≈ [1.0, 2.0, 3.0] .* parallel_nprocs()
        end

        @testset "Box, RealBox, Geometry, BoxArray" begin
            bx = Box(0, 0, 0, 15, 15, 0)
            @test bounds(bx) == (0:15, 0:15, 0:0)
            @test num_pts(bx) == 256
            @test contains(bx, Int32(5), Int32(5), Int32(0))
            @test !contains(bx, Int32(16), Int32(5), Int32(0))

            nbx = surrounding_nodes(bx)
            @test bounds(nbx) == (0:16, 0:16, 0:1)
            @test is_nodal(nbx, Int32(0))

            rb = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            geom = Geometry(bx, rb; is_periodic=(true, true, true))
            @test cell_sizes(geom) == (1.0, 1.0, 1.0)
            @test inv_cell_sizes(geom) == (1.0, 1.0, 1.0)
            @test is_periodic(geom, Int32(0))

            ba = BoxArray(bx; max_grid_size=(8, 8, 1))
            @test AMReX.size(ba) == 4
            nba = nodal_box_array(ba)
            @test is_nodal(get_box(nba, 0), Int32(0))
        end

        @testset "MultiFab, iMultiFab, Zero-Copy OffsetArrays, FillBoundary & SumBoundary" begin
            bx = Box(0, 0, 0, 7, 7, 0)
            rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
            geom = Geometry(bx, rb; is_periodic=(true, true, true))
            ba = BoxArray(bx; max_grid_size=(4, 4, 1))
            dm = DistributionMapping(ba)

            mf = MultiFab(ba, dm, 2, 1)
            set_val!(mf, 0.0)

            # Mutate valid cells directly in Julia via zero-copy OffsetArray
            each_fab(mf) do mfi, vbx, arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    arr[i, j, k, 1] = Float64(10 * i + j)
                    arr[i, j, k, 2] = 1.0
                end
            end

            fill_boundary!(mf, geom)

            # Verify periodic guard cells were populated by AMReX FillBoundary
            each_fab(mf) do mfi, vbx, arr
                rx, ry, rz = bounds(vbx)
                if first(rx) == 0 && first(ry) == 0
                    # Periodic wrap from i = 7, j = 0 -> 10*7 + 0 = 70.0
                    @test arr[-1, 0, 0, 1] ≈ 70.0
                end
            end

            # Test SumBoundary! accumulation from guard cells into valid cells
            mf_sum = MultiFab(ba, dm, 1, 1)
            set_val!(mf_sum, 0.0)
            each_fab(mf_sum) do mfi, vbx, arr
                rx, ry, rz = bounds(vbx)
                if first(rx) == 0 && first(ry) == 0
                    # Deposit into left guard cell (-1, 0, 0), which wraps to valid cell (7, 0, 0)
                    arr[-1, 0, 0, 1] = 5.5
                end
            end
            sum_boundary!(mf_sum, geom)
            @test sum_val(mf_sum, Int32(0), false) ≈ 5.5

            # Test iMultiFab bitmasks
            imf = iMultiFab(ba, dm, 1, 1)
            set_val!(imf, Int32(3))
            @test min_val(imf, Int32(0), Int32(0)) == 3
        end

        @testset "AmrCore Multi-Level Hierarchy & FillPatchTwoLevels" begin
            rb = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            amr = AmrCore(rb, (16, 16, 1); max_level=1, max_grid_size=(8, 8, 1), blocking_factor=(4, 4, 1))
            add_refine_box!(amr, Int32(0), 4.0, 4.0, 0.0, 12.0, 12.0, 1.0)
            init_from_scratch!(amr, 0.0)

            @test finest_level(amr) == 1
            @test ref_ratio_tuple(amr, 0) == (2, 2, 1)

            g0 = geom(amr, Int32(0))
            g1 = geom(amr, Int32(1))
            @test cell_sizes(g0) == (1.0, 1.0, 1.0)
            @test cell_sizes(g1) == (0.5, 0.5, 1.0)

            mf0 = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 1)
            mf1 = MultiFab(box_array(amr, Int32(1)), dmap(amr, Int32(1)), 1, 1)
            set_val!(mf0, 3.25)
            set_val!(mf1, 0.0)

            # Interpolate coarse -> fine (including fine ghost cells)
            interp_from_coarse_to_fine!(mf0, mf1, Int32(0), Int32(1),
                                        Int32(2), Int32(2), Int32(1),
                                        g0, g1, Int32(CellConservativeLinear), Int32(1))
            @test min_val(mf1, Int32(0), Int32(0)) ≈ 3.25
            @test max_val(mf1, Int32(0), Int32(0)) ≈ 3.25
        end

        @testset "Pure-SoA ParticleContainer, 64-bit idcpu & Redistribute!" begin
            rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
            amr = AmrCore(rb, (8, 8, 1); max_level=0, max_grid_size=(4, 4, 1))
            init_from_scratch!(amr, 0.0)

            pc = ParticleContainer(amr; n_real=4, n_int=1)
            my_cpu = parallel_myproc()

            # Inject 1 particle per cell via MFIter on a dummy MultiFab
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
                    # Shift x by +4.5 so Redistribute! migrates particles across box & periodic boundaries
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

            # Mark 4 particles invalid in the first tile and re-redistribute
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

            # Test Plotfile & Particle Plotfile output in a temp directory
            mktempdir() do tmpdir
                plt_dir = joinpath(tmpdir, "plt00000")
                write_plotfile(plt_dir, mf_dummy, g0; varnames=["rho"], time=0.0, step=0)
                @test isfile(joinpath(plt_dir, "Header"))
                write_plotfile(pc, plt_dir, "particles")
                @test isdir(joinpath(plt_dir, "particles"))
            end
        end
    end
finally
    amrex_finalize()
end
