@testset "Lifecycle, Parallel Reductions, Box, Geometry & AmrCore" begin
    @testset "Lifecycle & Parallel Reductions" begin
        let
            @test amrex_initialized()
            @test parallel_nprocs() >= 1
            @test parallel_myproc() >= 0
            @test parallel_reduce_sum_real(2.5) ≈ 2.5 * parallel_nprocs()
            v = [1.0, 2.0, 3.0]
            parallel_reduce_sum_real_vec!(v)
            @test v ≈ [1.0, 2.0, 3.0] .* parallel_nprocs()
        end
    end

    @testset "Box, RealBox, Geometry, BoxArray" begin
        let
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
    end

    @testset "AmrCore Multi-Level Hierarchy & FillPatchTwoLevels" begin
        let
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

            interp_from_coarse_to_fine!(mf0, mf1, Int32(0), Int32(1),
                                        Int32(2), Int32(2), Int32(1),
                                        g0, g1, Int32(CellConservativeLinear), Int32(1))
            @test min_val(mf1, Int32(0), Int32(0)) ≈ 3.25
            @test max_val(mf1, Int32(0), Int32(0)) ≈ 3.25
        end
    end

    @testset "Physical BCs (BCRec, fill_domain_boundary!) & Dynamic Tagging" begin
        let
            # 1. Non-periodic domain boundary filling (FOExtrap & ExtDir)
            bx = Box(0, 0, 0, 7, 7, 0)
            rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
            geom = Geometry(bx, rb; is_periodic=(false, true, true))
            ba = BoxArray(bx; max_grid_size=(8, 8, 1))
            dm = DistributionMapping(ba)
            mf = MultiFab(ba, dm, 1, 1)
            set_val!(mf, 2.5)
            # Zero out ghost cells first
            each_fab(mf) do mfi, vbx, arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry
                    arr[first(rx) - 1, j, k, 1] = 0.0
                    arr[last(rx) + 1, j, k, 1] = 0.0
                end
            end

            bcr_extrap = BCRec(BCFOExtrap, BCIntDir, BCIntDir, BCFOExtrap, BCIntDir, BCIntDir)
            @test lo(bcr_extrap, Int32(0)) == Int32(BCFOExtrap)
            fill_domain_boundary!(mf, geom, bcr_extrap)
            each_fab(mf) do mfi, vbx, arr
                @test arr[-1, 3, 0, 1] ≈ 2.5
                @test arr[8, 3, 0, 1] ≈ 2.5
            end

            bcr_dir = BCRec(BCExtDir, BCIntDir, BCIntDir, BCExtDir, BCIntDir, BCIntDir)
            fill_domain_boundary!(mf, geom, bcr_dir; val_lo=(-7.0, 0.0, 0.0), val_hi=(9.0, 0.0, 0.0))
            each_fab(mf) do mfi, vbx, arr
                @test arr[-1, 3, 0, 1] ≈ -7.0
                @test arr[8, 3, 0, 1] ≈ 9.0
            end

            # 2. Dynamic AMR tagging via tag_cells_by_value! and on_error_est! callback
            rb_amr = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            amr = AmrCore(rb_amr, (16, 16, 1); max_level=1, max_grid_size=(8, 8, 1), blocking_factor=(4, 4, 1))
            init_from_scratch!(amr, 0.0)
            @test finest_level(amr) == 0

            mf_tag = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 1)
            set_val!(mf_tag, 0.0)
            each_fab(mf_tag) do mfi, vbx, arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    if 4 <= i <= 11 && 4 <= j <= 11
                        arr[i, j, k, 1] = 10.0
                    end
                end
            end

            tag_cells_by_value!(amr, 0, mf_tag, 5.0)
            regrid_levels!(amr, Int32(0), 0.0)
            @test finest_level(amr) == 1

            # Clear mask and test on_error_est! callback with TagBoxArray
            clear_tag_masks!(amr)
            regrid_levels!(amr, Int32(0), 0.0)
            @test finest_level(amr) == 0

            on_error_est!(amr) do lev, tags, t
                if lev == 0
                    each_fab(tags) do mfi, vbx, tarr
                        rx, ry, rz = bounds(vbx)
                        for k in rz, j in ry, i in rx
                            if i < 4 && j < 4
                                tarr[i, j, k, 1] = Int8(TagSet)
                            end
                        end
                    end
                end
            end
            regrid_levels!(amr, Int32(0), 0.0)
            @test finest_level(amr) == 1
            on_error_est!(nothing, amr)
        end
    end
end
