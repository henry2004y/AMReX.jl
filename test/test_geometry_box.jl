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
            geom = Geometry(bx, rb; is_periodic = (true, true, true))
            @test cell_sizes(geom) == (1.0, 1.0, 1.0)
            @test inv_cell_sizes(geom) == (1.0, 1.0, 1.0)
            @test is_periodic(geom, Int32(0))

            ba = BoxArray(bx; max_grid_size = (8, 8, 1))
            @test AMReX.size(ba) == 4
            nba = nodal_box_array(ba)
            @test is_nodal(get_box(nba, 0), Int32(0))
        end
    end

    @testset "AmrCore Multi-Level Hierarchy & FillPatchTwoLevels" begin
        let
            rb = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            amr = AmrCore(rb, (16, 16, 1); max_level = 1, max_grid_size = (8, 8, 1), blocking_factor = (4, 4, 1))
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

            interp_from_coarse_to_fine!(
                mf0, mf1, Int32(0), Int32(1),
                Int32(2), Int32(2), Int32(1),
                g0, g1, Int32(CellConservativeLinear), Int32(1)
            )
            @test min_val(mf1, Int32(0), Int32(0)) ≈ 3.25
            @test max_val(mf1, Int32(0), Int32(0)) ≈ 3.25
        end
    end

    @testset "Physical BCs (BCRec, fill_domain_boundary!) & Dynamic Tagging" begin
        let
            # 1. Non-periodic domain boundary filling (FOExtrap & ExtDir)
            bx = Box(0, 0, 0, 7, 7, 0)
            rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
            geom = Geometry(bx, rb; is_periodic = (false, true, true))
            ba = BoxArray(bx; max_grid_size = (8, 8, 1))
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
            fill_domain_boundary!(mf, geom, bcr_dir; val_lo = (-7.0, 0.0, 0.0), val_hi = (9.0, 0.0, 0.0))
            each_fab(mf) do mfi, vbx, arr
                @test arr[-1, 3, 0, 1] ≈ -7.0
                @test arr[8, 3, 0, 1] ≈ 9.0
            end

            # 2. Dynamic AMR tagging via tag_cells_by_value! and on_error_est! callback
            rb_amr = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            amr = AmrCore(rb_amr, (16, 16, 1); max_level = 1, max_grid_size = (8, 8, 1), blocking_factor = (4, 4, 1))
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

    @testset "AMR Geometry Ops, 1D AmrCore, make_fine_mask, Load Balancing & Time-Interp FillPatch" begin
        let
            # 1. Box & BoxArray AMR geometry ops
            bx = Box(0, 0, 0, 15, 7, 0)
            bx_c = coarsen(bx, (2, 2, 1))
            @test bounds(bx_c) == (0:7, 0:3, 0:0)
            @test bounds(refine(bx_c, (2, 2, 1))) == (0:15, 0:7, 0:0)
            @test bounds(grow(bx, (1, 2, 0))) == (-1:16, -2:9, 0:0)
            @test bounds(shift(bx_c, (2, -1, 0))) == (2:9, -1:2, 0:0)
            @test intersects(bx, bx_c)
            @test !intersects(bx, shift(bx, (20, 0, 0)))
            @test bounds(surrounding_nodes(bx_c, 0)) == (0:8, 0:3, 0:0)
            @test bounds(enclosed_cells(surrounding_nodes(bx_c, 0), 0)) == (0:7, 0:3, 0:0)

            ba_custom = BoxArray([Box(0, 0, 0, 7, 7, 0), Box(8, 0, 0, 15, 7, 0)])
            @test AMReX.size(ba_custom) == 2
            @test num_pts(ba_custom) == 128
            @test coarsenable(ba_custom, (2, 2, 1))
            ba_c = coarsen(ba_custom, (2, 2, 1))
            @test num_pts(ba_c) == 32
            @test num_pts(refine(ba_c, (2, 2, 1))) == 128
            @test num_pts(grow(ba_c, (1, 1, 0))) == 2 * 6 * 6
            @test intersects(ba_custom, Box(6, 0, 0, 9, 3, 0))
            @test !intersects(ba_custom, Box(16, 0, 0, 20, 3, 0))

            # 2. 1D AmrCore default ref_ratio == (2, 1, 1) and n_error_buf
            rb1d = RealBox(0.0, 0.0, 0.0, 8.0, 1.0, 1.0)
            amr1d = AmrCore(
                rb1d, (8, 1, 1); max_level = 1, max_grid_size = (8, 1, 1),
                blocking_factor = (2, 1, 1), n_error_buf = 1
            )
            add_tag_box!(amr1d, 2.0, 0.0, 0.0, 6.0, 1.0, 1.0)
            init_from_scratch!(amr1d, 0.0)
            @test finest_level(amr1d) == 1
            @test ref_ratio_tuple(amr1d, 0) == (2, 1, 1)
            dom1 = domain(geom(amr1d, Int32(1)))
            @test bounds(dom1) == (0:15, 0:0, 0:0)

            # 3. make_fine_mask & coarsened_multifab
            mask = make_fine_mask(amr1d, 0; crse_val = 0, fine_val = 1, nghost = 0)
            @test min_val(mask, Int32(0), Int32(0)) == 0
            @test max_val(mask, Int32(0), Int32(0)) == 1

            g0_1d = geom(amr1d, Int32(0))
            mf0_1d = MultiFab(box_array(amr1d, Int32(0)), dmap(amr1d, Int32(0)), 2, 1)
            each_fab(mf0_1d) do _, vbx, arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    arr[i, j, k, 1] = Float64(i + 1)
                    arr[i, j, k, 2] = Float64(10 * i)
                end
            end
            fill_boundary!(mf0_1d, g0_1d)
            mf_stage = coarsened_multifab(amr1d, 1, 2, (1, 0, 0))
            parallel_copy!(mf_stage, mf0_1d, g0_1d; snghost = (1, 0, 0), dnghost = (1, 0, 0))
            each_fab(mf_stage) do _, vbx, arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    @test arr[i, j, k, 1] ≈ Float64(i + 1)
                    @test arr[i, j, k, 2] ≈ Float64(10 * i)
                end
            end

            # 4. Weighted DistributionMapping (Knapsack & SFC)
            dm_ks = make_knapsack_dmap([1.0, 5.0])
            @test AMReX.size(dm_ks) == 2
            dm_sfc = make_sfc_dmap([1.0, 5.0], ba_custom)
            @test AMReX.size(dm_sfc) == 2

            # 5. Time-interpolated fill_patch_two_levels!
            rb2 = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
            amr2 = AmrCore(rb2, (16, 16, 1); max_level = 1, max_grid_size = (8, 8, 1), blocking_factor = (4, 4, 1))
            add_tag_box!(amr2, 4.0, 4.0, 0.0, 12.0, 12.0, 1.0)
            init_from_scratch!(amr2, 0.0)
            g0 = geom(amr2, Int32(0)); g1 = geom(amr2, Int32(1))
            ba0 = box_array(amr2, Int32(0)); dm0 = dmap(amr2, Int32(0))
            ba1 = box_array(amr2, Int32(1)); dm1 = dmap(amr2, Int32(1))
            rr2 = ref_ratio_tuple(amr2, 0)

            cmf_old = MultiFab(ba0, dm0, 1, 1); set_val!(cmf_old, 10.0)
            cmf_new = MultiFab(ba0, dm0, 1, 1); set_val!(cmf_new, 20.0)
            fmf_cur = MultiFab(ba1, dm1, 1, 1); set_val!(fmf_cur, 15.0)
            dst_f   = MultiFab(ba1, dm1, 1, 1); set_val!(dst_f, 0.0)

            fill_patch_two_levels!(
                dst_f, 0.25,
                cmf_old, 0.0, cmf_new, 1.0,
                fmf_cur, 0.25,
                0, 0, 1, g0, g1, rr2, CellConservativeLinear
            )
            # Ghost cell outside fine valid box gets linearly time-interpolated value: 10 + 0.25*(20-10) = 12.5
            each_fab(dst_f) do _, vbx, arr
                rx, ry, rz = bounds(vbx)
                # Interior copied from fmf_cur (15.0), coarse-fine ghost interpolated to 12.5
                @test arr[first(rx), first(ry), 0, 1] ≈ 15.0
                if first(rx) == 8
                    @test arr[first(rx) - 1, first(ry), 0, 1] ≈ 12.5
                end
            end
        end
    end
end
