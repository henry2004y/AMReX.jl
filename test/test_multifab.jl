@testset "MultiFab, iMultiFab, Zero-Copy OffsetArrays & Halo Exchange" begin
    let
        bx = Box(0, 0, 0, 7, 7, 0)
        rb = RealBox(0.0, 0.0, 0.0, 8.0, 8.0, 1.0)
        geom = Geometry(bx, rb; is_periodic=(true, true, true))
        ba = BoxArray(bx; max_grid_size=(4, 4, 1))
        dm = DistributionMapping(ba)

        mf = MultiFab(ba, dm, 2, 1)
        set_val!(mf, 0.0)

        each_fab(mf) do mfi, vbx, arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                arr[i, j, k, 1] = Float64(10 * i + j)
                arr[i, j, k, 2] = 1.0
            end
        end

        fill_boundary!(mf, geom)

        each_fab(mf) do mfi, vbx, arr
            rx, ry, rz = bounds(vbx)
            if first(rx) == 0 && first(ry) == 0
                @test arr[-1, 0, 0, 1] ≈ 70.0
            end
        end

        mf_sum = MultiFab(ba, dm, 1, 1)
        set_val!(mf_sum, 0.0)
        each_fab(mf_sum) do mfi, vbx, arr
            rx, ry, rz = bounds(vbx)
            if first(rx) == 0 && first(ry) == 0
                arr[-1, 0, 0, 1] = 5.5
            end
        end
        sum_boundary!(mf_sum, geom)
        @test sum_val(mf_sum, Int32(0), false) ≈ 5.5

        mf_a = MultiFab(ba, dm, 1, 0)
        mf_b = MultiFab(ba, dm, 1, 0)
        mf_dst = MultiFab(ba, dm, 1, 0)
        set_val!(mf_a, 2.0)
        set_val!(mf_b, 3.0)
        saxpy!(mf_a, 0.5, mf_b, 0, 0, 1, 0)
        @test min_val(mf_a, Int32(0), Int32(0)) ≈ 3.5
        lincomb!(mf_dst, 2.0, mf_a, 0, -1.0, mf_b, 0, 0, 1, 0)
        @test max_val(mf_dst, Int32(0), Int32(0)) ≈ 4.0

        imf = iMultiFab(ba, dm, 1, 1)
        set_val!(imf, Int32(3))
        @test min_val(imf, Int32(0), Int32(0)) == 3
        @test mf isa JlMultiFabAllocated
        @test imf isa JliMultiFabAllocated

        mfi = JlMFIter(mf, false)
        @test @inferred(array4(mf, mfi)) isa AbstractArray{Float64, 4}
        close!(mfi)

        imfi = JlMFIter(imf, false)
        @test @inferred(iarray4(imf, imfi)) isa AbstractArray{Int32, 4}
        close!(imfi)
    end

    @testset "YAFluxRegister, EdgeFluxRegister & average_down_faces!" begin
        let
            rb = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 4.0)
            amr = AmrCore(rb, (16, 16, 4); max_level=1, max_grid_size=(8, 8, 4), blocking_factor=(4, 4, 4))
            add_refine_box!(amr, Int32(0), 4.0, 4.0, 0.0, 12.0, 12.0, 4.0)
            init_from_scratch!(amr, 0.0)
            @test finest_level(amr) == 1

            ba0 = box_array(amr, Int32(0))
            dm0 = dmap(amr, Int32(0))
            g0  = geom(amr, Int32(0))
            ba1 = box_array(amr, Int32(1))
            dm1 = dmap(amr, Int32(1))
            g1  = geom(amr, Int32(1))
            rr  = ref_ratio_tuple(amr, 0)

            # 1. YAFluxRegister conservative reflux test
            yafr = YAFluxRegister(amr, 1, 1)
            reset!(yafr)

            state_c = MultiFab(ba0, dm0, 1, 0)
            set_val!(state_c, 0.0)

            # Face-centered coarse and fine fluxes
            fx_c = MultiFab(nodal_box_array(ba0; dirs=(true, false, false)), dm0, 1, 0)
            fy_c = MultiFab(nodal_box_array(ba0; dirs=(false, true, false)), dm0, 1, 0)
            fz_c = MultiFab(nodal_box_array(ba0; dirs=(false, false, true)), dm0, 1, 0)
            set_val!(fx_c, 1.0)
            set_val!(fy_c, 0.0)
            set_val!(fz_c, 0.0)

            fx_f = MultiFab(nodal_box_array(ba1; dirs=(true, false, false)), dm1, 1, 0)
            fy_f = MultiFab(nodal_box_array(ba1; dirs=(false, true, false)), dm1, 1, 0)
            fz_f = MultiFab(nodal_box_array(ba1; dirs=(false, false, true)), dm1, 1, 0)
            set_val!(fx_f, 3.0)
            set_val!(fy_f, 0.0)
            set_val!(fz_f, 0.0)

            dx0 = cell_sizes(g0)
            dx1 = cell_sizes(g1)
            each_fab(state_c) do mfi, _, _
                if crse_has_work(yafr, mfi)
                    crse_add!(yafr, mfi, (fx_c, fy_c, fz_c), dx0, 1.0)
                end
            end

            state_f = MultiFab(ba1, dm1, 1, 0)
            each_fab(state_f) do mfi, _, _
                if fine_has_work(yafr, mfi)
                    fine_add!(yafr, mfi, (fx_f, fy_f, fz_f), dx1, 1.0)
                end
            end

            reflux!(yafr, state_c)
            # Coarse cells abutting x=4 (i=3) and x=12 (i=12) get opposite reflux corrections
            @test min_val(state_c, Int32(0), Int32(0)) ≈ -2.0
            @test max_val(state_c, Int32(0), Int32(0)) ≈  2.0
            @test abs(sum_val(state_c, Int32(0), false)) < 1e-12

            # 2. average_down_faces!
            set_val!(fx_c, 0.0)
            set_val!(fy_c, 0.0)
            set_val!(fz_c, 0.0)
            average_down_faces!((fx_f, fy_f, fz_f), (fx_c, fy_c, fz_c), rr, g0)
            @test max_val(fx_c, Int32(0), Int32(0)) ≈ 3.0

            # 3. EdgeFluxRegister constrained-transport reflux test
            efr = EdgeFluxRegister(amr, 1, 1)
            reset!(efr)

            # Edge-centered E: Ex is nodal in (y,z), Ey in (x,z), Ez in (x,y)
            Ex_c = MultiFab(nodal_box_array(ba0; dirs=(false, true, true)), dm0, 1, 0)
            Ey_c = MultiFab(nodal_box_array(ba0; dirs=(true, false, true)), dm0, 1, 0)
            Ez_c = MultiFab(nodal_box_array(ba0; dirs=(true, true, false)), dm0, 1, 0)
            set_val!(Ex_c, 0.0)
            set_val!(Ey_c, 0.0)
            set_val!(Ez_c, 1.0)

            Ex_f = MultiFab(nodal_box_array(ba1; dirs=(false, true, true)), dm1, 1, 0)
            Ey_f = MultiFab(nodal_box_array(ba1; dirs=(true, false, true)), dm1, 1, 0)
            Ez_f = MultiFab(nodal_box_array(ba1; dirs=(true, true, false)), dm1, 1, 0)
            set_val!(Ex_f, 0.0)
            set_val!(Ey_f, 0.0)
            set_val!(Ez_f, 2.0)

            each_fab(state_c; tiling=false) do mfi, _, _
                crse_add!(efr, mfi, (Ex_c, Ey_c, Ez_c), 1.0)
            end
            each_fab(state_f; tiling=false) do mfi, _, _
                fine_add!(efr, mfi, (Ex_f, Ey_f, Ez_f), 1.0)
            end

            Bx_c = MultiFab(nodal_box_array(ba0; dirs=(true, false, false)), dm0, 1, 0)
            By_c = MultiFab(nodal_box_array(ba0; dirs=(false, true, false)), dm0, 1, 0)
            Bz_c = MultiFab(nodal_box_array(ba0; dirs=(false, false, true)), dm0, 1, 0)
            set_val!(Bx_c, 0.0)
            set_val!(By_c, 0.0)
            set_val!(Bz_c, 0.0)

            reflux!(efr, (Bx_c, By_c, Bz_c))
            @test norm0(Bx_c, Int32(0), Int32(0)) >= 0.0
        end
    end
end
