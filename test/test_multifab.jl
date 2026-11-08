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
end
