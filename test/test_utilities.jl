using Krylov: Krylov
using LinearAlgebra: dot, mul!, norm
using LinearOperators: LinearOperator

@testset "AmrCore Ergonomics, ScratchMultiFabPool, MPIDistVector & Plotfile Helpers" begin
    ensure_amrex_init()
    @test amrex_initialized()

    # 1. build_amrcore Fake-1D, Fake-2D, and 3D + cache_level_geometries + allocate_*
    amr1 = build_amrcore((16, 1, 1), (0.0, 0.0, 0.0), (16.0, 1.0, 1.0))
    @test n_levels(amr1) == 1
    @test cell_center((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0, 0, 0) == (0.5, 0.5, 0.5)

    amr2 = build_amrcore(
        (16, 16, 1), (-4.0, -4.0, 0.0), (4.0, 4.0, 1.0);
        is_periodic = (true, false, true),
        max_level = 1,
        refine_boxes = [(-1.0, -1.0, 0.0, 1.0, 1.0, 1.0)],
    )
    @test n_levels(amr2) == 2
    geoms2 = cache_level_geometries(amr2)
    @test length(geoms2) == 2

    mf_c = allocate_cell_mf(amr2, 0, 2, 1; init_val = 1.5)
    mf_f = allocate_face_mf(amr2, 0, 1, 1, 0; init_val = 2.0)
    mf_n = allocate_node_mf(amr2, 0, 1, 0; init_val = -1.0)
    imf_c = allocate_cell_imf(amr2, 0, 1, 0; init_val = Int32(3))
    imf_n = allocate_node_imf(amr2, 0, 1, 0; init_val = Int32(4))
    @test max_val(mf_c, 0, 0) ≈ 1.5
    @test min_val(mf_f, 0, 0) ≈ 2.0
    @test min_val(mf_n, 0, 0) ≈ -1.0
    @test min_val(imf_c, Int32(0), Int32(0)) == 3
    @test min_val(imf_n, Int32(0), Int32(0)) == 4

    # 2. ScratchMultiFabPool
    pool = ScratchMultiFabPool()
    mf1 = checkout_scratch_mf!(pool, amr2, 0, 3, 1; init_val = 2.0)
    mf2 = checkout_scratch_mf!(pool, amr2, 0, 3, 1; init_val = 5.0)
    @test pool.misses == 2 && pool.hits == 0
    release_scratch_mf!(pool, mf1, 0, 3, 1)
    release_scratch_mf!(pool, mf2, 0, 3, 1)

    with_scratch_mf!(pool, amr2, 0, 3, 1; init_val = 7.0) do mf
        @test max_val(mf, 0, 0) ≈ 7.0
    end
    @test pool.hits == 1 && pool.misses == 2

    invalidate_scratch_pool!(pool)
    mf3 = checkout_scratch_mf!(pool, amr2, 0, 3, 1)
    release_scratch_mf!(pool, mf3, 0, 3, 1)
    @test pool.misses == 3

    # 3. MPIDistVector + AMReXKrylovExt
    @test isconcretetype(MPIDistVector)
    x = MPIDistVector([1.0, 2.0, 3.0, 4.0])
    y = @. 2.0 * x + 1.0
    @test y isa MPIDistVector
    @test y.data == [3.0, 5.0, 7.0, 9.0]
    @. y = 0.5 * x
    @test y.data == [0.5, 1.0, 1.5, 2.0]
    @test dot(x, y) ≈ 15.0
    @test norm(y) ≈ sqrt(7.5)

    A_mat = [4.0 1.0 0.0 0.0; 1.0 4.0 1.0 0.0; 0.0 1.0 4.0 1.0; 0.0 0.0 1.0 4.0]
    b = MPIDistVector([1.0, 2.0, 3.0, 4.0])
    x_exact = A_mat \ b.data
    A_op = LinearOperator(Float64, 4, 4, true, true, (out, v) -> (mul!(out.data, A_mat, v.data); out))
    M_op = LinearOperator(Float64, 4, 4, true, true, (out, v) -> (@. out = 0.25 * v))

    sol1, stats1 = Krylov.gmres(A_op, b; M = M_op, rtol = 1.0e-12, atol = 1.0e-12)
    @test stats1.solved
    @test norm(sol1.data - x_exact) < 1.0e-11

    ws_g = Krylov.GmresWorkspace(4, 4, MPIDistVector)
    Krylov.gmres!(ws_g, A_op, b; rtol = 1.0e-12, atol = 1.0e-12)
    @test norm(ws_g.x.data - x_exact) < 1.0e-11

    ws_b = Krylov.BicgstabWorkspace(4, 4, MPIDistVector)
    Krylov.bicgstab!(ws_b, A_op, b; rtol = 1.0e-12, atol = 1.0e-12)
    @test norm(ws_b.x.data - x_exact) < 1.0e-11

    # 4. write_plotfile(dir, mfs, amr) + auto-mkpath parent dir
    mktempdir() do tmp
        plt1 = joinpath(tmp, "nested", "plt_single")
        write_plotfile(plt1, [mf_c], amr2; varnames = ["a", "b"], time = 0.0, step = 0)
        @test isfile(joinpath(plt1, "Header"))

        mf_c1 = allocate_cell_mf(amr2, 1, 2, 1; init_val = 2.5)
        plt2 = joinpath(tmp, "nested2", "plt_multi")
        write_plotfile(plt2, [mf_c, mf_c1], amr2; varnames = ["a", "b"], time = 0.5, step = 2)
        @test isfile(joinpath(plt2, "Header"))
    end
end
