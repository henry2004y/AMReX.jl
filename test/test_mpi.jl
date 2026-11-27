ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")

using Test
using AMReX
using Krylov
using LinearAlgebra: mul!, norm
using LinearOperators: LinearOperator

if !amrex_initialized()
    amrex_init(false)
end

try
    @testset "2-Process MPI Parallel Execution Suite" begin
        nprocs = Int(parallel_nprocs())
        myproc = Int(parallel_myproc())
        @test nprocs == 2
        @test 0 <= myproc < 2
        @test mpi_rank() == myproc

        # 1. Parallel Reductions across 2 MPI ranks
        @test parallel_reduce_sum_real(Float64(myproc + 1)) ≈ 3.0
        @test parallel_reduce_max_real(Float64(myproc + 10)) ≈ 11.0
        @test parallel_reduce_min_real(Float64(myproc + 10)) ≈ 10.0
        @test parallel_reduce_sum_long(Int64(myproc + 100)) == 201

        vec = [1.0, Float64(myproc + 1)]
        parallel_reduce_sum_real_vec!(vec)
        @test vec ≈ [2.0, 3.0]

        # 2. Multi-Box DistributionMapping across 2 MPI ranks
        dom = Box(0, 0, 0, 15, 7, 7)
        rb = RealBox(0.0, 0.0, 0.0, 16.0, 8.0, 8.0)
        geom_per = Geometry(dom, rb; is_periodic = (true, true, true))

        ba = BoxArray(dom; max_grid_size = 8)
        @test size(ba) == 2
        dm = DistributionMapping(ba)
        @test size(dm) == 2
        r0 = Int(get_rank(dm, 0))
        r1 = Int(get_rank(dm, 1))
        @test Set([r0, r1]) == Set([0, 1])

        # 3. Cross-Rank fill_boundary!, sum_boundary!, and parallel_copy_from!
        mf = MultiFab(ba, dm, 1, 1)
        @test local_size(mf) == 1
        set_val!(mf, 0.0)

        my_val = Float64(10 * (myproc + 1))
        each_fab(mf) do _, vbx, arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                arr[i, j, k, 1] = my_val
            end
        end

        fill_boundary!(mf, geom_per)

        other_val = Float64(10 * ((1 - myproc) + 1))
        each_fab(mf) do _, vbx, arr
            rx, ry, rz = bounds(vbx)
            # Check ghost cell across the interior rank interface (i = 7 or i = 8)
            if first(rx) == 0
                @test arr[8, 0, 0, 1] ≈ other_val
                @test arr[-1, 0, 0, 1] ≈ other_val
            else
                @test arr[7, 0, 0, 1] ≈ other_val
                @test arr[16, 0, 0, 1] ≈ other_val
            end
        end

        # Nodal sum_boundary! across shared interface at i = 8
        nba = nodal_box_array(ba)
        nmf = MultiFab(nba, dm, 1, 0)
        set_val!(nmf, 1.0)
        sum_boundary!(nmf, geom_per)
        each_fab(nmf) do _, vbx, arr
            rx, _, _ = bounds(vbx)
            if 8 in rx
                # Shared interface between box 0 and box 1 (and periodic at 0/16)
                @test arr[8, 2, 2, 1] ≈ 2.0
            end
            # Interior x-node not on box boundary
            mid_i = first(rx) + 2
            @test arr[mid_i, 2, 2, 1] ≈ 1.0
        end

        # Cross-rank parallel_copy_from! between different BoxArray decompositions
        ba_alt = BoxArray(dom; max_grid_size = (16, 4, 8))
        dm_alt = DistributionMapping(ba_alt)
        mf_alt = MultiFab(ba_alt, dm_alt, 1, 0)
        set_val!(mf_alt, -1.0)
        parallel_copy_from!(mf_alt, mf, 0, 0, 1, 0, 0, geom_per)
        @test min_val(mf_alt, 0, 0) ≈ 10.0
        @test max_val(mf_alt, 0, 0) ≈ 20.0

        # Non-blocking fill_boundary_nowait! + fill_boundary_finish! across 2 ranks
        mf_nb = MultiFab(ba, dm, 1, 1)
        set_val!(mf_nb, 0.0)
        each_fab(mf_nb) do _, vbx, arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                arr[i, j, k, 1] = my_val
            end
        end
        fill_boundary_nowait!(mf_nb, geom_per)
        fill_boundary_finish!(mf_nb)
        each_fab(mf_nb) do _, vbx, arr
            rx, _, _ = bounds(vbx)
            if first(rx) == 0
                @test arr[8, 0, 0, 1] ≈ other_val
            else
                @test arr[7, 0, 0, 1] ≈ other_val
            end
        end

        # Native MPI-global MultiFab dot product
        @test dot_mf(mf, mf; comp1 = 0, comp2 = 0, ncomp = 1, nghost = 0) ≈ (10.0^2 + 20.0^2) * 512.0

        # 3b. 2-Rank Multi-Level AMR: make_fine_mask, coarsened_multifab + parallel_copy!,
        #     weighted DistributionMapping, and time-interpolated fill_patch_two_levels!
        rb_amr = RealBox(0.0, 0.0, 0.0, 16.0, 16.0, 1.0)
        amr_ml = AmrCore(
            rb_amr, (16, 16, 1); max_level = 1,
            max_grid_size = (8, 8, 1), blocking_factor = (4, 4, 1)
        )
        add_tag_box!(amr_ml, 4.0, 4.0, 0.0, 12.0, 12.0, 1.0)
        init_from_scratch!(amr_ml, 0.0)
        @test finest_level(amr_ml) == 1

        g0_ml = geom(amr_ml, Int32(0)); g1_ml = geom(amr_ml, Int32(1))
        ba0_ml = box_array(amr_ml, Int32(0)); dm0_ml = dmap(amr_ml, Int32(0))
        ba1_ml = box_array(amr_ml, Int32(1)); dm1_ml = dmap(amr_ml, Int32(1))
        rr_ml = ref_ratio_tuple(amr_ml, 0)

        fmask = make_fine_mask(amr_ml, 0; crse_val = 0, fine_val = 1)
        @test min_val(fmask, Int32(0), Int32(0)) == 0
        @test max_val(fmask, Int32(0), Int32(0)) == 1

        # MPI-safe coarse -> fine staging via coarsened_multifab + parallel_copy!
        mf_c0 = MultiFab(ba0_ml, dm0_ml, 1, 1)
        each_fab(mf_c0) do _, vbx, arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                arr[i, j, k, 1] = Float64(100 * i + j)
            end
        end
        fill_boundary!(mf_c0, g0_ml)

        mf_stage_c = coarsened_multifab(amr_ml, 1, 1, (1, 1, 0))
        set_val!(mf_stage_c, -1.0)
        parallel_copy!(mf_stage_c, mf_c0, g0_ml; snghost = (1, 1, 0), dnghost = (1, 1, 0))
        each_fab(mf_stage_c) do _, vbx, arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                @test arr[i, j, k, 1] ≈ Float64(100 * i + j)
            end
        end

        # MPI-safe fine -> coarse staging via coarsened_multifab + parallel_copy!
        set_val!(mf_stage_c, 77.0)
        parallel_copy!(mf_c0, mf_stage_c, g0_ml; snghost = 0, dnghost = 0)
        each_fab(mf_c0) do mfi, vbx, arr
            m_arr = iarray4(fmask, mfi)
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                if m_arr[i, j, k, 1] == 1
                    @test arr[i, j, k, 1] ≈ 77.0
                else
                    @test arr[i, j, k, 1] ≈ Float64(100 * i + j)
                end
            end
        end

        # Weighted Knapsack & SFC DistributionMapping across 2 ranks
        nboxes0 = Int(AMReX.size(ba0_ml))
        costs0 = [Float64(b) for b in 1:nboxes0]
        dm_ks = make_knapsack_dmap(costs0)
        dm_sfc = make_sfc_dmap(costs0, ba0_ml)
        @test Int(AMReX.size(dm_ks)) == nboxes0
        @test Int(AMReX.size(dm_sfc)) == nboxes0
        @test Set([Int(get_rank(dm_ks, b - 1)) for b in 1:nboxes0]) == Set([0, 1])

        # Time-interpolated fill_patch_two_levels! across 2 ranks
        cmf_t0 = MultiFab(ba0_ml, dm0_ml, 1, 1); set_val!(cmf_t0, 4.0)
        cmf_t1 = MultiFab(ba0_ml, dm0_ml, 1, 1); set_val!(cmf_t1, 8.0)
        fmf_t = MultiFab(ba1_ml, dm1_ml, 1, 1); set_val!(fmf_t, 6.0)
        dst_t = MultiFab(ba1_ml, dm1_ml, 1, 1); set_val!(dst_t, 0.0)
        fill_patch_two_levels!(
            dst_t, 0.5, cmf_t0, 0.0, cmf_t1, 1.0, fmf_t, 0.5,
            0, 0, 1, g0_ml, g1_ml, rr_ml, CellConservativeLinear
        )
        @test min_val(dst_t, 0, 1) ≈ 6.0
        @test max_val(dst_t, 0, 1) ≈ 6.0

        # 4. Cross-Rank ParticleContainer redistribute! (particles migrating between Rank 0 and Rank 1)
        amr = AmrCore(
            rb, (16, 8, 8); max_level = 0, is_periodic = (true, true, true),
            max_grid_size = (8, 8, 8)
        )
        init_from_scratch!(amr, 0.0)
        pc = ParticleContainer(amr; n_real = 2, n_int = 1)
        mf_pc = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 0)

        each_fab(mf_pc) do mfi, vbx, _
            gid = index(mfi)
            tid = local_tile_index(mfi)
            rx, _, _ = bounds(vbx)
            x_base = Float64(first(rx))
            resize_tile!(pc, Int32(0), gid, tid, Int64(10))
            idcpus = tile_idcpu_vec(pc, 0, gid, tid)
            px, py, pz = tile_pos_vecs(pc, 0, gid, tid)
            w = tile_real_vec(pc, 0, gid, tid, 1)
            orig_rank = tile_int_vec(pc, 0, gid, tid, 1)
            for p in 1:10
                pid = next_particle_id()
                idcpus[p] = pack_idcpu(pid, myproc)
                # Place in the OTHER rank's half of the domain so redistribute! must send via MPI
                px[p] = mod(x_base + 0.5 * p + 8.0, 16.0)
                py[p] = 4.0
                pz[p] = 4.0
                w[p] = Float64(myproc + 1) * 5.0
                orig_rank[p] = Int32(myproc)
            end
        end

        redistribute!(pc)
        @test total_number_of_particles(pc; only_valid = true, local_only = false) == 20
        @test total_number_of_particles(pc; only_valid = true, local_only = true) == 10

        each_particle_tile(pc, 0) do pti
            np = num_particles(pti)
            @test np == 10
            idcpus = idcpu_vec(pti)
            w = real_vec(pti, 1)
            orig_rank = int_vec(pti, 1)
            expected_sender = 1 - myproc
            for p in 1:np
                @test unpack_cpu(idcpus[p]) == expected_sender
                @test orig_rank[p] == Int32(expected_sender)
                @test w[p] ≈ Float64(expected_sender + 1) * 5.0
            end
        end

        # 5. 2-Rank MLMG MLPoisson Solve
        dom_mg = Box(0, 0, 0, 15, 15, 15)
        rb_mg = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        geom_mg = Geometry(dom_mg, rb_mg; is_periodic = (false, false, false))
        ba_mg = BoxArray(dom_mg; max_grid_size = 8)
        dm_mg = DistributionMapping(ba_mg)

        sol = MultiFab(ba_mg, dm_mg, 1, 1)
        rhs = MultiFab(ba_mg, dm_mg, 1, 0)
        exact = MultiFab(ba_mg, dm_mg, 1, 0)
        set_val!(sol, 0.0)

        dx, dy, dz = cell_sizes(geom_mg)
        each_fab(rhs) do _, vbx, r_arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                z = (k + 0.5) * dz
                r_arr[i, j, k, 1] = -3.0 * π^2 * sin(π * x) * sin(π * y) * sin(π * z)
            end
        end
        each_fab(exact) do _, vbx, e_arr
            rx, ry, rz = bounds(vbx)
            for k in rz, j in ry, i in rx
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                z = (k + 0.5) * dz
                e_arr[i, j, k, 1] = sin(π * x) * sin(π * y) * sin(π * z)
            end
        end

        poisson = MLPoisson(geom_mg, ba_mg, dm_mg)
        set_domain_bc!(
            poisson,
            (LODirichlet, LODirichlet, LODirichlet),
            (LODirichlet, LODirichlet, LODirichlet)
        )
        set_level_bc!(poisson, 0, nothing)
        mlmg = MLMG(poisson; verbose = 0)
        res = solve!(mlmg, sol, rhs; reltol = 1.0e-10, abstol = 0.0)
        @test res < 1.0e-7
        saxpy!(exact, -1.0, sol, 0, 0, 1, 0)
        @test norm0(exact, 0, 0) < 5.0e-3

        # 6. 2-Rank Collective Plotfile, VisMF & Particle Checkpoint/Restart I/O
        tok = myproc == 0 ? Int64(rand(100_000:999_999_999)) : Int64(0)
        shared_tok = parallel_reduce_sum_long(tok)
        tmpdir = joinpath(tempdir(), "amrex_jl_mpi_test_$(shared_tok)")
        if parallel_ioprocessor()
            mkpath(tmpdir)
        end
        parallel_barrier()

        try
            plt_dir = joinpath(tmpdir, "plt_mpi")
            write_plotfile(plt_dir, sol, geom_mg; varnames = ["phi"], time = 0.5, step = 1)
            parallel_barrier()
            @test isfile(joinpath(plt_dir, "Header"))

            vismf_path = joinpath(tmpdir, "vismf", "sol")
            vismf_write(sol, vismf_path)
            parallel_barrier()
            sol_read = MultiFab(ba_mg, dm_mg, 1, 1)
            set_val!(sol_read, 0.0)
            vismf_read!(sol_read, vismf_path)
            saxpy!(sol_read, -1.0, sol, 0, 0, 1, 0)
            @test norm0(sol_read, 0, 0) < 1.0e-14

            chk_dir = joinpath(tmpdir, "chk_mpi")
            checkpoint(pc, chk_dir, "particles")
            parallel_barrier()
            pc2 = ParticleContainer(amr; n_real = 2, n_int = 1)
            restart!(pc2, chk_dir, "particles")
            @test total_number_of_particles(pc2; only_valid = true, local_only = false) == 20

            # 7. ABI-Portable MPI Point-to-Point & Collective Primitives
            reqs = MPIRequestBuffer(2)
            ensure_requests!(reqs, 4)
            peer = 1 - myproc
            send_explicit = Float64[100.0 * (myproc + 1), 200.0 * (myproc + 1)]
            recv_explicit = zeros(Float64, 2)
            parallel_irecv!(recv_explicit, peer, 701, reqs, 1)
            parallel_isend!(send_explicit, peer, 701, reqs, 2)
            parallel_waitall!(reqs, 2)
            @test recv_explicit ≈ Float64[100.0 * (peer + 1), 200.0 * (peer + 1)]

            send_any = Int32[myproc + 10, myproc + 20]
            recv_any = zeros(Int32, 2)
            parallel_irecv!(recv_any, PARALLEL_ANY_SOURCE, 702, reqs, 1)
            parallel_isend!(send_any, peer, 702, reqs, 2)
            parallel_waitall!(reqs, 2)
            @test recv_any == Int32[peer + 10, peer + 20]

            gather_send = Float64[Float64(myproc + 1)]
            gather_recv = zeros(Float64, 2)
            parallel_allgather!(gather_recv, gather_send)
            @test gather_recv ≈ [1.0, 2.0]

            # 8. MPIDistVector + Krylov & solve_cell_poisson_mlmg! across 2 ranks
            b_loc = MPIDistVector([1.0 + myproc, 2.0 + myproc])
            A_loc = [3.0 0.5; 0.5 4.0]
            A_op = LinearOperator(Float64, 2, 2, true, true, (out, v) -> (mul!(out.data, A_loc, v.data); out))
            sol_cg, stats_cg = Krylov.cg(A_op, b_loc; rtol = 1.0e-12, atol = 1.0e-12)
            @test stats_cg.solved
            @test norm(A_loc * sol_cg.data - b_loc.data) < 1.0e-11

            sol_g, stats_g = Krylov.gmres(A_op, b_loc; rtol = 1.0e-12, atol = 1.0e-12)
            @test stats_g.solved
            @test norm(A_loc * sol_g.data - b_loc.data) < 1.0e-11

            amr_p = build_amrcore((16, 16, 1), (0.0, 0.0, 0.0), (2π, 2π, 1.0); is_periodic = (true, false, true), max_grid_size = 8)
            rhs_p = allocate_cell_mf(amr_p, 0, 1, 0)
            sol_p = allocate_cell_mf(amr_p, 0, 1, 1; init_val = 0.0)
            geom_p = geometry(amr_p, 0)
            dx_p = cell_sizes(geom_p)
            plo_p = prob_lo_tuple(geom_p)
            each_fab(rhs_p) do _, vbx, r_arr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    xc, yc, _ = cell_center(plo_p, dx_p, i, j, k)
                    r_arr[i, j, k, 1] = -cos(xc) * sin(0.5 * yc)
                end
            end
            solve_cell_poisson_mlmg!(
                sol_p, rhs_p, geom_p, box_array(amr_p, Int32(0)), dmap(amr_p, Int32(0));
                is_periodic = (true, false, true), hidden_direction = 2,
                subtract_mean = false, reltol = 1.0e-10,
            )
            @test isfinite(norm2(sol_p, 0, geom_p))
        finally
            parallel_barrier()
            if parallel_ioprocessor()
                rm(tmpdir; recursive = true, force = true)
            end
            parallel_barrier()
        end
    end
finally
    GC.gc()
    if amrex_initialized()
        amrex_finalize()
    end
end
