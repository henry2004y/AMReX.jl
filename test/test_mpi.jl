ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")

using Test
using AMReX

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

        # 4. Cross-Rank ParticleContainer redistribute! (particles migrating between Rank 0 and Rank 1)
        amr = AmrCore(
            rb, (16, 8, 8); max_level = 0, is_periodic = (true, true, true),
            max_grid_size = (8, 8, 8)
        )
        init_from_scratch!(amr, 0.0)
        pc = ParticleContainer(amr; n_real = 2, n_int = 1)

        each_fab(mf) do mfi, vbx, _
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
