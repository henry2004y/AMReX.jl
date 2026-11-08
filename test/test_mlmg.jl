@testset "Linear Solvers: MLMG, MLPoisson, MLABecLaplacian & MLNodeLaplacian" begin
    @testset "MLPoisson Single-Level & Multi-Level Composite Solve" begin
        let
            # 1. Single-level 3D Poisson solve: ∇² φ = -3 π² sin(π x) sin(π y) sin(π z) on [0,1]³
            # with homogeneous Dirichlet BCs (φ = 0 on ∂Ω)
            nx, ny, nz = 16, 16, 16
            bx = Box(0, 0, 0, nx - 1, ny - 1, nz - 1)
            rb = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            geom = Geometry(bx, rb; is_periodic = (false, false, false))
            ba = BoxArray(bx; max_grid_size = (8, 8, 8))
            dm = DistributionMapping(ba)

            sol = MultiFab(ba, dm, 1, 1)
            rhs = MultiFab(ba, dm, 1, 0)
            exact = MultiFab(ba, dm, 1, 0)
            set_val!(sol, 0.0)

            dx, dy, dz = cell_sizes(geom)
            each_fab(rhs) do mfi, vbx, rarr
                earr = array4(exact, mfi)
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    z = (k + 0.5) * dz
                    val = sin(pi * x) * sin(pi * y) * sin(pi * z)
                    earr[i, j, k, 1] = val
                    rarr[i, j, k, 1] = -3.0 * pi^2 * val
                end
            end

            info = LPInfo(; max_coarsening_level = 10)
            poisson = MLPoisson(geom, ba, dm; info = info)
            set_max_order!(poisson, 2)
            @test get_max_order(poisson) == 2
            set_domain_bc!(
                poisson, (LODirichlet, LODirichlet, LODirichlet),
                (LODirichlet, LODirichlet, LODirichlet)
            )
            set_level_bc!(poisson, 0, nothing)

            mlmg = MLMG(poisson; verbose = 0, max_iter = 100)
            @test num_amr_levels(mlmg) == 1
            res_norm = solve!(mlmg, sol, rhs; reltol = 1.0e-10, abstol = 0.0)
            @test res_norm < 1.0e-7
            @test get_final_residual(mlmg) < 1.0e-7

            # Verify L∞ error against analytical solution (2nd-order O(h²) accuracy on 16³)
            err = MultiFab(ba, dm, 1, 0)
            copy_from!(err, sol, Int32(0), Int32(0), Int32(1), Int32(0))
            saxpy!(err, -1.0, exact, 0, 0, 1, 0)
            @test norm0(err, Int32(0), Int32(0)) < 1.5e-2

            # Verify get_grad_solution!, get_fluxes!, comp_residual!, apply!
            gx = MultiFab(nodal_box_array(ba; dirs = (true, false, false)), dm, 1, 0)
            gy = MultiFab(nodal_box_array(ba; dirs = (false, true, false)), dm, 1, 0)
            gz = MultiFab(nodal_box_array(ba; dirs = (false, false, true)), dm, 1, 0)
            get_grad_solution!(mlmg, (gx, gy, gz))
            @test max_val(gx, Int32(0), Int32(0)) > 2.5

            cc_flux = MultiFab(ba, dm, 3, 0)
            get_fluxes!(mlmg, cc_flux)
            @test norm0(cc_flux, Int32(0), Int32(0)) > 2.5

            res_mf = MultiFab(ba, dm, 1, 0)
            comp_residual!(mlmg, res_mf, sol, rhs)
            @test norm0(res_mf, Int32(0), Int32(0)) < 1.0e-6

            l_out = MultiFab(ba, dm, 1, 0)
            apply!(mlmg, l_out, sol)
            saxpy!(l_out, -1.0, rhs, 0, 0, 1, 0)
            @test norm0(l_out, Int32(0), Int32(0)) < 1.0e-6

            # 2. Multi-level (2-level AmrCore) composite Poisson solve
            amr = AmrCore(
                rb, (16, 16, 16); max_level = 1, max_grid_size = (8, 8, 8),
                blocking_factor = (4, 4, 4), is_periodic = (false, false, false)
            )
            add_refine_box!(amr, Int32(0), 0.25, 0.25, 0.25, 0.75, 0.75, 0.75)
            init_from_scratch!(amr, 0.0)
            @test finest_level(amr) == 1

            ml_poisson = MLPoisson(amr; info = info)
            set_domain_bc!(
                ml_poisson, (LODirichlet, LODirichlet, LODirichlet),
                (LODirichlet, LODirichlet, LODirichlet)
            )
            sol0 = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 1)
            sol1 = MultiFab(box_array(amr, Int32(1)), dmap(amr, Int32(1)), 1, 1)
            rhs0 = MultiFab(box_array(amr, Int32(0)), dmap(amr, Int32(0)), 1, 0)
            rhs1 = MultiFab(box_array(amr, Int32(1)), dmap(amr, Int32(1)), 1, 0)
            set_val!(sol0, 0.0)
            set_val!(sol1, 0.0)
            set_val!(rhs0, -1.0)
            set_val!(rhs1, -1.0)
            set_level_bc!(ml_poisson, 0, nothing)
            set_level_bc!(ml_poisson, 1, nothing)

            ml_solver = MLMG(ml_poisson; verbose = 0)
            @test num_amr_levels(ml_solver) == 2
            ml_res = solve!(ml_solver, [sol0, sol1], [rhs0, rhs1]; reltol = 1.0e-10)
            @test ml_res < 1.0e-8
            @test max_val(sol1, Int32(0), Int32(0)) > 0.05
        end
    end

    @testset "MLABecLaplacian Helmholtz & MLNodeLaplacian Projection" begin
        let
            nx, ny, nz = 16, 16, 16
            bx = Box(0, 0, 0, nx - 1, ny - 1, nz - 1)
            rb = RealBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            geom = Geometry(bx, rb; is_periodic = (false, false, false))
            ba = BoxArray(bx; max_grid_size = (8, 8, 8))
            dm = DistributionMapping(ba)

            # 1. MLABecLaplacian: (A α - B ∇⋅(β ∇)) φ = rhs with A=1, α=2, B=1, β=1
            # For φ = sin(π x) sin(π y) sin(π z), (2 + 3 π²) φ = rhs
            abec = MLABecLaplacian(geom, ba, dm)
            set_domain_bc!(
                abec, (LODirichlet, LODirichlet, LODirichlet),
                (LODirichlet, LODirichlet, LODirichlet)
            )
            set_level_bc!(abec, 0, nothing)
            set_scalars!(abec, 1.0, 1.0)
            set_a_coeffs!(abec, 0, 2.0)
            set_b_coeffs!(abec, 0, 1.0)

            sol = MultiFab(ba, dm, 1, 1)
            rhs = MultiFab(ba, dm, 1, 0)
            set_val!(sol, 0.0)
            dx, dy, dz = cell_sizes(geom)
            factor = 2.0 + 3.0 * pi^2
            each_fab(rhs) do mfi, vbx, rarr
                rx, ry, rz = bounds(vbx)
                for k in rz, j in ry, i in rx
                    x = (i + 0.5) * dx
                    y = (j + 0.5) * dy
                    z = (k + 0.5) * dz
                    rarr[i, j, k, 1] = factor * sin(pi * x) * sin(pi * y) * sin(pi * z)
                end
            end

            mlmg_abec = MLMG(abec; verbose = 0)
            res_abec = solve!(mlmg_abec, sol, rhs; reltol = 1.0e-10)
            @test res_abec < 1.0e-6
            @test isapprox(max_val(sol, Int32(0), Int32(0)), 1.0; atol = 2.0e-2)

            # 2. MLNodeLaplacian: nodal Poisson / divergence projection
            ndlap = MLNodeLaplacian(geom, ba, dm; const_sigma = 1.0)
            set_domain_bc!(
                ndlap, (LODirichlet, LODirichlet, LODirichlet),
                (LODirichlet, LODirichlet, LODirichlet)
            )
            nba = nodal_box_array(ba)
            nd_sol = MultiFab(nba, dm, 1, 1)
            nd_rhs = MultiFab(nba, dm, 1, 0)
            set_val!(nd_sol, 0.0)
            set_val!(nd_rhs, -1.0)

            mlmg_nd = MLMG(ndlap; verbose = 0)
            res_nd = solve!(mlmg_nd, nd_sol, nd_rhs; reltol = 1.0e-10)
            @test res_nd < 1.0e-7
            @test max_val(nd_sol, Int32(0), Int32(0)) > 0.05
        end
    end
end
