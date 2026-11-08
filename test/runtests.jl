ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "1")

using Test
using AMReX

function parse_test_filters(args::Vector{String})
    filters = String[]
    for a in args
        if startswith(a, "--test=")
            push!(filters, lowercase(split(a, '='; limit=2)[2]))
        elseif !startswith(a, "-")
            push!(filters, lowercase(a))
        end
    end
    return filters
end

function should_run_suite(name::String, filters::Vector{String})
    isempty(filters) && return true
    lname = lowercase(name)
    return any(f -> occursin(f, lname), filters)
end

const TEST_FILTERS = parse_test_filters(ARGS)

if should_run_suite("hygiene", TEST_FILTERS)
    include("test_hygiene.jl")
end

const AMREX_SUITES = [
    ("geometry_box", "test_geometry_box.jl"),
    ("multifab",     "test_multifab.jl"),
    ("mlmg",         "test_mlmg.jl"),
    ("particles_io", "test_particles_io.jl"),
]

const ACTIVE_AMREX_SUITES = filter(s -> should_run_suite(s[1], TEST_FILTERS), AMREX_SUITES)

if !isempty(ACTIVE_AMREX_SUITES)
    if !amrex_initialized()
        amrex_init(false)
    end

    try
        @testset "AMReX.jl Modular Test Suite" begin
            for (_, file) in ACTIVE_AMREX_SUITES
                include(file)
            end
        end
    finally
        GC.gc()
        if amrex_initialized()
            amrex_finalize()
        end
    end
end

if should_run_suite("mpi", TEST_FILTERS)
    @testset "2-Process MPI Subprocess Suite" begin
        mpi_test_script = joinpath(@__DIR__, "test_mpi.jl")
        cmd = `$(AMReX.AMReX_jll.MPICH_jll.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) $mpi_test_script`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end
