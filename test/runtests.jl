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

if !amrex_initialized()
    amrex_init(false)
end

try
    @testset "AMReX.jl Modular Test Suite" begin
        for (suite_name, file) in [
            ("geometry_box", "test_geometry_box.jl"),
            ("multifab",     "test_multifab.jl"),
            ("particles_io", "test_particles_io.jl"),
        ]
            if should_run_suite(suite_name, TEST_FILTERS)
                include(file)
            end
        end
    end
finally
    if amrex_initialized()
        amrex_finalize()
    end
end
