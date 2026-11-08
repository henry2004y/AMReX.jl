using Aqua

@testset "Code Quality & Hygiene (Aqua.jl)" begin
    Aqua.test_all(
        AMReX;
        ambiguities = false,
        piracies = false,
        stale_deps = false,
        persistent_tasks = false,
        deps_compat = (check_extras = false, ignore = [:Libdl])
    )
    own_piracies = filter(Aqua.Piracy.hunt(AMReX)) do m
        !occursin("CxxWrap", string(m.sig))
    end
    @test isempty(own_piracies)
    @test isdefined(AMReX, :set_amrex_prefix!)
end
