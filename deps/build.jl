using AMReX_jll
using CxxWrap

function build_cshim(;
        custom_amrex_prefix::Union{Nothing, String} = nothing,
        cxx::String = get(ENV, "CXX", "g++")
    )
    deps_dir = @__DIR__
    src_file = joinpath(deps_dir, "cshim", "amrex_jl_shim.cpp")
    lib_dir = joinpath(deps_dir, "lib")
    mkpath(lib_dir)
    out_lib = joinpath(lib_dir, "libamrex_jl_shim.so")

    julia_inc = abspath(joinpath(Sys.BINDIR, "..", "include", "julia"))
    julia_lib = abspath(joinpath(Sys.BINDIR, "..", "lib"))

    cxxwrap_prefix = CxxWrap.prefix_path()
    cxxwrap_inc = joinpath(cxxwrap_prefix, "include")
    cxxwrap_lib = joinpath(cxxwrap_prefix, "lib")

    amrex_prefix = something(custom_amrex_prefix, AMReX_jll.artifact_dir)
    amrex_inc = joinpath(amrex_prefix, "include")
    amrex_lib = joinpath(amrex_prefix, "lib")

    extra_libs = filter(isdir, AMReX_jll.LIBPATH_list)
    extra_incs = filter(isdir, [joinpath(dirname(p), "include") for p in extra_libs])

    inc_flags = String[
        "-I$julia_inc",
        "-I$cxxwrap_inc",
        "-I$amrex_inc",
    ]
    for inc in extra_incs
        push!(inc_flags, "-I$inc")
    end

    lib_flags = String[
        "-L$julia_lib",
        "-L$cxxwrap_lib",
        "-L$amrex_lib",
        "-Wl,-rpath,$julia_lib",
        "-Wl,-rpath,$cxxwrap_lib",
        "-Wl,-rpath,$amrex_lib",
    ]
    for ldir in extra_libs
        push!(lib_flags, "-L$ldir")
        push!(lib_flags, "-Wl,-rpath,$ldir")
    end

    cmd_args = String[
        cxx,
        "-shared", "-fPIC", "-O3", "-std=c++20", "-fopenmp",
        src_file,
        "-o", out_lib,
        inc_flags...,
        lib_flags...,
        "-lamrex_3d",
        "-lcxxwrap_julia",
        "-ljulia",
        "-lmpi",
    ]

    println("Compiling $out_lib ...")
    run(Cmd(cmd_args))
    println("Successfully built $out_lib")
    return out_lib
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_cshim()
end
