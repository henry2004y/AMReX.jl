using AMReX_jll
using CxxWrap
using Libdl

function build_cshim(;
        custom_amrex_prefix::Union{Nothing, String} = nothing,
        out_dir::Union{Nothing, String} = nothing,
        cxx::String = get(ENV, "CXX", "g++")
    )
    deps_dir = @__DIR__
    src_file = joinpath(deps_dir, "cshim", "amrex_jl_shim.cpp")
    lib_dir = something(out_dir, joinpath(deps_dir, "lib"))
    mkpath(lib_dir)
    out_lib = joinpath(lib_dir, "libamrex_jl_shim." * Libdl.dlext)

    julia_inc = abspath(joinpath(Sys.BINDIR, "..", "include", "julia"))
    julia_lib = abspath(joinpath(Sys.BINDIR, "..", "lib"))

    cxxwrap_prefix = CxxWrap.prefix_path()
    cxxwrap_inc = joinpath(cxxwrap_prefix, "include")
    cxxwrap_lib = joinpath(cxxwrap_prefix, "lib")

    amrex_prefix = something(custom_amrex_prefix, AMReX_jll.artifact_dir)
    amrex_inc = joinpath(amrex_prefix, "include")
    amrex_lib = joinpath(amrex_prefix, "lib")

    inc_flags = String[
        "-I$julia_inc",
        "-I$cxxwrap_inc",
        "-I$amrex_inc",
    ]
    lib_flags = String[
        "-L$julia_lib",
        "-L$cxxwrap_lib",
        "-L$amrex_lib",
        "-Wl,-rpath,$julia_lib",
        "-Wl,-rpath,$cxxwrap_lib",
        "-Wl,-rpath,$amrex_lib",
    ]

    # Only include AMReX_jll transitive library paths (e.g., MPICH_jll) when building against AMReX_jll.
    if custom_amrex_prefix === nothing
        extra_libs = filter(isdir, AMReX_jll.LIBPATH_list)
        extra_incs = filter(isdir, [joinpath(dirname(p), "include") for p in extra_libs])
        for inc in extra_incs
            push!(inc_flags, "-I$inc")
        end
        for ldir in extra_libs
            push!(lib_flags, "-L$ldir")
            push!(lib_flags, "-Wl,-rpath,$ldir")
        end
    end

    tmp_lib = out_lib * ".tmp.$(getpid())"
    cmd_args = String[
        cxx,
        "-shared", "-fPIC", "-O3", "-DNDEBUG", "-std=c++20", "-fopenmp",
        src_file,
        "-o", tmp_lib,
        inc_flags...,
        lib_flags...,
        "-lamrex_3d",
        "-lcxxwrap_julia",
        "-ljulia",
        "-lmpi",
    ]

    println("Compiling $out_lib ...")
    try
        run(Cmd(cmd_args))
        mv(tmp_lib, out_lib; force = true)
    catch
        rm(tmp_lib; force = true)
        rethrow()
    end
    println("Successfully built $out_lib")
    return out_lib
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_cshim()
end
