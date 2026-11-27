function build_cshim(;
        custom_amrex_prefix::Union{Nothing, String} = nothing,
        out_dir::Union{Nothing, String} = nothing,
        cxx::String = get(ENV, "CXX", "g++")
    )
    if custom_amrex_prefix === nothing && !AMReX_jll.is_available()
        throw(
            ErrorException(
                "AMReX_jll is not available on this platform. " *
                    "Configure a local AMReX installation with AMReX.set_amrex_prefix!(prefix)."
            )
        )
    end
    if Sys.which(cxx) === nothing && !isfile(cxx)
        throw(
            ErrorException(
                "C++ compiler '$cxx' not found. Compiling the AMReX C++ shim requires a C++20 " *
                    "compiler with OpenMP support (set via ENV[\"CXX\"], defaulting to \"g++\")."
            )
        )
    end

    cshim_dir = joinpath(@__DIR__, "..", "deps", "cshim")
    src_file = joinpath(cshim_dir, "amrex_jl_shim.cpp")
    lib_dir = something(out_dir, _shim_scratch_dir(custom_amrex_prefix))
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
