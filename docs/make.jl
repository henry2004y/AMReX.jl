using Documenter
using DocumenterVitepress
using AMReX

DocMeta.setdocmeta!(AMReX, :DocTestSetup, :(using AMReX); recursive=true)

makedocs(;
    modules = [AMReX],
    authors = "Hongyang Zhou <hyzhou@umich.edu>",
    repo = Remotes.GitHub("henry2004y", "AMReX.jl"),
    sitename = "AMReX.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/henry2004y/AMReX.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = get(ENV, "BUILD_VITEPRESS", "true") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting_started.md",
            "MultiFab & Zero-Copy Iterators" => "multifab_iterators.md",
            "Multi-Level AMR & Particles" => "amr_particles.md",
            "GPU & Custom AMReX Builds" => "gpu_preferences.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/henry2004y/AMReX.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)

