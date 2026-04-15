using RCCL
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(RCCL, :DocTestSetup, :(using RCCL, AMDGPU); recursive=true)

makedocs(;
    modules=[RCCL],
    authors="Francesco Fiusco (ffiusco94@gmail.com)",
    sitename="RCCL.jl",
    format=DocumenterVitepress.MarkdownVitepress(;
        canonical="https://ffrancesco94.github.io/RCCL.jl",
        edit_link="main",
        assets=String[],
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/ffrancesco94/RCCL.jl",
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
