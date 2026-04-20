using RCCL
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(RCCL, :DocTestSetup, :(using RCCL, AMDGPU); recursive=true)

makedocs(;
    modules=[RCCL],
    authors="Francesco Fiusco (ffiusco94@gmail.com)",
    sitename="RCCL.jl",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/ffrancesco94/RCCL.jl",
        devbranch="main",
        devurl="dev"
    ),
    pages=[
        "Home" => "index.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/ffrancesco94/RCCL.jl",
    branch="gh-pages",
    target = joinpath(@__DIR__, "build"),
    devbranch="main",
    push_preview=true,
)
