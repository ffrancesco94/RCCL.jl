using RCCL
using Documenter

DocMeta.setdocmeta!(RCCL, :DocTestSetup, :(using RCCL); recursive=true)

makedocs(;
    modules=[RCCL],
    authors="Francesco Fiusco (ffiusco94@gmail.com)",
    sitename="RCCL.jl",
    format=Documenter.HTML(;
        canonical="https://ffrancesco94.github.io/RCCL.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ffrancesco94/RCCL.jl",
    devbranch="main",
)
