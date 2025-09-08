using Clang.Generators, JuliaFormatter


# Headers to wrap
header_dir = "/opt/rocm/inclde/rccl/"

args = get_default_args()
push!(args, "=I$header_dir")
push!(args, "-D__HIP_PLATFORM_AMD__")
headers = joinpath(header_dir, "rccl.h")
#output_file = joinpath(@__DIR__, "..", "src", "librccl.jl")
options = load_options(joinpath(@__DIR__, "generator.toml"))
ctx = create_context(headers, args, options)
build!(ctx)

format_file(options["general"]["output_file_path"], YASStyle())
