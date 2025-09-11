using Clang.Generators, JuliaFormatter, Clang

cd(@__DIR__)
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using RCCLLoader: RCCL_HANDLE


# Headers to wrap
#header_dir = "/opt/rocm/include/"
header_dir = "/opt/rocm/include/rccl"

#=function discover_header_dirs(root::String)=#
#=    dirs = String[]=#
#=    for entry in readdir(root; join=true)=#
#=        isdir(entry) || continue=#
#=        any(endswith.(readdir(entry), ".h")) && push!(dirs, entry)=#
#=    end=#
#=    return dirs=#
#=end=#
#==#
#=header_dirs = discover_header_dirs(header_dir)=#
#=println(header_dirs)=#

args = get_default_args()
#=for d in header_dirs=#
#=    push!(args, "-I$d")=#
#=end=#
push!(args, "-I$header_dir")
push!(args, "-D__HIP_PLATFORM_AMD__")
#headers = joinpath(header_dir, "rccl", "rccl.h")
headers = joinpath(header_dir, "rccl.h")
#output_file = joinpath(@__DIR__, "..", "src", "librccl.jl")
options = load_options(joinpath(@__DIR__, "generator.toml"))


function rewriter!(ctx, options)
    for node in get_nodes(ctx.dag)
        if Generators.is_function(node) && !Generators.is_variadic_function(node)
            expr = node.exprs[1]
            call_expr = expr.args[2].args[1].args[3]    # assumes `@ccall`

            # rewrite pointer argument types
            arg_exprs = call_expr.args[1].args[2:end]
            #=for expr in arg_exprs=#
            #=    if expr.args[1] == :stream=#
            #=        expr.args[2] = :(HIP.stream)=#
            #=    end=#
            #=end=#

            rettyp = call_expr.args[2]
            if rettyp isa Symbol && String(rettyp) == "ncclResult_t"
                call_expr = Expr(:macrocall, Symbol("@check"), nothing, expr)
            end
        end
    end
end

using AMDGPU
ctx = create_context(headers, args, options)
build!(ctx, BUILDSTAGE_NO_PRINTING)
rewriter!(ctx, options)
build!(ctx, BUILDSTAGE_PRINTING_ONLY)

format_file(options["general"]["output_file_path"], YASStyle())
