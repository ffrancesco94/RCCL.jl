module RCCL

using Libdl, CEnum


include("RCCLLoader.jl")
export librccl, is_available

include("librccl.jl")
using .LibRCCL

include("version.jl")
include("base.jl")
include("communicator.jl")
include("group.jl")
include("pointtopoint.jl")
include("collective.jl")


end

