module RCCL

using Libdl, AMDGPU, CEnum


include("RCCLLoader.jl")
using .RCCLLoader
include("librccl.jl")
using .LibRCCL

include("version.jl")
include("base.jl")
include("communicator.jl")
include("group.jl")
include("pointtopoint.jl")
include("collective.jl")


end

