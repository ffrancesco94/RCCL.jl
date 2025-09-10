module RCCL

using Libdl, AMDGPU, CEnum


include("librccl.jl")
using .LibRCCL

include("base.jl")
include("communicator.jl")
include("group.jl")
include("pointtopoint.jl")
include("collective.jl")


end

