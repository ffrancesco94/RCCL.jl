# Group calls
"""
    RCCL.groupStart()

Start a RCCL group call.
"""
groupStart() = ncclGroupStart()

"""
    RCCL.groupEnd()

End a RCCL group call.
"""
groupEnd() = ncclGroupEnd()

"""
    RCCL.group(f)

Evaluate `f()` between [`RCCL.groupStart()`](@ref) and [`RCCL.groupEnd()`](@ref).
"""
function group(f)
    groupStart()
    try
        f()
    finally
        groupEnd()
    end
end
