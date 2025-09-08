module RCCL

using Libdl
export is_available, librrcl_path

# Ok, let's find out where ROCm is 
const DEFAULT_ROCM_PATH = Ref{String}("/opt/rocm")

# Well, the user can point us to where it is 
if haskey(env, "ROCM_PATH")
    DEFAULT_ROCM_PATH[] = ENV["ROCM_PATH"]
end

const LIB_NAME = "librccl.so"
const _cached_lib_path = Ref{Union{Nothing,String}}(nothing)

"""
    librccl_path -> Union{String, nothing}

Return the absolute path to `librccl.so` if it can be found. We look inside:
1. `$RCCL_LIBRARY_PATH` if it exists
2. `$ROCM_PATH/lib` 
3. `$ROCM_PATH/lib64`
4. Let the system figure it out from `$LD_LIBRARY_PATH`. 
"""
function librrcl_path()
    # 0. cached
    if _cached_lib_path[] !== nothing
        return _cached_lib_path[]
    end

    # 1. RCCL_LIBRARY_PATH
    if haskey(ENV, "RCLL_LIBRARY_PATH")
        p = ENV["RCLL_LIBRARY_PATH"]
        isfile(p) && (return _cached_lib_path[] = p)
    end

    # 2. Search inside ROCM_PATH 
    roots = [DEFAULT_ROCM_PATH[], joinpath(DEFAULT_ROCM_PATH[], "lib"), joinpath(DEFAULT_ROCM_PATH[], "lib64")]
    for r in roots
        candidate = joinpath(r, LIB_NAME)
        if isfile(candidate)
            return _cached_lib_path[] = candidate
        end
    end

    # 3. Duck tape and prayers - Let the system figure it out 
    _cached_lib_path = LIB_NAME
    return _cached_lib_path[]
end


"""
    is_available -> Bool 

Return `true` if I can load `librccl.so`, otherwise `false`. 
We also cache the result so that calling this in the same Julia session is quick.
"""
function is_available()
    lib = librccl_path()
    try
        handle = Libdl.dlopen(lib)
        Libdl.close(handle)
        return true
    catch err
        @debug "RCCL library not found" exception = (err, catch_backtrace())
        return false
    end
end


"""
    RCCL.version() :: VersionNumber

Get the version of the current RCCL library.
"""
function version()
    ver_r = Ref{Cint}()
    ncclGetVersion(ver_r)
    ver = ver_r[]

    if ver < 2900
        major, ver = divrem(ver, 1000)
        minor, patch = divrem(ver, 100)
    else
        major, ver = divrem(ver, 10000)
        minor, patch = divrem(ver, 100)
    end
    VersionNumber(major, minor, patch)
end

end

import .LibRCCL: ncclRedOp_t, ncclDataType_t

ncclRedOp_t(::typeof(+)) = ncclSum
ncclRedOp_t(::typeof(*)) = ncclProd
ncclRedOp_t(::typeof(max)) = ncclMax
ncclRedOp_t(::typeof(min)) = ncclMin
# Handles the case where user directly passed in the ncclRedOp_t (eg. `NCCL.avg`)
ncclRedOp_t(x::ncclRedOp_t) = x

"""
    RCCL.avg

Perform an average (over the ranks) operation.
"""

const avg = ncclAvg

ncclDataType_t(::Type{Int8}) = ncclInt8
ncclDataType_t(::Type{UInt8}) = ncclUint8
ncclDataType_t(::Type{Int32}) = ncclInt32
ncclDataType_t(::Type{UInt32}) = ncclUint32
ncclDataType_t(::Type{Int64}) = ncclInt64
ncclDataType_t(::Type{UInt64}) = ncclUint64
ncclDataType_t(::Type{Float16}) = ncclFloat16
ncclDataType_t(::Type{Float32}) = ncclFloat32
ncclDataType_t(::Type{Float64}) = ncclFloat64


