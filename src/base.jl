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


