module LibRCCL

using CEnum: CEnum, @cenum

const NULL = C_NULL
const INT_MIN = typemin(Cint)

using RCCL.RCCLLoader: librccl

#import AMDGPU: @check

@cenum ncclResult_t::UInt32 begin
    ncclSuccess = 0
    ncclUnhandledCudaError = 1
    ncclSystemError = 2
    ncclInternalError = 3
    ncclInvalidArgument = 4
    ncclInvalidUsage = 5
    ncclRemoteError = 6
    ncclInProgress = 7
    ncclNumResults = 8
end


function check(status::ncclResult_t)
    if status != ncclSuccess
        throw(NCCLError(status))
    end
    return
end

#=
function check(::ncclResult_t)
	return
end
=#

macro check(f)
    quote
        local err
        err = $(esc(f::Expr))
        $check(err)
        err
    end
end

struct ncclConfig_v21700
    size::Cint
    magic::Cuint
    version::Cuint
    blocking::Cint
    cgaClusterSize::Cint
    minCTAs::Cint
    maxCTAs::Cint
    netName::Cstring
    splitShare::Cint
end

const ncclConfig_t = ncclConfig_v21700

struct ncclSimInfo_v22200
    size::Cint
    magic::Cuint
    version::Cuint
    estimatedTime::Cfloat
end

const ncclSimInfo_t = ncclSimInfo_v22200

mutable struct ncclComm end

const ncclComm_t = Ptr{ncclComm}

struct ncclUniqueId
    internal::NTuple{128,Cchar}
end

function ncclMemAlloc(ptr, size)
     @check @ccall librccl.ncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Cint)::ncclResult_t
end

function pncclMemAlloc(ptr, size)
     @check @ccall librccl.pncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Cint)::ncclResult_t
end

function ncclMemFree(ptr)
     @check @ccall librccl.ncclMemFree(ptr::Ptr{Cvoid})::ncclResult_t
end

function pncclMemFree(ptr)
     @check @ccall librccl.pncclMemFree(ptr::Ptr{Cvoid})::ncclResult_t
end

function ncclGetVersion(version)
     @check @ccall librccl.ncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

function pncclGetVersion(version)
     @check @ccall librccl.pncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

function ncclGetUniqueId(uniqueId)
     @check @ccall librccl.ncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

function pncclGetUniqueId(uniqueId)
     @check @ccall librccl.pncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

function ncclCommInitRankConfig(comm, nranks, commId, rank, config)
     @check @ccall librccl.ncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
        commId::ncclUniqueId, rank::Cint,
        config::Ptr{ncclConfig_t})::ncclResult_t
end

function pncclCommInitRankConfig(comm, nranks, commId, rank, config)
     @check @ccall librccl.pncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
        commId::ncclUniqueId, rank::Cint,
        config::Ptr{ncclConfig_t})::ncclResult_t
end

function ncclCommInitRank(comm, nranks, commId, rank)
     @check @ccall librccl.ncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
        commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

function pncclCommInitRank(comm, nranks, commId, rank)
     @check @ccall librccl.pncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
        commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

function ncclCommInitAll(comm, ndev, devlist)
     @check @ccall librccl.ncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
        devlist::Ptr{Cint})::ncclResult_t
end

function pncclCommInitAll(comm, ndev, devlist)
     @check @ccall librccl.pncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
        devlist::Ptr{Cint})::ncclResult_t
end

function ncclCommFinalize(comm)
     @check @ccall librccl.ncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

function pncclCommFinalize(comm)
     @check @ccall librccl.pncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

function ncclCommDestroy(comm)
     @check @ccall librccl.ncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

function pncclCommDestroy(comm)
     @check @ccall librccl.pncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

function ncclCommAbort(comm)
     @check @ccall librccl.ncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

function pncclCommAbort(comm)
     @check @ccall librccl.pncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

function ncclCommSplit(comm, color, key, newcomm, config)
     @check @ccall librccl.ncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
        newcomm::Ptr{ncclComm_t},
        config::Ptr{ncclConfig_t})::ncclResult_t
end

function pncclCommSplit(comm, color, key, newcomm, config)
     @check @ccall librccl.pncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
        newcomm::Ptr{ncclComm_t},
        config::Ptr{ncclConfig_t})::ncclResult_t
end

function ncclGetErrorString(result)
     @ccall librccl.ncclGetErrorString(result::ncclResult_t)::Cstring
end

function pncclGetErrorString(result)
     @ccall librccl.pncclGetErrorString(result::ncclResult_t)::Cstring
end

function ncclGetLastError(comm)
     @ccall librccl.ncclGetLastError(comm::ncclComm_t)::Cstring
end

function pncclGetLastError(comm)
     @ccall librccl.pncclGetLastError(comm::ncclComm_t)::Cstring
end

function ncclCommGetAsyncError(comm, asyncError)
     @check @ccall librccl.ncclCommGetAsyncError(comm::ncclComm_t,
        asyncError::Ptr{ncclResult_t})::ncclResult_t
end

function pncclCommGetAsyncError(comm, asyncError)
     @check @ccall librccl.pncclCommGetAsyncError(comm::ncclComm_t,
        asyncError::Ptr{ncclResult_t})::ncclResult_t
end

function ncclCommCount(comm, count)
     @check @ccall librccl.ncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

function pncclCommCount(comm, count)
     @check @ccall librccl.pncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

function ncclCommCuDevice(comm, device)
     @check @ccall librccl.ncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

function pncclCommCuDevice(comm, device)
     @check @ccall librccl.pncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

function ncclCommUserRank(comm, rank)
     @check @ccall librccl.ncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

function pncclCommUserRank(comm, rank)
     @check @ccall librccl.pncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

function ncclCommRegister(comm, buff, size, handle)
     @check @ccall librccl.ncclCommRegister(comm::ncclComm_t, buff::Ptr{Cvoid}, size::Cint,
        handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

function pncclCommRegister(comm, buff, size, handle)
     @check @ccall librccl.pncclCommRegister(comm::ncclComm_t, buff::Ptr{Cvoid}, size::Cint,
        handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

function ncclCommDeregister(comm, handle)
     @check @ccall librccl.ncclCommDeregister(comm::ncclComm_t, handle::Ptr{Cvoid})::ncclResult_t
end

function pncclCommDeregister(comm, handle)
     @check @ccall librccl.pncclCommDeregister(comm::ncclComm_t, handle::Ptr{Cvoid})::ncclResult_t
end

@cenum ncclRedOp_dummy_t::UInt32 begin
    ncclNumOps_dummy = 5
end

@cenum ncclRedOp_t::UInt32 begin
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5
    ncclMaxRedOp = 2147483647
end

@cenum ncclDataType_t::UInt32 begin
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclFp8E4M3 = 10
    ncclFp8E5M2 = 11
    ncclNumTypes = 12
end

@cenum ncclScalarResidence_t::UInt32 begin
    ncclScalarDevice = 0
    ncclScalarHostImmediate = 1
end

function ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
     @check @ccall librccl.ncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
        datatype::ncclDataType_t,
        residence::ncclScalarResidence_t,
        comm::ncclComm_t)::ncclResult_t
end

function pncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
     @check @ccall librccl.pncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
        datatype::ncclDataType_t,
        residence::ncclScalarResidence_t,
        comm::ncclComm_t)::ncclResult_t
end

function ncclRedOpDestroy(op, comm)
     @check @ccall librccl.ncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

function pncclRedOpDestroy(op, comm)
     @check @ccall librccl.pncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
     @check @ccall librccl.ncclReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
     @check @ccall librccl.pncclReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclBcast(buff, count, datatype, root, comm, stream)
     @check @ccall librccl.ncclBcast(buff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        root::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclBcast(buff, count, datatype, root, comm, stream)
     @check @ccall librccl.pncclBcast(buff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        root::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
     @check @ccall librccl.ncclBroadcast(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
     @check @ccall librccl.pncclBroadcast(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
     @check @ccall librccl.ncclAllReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, op::ncclRedOp_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
     @check @ccall librccl.pncclAllReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, op::ncclRedOp_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
     @check @ccall librccl.ncclReduceScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
        recvcount::Cint, datatype::ncclDataType_t,
        op::ncclRedOp_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
     @check @ccall librccl.pncclReduceScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
        recvcount::Cint, datatype::ncclDataType_t,
        op::ncclRedOp_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
     @check @ccall librccl.ncclAllGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
        sendcount::Cint, datatype::ncclDataType_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
     @check @ccall librccl.pncclAllGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
        sendcount::Cint, datatype::ncclDataType_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclSend(sendbuff, count, datatype, peer, comm, stream)
     @check @ccall librccl.ncclSend(sendbuff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        peer::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclSend(sendbuff, count, datatype, peer, comm, stream)
     @check @ccall librccl.pncclSend(sendbuff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        peer::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclRecv(recvbuff, count, datatype, peer, comm, stream)
     @check @ccall librccl.ncclRecv(recvbuff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        peer::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pncclRecv(recvbuff, count, datatype, peer, comm, stream)
     @check @ccall librccl.pncclRecv(recvbuff::Ptr{Cvoid}, count::Cint, datatype::ncclDataType_t,
        peer::Cint, comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function ncclGather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
     @check @ccall librccl.ncclGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, sendcount::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclGather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
     @check @ccall librccl.pncclGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, sendcount::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function ncclScatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
     @check @ccall librccl.ncclScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, recvcount::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclScatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
     @check @ccall librccl.pncclScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, recvcount::Cint,
        datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)
     @check @ccall librccl.ncclAllToAll(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)
     @check @ccall librccl.pncclAllToAll(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Cint,
        datatype::ncclDataType_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function ncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
    datatype, comm, stream)
     @check @ccall librccl.ncclAllToAllv(sendbuff::Ptr{Cvoid}, sendcounts::Ptr{Cint},
        sdispls::Ptr{Cint}, recvbuff::Ptr{Cvoid},
        recvcounts::Ptr{Cint}, rdispls::Ptr{Cint},
        datatype::ncclDataType_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

function pncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
    datatype, comm, stream)
     @check @ccall librccl.pncclAllToAllv(sendbuff::Ptr{Cvoid}, sendcounts::Ptr{Cint},
        sdispls::Ptr{Cint}, recvbuff::Ptr{Cvoid},
        recvcounts::Ptr{Cint}, rdispls::Ptr{Cint},
        datatype::ncclDataType_t, comm::ncclComm_t,
        stream::Cint)::ncclResult_t
end

const mscclAlgoHandle_t = Cint

function mscclLoadAlgo(mscclAlgoFilePath, mscclAlgoHandle, rank)
     @check @ccall librccl.mscclLoadAlgo(mscclAlgoFilePath::Cstring,
        mscclAlgoHandle::Ptr{mscclAlgoHandle_t},
        rank::Cint)::ncclResult_t
end

function pmscclLoadAlgo(mscclAlgoFilePath, mscclAlgoHandle, rank)
     @check @ccall librccl.pmscclLoadAlgo(mscclAlgoFilePath::Cstring,
        mscclAlgoHandle::Ptr{mscclAlgoHandle_t},
        rank::Cint)::ncclResult_t
end

function mscclRunAlgo(sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls, count,
    dataType, root, peer, op, mscclAlgoHandle, comm, stream)
     @check @ccall librccl.mscclRunAlgo(sendBuff::Ptr{Cvoid}, sendCounts::Ptr{Cint},
        sDisPls::Ptr{Cint}, recvBuff::Ptr{Cvoid},
        recvCounts::Ptr{Cint}, rDisPls::Ptr{Cint}, count::Cint,
        dataType::ncclDataType_t, root::Cint, peer::Cint,
        op::ncclRedOp_t, mscclAlgoHandle::mscclAlgoHandle_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function pmscclRunAlgo(sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls, count,
    dataType, root, peer, op, mscclAlgoHandle, comm, stream)
     @check @ccall librccl.pmscclRunAlgo(sendBuff::Ptr{Cvoid}, sendCounts::Ptr{Cint},
        sDisPls::Ptr{Cint}, recvBuff::Ptr{Cvoid},
        recvCounts::Ptr{Cint}, rDisPls::Ptr{Cint}, count::Cint,
        dataType::ncclDataType_t, root::Cint, peer::Cint,
        op::ncclRedOp_t, mscclAlgoHandle::mscclAlgoHandle_t,
        comm::ncclComm_t, stream::Cint)::ncclResult_t
end

function mscclUnloadAlgo(mscclAlgoHandle)
     @check @ccall librccl.mscclUnloadAlgo(mscclAlgoHandle::mscclAlgoHandle_t)::ncclResult_t
end

function pmscclUnloadAlgo(mscclAlgoHandle)
     @check @ccall librccl.pmscclUnloadAlgo(mscclAlgoHandle::mscclAlgoHandle_t)::ncclResult_t
end

# no prototype is found for this function at rccl.h:826:15, please use with caution
function ncclGroupStart()
     @check @ccall librccl.ncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at rccl.h:828:14, please use with caution
function pncclGroupStart()
     @check @ccall librccl.pncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at rccl.h:836:15, please use with caution
function ncclGroupEnd()
     @check @ccall librccl.ncclGroupEnd()::ncclResult_t
end

# no prototype is found for this function at rccl.h:838:14, please use with caution
function pncclGroupEnd()
     @check @ccall librccl.pncclGroupEnd()::ncclResult_t
end

function ncclGroupSimulateEnd(simInfo)
     @check @ccall librccl.ncclGroupSimulateEnd(simInfo::Ptr{ncclSimInfo_t})::ncclResult_t
end

function pncclGroupSimulateEnd(simInfo)
     @check @ccall librccl.pncclGroupSimulateEnd(simInfo::Ptr{ncclSimInfo_t})::ncclResult_t
end

const NCCL_MAJOR = 2

const NCCL_MINOR = 22

const NCCL_PATCH = 3

const NCCL_SUFFIX = ""

const NCCL_VERSION_CODE = 22203

const RCCL_BFLOAT16 = 1

const RCCL_FLOAT8 = 1

const RCCL_GATHER_SCATTER = 1

const RCCL_ALLTOALLV = 1

const NCCL_COMM_NULL = NULL

const NCCL_UNIQUE_ID_BYTES = 128

const NCCL_CONFIG_UNDEF_INT = INT_MIN

const NCCL_CONFIG_UNDEF_PTR = NULL

const NCCL_SPLIT_NOCOLOR = -1

const NCCL_UNDEF_FLOAT = -(Float32(1.0))

# Skipping MacroDefinition: NCCL_CONFIG_INITIALIZER { sizeof ( ncclConfig_t ) , /* size */ 0xcafebeef , /* magic */ NCCL_VERSION ( NCCL_MAJOR , NCCL_MINOR , NCCL_PATCH ) , /* version */ NCCL_CONFIG_UNDEF_INT , /* blocking */ NCCL_CONFIG_UNDEF_INT , /* cgaClusterSize */ NCCL_CONFIG_UNDEF_INT , /* minCTAs */ NCCL_CONFIG_UNDEF_INT , /* maxCTAs */ NCCL_CONFIG_UNDEF_PTR , /* netName */ NCCL_CONFIG_UNDEF_INT /* splitShare */ \
#}

# Skipping MacroDefinition: NCCL_SIM_INFO_INITIALIZER { sizeof ( ncclSimInfo_t ) , /* size */ 0x74685283 , /* magic */ NCCL_VERSION ( NCCL_MAJOR , NCCL_MINOR , NCCL_PATCH ) , /* version */ NCCL_UNDEF_FLOAT /* estimated time */ \
#}

export NCCLError

struct NCCLError <: Exception
    code::ncclResult_t
    msg::AbstractString
end
Base.show(io::IO, err::NCCLError) = print(io, "NCCLError(code $(err.code), $(err.msg))")

function NCCLError(code::ncclResult_t)
    msg = status_message(code)
    return NCCLError(code, msg)
end

function status_message(status)
    if status == ncclSuccess
        return "function succeeded"
    elseif status == ncclUnhandledCudaError
        return "a call to a CUDA function failed"
    elseif status == ncclSystemError
        return "a call to the system failed"
    elseif status == ncclInternalError
        return "an internal check failed. This is either a bug in NCCL or due to memory corruption"
    elseif status == ncclInvalidArgument
        return "one argument has an invalid value"
    elseif status == ncclInvalidUsage
        return "the call to NCCL is incorrect. This is usually reflecting a programming error"
    elseif status == ncclRemoteError
        return "A call failed possibly due to a network error or a remote process exiting prematurely."
    elseif status == ncclInProgress
        return "A NCCL operation on the communicator is being enqueued and is being progressed in the background."
    else
        return "unknown status"
    end
end



# exports
const PREFIXES = ["nccl"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
