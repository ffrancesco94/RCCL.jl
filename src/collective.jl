"""
    RCCL.Allreduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::HIPStream=default_device_stream(comm)
    )

Reduce array `sendbuf` using `op` (`+`, `*`, `min`, `max`, [`RCCL.avg`](@ref)),
and write the result on `recvbuf` on all ranks.
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator;
    stream::HIPStream=default_device_stream(comm))
    count = length(recvbuf)
    @assert length(sendbuf) == count
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclAllReduce(sendbuf, recvbuf, count, data_type, _op, comm, stream)
    return recvbuf
end

"""
    RCCL.Allreduce!(
        sendrecvbuf, op, comm::Communicator;
        stream::HIPStream = default_device_stream(comm)
    )

Reduce the array `sendrecvbuf` using `op` (`+`, `*`, `min`, `max`, [`RCCL.avg`](@ref)),
writing the result inplace to all ranks.
"""
function Allreduce!(sendrecvbuf, op, comm::Communicator;
    stream::HIPStream=default_device_stream(comm))
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)
end

"""
    RCCL.Broadcast!(
        sendbuf, recvbuf, comm::Communicator; root=0,
        stream::HIPStream = default_device_stream(comm)
    )

Copies the `sendbuf` array sitting on rank `root` to `recvbuf` on all ranks.
"""
function Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer=0,
    stream::HIPStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm, stream)
    return recvbuf
end

"""
    RCCL.Broadcast!(
        sendbuf, recvbuf, comm::Communicator; root=0,
        stream::HIPStream = default_device_stream(comm)
    )

Copies the `sendrecvbuf` array sitting on rank `root` inplace on all ranks.
"""
function Broadcast!(sendrecvbuf, comm::Communicator; root::Integer=0,
    stream::HIPStream=default_device_stream(comm))
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)
end

"""
    RCCL.Reduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        root=0, stream::HIPStream=default_device_stream(comm)
    )

Reduce the array `sendbuf` onto `recvbuf` on rank `root` using `op` 
(`+`, `*`, `min`, `max`, [`RCCL.avg`](@ref)).
"""
function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0,
    stream::HIPStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    _op = ncclRedOp_t(op)
    ncclReduce(sendbuf, recvbuf, count, data_type, _op, root, comm, stream)
    return recvbuf
end

"""
    RCCL.Reduce!(
        sendrecvbuf, op, comm::Communicator;
        root=0, stream::HIPStream=default_device_stream(comm)
    )

Reduce the array `sendrecvbuf` in-place on rank `root` using `op` 
(`+`, `*`, `min`, `max`, [`RCCL.avg`](@ref)).
"""
function Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0,
    stream::HIPStream=default_device_stream(comm))
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)
end

"""
    RCCL.Allgather!(
        sendbuf, recvbuf, comm::Communicator;
        stream::HIPStream = default_device_stream(comm)
    )

Gather `sendbuf` from each rank into `recvbuf` on all ranks.
"""
function Allgather!(sendbuf, recvbuf, comm::Communicator;
    stream::HIPStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    sendcount = length(sendbuf)
    @assert length(recvbuf) == sendcount * size(comm)
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm, stream)
    return recvbuf
end

"""
    RCCL.ReduceScatter!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::HIPStream = default_device_stream(comm)
    )

Reduce `sendbuf` from each rank using `op`, and put the result in `recvbuf` 
in all the ranks so that `recvbuf` on rank `i` has the i-th block of the 
result.
"""
function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator;
    stream::HIPStream=default_device_stream(comm))
    recvcount = length(recvbuf)
    @assert length(sendbuf) == recvount * size(comm)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, _op, comm, stream)
    return recvbuf
end


