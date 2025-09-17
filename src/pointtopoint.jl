"""
    RCCL.Send(
        sendbuf, comm::Communicator;
        dest::Integer,
        stream::HIPStream = default_device_stream(comm)
    )

Send `sendbuf` data to rank `dest`. A matching [`Recv!`](@ref)
must be present.
"""
function Send(sendbuf, comm::Communicator; dest::Integer,
    stream::HIPStream=default_device_stream(comm))
    count = length(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    ncclSend(sendbuf, count, datatype, dest, comm, stream.stream)
    return nothing
end

"""
    RCCL.Recv!(
        recvbuf, comm::Communicator;
        source::Integer,
        stream::HIPStream = default_device_stream(comm)
    )

Write the data from a matching [`Send`](@ref) on rank `source` into `recvbuf`.
"""
function Recv!(recvbuf, comm::Communicator; source::Integer,
    stream::HIPStream=default_device_stream(comm))
    count = length(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    ncclRecv(recvbuf, count, datatype, source, comm, stream.stream)
    return recvbuf.data
end


