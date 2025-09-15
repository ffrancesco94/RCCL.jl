export Communicator

const UniqueID = LibRCCL.ncclUniqueId

using AMDGPU: device_id

function UniqueID()
    r = Ref{UniqueID}()
    ncclGetUniqueId(r)
    return r[]
end

mutable struct Communicator
    handle::ncclComm_t
end

function destroy(comm::Communicator)
    if comm.handle != C_NULL
        ncclCommDestroy(comm)
        comm.handle = ncclComm_t(C_NULL)
    end
    return nothing
end
Base.unsafe_convert(::Type{LibRCCL.ncclComm_t}, comm::Communicator) = comm.handle

"""
    RCCL.communicator(nranks, rank; [unique_id])

Instantiate a communicator to be used with multithreading/multiprocessing. `nranks`
is the number of ranks, `rank` is the index of the current rank (starting from 0).
`unique_id` is an optional (unique) identifier of the communicator.

# Examples
```
comm = Communicator(length(AMDGPU.devices()), id, myid()))
# keep in mind that this is blocking.
```
"""
function Communicator(nranks::Integer, rank::Integer; unique_id::UniqueID=UniqueID())
    0 <= rank < nranks || throw(ArgumentError("Rank must be in [0, nranks)"))
    handle_ref = Ref{ncclComm_t}(C_NULL)
    ncclCommInitRank(handle_ref, nranks, unique_id, rank)
    c = Communicator(handle_ref[])
    return finalizer(destroy, c)
end


"""
    RCCL.Communicators(devices) :: Vector{Communicator}

Construct and initialise a clique of RCCL communicators over the devices 
on a single host. 

`devices` can be either a collection of identifiers, or `HIPDevice`s.

# Examples 
```
comms = RCCL.Communicators(CUDA.devices())
```
"""
function Communicators(deviceids::Vector{Cint})
    deviceids .-= 1
    ndev = length(deviceids)
    comms = Vector{ncclComm_t}(undef, ndev)
    ncclCommInitAll(comms, ndev, deviceids)
    return map(comms) do ch
        c = Communicator(ch)
        finalizer(destroy, c)
    end
end
function Communicators(deviceids::AbstractVector{<:Integer})
    Communicators(Cint[d for d in deviceids])
end
function Communicators(devices)
    Communicators(Cint[device_id(d) for d in devices])
end

"""
    RCCL.device(comm::Communicator) :: HIPDevice 

The device of the communicator.
"""
function device(comm::Communicator)
    dev_ref = Ref{Cint}(C_NULL)
    ncclCommCuDevice(comm, dev_ref)
    return HIPDevice(dev_ref[]+1)
end

"""
    RCCL.size(comm::Communicator) :: Int 

The number of devices in the Communicator.
"""
function size(comm::Communicator)
    size_ref = Ref{Cint}(C_NULL)
    ncclCommCount(comm, size_ref)
    return Int(size_ref[])
end

"""
    RCCL.rank(comm::Communicator) :: Int 

The 0-based rank of the device in the communicator.
"""
function rank(comm::Communicator)
    rank_ref = Ref{Cint}(C_NULL)
    ncclCommUserRank(comm, rank_ref)
    return Int(rank_ref[])
end

"""
    RCCL.abort(comm::Communicator)

Frees `comm`. The communicator is destroyed and 
uncompleted operations are killed.
"""
function abort(comm::Communicator)
    ncclCommAbort(comm)
    return
end

"""
    RCCL.default_device_stream(comm::Communicator) :: HIPStream 

Get the default stream for the device corresponding to communicator
`comm`.
"""
function default_device_stream(comm::Communicator)
    dev = device(comm)
    device!(dev) do
        stream()
    end
end


