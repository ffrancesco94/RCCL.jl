```@meta
CurrentModule = RCCL
```

# RCCL.jl Documentation

RCCL.jl is a Julia wrapper for the AMD Radeon Collective Communication Library (RCCL), providing high-performance multi-GPU communication primitives optimized for AMD GPUs. This package is heavily inspired by and maintains API compatibility with [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl).

## Installation

To install RCCL.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("RCCL")
```

## Quick Start

```julia
using RCCL, AMDGPU

# Initialize communicators for all available GPUs
comms = RCCL.Communicators(AMDGPU.devices())

# Perform an all-reduce operation
sendbuf = AMDGPU.fill(1.0f0, 1024)
recvbuf = AMDGPU.fill(0.0f0, 1024)
RCCL.Allreduce!(sendbuf, recvbuf, +, comms[1])
```

## API Overview

RCCL.jl provides the following main components:

### Communicators

- [`Communicator`](@ref): The main communication object
- [`Communicators`](@ref): Create multiple communicators for all devices
- [`device`](@ref): Get the device associated with a communicator
- [`size`](@ref): Get the number of devices in a communicator
- [`rank`](@ref): Get the rank of the current device

### Collective Operations

- [`Allreduce!`](@ref): Reduce data across all ranks
- [`Broadcast!`](@ref): Broadcast data from one rank to all others
- [`Reduce!`](@ref): Reduce data to a single rank
- [`Allgather!`](@ref): Gather data from all ranks to all ranks
- [`ReduceScatter!`](@ref): Reduce and scatter data in one operation

### Point-to-Point Operations

- [`Send`](@ref): Send data to another rank
- [`Recv!`](@ref): Receive data from another rank

### Reduction Operations

- [`RCCL.avg`](@ref): Perform average reduction

## Detailed API Reference

```@autodocs
Modules = [RCCL]
```

## Internal discovery mechanism

```@autodocs
Modules = [RCCL.RCCLLoader]
```

## Examples

### Basic All-Reduce

```julia
using RCCL, AMDGPU

# Initialize
comms = RCCL.Communicators(AMDGPU.devices())
comm = comms[1]

# Create buffers
sendbuf = AMDGPU.fill(1.0f0, 1024)
recvbuf = AMDGPU.fill(0.0f0, 1024)

# Perform all-reduce
RCCL.Allreduce!(sendbuf, recvbuf, +, comm)

# recvbuf now contains the sum across all ranks
```

### Broadcast

```julia
using RCCL, AMDGPU

comms = RCCL.Communicators(AMDGPU.devices())
comm = comms[1]

# Root rank prepares data
if RCCL.rank(comm) == 0
    sendbuf = AMDGPU.fill(42.0f0, 1024)
else
    sendbuf = AMDGPU.fill(0.0f0, 1024)
end

recvbuf = AMDGPU.fill(0.0f0, 1024)

# Broadcast from root (rank 0)
RCCL.Broadcast!(sendbuf, recvbuf, comm; root=0)

# All ranks now have the same data in recvbuf
```

### Multi-GPU Training

```julia
using RCCL, AMDGPU

# Initialize communicators for all GPUs
comms = RCCL.Communicators(AMDGPU.devices())

# Simulate gradient averaging
for epoch in 1:10
    # Each GPU computes its own gradients
    gradients = AMDGPU.rand(Float32, 1000)
    
    # Average gradients across all GPUs
    averaged_gradients = AMDGPU.zeros(Float32, 1000)
    RCCL.Allreduce!(gradients, averaged_gradients, RCCL.avg, comms[1])
    
    # Update model parameters...
end
```

## Implementation Notes

RCCL.jl differs from NCCL.jl in a few implementation details due to differences between AMDGPU.jl and CUDA.jl:

1. **Stream Handling**: AMDGPU.jl uses `HIPStream` objects which contain handles to `hipStream_t`, while CUDA.jl exports `CUstream` as an alias to `cuStream_t`. RCCL.jl passes `Ptr{Cvoid}` to RCCL functions.

2. **Device Management**: The device handling follows AMDGPU.jl conventions.

## Troubleshooting

### Common Issues

- **Library Not Found**: Ensure RCCL is properly installed on your system and the library path is correctly configured.

- **Device Mismatch**: Make sure you're using the correct device IDs when creating communicators.

- **Communication Hangs**: Verify that all ranks are participating in collective operations and that your network configuration supports GPU-to-GPU communication.

## Performance Considerations

- Use the default device stream for best performance unless you have specific synchronization requirements.
- For multi-node communication, ensure your network infrastructure supports RDMA for optimal performance.
- Consider using non-blocking operations and overlapping communication with computation when possible.

## Related Packages

- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl): AMD GPU programming in Julia
- [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl): NVIDIA's equivalent library (inspiration for this package)
- [MPI.jl](https://github.com/JuliaParallel/MPI.jl): Message Passing Interface for Julia

## Index

```@index
```
