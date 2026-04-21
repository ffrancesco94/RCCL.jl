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

## Index

```@index
```
