# RCCL.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ffrancesco94.github.io/RCCL.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ffrancesco94.github.io/RCCL.jl/dev/)
[![Build Status](https://github.com/ffrancesco94/RCCL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ffrancesco94/RCCL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ffrancesco94/RCCL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ffrancesco94/RCCL.jl)

**Julia bindings for the AMD Radeon Collective Communication Library (RCCL)**

RCCL.jl provides Julia bindings for AMD's RCCL library, enabling high-performance multi-GPU communication on AMD GPU systems. The API is designed to be similar to [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) and passes the same set of tests. **Currently tested on a single node of MI250X, using ROCm 6.4. If anyone has access to a different HW/SW stack, feel free to try and report any hiccups!**

## Installation

```julia
using Pkg
Pkg.add("RCCL")
```

## Documentation

For complete documentation, see:
- [Stable Documentation](https://ffrancesco94.github.io/RCCL.jl/stable/)
- [Development Documentation](https://ffrancesco94.github.io/RCCL.jl/dev/)


## Implementation Notes

RCCL.jl differs from NCCL.jl in a few implementation details due to differences between AMDGPU.jl and CUDA.jl:

1. **Stream Handling**: AMDGPU.jl uses `HIPStream` objects which contain handles to `hipStream_t`, while CUDA.jl exports `CUstream` as an alias to `cuStream_t`. RCCL.jl passes `Ptr{Cvoid}` to RCCL functions.

2. **Device Management**: The device handling follows AMDGPU.jl conventions where device IDs are 1-based.

## Testing

The package includes the same testing suite as NCCL.jl

## Requirements

- Julia 1.6+
- AMDGPU.jl
- RCCL library installed on your system
- AMD GPUs with ROCm support

## Related Packages

- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) - AMD GPU programming in Julia
- [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) - NVIDIA's equivalent (inspiration for this package)
- [MPI.jl](https://github.com/JuliaParallel/MPI.jl) - Message Passing Interface for Julia

## Contributing

Contributions are welcome! Please open issues or pull requests on the GitHub repository.

## License

RCCL.jl is licensed under the MIT License.
