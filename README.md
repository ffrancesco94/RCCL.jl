# RCCL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ffrancesco94.github.io/RCCL.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ffrancesco94.github.io/RCCL.jl/dev/)
[![Build Status](https://github.com/ffrancesco94/RCCL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ffrancesco94/RCCL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ffrancesco94/RCCL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ffrancesco94/RCCL.jl)

Wrapper of the RCCL (Radeon Collective Communication Library) for Julia.
The bindings to the C functions were generated with [Clang.jl](https://github.com/JuliaInterop/Clang.jl). The API is very similar to that of [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) and passes
the same set of tests as that.

## General notes

A couple of implementation details differ between RCCL.jl and NCCL.jl
due to discrepancies between the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl); specifically:

- `CUDA.jl` exports a `CUstream` type which is simply an alias of `cuStream_t` which
  then gets wrapped in a `CuStream` struct. `CUstream` can be passed to all NCCL
  C functions since it is basically a C type. `AMDGPU.jl` exposes a Julia struct
  called `HIPStream` which contains a handle to a `hipStream_t` C type (which itself
  is not exported). Thus, when `@ccall`ing RCCL functions, a simple `Ptr{Cvoid}`
  is passed. This should change if somehow a `hipStream_t` is exposed by `AMDGPU.jl`;
- `CUDA.jl` has a
