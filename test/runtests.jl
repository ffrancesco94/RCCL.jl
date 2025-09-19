using Test

import AMDGPU
using AMDGPU: ROCVector, ROCArray, zeros, device!, devices, versioninfo, HIPDevice
using AMDGPU.HIP: device_id
@info AMDGPU.versioninfo()

using RCCL 
@info "RCCL info: $(RCCL.version())"


@testset "RCCL.jl" begin

 @testset "Communicator" begin
    # clique of communicators
    comms = RCCL.Communicators(AMDGPU.devices())
    for (i,dev) in enumerate(AMDGPU.devices())
        @test RCCL.rank(comms[i]) == i-1
        @test RCCL.device(comms[i]) == dev
        @test RCCL.size(comms[i]) == length(AMDGPU.devices())
    end

    # single communicator (with nranks=1 or this would block)
    comm  = Communicator(1, 0)
    @test RCCL.device(comm) == HIPDevice(1)
end

@testset "Allreduce!" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)

    @testset "sum" begin
        recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
        sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
        N = 512
        for (ii, dev) in enumerate(devs)
	    AMDGPU.device!(dev)
            sendbuf[ii] = ROCArray(fill(Float64(ii), N))
            recvbuf[ii] = AMDGPU.zeros(Float64, N)
        end
        RCCL.group() do
            for ii in 1:length(devs)
                RCCL.Allreduce!(sendbuf[ii], recvbuf[ii], +, comms[ii])
            end
        end
        answer = sum(1:length(devs))
        for (ii, dev) in enumerate(devs)
	    AMDGPU.device!(dev)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end

    @testset "RCCL.avg" begin
        recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
        sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
        N = 512
        for (ii, dev) in enumerate(devs)
	    AMDGPU.device!(dev)
            sendbuf[ii] = ROCArray(fill(Float64(ii), N))
            recvbuf[ii] = AMDGPU.zeros(Float64, N)
        end
        RCCL.group() do
            for ii in 1:length(devs)
                RCCL.Allreduce!(sendbuf[ii], recvbuf[ii], RCCL.avg, comms[ii])
            end
        end
        answer = sum(1:length(devs)) / length(devs)
        for (ii, dev) in enumerate(devs)
	    AMDGPU.device!(dev)
            crecv = collect(recvbuf[ii])
            @test all(crecv .≈ answer)
        end
    end
end

@testset "Broadcast!" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)
    recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        sendbuf[ii] = (ii-1) == root ? ROCArray(fill(Float64(1.0), 512)) : AMDGPU.zeros(Float64, 512)
        recvbuf[ii] = AMDGPU.zeros(Float64, 512)
    end
    RCCL.group() do
        for ii in 1:length(devs)
            RCCL.Broadcast!(sendbuf[ii], recvbuf[ii], comms[ii]; root=root)
        end
    end
    answer = 1.0
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Reduce!" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)
    recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        sendbuf[ii] = ROCArray(fill(Float64(ii), 512))
        recvbuf[ii] = AMDGPU.zeros(Float64, 512)
    end
    RCCL.group() do
        for ii in 1:length(devs)
            RCCL.Reduce!(sendbuf[ii], recvbuf[ii], +, comms[ii]; root=root)
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = (ii - 1) == root ? sum(1:length(devs)) : 0.0
	AMDGPU.device!(dev)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Allgather!" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)
    recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        sendbuf[ii] = ROCArray(fill(Float64(ii), 512))
        recvbuf[ii] = AMDGPU.zeros(Float64, length(devs)*512)
    end
    RCCL.group() do
        for ii in 1:length(devs)
            RCCL.Allgather!(sendbuf[ii], recvbuf[ii], comms[ii])
        end
    end
    answer = vec(repeat(1:length(devs), inner=512))
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "ReduceScatter!" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)
    recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        sendbuf[ii] = ROCArray(vec(repeat(collect(1:length(devs)), inner=2)))
        recvbuf[ii] = AMDGPU.zeros(Float64, 2)
    end
    RCCL.group() do
        for ii in 1:length(devs)
            RCCL.ReduceScatter!(sendbuf[ii], recvbuf[ii], +, comms[ii])
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = length(devs)*ii
	AMDGPU.device!(dev)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Send/Recv" begin
    devs  = AMDGPU.devices()
    comms = RCCL.Communicators(devs)
    recvbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    sendbuf = Vector{ROCVector{Float64}}(undef, length(devs))
    N = 512
    for (ii, dev) in enumerate(devs)
	AMDGPU.device!(dev)
        sendbuf[ii] = ROCArray(fill(Float64(ii), N))
        recvbuf[ii] = AMDGPU.zeros(Float64, N)
    end

    RCCL.group() do
        for ii in 1:length(devs)
            comm = comms[ii]
            dest = mod(RCCL.rank(comm)+1, RCCL.size(comm))
            source = mod(RCCL.rank(comm), RCCL.size(comm))
            RCCL.Send(sendbuf[ii], comm; dest)
            RCCL.Recv!(recvbuf[ii], comm; source)
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = mod1(ii, length(devs))
	AMDGPU.device!(dev)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end
  
end
