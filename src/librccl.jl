module LibRCCL

using CEnum: CEnum, @cenum

const NULL = C_NULL
const INT_MIN = typemin(Cint)

import AMDGPU: HIPStream, @check

function check(f)
    res = f()::ncclResult_t
    if res != ncclSuccess
        throw(NCCLError(err))
    end
    return
end

struct hipUUID_t
    bytes::NTuple{16,Cchar}
end

const hipUUID = hipUUID_t

struct hipDeviceArch_t
    data::NTuple{4,UInt8}
end

function Base.getproperty(x::Ptr{hipDeviceArch_t}, f::Symbol)
    f === :hasGlobalInt32Atomics && return (Ptr{Cuint}(x + 0), 0, 1)
    f === :hasGlobalFloatAtomicExch && return (Ptr{Cuint}(x + 0), 1, 1)
    f === :hasSharedInt32Atomics && return (Ptr{Cuint}(x + 0), 2, 1)
    f === :hasSharedFloatAtomicExch && return (Ptr{Cuint}(x + 0), 3, 1)
    f === :hasFloatAtomicAdd && return (Ptr{Cuint}(x + 0), 4, 1)
    f === :hasGlobalInt64Atomics && return (Ptr{Cuint}(x + 0), 5, 1)
    f === :hasSharedInt64Atomics && return (Ptr{Cuint}(x + 0), 6, 1)
    f === :hasDoubles && return (Ptr{Cuint}(x + 0), 7, 1)
    f === :hasWarpVote && return (Ptr{Cuint}(x + 0), 8, 1)
    f === :hasWarpBallot && return (Ptr{Cuint}(x + 0), 9, 1)
    f === :hasWarpShuffle && return (Ptr{Cuint}(x + 0), 10, 1)
    f === :hasFunnelShift && return (Ptr{Cuint}(x + 0), 11, 1)
    f === :hasThreadFenceSystem && return (Ptr{Cuint}(x + 0), 12, 1)
    f === :hasSyncThreadsExt && return (Ptr{Cuint}(x + 0), 13, 1)
    f === :hasSurfaceFuncs && return (Ptr{Cuint}(x + 0), 14, 1)
    f === :has3dGrid && return (Ptr{Cuint}(x + 0), 15, 1)
    f === :hasDynamicParallelism && return (Ptr{Cuint}(x + 0), 16, 1)
    return getfield(x, f)
end

function Base.getproperty(x::hipDeviceArch_t, f::Symbol)
    r = Ref{hipDeviceArch_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipDeviceArch_t}, r)
    fptr = getproperty(ptr, f)
    begin
        if fptr isa Ptr
            return GC.@preserve(r, unsafe_load(fptr))
        else
            (baseptr, offset, width) = fptr
            ty = eltype(baseptr)
            baseptr32 = convert(Ptr{UInt32}, baseptr)
            u64 = GC.@preserve(r, unsafe_load(baseptr32))
            if offset + width > 32
                u64 |= GC.@preserve(r, unsafe_load(baseptr32 + 4)) << 32
            end
            u64 = u64 >> offset & (1 << width - 1)
            return u64 % ty
        end
    end
end

function Base.setproperty!(x::Ptr{hipDeviceArch_t}, f::Symbol, v)
    fptr = getproperty(x, f)
    if fptr isa Ptr
        unsafe_store!(getproperty(x, f), v)
    else
        (baseptr, offset, width) = fptr
        baseptr32 = convert(Ptr{UInt32}, baseptr)
        u64 = unsafe_load(baseptr32)
        straddle = offset + width > 32
        if straddle
            u64 |= unsafe_load(baseptr32 + 4) << 32
        end
        mask = 1 << width - 1
        u64 &= ~(mask << offset)
        u64 |= (unsigned(v) & mask) << offset
        unsafe_store!(baseptr32, u64 & typemax(UInt32))
        if straddle
            unsafe_store!(baseptr32 + 4, u64 >> 32)
        end
    end
end

function Base.propertynames(x::hipDeviceArch_t, private::Bool=false)
    return (:hasGlobalInt32Atomics, :hasGlobalFloatAtomicExch, :hasSharedInt32Atomics,
            :hasSharedFloatAtomicExch, :hasFloatAtomicAdd, :hasGlobalInt64Atomics,
            :hasSharedInt64Atomics, :hasDoubles, :hasWarpVote, :hasWarpBallot,
            :hasWarpShuffle, :hasFunnelShift, :hasThreadFenceSystem, :hasSyncThreadsExt,
            :hasSurfaceFuncs, :has3dGrid, :hasDynamicParallelism,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipDeviceProp_tR0600
    data::NTuple{1472,UInt8}
end

function Base.getproperty(x::Ptr{hipDeviceProp_tR0600}, f::Symbol)
    f === :name && return Ptr{NTuple{256,Cchar}}(x + 0)
    f === :uuid && return Ptr{hipUUID}(x + 256)
    f === :luid && return Ptr{NTuple{8,Cchar}}(x + 272)
    f === :luidDeviceNodeMask && return Ptr{Cuint}(x + 280)
    f === :totalGlobalMem && return Ptr{Csize_t}(x + 288)
    f === :sharedMemPerBlock && return Ptr{Csize_t}(x + 296)
    f === :regsPerBlock && return Ptr{Cint}(x + 304)
    f === :warpSize && return Ptr{Cint}(x + 308)
    f === :memPitch && return Ptr{Csize_t}(x + 312)
    f === :maxThreadsPerBlock && return Ptr{Cint}(x + 320)
    f === :maxThreadsDim && return Ptr{NTuple{3,Cint}}(x + 324)
    f === :maxGridSize && return Ptr{NTuple{3,Cint}}(x + 336)
    f === :clockRate && return Ptr{Cint}(x + 348)
    f === :totalConstMem && return Ptr{Csize_t}(x + 352)
    f === :major && return Ptr{Cint}(x + 360)
    f === :minor && return Ptr{Cint}(x + 364)
    f === :textureAlignment && return Ptr{Csize_t}(x + 368)
    f === :texturePitchAlignment && return Ptr{Csize_t}(x + 376)
    f === :deviceOverlap && return Ptr{Cint}(x + 384)
    f === :multiProcessorCount && return Ptr{Cint}(x + 388)
    f === :kernelExecTimeoutEnabled && return Ptr{Cint}(x + 392)
    f === :integrated && return Ptr{Cint}(x + 396)
    f === :canMapHostMemory && return Ptr{Cint}(x + 400)
    f === :computeMode && return Ptr{Cint}(x + 404)
    f === :maxTexture1D && return Ptr{Cint}(x + 408)
    f === :maxTexture1DMipmap && return Ptr{Cint}(x + 412)
    f === :maxTexture1DLinear && return Ptr{Cint}(x + 416)
    f === :maxTexture2D && return Ptr{NTuple{2,Cint}}(x + 420)
    f === :maxTexture2DMipmap && return Ptr{NTuple{2,Cint}}(x + 428)
    f === :maxTexture2DLinear && return Ptr{NTuple{3,Cint}}(x + 436)
    f === :maxTexture2DGather && return Ptr{NTuple{2,Cint}}(x + 448)
    f === :maxTexture3D && return Ptr{NTuple{3,Cint}}(x + 456)
    f === :maxTexture3DAlt && return Ptr{NTuple{3,Cint}}(x + 468)
    f === :maxTextureCubemap && return Ptr{Cint}(x + 480)
    f === :maxTexture1DLayered && return Ptr{NTuple{2,Cint}}(x + 484)
    f === :maxTexture2DLayered && return Ptr{NTuple{3,Cint}}(x + 492)
    f === :maxTextureCubemapLayered && return Ptr{NTuple{2,Cint}}(x + 504)
    f === :maxSurface1D && return Ptr{Cint}(x + 512)
    f === :maxSurface2D && return Ptr{NTuple{2,Cint}}(x + 516)
    f === :maxSurface3D && return Ptr{NTuple{3,Cint}}(x + 524)
    f === :maxSurface1DLayered && return Ptr{NTuple{2,Cint}}(x + 536)
    f === :maxSurface2DLayered && return Ptr{NTuple{3,Cint}}(x + 544)
    f === :maxSurfaceCubemap && return Ptr{Cint}(x + 556)
    f === :maxSurfaceCubemapLayered && return Ptr{NTuple{2,Cint}}(x + 560)
    f === :surfaceAlignment && return Ptr{Csize_t}(x + 568)
    f === :concurrentKernels && return Ptr{Cint}(x + 576)
    f === :ECCEnabled && return Ptr{Cint}(x + 580)
    f === :pciBusID && return Ptr{Cint}(x + 584)
    f === :pciDeviceID && return Ptr{Cint}(x + 588)
    f === :pciDomainID && return Ptr{Cint}(x + 592)
    f === :tccDriver && return Ptr{Cint}(x + 596)
    f === :asyncEngineCount && return Ptr{Cint}(x + 600)
    f === :unifiedAddressing && return Ptr{Cint}(x + 604)
    f === :memoryClockRate && return Ptr{Cint}(x + 608)
    f === :memoryBusWidth && return Ptr{Cint}(x + 612)
    f === :l2CacheSize && return Ptr{Cint}(x + 616)
    f === :persistingL2CacheMaxSize && return Ptr{Cint}(x + 620)
    f === :maxThreadsPerMultiProcessor && return Ptr{Cint}(x + 624)
    f === :streamPrioritiesSupported && return Ptr{Cint}(x + 628)
    f === :globalL1CacheSupported && return Ptr{Cint}(x + 632)
    f === :localL1CacheSupported && return Ptr{Cint}(x + 636)
    f === :sharedMemPerMultiprocessor && return Ptr{Csize_t}(x + 640)
    f === :regsPerMultiprocessor && return Ptr{Cint}(x + 648)
    f === :managedMemory && return Ptr{Cint}(x + 652)
    f === :isMultiGpuBoard && return Ptr{Cint}(x + 656)
    f === :multiGpuBoardGroupID && return Ptr{Cint}(x + 660)
    f === :hostNativeAtomicSupported && return Ptr{Cint}(x + 664)
    f === :singleToDoublePrecisionPerfRatio && return Ptr{Cint}(x + 668)
    f === :pageableMemoryAccess && return Ptr{Cint}(x + 672)
    f === :concurrentManagedAccess && return Ptr{Cint}(x + 676)
    f === :computePreemptionSupported && return Ptr{Cint}(x + 680)
    f === :canUseHostPointerForRegisteredMem && return Ptr{Cint}(x + 684)
    f === :cooperativeLaunch && return Ptr{Cint}(x + 688)
    f === :cooperativeMultiDeviceLaunch && return Ptr{Cint}(x + 692)
    f === :sharedMemPerBlockOptin && return Ptr{Csize_t}(x + 696)
    f === :pageableMemoryAccessUsesHostPageTables && return Ptr{Cint}(x + 704)
    f === :directManagedMemAccessFromHost && return Ptr{Cint}(x + 708)
    f === :maxBlocksPerMultiProcessor && return Ptr{Cint}(x + 712)
    f === :accessPolicyMaxWindowSize && return Ptr{Cint}(x + 716)
    f === :reservedSharedMemPerBlock && return Ptr{Csize_t}(x + 720)
    f === :hostRegisterSupported && return Ptr{Cint}(x + 728)
    f === :sparseHipArraySupported && return Ptr{Cint}(x + 732)
    f === :hostRegisterReadOnlySupported && return Ptr{Cint}(x + 736)
    f === :timelineSemaphoreInteropSupported && return Ptr{Cint}(x + 740)
    f === :memoryPoolsSupported && return Ptr{Cint}(x + 744)
    f === :gpuDirectRDMASupported && return Ptr{Cint}(x + 748)
    f === :gpuDirectRDMAFlushWritesOptions && return Ptr{Cuint}(x + 752)
    f === :gpuDirectRDMAWritesOrdering && return Ptr{Cint}(x + 756)
    f === :memoryPoolSupportedHandleTypes && return Ptr{Cuint}(x + 760)
    f === :deferredMappingHipArraySupported && return Ptr{Cint}(x + 764)
    f === :ipcEventSupported && return Ptr{Cint}(x + 768)
    f === :clusterLaunch && return Ptr{Cint}(x + 772)
    f === :unifiedFunctionPointers && return Ptr{Cint}(x + 776)
    f === :reserved && return Ptr{NTuple{63,Cint}}(x + 780)
    f === :hipReserved && return Ptr{NTuple{32,Cint}}(x + 1032)
    f === :gcnArchName && return Ptr{NTuple{256,Cchar}}(x + 1160)
    f === :maxSharedMemoryPerMultiProcessor && return Ptr{Csize_t}(x + 1416)
    f === :clockInstructionRate && return Ptr{Cint}(x + 1424)
    f === :arch && return Ptr{hipDeviceArch_t}(x + 1428)
    f === :hdpMemFlushCntl && return Ptr{Ptr{Cuint}}(x + 1432)
    f === :hdpRegFlushCntl && return Ptr{Ptr{Cuint}}(x + 1440)
    f === :cooperativeMultiDeviceUnmatchedFunc && return Ptr{Cint}(x + 1448)
    f === :cooperativeMultiDeviceUnmatchedGridDim && return Ptr{Cint}(x + 1452)
    f === :cooperativeMultiDeviceUnmatchedBlockDim && return Ptr{Cint}(x + 1456)
    f === :cooperativeMultiDeviceUnmatchedSharedMem && return Ptr{Cint}(x + 1460)
    f === :isLargeBar && return Ptr{Cint}(x + 1464)
    f === :asicRevision && return Ptr{Cint}(x + 1468)
    return getfield(x, f)
end

function Base.getproperty(x::hipDeviceProp_tR0600, f::Symbol)
    r = Ref{hipDeviceProp_tR0600}(x)
    ptr = Base.unsafe_convert(Ptr{hipDeviceProp_tR0600}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipDeviceProp_tR0600}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipDeviceProp_tR0600, private::Bool=false)
    return (:name, :uuid, :luid, :luidDeviceNodeMask, :totalGlobalMem, :sharedMemPerBlock,
            :regsPerBlock, :warpSize, :memPitch, :maxThreadsPerBlock, :maxThreadsDim,
            :maxGridSize, :clockRate, :totalConstMem, :major, :minor, :textureAlignment,
            :texturePitchAlignment, :deviceOverlap, :multiProcessorCount,
            :kernelExecTimeoutEnabled, :integrated, :canMapHostMemory, :computeMode,
            :maxTexture1D, :maxTexture1DMipmap, :maxTexture1DLinear, :maxTexture2D,
            :maxTexture2DMipmap, :maxTexture2DLinear, :maxTexture2DGather, :maxTexture3D,
            :maxTexture3DAlt, :maxTextureCubemap, :maxTexture1DLayered,
            :maxTexture2DLayered, :maxTextureCubemapLayered, :maxSurface1D, :maxSurface2D,
            :maxSurface3D, :maxSurface1DLayered, :maxSurface2DLayered, :maxSurfaceCubemap,
            :maxSurfaceCubemapLayered, :surfaceAlignment, :concurrentKernels, :ECCEnabled,
            :pciBusID, :pciDeviceID, :pciDomainID, :tccDriver, :asyncEngineCount,
            :unifiedAddressing, :memoryClockRate, :memoryBusWidth, :l2CacheSize,
            :persistingL2CacheMaxSize, :maxThreadsPerMultiProcessor,
            :streamPrioritiesSupported, :globalL1CacheSupported, :localL1CacheSupported,
            :sharedMemPerMultiprocessor, :regsPerMultiprocessor, :managedMemory,
            :isMultiGpuBoard, :multiGpuBoardGroupID, :hostNativeAtomicSupported,
            :singleToDoublePrecisionPerfRatio, :pageableMemoryAccess,
            :concurrentManagedAccess, :computePreemptionSupported,
            :canUseHostPointerForRegisteredMem, :cooperativeLaunch,
            :cooperativeMultiDeviceLaunch, :sharedMemPerBlockOptin,
            :pageableMemoryAccessUsesHostPageTables, :directManagedMemAccessFromHost,
            :maxBlocksPerMultiProcessor, :accessPolicyMaxWindowSize,
            :reservedSharedMemPerBlock, :hostRegisterSupported, :sparseHipArraySupported,
            :hostRegisterReadOnlySupported, :timelineSemaphoreInteropSupported,
            :memoryPoolsSupported, :gpuDirectRDMASupported,
            :gpuDirectRDMAFlushWritesOptions, :gpuDirectRDMAWritesOrdering,
            :memoryPoolSupportedHandleTypes, :deferredMappingHipArraySupported,
            :ipcEventSupported, :clusterLaunch, :unifiedFunctionPointers, :reserved,
            :hipReserved, :gcnArchName, :maxSharedMemoryPerMultiProcessor,
            :clockInstructionRate, :arch, :hdpMemFlushCntl, :hdpRegFlushCntl,
            :cooperativeMultiDeviceUnmatchedFunc, :cooperativeMultiDeviceUnmatchedGridDim,
            :cooperativeMultiDeviceUnmatchedBlockDim,
            :cooperativeMultiDeviceUnmatchedSharedMem, :isLargeBar, :asicRevision,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipError_t::UInt32 begin
    hipSuccess = 0
    hipErrorInvalidValue = 1
    hipErrorOutOfMemory = 2
    hipErrorMemoryAllocation = 2
    hipErrorNotInitialized = 3
    hipErrorInitializationError = 3
    hipErrorDeinitialized = 4
    hipErrorProfilerDisabled = 5
    hipErrorProfilerNotInitialized = 6
    hipErrorProfilerAlreadyStarted = 7
    hipErrorProfilerAlreadyStopped = 8
    hipErrorInvalidConfiguration = 9
    hipErrorInvalidPitchValue = 12
    hipErrorInvalidSymbol = 13
    hipErrorInvalidDevicePointer = 17
    hipErrorInvalidMemcpyDirection = 21
    hipErrorInsufficientDriver = 35
    hipErrorMissingConfiguration = 52
    hipErrorPriorLaunchFailure = 53
    hipErrorInvalidDeviceFunction = 98
    hipErrorNoDevice = 100
    hipErrorInvalidDevice = 101
    hipErrorInvalidImage = 200
    hipErrorInvalidContext = 201
    hipErrorContextAlreadyCurrent = 202
    hipErrorMapFailed = 205
    hipErrorMapBufferObjectFailed = 205
    hipErrorUnmapFailed = 206
    hipErrorArrayIsMapped = 207
    hipErrorAlreadyMapped = 208
    hipErrorNoBinaryForGpu = 209
    hipErrorAlreadyAcquired = 210
    hipErrorNotMapped = 211
    hipErrorNotMappedAsArray = 212
    hipErrorNotMappedAsPointer = 213
    hipErrorECCNotCorrectable = 214
    hipErrorUnsupportedLimit = 215
    hipErrorContextAlreadyInUse = 216
    hipErrorPeerAccessUnsupported = 217
    hipErrorInvalidKernelFile = 218
    hipErrorInvalidGraphicsContext = 219
    hipErrorInvalidSource = 300
    hipErrorFileNotFound = 301
    hipErrorSharedObjectSymbolNotFound = 302
    hipErrorSharedObjectInitFailed = 303
    hipErrorOperatingSystem = 304
    hipErrorInvalidHandle = 400
    hipErrorInvalidResourceHandle = 400
    hipErrorIllegalState = 401
    hipErrorNotFound = 500
    hipErrorNotReady = 600
    hipErrorIllegalAddress = 700
    hipErrorLaunchOutOfResources = 701
    hipErrorLaunchTimeOut = 702
    hipErrorPeerAccessAlreadyEnabled = 704
    hipErrorPeerAccessNotEnabled = 705
    hipErrorSetOnActiveProcess = 708
    hipErrorContextIsDestroyed = 709
    hipErrorAssert = 710
    hipErrorHostMemoryAlreadyRegistered = 712
    hipErrorHostMemoryNotRegistered = 713
    hipErrorLaunchFailure = 719
    hipErrorCooperativeLaunchTooLarge = 720
    hipErrorNotSupported = 801
    hipErrorStreamCaptureUnsupported = 900
    hipErrorStreamCaptureInvalidated = 901
    hipErrorStreamCaptureMerge = 902
    hipErrorStreamCaptureUnmatched = 903
    hipErrorStreamCaptureUnjoined = 904
    hipErrorStreamCaptureIsolation = 905
    hipErrorStreamCaptureImplicit = 906
    hipErrorCapturedEvent = 907
    hipErrorStreamCaptureWrongThread = 908
    hipErrorGraphExecUpdateFailure = 910
    hipErrorInvalidChannelDescriptor = 911
    hipErrorInvalidTexture = 912
    hipErrorUnknown = 999
    hipErrorRuntimeMemory = 1052
    hipErrorRuntimeOther = 1053
    hipErrorTbd = 1054
end

function hipGetDevicePropertiesR0600(prop, deviceId)
    @ccall librccl.hipGetDevicePropertiesR0600(prop::Ptr{hipDeviceProp_tR0600},
                                               deviceId::Cint)::hipError_t
end

function hipChooseDeviceR0600(device, prop)
    @ccall librccl.hipChooseDeviceR0600(device::Ptr{Cint},
                                        prop::Ptr{hipDeviceProp_tR0600})::hipError_t
end

mutable struct ihipStream_t end

const hipStream_t = Ptr{ihipStream_t}

@cenum hipLaunchAttributeID::UInt32 begin
    hipLaunchAttributeAccessPolicyWindow = 1
    hipLaunchAttributeCooperative = 2
    hipLaunchAttributePriority = 8
end

struct hipLaunchAttributeValue
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{hipLaunchAttributeValue}, f::Symbol)
    f === :accessPolicyWindow && return Ptr{hipAccessPolicyWindow}(x + 0)
    f === :cooperative && return Ptr{Cint}(x + 0)
    f === :priority && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipLaunchAttributeValue, f::Symbol)
    r = Ref{hipLaunchAttributeValue}(x)
    ptr = Base.unsafe_convert(Ptr{hipLaunchAttributeValue}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipLaunchAttributeValue}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipLaunchAttributeValue, private::Bool=false)
    return (:accessPolicyWindow, :cooperative, :priority, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct ncclConfig_v21700
    size::Csize_t
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
    size::Csize_t
    magic::Cuint
    version::Cuint
    estimatedTime::Cfloat
end

const ncclSimInfo_t = ncclSimInfo_v22200

# no prototype is found for this function at amd_hip_runtime.h:54:13, please use with caution
function amd_dbgapi_get_build_name()
    @ccall librccl.amd_dbgapi_get_build_name()::Cstring
end

# no prototype is found for this function at amd_hip_runtime.h:63:13, please use with caution
function amd_dbgapi_get_git_hash()
    @ccall librccl.amd_dbgapi_get_git_hash()::Cstring
end

# no prototype is found for this function at amd_hip_runtime.h:72:8, please use with caution
function amd_dbgapi_get_build_id()
    @ccall librccl.amd_dbgapi_get_build_id()::Csize_t
end

@cenum hipJitOption::UInt32 begin
    hipJitOptionMaxRegisters = 0
    hipJitOptionThreadsPerBlock = 1
    hipJitOptionWallTime = 2
    hipJitOptionInfoLogBuffer = 3
    hipJitOptionInfoLogBufferSizeBytes = 4
    hipJitOptionErrorLogBuffer = 5
    hipJitOptionErrorLogBufferSizeBytes = 6
    hipJitOptionOptimizationLevel = 7
    hipJitOptionTargetFromContext = 8
    hipJitOptionTarget = 9
    hipJitOptionFallbackStrategy = 10
    hipJitOptionGenerateDebugInfo = 11
    hipJitOptionLogVerbose = 12
    hipJitOptionGenerateLineInfo = 13
    hipJitOptionCacheMode = 14
    hipJitOptionSm3xOpt = 15
    hipJitOptionFastCompile = 16
    hipJitOptionGlobalSymbolNames = 17
    hipJitOptionGlobalSymbolAddresses = 18
    hipJitOptionGlobalSymbolCount = 19
    hipJitOptionLto = 20
    hipJitOptionFtz = 21
    hipJitOptionPrecDiv = 22
    hipJitOptionPrecSqrt = 23
    hipJitOptionFma = 24
    hipJitOptionPositionIndependentCode = 25
    hipJitOptionMinCTAPerSM = 26
    hipJitOptionMaxThreadsPerBlock = 27
    hipJitOptionOverrideDirectiveValues = 28
    hipJitOptionNumOptions = 29
    hipJitOptionIRtoISAOptExt = 10000
    hipJitOptionIRtoISAOptCountExt = 10001
end

@cenum hipJitInputType::UInt32 begin
    hipJitInputCubin = 0
    hipJitInputPtx = 1
    hipJitInputFatBinary = 2
    hipJitInputObject = 3
    hipJitInputLibrary = 4
    hipJitInputNvvm = 5
    hipJitNumLegacyInputTypes = 6
    hipJitInputLLVMBitcode = 100
    hipJitInputLLVMBundledBitcode = 101
    hipJitInputLLVMArchivesOfBundledBitcode = 102
    hipJitInputSpirv = 103
    hipJitNumInputTypes = 10
end

@cenum hipJitCacheMode::UInt32 begin
    hipJitCacheOptionNone = 0
    hipJitCacheOptionCG = 1
    hipJitCacheOptionCA = 2
end

@cenum hipJitFallback::UInt32 begin
    hipJitPreferPTX = 0
    hipJitPreferBinary = 1
end

@cenum var"##Ctag#231"::UInt32 begin
    HIP_SUCCESS = 0
    HIP_ERROR_INVALID_VALUE = 1
    HIP_ERROR_NOT_INITIALIZED = 2
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES = 3
end

@cenum hipMemoryType::UInt32 begin
    hipMemoryTypeUnregistered = 0
    hipMemoryTypeHost = 1
    hipMemoryTypeDevice = 2
    hipMemoryTypeManaged = 3
    hipMemoryTypeArray = 10
    hipMemoryTypeUnified = 11
end

struct hipPointerAttribute_t
    type::hipMemoryType
    device::Cint
    devicePointer::Ptr{Cvoid}
    hostPointer::Ptr{Cvoid}
    isManaged::Cint
    allocationFlags::Cuint
end

@cenum hipDeviceAttribute_t::UInt32 begin
    hipDeviceAttributeCudaCompatibleBegin = 0
    hipDeviceAttributeEccEnabled = 0
    hipDeviceAttributeAccessPolicyMaxWindowSize = 1
    hipDeviceAttributeAsyncEngineCount = 2
    hipDeviceAttributeCanMapHostMemory = 3
    hipDeviceAttributeCanUseHostPointerForRegisteredMem = 4
    hipDeviceAttributeClockRate = 5
    hipDeviceAttributeComputeMode = 6
    hipDeviceAttributeComputePreemptionSupported = 7
    hipDeviceAttributeConcurrentKernels = 8
    hipDeviceAttributeConcurrentManagedAccess = 9
    hipDeviceAttributeCooperativeLaunch = 10
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 11
    hipDeviceAttributeDeviceOverlap = 12
    hipDeviceAttributeDirectManagedMemAccessFromHost = 13
    hipDeviceAttributeGlobalL1CacheSupported = 14
    hipDeviceAttributeHostNativeAtomicSupported = 15
    hipDeviceAttributeIntegrated = 16
    hipDeviceAttributeIsMultiGpuBoard = 17
    hipDeviceAttributeKernelExecTimeout = 18
    hipDeviceAttributeL2CacheSize = 19
    hipDeviceAttributeLocalL1CacheSupported = 20
    hipDeviceAttributeLuid = 21
    hipDeviceAttributeLuidDeviceNodeMask = 22
    hipDeviceAttributeComputeCapabilityMajor = 23
    hipDeviceAttributeManagedMemory = 24
    hipDeviceAttributeMaxBlocksPerMultiProcessor = 25
    hipDeviceAttributeMaxBlockDimX = 26
    hipDeviceAttributeMaxBlockDimY = 27
    hipDeviceAttributeMaxBlockDimZ = 28
    hipDeviceAttributeMaxGridDimX = 29
    hipDeviceAttributeMaxGridDimY = 30
    hipDeviceAttributeMaxGridDimZ = 31
    hipDeviceAttributeMaxSurface1D = 32
    hipDeviceAttributeMaxSurface1DLayered = 33
    hipDeviceAttributeMaxSurface2D = 34
    hipDeviceAttributeMaxSurface2DLayered = 35
    hipDeviceAttributeMaxSurface3D = 36
    hipDeviceAttributeMaxSurfaceCubemap = 37
    hipDeviceAttributeMaxSurfaceCubemapLayered = 38
    hipDeviceAttributeMaxTexture1DWidth = 39
    hipDeviceAttributeMaxTexture1DLayered = 40
    hipDeviceAttributeMaxTexture1DLinear = 41
    hipDeviceAttributeMaxTexture1DMipmap = 42
    hipDeviceAttributeMaxTexture2DWidth = 43
    hipDeviceAttributeMaxTexture2DHeight = 44
    hipDeviceAttributeMaxTexture2DGather = 45
    hipDeviceAttributeMaxTexture2DLayered = 46
    hipDeviceAttributeMaxTexture2DLinear = 47
    hipDeviceAttributeMaxTexture2DMipmap = 48
    hipDeviceAttributeMaxTexture3DWidth = 49
    hipDeviceAttributeMaxTexture3DHeight = 50
    hipDeviceAttributeMaxTexture3DDepth = 51
    hipDeviceAttributeMaxTexture3DAlt = 52
    hipDeviceAttributeMaxTextureCubemap = 53
    hipDeviceAttributeMaxTextureCubemapLayered = 54
    hipDeviceAttributeMaxThreadsDim = 55
    hipDeviceAttributeMaxThreadsPerBlock = 56
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 57
    hipDeviceAttributeMaxPitch = 58
    hipDeviceAttributeMemoryBusWidth = 59
    hipDeviceAttributeMemoryClockRate = 60
    hipDeviceAttributeComputeCapabilityMinor = 61
    hipDeviceAttributeMultiGpuBoardGroupID = 62
    hipDeviceAttributeMultiprocessorCount = 63
    hipDeviceAttributeUnused1 = 64
    hipDeviceAttributePageableMemoryAccess = 65
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 66
    hipDeviceAttributePciBusId = 67
    hipDeviceAttributePciDeviceId = 68
    hipDeviceAttributePciDomainID = 69
    hipDeviceAttributePersistingL2CacheMaxSize = 70
    hipDeviceAttributeMaxRegistersPerBlock = 71
    hipDeviceAttributeMaxRegistersPerMultiprocessor = 72
    hipDeviceAttributeReservedSharedMemPerBlock = 73
    hipDeviceAttributeMaxSharedMemoryPerBlock = 74
    hipDeviceAttributeSharedMemPerBlockOptin = 75
    hipDeviceAttributeSharedMemPerMultiprocessor = 76
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio = 77
    hipDeviceAttributeStreamPrioritiesSupported = 78
    hipDeviceAttributeSurfaceAlignment = 79
    hipDeviceAttributeTccDriver = 80
    hipDeviceAttributeTextureAlignment = 81
    hipDeviceAttributeTexturePitchAlignment = 82
    hipDeviceAttributeTotalConstantMemory = 83
    hipDeviceAttributeTotalGlobalMem = 84
    hipDeviceAttributeUnifiedAddressing = 85
    hipDeviceAttributeUnused2 = 86
    hipDeviceAttributeWarpSize = 87
    hipDeviceAttributeMemoryPoolsSupported = 88
    hipDeviceAttributeVirtualMemoryManagementSupported = 89
    hipDeviceAttributeHostRegisterSupported = 90
    hipDeviceAttributeMemoryPoolSupportedHandleTypes = 91
    hipDeviceAttributeCudaCompatibleEnd = 9999
    hipDeviceAttributeAmdSpecificBegin = 10000
    hipDeviceAttributeClockInstructionRate = 10000
    hipDeviceAttributeUnused3 = 10001
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 10002
    hipDeviceAttributeUnused4 = 10003
    hipDeviceAttributeUnused5 = 10004
    hipDeviceAttributeHdpMemFlushCntl = 10005
    hipDeviceAttributeHdpRegFlushCntl = 10006
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = 10007
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = 10008
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = 10009
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = 10010
    hipDeviceAttributeIsLargeBar = 10011
    hipDeviceAttributeAsicRevision = 10012
    hipDeviceAttributeCanUseStreamWaitValue = 10013
    hipDeviceAttributeImageSupport = 10014
    hipDeviceAttributePhysicalMultiProcessorCount = 10015
    hipDeviceAttributeFineGrainSupport = 10016
    hipDeviceAttributeWallClockRate = 10017
    hipDeviceAttributeAmdSpecificEnd = 19999
    hipDeviceAttributeVendorSpecificBegin = 20000
end

@cenum hipDriverProcAddressQueryResult::UInt32 begin
    HIP_GET_PROC_ADDRESS_SUCCESS = 0
    HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1
    HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
end

@cenum hipComputeMode::UInt32 begin
    hipComputeModeDefault = 0
    hipComputeModeExclusive = 1
    hipComputeModeProhibited = 2
    hipComputeModeExclusiveProcess = 3
end

@cenum hipFlushGPUDirectRDMAWritesOptions::UInt32 begin
    hipFlushGPUDirectRDMAWritesOptionHost = 1
    hipFlushGPUDirectRDMAWritesOptionMemOps = 2
end

@cenum hipGPUDirectRDMAWritesOrdering::UInt32 begin
    hipGPUDirectRDMAWritesOrderingNone = 0
    hipGPUDirectRDMAWritesOrderingOwner = 100
    hipGPUDirectRDMAWritesOrderingAllDevices = 200
end

const hipDeviceptr_t = Ptr{Cvoid}

@cenum hipChannelFormatKind::UInt32 begin
    hipChannelFormatKindSigned = 0
    hipChannelFormatKindUnsigned = 1
    hipChannelFormatKindFloat = 2
    hipChannelFormatKindNone = 3
end

struct hipChannelFormatDesc
    x::Cint
    y::Cint
    z::Cint
    w::Cint
    f::hipChannelFormatKind
end

mutable struct hipArray end

const hipArray_t = Ptr{hipArray}

const hipArray_const_t = Ptr{hipArray}

@cenum hipArray_Format::UInt32 begin
    HIP_AD_FORMAT_UNSIGNED_INT8 = 1
    HIP_AD_FORMAT_UNSIGNED_INT16 = 2
    HIP_AD_FORMAT_UNSIGNED_INT32 = 3
    HIP_AD_FORMAT_SIGNED_INT8 = 8
    HIP_AD_FORMAT_SIGNED_INT16 = 9
    HIP_AD_FORMAT_SIGNED_INT32 = 10
    HIP_AD_FORMAT_HALF = 16
    HIP_AD_FORMAT_FLOAT = 32
end

struct HIP_ARRAY_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t
    Format::hipArray_Format
    NumChannels::Cuint
end

struct HIP_ARRAY3D_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t
    Format::hipArray_Format
    NumChannels::Cuint
    Flags::Cuint
end

struct hip_Memcpy2D
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcMemoryType::hipMemoryType
    srcHost::Ptr{Cvoid}
    srcDevice::hipDeviceptr_t
    srcArray::hipArray_t
    srcPitch::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstMemoryType::hipMemoryType
    dstHost::Ptr{Cvoid}
    dstDevice::hipDeviceptr_t
    dstArray::hipArray_t
    dstPitch::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
end

struct hipMipmappedArray
    data::Ptr{Cvoid}
    desc::hipChannelFormatDesc
    type::Cuint
    width::Cuint
    height::Cuint
    depth::Cuint
    min_mipmap_level::Cuint
    max_mipmap_level::Cuint
    flags::Cuint
    format::hipArray_Format
    num_channels::Cuint
end

const hipMipmappedArray_t = Ptr{hipMipmappedArray}

const hipmipmappedArray = hipMipmappedArray_t

const hipMipmappedArray_const_t = Ptr{hipMipmappedArray}

@cenum hipResourceType::UInt32 begin
    hipResourceTypeArray = 0
    hipResourceTypeMipmappedArray = 1
    hipResourceTypeLinear = 2
    hipResourceTypePitch2D = 3
end

@cenum HIPresourcetype_enum::UInt32 begin
    HIP_RESOURCE_TYPE_ARRAY = 0
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    HIP_RESOURCE_TYPE_LINEAR = 2
    HIP_RESOURCE_TYPE_PITCH2D = 3
end

const HIPresourcetype = HIPresourcetype_enum

const hipResourcetype = HIPresourcetype_enum

@cenum HIPaddress_mode_enum::UInt32 begin
    HIP_TR_ADDRESS_MODE_WRAP = 0
    HIP_TR_ADDRESS_MODE_CLAMP = 1
    HIP_TR_ADDRESS_MODE_MIRROR = 2
    HIP_TR_ADDRESS_MODE_BORDER = 3
end

const HIPaddress_mode = HIPaddress_mode_enum

@cenum HIPfilter_mode_enum::UInt32 begin
    HIP_TR_FILTER_MODE_POINT = 0
    HIP_TR_FILTER_MODE_LINEAR = 1
end

const HIPfilter_mode = HIPfilter_mode_enum

struct HIP_TEXTURE_DESC_st
    addressMode::NTuple{3,HIPaddress_mode}
    filterMode::HIPfilter_mode
    flags::Cuint
    maxAnisotropy::Cuint
    mipmapFilterMode::HIPfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    borderColor::NTuple{4,Cfloat}
    reserved::NTuple{12,Cint}
end

const HIP_TEXTURE_DESC = HIP_TEXTURE_DESC_st

@cenum hipResourceViewFormat::UInt32 begin
    hipResViewFormatNone = 0
    hipResViewFormatUnsignedChar1 = 1
    hipResViewFormatUnsignedChar2 = 2
    hipResViewFormatUnsignedChar4 = 3
    hipResViewFormatSignedChar1 = 4
    hipResViewFormatSignedChar2 = 5
    hipResViewFormatSignedChar4 = 6
    hipResViewFormatUnsignedShort1 = 7
    hipResViewFormatUnsignedShort2 = 8
    hipResViewFormatUnsignedShort4 = 9
    hipResViewFormatSignedShort1 = 10
    hipResViewFormatSignedShort2 = 11
    hipResViewFormatSignedShort4 = 12
    hipResViewFormatUnsignedInt1 = 13
    hipResViewFormatUnsignedInt2 = 14
    hipResViewFormatUnsignedInt4 = 15
    hipResViewFormatSignedInt1 = 16
    hipResViewFormatSignedInt2 = 17
    hipResViewFormatSignedInt4 = 18
    hipResViewFormatHalf1 = 19
    hipResViewFormatHalf2 = 20
    hipResViewFormatHalf4 = 21
    hipResViewFormatFloat1 = 22
    hipResViewFormatFloat2 = 23
    hipResViewFormatFloat4 = 24
    hipResViewFormatUnsignedBlockCompressed1 = 25
    hipResViewFormatUnsignedBlockCompressed2 = 26
    hipResViewFormatUnsignedBlockCompressed3 = 27
    hipResViewFormatUnsignedBlockCompressed4 = 28
    hipResViewFormatSignedBlockCompressed4 = 29
    hipResViewFormatUnsignedBlockCompressed5 = 30
    hipResViewFormatSignedBlockCompressed5 = 31
    hipResViewFormatUnsignedBlockCompressed6H = 32
    hipResViewFormatSignedBlockCompressed6H = 33
    hipResViewFormatUnsignedBlockCompressed7 = 34
end

@cenum HIPresourceViewFormat_enum::UInt32 begin
    HIP_RES_VIEW_FORMAT_NONE = 0
    HIP_RES_VIEW_FORMAT_UINT_1X8 = 1
    HIP_RES_VIEW_FORMAT_UINT_2X8 = 2
    HIP_RES_VIEW_FORMAT_UINT_4X8 = 3
    HIP_RES_VIEW_FORMAT_SINT_1X8 = 4
    HIP_RES_VIEW_FORMAT_SINT_2X8 = 5
    HIP_RES_VIEW_FORMAT_SINT_4X8 = 6
    HIP_RES_VIEW_FORMAT_UINT_1X16 = 7
    HIP_RES_VIEW_FORMAT_UINT_2X16 = 8
    HIP_RES_VIEW_FORMAT_UINT_4X16 = 9
    HIP_RES_VIEW_FORMAT_SINT_1X16 = 10
    HIP_RES_VIEW_FORMAT_SINT_2X16 = 11
    HIP_RES_VIEW_FORMAT_SINT_4X16 = 12
    HIP_RES_VIEW_FORMAT_UINT_1X32 = 13
    HIP_RES_VIEW_FORMAT_UINT_2X32 = 14
    HIP_RES_VIEW_FORMAT_UINT_4X32 = 15
    HIP_RES_VIEW_FORMAT_SINT_1X32 = 16
    HIP_RES_VIEW_FORMAT_SINT_2X32 = 17
    HIP_RES_VIEW_FORMAT_SINT_4X32 = 18
    HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
end

const HIPresourceViewFormat = HIPresourceViewFormat_enum

struct var"##Ctag#243"
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#243"}, f::Symbol)
    f === :array && return Ptr{var"##Ctag#244"}(x + 0)
    f === :mipmap && return Ptr{var"##Ctag#245"}(x + 0)
    f === :linear && return Ptr{var"##Ctag#246"}(x + 0)
    f === :pitch2D && return Ptr{var"##Ctag#247"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#243", f::Symbol)
    r = Ref{var"##Ctag#243"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#243"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#243"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#243", private::Bool=false)
    return (:array, :mipmap, :linear, :pitch2D, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipResourceDesc
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipResourceDesc}, f::Symbol)
    f === :resType && return Ptr{hipResourceType}(x + 0)
    f === :res && return Ptr{var"##Ctag#243"}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::hipResourceDesc, f::Symbol)
    r = Ref{hipResourceDesc}(x)
    ptr = Base.unsafe_convert(Ptr{hipResourceDesc}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipResourceDesc}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipResourceDesc, private::Bool=false)
    return (:resType, :res, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#250"
    data::NTuple{128,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#250"}, f::Symbol)
    f === :array && return Ptr{var"##Ctag#251"}(x + 0)
    f === :mipmap && return Ptr{var"##Ctag#252"}(x + 0)
    f === :linear && return Ptr{var"##Ctag#253"}(x + 0)
    f === :pitch2D && return Ptr{var"##Ctag#254"}(x + 0)
    f === :reserved && return Ptr{var"##Ctag#255"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#250", f::Symbol)
    r = Ref{var"##Ctag#250"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#250"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#250"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#250", private::Bool=false)
    return (:array, :mipmap, :linear, :pitch2D, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct HIP_RESOURCE_DESC_st
    data::NTuple{144,UInt8}
end

function Base.getproperty(x::Ptr{HIP_RESOURCE_DESC_st}, f::Symbol)
    f === :resType && return Ptr{HIPresourcetype}(x + 0)
    f === :res && return Ptr{var"##Ctag#250"}(x + 8)
    f === :flags && return Ptr{Cuint}(x + 136)
    return getfield(x, f)
end

function Base.getproperty(x::HIP_RESOURCE_DESC_st, f::Symbol)
    r = Ref{HIP_RESOURCE_DESC_st}(x)
    ptr = Base.unsafe_convert(Ptr{HIP_RESOURCE_DESC_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{HIP_RESOURCE_DESC_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::HIP_RESOURCE_DESC_st, private::Bool=false)
    return (:resType, :res, :flags, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const HIP_RESOURCE_DESC = HIP_RESOURCE_DESC_st

struct hipResourceViewDesc
    format::hipResourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::Cuint
    lastMipmapLevel::Cuint
    firstLayer::Cuint
    lastLayer::Cuint
end

struct HIP_RESOURCE_VIEW_DESC_st
    format::HIPresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::Cuint
    lastMipmapLevel::Cuint
    firstLayer::Cuint
    lastLayer::Cuint
    reserved::NTuple{16,Cuint}
end

const HIP_RESOURCE_VIEW_DESC = HIP_RESOURCE_VIEW_DESC_st

@cenum hipMemcpyKind::UInt32 begin
    hipMemcpyHostToHost = 0
    hipMemcpyHostToDevice = 1
    hipMemcpyDeviceToHost = 2
    hipMemcpyDeviceToDevice = 3
    hipMemcpyDefault = 4
    hipMemcpyDeviceToDeviceNoCU = 1024
end

struct hipPitchedPtr
    ptr::Ptr{Cvoid}
    pitch::Csize_t
    xsize::Csize_t
    ysize::Csize_t
end

struct hipExtent
    width::Csize_t
    height::Csize_t
    depth::Csize_t
end

struct hipPos
    x::Csize_t
    y::Csize_t
    z::Csize_t
end

struct hipMemcpy3DParms
    srcArray::hipArray_t
    srcPos::hipPos
    srcPtr::hipPitchedPtr
    dstArray::hipArray_t
    dstPos::hipPos
    dstPtr::hipPitchedPtr
    extent::hipExtent
    kind::hipMemcpyKind
end

struct HIP_MEMCPY3D
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::hipMemoryType
    srcHost::Ptr{Cvoid}
    srcDevice::hipDeviceptr_t
    srcArray::hipArray_t
    srcPitch::Csize_t
    srcHeight::Csize_t
    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::hipMemoryType
    dstHost::Ptr{Cvoid}
    dstDevice::hipDeviceptr_t
    dstArray::hipArray_t
    dstPitch::Csize_t
    dstHeight::Csize_t
    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

function make_hipPitchedPtr(d, p, xsz, ysz)
    @ccall librccl.make_hipPitchedPtr(d::Ptr{Cvoid}, p::Csize_t, xsz::Csize_t,
                                      ysz::Csize_t)::hipPitchedPtr
end

function make_hipPos(x, y, z)
    @ccall librccl.make_hipPos(x::Csize_t, y::Csize_t, z::Csize_t)::hipPos
end

function make_hipExtent(w, h, d)
    @ccall librccl.make_hipExtent(w::Csize_t, h::Csize_t, d::Csize_t)::hipExtent
end

@cenum hipFunction_attribute::UInt32 begin
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    HIP_FUNC_ATTRIBUTE_MAX = 10
end

@cenum hipPointer_attribute::UInt32 begin
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    HIP_POINTER_ATTRIBUTE_HOST_POINTER = 4
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS = 5
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    HIP_POINTER_ATTRIBUTE_BUFFER_ID = 7
    HIP_POINTER_ATTRIBUTE_IS_MANAGED = 8
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE = 10
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE = 12
    HIP_POINTER_ATTRIBUTE_MAPPED = 13
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
end

struct uchar1
    x::Cuchar
end

struct uchar2
    x::Cuchar
    y::Cuchar
end

struct uchar3
    x::Cuchar
    y::Cuchar
    z::Cuchar
end

struct uchar4
    x::Cuchar
    y::Cuchar
    z::Cuchar
    w::Cuchar
end

struct char1
    x::Cchar
end

struct char2
    x::Cchar
    y::Cchar
end

struct char3
    x::Cchar
    y::Cchar
    z::Cchar
end

struct char4
    x::Cchar
    y::Cchar
    z::Cchar
    w::Cchar
end

struct ushort1
    x::Cushort
end

struct ushort2
    x::Cushort
    y::Cushort
end

struct ushort3
    x::Cushort
    y::Cushort
    z::Cushort
end

struct ushort4
    x::Cushort
    y::Cushort
    z::Cushort
    w::Cushort
end

struct short1
    x::Cshort
end

struct short2
    x::Cshort
    y::Cshort
end

struct short3
    x::Cshort
    y::Cshort
    z::Cshort
end

struct short4
    x::Cshort
    y::Cshort
    z::Cshort
    w::Cshort
end

struct uint1
    x::Cuint
end

struct uint2
    x::Cuint
    y::Cuint
end

struct uint3
    x::Cuint
    y::Cuint
    z::Cuint
end

struct uint4
    x::Cuint
    y::Cuint
    z::Cuint
    w::Cuint
end

struct int1
    x::Cint
end

struct int2
    x::Cint
    y::Cint
end

struct int3
    x::Cint
    y::Cint
    z::Cint
end

struct int4
    x::Cint
    y::Cint
    z::Cint
    w::Cint
end

struct ulong1
    x::Culong
end

struct ulong2
    x::Culong
    y::Culong
end

struct ulong3
    x::Culong
    y::Culong
    z::Culong
end

struct ulong4
    x::Culong
    y::Culong
    z::Culong
    w::Culong
end

struct long1
    x::Clong
end

struct long2
    x::Clong
    y::Clong
end

struct long3
    x::Clong
    y::Clong
    z::Clong
end

struct long4
    x::Clong
    y::Clong
    z::Clong
    w::Clong
end

struct ulonglong1
    x::Culonglong
end

struct ulonglong2
    x::Culonglong
    y::Culonglong
end

struct ulonglong3
    x::Culonglong
    y::Culonglong
    z::Culonglong
end

struct ulonglong4
    x::Culonglong
    y::Culonglong
    z::Culonglong
    w::Culonglong
end

struct longlong1
    x::Clonglong
end

struct longlong2
    x::Clonglong
    y::Clonglong
end

struct longlong3
    x::Clonglong
    y::Clonglong
    z::Clonglong
end

struct longlong4
    x::Clonglong
    y::Clonglong
    z::Clonglong
    w::Clonglong
end

struct float1
    x::Cfloat
end

struct float2
    x::Cfloat
    y::Cfloat
end

struct float3
    x::Cfloat
    y::Cfloat
    z::Cfloat
end

struct float4
    x::Cfloat
    y::Cfloat
    z::Cfloat
    w::Cfloat
end

struct double1
    x::Cdouble
end

struct double2
    x::Cdouble
    y::Cdouble
end

struct double3
    x::Cdouble
    y::Cdouble
    z::Cdouble
end

struct double4
    x::Cdouble
    y::Cdouble
    z::Cdouble
    w::Cdouble
end

function make_uchar1(x)
    @ccall librccl.make_uchar1(x::Cuchar)::uchar1
end

function make_uchar2(x, y)
    @ccall librccl.make_uchar2(x::Cuchar, y::Cuchar)::uchar2
end

function make_uchar3(x, y, z)
    @ccall librccl.make_uchar3(x::Cuchar, y::Cuchar, z::Cuchar)::uchar3
end

function make_uchar4(x, y, z, w)
    @ccall librccl.make_uchar4(x::Cuchar, y::Cuchar, z::Cuchar, w::Cuchar)::uchar4
end

function make_char1(x)
    @ccall librccl.make_char1(x::Int8)::char1
end

function make_char2(x, y)
    @ccall librccl.make_char2(x::Int8, y::Int8)::char2
end

function make_char3(x, y, z)
    @ccall librccl.make_char3(x::Int8, y::Int8, z::Int8)::char3
end

function make_char4(x, y, z, w)
    @ccall librccl.make_char4(x::Int8, y::Int8, z::Int8, w::Int8)::char4
end

function make_ushort1(x)
    @ccall librccl.make_ushort1(x::Cushort)::ushort1
end

function make_ushort2(x, y)
    @ccall librccl.make_ushort2(x::Cushort, y::Cushort)::ushort2
end

function make_ushort3(x, y, z)
    @ccall librccl.make_ushort3(x::Cushort, y::Cushort, z::Cushort)::ushort3
end

function make_ushort4(x, y, z, w)
    @ccall librccl.make_ushort4(x::Cushort, y::Cushort, z::Cushort, w::Cushort)::ushort4
end

function make_short1(x)
    @ccall librccl.make_short1(x::Cshort)::short1
end

function make_short2(x, y)
    @ccall librccl.make_short2(x::Cshort, y::Cshort)::short2
end

function make_short3(x, y, z)
    @ccall librccl.make_short3(x::Cshort, y::Cshort, z::Cshort)::short3
end

function make_short4(x, y, z, w)
    @ccall librccl.make_short4(x::Cshort, y::Cshort, z::Cshort, w::Cshort)::short4
end

function make_uint1(x)
    @ccall librccl.make_uint1(x::Cuint)::uint1
end

function make_uint2(x, y)
    @ccall librccl.make_uint2(x::Cuint, y::Cuint)::uint2
end

function make_uint3(x, y, z)
    @ccall librccl.make_uint3(x::Cuint, y::Cuint, z::Cuint)::uint3
end

function make_uint4(x, y, z, w)
    @ccall librccl.make_uint4(x::Cuint, y::Cuint, z::Cuint, w::Cuint)::uint4
end

function make_int1(x)
    @ccall librccl.make_int1(x::Cint)::int1
end

function make_int2(x, y)
    @ccall librccl.make_int2(x::Cint, y::Cint)::int2
end

function make_int3(x, y, z)
    @ccall librccl.make_int3(x::Cint, y::Cint, z::Cint)::int3
end

function make_int4(x, y, z, w)
    @ccall librccl.make_int4(x::Cint, y::Cint, z::Cint, w::Cint)::int4
end

function make_float1(x)
    @ccall librccl.make_float1(x::Cfloat)::float1
end

function make_float2(x, y)
    @ccall librccl.make_float2(x::Cfloat, y::Cfloat)::float2
end

function make_float3(x, y, z)
    @ccall librccl.make_float3(x::Cfloat, y::Cfloat, z::Cfloat)::float3
end

function make_float4(x, y, z, w)
    @ccall librccl.make_float4(x::Cfloat, y::Cfloat, z::Cfloat, w::Cfloat)::float4
end

function make_double1(x)
    @ccall librccl.make_double1(x::Cdouble)::double1
end

function make_double2(x, y)
    @ccall librccl.make_double2(x::Cdouble, y::Cdouble)::double2
end

function make_double3(x, y, z)
    @ccall librccl.make_double3(x::Cdouble, y::Cdouble, z::Cdouble)::double3
end

function make_double4(x, y, z, w)
    @ccall librccl.make_double4(x::Cdouble, y::Cdouble, z::Cdouble, w::Cdouble)::double4
end

function make_ulong1(x)
    @ccall librccl.make_ulong1(x::Culong)::ulong1
end

function make_ulong2(x, y)
    @ccall librccl.make_ulong2(x::Culong, y::Culong)::ulong2
end

function make_ulong3(x, y, z)
    @ccall librccl.make_ulong3(x::Culong, y::Culong, z::Culong)::ulong3
end

function make_ulong4(x, y, z, w)
    @ccall librccl.make_ulong4(x::Culong, y::Culong, z::Culong, w::Culong)::ulong4
end

function make_long1(x)
    @ccall librccl.make_long1(x::Clong)::long1
end

function make_long2(x, y)
    @ccall librccl.make_long2(x::Clong, y::Clong)::long2
end

function make_long3(x, y, z)
    @ccall librccl.make_long3(x::Clong, y::Clong, z::Clong)::long3
end

function make_long4(x, y, z, w)
    @ccall librccl.make_long4(x::Clong, y::Clong, z::Clong, w::Clong)::long4
end

function make_ulonglong1(x)
    @ccall librccl.make_ulonglong1(x::Culonglong)::ulonglong1
end

function make_ulonglong2(x, y)
    @ccall librccl.make_ulonglong2(x::Culonglong, y::Culonglong)::ulonglong2
end

function make_ulonglong3(x, y, z)
    @ccall librccl.make_ulonglong3(x::Culonglong, y::Culonglong, z::Culonglong)::ulonglong3
end

function make_ulonglong4(x, y, z, w)
    @ccall librccl.make_ulonglong4(x::Culonglong, y::Culonglong, z::Culonglong,
                                   w::Culonglong)::ulonglong4
end

function make_longlong1(x)
    @ccall librccl.make_longlong1(x::Clonglong)::longlong1
end

function make_longlong2(x, y)
    @ccall librccl.make_longlong2(x::Clonglong, y::Clonglong)::longlong2
end

function make_longlong3(x, y, z)
    @ccall librccl.make_longlong3(x::Clonglong, y::Clonglong, z::Clonglong)::longlong3
end

function make_longlong4(x, y, z, w)
    @ccall librccl.make_longlong4(x::Clonglong, y::Clonglong, z::Clonglong,
                                  w::Clonglong)::longlong4
end

function hipCreateChannelDesc(x, y, z, w, f)
    @ccall librccl.hipCreateChannelDesc(x::Cint, y::Cint, z::Cint, w::Cint,
                                        f::hipChannelFormatKind)::hipChannelFormatDesc
end

mutable struct __hip_texture end

const hipTextureObject_t = Ptr{__hip_texture}

@cenum hipTextureAddressMode::UInt32 begin
    hipAddressModeWrap = 0
    hipAddressModeClamp = 1
    hipAddressModeMirror = 2
    hipAddressModeBorder = 3
end

@cenum hipTextureFilterMode::UInt32 begin
    hipFilterModePoint = 0
    hipFilterModeLinear = 1
end

@cenum hipTextureReadMode::UInt32 begin
    hipReadModeElementType = 0
    hipReadModeNormalizedFloat = 1
end

struct textureReference
    normalized::Cint
    readMode::hipTextureReadMode
    filterMode::hipTextureFilterMode
    addressMode::NTuple{3,hipTextureAddressMode}
    channelDesc::hipChannelFormatDesc
    sRGB::Cint
    maxAnisotropy::Cuint
    mipmapFilterMode::hipTextureFilterMode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    textureObject::hipTextureObject_t
    numChannels::Cint
    format::hipArray_Format
end

struct hipTextureDesc
    addressMode::NTuple{3,hipTextureAddressMode}
    filterMode::hipTextureFilterMode
    readMode::hipTextureReadMode
    sRGB::Cint
    borderColor::NTuple{4,Cfloat}
    normalizedCoords::Cint
    maxAnisotropy::Cuint
    mipmapFilterMode::hipTextureFilterMode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
end

mutable struct __hip_surface end

const hipSurfaceObject_t = Ptr{__hip_surface}

struct surfaceReference
    surfaceObject::hipSurfaceObject_t
end

@cenum hipSurfaceBoundaryMode::UInt32 begin
    hipBoundaryModeZero = 0
    hipBoundaryModeTrap = 1
    hipBoundaryModeClamp = 2
end

mutable struct ihipCtx_t end

const hipCtx_t = Ptr{ihipCtx_t}

const hipDevice_t = Cint

@cenum hipDeviceP2PAttr::UInt32 begin
    hipDevP2PAttrPerformanceRank = 0
    hipDevP2PAttrAccessSupported = 1
    hipDevP2PAttrNativeAtomicSupported = 2
    hipDevP2PAttrHipArrayAccessSupported = 3
end

struct hipIpcMemHandle_st
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipIpcMemHandle_st}, f::Symbol)
    f === :reserved && return Ptr{NTuple{64,Cchar}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipIpcMemHandle_st, f::Symbol)
    r = Ref{hipIpcMemHandle_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipIpcMemHandle_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipIpcMemHandle_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipIpcMemHandle_st, private::Bool=false)
    return (:reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipIpcMemHandle_t = hipIpcMemHandle_st

struct hipIpcEventHandle_st
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipIpcEventHandle_st}, f::Symbol)
    f === :reserved && return Ptr{NTuple{64,Cchar}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipIpcEventHandle_st, f::Symbol)
    r = Ref{hipIpcEventHandle_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipIpcEventHandle_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipIpcEventHandle_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipIpcEventHandle_st, private::Bool=false)
    return (:reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipIpcEventHandle_t = hipIpcEventHandle_st

mutable struct ihipModule_t end

const hipModule_t = Ptr{ihipModule_t}

mutable struct ihipModuleSymbol_t end

const hipFunction_t = Ptr{ihipModuleSymbol_t}

mutable struct ihipLinkState_t end

const hipLinkState_t = Ptr{ihipLinkState_t}

mutable struct ihipMemPoolHandle_t end

const hipMemPool_t = Ptr{ihipMemPoolHandle_t}

struct hipFuncAttributes
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{hipFuncAttributes}, f::Symbol)
    f === :binaryVersion && return Ptr{Cint}(x + 0)
    f === :cacheModeCA && return Ptr{Cint}(x + 4)
    f === :constSizeBytes && return Ptr{Csize_t}(x + 8)
    f === :localSizeBytes && return Ptr{Csize_t}(x + 16)
    f === :maxDynamicSharedSizeBytes && return Ptr{Cint}(x + 24)
    f === :maxThreadsPerBlock && return Ptr{Cint}(x + 28)
    f === :numRegs && return Ptr{Cint}(x + 32)
    f === :preferredShmemCarveout && return Ptr{Cint}(x + 36)
    f === :ptxVersion && return Ptr{Cint}(x + 40)
    f === :sharedSizeBytes && return Ptr{Csize_t}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::hipFuncAttributes, f::Symbol)
    r = Ref{hipFuncAttributes}(x)
    ptr = Base.unsafe_convert(Ptr{hipFuncAttributes}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipFuncAttributes}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipFuncAttributes, private::Bool=false)
    return (:binaryVersion, :cacheModeCA, :constSizeBytes, :localSizeBytes,
            :maxDynamicSharedSizeBytes, :maxThreadsPerBlock, :numRegs,
            :preferredShmemCarveout, :ptxVersion, :sharedSizeBytes,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

mutable struct ihipEvent_t end

const hipEvent_t = Ptr{ihipEvent_t}

@cenum hipLimit_t::UInt32 begin
    hipLimitStackSize = 0
    hipLimitPrintfFifoSize = 1
    hipLimitMallocHeapSize = 2
    hipLimitRange = 3
end

@cenum hipStreamBatchMemOpType::UInt32 begin
    hipStreamMemOpWaitValue32 = 1
    hipStreamMemOpWriteValue32 = 2
    hipStreamMemOpWaitValue64 = 4
    hipStreamMemOpWriteValue64 = 5
    hipStreamMemOpBarrier = 6
    hipStreamMemOpFlushRemoteWrites = 3
end

struct hipStreamBatchMemOpParams_union
    data::NTuple{48,UInt8}
end

function Base.getproperty(x::Ptr{hipStreamBatchMemOpParams_union}, f::Symbol)
    f === :operation && return Ptr{hipStreamBatchMemOpType}(x + 0)
    f === :waitValue && return Ptr{hipStreamMemOpWaitValueParams_t}(x + 0)
    f === :writeValue && return Ptr{hipStreamMemOpWriteValueParams_t}(x + 0)
    f === :flushRemoteWrites && return Ptr{hipStreamMemOpFlushRemoteWritesParams_t}(x + 0)
    f === :memoryBarrier && return Ptr{hipStreamMemOpMemoryBarrierParams_t}(x + 0)
    f === :pad && return Ptr{NTuple{6,UInt64}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipStreamBatchMemOpParams_union, f::Symbol)
    r = Ref{hipStreamBatchMemOpParams_union}(x)
    ptr = Base.unsafe_convert(Ptr{hipStreamBatchMemOpParams_union}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipStreamBatchMemOpParams_union}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipStreamBatchMemOpParams_union, private::Bool=false)
    return (:operation, :waitValue, :writeValue, :flushRemoteWrites, :memoryBarrier, :pad,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipStreamBatchMemOpParams = hipStreamBatchMemOpParams_union

struct hipBatchMemOpNodeParams
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{hipBatchMemOpNodeParams}, f::Symbol)
    f === :ctx && return Ptr{hipCtx_t}(x + 0)
    f === :count && return Ptr{Cuint}(x + 8)
    f === :paramArray && return Ptr{Ptr{hipStreamBatchMemOpParams}}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::hipBatchMemOpNodeParams, f::Symbol)
    r = Ref{hipBatchMemOpNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipBatchMemOpNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipBatchMemOpNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipBatchMemOpNodeParams, private::Bool=false)
    return (:ctx, :count, :paramArray, :flags, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipMemoryAdvise::UInt32 begin
    hipMemAdviseSetReadMostly = 1
    hipMemAdviseUnsetReadMostly = 2
    hipMemAdviseSetPreferredLocation = 3
    hipMemAdviseUnsetPreferredLocation = 4
    hipMemAdviseSetAccessedBy = 5
    hipMemAdviseUnsetAccessedBy = 6
    hipMemAdviseSetCoarseGrain = 100
    hipMemAdviseUnsetCoarseGrain = 101
end

@cenum hipMemRangeCoherencyMode::UInt32 begin
    hipMemRangeCoherencyModeFineGrain = 0
    hipMemRangeCoherencyModeCoarseGrain = 1
    hipMemRangeCoherencyModeIndeterminate = 2
end

@cenum hipMemRangeAttribute::UInt32 begin
    hipMemRangeAttributeReadMostly = 1
    hipMemRangeAttributePreferredLocation = 2
    hipMemRangeAttributeAccessedBy = 3
    hipMemRangeAttributeLastPrefetchLocation = 4
    hipMemRangeAttributeCoherencyMode = 100
end

@cenum hipMemPoolAttr::UInt32 begin
    hipMemPoolReuseFollowEventDependencies = 1
    hipMemPoolReuseAllowOpportunistic = 2
    hipMemPoolReuseAllowInternalDependencies = 3
    hipMemPoolAttrReleaseThreshold = 4
    hipMemPoolAttrReservedMemCurrent = 5
    hipMemPoolAttrReservedMemHigh = 6
    hipMemPoolAttrUsedMemCurrent = 7
    hipMemPoolAttrUsedMemHigh = 8
end

@cenum hipMemLocationType::UInt32 begin
    hipMemLocationTypeInvalid = 0
    hipMemLocationTypeDevice = 1
end

struct hipMemLocation
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipMemLocation}, f::Symbol)
    f === :type && return Ptr{hipMemLocationType}(x + 0)
    f === :id && return Ptr{Cint}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemLocation, f::Symbol)
    r = Ref{hipMemLocation}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemLocation}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemLocation}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemLocation, private::Bool=false)
    return (:type, :id, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipMemAccessFlags::UInt32 begin
    hipMemAccessFlagsProtNone = 0
    hipMemAccessFlagsProtRead = 1
    hipMemAccessFlagsProtReadWrite = 3
end

struct hipMemAccessDesc
    data::NTuple{12,UInt8}
end

function Base.getproperty(x::Ptr{hipMemAccessDesc}, f::Symbol)
    f === :location && return Ptr{hipMemLocation}(x + 0)
    f === :flags && return Ptr{hipMemAccessFlags}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemAccessDesc, f::Symbol)
    r = Ref{hipMemAccessDesc}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemAccessDesc}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemAccessDesc}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemAccessDesc, private::Bool=false)
    return (:location, :flags, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipMemAllocationType::UInt32 begin
    hipMemAllocationTypeInvalid = 0
    hipMemAllocationTypePinned = 1
    hipMemAllocationTypeMax = 2147483647
end

@cenum hipMemAllocationHandleType::UInt32 begin
    hipMemHandleTypeNone = 0
    hipMemHandleTypePosixFileDescriptor = 1
    hipMemHandleTypeWin32 = 2
    hipMemHandleTypeWin32Kmt = 4
end

struct hipMemPoolProps
    data::NTuple{88,UInt8}
end

function Base.getproperty(x::Ptr{hipMemPoolProps}, f::Symbol)
    f === :allocType && return Ptr{hipMemAllocationType}(x + 0)
    f === :handleTypes && return Ptr{hipMemAllocationHandleType}(x + 4)
    f === :location && return Ptr{hipMemLocation}(x + 8)
    f === :win32SecurityAttributes && return Ptr{Ptr{Cvoid}}(x + 16)
    f === :maxSize && return Ptr{Csize_t}(x + 24)
    f === :reserved && return Ptr{NTuple{56,Cuchar}}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemPoolProps, f::Symbol)
    r = Ref{hipMemPoolProps}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemPoolProps}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemPoolProps}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemPoolProps, private::Bool=false)
    return (:allocType, :handleTypes, :location, :win32SecurityAttributes, :maxSize,
            :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipMemPoolPtrExportData
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipMemPoolPtrExportData}, f::Symbol)
    f === :reserved && return Ptr{NTuple{64,Cuchar}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemPoolPtrExportData, f::Symbol)
    r = Ref{hipMemPoolPtrExportData}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemPoolPtrExportData}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemPoolPtrExportData}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemPoolPtrExportData, private::Bool=false)
    return (:reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipFuncAttribute::UInt32 begin
    hipFuncAttributeMaxDynamicSharedMemorySize = 8
    hipFuncAttributePreferredSharedMemoryCarveout = 9
    hipFuncAttributeMax = 10
end

@cenum hipFuncCache_t::UInt32 begin
    hipFuncCachePreferNone = 0
    hipFuncCachePreferShared = 1
    hipFuncCachePreferL1 = 2
    hipFuncCachePreferEqual = 3
end

@cenum hipSharedMemConfig::UInt32 begin
    hipSharedMemBankSizeDefault = 0
    hipSharedMemBankSizeFourByte = 1
    hipSharedMemBankSizeEightByte = 2
end

struct dim3
    data::NTuple{12,UInt8}
end

function Base.getproperty(x::Ptr{dim3}, f::Symbol)
    f === :x && return Ptr{UInt32}(x + 0)
    f === :y && return Ptr{UInt32}(x + 4)
    f === :z && return Ptr{UInt32}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::dim3, f::Symbol)
    r = Ref{dim3}(x)
    ptr = Base.unsafe_convert(Ptr{dim3}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{dim3}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::dim3, private::Bool=false)
    return (:x, :y, :z, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipLaunchParams_t
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{hipLaunchParams_t}, f::Symbol)
    f === :func && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :gridDim && return Ptr{dim3}(x + 8)
    f === :blockDim && return Ptr{dim3}(x + 20)
    f === :args && return Ptr{Ptr{Ptr{Cvoid}}}(x + 32)
    f === :sharedMem && return Ptr{Csize_t}(x + 40)
    f === :stream && return Ptr{hipStream_t}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::hipLaunchParams_t, f::Symbol)
    r = Ref{hipLaunchParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipLaunchParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipLaunchParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipLaunchParams_t, private::Bool=false)
    return (:func, :gridDim, :blockDim, :args, :sharedMem, :stream,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipLaunchParams = hipLaunchParams_t

struct hipFunctionLaunchParams_t
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{hipFunctionLaunchParams_t}, f::Symbol)
    f === :_function && return Ptr{hipFunction_t}(x + 0)
    f === :gridDimX && return Ptr{Cuint}(x + 8)
    f === :gridDimY && return Ptr{Cuint}(x + 12)
    f === :gridDimZ && return Ptr{Cuint}(x + 16)
    f === :blockDimX && return Ptr{Cuint}(x + 20)
    f === :blockDimY && return Ptr{Cuint}(x + 24)
    f === :blockDimZ && return Ptr{Cuint}(x + 28)
    f === :sharedMemBytes && return Ptr{Cuint}(x + 32)
    f === :hStream && return Ptr{hipStream_t}(x + 40)
    f === :kernelParams && return Ptr{Ptr{Ptr{Cvoid}}}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::hipFunctionLaunchParams_t, f::Symbol)
    r = Ref{hipFunctionLaunchParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipFunctionLaunchParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipFunctionLaunchParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipFunctionLaunchParams_t, private::Bool=false)
    return (:_function, :gridDimX, :gridDimY, :gridDimZ, :blockDimX, :blockDimY, :blockDimZ,
            :sharedMemBytes, :hStream, :kernelParams, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipFunctionLaunchParams = hipFunctionLaunchParams_t

@cenum hipExternalMemoryHandleType_enum::UInt32 begin
    hipExternalMemoryHandleTypeOpaqueFd = 1
    hipExternalMemoryHandleTypeOpaqueWin32 = 2
    hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    hipExternalMemoryHandleTypeD3D12Heap = 4
    hipExternalMemoryHandleTypeD3D12Resource = 5
    hipExternalMemoryHandleTypeD3D11Resource = 6
    hipExternalMemoryHandleTypeD3D11ResourceKmt = 7
    hipExternalMemoryHandleTypeNvSciBuf = 8
end

const hipExternalMemoryHandleType = hipExternalMemoryHandleType_enum

struct var"##Ctag#248"
    data::NTuple{16,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#248"}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    f === :win32 && return Ptr{var"##Ctag#249"}(x + 0)
    f === :nvSciBufObject && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#248", f::Symbol)
    r = Ref{var"##Ctag#248"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#248"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#248"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#248", private::Bool=false)
    return (:fd, :win32, :nvSciBufObject, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalMemoryHandleDesc_st
    data::NTuple{104,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalMemoryHandleDesc_st}, f::Symbol)
    f === :type && return Ptr{hipExternalMemoryHandleType}(x + 0)
    f === :handle && return Ptr{var"##Ctag#248"}(x + 8)
    f === :size && return Ptr{Culonglong}(x + 24)
    f === :flags && return Ptr{Cuint}(x + 32)
    f === :reserved && return Ptr{NTuple{16,Cuint}}(x + 36)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalMemoryHandleDesc_st, f::Symbol)
    r = Ref{hipExternalMemoryHandleDesc_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalMemoryHandleDesc_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalMemoryHandleDesc_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalMemoryHandleDesc_st, private::Bool=false)
    return (:type, :handle, :size, :flags, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalMemoryHandleDesc = hipExternalMemoryHandleDesc_st

struct hipExternalMemoryBufferDesc_st
    data::NTuple{88,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalMemoryBufferDesc_st}, f::Symbol)
    f === :offset && return Ptr{Culonglong}(x + 0)
    f === :size && return Ptr{Culonglong}(x + 8)
    f === :flags && return Ptr{Cuint}(x + 16)
    f === :reserved && return Ptr{NTuple{16,Cuint}}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalMemoryBufferDesc_st, f::Symbol)
    r = Ref{hipExternalMemoryBufferDesc_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalMemoryBufferDesc_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalMemoryBufferDesc_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalMemoryBufferDesc_st, private::Bool=false)
    return (:offset, :size, :flags, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalMemoryBufferDesc = hipExternalMemoryBufferDesc_st

struct hipExternalMemoryMipmappedArrayDesc_st
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalMemoryMipmappedArrayDesc_st}, f::Symbol)
    f === :offset && return Ptr{Culonglong}(x + 0)
    f === :formatDesc && return Ptr{hipChannelFormatDesc}(x + 8)
    f === :extent && return Ptr{hipExtent}(x + 32)
    f === :flags && return Ptr{Cuint}(x + 56)
    f === :numLevels && return Ptr{Cuint}(x + 60)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalMemoryMipmappedArrayDesc_st, f::Symbol)
    r = Ref{hipExternalMemoryMipmappedArrayDesc_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalMemoryMipmappedArrayDesc_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalMemoryMipmappedArrayDesc_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalMemoryMipmappedArrayDesc_st, private::Bool=false)
    return (:offset, :formatDesc, :extent, :flags, :numLevels, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalMemoryMipmappedArrayDesc = hipExternalMemoryMipmappedArrayDesc_st

const hipExternalMemory_t = Ptr{Cvoid}

@cenum hipExternalSemaphoreHandleType_enum::UInt32 begin
    hipExternalSemaphoreHandleTypeOpaqueFd = 1
    hipExternalSemaphoreHandleTypeOpaqueWin32 = 2
    hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    hipExternalSemaphoreHandleTypeD3D12Fence = 4
    hipExternalSemaphoreHandleTypeD3D11Fence = 5
    hipExternalSemaphoreHandleTypeNvSciSync = 6
    hipExternalSemaphoreHandleTypeKeyedMutex = 7
    hipExternalSemaphoreHandleTypeKeyedMutexKmt = 8
    hipExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9
    hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10
end

const hipExternalSemaphoreHandleType = hipExternalSemaphoreHandleType_enum

struct var"##Ctag#241"
    data::NTuple{16,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#241"}, f::Symbol)
    f === :fd && return Ptr{Cint}(x + 0)
    f === :win32 && return Ptr{var"##Ctag#242"}(x + 0)
    f === :NvSciSyncObj && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#241", f::Symbol)
    r = Ref{var"##Ctag#241"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#241"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#241"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#241", private::Bool=false)
    return (:fd, :win32, :NvSciSyncObj, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalSemaphoreHandleDesc_st
    data::NTuple{96,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalSemaphoreHandleDesc_st}, f::Symbol)
    f === :type && return Ptr{hipExternalSemaphoreHandleType}(x + 0)
    f === :handle && return Ptr{var"##Ctag#241"}(x + 8)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :reserved && return Ptr{NTuple{16,Cuint}}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalSemaphoreHandleDesc_st, f::Symbol)
    r = Ref{hipExternalSemaphoreHandleDesc_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalSemaphoreHandleDesc_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalSemaphoreHandleDesc_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalSemaphoreHandleDesc_st, private::Bool=false)
    return (:type, :handle, :flags, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalSemaphoreHandleDesc = hipExternalSemaphoreHandleDesc_st

const hipExternalSemaphore_t = Ptr{Cvoid}

struct var"##Ctag#238"
    value::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#238"}, f::Symbol)
    f === :value && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#238", f::Symbol)
    r = Ref{var"##Ctag#238"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#238"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#238"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#239"
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#239"}, f::Symbol)
    f === :fence && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :reserved && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#239", f::Symbol)
    r = Ref{var"##Ctag#239"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#239"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#239"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#239", private::Bool=false)
    return (:fence, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#240"
    key::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#240"}, f::Symbol)
    f === :key && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#240", f::Symbol)
    r = Ref{var"##Ctag#240"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#240"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#240"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#237"
    data::NTuple{72,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#237"}, f::Symbol)
    f === :fence && return Ptr{var"##Ctag#238"}(x + 0)
    f === :nvSciSync && return Ptr{var"##Ctag#239"}(x + 8)
    f === :keyedMutex && return Ptr{var"##Ctag#240"}(x + 16)
    f === :reserved && return Ptr{NTuple{12,Cuint}}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#237", f::Symbol)
    r = Ref{var"##Ctag#237"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#237"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#237"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#237", private::Bool=false)
    return (:fence, :nvSciSync, :keyedMutex, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalSemaphoreSignalParams_st
    data::NTuple{144,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalSemaphoreSignalParams_st}, f::Symbol)
    f === :params && return Ptr{Cvoid}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 72)
    f === :reserved && return Ptr{NTuple{16,Cuint}}(x + 76)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalSemaphoreSignalParams_st, f::Symbol)
    r = Ref{hipExternalSemaphoreSignalParams_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalSemaphoreSignalParams_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalSemaphoreSignalParams_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalSemaphoreSignalParams_st, private::Bool=false)
    return (:params, :flags, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalSemaphoreSignalParams = hipExternalSemaphoreSignalParams_st

struct var"##Ctag#258"
    value::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#258"}, f::Symbol)
    f === :value && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#258", f::Symbol)
    r = Ref{var"##Ctag#258"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#258"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#258"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#259"
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#259"}, f::Symbol)
    f === :fence && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :reserved && return Ptr{Culonglong}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#259", f::Symbol)
    r = Ref{var"##Ctag#259"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#259"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#259"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#259", private::Bool=false)
    return (:fence, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#260"
    key::Culonglong
    timeoutMs::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#260"}, f::Symbol)
    f === :key && return Ptr{Culonglong}(x + 0)
    f === :timeoutMs && return Ptr{Cuint}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#260", f::Symbol)
    r = Ref{var"##Ctag#260"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#260"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#260"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#257"
    data::NTuple{72,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#257"}, f::Symbol)
    f === :fence && return Ptr{var"##Ctag#258"}(x + 0)
    f === :nvSciSync && return Ptr{var"##Ctag#259"}(x + 8)
    f === :keyedMutex && return Ptr{var"##Ctag#260"}(x + 16)
    f === :reserved && return Ptr{NTuple{10,Cuint}}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#257", f::Symbol)
    r = Ref{var"##Ctag#257"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#257"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#257"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#257", private::Bool=false)
    return (:fence, :nvSciSync, :keyedMutex, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalSemaphoreWaitParams_st
    data::NTuple{144,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalSemaphoreWaitParams_st}, f::Symbol)
    f === :params && return Ptr{Cvoid}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 72)
    f === :reserved && return Ptr{NTuple{16,Cuint}}(x + 76)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalSemaphoreWaitParams_st, f::Symbol)
    r = Ref{hipExternalSemaphoreWaitParams_st}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalSemaphoreWaitParams_st}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalSemaphoreWaitParams_st}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalSemaphoreWaitParams_st, private::Bool=false)
    return (:params, :flags, :reserved, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

const hipExternalSemaphoreWaitParams = hipExternalSemaphoreWaitParams_st

function __hipGetPCH(pch, size)
    @ccall librccl.__hipGetPCH(pch::Ptr{Cstring}, size::Ptr{Cuint})::Cvoid
end

@cenum hipGraphicsRegisterFlags::UInt32 begin
    hipGraphicsRegisterFlagsNone = 0
    hipGraphicsRegisterFlagsReadOnly = 1
    hipGraphicsRegisterFlagsWriteDiscard = 2
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4
    hipGraphicsRegisterFlagsTextureGather = 8
end

mutable struct _hipGraphicsResource end

const hipGraphicsResource = _hipGraphicsResource

const hipGraphicsResource_t = Ptr{hipGraphicsResource}

mutable struct ihipGraph end

const hipGraph_t = Ptr{ihipGraph}

mutable struct hipGraphNode end

const hipGraphNode_t = Ptr{hipGraphNode}

mutable struct hipGraphExec end

const hipGraphExec_t = Ptr{hipGraphExec}

mutable struct hipUserObject end

const hipUserObject_t = Ptr{hipUserObject}

@cenum hipGraphNodeType::UInt32 begin
    hipGraphNodeTypeKernel = 0
    hipGraphNodeTypeMemcpy = 1
    hipGraphNodeTypeMemset = 2
    hipGraphNodeTypeHost = 3
    hipGraphNodeTypeGraph = 4
    hipGraphNodeTypeEmpty = 5
    hipGraphNodeTypeWaitEvent = 6
    hipGraphNodeTypeEventRecord = 7
    hipGraphNodeTypeExtSemaphoreSignal = 8
    hipGraphNodeTypeExtSemaphoreWait = 9
    hipGraphNodeTypeMemAlloc = 10
    hipGraphNodeTypeMemFree = 11
    hipGraphNodeTypeMemcpyFromSymbol = 12
    hipGraphNodeTypeMemcpyToSymbol = 13
    hipGraphNodeTypeBatchMemOp = 14
    hipGraphNodeTypeCount = 15
end

# typedef void ( * hipHostFn_t ) ( void * userData )
const hipHostFn_t = Ptr{Cvoid}

struct hipHostNodeParams
    data::NTuple{16,UInt8}
end

function Base.getproperty(x::Ptr{hipHostNodeParams}, f::Symbol)
    f === :fn && return Ptr{hipHostFn_t}(x + 0)
    f === :userData && return Ptr{Ptr{Cvoid}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::hipHostNodeParams, f::Symbol)
    r = Ref{hipHostNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipHostNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipHostNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipHostNodeParams, private::Bool=false)
    return (:fn, :userData, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipKernelNodeParams
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{hipKernelNodeParams}, f::Symbol)
    f === :blockDim && return Ptr{dim3}(x + 0)
    f === :extra && return Ptr{Ptr{Ptr{Cvoid}}}(x + 16)
    f === :func && return Ptr{Ptr{Cvoid}}(x + 24)
    f === :gridDim && return Ptr{dim3}(x + 32)
    f === :kernelParams && return Ptr{Ptr{Ptr{Cvoid}}}(x + 48)
    f === :sharedMemBytes && return Ptr{Cuint}(x + 56)
    return getfield(x, f)
end

function Base.getproperty(x::hipKernelNodeParams, f::Symbol)
    r = Ref{hipKernelNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipKernelNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipKernelNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipKernelNodeParams, private::Bool=false)
    return (:blockDim, :extra, :func, :gridDim, :kernelParams, :sharedMemBytes,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipMemsetParams
    data::NTuple{48,UInt8}
end

function Base.getproperty(x::Ptr{hipMemsetParams}, f::Symbol)
    f === :dst && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :elementSize && return Ptr{Cuint}(x + 8)
    f === :height && return Ptr{Csize_t}(x + 16)
    f === :pitch && return Ptr{Csize_t}(x + 24)
    f === :value && return Ptr{Cuint}(x + 32)
    f === :width && return Ptr{Csize_t}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemsetParams, f::Symbol)
    r = Ref{hipMemsetParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemsetParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemsetParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemsetParams, private::Bool=false)
    return (:dst, :elementSize, :height, :pitch, :value, :width,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipMemAllocNodeParams
    data::NTuple{120,UInt8}
end

function Base.getproperty(x::Ptr{hipMemAllocNodeParams}, f::Symbol)
    f === :poolProps && return Ptr{hipMemPoolProps}(x + 0)
    f === :accessDescs && return Ptr{Ptr{hipMemAccessDesc}}(x + 88)
    f === :accessDescCount && return Ptr{Csize_t}(x + 96)
    f === :bytesize && return Ptr{Csize_t}(x + 104)
    f === :dptr && return Ptr{Ptr{Cvoid}}(x + 112)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemAllocNodeParams, f::Symbol)
    r = Ref{hipMemAllocNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemAllocNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemAllocNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemAllocNodeParams, private::Bool=false)
    return (:poolProps, :accessDescs, :accessDescCount, :bytesize, :dptr,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipAccessProperty::UInt32 begin
    hipAccessPropertyNormal = 0
    hipAccessPropertyStreaming = 1
    hipAccessPropertyPersisting = 2
end

struct hipAccessPolicyWindow
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{hipAccessPolicyWindow}, f::Symbol)
    f === :base_ptr && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :hitProp && return Ptr{hipAccessProperty}(x + 8)
    f === :hitRatio && return Ptr{Cfloat}(x + 12)
    f === :missProp && return Ptr{hipAccessProperty}(x + 16)
    f === :num_bytes && return Ptr{Csize_t}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::hipAccessPolicyWindow, f::Symbol)
    r = Ref{hipAccessPolicyWindow}(x)
    ptr = Base.unsafe_convert(Ptr{hipAccessPolicyWindow}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipAccessPolicyWindow}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipAccessPolicyWindow, private::Bool=false)
    return (:base_ptr, :hitProp, :hitRatio, :missProp, :num_bytes,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct HIP_MEMSET_NODE_PARAMS
    data::NTuple{40,UInt8}
end

function Base.getproperty(x::Ptr{HIP_MEMSET_NODE_PARAMS}, f::Symbol)
    f === :dst && return Ptr{hipDeviceptr_t}(x + 0)
    f === :pitch && return Ptr{Csize_t}(x + 8)
    f === :value && return Ptr{Cuint}(x + 16)
    f === :elementSize && return Ptr{Cuint}(x + 20)
    f === :width && return Ptr{Csize_t}(x + 24)
    f === :height && return Ptr{Csize_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::HIP_MEMSET_NODE_PARAMS, f::Symbol)
    r = Ref{HIP_MEMSET_NODE_PARAMS}(x)
    ptr = Base.unsafe_convert(Ptr{HIP_MEMSET_NODE_PARAMS}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{HIP_MEMSET_NODE_PARAMS}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::HIP_MEMSET_NODE_PARAMS, private::Bool=false)
    return (:dst, :pitch, :value, :elementSize, :width, :height,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipGraphExecUpdateResult::UInt32 begin
    hipGraphExecUpdateSuccess = 0
    hipGraphExecUpdateError = 1
    hipGraphExecUpdateErrorTopologyChanged = 2
    hipGraphExecUpdateErrorNodeTypeChanged = 3
    hipGraphExecUpdateErrorFunctionChanged = 4
    hipGraphExecUpdateErrorParametersChanged = 5
    hipGraphExecUpdateErrorNotSupported = 6
    hipGraphExecUpdateErrorUnsupportedFunctionChange = 7
end

@cenum hipStreamCaptureMode::UInt32 begin
    hipStreamCaptureModeGlobal = 0
    hipStreamCaptureModeThreadLocal = 1
    hipStreamCaptureModeRelaxed = 2
end

@cenum hipStreamCaptureStatus::UInt32 begin
    hipStreamCaptureStatusNone = 0
    hipStreamCaptureStatusActive = 1
    hipStreamCaptureStatusInvalidated = 2
end

@cenum hipStreamUpdateCaptureDependenciesFlags::UInt32 begin
    hipStreamAddCaptureDependencies = 0
    hipStreamSetCaptureDependencies = 1
end

@cenum hipGraphMemAttributeType::UInt32 begin
    hipGraphMemAttrUsedMemCurrent = 0
    hipGraphMemAttrUsedMemHigh = 1
    hipGraphMemAttrReservedMemCurrent = 2
    hipGraphMemAttrReservedMemHigh = 3
end

@cenum hipUserObjectFlags::UInt32 begin
    hipUserObjectNoDestructorSync = 1
end

@cenum hipUserObjectRetainFlags::UInt32 begin
    hipGraphUserObjectMove = 1
end

@cenum hipGraphInstantiateFlags::UInt32 begin
    hipGraphInstantiateFlagAutoFreeOnLaunch = 1
    hipGraphInstantiateFlagUpload = 2
    hipGraphInstantiateFlagDeviceLaunch = 4
    hipGraphInstantiateFlagUseNodePriority = 8
end

@cenum hipGraphDebugDotFlags::UInt32 begin
    hipGraphDebugDotFlagsVerbose = 1
    hipGraphDebugDotFlagsKernelNodeParams = 4
    hipGraphDebugDotFlagsMemcpyNodeParams = 8
    hipGraphDebugDotFlagsMemsetNodeParams = 16
    hipGraphDebugDotFlagsHostNodeParams = 32
    hipGraphDebugDotFlagsEventNodeParams = 64
    hipGraphDebugDotFlagsExtSemasSignalNodeParams = 128
    hipGraphDebugDotFlagsExtSemasWaitNodeParams = 256
    hipGraphDebugDotFlagsKernelNodeAttributes = 512
    hipGraphDebugDotFlagsHandles = 1024
end

@cenum hipGraphInstantiateResult::UInt32 begin
    hipGraphInstantiateSuccess = 0
    hipGraphInstantiateError = 1
    hipGraphInstantiateInvalidStructure = 2
    hipGraphInstantiateNodeOperationNotSupported = 3
    hipGraphInstantiateMultipleDevicesNotSupported = 4
end

struct hipGraphInstantiateParams
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{hipGraphInstantiateParams}, f::Symbol)
    f === :errNode_out && return Ptr{hipGraphNode_t}(x + 0)
    f === :flags && return Ptr{Culonglong}(x + 8)
    f === :result_out && return Ptr{hipGraphInstantiateResult}(x + 16)
    f === :uploadStream && return Ptr{hipStream_t}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::hipGraphInstantiateParams, f::Symbol)
    r = Ref{hipGraphInstantiateParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipGraphInstantiateParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipGraphInstantiateParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipGraphInstantiateParams, private::Bool=false)
    return (:errNode_out, :flags, :result_out, :uploadStream, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#256"
    compressionType::Cuchar
    gpuDirectRDMACapable::Cuchar
    usage::Cushort
end
function Base.getproperty(x::Ptr{var"##Ctag#256"}, f::Symbol)
    f === :compressionType && return Ptr{Cuchar}(x + 0)
    f === :gpuDirectRDMACapable && return Ptr{Cuchar}(x + 1)
    f === :usage && return Ptr{Cushort}(x + 2)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#256", f::Symbol)
    r = Ref{var"##Ctag#256"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#256"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#256"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct hipMemAllocationProp
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{hipMemAllocationProp}, f::Symbol)
    f === :type && return Ptr{hipMemAllocationType}(x + 0)
    f === :requestedHandleType && return Ptr{hipMemAllocationHandleType}(x + 4)
    f === :location && return Ptr{hipMemLocation}(x + 8)
    f === :win32HandleMetaData && return Ptr{Ptr{Cvoid}}(x + 16)
    f === :allocFlags && return Ptr{var"##Ctag#256"}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemAllocationProp, f::Symbol)
    r = Ref{hipMemAllocationProp}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemAllocationProp}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemAllocationProp}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemAllocationProp, private::Bool=false)
    return (:type, :requestedHandleType, :location, :win32HandleMetaData, :allocFlags,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalSemaphoreSignalNodeParams
    data::NTuple{24,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalSemaphoreSignalNodeParams}, f::Symbol)
    f === :extSemArray && return Ptr{Ptr{hipExternalSemaphore_t}}(x + 0)
    f === :paramsArray && return Ptr{Ptr{hipExternalSemaphoreSignalParams}}(x + 8)
    f === :numExtSems && return Ptr{Cuint}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalSemaphoreSignalNodeParams, f::Symbol)
    r = Ref{hipExternalSemaphoreSignalNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalSemaphoreSignalNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalSemaphoreSignalNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalSemaphoreSignalNodeParams, private::Bool=false)
    return (:extSemArray, :paramsArray, :numExtSems, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipExternalSemaphoreWaitNodeParams
    data::NTuple{24,UInt8}
end

function Base.getproperty(x::Ptr{hipExternalSemaphoreWaitNodeParams}, f::Symbol)
    f === :extSemArray && return Ptr{Ptr{hipExternalSemaphore_t}}(x + 0)
    f === :paramsArray && return Ptr{Ptr{hipExternalSemaphoreWaitParams}}(x + 8)
    f === :numExtSems && return Ptr{Cuint}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::hipExternalSemaphoreWaitNodeParams, f::Symbol)
    r = Ref{hipExternalSemaphoreWaitNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipExternalSemaphoreWaitNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipExternalSemaphoreWaitNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipExternalSemaphoreWaitNodeParams, private::Bool=false)
    return (:extSemArray, :paramsArray, :numExtSems, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

mutable struct ihipMemGenericAllocationHandle end

const hipMemGenericAllocationHandle_t = Ptr{ihipMemGenericAllocationHandle}

@cenum hipMemAllocationGranularity_flags::UInt32 begin
    hipMemAllocationGranularityMinimum = 0
    hipMemAllocationGranularityRecommended = 1
end

@cenum hipMemHandleType::UInt32 begin
    hipMemHandleTypeGeneric = 0
end

@cenum hipMemOperationType::UInt32 begin
    hipMemOperationTypeMap = 1
    hipMemOperationTypeUnmap = 2
end

@cenum hipArraySparseSubresourceType::UInt32 begin
    hipArraySparseSubresourceTypeSparseLevel = 0
    hipArraySparseSubresourceTypeMiptail = 1
end

struct var"##Ctag#232"
    data::NTuple{64,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#232"}, f::Symbol)
    f === :mipmap && return Ptr{hipMipmappedArray}(x + 0)
    f === :array && return Ptr{hipArray_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#232", f::Symbol)
    r = Ref{var"##Ctag#232"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#232"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#232"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#232", private::Bool=false)
    return (:mipmap, :array, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#233"
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#233"}, f::Symbol)
    f === :sparseLevel && return Ptr{var"##Ctag#234"}(x + 0)
    f === :miptail && return Ptr{var"##Ctag#235"}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#233", f::Symbol)
    r = Ref{var"##Ctag#233"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#233"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#233"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#233", private::Bool=false)
    return (:sparseLevel, :miptail, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#236"
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{var"##Ctag#236"}, f::Symbol)
    f === :memHandle && return Ptr{hipMemGenericAllocationHandle_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#236", f::Symbol)
    r = Ref{var"##Ctag#236"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#236"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#236"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::var"##Ctag#236", private::Bool=false)
    return (:memHandle, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipArrayMapInfo
    data::NTuple{152,UInt8}
end

function Base.getproperty(x::Ptr{hipArrayMapInfo}, f::Symbol)
    f === :resourceType && return Ptr{hipResourceType}(x + 0)
    f === :resource && return Ptr{var"##Ctag#232"}(x + 8)
    f === :subresourceType && return Ptr{hipArraySparseSubresourceType}(x + 72)
    f === :subresource && return Ptr{var"##Ctag#233"}(x + 80)
    f === :memOperationType && return Ptr{hipMemOperationType}(x + 112)
    f === :memHandleType && return Ptr{hipMemHandleType}(x + 116)
    f === :memHandle && return Ptr{var"##Ctag#236"}(x + 120)
    f === :offset && return Ptr{Culonglong}(x + 128)
    f === :deviceBitMask && return Ptr{Cuint}(x + 136)
    f === :flags && return Ptr{Cuint}(x + 140)
    f === :reserved && return Ptr{NTuple{2,Cuint}}(x + 144)
    return getfield(x, f)
end

function Base.getproperty(x::hipArrayMapInfo, f::Symbol)
    r = Ref{hipArrayMapInfo}(x)
    ptr = Base.unsafe_convert(Ptr{hipArrayMapInfo}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipArrayMapInfo}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipArrayMapInfo, private::Bool=false)
    return (:resourceType, :resource, :subresourceType, :subresource, :memOperationType,
            :memHandleType, :memHandle, :offset, :deviceBitMask, :flags, :reserved,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipMemcpyNodeParams
    data::NTuple{176,UInt8}
end

function Base.getproperty(x::Ptr{hipMemcpyNodeParams}, f::Symbol)
    f === :flags && return Ptr{Cint}(x + 0)
    f === :reserved && return Ptr{NTuple{3,Cint}}(x + 4)
    f === :copyParams && return Ptr{hipMemcpy3DParms}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemcpyNodeParams, f::Symbol)
    r = Ref{hipMemcpyNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemcpyNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemcpyNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemcpyNodeParams, private::Bool=false)
    return (:flags, :reserved, :copyParams, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipChildGraphNodeParams
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipChildGraphNodeParams}, f::Symbol)
    f === :graph && return Ptr{hipGraph_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipChildGraphNodeParams, f::Symbol)
    r = Ref{hipChildGraphNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipChildGraphNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipChildGraphNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipChildGraphNodeParams, private::Bool=false)
    return (:graph, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipEventWaitNodeParams
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipEventWaitNodeParams}, f::Symbol)
    f === :event && return Ptr{hipEvent_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipEventWaitNodeParams, f::Symbol)
    r = Ref{hipEventWaitNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipEventWaitNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipEventWaitNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipEventWaitNodeParams, private::Bool=false)
    return (:event, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipEventRecordNodeParams
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipEventRecordNodeParams}, f::Symbol)
    f === :event && return Ptr{hipEvent_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipEventRecordNodeParams, f::Symbol)
    r = Ref{hipEventRecordNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipEventRecordNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipEventRecordNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipEventRecordNodeParams, private::Bool=false)
    return (:event, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipMemFreeNodeParams
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipMemFreeNodeParams}, f::Symbol)
    f === :dptr && return Ptr{Ptr{Cvoid}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::hipMemFreeNodeParams, f::Symbol)
    r = Ref{hipMemFreeNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipMemFreeNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipMemFreeNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipMemFreeNodeParams, private::Bool=false)
    return (:dptr, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipGraphNodeParams
    data::NTuple{256,UInt8}
end

function Base.getproperty(x::Ptr{hipGraphNodeParams}, f::Symbol)
    f === :type && return Ptr{hipGraphNodeType}(x + 0)
    f === :reserved0 && return Ptr{NTuple{3,Cint}}(x + 4)
    f === :reserved1 && return Ptr{NTuple{29,Clonglong}}(x + 16)
    f === :kernel && return Ptr{hipKernelNodeParams}(x + 16)
    f === :memcpy && return Ptr{hipMemcpyNodeParams}(x + 16)
    f === :memset && return Ptr{hipMemsetParams}(x + 16)
    f === :host && return Ptr{hipHostNodeParams}(x + 16)
    f === :graph && return Ptr{hipChildGraphNodeParams}(x + 16)
    f === :eventWait && return Ptr{hipEventWaitNodeParams}(x + 16)
    f === :eventRecord && return Ptr{hipEventRecordNodeParams}(x + 16)
    f === :extSemSignal && return Ptr{hipExternalSemaphoreSignalNodeParams}(x + 16)
    f === :extSemWait && return Ptr{hipExternalSemaphoreWaitNodeParams}(x + 16)
    f === :alloc && return Ptr{hipMemAllocNodeParams}(x + 16)
    f === :free && return Ptr{hipMemFreeNodeParams}(x + 16)
    f === :reserved2 && return Ptr{Clonglong}(x + 248)
    return getfield(x, f)
end

function Base.getproperty(x::hipGraphNodeParams, f::Symbol)
    r = Ref{hipGraphNodeParams}(x)
    ptr = Base.unsafe_convert(Ptr{hipGraphNodeParams}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipGraphNodeParams}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipGraphNodeParams, private::Bool=false)
    return (:type, :reserved0, :reserved1, :kernel, :memcpy, :memset, :host, :graph,
            :eventWait, :eventRecord, :extSemSignal, :extSemWait, :alloc, :free, :reserved2,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

@cenum hipGraphDependencyType::UInt32 begin
    hipGraphDependencyTypeDefault = 0
    hipGraphDependencyTypeProgrammatic = 1
end

struct hipGraphEdgeData
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipGraphEdgeData}, f::Symbol)
    f === :from_port && return Ptr{Cuchar}(x + 0)
    f === :reserved && return Ptr{NTuple{5,Cuchar}}(x + 1)
    f === :to_port && return Ptr{Cuchar}(x + 6)
    f === :type && return Ptr{Cuchar}(x + 7)
    return getfield(x, f)
end

function Base.getproperty(x::hipGraphEdgeData, f::Symbol)
    r = Ref{hipGraphEdgeData}(x)
    ptr = Base.unsafe_convert(Ptr{hipGraphEdgeData}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipGraphEdgeData}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipGraphEdgeData, private::Bool=false)
    return (:from_port, :reserved, :to_port, :type, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

function hipInit(flags)
    @ccall librccl.hipInit(flags::Cuint)::hipError_t
end

function hipDriverGetVersion(driverVersion)
    @ccall librccl.hipDriverGetVersion(driverVersion::Ptr{Cint})::hipError_t
end

function hipRuntimeGetVersion(runtimeVersion)
    @ccall librccl.hipRuntimeGetVersion(runtimeVersion::Ptr{Cint})::hipError_t
end

function hipDeviceGet(device, ordinal)
    @ccall librccl.hipDeviceGet(device::Ptr{hipDevice_t}, ordinal::Cint)::hipError_t
end

function hipDeviceComputeCapability(major, minor, device)
    @ccall librccl.hipDeviceComputeCapability(major::Ptr{Cint}, minor::Ptr{Cint},
                                              device::hipDevice_t)::hipError_t
end

function hipDeviceGetName(name, len, device)
    @ccall librccl.hipDeviceGetName(name::Cstring, len::Cint,
                                    device::hipDevice_t)::hipError_t
end

function hipDeviceGetUuid(uuid, device)
    @ccall librccl.hipDeviceGetUuid(uuid::Ptr{hipUUID}, device::hipDevice_t)::hipError_t
end

function hipDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)
    @ccall librccl.hipDeviceGetP2PAttribute(value::Ptr{Cint}, attr::hipDeviceP2PAttr,
                                            srcDevice::Cint, dstDevice::Cint)::hipError_t
end

function hipDeviceGetPCIBusId(pciBusId, len, device)
    @ccall librccl.hipDeviceGetPCIBusId(pciBusId::Cstring, len::Cint,
                                        device::Cint)::hipError_t
end

function hipDeviceGetByPCIBusId(device, pciBusId)
    @ccall librccl.hipDeviceGetByPCIBusId(device::Ptr{Cint}, pciBusId::Cstring)::hipError_t
end

function hipDeviceTotalMem(bytes, device)
    @ccall librccl.hipDeviceTotalMem(bytes::Ptr{Csize_t}, device::hipDevice_t)::hipError_t
end

function hipDeviceSynchronize()
    @ccall librccl.hipDeviceSynchronize()::hipError_t
end

function hipDeviceReset()
    @ccall librccl.hipDeviceReset()::hipError_t
end

function hipSetDevice(deviceId)
    @ccall librccl.hipSetDevice(deviceId::Cint)::hipError_t
end

function hipSetValidDevices(device_arr, len)
    @ccall librccl.hipSetValidDevices(device_arr::Ptr{Cint}, len::Cint)::hipError_t
end

function hipGetDevice(deviceId)
    @ccall librccl.hipGetDevice(deviceId::Ptr{Cint})::hipError_t
end

function hipGetDeviceCount(count)
    @ccall librccl.hipGetDeviceCount(count::Ptr{Cint})::hipError_t
end

function hipDeviceGetAttribute(pi, attr, deviceId)
    @ccall librccl.hipDeviceGetAttribute(pi::Ptr{Cint}, attr::hipDeviceAttribute_t,
                                         deviceId::Cint)::hipError_t
end

function hipDeviceGetDefaultMemPool(mem_pool, device)
    @ccall librccl.hipDeviceGetDefaultMemPool(mem_pool::Ptr{hipMemPool_t},
                                              device::Cint)::hipError_t
end

function hipDeviceSetMemPool(device, mem_pool)
    @ccall librccl.hipDeviceSetMemPool(device::Cint, mem_pool::hipMemPool_t)::hipError_t
end

function hipDeviceGetMemPool(mem_pool, device)
    @ccall librccl.hipDeviceGetMemPool(mem_pool::Ptr{hipMemPool_t},
                                       device::Cint)::hipError_t
end

function hipDeviceGetTexture1DLinearMaxWidth(mem_pool, device)
    @ccall librccl.hipDeviceGetTexture1DLinearMaxWidth(mem_pool::Ptr{hipMemPool_t},
                                                       device::Cint)::hipError_t
end

function hipDeviceSetCacheConfig(cacheConfig)
    @ccall librccl.hipDeviceSetCacheConfig(cacheConfig::hipFuncCache_t)::hipError_t
end

function hipDeviceGetCacheConfig(cacheConfig)
    @ccall librccl.hipDeviceGetCacheConfig(cacheConfig::Ptr{hipFuncCache_t})::hipError_t
end

function hipDeviceGetLimit(pValue, limit)
    @ccall librccl.hipDeviceGetLimit(pValue::Ptr{Csize_t}, limit::hipLimit_t)::hipError_t
end

function hipDeviceSetLimit(limit, value)
    @ccall librccl.hipDeviceSetLimit(limit::hipLimit_t, value::Csize_t)::hipError_t
end

function hipDeviceGetSharedMemConfig(pConfig)
    @ccall librccl.hipDeviceGetSharedMemConfig(pConfig::Ptr{hipSharedMemConfig})::hipError_t
end

function hipGetDeviceFlags(flags)
    @ccall librccl.hipGetDeviceFlags(flags::Ptr{Cuint})::hipError_t
end

function hipDeviceSetSharedMemConfig(config)
    @ccall librccl.hipDeviceSetSharedMemConfig(config::hipSharedMemConfig)::hipError_t
end

function hipSetDeviceFlags(flags)
    @ccall librccl.hipSetDeviceFlags(flags::Cuint)::hipError_t
end

function hipExtGetLinkTypeAndHopCount(device1, device2, linktype, hopcount)
    @ccall librccl.hipExtGetLinkTypeAndHopCount(device1::Cint, device2::Cint,
                                                linktype::Ptr{UInt32},
                                                hopcount::Ptr{UInt32})::hipError_t
end

function hipIpcGetMemHandle(handle, devPtr)
    @ccall librccl.hipIpcGetMemHandle(handle::Ptr{hipIpcMemHandle_t},
                                      devPtr::Ptr{Cvoid})::hipError_t
end

function hipIpcOpenMemHandle(devPtr, handle, flags)
    @ccall librccl.hipIpcOpenMemHandle(devPtr::Ptr{Ptr{Cvoid}}, handle::hipIpcMemHandle_t,
                                       flags::Cuint)::hipError_t
end

function hipIpcCloseMemHandle(devPtr)
    @ccall librccl.hipIpcCloseMemHandle(devPtr::Ptr{Cvoid})::hipError_t
end

function hipIpcGetEventHandle(handle, event)
    @ccall librccl.hipIpcGetEventHandle(handle::Ptr{hipIpcEventHandle_t},
                                        event::hipEvent_t)::hipError_t
end

function hipIpcOpenEventHandle(event, handle)
    @ccall librccl.hipIpcOpenEventHandle(event::Ptr{hipEvent_t},
                                         handle::hipIpcEventHandle_t)::hipError_t
end

function hipFuncSetAttribute(func, attr, value)
    @ccall librccl.hipFuncSetAttribute(func::Ptr{Cvoid}, attr::hipFuncAttribute,
                                       value::Cint)::hipError_t
end

function hipFuncSetCacheConfig(func, config)
    @ccall librccl.hipFuncSetCacheConfig(func::Ptr{Cvoid},
                                         config::hipFuncCache_t)::hipError_t
end

function hipFuncSetSharedMemConfig(func, config)
    @ccall librccl.hipFuncSetSharedMemConfig(func::Ptr{Cvoid},
                                             config::hipSharedMemConfig)::hipError_t
end

function hipGetLastError()
    @ccall librccl.hipGetLastError()::hipError_t
end

function hipExtGetLastError()
    @ccall librccl.hipExtGetLastError()::hipError_t
end

function hipPeekAtLastError()
    @ccall librccl.hipPeekAtLastError()::hipError_t
end

function hipGetErrorName(hip_error)
    @ccall librccl.hipGetErrorName(hip_error::hipError_t)::Cstring
end

function hipGetErrorString(hipError)
    @ccall librccl.hipGetErrorString(hipError::hipError_t)::Cstring
end

function hipDrvGetErrorName(hipError, errorString)
    @ccall librccl.hipDrvGetErrorName(hipError::hipError_t,
                                      errorString::Ptr{Cstring})::hipError_t
end

function hipDrvGetErrorString(hipError, errorString)
    @ccall librccl.hipDrvGetErrorString(hipError::hipError_t,
                                        errorString::Ptr{Cstring})::hipError_t
end

function hipStreamCreate(stream)
    @ccall librccl.hipStreamCreate(stream::HIPStream)::hipError_t
end

function hipStreamCreateWithFlags(stream, flags)
    @ccall librccl.hipStreamCreateWithFlags(stream::HIPStream, flags::Cuint)::hipError_t
end

function hipStreamCreateWithPriority(stream, flags, priority)
    @ccall librccl.hipStreamCreateWithPriority(stream::HIPStream, flags::Cuint,
                                               priority::Cint)::hipError_t
end

function hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority)
    @ccall librccl.hipDeviceGetStreamPriorityRange(leastPriority::Ptr{Cint},
                                                   greatestPriority::Ptr{Cint})::hipError_t
end

function hipStreamDestroy(stream)
    @ccall librccl.hipStreamDestroy(stream::HIPStream)::hipError_t
end

function hipStreamQuery(stream)
    @ccall librccl.hipStreamQuery(stream::HIPStream)::hipError_t
end

function hipStreamSynchronize(stream)
    @ccall librccl.hipStreamSynchronize(stream::HIPStream)::hipError_t
end

function hipStreamWaitEvent(stream, event, flags)
    @ccall librccl.hipStreamWaitEvent(stream::HIPStream, event::hipEvent_t,
                                      flags::Cuint)::hipError_t
end

function hipStreamGetFlags(stream, flags)
    @ccall librccl.hipStreamGetFlags(stream::HIPStream, flags::Ptr{Cuint})::hipError_t
end

function hipStreamGetPriority(stream, priority)
    @ccall librccl.hipStreamGetPriority(stream::HIPStream, priority::Ptr{Cint})::hipError_t
end

function hipStreamGetDevice(stream, device)
    @ccall librccl.hipStreamGetDevice(stream::HIPStream,
                                      device::Ptr{hipDevice_t})::hipError_t
end

function hipExtStreamCreateWithCUMask(stream, cuMaskSize, cuMask)
    @ccall librccl.hipExtStreamCreateWithCUMask(stream::HIPStream, cuMaskSize::UInt32,
                                                cuMask::Ptr{UInt32})::hipError_t
end

function hipExtStreamGetCUMask(stream, cuMaskSize, cuMask)
    @ccall librccl.hipExtStreamGetCUMask(stream::HIPStream, cuMaskSize::UInt32,
                                         cuMask::Ptr{UInt32})::hipError_t
end

# typedef void ( * hipStreamCallback_t ) ( hipStream_t stream , hipError_t status , void * userData )
const hipStreamCallback_t = Ptr{Cvoid}

function hipStreamAddCallback(stream, callback, userData, flags)
    @ccall librccl.hipStreamAddCallback(stream::HIPStream, callback::hipStreamCallback_t,
                                        userData::Ptr{Cvoid}, flags::Cuint)::hipError_t
end

function hipStreamWaitValue32(stream, ptr, value, flags, mask)
    @ccall librccl.hipStreamWaitValue32(stream::HIPStream, ptr::Ptr{Cvoid}, value::UInt32,
                                        flags::Cuint, mask::UInt32)::hipError_t
end

function hipStreamWaitValue64(stream, ptr, value, flags, mask)
    @ccall librccl.hipStreamWaitValue64(stream::HIPStream, ptr::Ptr{Cvoid}, value::UInt64,
                                        flags::Cuint, mask::UInt64)::hipError_t
end

function hipStreamWriteValue32(stream, ptr, value, flags)
    @ccall librccl.hipStreamWriteValue32(stream::HIPStream, ptr::Ptr{Cvoid}, value::UInt32,
                                         flags::Cuint)::hipError_t
end

function hipStreamWriteValue64(stream, ptr, value, flags)
    @ccall librccl.hipStreamWriteValue64(stream::HIPStream, ptr::Ptr{Cvoid}, value::UInt64,
                                         flags::Cuint)::hipError_t
end

function hipStreamBatchMemOp(stream, count, paramArray, flags)
    @ccall librccl.hipStreamBatchMemOp(stream::HIPStream, count::Cuint,
                                       paramArray::Ptr{hipStreamBatchMemOpParams},
                                       flags::Cuint)::hipError_t
end

function hipGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies,
                                   nodeParams)
    @ccall librccl.hipGraphAddBatchMemOpNode(phGraphNode::Ptr{hipGraphNode_t},
                                             hGraph::hipGraph_t,
                                             dependencies::Ptr{hipGraphNode_t},
                                             numDependencies::Csize_t,
                                             nodeParams::Ptr{hipBatchMemOpNodeParams})::hipError_t
end

function hipGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)
    @ccall librccl.hipGraphBatchMemOpNodeGetParams(hNode::hipGraphNode_t,
                                                   nodeParams_out::Ptr{hipBatchMemOpNodeParams})::hipError_t
end

function hipGraphBatchMemOpNodeSetParams(hNode, nodeParams)
    @ccall librccl.hipGraphBatchMemOpNodeSetParams(hNode::hipGraphNode_t,
                                                   nodeParams::Ptr{hipBatchMemOpNodeParams})::hipError_t
end

function hipGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)
    @ccall librccl.hipGraphExecBatchMemOpNodeSetParams(hGraphExec::hipGraphExec_t,
                                                       hNode::hipGraphNode_t,
                                                       nodeParams::Ptr{hipBatchMemOpNodeParams})::hipError_t
end

function hipEventCreateWithFlags(event, flags)
    @ccall librccl.hipEventCreateWithFlags(event::Ptr{hipEvent_t}, flags::Cuint)::hipError_t
end

function hipEventCreate(event)
    @ccall librccl.hipEventCreate(event::Ptr{hipEvent_t})::hipError_t
end

function hipEventRecordWithFlags(event, stream, flags)
    @ccall librccl.hipEventRecordWithFlags(event::hipEvent_t, stream::HIPStream,
                                           flags::Cuint)::hipError_t
end

function hipEventRecord(event, stream)
    @ccall librccl.hipEventRecord(event::hipEvent_t, stream::HIPStream)::hipError_t
end

function hipEventDestroy(event)
    @ccall librccl.hipEventDestroy(event::hipEvent_t)::hipError_t
end

function hipEventSynchronize(event)
    @ccall librccl.hipEventSynchronize(event::hipEvent_t)::hipError_t
end

function hipEventElapsedTime(ms, start, stop)
    @ccall librccl.hipEventElapsedTime(ms::Ptr{Cfloat}, start::hipEvent_t,
                                       stop::hipEvent_t)::hipError_t
end

function hipEventQuery(event)
    @ccall librccl.hipEventQuery(event::hipEvent_t)::hipError_t
end

function hipPointerSetAttribute(value, attribute, ptr)
    @ccall librccl.hipPointerSetAttribute(value::Ptr{Cvoid},
                                          attribute::hipPointer_attribute,
                                          ptr::hipDeviceptr_t)::hipError_t
end

function hipPointerGetAttributes(attributes, ptr)
    @ccall librccl.hipPointerGetAttributes(attributes::Ptr{hipPointerAttribute_t},
                                           ptr::Ptr{Cvoid})::hipError_t
end

function hipPointerGetAttribute(data, attribute, ptr)
    @ccall librccl.hipPointerGetAttribute(data::Ptr{Cvoid}, attribute::hipPointer_attribute,
                                          ptr::hipDeviceptr_t)::hipError_t
end

function hipDrvPointerGetAttributes(numAttributes, attributes, data, ptr)
    @ccall librccl.hipDrvPointerGetAttributes(numAttributes::Cuint,
                                              attributes::Ptr{hipPointer_attribute},
                                              data::Ptr{Ptr{Cvoid}},
                                              ptr::hipDeviceptr_t)::hipError_t
end

function hipImportExternalSemaphore(extSem_out, semHandleDesc)
    @ccall librccl.hipImportExternalSemaphore(extSem_out::Ptr{hipExternalSemaphore_t},
                                              semHandleDesc::Ptr{hipExternalSemaphoreHandleDesc})::hipError_t
end

function hipSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    @ccall librccl.hipSignalExternalSemaphoresAsync(extSemArray::Ptr{hipExternalSemaphore_t},
                                                    paramsArray::Ptr{hipExternalSemaphoreSignalParams},
                                                    numExtSems::Cuint,
                                                    stream::HIPStream)::hipError_t
end

function hipWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    @ccall librccl.hipWaitExternalSemaphoresAsync(extSemArray::Ptr{hipExternalSemaphore_t},
                                                  paramsArray::Ptr{hipExternalSemaphoreWaitParams},
                                                  numExtSems::Cuint,
                                                  stream::HIPStream)::hipError_t
end

function hipDestroyExternalSemaphore(extSem)
    @ccall librccl.hipDestroyExternalSemaphore(extSem::hipExternalSemaphore_t)::hipError_t
end

function hipImportExternalMemory(extMem_out, memHandleDesc)
    @ccall librccl.hipImportExternalMemory(extMem_out::Ptr{hipExternalMemory_t},
                                           memHandleDesc::Ptr{hipExternalMemoryHandleDesc})::hipError_t
end

function hipExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    @ccall librccl.hipExternalMemoryGetMappedBuffer(devPtr::Ptr{Ptr{Cvoid}},
                                                    extMem::hipExternalMemory_t,
                                                    bufferDesc::Ptr{hipExternalMemoryBufferDesc})::hipError_t
end

function hipDestroyExternalMemory(extMem)
    @ccall librccl.hipDestroyExternalMemory(extMem::hipExternalMemory_t)::hipError_t
end

function hipExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    @ccall librccl.hipExternalMemoryGetMappedMipmappedArray(mipmap::Ptr{hipMipmappedArray_t},
                                                            extMem::hipExternalMemory_t,
                                                            mipmapDesc::Ptr{hipExternalMemoryMipmappedArrayDesc})::hipError_t
end

function hipMalloc(ptr, size)
    @ccall librccl.hipMalloc(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::hipError_t
end

function hipExtMallocWithFlags(ptr, sizeBytes, flags)
    @ccall librccl.hipExtMallocWithFlags(ptr::Ptr{Ptr{Cvoid}}, sizeBytes::Csize_t,
                                         flags::Cuint)::hipError_t
end

function hipMallocHost(ptr, size)
    @ccall librccl.hipMallocHost(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::hipError_t
end

function hipMemAllocHost(ptr, size)
    @ccall librccl.hipMemAllocHost(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::hipError_t
end

function hipHostMalloc(ptr, size, flags)
    @ccall librccl.hipHostMalloc(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                 flags::Cuint)::hipError_t
end

function hipMallocManaged(dev_ptr, size, flags)
    @ccall librccl.hipMallocManaged(dev_ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                    flags::Cuint)::hipError_t
end

function hipMemPrefetchAsync(dev_ptr, count, device, stream)
    @ccall librccl.hipMemPrefetchAsync(dev_ptr::Ptr{Cvoid}, count::Csize_t, device::Cint,
                                       stream::HIPStream)::hipError_t
end

function hipMemAdvise(dev_ptr, count, advice, device)
    @ccall librccl.hipMemAdvise(dev_ptr::Ptr{Cvoid}, count::Csize_t,
                                advice::hipMemoryAdvise, device::Cint)::hipError_t
end

function hipMemRangeGetAttribute(data, data_size, attribute, dev_ptr, count)
    @ccall librccl.hipMemRangeGetAttribute(data::Ptr{Cvoid}, data_size::Csize_t,
                                           attribute::hipMemRangeAttribute,
                                           dev_ptr::Ptr{Cvoid}, count::Csize_t)::hipError_t
end

function hipMemRangeGetAttributes(data, data_sizes, attributes, num_attributes, dev_ptr,
                                  count)
    @ccall librccl.hipMemRangeGetAttributes(data::Ptr{Ptr{Cvoid}}, data_sizes::Ptr{Csize_t},
                                            attributes::Ptr{hipMemRangeAttribute},
                                            num_attributes::Csize_t, dev_ptr::Ptr{Cvoid},
                                            count::Csize_t)::hipError_t
end

function hipStreamAttachMemAsync(stream, dev_ptr, length, flags)
    @ccall librccl.hipStreamAttachMemAsync(stream::HIPStream, dev_ptr::Ptr{Cvoid},
                                           length::Csize_t, flags::Cuint)::hipError_t
end

function hipMallocAsync(dev_ptr, size, stream)
    @ccall librccl.hipMallocAsync(dev_ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                  stream::HIPStream)::hipError_t
end

function hipFreeAsync(dev_ptr, stream)
    @ccall librccl.hipFreeAsync(dev_ptr::Ptr{Cvoid}, stream::HIPStream)::hipError_t
end

function hipMemPoolTrimTo(mem_pool, min_bytes_to_hold)
    @ccall librccl.hipMemPoolTrimTo(mem_pool::hipMemPool_t,
                                    min_bytes_to_hold::Csize_t)::hipError_t
end

function hipMemPoolSetAttribute(mem_pool, attr, value)
    @ccall librccl.hipMemPoolSetAttribute(mem_pool::hipMemPool_t, attr::hipMemPoolAttr,
                                          value::Ptr{Cvoid})::hipError_t
end

function hipMemPoolGetAttribute(mem_pool, attr, value)
    @ccall librccl.hipMemPoolGetAttribute(mem_pool::hipMemPool_t, attr::hipMemPoolAttr,
                                          value::Ptr{Cvoid})::hipError_t
end

function hipMemPoolSetAccess(mem_pool, desc_list, count)
    @ccall librccl.hipMemPoolSetAccess(mem_pool::hipMemPool_t,
                                       desc_list::Ptr{hipMemAccessDesc},
                                       count::Csize_t)::hipError_t
end

function hipMemPoolGetAccess(flags, mem_pool, location)
    @ccall librccl.hipMemPoolGetAccess(flags::Ptr{hipMemAccessFlags},
                                       mem_pool::hipMemPool_t,
                                       location::Ptr{hipMemLocation})::hipError_t
end

function hipMemPoolCreate(mem_pool, pool_props)
    @ccall librccl.hipMemPoolCreate(mem_pool::Ptr{hipMemPool_t},
                                    pool_props::Ptr{hipMemPoolProps})::hipError_t
end

function hipMemPoolDestroy(mem_pool)
    @ccall librccl.hipMemPoolDestroy(mem_pool::hipMemPool_t)::hipError_t
end

function hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream)
    @ccall librccl.hipMallocFromPoolAsync(dev_ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                          mem_pool::hipMemPool_t,
                                          stream::HIPStream)::hipError_t
end

function hipMemPoolExportToShareableHandle(shared_handle, mem_pool, handle_type, flags)
    @ccall librccl.hipMemPoolExportToShareableHandle(shared_handle::Ptr{Cvoid},
                                                     mem_pool::hipMemPool_t,
                                                     handle_type::hipMemAllocationHandleType,
                                                     flags::Cuint)::hipError_t
end

function hipMemPoolImportFromShareableHandle(mem_pool, shared_handle, handle_type, flags)
    @ccall librccl.hipMemPoolImportFromShareableHandle(mem_pool::Ptr{hipMemPool_t},
                                                       shared_handle::Ptr{Cvoid},
                                                       handle_type::hipMemAllocationHandleType,
                                                       flags::Cuint)::hipError_t
end

function hipMemPoolExportPointer(export_data, dev_ptr)
    @ccall librccl.hipMemPoolExportPointer(export_data::Ptr{hipMemPoolPtrExportData},
                                           dev_ptr::Ptr{Cvoid})::hipError_t
end

function hipMemPoolImportPointer(dev_ptr, mem_pool, export_data)
    @ccall librccl.hipMemPoolImportPointer(dev_ptr::Ptr{Ptr{Cvoid}}, mem_pool::hipMemPool_t,
                                           export_data::Ptr{hipMemPoolPtrExportData})::hipError_t
end

function hipHostAlloc(ptr, size, flags)
    @ccall librccl.hipHostAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                flags::Cuint)::hipError_t
end

function hipHostGetDevicePointer(devPtr, hstPtr, flags)
    @ccall librccl.hipHostGetDevicePointer(devPtr::Ptr{Ptr{Cvoid}}, hstPtr::Ptr{Cvoid},
                                           flags::Cuint)::hipError_t
end

function hipHostGetFlags(flagsPtr, hostPtr)
    @ccall librccl.hipHostGetFlags(flagsPtr::Ptr{Cuint}, hostPtr::Ptr{Cvoid})::hipError_t
end

function hipHostRegister(hostPtr, sizeBytes, flags)
    @ccall librccl.hipHostRegister(hostPtr::Ptr{Cvoid}, sizeBytes::Csize_t,
                                   flags::Cuint)::hipError_t
end

function hipHostUnregister(hostPtr)
    @ccall librccl.hipHostUnregister(hostPtr::Ptr{Cvoid})::hipError_t
end

function hipMallocPitch(ptr, pitch, width, height)
    @ccall librccl.hipMallocPitch(ptr::Ptr{Ptr{Cvoid}}, pitch::Ptr{Csize_t}, width::Csize_t,
                                  height::Csize_t)::hipError_t
end

function hipMemAllocPitch(dptr, pitch, widthInBytes, height, elementSizeBytes)
    @ccall librccl.hipMemAllocPitch(dptr::Ptr{hipDeviceptr_t}, pitch::Ptr{Csize_t},
                                    widthInBytes::Csize_t, height::Csize_t,
                                    elementSizeBytes::Cuint)::hipError_t
end

function hipFree(ptr)
    @ccall librccl.hipFree(ptr::Ptr{Cvoid})::hipError_t
end

function hipFreeHost(ptr)
    @ccall librccl.hipFreeHost(ptr::Ptr{Cvoid})::hipError_t
end

function hipHostFree(ptr)
    @ccall librccl.hipHostFree(ptr::Ptr{Cvoid})::hipError_t
end

function hipMemcpy(dst, src, sizeBytes, kind)
    @ccall librccl.hipMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, sizeBytes::Csize_t,
                             kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyWithStream(dst, src, sizeBytes, kind, stream)
    @ccall librccl.hipMemcpyWithStream(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, sizeBytes::Csize_t,
                                       kind::hipMemcpyKind, stream::HIPStream)::hipError_t
end

function hipMemcpyHtoD(dst, src, sizeBytes)
    @ccall librccl.hipMemcpyHtoD(dst::hipDeviceptr_t, src::Ptr{Cvoid},
                                 sizeBytes::Csize_t)::hipError_t
end

function hipMemcpyDtoH(dst, src, sizeBytes)
    @ccall librccl.hipMemcpyDtoH(dst::Ptr{Cvoid}, src::hipDeviceptr_t,
                                 sizeBytes::Csize_t)::hipError_t
end

function hipMemcpyDtoD(dst, src, sizeBytes)
    @ccall librccl.hipMemcpyDtoD(dst::hipDeviceptr_t, src::hipDeviceptr_t,
                                 sizeBytes::Csize_t)::hipError_t
end

function hipMemcpyAtoD(dstDevice, srcArray, srcOffset, ByteCount)
    @ccall librccl.hipMemcpyAtoD(dstDevice::hipDeviceptr_t, srcArray::hipArray_t,
                                 srcOffset::Csize_t, ByteCount::Csize_t)::hipError_t
end

function hipMemcpyDtoA(dstArray, dstOffset, srcDevice, ByteCount)
    @ccall librccl.hipMemcpyDtoA(dstArray::hipArray_t, dstOffset::Csize_t,
                                 srcDevice::hipDeviceptr_t, ByteCount::Csize_t)::hipError_t
end

function hipMemcpyAtoA(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    @ccall librccl.hipMemcpyAtoA(dstArray::hipArray_t, dstOffset::Csize_t,
                                 srcArray::hipArray_t, srcOffset::Csize_t,
                                 ByteCount::Csize_t)::hipError_t
end

function hipMemcpyHtoDAsync(dst, src, sizeBytes, stream)
    @ccall librccl.hipMemcpyHtoDAsync(dst::hipDeviceptr_t, src::Ptr{Cvoid},
                                      sizeBytes::Csize_t, stream::HIPStream)::hipError_t
end

function hipMemcpyDtoHAsync(dst, src, sizeBytes, stream)
    @ccall librccl.hipMemcpyDtoHAsync(dst::Ptr{Cvoid}, src::hipDeviceptr_t,
                                      sizeBytes::Csize_t, stream::HIPStream)::hipError_t
end

function hipMemcpyDtoDAsync(dst, src, sizeBytes, stream)
    @ccall librccl.hipMemcpyDtoDAsync(dst::hipDeviceptr_t, src::hipDeviceptr_t,
                                      sizeBytes::Csize_t, stream::HIPStream)::hipError_t
end

function hipMemcpyAtoHAsync(dstHost, srcArray, srcOffset, ByteCount, stream)
    @ccall librccl.hipMemcpyAtoHAsync(dstHost::Ptr{Cvoid}, srcArray::hipArray_t,
                                      srcOffset::Csize_t, ByteCount::Csize_t,
                                      stream::HIPStream)::hipError_t
end

function hipMemcpyHtoAAsync(dstArray, dstOffset, srcHost, ByteCount, stream)
    @ccall librccl.hipMemcpyHtoAAsync(dstArray::hipArray_t, dstOffset::Csize_t,
                                      srcHost::Ptr{Cvoid}, ByteCount::Csize_t,
                                      stream::HIPStream)::hipError_t
end

function hipModuleGetGlobal(dptr, bytes, hmod, name)
    @ccall librccl.hipModuleGetGlobal(dptr::Ptr{hipDeviceptr_t}, bytes::Ptr{Csize_t},
                                      hmod::hipModule_t, name::Cstring)::hipError_t
end

function hipGetSymbolAddress(devPtr, symbol)
    @ccall librccl.hipGetSymbolAddress(devPtr::Ptr{Ptr{Cvoid}},
                                       symbol::Ptr{Cvoid})::hipError_t
end

function hipGetSymbolSize(size, symbol)
    @ccall librccl.hipGetSymbolSize(size::Ptr{Csize_t}, symbol::Ptr{Cvoid})::hipError_t
end

function hipGetProcAddress(symbol, pfn, hipVersion, flags, symbolStatus)
    @ccall librccl.hipGetProcAddress(symbol::Cstring, pfn::Ptr{Ptr{Cvoid}},
                                     hipVersion::Cint, flags::UInt64,
                                     symbolStatus::Ptr{hipDriverProcAddressQueryResult})::hipError_t
end

function hipMemcpyToSymbol(symbol, src, sizeBytes, offset, kind)
    @ccall librccl.hipMemcpyToSymbol(symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                     sizeBytes::Csize_t, offset::Csize_t,
                                     kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind, stream)
    @ccall librccl.hipMemcpyToSymbolAsync(symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                          sizeBytes::Csize_t, offset::Csize_t,
                                          kind::hipMemcpyKind,
                                          stream::HIPStream)::hipError_t
end

function hipMemcpyFromSymbol(dst, symbol, sizeBytes, offset, kind)
    @ccall librccl.hipMemcpyFromSymbol(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid},
                                       sizeBytes::Csize_t, offset::Csize_t,
                                       kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind, stream)
    @ccall librccl.hipMemcpyFromSymbolAsync(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid},
                                            sizeBytes::Csize_t, offset::Csize_t,
                                            kind::hipMemcpyKind,
                                            stream::HIPStream)::hipError_t
end

function hipMemcpyAsync(dst, src, sizeBytes, kind, stream)
    @ccall librccl.hipMemcpyAsync(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, sizeBytes::Csize_t,
                                  kind::hipMemcpyKind, stream::HIPStream)::hipError_t
end

function hipMemset(dst, value, sizeBytes)
    @ccall librccl.hipMemset(dst::Ptr{Cvoid}, value::Cint, sizeBytes::Csize_t)::hipError_t
end

function hipMemsetD8(dest, value, count)
    @ccall librccl.hipMemsetD8(dest::hipDeviceptr_t, value::Cuchar,
                               count::Csize_t)::hipError_t
end

function hipMemsetD8Async(dest, value, count, stream)
    @ccall librccl.hipMemsetD8Async(dest::hipDeviceptr_t, value::Cuchar, count::Csize_t,
                                    stream::HIPStream)::hipError_t
end

function hipMemsetD16(dest, value, count)
    @ccall librccl.hipMemsetD16(dest::hipDeviceptr_t, value::Cushort,
                                count::Csize_t)::hipError_t
end

function hipMemsetD16Async(dest, value, count, stream)
    @ccall librccl.hipMemsetD16Async(dest::hipDeviceptr_t, value::Cushort, count::Csize_t,
                                     stream::HIPStream)::hipError_t
end

function hipMemsetD32(dest, value, count)
    @ccall librccl.hipMemsetD32(dest::hipDeviceptr_t, value::Cint,
                                count::Csize_t)::hipError_t
end

function hipMemsetAsync(dst, value, sizeBytes, stream)
    @ccall librccl.hipMemsetAsync(dst::Ptr{Cvoid}, value::Cint, sizeBytes::Csize_t,
                                  stream::HIPStream)::hipError_t
end

function hipMemsetD32Async(dst, value, count, stream)
    @ccall librccl.hipMemsetD32Async(dst::hipDeviceptr_t, value::Cint, count::Csize_t,
                                     stream::HIPStream)::hipError_t
end

function hipMemset2D(dst, pitch, value, width, height)
    @ccall librccl.hipMemset2D(dst::Ptr{Cvoid}, pitch::Csize_t, value::Cint, width::Csize_t,
                               height::Csize_t)::hipError_t
end

function hipMemset2DAsync(dst, pitch, value, width, height, stream)
    @ccall librccl.hipMemset2DAsync(dst::Ptr{Cvoid}, pitch::Csize_t, value::Cint,
                                    width::Csize_t, height::Csize_t,
                                    stream::HIPStream)::hipError_t
end

function hipMemset3D(pitchedDevPtr, value, extent)
    @ccall librccl.hipMemset3D(pitchedDevPtr::hipPitchedPtr, value::Cint,
                               extent::hipExtent)::hipError_t
end

function hipMemset3DAsync(pitchedDevPtr, value, extent, stream)
    @ccall librccl.hipMemset3DAsync(pitchedDevPtr::hipPitchedPtr, value::Cint,
                                    extent::hipExtent, stream::HIPStream)::hipError_t
end

function hipMemGetInfo(free, total)
    @ccall librccl.hipMemGetInfo(free::Ptr{Csize_t}, total::Ptr{Csize_t})::hipError_t
end

function hipMemPtrGetInfo(ptr, size)
    @ccall librccl.hipMemPtrGetInfo(ptr::Ptr{Cvoid}, size::Ptr{Csize_t})::hipError_t
end

function hipMallocArray(array, desc, width, height, flags)
    @ccall librccl.hipMallocArray(array::Ptr{hipArray_t}, desc::Ptr{hipChannelFormatDesc},
                                  width::Csize_t, height::Csize_t, flags::Cuint)::hipError_t
end

function hipArrayCreate(pHandle, pAllocateArray)
    @ccall librccl.hipArrayCreate(pHandle::Ptr{hipArray_t},
                                  pAllocateArray::Ptr{HIP_ARRAY_DESCRIPTOR})::hipError_t
end

function hipArrayDestroy(array)
    @ccall librccl.hipArrayDestroy(array::hipArray_t)::hipError_t
end

function hipArray3DCreate(array, pAllocateArray)
    @ccall librccl.hipArray3DCreate(array::Ptr{hipArray_t},
                                    pAllocateArray::Ptr{HIP_ARRAY3D_DESCRIPTOR})::hipError_t
end

function hipMalloc3D(pitchedDevPtr, extent)
    @ccall librccl.hipMalloc3D(pitchedDevPtr::Ptr{hipPitchedPtr},
                               extent::hipExtent)::hipError_t
end

function hipFreeArray(array)
    @ccall librccl.hipFreeArray(array::hipArray_t)::hipError_t
end

function hipMalloc3DArray(array, desc, extent, flags)
    @ccall librccl.hipMalloc3DArray(array::Ptr{hipArray_t}, desc::Ptr{hipChannelFormatDesc},
                                    extent::hipExtent, flags::Cuint)::hipError_t
end

function hipArrayGetInfo(desc, extent, flags, array)
    @ccall librccl.hipArrayGetInfo(desc::Ptr{hipChannelFormatDesc}, extent::Ptr{hipExtent},
                                   flags::Ptr{Cuint}, array::hipArray_t)::hipError_t
end

function hipArrayGetDescriptor(pArrayDescriptor, array)
    @ccall librccl.hipArrayGetDescriptor(pArrayDescriptor::Ptr{HIP_ARRAY_DESCRIPTOR},
                                         array::hipArray_t)::hipError_t
end

function hipArray3DGetDescriptor(pArrayDescriptor, array)
    @ccall librccl.hipArray3DGetDescriptor(pArrayDescriptor::Ptr{HIP_ARRAY3D_DESCRIPTOR},
                                           array::hipArray_t)::hipError_t
end

function hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
    @ccall librccl.hipMemcpy2D(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid},
                               spitch::Csize_t, width::Csize_t, height::Csize_t,
                               kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyParam2D(pCopy)
    @ccall librccl.hipMemcpyParam2D(pCopy::Ptr{hip_Memcpy2D})::hipError_t
end

function hipMemcpyParam2DAsync(pCopy, stream)
    @ccall librccl.hipMemcpyParam2DAsync(pCopy::Ptr{hip_Memcpy2D},
                                         stream::HIPStream)::hipError_t
end

function hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
    @ccall librccl.hipMemcpy2DAsync(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid},
                                    spitch::Csize_t, width::Csize_t, height::Csize_t,
                                    kind::hipMemcpyKind, stream::HIPStream)::hipError_t
end

function hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)
    @ccall librccl.hipMemcpy2DToArray(dst::hipArray_t, wOffset::Csize_t, hOffset::Csize_t,
                                      src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t,
                                      height::Csize_t, kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind,
                                 stream)
    @ccall librccl.hipMemcpy2DToArrayAsync(dst::hipArray_t, wOffset::Csize_t,
                                           hOffset::Csize_t, src::Ptr{Cvoid},
                                           spitch::Csize_t, width::Csize_t, height::Csize_t,
                                           kind::hipMemcpyKind,
                                           stream::HIPStream)::hipError_t
end

function hipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc,
                                 width, height, kind)
    @ccall librccl.hipMemcpy2DArrayToArray(dst::hipArray_t, wOffsetDst::Csize_t,
                                           hOffsetDst::Csize_t, src::hipArray_const_t,
                                           wOffsetSrc::Csize_t, hOffsetSrc::Csize_t,
                                           width::Csize_t, height::Csize_t,
                                           kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyToArray(dst, wOffset, hOffset, src, count, kind)
    @ccall librccl.hipMemcpyToArray(dst::hipArray_t, wOffset::Csize_t, hOffset::Csize_t,
                                    src::Ptr{Cvoid}, count::Csize_t,
                                    kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyFromArray(dst, srcArray, wOffset, hOffset, count, kind)
    @ccall librccl.hipMemcpyFromArray(dst::Ptr{Cvoid}, srcArray::hipArray_const_t,
                                      wOffset::Csize_t, hOffset::Csize_t, count::Csize_t,
                                      kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    @ccall librccl.hipMemcpy2DFromArray(dst::Ptr{Cvoid}, dpitch::Csize_t,
                                        src::hipArray_const_t, wOffset::Csize_t,
                                        hOffset::Csize_t, width::Csize_t, height::Csize_t,
                                        kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind,
                                   stream)
    @ccall librccl.hipMemcpy2DFromArrayAsync(dst::Ptr{Cvoid}, dpitch::Csize_t,
                                             src::hipArray_const_t, wOffset::Csize_t,
                                             hOffset::Csize_t, width::Csize_t,
                                             height::Csize_t, kind::hipMemcpyKind,
                                             stream::HIPStream)::hipError_t
end

function hipMemcpyAtoH(dst, srcArray, srcOffset, count)
    @ccall librccl.hipMemcpyAtoH(dst::Ptr{Cvoid}, srcArray::hipArray_t, srcOffset::Csize_t,
                                 count::Csize_t)::hipError_t
end

function hipMemcpyHtoA(dstArray, dstOffset, srcHost, count)
    @ccall librccl.hipMemcpyHtoA(dstArray::hipArray_t, dstOffset::Csize_t,
                                 srcHost::Ptr{Cvoid}, count::Csize_t)::hipError_t
end

function hipMemcpy3D(p)
    @ccall librccl.hipMemcpy3D(p::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipMemcpy3DAsync(p, stream)
    @ccall librccl.hipMemcpy3DAsync(p::Ptr{hipMemcpy3DParms}, stream::HIPStream)::hipError_t
end

function hipDrvMemcpy3D(pCopy)
    @ccall librccl.hipDrvMemcpy3D(pCopy::Ptr{HIP_MEMCPY3D})::hipError_t
end

function hipDrvMemcpy3DAsync(pCopy, stream)
    @ccall librccl.hipDrvMemcpy3DAsync(pCopy::Ptr{HIP_MEMCPY3D},
                                       stream::HIPStream)::hipError_t
end

function hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId)
    @ccall librccl.hipDeviceCanAccessPeer(canAccessPeer::Ptr{Cint}, deviceId::Cint,
                                          peerDeviceId::Cint)::hipError_t
end

function hipDeviceEnablePeerAccess(peerDeviceId, flags)
    @ccall librccl.hipDeviceEnablePeerAccess(peerDeviceId::Cint, flags::Cuint)::hipError_t
end

function hipDeviceDisablePeerAccess(peerDeviceId)
    @ccall librccl.hipDeviceDisablePeerAccess(peerDeviceId::Cint)::hipError_t
end

function hipMemGetAddressRange(pbase, psize, dptr)
    @ccall librccl.hipMemGetAddressRange(pbase::Ptr{hipDeviceptr_t}, psize::Ptr{Csize_t},
                                         dptr::hipDeviceptr_t)::hipError_t
end

function hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes)
    @ccall librccl.hipMemcpyPeer(dst::Ptr{Cvoid}, dstDeviceId::Cint, src::Ptr{Cvoid},
                                 srcDeviceId::Cint, sizeBytes::Csize_t)::hipError_t
end

function hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDevice, sizeBytes, stream)
    @ccall librccl.hipMemcpyPeerAsync(dst::Ptr{Cvoid}, dstDeviceId::Cint, src::Ptr{Cvoid},
                                      srcDevice::Cint, sizeBytes::Csize_t,
                                      stream::HIPStream)::hipError_t
end

function hipCtxCreate(ctx, flags, device)
    @ccall librccl.hipCtxCreate(ctx::Ptr{hipCtx_t}, flags::Cuint,
                                device::hipDevice_t)::hipError_t
end

function hipCtxDestroy(ctx)
    @ccall librccl.hipCtxDestroy(ctx::hipCtx_t)::hipError_t
end

function hipCtxPopCurrent(ctx)
    @ccall librccl.hipCtxPopCurrent(ctx::Ptr{hipCtx_t})::hipError_t
end

function hipCtxPushCurrent(ctx)
    @ccall librccl.hipCtxPushCurrent(ctx::hipCtx_t)::hipError_t
end

function hipCtxSetCurrent(ctx)
    @ccall librccl.hipCtxSetCurrent(ctx::hipCtx_t)::hipError_t
end

function hipCtxGetCurrent(ctx)
    @ccall librccl.hipCtxGetCurrent(ctx::Ptr{hipCtx_t})::hipError_t
end

function hipCtxGetDevice(device)
    @ccall librccl.hipCtxGetDevice(device::Ptr{hipDevice_t})::hipError_t
end

function hipCtxGetApiVersion(ctx, apiVersion)
    @ccall librccl.hipCtxGetApiVersion(ctx::hipCtx_t, apiVersion::Ptr{Cint})::hipError_t
end

function hipCtxGetCacheConfig(cacheConfig)
    @ccall librccl.hipCtxGetCacheConfig(cacheConfig::Ptr{hipFuncCache_t})::hipError_t
end

function hipCtxSetCacheConfig(cacheConfig)
    @ccall librccl.hipCtxSetCacheConfig(cacheConfig::hipFuncCache_t)::hipError_t
end

function hipCtxSetSharedMemConfig(config)
    @ccall librccl.hipCtxSetSharedMemConfig(config::hipSharedMemConfig)::hipError_t
end

function hipCtxGetSharedMemConfig(pConfig)
    @ccall librccl.hipCtxGetSharedMemConfig(pConfig::Ptr{hipSharedMemConfig})::hipError_t
end

function hipCtxSynchronize()
    @ccall librccl.hipCtxSynchronize()::hipError_t
end

function hipCtxGetFlags(flags)
    @ccall librccl.hipCtxGetFlags(flags::Ptr{Cuint})::hipError_t
end

function hipCtxEnablePeerAccess(peerCtx, flags)
    @ccall librccl.hipCtxEnablePeerAccess(peerCtx::hipCtx_t, flags::Cuint)::hipError_t
end

function hipCtxDisablePeerAccess(peerCtx)
    @ccall librccl.hipCtxDisablePeerAccess(peerCtx::hipCtx_t)::hipError_t
end

function hipDevicePrimaryCtxGetState(dev, flags, active)
    @ccall librccl.hipDevicePrimaryCtxGetState(dev::hipDevice_t, flags::Ptr{Cuint},
                                               active::Ptr{Cint})::hipError_t
end

function hipDevicePrimaryCtxRelease(dev)
    @ccall librccl.hipDevicePrimaryCtxRelease(dev::hipDevice_t)::hipError_t
end

function hipDevicePrimaryCtxRetain(pctx, dev)
    @ccall librccl.hipDevicePrimaryCtxRetain(pctx::Ptr{hipCtx_t},
                                             dev::hipDevice_t)::hipError_t
end

function hipDevicePrimaryCtxReset(dev)
    @ccall librccl.hipDevicePrimaryCtxReset(dev::hipDevice_t)::hipError_t
end

function hipDevicePrimaryCtxSetFlags(dev, flags)
    @ccall librccl.hipDevicePrimaryCtxSetFlags(dev::hipDevice_t, flags::Cuint)::hipError_t
end

function hipModuleLoad(_module, fname)
    @ccall librccl.hipModuleLoad(_module::Ptr{hipModule_t}, fname::Cstring)::hipError_t
end

function hipModuleUnload(_module)
    @ccall librccl.hipModuleUnload(_module::hipModule_t)::hipError_t
end

function hipModuleGetFunction(_function, _module, kname)
    @ccall librccl.hipModuleGetFunction(_function::Ptr{hipFunction_t}, _module::hipModule_t,
                                        kname::Cstring)::hipError_t
end

function hipFuncGetAttributes(attr, func)
    @ccall librccl.hipFuncGetAttributes(attr::Ptr{hipFuncAttributes},
                                        func::Ptr{Cvoid})::hipError_t
end

function hipFuncGetAttribute(value, attrib, hfunc)
    @ccall librccl.hipFuncGetAttribute(value::Ptr{Cint}, attrib::hipFunction_attribute,
                                       hfunc::hipFunction_t)::hipError_t
end

function hipGetFuncBySymbol(functionPtr, symbolPtr)
    @ccall librccl.hipGetFuncBySymbol(functionPtr::Ptr{hipFunction_t},
                                      symbolPtr::Ptr{Cvoid})::hipError_t
end

function hipModuleGetTexRef(texRef, hmod, name)
    @ccall librccl.hipModuleGetTexRef(texRef::Ptr{Ptr{textureReference}}, hmod::hipModule_t,
                                      name::Cstring)::hipError_t
end

function hipModuleLoadData(_module, image)
    @ccall librccl.hipModuleLoadData(_module::Ptr{hipModule_t},
                                     image::Ptr{Cvoid})::hipError_t
end

function hipModuleLoadDataEx(_module, image, numOptions, options, optionValues)
    @ccall librccl.hipModuleLoadDataEx(_module::Ptr{hipModule_t}, image::Ptr{Cvoid},
                                       numOptions::Cuint, options::Ptr{hipJitOption},
                                       optionValues::Ptr{Ptr{Cvoid}})::hipError_t
end

function hipLinkAddData(state, type, data, size, name, numOptions, options, optionValues)
    @ccall librccl.hipLinkAddData(state::hipLinkState_t, type::hipJitInputType,
                                  data::Ptr{Cvoid}, size::Csize_t, name::Cstring,
                                  numOptions::Cuint, options::Ptr{hipJitOption},
                                  optionValues::Ptr{Ptr{Cvoid}})::hipError_t
end

function hipLinkAddFile(state, type, path, numOptions, options, optionValues)
    @ccall librccl.hipLinkAddFile(state::hipLinkState_t, type::hipJitInputType,
                                  path::Cstring, numOptions::Cuint,
                                  options::Ptr{hipJitOption},
                                  optionValues::Ptr{Ptr{Cvoid}})::hipError_t
end

function hipLinkComplete(state, hipBinOut, sizeOut)
    @ccall librccl.hipLinkComplete(state::hipLinkState_t, hipBinOut::Ptr{Ptr{Cvoid}},
                                   sizeOut::Ptr{Csize_t})::hipError_t
end

function hipLinkCreate(numOptions, options, optionValues, stateOut)
    @ccall librccl.hipLinkCreate(numOptions::Cuint, options::Ptr{hipJitOption},
                                 optionValues::Ptr{Ptr{Cvoid}},
                                 stateOut::Ptr{hipLinkState_t})::hipError_t
end

function hipLinkDestroy(state)
    @ccall librccl.hipLinkDestroy(state::hipLinkState_t)::hipError_t
end

function hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                               blockDimZ, sharedMemBytes, stream, kernelParams, extra)
    @ccall librccl.hipModuleLaunchKernel(f::hipFunction_t, gridDimX::Cuint, gridDimY::Cuint,
                                         gridDimZ::Cuint, blockDimX::Cuint,
                                         blockDimY::Cuint, blockDimZ::Cuint,
                                         sharedMemBytes::Cuint, stream::HIPStream,
                                         kernelParams::Ptr{Ptr{Cvoid}},
                                         extra::Ptr{Ptr{Cvoid}})::hipError_t
end

function hipModuleLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                          blockDimY, blockDimZ, sharedMemBytes, stream,
                                          kernelParams)
    @ccall librccl.hipModuleLaunchCooperativeKernel(f::hipFunction_t, gridDimX::Cuint,
                                                    gridDimY::Cuint, gridDimZ::Cuint,
                                                    blockDimX::Cuint, blockDimY::Cuint,
                                                    blockDimZ::Cuint, sharedMemBytes::Cuint,
                                                    stream::HIPStream,
                                                    kernelParams::Ptr{Ptr{Cvoid}})::hipError_t
end

function hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    @ccall librccl.hipModuleLaunchCooperativeKernelMultiDevice(launchParamsList::Ptr{hipFunctionLaunchParams},
                                                               numDevices::Cuint,
                                                               flags::Cuint)::hipError_t
end

function hipLaunchCooperativeKernel(f, gridDim, blockDimX, kernelParams, sharedMemBytes,
                                    stream)
    @ccall librccl.hipLaunchCooperativeKernel(f::Ptr{Cvoid}, gridDim::dim3, blockDimX::dim3,
                                              kernelParams::Ptr{Ptr{Cvoid}},
                                              sharedMemBytes::Cuint,
                                              stream::HIPStream)::hipError_t
end

function hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    @ccall librccl.hipLaunchCooperativeKernelMultiDevice(launchParamsList::Ptr{hipLaunchParams},
                                                         numDevices::Cint,
                                                         flags::Cuint)::hipError_t
end

function hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags)
    @ccall librccl.hipExtLaunchMultiKernelMultiDevice(launchParamsList::Ptr{hipLaunchParams},
                                                      numDevices::Cint,
                                                      flags::Cuint)::hipError_t
end

function hipModuleOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk,
                                                 blockSizeLimit)
    @ccall librccl.hipModuleOccupancyMaxPotentialBlockSize(gridSize::Ptr{Cint},
                                                           blockSize::Ptr{Cint},
                                                           f::hipFunction_t,
                                                           dynSharedMemPerBlk::Csize_t,
                                                           blockSizeLimit::Cint)::hipError_t
end

function hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, f,
                                                          dynSharedMemPerBlk,
                                                          blockSizeLimit, flags)
    @ccall librccl.hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize::Ptr{Cint},
                                                                    blockSize::Ptr{Cint},
                                                                    f::hipFunction_t,
                                                                    dynSharedMemPerBlk::Csize_t,
                                                                    blockSizeLimit::Cint,
                                                                    flags::Cuint)::hipError_t
end

function hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize,
                                                            dynSharedMemPerBlk)
    @ccall librccl.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Ptr{Cint},
                                                                      f::hipFunction_t,
                                                                      blockSize::Cint,
                                                                      dynSharedMemPerBlk::Csize_t)::hipError_t
end

function hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f,
                                                                     blockSize,
                                                                     dynSharedMemPerBlk,
                                                                     flags)
    @ccall librccl.hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Ptr{Cint},
                                                                               f::hipFunction_t,
                                                                               blockSize::Cint,
                                                                               dynSharedMemPerBlk::Csize_t,
                                                                               flags::Cuint)::hipError_t
end

function hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize,
                                                      dynSharedMemPerBlk)
    @ccall librccl.hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Ptr{Cint},
                                                                f::Ptr{Cvoid},
                                                                blockSize::Cint,
                                                                dynSharedMemPerBlk::Csize_t)::hipError_t
end

function hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, f, blockSize,
                                                               dynSharedMemPerBlk, flags)
    @ccall librccl.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Ptr{Cint},
                                                                         f::Ptr{Cvoid},
                                                                         blockSize::Cint,
                                                                         dynSharedMemPerBlk::Csize_t,
                                                                         flags::Cuint)::hipError_t
end

function hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, dynSharedMemPerBlk,
                                           blockSizeLimit)
    @ccall librccl.hipOccupancyMaxPotentialBlockSize(gridSize::Ptr{Cint},
                                                     blockSize::Ptr{Cint}, f::Ptr{Cvoid},
                                                     dynSharedMemPerBlk::Csize_t,
                                                     blockSizeLimit::Cint)::hipError_t
end

# no prototype is found for this function at hip_runtime_api.h:6243:12, please use with caution
function hipProfilerStart()
    @ccall librccl.hipProfilerStart()::hipError_t
end

# no prototype is found for this function at hip_runtime_api.h:6251:12, please use with caution
function hipProfilerStop()
    @ccall librccl.hipProfilerStop()::hipError_t
end

function hipConfigureCall(gridDim, blockDim, sharedMem, stream)
    @ccall librccl.hipConfigureCall(gridDim::dim3, blockDim::dim3, sharedMem::Csize_t,
                                    stream::HIPStream)::hipError_t
end

function hipSetupArgument(arg, size, offset)
    @ccall librccl.hipSetupArgument(arg::Ptr{Cvoid}, size::Csize_t,
                                    offset::Csize_t)::hipError_t
end

function hipLaunchByPtr(func)
    @ccall librccl.hipLaunchByPtr(func::Ptr{Cvoid})::hipError_t
end

function __hipPushCallConfiguration(gridDim, blockDim, sharedMem, stream)
    @ccall librccl.__hipPushCallConfiguration(gridDim::dim3, blockDim::dim3,
                                              sharedMem::Csize_t,
                                              stream::HIPStream)::hipError_t
end

function __hipPopCallConfiguration(gridDim, blockDim, sharedMem, stream)
    @ccall librccl.__hipPopCallConfiguration(gridDim::Ptr{dim3}, blockDim::Ptr{dim3},
                                             sharedMem::Ptr{Csize_t},
                                             stream::HIPStream)::hipError_t
end

function hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                         stream)
    @ccall librccl.hipLaunchKernel(function_address::Ptr{Cvoid}, numBlocks::dim3,
                                   dimBlocks::dim3, args::Ptr{Ptr{Cvoid}},
                                   sharedMemBytes::Csize_t, stream::HIPStream)::hipError_t
end

function hipLaunchHostFunc(stream, fn, userData)
    @ccall librccl.hipLaunchHostFunc(stream::HIPStream, fn::hipHostFn_t,
                                     userData::Ptr{Cvoid})::hipError_t
end

function hipDrvMemcpy2DUnaligned(pCopy)
    @ccall librccl.hipDrvMemcpy2DUnaligned(pCopy::Ptr{hip_Memcpy2D})::hipError_t
end

function hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                            stream, startEvent, stopEvent, flags)
    @ccall librccl.hipExtLaunchKernel(function_address::Ptr{Cvoid}, numBlocks::dim3,
                                      dimBlocks::dim3, args::Ptr{Ptr{Cvoid}},
                                      sharedMemBytes::Csize_t, stream::HIPStream,
                                      startEvent::hipEvent_t, stopEvent::hipEvent_t,
                                      flags::Cint)::hipError_t
end

function hipCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    @ccall librccl.hipCreateTextureObject(pTexObject::Ptr{hipTextureObject_t},
                                          pResDesc::Ptr{hipResourceDesc},
                                          pTexDesc::Ptr{hipTextureDesc},
                                          pResViewDesc::Ptr{hipResourceViewDesc})::hipError_t
end

function hipDestroyTextureObject(textureObject)
    @ccall librccl.hipDestroyTextureObject(textureObject::hipTextureObject_t)::hipError_t
end

function hipGetChannelDesc(desc, array)
    @ccall librccl.hipGetChannelDesc(desc::Ptr{hipChannelFormatDesc},
                                     array::hipArray_const_t)::hipError_t
end

function hipGetTextureObjectResourceDesc(pResDesc, textureObject)
    @ccall librccl.hipGetTextureObjectResourceDesc(pResDesc::Ptr{hipResourceDesc},
                                                   textureObject::hipTextureObject_t)::hipError_t
end

function hipGetTextureObjectResourceViewDesc(pResViewDesc, textureObject)
    @ccall librccl.hipGetTextureObjectResourceViewDesc(pResViewDesc::Ptr{hipResourceViewDesc},
                                                       textureObject::hipTextureObject_t)::hipError_t
end

function hipGetTextureObjectTextureDesc(pTexDesc, textureObject)
    @ccall librccl.hipGetTextureObjectTextureDesc(pTexDesc::Ptr{hipTextureDesc},
                                                  textureObject::hipTextureObject_t)::hipError_t
end

function hipTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    @ccall librccl.hipTexObjectCreate(pTexObject::Ptr{hipTextureObject_t},
                                      pResDesc::Ptr{HIP_RESOURCE_DESC},
                                      pTexDesc::Ptr{HIP_TEXTURE_DESC},
                                      pResViewDesc::Ptr{HIP_RESOURCE_VIEW_DESC})::hipError_t
end

function hipTexObjectDestroy(texObject)
    @ccall librccl.hipTexObjectDestroy(texObject::hipTextureObject_t)::hipError_t
end

function hipTexObjectGetResourceDesc(pResDesc, texObject)
    @ccall librccl.hipTexObjectGetResourceDesc(pResDesc::Ptr{HIP_RESOURCE_DESC},
                                               texObject::hipTextureObject_t)::hipError_t
end

function hipTexObjectGetResourceViewDesc(pResViewDesc, texObject)
    @ccall librccl.hipTexObjectGetResourceViewDesc(pResViewDesc::Ptr{HIP_RESOURCE_VIEW_DESC},
                                                   texObject::hipTextureObject_t)::hipError_t
end

function hipTexObjectGetTextureDesc(pTexDesc, texObject)
    @ccall librccl.hipTexObjectGetTextureDesc(pTexDesc::Ptr{HIP_TEXTURE_DESC},
                                              texObject::hipTextureObject_t)::hipError_t
end

function hipMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)
    @ccall librccl.hipMallocMipmappedArray(mipmappedArray::Ptr{hipMipmappedArray_t},
                                           desc::Ptr{hipChannelFormatDesc},
                                           extent::hipExtent, numLevels::Cuint,
                                           flags::Cuint)::hipError_t
end

function hipFreeMipmappedArray(mipmappedArray)
    @ccall librccl.hipFreeMipmappedArray(mipmappedArray::hipMipmappedArray_t)::hipError_t
end

function hipGetMipmappedArrayLevel(levelArray, mipmappedArray, level)
    @ccall librccl.hipGetMipmappedArrayLevel(levelArray::Ptr{hipArray_t},
                                             mipmappedArray::hipMipmappedArray_const_t,
                                             level::Cuint)::hipError_t
end

function hipMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    @ccall librccl.hipMipmappedArrayCreate(pHandle::Ptr{hipMipmappedArray_t},
                                           pMipmappedArrayDesc::Ptr{HIP_ARRAY3D_DESCRIPTOR},
                                           numMipmapLevels::Cuint)::hipError_t
end

function hipMipmappedArrayDestroy(hMipmappedArray)
    @ccall librccl.hipMipmappedArrayDestroy(hMipmappedArray::hipMipmappedArray_t)::hipError_t
end

function hipMipmappedArrayGetLevel(pLevelArray, hMipMappedArray, level)
    @ccall librccl.hipMipmappedArrayGetLevel(pLevelArray::Ptr{hipArray_t},
                                             hMipMappedArray::hipMipmappedArray_t,
                                             level::Cuint)::hipError_t
end

function hipBindTextureToMipmappedArray(tex, mipmappedArray, desc)
    @ccall librccl.hipBindTextureToMipmappedArray(tex::Ptr{textureReference},
                                                  mipmappedArray::hipMipmappedArray_const_t,
                                                  desc::Ptr{hipChannelFormatDesc})::hipError_t
end

function hipGetTextureReference(texref, symbol)
    @ccall librccl.hipGetTextureReference(texref::Ptr{Ptr{textureReference}},
                                          symbol::Ptr{Cvoid})::hipError_t
end

function hipTexRefGetBorderColor(pBorderColor, texRef)
    @ccall librccl.hipTexRefGetBorderColor(pBorderColor::Ptr{Cfloat},
                                           texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetArray(pArray, texRef)
    @ccall librccl.hipTexRefGetArray(pArray::Ptr{hipArray_t},
                                     texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefSetAddressMode(texRef, dim, am)
    @ccall librccl.hipTexRefSetAddressMode(texRef::Ptr{textureReference}, dim::Cint,
                                           am::hipTextureAddressMode)::hipError_t
end

function hipTexRefSetArray(tex, array, flags)
    @ccall librccl.hipTexRefSetArray(tex::Ptr{textureReference}, array::hipArray_const_t,
                                     flags::Cuint)::hipError_t
end

function hipTexRefSetFilterMode(texRef, fm)
    @ccall librccl.hipTexRefSetFilterMode(texRef::Ptr{textureReference},
                                          fm::hipTextureFilterMode)::hipError_t
end

function hipTexRefSetFlags(texRef, Flags)
    @ccall librccl.hipTexRefSetFlags(texRef::Ptr{textureReference},
                                     Flags::Cuint)::hipError_t
end

function hipTexRefSetFormat(texRef, fmt, NumPackedComponents)
    @ccall librccl.hipTexRefSetFormat(texRef::Ptr{textureReference}, fmt::hipArray_Format,
                                      NumPackedComponents::Cint)::hipError_t
end

function hipBindTexture(offset, tex, devPtr, desc, size)
    @ccall librccl.hipBindTexture(offset::Ptr{Csize_t}, tex::Ptr{textureReference},
                                  devPtr::Ptr{Cvoid}, desc::Ptr{hipChannelFormatDesc},
                                  size::Csize_t)::hipError_t
end

function hipBindTexture2D(offset, tex, devPtr, desc, width, height, pitch)
    @ccall librccl.hipBindTexture2D(offset::Ptr{Csize_t}, tex::Ptr{textureReference},
                                    devPtr::Ptr{Cvoid}, desc::Ptr{hipChannelFormatDesc},
                                    width::Csize_t, height::Csize_t,
                                    pitch::Csize_t)::hipError_t
end

function hipBindTextureToArray(tex, array, desc)
    @ccall librccl.hipBindTextureToArray(tex::Ptr{textureReference},
                                         array::hipArray_const_t,
                                         desc::Ptr{hipChannelFormatDesc})::hipError_t
end

function hipGetTextureAlignmentOffset(offset, texref)
    @ccall librccl.hipGetTextureAlignmentOffset(offset::Ptr{Csize_t},
                                                texref::Ptr{textureReference})::hipError_t
end

function hipUnbindTexture(tex)
    @ccall librccl.hipUnbindTexture(tex::Ptr{textureReference})::hipError_t
end

function hipTexRefGetAddress(dev_ptr, texRef)
    @ccall librccl.hipTexRefGetAddress(dev_ptr::Ptr{hipDeviceptr_t},
                                       texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetAddressMode(pam, texRef, dim)
    @ccall librccl.hipTexRefGetAddressMode(pam::Ptr{hipTextureAddressMode},
                                           texRef::Ptr{textureReference},
                                           dim::Cint)::hipError_t
end

function hipTexRefGetFilterMode(pfm, texRef)
    @ccall librccl.hipTexRefGetFilterMode(pfm::Ptr{hipTextureFilterMode},
                                          texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetFlags(pFlags, texRef)
    @ccall librccl.hipTexRefGetFlags(pFlags::Ptr{Cuint},
                                     texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetFormat(pFormat, pNumChannels, texRef)
    @ccall librccl.hipTexRefGetFormat(pFormat::Ptr{hipArray_Format},
                                      pNumChannels::Ptr{Cint},
                                      texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetMaxAnisotropy(pmaxAnsio, texRef)
    @ccall librccl.hipTexRefGetMaxAnisotropy(pmaxAnsio::Ptr{Cint},
                                             texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetMipmapFilterMode(pfm, texRef)
    @ccall librccl.hipTexRefGetMipmapFilterMode(pfm::Ptr{hipTextureFilterMode},
                                                texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetMipmapLevelBias(pbias, texRef)
    @ccall librccl.hipTexRefGetMipmapLevelBias(pbias::Ptr{Cfloat},
                                               texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef)
    @ccall librccl.hipTexRefGetMipmapLevelClamp(pminMipmapLevelClamp::Ptr{Cfloat},
                                                pmaxMipmapLevelClamp::Ptr{Cfloat},
                                                texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefGetMipMappedArray(pArray, texRef)
    @ccall librccl.hipTexRefGetMipMappedArray(pArray::Ptr{hipMipmappedArray_t},
                                              texRef::Ptr{textureReference})::hipError_t
end

function hipTexRefSetAddress(ByteOffset, texRef, dptr, bytes)
    @ccall librccl.hipTexRefSetAddress(ByteOffset::Ptr{Csize_t},
                                       texRef::Ptr{textureReference}, dptr::hipDeviceptr_t,
                                       bytes::Csize_t)::hipError_t
end

function hipTexRefSetAddress2D(texRef, desc, dptr, Pitch)
    @ccall librccl.hipTexRefSetAddress2D(texRef::Ptr{textureReference},
                                         desc::Ptr{HIP_ARRAY_DESCRIPTOR},
                                         dptr::hipDeviceptr_t, Pitch::Csize_t)::hipError_t
end

function hipTexRefSetMaxAnisotropy(texRef, maxAniso)
    @ccall librccl.hipTexRefSetMaxAnisotropy(texRef::Ptr{textureReference},
                                             maxAniso::Cuint)::hipError_t
end

function hipTexRefSetBorderColor(texRef, pBorderColor)
    @ccall librccl.hipTexRefSetBorderColor(texRef::Ptr{textureReference},
                                           pBorderColor::Ptr{Cfloat})::hipError_t
end

function hipTexRefSetMipmapFilterMode(texRef, fm)
    @ccall librccl.hipTexRefSetMipmapFilterMode(texRef::Ptr{textureReference},
                                                fm::hipTextureFilterMode)::hipError_t
end

function hipTexRefSetMipmapLevelBias(texRef, bias)
    @ccall librccl.hipTexRefSetMipmapLevelBias(texRef::Ptr{textureReference},
                                               bias::Cfloat)::hipError_t
end

function hipTexRefSetMipmapLevelClamp(texRef, minMipMapLevelClamp, maxMipMapLevelClamp)
    @ccall librccl.hipTexRefSetMipmapLevelClamp(texRef::Ptr{textureReference},
                                                minMipMapLevelClamp::Cfloat,
                                                maxMipMapLevelClamp::Cfloat)::hipError_t
end

function hipTexRefSetMipmappedArray(texRef, mipmappedArray, Flags)
    @ccall librccl.hipTexRefSetMipmappedArray(texRef::Ptr{textureReference},
                                              mipmappedArray::Ptr{hipMipmappedArray},
                                              Flags::Cuint)::hipError_t
end

function hipApiName(id)
    @ccall librccl.hipApiName(id::UInt32)::Cstring
end

function hipKernelNameRef(f)
    @ccall librccl.hipKernelNameRef(f::hipFunction_t)::Cstring
end

function hipKernelNameRefByPtr(hostFunction, stream)
    @ccall librccl.hipKernelNameRefByPtr(hostFunction::Ptr{Cvoid},
                                         stream::HIPStream)::Cstring
end

function hipGetStreamDeviceId(stream)
    @ccall librccl.hipGetStreamDeviceId(stream::HIPStream)::Cint
end

function hipStreamBeginCapture(stream, mode)
    @ccall librccl.hipStreamBeginCapture(stream::HIPStream,
                                         mode::hipStreamCaptureMode)::hipError_t
end

function hipStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData,
                                      numDependencies, mode)
    @ccall librccl.hipStreamBeginCaptureToGraph(stream::HIPStream, graph::hipGraph_t,
                                                dependencies::Ptr{hipGraphNode_t},
                                                dependencyData::Ptr{hipGraphEdgeData},
                                                numDependencies::Csize_t,
                                                mode::hipStreamCaptureMode)::hipError_t
end

function hipStreamEndCapture(stream, pGraph)
    @ccall librccl.hipStreamEndCapture(stream::HIPStream,
                                       pGraph::Ptr{hipGraph_t})::hipError_t
end

function hipStreamGetCaptureInfo(stream, pCaptureStatus, pId)
    @ccall librccl.hipStreamGetCaptureInfo(stream::HIPStream,
                                           pCaptureStatus::Ptr{hipStreamCaptureStatus},
                                           pId::Ptr{Culonglong})::hipError_t
end

function hipStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out,
                                    dependencies_out, numDependencies_out)
    @ccall librccl.hipStreamGetCaptureInfo_v2(stream::HIPStream,
                                              captureStatus_out::Ptr{hipStreamCaptureStatus},
                                              id_out::Ptr{Culonglong},
                                              graph_out::Ptr{hipGraph_t},
                                              dependencies_out::Ptr{Ptr{hipGraphNode_t}},
                                              numDependencies_out::Ptr{Csize_t})::hipError_t
end

function hipStreamIsCapturing(stream, pCaptureStatus)
    @ccall librccl.hipStreamIsCapturing(stream::HIPStream,
                                        pCaptureStatus::Ptr{hipStreamCaptureStatus})::hipError_t
end

function hipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)
    @ccall librccl.hipStreamUpdateCaptureDependencies(stream::HIPStream,
                                                      dependencies::Ptr{hipGraphNode_t},
                                                      numDependencies::Csize_t,
                                                      flags::Cuint)::hipError_t
end

function hipThreadExchangeStreamCaptureMode(mode)
    @ccall librccl.hipThreadExchangeStreamCaptureMode(mode::Ptr{hipStreamCaptureMode})::hipError_t
end

function hipGraphCreate(pGraph, flags)
    @ccall librccl.hipGraphCreate(pGraph::Ptr{hipGraph_t}, flags::Cuint)::hipError_t
end

function hipGraphDestroy(graph)
    @ccall librccl.hipGraphDestroy(graph::hipGraph_t)::hipError_t
end

function hipGraphAddDependencies(graph, from, to, numDependencies)
    @ccall librccl.hipGraphAddDependencies(graph::hipGraph_t, from::Ptr{hipGraphNode_t},
                                           to::Ptr{hipGraphNode_t},
                                           numDependencies::Csize_t)::hipError_t
end

function hipGraphRemoveDependencies(graph, from, to, numDependencies)
    @ccall librccl.hipGraphRemoveDependencies(graph::hipGraph_t, from::Ptr{hipGraphNode_t},
                                              to::Ptr{hipGraphNode_t},
                                              numDependencies::Csize_t)::hipError_t
end

function hipGraphGetEdges(graph, from, to, numEdges)
    @ccall librccl.hipGraphGetEdges(graph::hipGraph_t, from::Ptr{hipGraphNode_t},
                                    to::Ptr{hipGraphNode_t},
                                    numEdges::Ptr{Csize_t})::hipError_t
end

function hipGraphGetNodes(graph, nodes, numNodes)
    @ccall librccl.hipGraphGetNodes(graph::hipGraph_t, nodes::Ptr{hipGraphNode_t},
                                    numNodes::Ptr{Csize_t})::hipError_t
end

function hipGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)
    @ccall librccl.hipGraphGetRootNodes(graph::hipGraph_t, pRootNodes::Ptr{hipGraphNode_t},
                                        pNumRootNodes::Ptr{Csize_t})::hipError_t
end

function hipGraphNodeGetDependencies(node, pDependencies, pNumDependencies)
    @ccall librccl.hipGraphNodeGetDependencies(node::hipGraphNode_t,
                                               pDependencies::Ptr{hipGraphNode_t},
                                               pNumDependencies::Ptr{Csize_t})::hipError_t
end

function hipGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes)
    @ccall librccl.hipGraphNodeGetDependentNodes(node::hipGraphNode_t,
                                                 pDependentNodes::Ptr{hipGraphNode_t},
                                                 pNumDependentNodes::Ptr{Csize_t})::hipError_t
end

function hipGraphNodeGetType(node, pType)
    @ccall librccl.hipGraphNodeGetType(node::hipGraphNode_t,
                                       pType::Ptr{hipGraphNodeType})::hipError_t
end

function hipGraphDestroyNode(node)
    @ccall librccl.hipGraphDestroyNode(node::hipGraphNode_t)::hipError_t
end

function hipGraphClone(pGraphClone, originalGraph)
    @ccall librccl.hipGraphClone(pGraphClone::Ptr{hipGraph_t},
                                 originalGraph::hipGraph_t)::hipError_t
end

function hipGraphNodeFindInClone(pNode, originalNode, clonedGraph)
    @ccall librccl.hipGraphNodeFindInClone(pNode::Ptr{hipGraphNode_t},
                                           originalNode::hipGraphNode_t,
                                           clonedGraph::hipGraph_t)::hipError_t
end

function hipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize)
    @ccall librccl.hipGraphInstantiate(pGraphExec::Ptr{hipGraphExec_t}, graph::hipGraph_t,
                                       pErrorNode::Ptr{hipGraphNode_t}, pLogBuffer::Cstring,
                                       bufferSize::Csize_t)::hipError_t
end

function hipGraphInstantiateWithFlags(pGraphExec, graph, flags)
    @ccall librccl.hipGraphInstantiateWithFlags(pGraphExec::Ptr{hipGraphExec_t},
                                                graph::hipGraph_t,
                                                flags::Culonglong)::hipError_t
end

function hipGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)
    @ccall librccl.hipGraphInstantiateWithParams(pGraphExec::Ptr{hipGraphExec_t},
                                                 graph::hipGraph_t,
                                                 instantiateParams::Ptr{hipGraphInstantiateParams})::hipError_t
end

function hipGraphLaunch(graphExec, stream)
    @ccall librccl.hipGraphLaunch(graphExec::hipGraphExec_t, stream::HIPStream)::hipError_t
end

function hipGraphUpload(graphExec, stream)
    @ccall librccl.hipGraphUpload(graphExec::hipGraphExec_t, stream::HIPStream)::hipError_t
end

function hipGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    @ccall librccl.hipGraphAddNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                   pDependencies::Ptr{hipGraphNode_t},
                                   numDependencies::Csize_t,
                                   nodeParams::Ptr{hipGraphNodeParams})::hipError_t
end

function hipGraphExecGetFlags(graphExec, flags)
    @ccall librccl.hipGraphExecGetFlags(graphExec::hipGraphExec_t,
                                        flags::Ptr{Culonglong})::hipError_t
end

function hipGraphNodeSetParams(node, nodeParams)
    @ccall librccl.hipGraphNodeSetParams(node::hipGraphNode_t,
                                         nodeParams::Ptr{hipGraphNodeParams})::hipError_t
end

function hipGraphExecNodeSetParams(graphExec, node, nodeParams)
    @ccall librccl.hipGraphExecNodeSetParams(graphExec::hipGraphExec_t,
                                             node::hipGraphNode_t,
                                             nodeParams::Ptr{hipGraphNodeParams})::hipError_t
end

function hipGraphExecDestroy(graphExec)
    @ccall librccl.hipGraphExecDestroy(graphExec::hipGraphExec_t)::hipError_t
end

function hipGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    @ccall librccl.hipGraphExecUpdate(hGraphExec::hipGraphExec_t, hGraph::hipGraph_t,
                                      hErrorNode_out::Ptr{hipGraphNode_t},
                                      updateResult_out::Ptr{hipGraphExecUpdateResult})::hipError_t
end

function hipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies,
                               pNodeParams)
    @ccall librccl.hipGraphAddKernelNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                         pDependencies::Ptr{hipGraphNode_t},
                                         numDependencies::Csize_t,
                                         pNodeParams::Ptr{hipKernelNodeParams})::hipError_t
end

function hipGraphKernelNodeGetParams(node, pNodeParams)
    @ccall librccl.hipGraphKernelNodeGetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipKernelNodeParams})::hipError_t
end

function hipGraphKernelNodeSetParams(node, pNodeParams)
    @ccall librccl.hipGraphKernelNodeSetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipKernelNodeParams})::hipError_t
end

function hipGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)
    @ccall librccl.hipGraphExecKernelNodeSetParams(hGraphExec::hipGraphExec_t,
                                                   node::hipGraphNode_t,
                                                   pNodeParams::Ptr{hipKernelNodeParams})::hipError_t
end

function hipDrvGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies,
                                  copyParams, ctx)
    @ccall librccl.hipDrvGraphAddMemcpyNode(phGraphNode::Ptr{hipGraphNode_t},
                                            hGraph::hipGraph_t,
                                            dependencies::Ptr{hipGraphNode_t},
                                            numDependencies::Csize_t,
                                            copyParams::Ptr{HIP_MEMCPY3D},
                                            ctx::hipCtx_t)::hipError_t
end

function hipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies,
                               pCopyParams)
    @ccall librccl.hipGraphAddMemcpyNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                         pDependencies::Ptr{hipGraphNode_t},
                                         numDependencies::Csize_t,
                                         pCopyParams::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipGraphMemcpyNodeGetParams(node, pNodeParams)
    @ccall librccl.hipGraphMemcpyNodeGetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipGraphMemcpyNodeSetParams(node, pNodeParams)
    @ccall librccl.hipGraphMemcpyNodeSetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipGraphKernelNodeSetAttribute(hNode, attr, value)
    @ccall librccl.hipGraphKernelNodeSetAttribute(hNode::hipGraphNode_t,
                                                  attr::hipLaunchAttributeID,
                                                  value::Ptr{hipLaunchAttributeValue})::hipError_t
end

function hipGraphKernelNodeGetAttribute(hNode, attr, value)
    @ccall librccl.hipGraphKernelNodeGetAttribute(hNode::hipGraphNode_t,
                                                  attr::hipLaunchAttributeID,
                                                  value::Ptr{hipLaunchAttributeValue})::hipError_t
end

function hipGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)
    @ccall librccl.hipGraphExecMemcpyNodeSetParams(hGraphExec::hipGraphExec_t,
                                                   node::hipGraphNode_t,
                                                   pNodeParams::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst,
                                 src, count, kind)
    @ccall librccl.hipGraphAddMemcpyNode1D(pGraphNode::Ptr{hipGraphNode_t},
                                           graph::hipGraph_t,
                                           pDependencies::Ptr{hipGraphNode_t},
                                           numDependencies::Csize_t, dst::Ptr{Cvoid},
                                           src::Ptr{Cvoid}, count::Csize_t,
                                           kind::hipMemcpyKind)::hipError_t
end

function hipGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)
    @ccall librccl.hipGraphMemcpyNodeSetParams1D(node::hipGraphNode_t, dst::Ptr{Cvoid},
                                                 src::Ptr{Cvoid}, count::Csize_t,
                                                 kind::hipMemcpyKind)::hipError_t
end

function hipGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)
    @ccall librccl.hipGraphExecMemcpyNodeSetParams1D(hGraphExec::hipGraphExec_t,
                                                     node::hipGraphNode_t, dst::Ptr{Cvoid},
                                                     src::Ptr{Cvoid}, count::Csize_t,
                                                     kind::hipMemcpyKind)::hipError_t
end

function hipGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies,
                                         dst, symbol, count, offset, kind)
    @ccall librccl.hipGraphAddMemcpyNodeFromSymbol(pGraphNode::Ptr{hipGraphNode_t},
                                                   graph::hipGraph_t,
                                                   pDependencies::Ptr{hipGraphNode_t},
                                                   numDependencies::Csize_t,
                                                   dst::Ptr{Cvoid}, symbol::Ptr{Cvoid},
                                                   count::Csize_t, offset::Csize_t,
                                                   kind::hipMemcpyKind)::hipError_t
end

function hipGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind)
    @ccall librccl.hipGraphMemcpyNodeSetParamsFromSymbol(node::hipGraphNode_t,
                                                         dst::Ptr{Cvoid},
                                                         symbol::Ptr{Cvoid}, count::Csize_t,
                                                         offset::Csize_t,
                                                         kind::hipMemcpyKind)::hipError_t
end

function hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count,
                                                   offset, kind)
    @ccall librccl.hipGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec::hipGraphExec_t,
                                                             node::hipGraphNode_t,
                                                             dst::Ptr{Cvoid},
                                                             symbol::Ptr{Cvoid},
                                                             count::Csize_t,
                                                             offset::Csize_t,
                                                             kind::hipMemcpyKind)::hipError_t
end

function hipGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies,
                                       symbol, src, count, offset, kind)
    @ccall librccl.hipGraphAddMemcpyNodeToSymbol(pGraphNode::Ptr{hipGraphNode_t},
                                                 graph::hipGraph_t,
                                                 pDependencies::Ptr{hipGraphNode_t},
                                                 numDependencies::Csize_t,
                                                 symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                                 count::Csize_t, offset::Csize_t,
                                                 kind::hipMemcpyKind)::hipError_t
end

function hipGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind)
    @ccall librccl.hipGraphMemcpyNodeSetParamsToSymbol(node::hipGraphNode_t,
                                                       symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                                       count::Csize_t, offset::Csize_t,
                                                       kind::hipMemcpyKind)::hipError_t
end

function hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count,
                                                 offset, kind)
    @ccall librccl.hipGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec::hipGraphExec_t,
                                                           node::hipGraphNode_t,
                                                           symbol::Ptr{Cvoid},
                                                           src::Ptr{Cvoid}, count::Csize_t,
                                                           offset::Csize_t,
                                                           kind::hipMemcpyKind)::hipError_t
end

function hipGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies,
                               pMemsetParams)
    @ccall librccl.hipGraphAddMemsetNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                         pDependencies::Ptr{hipGraphNode_t},
                                         numDependencies::Csize_t,
                                         pMemsetParams::Ptr{hipMemsetParams})::hipError_t
end

function hipGraphMemsetNodeGetParams(node, pNodeParams)
    @ccall librccl.hipGraphMemsetNodeGetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipMemsetParams})::hipError_t
end

function hipGraphMemsetNodeSetParams(node, pNodeParams)
    @ccall librccl.hipGraphMemsetNodeSetParams(node::hipGraphNode_t,
                                               pNodeParams::Ptr{hipMemsetParams})::hipError_t
end

function hipGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)
    @ccall librccl.hipGraphExecMemsetNodeSetParams(hGraphExec::hipGraphExec_t,
                                                   node::hipGraphNode_t,
                                                   pNodeParams::Ptr{hipMemsetParams})::hipError_t
end

function hipGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    @ccall librccl.hipGraphAddHostNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                       pDependencies::Ptr{hipGraphNode_t},
                                       numDependencies::Csize_t,
                                       pNodeParams::Ptr{hipHostNodeParams})::hipError_t
end

function hipGraphHostNodeGetParams(node, pNodeParams)
    @ccall librccl.hipGraphHostNodeGetParams(node::hipGraphNode_t,
                                             pNodeParams::Ptr{hipHostNodeParams})::hipError_t
end

function hipGraphHostNodeSetParams(node, pNodeParams)
    @ccall librccl.hipGraphHostNodeSetParams(node::hipGraphNode_t,
                                             pNodeParams::Ptr{hipHostNodeParams})::hipError_t
end

function hipGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)
    @ccall librccl.hipGraphExecHostNodeSetParams(hGraphExec::hipGraphExec_t,
                                                 node::hipGraphNode_t,
                                                 pNodeParams::Ptr{hipHostNodeParams})::hipError_t
end

function hipGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies,
                                   childGraph)
    @ccall librccl.hipGraphAddChildGraphNode(pGraphNode::Ptr{hipGraphNode_t},
                                             graph::hipGraph_t,
                                             pDependencies::Ptr{hipGraphNode_t},
                                             numDependencies::Csize_t,
                                             childGraph::hipGraph_t)::hipError_t
end

function hipGraphChildGraphNodeGetGraph(node, pGraph)
    @ccall librccl.hipGraphChildGraphNodeGetGraph(node::hipGraphNode_t,
                                                  pGraph::Ptr{hipGraph_t})::hipError_t
end

function hipGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)
    @ccall librccl.hipGraphExecChildGraphNodeSetParams(hGraphExec::hipGraphExec_t,
                                                       node::hipGraphNode_t,
                                                       childGraph::hipGraph_t)::hipError_t
end

function hipGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)
    @ccall librccl.hipGraphAddEmptyNode(pGraphNode::Ptr{hipGraphNode_t}, graph::hipGraph_t,
                                        pDependencies::Ptr{hipGraphNode_t},
                                        numDependencies::Csize_t)::hipError_t
end

function hipGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies,
                                    event)
    @ccall librccl.hipGraphAddEventRecordNode(pGraphNode::Ptr{hipGraphNode_t},
                                              graph::hipGraph_t,
                                              pDependencies::Ptr{hipGraphNode_t},
                                              numDependencies::Csize_t,
                                              event::hipEvent_t)::hipError_t
end

function hipGraphEventRecordNodeGetEvent(node, event_out)
    @ccall librccl.hipGraphEventRecordNodeGetEvent(node::hipGraphNode_t,
                                                   event_out::Ptr{hipEvent_t})::hipError_t
end

function hipGraphEventRecordNodeSetEvent(node, event)
    @ccall librccl.hipGraphEventRecordNodeSetEvent(node::hipGraphNode_t,
                                                   event::hipEvent_t)::hipError_t
end

function hipGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
    @ccall librccl.hipGraphExecEventRecordNodeSetEvent(hGraphExec::hipGraphExec_t,
                                                       hNode::hipGraphNode_t,
                                                       event::hipEvent_t)::hipError_t
end

function hipGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)
    @ccall librccl.hipGraphAddEventWaitNode(pGraphNode::Ptr{hipGraphNode_t},
                                            graph::hipGraph_t,
                                            pDependencies::Ptr{hipGraphNode_t},
                                            numDependencies::Csize_t,
                                            event::hipEvent_t)::hipError_t
end

function hipGraphEventWaitNodeGetEvent(node, event_out)
    @ccall librccl.hipGraphEventWaitNodeGetEvent(node::hipGraphNode_t,
                                                 event_out::Ptr{hipEvent_t})::hipError_t
end

function hipGraphEventWaitNodeSetEvent(node, event)
    @ccall librccl.hipGraphEventWaitNodeSetEvent(node::hipGraphNode_t,
                                                 event::hipEvent_t)::hipError_t
end

function hipGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
    @ccall librccl.hipGraphExecEventWaitNodeSetEvent(hGraphExec::hipGraphExec_t,
                                                     hNode::hipGraphNode_t,
                                                     event::hipEvent_t)::hipError_t
end

function hipGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies,
                                 pNodeParams)
    @ccall librccl.hipGraphAddMemAllocNode(pGraphNode::Ptr{hipGraphNode_t},
                                           graph::hipGraph_t,
                                           pDependencies::Ptr{hipGraphNode_t},
                                           numDependencies::Csize_t,
                                           pNodeParams::Ptr{hipMemAllocNodeParams})::hipError_t
end

function hipGraphMemAllocNodeGetParams(node, pNodeParams)
    @ccall librccl.hipGraphMemAllocNodeGetParams(node::hipGraphNode_t,
                                                 pNodeParams::Ptr{hipMemAllocNodeParams})::hipError_t
end

function hipGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dev_ptr)
    @ccall librccl.hipGraphAddMemFreeNode(pGraphNode::Ptr{hipGraphNode_t},
                                          graph::hipGraph_t,
                                          pDependencies::Ptr{hipGraphNode_t},
                                          numDependencies::Csize_t,
                                          dev_ptr::Ptr{Cvoid})::hipError_t
end

function hipGraphMemFreeNodeGetParams(node, dev_ptr)
    @ccall librccl.hipGraphMemFreeNodeGetParams(node::hipGraphNode_t,
                                                dev_ptr::Ptr{Cvoid})::hipError_t
end

function hipDeviceGetGraphMemAttribute(device, attr, value)
    @ccall librccl.hipDeviceGetGraphMemAttribute(device::Cint,
                                                 attr::hipGraphMemAttributeType,
                                                 value::Ptr{Cvoid})::hipError_t
end

function hipDeviceSetGraphMemAttribute(device, attr, value)
    @ccall librccl.hipDeviceSetGraphMemAttribute(device::Cint,
                                                 attr::hipGraphMemAttributeType,
                                                 value::Ptr{Cvoid})::hipError_t
end

function hipDeviceGraphMemTrim(device)
    @ccall librccl.hipDeviceGraphMemTrim(device::Cint)::hipError_t
end

function hipUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)
    @ccall librccl.hipUserObjectCreate(object_out::Ptr{hipUserObject_t}, ptr::Ptr{Cvoid},
                                       destroy::hipHostFn_t, initialRefcount::Cuint,
                                       flags::Cuint)::hipError_t
end

function hipUserObjectRelease(object, count)
    @ccall librccl.hipUserObjectRelease(object::hipUserObject_t, count::Cuint)::hipError_t
end

function hipUserObjectRetain(object, count)
    @ccall librccl.hipUserObjectRetain(object::hipUserObject_t, count::Cuint)::hipError_t
end

function hipGraphRetainUserObject(graph, object, count, flags)
    @ccall librccl.hipGraphRetainUserObject(graph::hipGraph_t, object::hipUserObject_t,
                                            count::Cuint, flags::Cuint)::hipError_t
end

function hipGraphReleaseUserObject(graph, object, count)
    @ccall librccl.hipGraphReleaseUserObject(graph::hipGraph_t, object::hipUserObject_t,
                                             count::Cuint)::hipError_t
end

function hipGraphDebugDotPrint(graph, path, flags)
    @ccall librccl.hipGraphDebugDotPrint(graph::hipGraph_t, path::Cstring,
                                         flags::Cuint)::hipError_t
end

function hipGraphKernelNodeCopyAttributes(hSrc, hDst)
    @ccall librccl.hipGraphKernelNodeCopyAttributes(hSrc::hipGraphNode_t,
                                                    hDst::hipGraphNode_t)::hipError_t
end

function hipGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)
    @ccall librccl.hipGraphNodeSetEnabled(hGraphExec::hipGraphExec_t, hNode::hipGraphNode_t,
                                          isEnabled::Cuint)::hipError_t
end

function hipGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)
    @ccall librccl.hipGraphNodeGetEnabled(hGraphExec::hipGraphExec_t, hNode::hipGraphNode_t,
                                          isEnabled::Ptr{Cuint})::hipError_t
end

function hipGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies,
                                               numDependencies, nodeParams)
    @ccall librccl.hipGraphAddExternalSemaphoresWaitNode(pGraphNode::Ptr{hipGraphNode_t},
                                                         graph::hipGraph_t,
                                                         pDependencies::Ptr{hipGraphNode_t},
                                                         numDependencies::Csize_t,
                                                         nodeParams::Ptr{hipExternalSemaphoreWaitNodeParams})::hipError_t
end

function hipGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies,
                                                 numDependencies, nodeParams)
    @ccall librccl.hipGraphAddExternalSemaphoresSignalNode(pGraphNode::Ptr{hipGraphNode_t},
                                                           graph::hipGraph_t,
                                                           pDependencies::Ptr{hipGraphNode_t},
                                                           numDependencies::Csize_t,
                                                           nodeParams::Ptr{hipExternalSemaphoreSignalNodeParams})::hipError_t
end

function hipGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)
    @ccall librccl.hipGraphExternalSemaphoresSignalNodeSetParams(hNode::hipGraphNode_t,
                                                                 nodeParams::Ptr{hipExternalSemaphoreSignalNodeParams})::hipError_t
end

function hipGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)
    @ccall librccl.hipGraphExternalSemaphoresWaitNodeSetParams(hNode::hipGraphNode_t,
                                                               nodeParams::Ptr{hipExternalSemaphoreWaitNodeParams})::hipError_t
end

function hipGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)
    @ccall librccl.hipGraphExternalSemaphoresSignalNodeGetParams(hNode::hipGraphNode_t,
                                                                 params_out::Ptr{hipExternalSemaphoreSignalNodeParams})::hipError_t
end

function hipGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)
    @ccall librccl.hipGraphExternalSemaphoresWaitNodeGetParams(hNode::hipGraphNode_t,
                                                               params_out::Ptr{hipExternalSemaphoreWaitNodeParams})::hipError_t
end

function hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)
    @ccall librccl.hipGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec::hipGraphExec_t,
                                                                     hNode::hipGraphNode_t,
                                                                     nodeParams::Ptr{hipExternalSemaphoreSignalNodeParams})::hipError_t
end

function hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)
    @ccall librccl.hipGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec::hipGraphExec_t,
                                                                   hNode::hipGraphNode_t,
                                                                   nodeParams::Ptr{hipExternalSemaphoreWaitNodeParams})::hipError_t
end

function hipDrvGraphMemcpyNodeGetParams(hNode, nodeParams)
    @ccall librccl.hipDrvGraphMemcpyNodeGetParams(hNode::hipGraphNode_t,
                                                  nodeParams::Ptr{HIP_MEMCPY3D})::hipError_t
end

function hipDrvGraphMemcpyNodeSetParams(hNode, nodeParams)
    @ccall librccl.hipDrvGraphMemcpyNodeSetParams(hNode::hipGraphNode_t,
                                                  nodeParams::Ptr{HIP_MEMCPY3D})::hipError_t
end

function hipDrvGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies,
                                  memsetParams, ctx)
    @ccall librccl.hipDrvGraphAddMemsetNode(phGraphNode::Ptr{hipGraphNode_t},
                                            hGraph::hipGraph_t,
                                            dependencies::Ptr{hipGraphNode_t},
                                            numDependencies::Csize_t,
                                            memsetParams::Ptr{HIP_MEMSET_NODE_PARAMS},
                                            ctx::hipCtx_t)::hipError_t
end

function hipDrvGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr)
    @ccall librccl.hipDrvGraphAddMemFreeNode(phGraphNode::Ptr{hipGraphNode_t},
                                             hGraph::hipGraph_t,
                                             dependencies::Ptr{hipGraphNode_t},
                                             numDependencies::Csize_t,
                                             dptr::hipDeviceptr_t)::hipError_t
end

function hipDrvGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)
    @ccall librccl.hipDrvGraphExecMemcpyNodeSetParams(hGraphExec::hipGraphExec_t,
                                                      hNode::hipGraphNode_t,
                                                      copyParams::Ptr{HIP_MEMCPY3D},
                                                      ctx::hipCtx_t)::hipError_t
end

function hipDrvGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)
    @ccall librccl.hipDrvGraphExecMemsetNodeSetParams(hGraphExec::hipGraphExec_t,
                                                      hNode::hipGraphNode_t,
                                                      memsetParams::Ptr{HIP_MEMSET_NODE_PARAMS},
                                                      ctx::hipCtx_t)::hipError_t
end

function hipMemAddressFree(devPtr, size)
    @ccall librccl.hipMemAddressFree(devPtr::Ptr{Cvoid}, size::Csize_t)::hipError_t
end

function hipMemAddressReserve(ptr, size, alignment, addr, flags)
    @ccall librccl.hipMemAddressReserve(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t,
                                        alignment::Csize_t, addr::Ptr{Cvoid},
                                        flags::Culonglong)::hipError_t
end

function hipMemCreate(handle, size, prop, flags)
    @ccall librccl.hipMemCreate(handle::Ptr{hipMemGenericAllocationHandle_t}, size::Csize_t,
                                prop::Ptr{hipMemAllocationProp},
                                flags::Culonglong)::hipError_t
end

function hipMemExportToShareableHandle(shareableHandle, handle, handleType, flags)
    @ccall librccl.hipMemExportToShareableHandle(shareableHandle::Ptr{Cvoid},
                                                 handle::hipMemGenericAllocationHandle_t,
                                                 handleType::hipMemAllocationHandleType,
                                                 flags::Culonglong)::hipError_t
end

function hipMemGetAccess(flags, location, ptr)
    @ccall librccl.hipMemGetAccess(flags::Ptr{Culonglong}, location::Ptr{hipMemLocation},
                                   ptr::Ptr{Cvoid})::hipError_t
end

function hipMemGetAllocationGranularity(granularity, prop, option)
    @ccall librccl.hipMemGetAllocationGranularity(granularity::Ptr{Csize_t},
                                                  prop::Ptr{hipMemAllocationProp},
                                                  option::hipMemAllocationGranularity_flags)::hipError_t
end

function hipMemGetAllocationPropertiesFromHandle(prop, handle)
    @ccall librccl.hipMemGetAllocationPropertiesFromHandle(prop::Ptr{hipMemAllocationProp},
                                                           handle::hipMemGenericAllocationHandle_t)::hipError_t
end

function hipMemImportFromShareableHandle(handle, osHandle, shHandleType)
    @ccall librccl.hipMemImportFromShareableHandle(handle::Ptr{hipMemGenericAllocationHandle_t},
                                                   osHandle::Ptr{Cvoid},
                                                   shHandleType::hipMemAllocationHandleType)::hipError_t
end

function hipMemMap(ptr, size, offset, handle, flags)
    @ccall librccl.hipMemMap(ptr::Ptr{Cvoid}, size::Csize_t, offset::Csize_t,
                             handle::hipMemGenericAllocationHandle_t,
                             flags::Culonglong)::hipError_t
end

function hipMemMapArrayAsync(mapInfoList, count, stream)
    @ccall librccl.hipMemMapArrayAsync(mapInfoList::Ptr{hipArrayMapInfo}, count::Cuint,
                                       stream::HIPStream)::hipError_t
end

function hipMemRelease(handle)
    @ccall librccl.hipMemRelease(handle::hipMemGenericAllocationHandle_t)::hipError_t
end

function hipMemRetainAllocationHandle(handle, addr)
    @ccall librccl.hipMemRetainAllocationHandle(handle::Ptr{hipMemGenericAllocationHandle_t},
                                                addr::Ptr{Cvoid})::hipError_t
end

function hipMemSetAccess(ptr, size, desc, count)
    @ccall librccl.hipMemSetAccess(ptr::Ptr{Cvoid}, size::Csize_t,
                                   desc::Ptr{hipMemAccessDesc}, count::Csize_t)::hipError_t
end

function hipMemUnmap(ptr, size)
    @ccall librccl.hipMemUnmap(ptr::Ptr{Cvoid}, size::Csize_t)::hipError_t
end

function hipGraphicsMapResources(count, resources, stream)
    @ccall librccl.hipGraphicsMapResources(count::Cint,
                                           resources::Ptr{hipGraphicsResource_t},
                                           stream::HIPStream)::hipError_t
end

function hipGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)
    @ccall librccl.hipGraphicsSubResourceGetMappedArray(array::Ptr{hipArray_t},
                                                        resource::hipGraphicsResource_t,
                                                        arrayIndex::Cuint,
                                                        mipLevel::Cuint)::hipError_t
end

function hipGraphicsResourceGetMappedPointer(devPtr, size, resource)
    @ccall librccl.hipGraphicsResourceGetMappedPointer(devPtr::Ptr{Ptr{Cvoid}},
                                                       size::Ptr{Csize_t},
                                                       resource::hipGraphicsResource_t)::hipError_t
end

function hipGraphicsUnmapResources(count, resources, stream)
    @ccall librccl.hipGraphicsUnmapResources(count::Cint,
                                             resources::Ptr{hipGraphicsResource_t},
                                             stream::HIPStream)::hipError_t
end

function hipGraphicsUnregisterResource(resource)
    @ccall librccl.hipGraphicsUnregisterResource(resource::hipGraphicsResource_t)::hipError_t
end

function hipCreateSurfaceObject(pSurfObject, pResDesc)
    @ccall librccl.hipCreateSurfaceObject(pSurfObject::Ptr{hipSurfaceObject_t},
                                          pResDesc::Ptr{hipResourceDesc})::hipError_t
end

function hipDestroySurfaceObject(surfaceObject)
    @ccall librccl.hipDestroySurfaceObject(surfaceObject::hipSurfaceObject_t)::hipError_t
end

function hipMemcpy_spt(dst, src, sizeBytes, kind)
    @ccall librccl.hipMemcpy_spt(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, sizeBytes::Csize_t,
                                 kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyToSymbol_spt(symbol, src, sizeBytes, offset, kind)
    @ccall librccl.hipMemcpyToSymbol_spt(symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                         sizeBytes::Csize_t, offset::Csize_t,
                                         kind::hipMemcpyKind)::hipError_t
end

function hipMemcpyFromSymbol_spt(dst, symbol, sizeBytes, offset, kind)
    @ccall librccl.hipMemcpyFromSymbol_spt(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid},
                                           sizeBytes::Csize_t, offset::Csize_t,
                                           kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2D_spt(dst, dpitch, src, spitch, width, height, kind)
    @ccall librccl.hipMemcpy2D_spt(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid},
                                   spitch::Csize_t, width::Csize_t, height::Csize_t,
                                   kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DFromArray_spt(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    @ccall librccl.hipMemcpy2DFromArray_spt(dst::Ptr{Cvoid}, dpitch::Csize_t,
                                            src::hipArray_const_t, wOffset::Csize_t,
                                            hOffset::Csize_t, width::Csize_t,
                                            height::Csize_t,
                                            kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy3D_spt(p)
    @ccall librccl.hipMemcpy3D_spt(p::Ptr{hipMemcpy3DParms})::hipError_t
end

function hipMemset_spt(dst, value, sizeBytes)
    @ccall librccl.hipMemset_spt(dst::Ptr{Cvoid}, value::Cint,
                                 sizeBytes::Csize_t)::hipError_t
end

function hipMemsetAsync_spt(dst, value, sizeBytes, stream)
    @ccall librccl.hipMemsetAsync_spt(dst::Ptr{Cvoid}, value::Cint, sizeBytes::Csize_t,
                                      stream::HIPStream)::hipError_t
end

function hipMemset2D_spt(dst, pitch, value, width, height)
    @ccall librccl.hipMemset2D_spt(dst::Ptr{Cvoid}, pitch::Csize_t, value::Cint,
                                   width::Csize_t, height::Csize_t)::hipError_t
end

function hipMemset2DAsync_spt(dst, pitch, value, width, height, stream)
    @ccall librccl.hipMemset2DAsync_spt(dst::Ptr{Cvoid}, pitch::Csize_t, value::Cint,
                                        width::Csize_t, height::Csize_t,
                                        stream::HIPStream)::hipError_t
end

function hipMemset3DAsync_spt(pitchedDevPtr, value, extent, stream)
    @ccall librccl.hipMemset3DAsync_spt(pitchedDevPtr::hipPitchedPtr, value::Cint,
                                        extent::hipExtent, stream::HIPStream)::hipError_t
end

function hipMemset3D_spt(pitchedDevPtr, value, extent)
    @ccall librccl.hipMemset3D_spt(pitchedDevPtr::hipPitchedPtr, value::Cint,
                                   extent::hipExtent)::hipError_t
end

function hipMemcpyAsync_spt(dst, src, sizeBytes, kind, stream)
    @ccall librccl.hipMemcpyAsync_spt(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, sizeBytes::Csize_t,
                                      kind::hipMemcpyKind, stream::HIPStream)::hipError_t
end

function hipMemcpy3DAsync_spt(p, stream)
    @ccall librccl.hipMemcpy3DAsync_spt(p::Ptr{hipMemcpy3DParms},
                                        stream::HIPStream)::hipError_t
end

function hipMemcpy2DAsync_spt(dst, dpitch, src, spitch, width, height, kind, stream)
    @ccall librccl.hipMemcpy2DAsync_spt(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid},
                                        spitch::Csize_t, width::Csize_t, height::Csize_t,
                                        kind::hipMemcpyKind, stream::HIPStream)::hipError_t
end

function hipMemcpyFromSymbolAsync_spt(dst, symbol, sizeBytes, offset, kind, stream)
    @ccall librccl.hipMemcpyFromSymbolAsync_spt(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid},
                                                sizeBytes::Csize_t, offset::Csize_t,
                                                kind::hipMemcpyKind,
                                                stream::HIPStream)::hipError_t
end

function hipMemcpyToSymbolAsync_spt(symbol, src, sizeBytes, offset, kind, stream)
    @ccall librccl.hipMemcpyToSymbolAsync_spt(symbol::Ptr{Cvoid}, src::Ptr{Cvoid},
                                              sizeBytes::Csize_t, offset::Csize_t,
                                              kind::hipMemcpyKind,
                                              stream::HIPStream)::hipError_t
end

function hipMemcpyFromArray_spt(dst, src, wOffsetSrc, hOffset, count, kind)
    @ccall librccl.hipMemcpyFromArray_spt(dst::Ptr{Cvoid}, src::hipArray_const_t,
                                          wOffsetSrc::Csize_t, hOffset::Csize_t,
                                          count::Csize_t, kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DToArray_spt(dst, wOffset, hOffset, src, spitch, width, height, kind)
    @ccall librccl.hipMemcpy2DToArray_spt(dst::hipArray_t, wOffset::Csize_t,
                                          hOffset::Csize_t, src::Ptr{Cvoid},
                                          spitch::Csize_t, width::Csize_t, height::Csize_t,
                                          kind::hipMemcpyKind)::hipError_t
end

function hipMemcpy2DFromArrayAsync_spt(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width,
                                       height, kind, stream)
    @ccall librccl.hipMemcpy2DFromArrayAsync_spt(dst::Ptr{Cvoid}, dpitch::Csize_t,
                                                 src::hipArray_const_t, wOffsetSrc::Csize_t,
                                                 hOffsetSrc::Csize_t, width::Csize_t,
                                                 height::Csize_t, kind::hipMemcpyKind,
                                                 stream::HIPStream)::hipError_t
end

function hipMemcpy2DToArrayAsync_spt(dst, wOffset, hOffset, src, spitch, width, height,
                                     kind, stream)
    @ccall librccl.hipMemcpy2DToArrayAsync_spt(dst::hipArray_t, wOffset::Csize_t,
                                               hOffset::Csize_t, src::Ptr{Cvoid},
                                               spitch::Csize_t, width::Csize_t,
                                               height::Csize_t, kind::hipMemcpyKind,
                                               stream::HIPStream)::hipError_t
end

function hipStreamQuery_spt(stream)
    @ccall librccl.hipStreamQuery_spt(stream::HIPStream)::hipError_t
end

function hipStreamSynchronize_spt(stream)
    @ccall librccl.hipStreamSynchronize_spt(stream::HIPStream)::hipError_t
end

function hipStreamGetPriority_spt(stream, priority)
    @ccall librccl.hipStreamGetPriority_spt(stream::HIPStream,
                                            priority::Ptr{Cint})::hipError_t
end

function hipStreamWaitEvent_spt(stream, event, flags)
    @ccall librccl.hipStreamWaitEvent_spt(stream::HIPStream, event::hipEvent_t,
                                          flags::Cuint)::hipError_t
end

function hipStreamGetFlags_spt(stream, flags)
    @ccall librccl.hipStreamGetFlags_spt(stream::HIPStream, flags::Ptr{Cuint})::hipError_t
end

function hipStreamAddCallback_spt(stream, callback, userData, flags)
    @ccall librccl.hipStreamAddCallback_spt(stream::HIPStream,
                                            callback::hipStreamCallback_t,
                                            userData::Ptr{Cvoid}, flags::Cuint)::hipError_t
end

function hipEventRecord_spt(event, stream)
    @ccall librccl.hipEventRecord_spt(event::hipEvent_t, stream::HIPStream)::hipError_t
end

function hipLaunchCooperativeKernel_spt(f, gridDim, blockDim, kernelParams, sharedMemBytes,
                                        hStream)
    @ccall librccl.hipLaunchCooperativeKernel_spt(f::Ptr{Cvoid}, gridDim::dim3,
                                                  blockDim::dim3,
                                                  kernelParams::Ptr{Ptr{Cvoid}},
                                                  sharedMemBytes::UInt32,
                                                  hStream::hipStream_t)::hipError_t
end

function hipLaunchKernel_spt(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                             stream)
    @ccall librccl.hipLaunchKernel_spt(function_address::Ptr{Cvoid}, numBlocks::dim3,
                                       dimBlocks::dim3, args::Ptr{Ptr{Cvoid}},
                                       sharedMemBytes::Csize_t,
                                       stream::HIPStream)::hipError_t
end

function hipGraphLaunch_spt(graphExec, stream)
    @ccall librccl.hipGraphLaunch_spt(graphExec::hipGraphExec_t,
                                      stream::HIPStream)::hipError_t
end

function hipStreamBeginCapture_spt(stream, mode)
    @ccall librccl.hipStreamBeginCapture_spt(stream::HIPStream,
                                             mode::hipStreamCaptureMode)::hipError_t
end

function hipStreamEndCapture_spt(stream, pGraph)
    @ccall librccl.hipStreamEndCapture_spt(stream::HIPStream,
                                           pGraph::Ptr{hipGraph_t})::hipError_t
end

function hipStreamIsCapturing_spt(stream, pCaptureStatus)
    @ccall librccl.hipStreamIsCapturing_spt(stream::HIPStream,
                                            pCaptureStatus::Ptr{hipStreamCaptureStatus})::hipError_t
end

function hipStreamGetCaptureInfo_spt(stream, pCaptureStatus, pId)
    @ccall librccl.hipStreamGetCaptureInfo_spt(stream::HIPStream,
                                               pCaptureStatus::Ptr{hipStreamCaptureStatus},
                                               pId::Ptr{Culonglong})::hipError_t
end

function hipStreamGetCaptureInfo_v2_spt(stream, captureStatus_out, id_out, graph_out,
                                        dependencies_out, numDependencies_out)
    @ccall librccl.hipStreamGetCaptureInfo_v2_spt(stream::HIPStream,
                                                  captureStatus_out::Ptr{hipStreamCaptureStatus},
                                                  id_out::Ptr{Culonglong},
                                                  graph_out::Ptr{hipGraph_t},
                                                  dependencies_out::Ptr{Ptr{hipGraphNode_t}},
                                                  numDependencies_out::Ptr{Csize_t})::hipError_t
end

function hipLaunchHostFunc_spt(stream, fn, userData)
    @ccall librccl.hipLaunchHostFunc_spt(stream::HIPStream, fn::hipHostFn_t,
                                         userData::Ptr{Cvoid})::hipError_t
end

@cenum hipDataType::UInt32 begin
    HIP_R_32F = 0
    HIP_R_64F = 1
    HIP_R_16F = 2
    HIP_R_8I = 3
    HIP_C_32F = 4
    HIP_C_64F = 5
    HIP_C_16F = 6
    HIP_C_8I = 7
    HIP_R_8U = 8
    HIP_C_8U = 9
    HIP_R_32I = 10
    HIP_C_32I = 11
    HIP_R_32U = 12
    HIP_C_32U = 13
    HIP_R_16BF = 14
    HIP_C_16BF = 15
    HIP_R_4I = 16
    HIP_C_4I = 17
    HIP_R_4U = 18
    HIP_C_4U = 19
    HIP_R_16I = 20
    HIP_C_16I = 21
    HIP_R_16U = 22
    HIP_C_16U = 23
    HIP_R_64I = 24
    HIP_C_64I = 25
    HIP_R_64U = 26
    HIP_C_64U = 27
    HIP_R_8F_E4M3 = 28
    HIP_R_8F_E5M2 = 29
    HIP_R_8F_E4M3_FNUZ = 1000
    HIP_R_8F_E5M2_FNUZ = 1001
end

@cenum hipLibraryPropertyType::UInt32 begin
    HIP_LIBRARY_MAJOR_VERSION = 0
    HIP_LIBRARY_MINOR_VERSION = 1
    HIP_LIBRARY_PATCH_LEVEL = 2
end

struct __half_raw
    x::Cushort
end

struct __half2_raw
    x::Cushort
    y::Cushort
end

mutable struct ncclComm end

const ncclComm_t = Ptr{ncclComm}

struct ncclUniqueId
    internal::NTuple{128,Cchar}
end

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

@check function ncclMemAlloc(ptr, size)
    @ccall librccl.ncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::ncclResult_t
end

@check function pncclMemAlloc(ptr, size)
    @ccall librccl.pncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::ncclResult_t
end

@check function ncclMemFree(ptr)
    @ccall librccl.ncclMemFree(ptr::Ptr{Cvoid})::ncclResult_t
end

@check function pncclMemFree(ptr)
    @ccall librccl.pncclMemFree(ptr::Ptr{Cvoid})::ncclResult_t
end

@check function ncclGetVersion(version)
    @ccall librccl.ncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

@check function pncclGetVersion(version)
    @ccall librccl.pncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

@check function ncclGetUniqueId(uniqueId)
    @ccall librccl.ncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

@check function pncclGetUniqueId(uniqueId)
    @ccall librccl.pncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

@check function ncclCommInitRankConfig(comm, nranks, commId, rank, config)
    @ccall librccl.ncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
                                          commId::ncclUniqueId, rank::Cint,
                                          config::Ptr{ncclConfig_t})::ncclResult_t
end

@check function pncclCommInitRankConfig(comm, nranks, commId, rank, config)
    @ccall librccl.pncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
                                           commId::ncclUniqueId, rank::Cint,
                                           config::Ptr{ncclConfig_t})::ncclResult_t
end

@check function ncclCommInitRank(comm, nranks, commId, rank)
    @ccall librccl.ncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
                                    commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

@check function pncclCommInitRank(comm, nranks, commId, rank)
    @ccall librccl.pncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
                                     commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

@check function ncclCommInitAll(comm, ndev, devlist)
    @ccall librccl.ncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
                                   devlist::Ptr{Cint})::ncclResult_t
end

@check function pncclCommInitAll(comm, ndev, devlist)
    @ccall librccl.pncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
                                    devlist::Ptr{Cint})::ncclResult_t
end

@check function ncclCommFinalize(comm)
    @ccall librccl.ncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

@check function pncclCommFinalize(comm)
    @ccall librccl.pncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

@check function ncclCommDestroy(comm)
    @ccall librccl.ncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

@check function pncclCommDestroy(comm)
    @ccall librccl.pncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

@check function ncclCommAbort(comm)
    @ccall librccl.ncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

@check function pncclCommAbort(comm)
    @ccall librccl.pncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

@check function ncclCommSplit(comm, color, key, newcomm, config)
    @ccall librccl.ncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
                                 newcomm::Ptr{ncclComm_t},
                                 config::Ptr{ncclConfig_t})::ncclResult_t
end

@check function pncclCommSplit(comm, color, key, newcomm, config)
    @ccall librccl.pncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
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

@check function ncclCommGetAsyncError(comm, asyncError)
    @ccall librccl.ncclCommGetAsyncError(comm::ncclComm_t,
                                         asyncError::Ptr{ncclResult_t})::ncclResult_t
end

@check function pncclCommGetAsyncError(comm, asyncError)
    @ccall librccl.pncclCommGetAsyncError(comm::ncclComm_t,
                                          asyncError::Ptr{ncclResult_t})::ncclResult_t
end

@check function ncclCommCount(comm, count)
    @ccall librccl.ncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

@check function pncclCommCount(comm, count)
    @ccall librccl.pncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

@check function ncclCommCuDevice(comm, device)
    @ccall librccl.ncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

@check function pncclCommCuDevice(comm, device)
    @ccall librccl.pncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

@check function ncclCommUserRank(comm, rank)
    @ccall librccl.ncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

@check function pncclCommUserRank(comm, rank)
    @ccall librccl.pncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

@check function ncclCommRegister(comm, buff, size, handle)
    @ccall librccl.ncclCommRegister(comm::ncclComm_t, buff::Ptr{Cvoid}, size::Csize_t,
                                    handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

@check function pncclCommRegister(comm, buff, size, handle)
    @ccall librccl.pncclCommRegister(comm::ncclComm_t, buff::Ptr{Cvoid}, size::Csize_t,
                                     handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

@check function ncclCommDeregister(comm, handle)
    @ccall librccl.ncclCommDeregister(comm::ncclComm_t, handle::Ptr{Cvoid})::ncclResult_t
end

@check function pncclCommDeregister(comm, handle)
    @ccall librccl.pncclCommDeregister(comm::ncclComm_t, handle::Ptr{Cvoid})::ncclResult_t
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

@check function ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
    @ccall librccl.ncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
                                            datatype::ncclDataType_t,
                                            residence::ncclScalarResidence_t,
                                            comm::ncclComm_t)::ncclResult_t
end

@check function pncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
    @ccall librccl.pncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
                                             datatype::ncclDataType_t,
                                             residence::ncclScalarResidence_t,
                                             comm::ncclComm_t)::ncclResult_t
end

@check function ncclRedOpDestroy(op, comm)
    @ccall librccl.ncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

@check function pncclRedOpDestroy(op, comm)
    @ccall librccl.pncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

@check function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @ccall librccl.ncclReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                              datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
                              comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @ccall librccl.pncclReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                               datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
                               comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclBcast(buff, count, datatype, root, comm, stream)
    @ccall librccl.ncclBcast(buff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                             root::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclBcast(buff, count, datatype, root, comm, stream)
    @ccall librccl.pncclBcast(buff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                              root::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @ccall librccl.ncclBroadcast(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                                 datatype::ncclDataType_t, root::Cint, comm::ncclComm_t,
                                 stream::HIPStream)::ncclResult_t
end

@check function pncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @ccall librccl.pncclBroadcast(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                  count::Csize_t, datatype::ncclDataType_t, root::Cint,
                                  comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @ccall librccl.ncclAllReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                                 datatype::ncclDataType_t, op::ncclRedOp_t,
                                 comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @ccall librccl.pncclAllReduce(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                  count::Csize_t, datatype::ncclDataType_t, op::ncclRedOp_t,
                                  comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    @ccall librccl.ncclReduceScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                     recvcount::Csize_t, datatype::ncclDataType_t,
                                     op::ncclRedOp_t, comm::ncclComm_t,
                                     stream::HIPStream)::ncclResult_t
end

@check function pncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm,
                                   stream)
    @ccall librccl.pncclReduceScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                      recvcount::Csize_t, datatype::ncclDataType_t,
                                      op::ncclRedOp_t, comm::ncclComm_t,
                                      stream::HIPStream)::ncclResult_t
end

@check function ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @ccall librccl.ncclAllGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                 sendcount::Csize_t, datatype::ncclDataType_t,
                                 comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @ccall librccl.pncclAllGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                  sendcount::Csize_t, datatype::ncclDataType_t,
                                  comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclSend(sendbuff, count, datatype, peer, comm, stream)
    @ccall librccl.ncclSend(sendbuff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                            peer::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclSend(sendbuff, count, datatype, peer, comm, stream)
    @ccall librccl.pncclSend(sendbuff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                             peer::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclRecv(recvbuff, count, datatype, peer, comm, stream)
    @ccall librccl.ncclRecv(recvbuff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                            peer::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclRecv(recvbuff, count, datatype, peer, comm, stream)
    @ccall librccl.pncclRecv(recvbuff::Ptr{Cvoid}, count::Csize_t, datatype::ncclDataType_t,
                             peer::Cint, comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclGather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
    @ccall librccl.ncclGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                              sendcount::Csize_t, datatype::ncclDataType_t, root::Cint,
                              comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclGather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
    @ccall librccl.pncclGather(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                               sendcount::Csize_t, datatype::ncclDataType_t, root::Cint,
                               comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclScatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
    @ccall librccl.ncclScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                               recvcount::Csize_t, datatype::ncclDataType_t, root::Cint,
                               comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function pncclScatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
    @ccall librccl.pncclScatter(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid},
                                recvcount::Csize_t, datatype::ncclDataType_t, root::Cint,
                                comm::ncclComm_t, stream::HIPStream)::ncclResult_t
end

@check function ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)
    @ccall librccl.ncclAllToAll(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                                datatype::ncclDataType_t, comm::ncclComm_t,
                                stream::HIPStream)::ncclResult_t
end

@check function pncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream)
    @ccall librccl.pncclAllToAll(sendbuff::Ptr{Cvoid}, recvbuff::Ptr{Cvoid}, count::Csize_t,
                                 datatype::ncclDataType_t, comm::ncclComm_t,
                                 stream::HIPStream)::ncclResult_t
end

@check function ncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
                              datatype, comm, stream)
    @ccall librccl.ncclAllToAllv(sendbuff::Ptr{Cvoid}, sendcounts::Ptr{Csize_t},
                                 sdispls::Ptr{Csize_t}, recvbuff::Ptr{Cvoid},
                                 recvcounts::Ptr{Csize_t}, rdispls::Ptr{Csize_t},
                                 datatype::ncclDataType_t, comm::ncclComm_t,
                                 stream::HIPStream)::ncclResult_t
end

@check function pncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
                               datatype, comm, stream)
    @ccall librccl.pncclAllToAllv(sendbuff::Ptr{Cvoid}, sendcounts::Ptr{Csize_t},
                                  sdispls::Ptr{Csize_t}, recvbuff::Ptr{Cvoid},
                                  recvcounts::Ptr{Csize_t}, rdispls::Ptr{Csize_t},
                                  datatype::ncclDataType_t, comm::ncclComm_t,
                                  stream::HIPStream)::ncclResult_t
end

const mscclAlgoHandle_t = Cint

@check function mscclLoadAlgo(mscclAlgoFilePath, mscclAlgoHandle, rank)
    @ccall librccl.mscclLoadAlgo(mscclAlgoFilePath::Cstring,
                                 mscclAlgoHandle::Ptr{mscclAlgoHandle_t},
                                 rank::Cint)::ncclResult_t
end

@check function pmscclLoadAlgo(mscclAlgoFilePath, mscclAlgoHandle, rank)
    @ccall librccl.pmscclLoadAlgo(mscclAlgoFilePath::Cstring,
                                  mscclAlgoHandle::Ptr{mscclAlgoHandle_t},
                                  rank::Cint)::ncclResult_t
end

@check function mscclRunAlgo(sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
                             count, dataType, root, peer, op, mscclAlgoHandle, comm, stream)
    @ccall librccl.mscclRunAlgo(sendBuff::Ptr{Cvoid}, sendCounts::Ptr{Csize_t},
                                sDisPls::Ptr{Csize_t}, recvBuff::Ptr{Cvoid},
                                recvCounts::Ptr{Csize_t}, rDisPls::Ptr{Csize_t},
                                count::Csize_t, dataType::ncclDataType_t, root::Cint,
                                peer::Cint, op::ncclRedOp_t,
                                mscclAlgoHandle::mscclAlgoHandle_t, comm::ncclComm_t,
                                stream::HIPStream)::ncclResult_t
end

@check function pmscclRunAlgo(sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
                              count, dataType, root, peer, op, mscclAlgoHandle, comm,
                              stream)
    @ccall librccl.pmscclRunAlgo(sendBuff::Ptr{Cvoid}, sendCounts::Ptr{Csize_t},
                                 sDisPls::Ptr{Csize_t}, recvBuff::Ptr{Cvoid},
                                 recvCounts::Ptr{Csize_t}, rDisPls::Ptr{Csize_t},
                                 count::Csize_t, dataType::ncclDataType_t, root::Cint,
                                 peer::Cint, op::ncclRedOp_t,
                                 mscclAlgoHandle::mscclAlgoHandle_t, comm::ncclComm_t,
                                 stream::HIPStream)::ncclResult_t
end

@check function mscclUnloadAlgo(mscclAlgoHandle)
    @ccall librccl.mscclUnloadAlgo(mscclAlgoHandle::mscclAlgoHandle_t)::ncclResult_t
end

@check function pmscclUnloadAlgo(mscclAlgoHandle)
    @ccall librccl.pmscclUnloadAlgo(mscclAlgoHandle::mscclAlgoHandle_t)::ncclResult_t
end

# no prototype is found for this function at rccl.h:826:15, please use with caution
@check function ncclGroupStart()
    @ccall librccl.ncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at rccl.h:828:14, please use with caution
@check function pncclGroupStart()
    @ccall librccl.pncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at rccl.h:836:15, please use with caution
@check function ncclGroupEnd()
    @ccall librccl.ncclGroupEnd()::ncclResult_t
end

# no prototype is found for this function at rccl.h:838:14, please use with caution
@check function pncclGroupEnd()
    @ccall librccl.pncclGroupEnd()::ncclResult_t
end

@check function ncclGroupSimulateEnd(simInfo)
    @ccall librccl.ncclGroupSimulateEnd(simInfo::Ptr{ncclSimInfo_t})::ncclResult_t
end

@check function pncclGroupSimulateEnd(simInfo)
    @ccall librccl.pncclGroupSimulateEnd(simInfo::Ptr{ncclSimInfo_t})::ncclResult_t
end

struct var"##Ctag#234"
    level::Cuint
    layer::Cuint
    offsetX::Cuint
    offsetY::Cuint
    offsetZ::Cuint
    extentWidth::Cuint
    extentHeight::Cuint
    extentDepth::Cuint
end
function Base.getproperty(x::Ptr{var"##Ctag#234"}, f::Symbol)
    f === :level && return Ptr{Cuint}(x + 0)
    f === :layer && return Ptr{Cuint}(x + 4)
    f === :offsetX && return Ptr{Cuint}(x + 8)
    f === :offsetY && return Ptr{Cuint}(x + 12)
    f === :offsetZ && return Ptr{Cuint}(x + 16)
    f === :extentWidth && return Ptr{Cuint}(x + 20)
    f === :extentHeight && return Ptr{Cuint}(x + 24)
    f === :extentDepth && return Ptr{Cuint}(x + 28)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#234", f::Symbol)
    r = Ref{var"##Ctag#234"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#234"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#234"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#235"
    layer::Cuint
    offset::Culonglong
    size::Culonglong
end
function Base.getproperty(x::Ptr{var"##Ctag#235"}, f::Symbol)
    f === :layer && return Ptr{Cuint}(x + 0)
    f === :offset && return Ptr{Culonglong}(x + 8)
    f === :size && return Ptr{Culonglong}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#235", f::Symbol)
    r = Ref{var"##Ctag#235"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#235"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#235"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#242"
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{var"##Ctag#242"}, f::Symbol)
    f === :handle && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :name && return Ptr{Ptr{Cvoid}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#242", f::Symbol)
    r = Ref{var"##Ctag#242"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#242"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#242"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#244"
    array::hipArray_t
end
function Base.getproperty(x::Ptr{var"##Ctag#244"}, f::Symbol)
    f === :array && return Ptr{hipArray_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#244", f::Symbol)
    r = Ref{var"##Ctag#244"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#244"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#244"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#245"
    mipmap::hipMipmappedArray_t
end
function Base.getproperty(x::Ptr{var"##Ctag#245"}, f::Symbol)
    f === :mipmap && return Ptr{hipMipmappedArray_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#245", f::Symbol)
    r = Ref{var"##Ctag#245"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#245"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#245"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#246"
    devPtr::Ptr{Cvoid}
    desc::hipChannelFormatDesc
    sizeInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#246"}, f::Symbol)
    f === :devPtr && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :desc && return Ptr{hipChannelFormatDesc}(x + 8)
    f === :sizeInBytes && return Ptr{Csize_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#246", f::Symbol)
    r = Ref{var"##Ctag#246"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#246"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#246"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#247"
    devPtr::Ptr{Cvoid}
    desc::hipChannelFormatDesc
    width::Csize_t
    height::Csize_t
    pitchInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#247"}, f::Symbol)
    f === :devPtr && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :desc && return Ptr{hipChannelFormatDesc}(x + 8)
    f === :width && return Ptr{Csize_t}(x + 32)
    f === :height && return Ptr{Csize_t}(x + 40)
    f === :pitchInBytes && return Ptr{Csize_t}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#247", f::Symbol)
    r = Ref{var"##Ctag#247"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#247"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#247"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct hipStreamMemOpWaitValueParams_t
    data::NTuple{40,UInt8}
end

function Base.getproperty(x::Ptr{hipStreamMemOpWaitValueParams_t}, f::Symbol)
    f === :operation && return Ptr{hipStreamBatchMemOpType}(x + 0)
    f === :address && return Ptr{hipDeviceptr_t}(x + 8)
    f === :value && return Ptr{UInt32}(x + 16)
    f === :value64 && return Ptr{UInt64}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :alias && return Ptr{hipDeviceptr_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::hipStreamMemOpWaitValueParams_t, f::Symbol)
    r = Ref{hipStreamMemOpWaitValueParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipStreamMemOpWaitValueParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipStreamMemOpWaitValueParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipStreamMemOpWaitValueParams_t, private::Bool=false)
    return (:operation, :address, :value, :value64, :flags, :alias,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipStreamMemOpWriteValueParams_t
    data::NTuple{40,UInt8}
end

function Base.getproperty(x::Ptr{hipStreamMemOpWriteValueParams_t}, f::Symbol)
    f === :operation && return Ptr{hipStreamBatchMemOpType}(x + 0)
    f === :address && return Ptr{hipDeviceptr_t}(x + 8)
    f === :value && return Ptr{UInt32}(x + 16)
    f === :value64 && return Ptr{UInt64}(x + 16)
    f === :flags && return Ptr{Cuint}(x + 24)
    f === :alias && return Ptr{hipDeviceptr_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::hipStreamMemOpWriteValueParams_t, f::Symbol)
    r = Ref{hipStreamMemOpWriteValueParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipStreamMemOpWriteValueParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipStreamMemOpWriteValueParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipStreamMemOpWriteValueParams_t, private::Bool=false)
    return (:operation, :address, :value, :value64, :flags, :alias,
            if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipStreamMemOpFlushRemoteWritesParams_t
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipStreamMemOpFlushRemoteWritesParams_t}, f::Symbol)
    f === :operation && return Ptr{hipStreamBatchMemOpType}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::hipStreamMemOpFlushRemoteWritesParams_t, f::Symbol)
    r = Ref{hipStreamMemOpFlushRemoteWritesParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipStreamMemOpFlushRemoteWritesParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipStreamMemOpFlushRemoteWritesParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipStreamMemOpFlushRemoteWritesParams_t, private::Bool=false)
    return (:operation, :flags, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct hipStreamMemOpMemoryBarrierParams_t
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{hipStreamMemOpMemoryBarrierParams_t}, f::Symbol)
    f === :operation && return Ptr{hipStreamBatchMemOpType}(x + 0)
    f === :flags && return Ptr{Cuint}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::hipStreamMemOpMemoryBarrierParams_t, f::Symbol)
    r = Ref{hipStreamMemOpMemoryBarrierParams_t}(x)
    ptr = Base.unsafe_convert(Ptr{hipStreamMemOpMemoryBarrierParams_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{hipStreamMemOpMemoryBarrierParams_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::hipStreamMemOpMemoryBarrierParams_t, private::Bool=false)
    return (:operation, :flags, if private
                fieldnames(typeof(x))
            else
                ()
            end...)
end

struct var"##Ctag#249"
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end
function Base.getproperty(x::Ptr{var"##Ctag#249"}, f::Symbol)
    f === :handle && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :name && return Ptr{Ptr{Cvoid}}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#249", f::Symbol)
    r = Ref{var"##Ctag#249"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#249"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#249"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#251"
    hArray::hipArray_t
end
function Base.getproperty(x::Ptr{var"##Ctag#251"}, f::Symbol)
    f === :hArray && return Ptr{hipArray_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#251", f::Symbol)
    r = Ref{var"##Ctag#251"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#251"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#251"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#252"
    hMipmappedArray::hipMipmappedArray_t
end
function Base.getproperty(x::Ptr{var"##Ctag#252"}, f::Symbol)
    f === :hMipmappedArray && return Ptr{hipMipmappedArray_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#252", f::Symbol)
    r = Ref{var"##Ctag#252"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#252"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#252"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#253"
    devPtr::hipDeviceptr_t
    format::hipArray_Format
    numChannels::Cuint
    sizeInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#253"}, f::Symbol)
    f === :devPtr && return Ptr{hipDeviceptr_t}(x + 0)
    f === :format && return Ptr{hipArray_Format}(x + 8)
    f === :numChannels && return Ptr{Cuint}(x + 12)
    f === :sizeInBytes && return Ptr{Csize_t}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#253", f::Symbol)
    r = Ref{var"##Ctag#253"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#253"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#253"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#254"
    devPtr::hipDeviceptr_t
    format::hipArray_Format
    numChannels::Cuint
    width::Csize_t
    height::Csize_t
    pitchInBytes::Csize_t
end
function Base.getproperty(x::Ptr{var"##Ctag#254"}, f::Symbol)
    f === :devPtr && return Ptr{hipDeviceptr_t}(x + 0)
    f === :format && return Ptr{hipArray_Format}(x + 8)
    f === :numChannels && return Ptr{Cuint}(x + 12)
    f === :width && return Ptr{Csize_t}(x + 16)
    f === :height && return Ptr{Csize_t}(x + 24)
    f === :pitchInBytes && return Ptr{Csize_t}(x + 32)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#254", f::Symbol)
    r = Ref{var"##Ctag#254"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#254"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#254"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct var"##Ctag#255"
    reserved::NTuple{32,Cint}
end
function Base.getproperty(x::Ptr{var"##Ctag#255"}, f::Symbol)
    f === :reserved && return Ptr{NTuple{32,Cint}}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#255", f::Symbol)
    r = Ref{var"##Ctag#255"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#255"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#255"}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const HIP_VERSION_MAJOR = 6

const HIP_VERSION_MINOR = 4

const HIP_VERSION_PATCH = 43484

const HIP_VERSION_GITHASH = "123eb5128"

const HIP_VERSION_BUILD_ID = 0

const HIP_VERSION_BUILD_NAME = ""

const HIP_VERSION = HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 +
                    HIP_VERSION_PATCH

const __HIP_HAS_GET_PCH = 1

# Skipping MacroDefinition: HIP_PUBLIC_API __attribute__ ( ( visibility ( "default" ) ) )

# Skipping MacroDefinition: HIP_INTERNAL_EXPORTED_API __attribute__ ( ( visibility ( "default" ) ) )

const __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ = 0

const __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ = 0

const __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ = 0

const __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ = 0

const __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ = 0

const __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ = 0

const __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ = 0

const __HIP_ARCH_HAS_DOUBLES__ = 0

const __HIP_ARCH_HAS_WARP_VOTE__ = 0

const __HIP_ARCH_HAS_WARP_BALLOT__ = 0

const __HIP_ARCH_HAS_WARP_SHUFFLE__ = 0

const __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ = 0

const __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ = 0

const __HIP_ARCH_HAS_SYNC_THREAD_EXT__ = 0

const __HIP_ARCH_HAS_SURFACE_FUNCS__ = 0

const __HIP_ARCH_HAS_3DGRID__ = 0

const __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ = 0

const __HIP_CLANG_ONLY__ = 0

const hipGetDeviceProperties = hipGetDevicePropertiesR0600

const hipDeviceProp_t = hipDeviceProp_tR0600

const hipChooseDevice = hipChooseDeviceR0600

const GENERIC_GRID_LAUNCH = 1

# Skipping MacroDefinition: __forceinline__ inline

const HIP_TRSA_OVERRIDE_FORMAT = 0x01

const HIP_TRSF_READ_AS_INTEGER = 0x01

const HIP_TRSF_NORMALIZED_COORDINATES = 0x02

const HIP_TRSF_SRGB = 0x10

const __HOST_DEVICE__ = __host__(__device__)

const __HIP_USE_NATIVE_VECTOR__ = 1

const hipTextureType1D = 0x01

const hipTextureType2D = 0x02

const hipTextureType3D = 0x03

const hipTextureTypeCubemap = 0x0c

const hipTextureType1DLayered = 0xf1

const hipTextureType2DLayered = 0xf2

const hipTextureTypeCubemapLayered = 0xfc

const HIP_IMAGE_OBJECT_SIZE_DWORD = 12

const HIP_SAMPLER_OBJECT_SIZE_DWORD = 8

const HIP_SAMPLER_OBJECT_OFFSET_DWORD = HIP_IMAGE_OBJECT_SIZE_DWORD

const HIP_TEXTURE_OBJECT_SIZE_DWORD = HIP_IMAGE_OBJECT_SIZE_DWORD +
                                      HIP_SAMPLER_OBJECT_SIZE_DWORD

const HIP_DEPRECATED_MSG = "This API is marked as deprecated and might not be supported in future releases. For more details please refer https://github.com/ROCm/HIP/blob/develop/docs/reference/deprecated_api_list.md"

# Skipping MacroDefinition: HIP_LAUNCH_PARAM_BUFFER_POINTER ( ( void * ) 0x01 )

# Skipping MacroDefinition: HIP_LAUNCH_PARAM_BUFFER_SIZE ( ( void * ) 0x02 )

# Skipping MacroDefinition: HIP_LAUNCH_PARAM_END ( ( void * ) 0x03 )

const hipIpcMemLazyEnablePeerAccess = 0x01

const HIP_IPC_HANDLE_SIZE = 64

const hipStreamDefault = 0x00

const hipStreamNonBlocking = 0x01

const hipEventDefault = 0x00

const hipEventBlockingSync = 0x01

const hipEventDisableTiming = 0x02

const hipEventInterprocess = 0x04

const hipEventRecordDefault = 0x00

const hipEventRecordExternal = 0x01

const hipEventWaitDefault = 0x00

const hipEventWaitExternal = 0x01

const hipEventDisableSystemFence = 0x20000000

const hipEventReleaseToDevice = 0x40000000

const hipEventReleaseToSystem = 0x80000000

const hipHostAllocDefault = 0x00

const hipHostMallocDefault = 0x00

const hipHostAllocPortable = 0x01

const hipHostMallocPortable = 0x01

const hipHostAllocMapped = 0x02

const hipHostMallocMapped = 0x02

const hipHostAllocWriteCombined = 0x04

const hipHostMallocWriteCombined = 0x04

const hipHostMallocNumaUser = 0x20000000

const hipHostMallocCoherent = 0x40000000

const hipHostMallocNonCoherent = 0x80000000

const hipMemAttachGlobal = 0x01

const hipMemAttachHost = 0x02

const hipMemAttachSingle = 0x04

const hipDeviceMallocDefault = 0x00

const hipDeviceMallocFinegrained = 0x01

const hipMallocSignalMemory = 0x02

const hipDeviceMallocUncached = 0x03

const hipDeviceMallocContiguous = 0x04

const hipHostRegisterDefault = 0x00

const hipHostRegisterPortable = 0x01

const hipHostRegisterMapped = 0x02

const hipHostRegisterIoMemory = 0x04

const hipHostRegisterReadOnly = 0x08

const hipExtHostRegisterCoarseGrained = 0x08

const hipDeviceScheduleAuto = 0x00

const hipDeviceScheduleSpin = 0x01

const hipDeviceScheduleYield = 0x02

const hipDeviceScheduleBlockingSync = 0x04

const hipDeviceScheduleMask = 0x07

const hipDeviceMapHost = 0x08

const hipDeviceLmemResizeToMax = 0x10

const hipArrayDefault = 0x00

const hipArrayLayered = 0x01

const hipArraySurfaceLoadStore = 0x02

const hipArrayCubemap = 0x04

const hipArrayTextureGather = 0x08

const hipOccupancyDefault = 0x00

const hipOccupancyDisableCachingOverride = 0x01

const hipCooperativeLaunchMultiDeviceNoPreSync = 0x01

const hipCooperativeLaunchMultiDeviceNoPostSync = 0x02

const hipCpuDeviceId = Cint - 1

const hipInvalidDeviceId = Cint - 2

const hipExtAnyOrderLaunch = 0x01

const hipStreamWaitValueGte = 0x00

const hipStreamWaitValueEq = 0x01

const hipStreamWaitValueAnd = 0x02

const hipStreamWaitValueNor = 0x03

const hipStreamPerThread = hipStream_t(2)

const hipStreamLegacy = hipStream_t(1)

const hipExternalMemoryDedicated = 0x01

const hipKernelNodeAttrID = hipLaunchAttributeID

const hipKernelNodeAttributeAccessPolicyWindow = hipLaunchAttributeAccessPolicyWindow

const hipKernelNodeAttributeCooperative = hipLaunchAttributeCooperative

const hipKernelNodeAttributePriority = hipLaunchAttributePriority

const hipKernelNodeAttrValue = hipLaunchAttributeValue

const hipGraphKernelNodePortDefault = 0

const hipGraphKernelNodePortLaunchCompletion = 2

const hipGraphKernelNodePortProgrammatic = 1

const USE_PEER_NON_UNIFIED = 1

const HIPRT_INF_FP16 = __ushort_as_half(Cushort(Cuint(0x7c00)))

const HIPRT_MAX_NORMAL_FP16 = __ushort_as_half(Cushort(Cuint(0x7bff)))

const HIPRT_MIN_DENORM_FP16 = __ushort_as_half(Cushort(Cuint(0x0001)))

const HIPRT_NAN_FP16 = __ushort_as_half(Cushort(Cuint(0x7fff)))

const HIPRT_NEG_ZERO_FP16 = __ushort_as_half(Cushort(Cuint(0x8000)))

const HIPRT_ONE_FP16 = __ushort_as_half(Cushort(Cuint(0x3c00)))

const HIPRT_ZERO_FP16 = __ushort_as_half(Cushort(Cuint(0x0000)))

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
const PREFIXES = ["rccl"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
