module RCCLLoader

using Libdl
export librccl

# Ok, let's find out where ROCm is 
const DEFAULT_ROCM_PATH = Ref{String}("/opt/rocm")

# Well, the user can point us to where it is 
if haskey(ENV, "ROCM_PATH")
    DEFAULT_ROCM_PATH[] = ENV["ROCM_PATH"]
end

const LIB_NAME = "librccl.so"
global const librccl::String = ""

"""
    librccl_path -> Union{String, nothing}

Return the absolute path to `librccl.so` if it can be found. We look inside:
1. `\$RCCL_LIBRARY_PATH` if it exists
2. `\$ROCM_PATH/lib` 
3. `\$ROCM_PATH/lib64`
4. Let the system figure it out from `\$LD_LIBRARY_PATH`. 
"""
function librccl_path()

    # 1. RCCL_LIBRARY_PATH
    if haskey(ENV, "RCCL_LIBRARY_PATH")
        p = ENV["RCCL_LIBRARY_PATH"]
        isfile(p) && (return p)
    end

    # 2. Search inside ROCM_PATH 
    roots = [DEFAULT_ROCM_PATH[], joinpath(DEFAULT_ROCM_PATH[], "lib"), joinpath(DEFAULT_ROCM_PATH[], "lib64")]
    for r in roots
        candidate = joinpath(r, LIB_NAME)
        if isfile(candidate)
            return candidate
        end
    end

    # 3. Duck tape and prayers - Let the system figure it out 
    return LIB_NAME
end

"""
    is_available -> Bool 

Return `true` if I can load `librccl.so`, otherwise `false`. 
We also cache the result so that calling this in the same Julia session is quick.
"""
function is_available()
    lib = librccl_path()
    try
        handle = Libdl.dlopen(lib)
        Libdl.dlclose(handle)
        return true
    catch err
        @debug "RCCL library not found" exception = (err, catch_backtrace())
        return false
    end
end

function __init__()
    if is_available()
        global librccl = librccl_path()
    else
        @error "Cannot find librccl.so"
    end
end

end
