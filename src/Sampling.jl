using QuasiMonteCarlo
using QuasiMonteCarlo: SamplingAlgorithm

# We need to convert the matrix that QuasiMonteCarlo produces into a vector of Tuples like Surrogates expects
# This will eventually be removed once we refactor the rest of the code to work with d x n matrices instead
# of vectors of Tuples
function sample(args...; kwargs...)
    s = QuasiMonteCarlo.sample(args...; kwargs...)
    if s isa Vector
        # 1D case: s is a Vector
        return s
    else
        # ND case: s is a d x n matrix, where d is the dimension and n is the number of samples
        return collect(reinterpret(reshape, NTuple{size(s, 1), eltype(s)}, s))
    end
end