using LinearAlgebra
using Distributions
using Sobol
using LatinHypercubeSampling
#=
1) Random sampling
2) Uniform sampling
3) Sobol sampling
4) Latin Hypercube
5) Sparse grid sampling
6) Monte carlo importance sampling
7) Low- discrepancy sampling
=#

"""
sample(f,kind_of_sample()) returns the evaluation of function f
and the sampling points, ready to be used in Kriging and Radials.
"""
function sample(f::Function,sample)
    n = Base.size(sample,1)
    vals = zeros(eltype(sample[1]),1,n)
    for i = 1:n
        vals[i] = f(sample[i])
    end
    return vec(vals),sample
end

"""
random_sample(n,d,bounds) returns a nxd Array containing
random numbers
"""
function random_sample(n,d,bounds)
    return rand(bounds[1]:0.0000001:bounds[2],n,d)
end

"""
uniform_sample
"""
function uniform_sample(n,d,bounds)
    return rand(Uniform(bounds[1],bounds[2]),n,d)
end

"""
Sobol
"""
function sobol_sample(n,lb,ub)
    x = Tuple[]
    s = SobolSeq(lb,ub)
    skip(s,n)
    for i = 1:n
        push!(x,Tuple(next!(s)))
    end
    if length(lb) == 1
        flat(arr::Array) = mapreduce(x -> isa(x, Array) ? flat(x) : x, append!, arr,init=[])
        return flat(x)
    else
        return x
    end
end

"""
Latin hypercube sapling
"""
function LHS_sample(n,d)
    return LHCoptim(n,d,1)[1]
end
