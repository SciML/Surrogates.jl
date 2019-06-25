abstract type SamplingAlgorithm end
struct GridSample{T} <: SamplingAlgorithm
    dx::T
end
struct UniformSample <: SamplingAlgorithm end
struct SobolSample <: SamplingAlgorithm end
struct LatinHypercubeSample <: SamplingAlgorithm end
struct LowDiscrepancySample{T} <: SamplingAlgorithm
    base::T
end

function sample(n,lb,ub,S::GridSample)
    dx = S.dx
    if lb isa Number
        return vec(rand(lb:S.dx:ub,n))
    else
        d = length(lb)
        x = [[rand(lb[j]:dx[j]:ub[j]) for j = 1:d] for i in 1:n]
        return x
    end
end

"""
sample(n,lb,ub,::UniformRandom)
returns a nxd Array containing uniform random numbers
"""
function sample(n,lb,ub,::UniformSample)
    if lb isa Number
        return rand(Uniform(lb,ub),n)
    else
        d = length(lb)
        x = [[rand(Uniform(lb[j],ub[j])) for j in 1:d] for i in 1:n]
        return x
    end
end

"""
sample(n,lb,ub,::SobolSampling)

Sobol
"""
function sample(n,lb,ub,::SobolSample)
    s = SobolSeq(lb,ub)
    skip(s,n)
    if lb isa Number
        return [next!(s)[1] for i = 1:n]
    else
        return [next!(s) for i = 1:n]
    end
end

"""
sample(n,lb,ub,::LatinHypercube)

Latin hypercube sapling
"""
function sample(n,lb,ub,::LatinHypercubeSample)
    d = length(lb)
    if lb isa Number
        x = vec(LHCoptim(n,d,1)[1])
        # x∈[0,n], so affine transform
        return @. (ub-lb) * x/(n) + lb
    else
        x = float(LHCoptim(n,d,1)[1])
        # x∈[0,n], so affine transform column-wise
        @inbounds for c = 1:d
            x[:,c] = (ub[c]-lb[c])*x[:,c]/n .+ lb[c]
        end
        return x
    end
end


"""
sample(n,lb,ub,S::LowDiscrepancySample)

LowDiscrepancySample for different bases.
If dimension d > 1, every bases must be coprime with each other.
"""
function sample(n,lb,ub,S::LowDiscrepancySample)
    d = length(lb)
    if d == 1
        #Van der Corput
        b = S.base
        x = zeros(Float32,n)
        for i = 1:n
            expansion = digits(i,base = b)
            L = length(expansion)
            val = zero(Float32)
            for k = 1:L
                val += expansion[k]*float(b)^(-(k-1)-1)
            end
            x[i] = val
        end
        # It is always defined on the unit interval, resizing:
        return @. (ub-lb) * x + lb
    else
        #Halton sequence
        x = zeros(Float32,n,d)
        for j = 1:d
            b = S.base[j]
            for i = 1:n
                val = zero(Float32)
                expansion = digits(i, base = b)
                L = length(expansion)
                val = zero(Float32)
                for k = 1:L
                    val += expansion[k]*float(b)^(-(k-1)-1)
                end
                x[i,j] = val
            end
        end
        #Resizing
        # x∈[0,1], so affine transform column-wise
        @inbounds for c = 1:d
            x[:,c] = (ub[c]-lb[c])*x[:,c] .+ lb[c]
        end
        return x
    end
end
