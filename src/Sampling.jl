abstract type SamplingAlgorithm end

"""
GridSample{T}

T is the step dx for lb:dx:ub
"""
struct GridSample{T} <: SamplingAlgorithm
    dx::T
end

struct UniformSample <: SamplingAlgorithm end
struct SobolSample <: SamplingAlgorithm end
struct LatinHypercubeSample <: SamplingAlgorithm end

"""
LowDiscrepancySample{T}

T is the base for the sequence
"""
struct LowDiscrepancySample{T} <: SamplingAlgorithm
    base::T
end

struct RandomSample <: SamplingAlgorithm end

struct KroneckerSample{A,B} <: SamplingAlgorithm
    alpha::A
    s0::B
end

struct GoldenSample <: SamplingAlgorithm end


"""
sample(n,lb,ub,S::GridSample)

Returns a tuple containing numbers in a grid.

"""
function sample(n,lb,ub,S::GridSample)
    dx = S.dx
    if lb isa Number
        return vec(rand(lb:S.dx:ub,n))
    else
        d = length(lb)
        x = [[rand(lb[j]:dx[j]:ub[j]) for j = 1:d] for i in 1:n]
        return Tuple.(x)
    end
end

"""
sample(n,lb,ub,::UniformRandom)

Returns a Tuple containing uniform random numbers.
"""
function sample(n,lb,ub,::UniformSample)
    if lb isa Number
        return rand(Uniform(lb,ub),n)
    else
        d = length(lb)
        x = [[rand(Uniform(lb[j],ub[j])) for j in 1:d] for i in 1:n]
        return Tuple.(x)
    end
end

"""
sample(n,lb,ub,::SobolSampling)

Returns a Tuple containing Sobol sequences.
"""
function sample(n,lb,ub,::SobolSample)
    s = SobolSeq(lb,ub)
    skip(s,n)
    if lb isa Number
        return [next!(s)[1] for i = 1:n]
    else
        return Tuple.([next!(s) for i = 1:n])
    end
end

"""
sample(n,lb,ub,::LatinHypercube)

Returns a Tuple containing LatinHypercube sequences.
"""
function sample(n,lb,ub,::LatinHypercubeSample)
    d = length(lb)
    if lb isa Number
        x = vec(LHCoptim(n,d,1)[1])
        # x∈[0,n], so affine transform
        return @. (ub-lb) * x/(n) + lb
    else
        lib_out = float(LHCoptim(n,d,1)[1])
        # x∈[0,n], so affine transform column-wise
        @inbounds for c = 1:d
            lib_out[:,c] = (ub[c]-lb[c])*lib_out[:,c]/n .+ lb[c]
        end
        x = [lib_out[i,:] for i = 1:n]
        return Tuple.(x)
    end
end


"""
sample(n,lb,ub,S::LowDiscrepancySample)

Low discrepancy sample:
- Dimension 1: Van der Corput sequence
- Dimension > 1: Halton sequence
If dimension d > 1, all bases must be coprime with each other.
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

        y = [x[i,:] for i = 1:n]
        return Tuple.(y)
    end
end

"""
sample(n,d,D::Distribution)

Returns a Tuple containing numbers distributed as D
"""
function sample(n,d,D::Distribution)
    if d == 1
        return rand(D,n)
    else
        x = [[rand(D) for j in 1:d] for i in 1:n]
        return Tuple.(x)
    end
end


"""
sample(n,d,K::KroneckerSample)

Returns a Tuple containing numbers following the Kronecker sample
"""
function sample(n,lb,ub,K::KroneckerSample)
    d = length(lb)
    alpha = K.alpha
    s0 = K.s0
    if d == 1
        x = zeros(n)
        @inbounds for i = 1:n
            x[i] = (s0+i*alpha)%1
        end
        return @. (ub-lb) * x + lb
    else
        x = zeros(n,d)
        @inbounds for j = 1:d
            for i = 1:n
                x[i,j] = (s0[j] + i*alpha[j])%i
            end
        end
        #Resizing
        # x∈[0,1], so affine transform column-wise
        @inbounds for c = 1:d
            x[:,c] = (ub[c]-lb[c])*x[:,c] .+ lb[c]
        end

        y = [x[i,:] for i = 1:n]
        return Tuple.(y)
    end
end

function sample(n,lb,ub,G::GoldenSample)
    d = length(lb)
    if d == 1
        x = zeros(n)
        g = (sqrt(5)+1)/2
        a = 1.0/g
        for i = 1:n
            x[i] = (0.5+a*i)%1
        end
        return @. (ub-lb) * x + lb
    else
        x = zeros(n,d)
        for j = 1:d
            #Approximate solution of x^(d+1) = x + 1, a simple newton is good enough
            y = 2.0
            for s = 1:10
                g = (1+y)^(1/(j+1))
            end
            a = 1.0/g
            for i = 1:n
                x[i,j] = (0.5+a*i)%1
            end
        end
        @inbounds for c = 1:d
            x[:,c] = (ub[c]-lb[c])*x[:,c] .+ lb[c]
        end
        y = [x[i,:] for i = 1:n]
        return Tuple.(y)
    end
end
