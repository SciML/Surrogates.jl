using QuasiMonteCarlo
using QuasiMonteCarlo: SamplingAlgorithm

# We need to convert the matrix that QuasiMonteCarlo produces into a vector of Tuples like Surrogates expects
function sample(args...; kwargs...)
    s = QuasiMonteCarlo.sample(args...; kwargs...)
    if s isa Vector
        # 1D case: s is a Vector
        return s
    else
        # ND case: s is a d x n matrix, where d is the dimension and n is the number of samples
        return reinterpret(reshape, NTuple{size(s, 1), eltype(s)}, s)
    end
end

struct KroneckerSample{A,B} <: QuasiMonteCarlo.SamplingAlgorithm
    alpha::A
    s0::B
end

struct GoldenSample <: QuasiMonteCarlo.SamplingAlgorithm end

struct SectionSample{T} <: QuasiMonteCarlo.SamplingAlgorithm
    x0::Vector{T}
    sa::QuasiMonteCarlo.SamplingAlgorithm
end

"""
sample(n,d,K::KroneckerSample)

Returns a Tuple containing numbers following the Kronecker sample
"""
function QuasiMonteCarlo.sample(n,lb,ub,K::KroneckerSample)
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

        y = collect(x')
        return y
    end
end

function QuasiMonteCarlo.sample(n,lb,ub, G::GoldenSample)
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
        y = collect(x')
        return y
    end
end

fixed_dimensions(
        section_sampler::SectionSample)::Vector{Int64} = findall(
    x->x == false, isnan.(section_sampler.x0))

free_dimensions(
        section_sampler::SectionSample)::Vector{Int64} = findall(
    x->x == true, isnan.(section_sampler.x0))

"""
sample(n,lb,ub,K::SectionSample)

Returns Tuples constrained to a section.

In surrogate-based identification and control, optimization can alternate between unconstrained sampling in the full-dimensional parameter space, and sampling constrained on specific sections (e.g. a planes in a 3D volume),

A SectionSampler allows sampling and optimizing on a subset of 'free' dimensions while keeping 'fixed' ones constrained.
The sampler is defined as in e.g.

`section_sampler_y_is_10 = SectionSample([NaN64, NaN64, 10.0, 10.0], Surrogates.UniformSample())`

where the first argument is a Vector{T} in which numbers are fixed coordinates and `NaN`s correspond to free dimensions, and the second argument is a SamplingAlgorithm which is used to sample in the free dimensions.
"""
function QuasiMonteCarlo.sample(n,lb,ub,section_sampler::SectionSample)
    if lb isa Number
        if isnan(section_sampler.x0[1])
            return sample(n, lb, ub, section_sampler.sa)
        else
            return fill(section_sampler.x0[1], n)
        end
    else
        d_free = Surrogates.free_dimensions(section_sampler)
        new_samples = sample(n, lb[d_free], ub[d_free], section_sampler.sa)
        out_as_vec = collect(repeat(section_sampler.x0', n, 1)')
        for y in 1:size(out_as_vec,2)
            for (xi, d) in enumerate(d_free)
                out_as_vec[d,y] = new_samples[y][xi]
            end
        end
        return out_as_vec
    end
end