using QuasiMonteCarlo
using QuasiMonteCarlo: SamplingAlgorithm

# We need to convert the matrix that QuasiMonteCarlo produces into a vector of Tuples like Surrogates expects
# This will eventually be removed once we refactor the rest of the code to work with d x n matrices instead
# of vectors of Tuples
function sample(args...; kwargs...)
    s = QuasiMonteCarlo.sample(args...; kwargs...)
    if isone(size(s, 1))
        # 1D case: s is a Vector
        return vec(s)
    else
        # ND case: s is a d x n matrix, where d is the dimension and n is the number of samples
        return collect(reinterpret(reshape, NTuple{size(s, 1), eltype(s)}, s))
    end
end

#### SectionSample #### 
"""
    SectionSample{T}(x0, sa)
`SectionSample(x0, sampler)` where `sampler` is any sampler above and `x0` is a vector of either `NaN` for a free dimension or some scalar for a constrained dimension.
"""
struct SectionSample{
    R <: Real,
    I <: Integer,
    VR <: AbstractVector{R},
    VI <: AbstractVector{I},
} <: SamplingAlgorithm
    x0::VR
    sa::SamplingAlgorithm
    fixed_dims::VI
end
fixed_dimensions(section_sampler::SectionSample)::Vector{Int64} = findall(x -> x == false,
    isnan.(section_sampler.x0))
free_dimensions(section_sampler::SectionSample)::Vector{Int64} = findall(x -> x == true,
    isnan.(section_sampler.x0))
"""
    sample(n,lb,ub,K::SectionSample)
Returns Tuples constrained to a section.
In surrogate-based identification and control, optimization can alternate between unconstrained sampling in the full-dimensional parameter space, and sampling constrained on specific sections (e.g. a planes in a 3D volume),
A SectionSample allows sampling and optimizing on a subset of 'free' dimensions while keeping 'fixed' ones constrained.
The sampler is defined as in e.g.
`section_sampler_y_is_10 = SectionSample([NaN64, NaN64, 10.0, 10.0], UniformSample())`
where the first argument is a Vector{T} in which numbers are fixed coordinates and `NaN`s correspond to free dimensions, and the second argument is a SamplingAlgorithm which is used to sample in the free dimensions.
"""
function sample(n::Integer,
        lb::T,
        ub::T,
        section_sampler::SectionSample) where {
        T <: Union{Base.AbstractVecOrTuple, Number}}
    @assert n>0 ZERO_SAMPLES_MESSAGE
    QuasiMonteCarlo._check_sequence(lb, ub, length(lb))
    if lb isa Number
        if isnan(section_sampler.x0[1])
            return vec(sample(n, lb, ub, section_sampler.sa))
        else
            return fill(section_sampler.x0[1], n)
        end
    else
        d_free = free_dimensions(section_sampler)
        @info d_free
        new_samples = QuasiMonteCarlo.sample(n, lb[d_free], ub[d_free], section_sampler.sa)
        out_as_vec = collect(repeat(section_sampler.x0', n, 1)')

        for y in 1:size(out_as_vec, 2)
            for (xi, d) in enumerate(d_free)
                out_as_vec[d, y] = new_samples[xi, y]
            end
        end
        return isone(size(out_as_vec, 1)) ? vec(out_as_vec) :
               collect(reinterpret(reshape,
            NTuple{size(out_as_vec, 1), eltype(out_as_vec)},
            out_as_vec))
    end
end

function SectionSample(x0::AbstractVector, sa::SamplingAlgorithm)
    SectionSample(x0, sa, findall(isnan, x0))
end

"""
    SectionSample(n, d, K::SectionSample)
In surrogate-based identification and control, optimization can alternate between unconstrained sampling in the full-dimensional parameter space, and sampling constrained on specific sections (e.g. planes in a 3D volume).
`SectionSample` allows sampling and optimizing on a subset of 'free' dimensions while keeping 'fixed' ones constrained.
The sampler is defined
`SectionSample([NaN64, NaN64, 10.0, 10.0], UniformSample())`
where the first argument is a Vector{T} in which numbers are fixed coordinates and `NaN`s correspond to free dimensions, and the second argument is a SamplingAlgorithm which is used to sample in the free dimensions.
"""
function sample(n::Integer,
        d::Integer,
        section_sampler::SectionSample,
        T = eltype(section_sampler.x0))
    QuasiMonteCarlo._check_sequence(n)
    @assert eltype(section_sampler.x0) == T
    @assert length(section_sampler.fixed_dims) == d
    return sample(n, section_sampler)
end

@views function sample(n::Integer, section_sampler::SectionSample{T}) where {T}
    samples = Matrix{T}(undef, n, length(section_sampler.x0))
    fixed_dims = section_sampler.fixed_dims
    samples[:, fixed_dims] .= sample(n, length(fixed_dims), section_sampler.sa, T)
    return vec(samples)
end
