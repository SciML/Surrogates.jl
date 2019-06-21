module Surrogates

using LinearAlgebra
using Distributions
using Sobol
using LatinHypercubeSampling

abstract type AbstractSurrogate <: Function end

include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")

export Kriging, RadialBasis, add_point!, current_estimate
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample

end
