module Surrogates

using LinearAlgebra

abstract type AbstractSurrogate <: Function end

include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")

export Kriging,std_error_at_point,RadialBasis, add_point!, current_estimate,
       sample, random_sample,uniform_sample, sobol_sample,
       LHS_sample

end
