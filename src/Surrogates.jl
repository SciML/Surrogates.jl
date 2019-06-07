module Surrogates

using LinearAlgebra

abstract type AbstractSurrogate <: Function end

include("Radials.jl")
include("Kriging.jl")

export Kriging, RadialBasis, add_point!, current_estimate

end
