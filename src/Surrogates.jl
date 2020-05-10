module Surrogates
using LinearAlgebra
using GLM
using Distributions
using Sobol
using LatinHypercubeSampling
using Stheno

abstract type AbstractSurrogate <: Function end
include("utils.jl")
include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")
include("Optimization.jl")
include("Lobachesky.jl")
include("LinearSurrogate.jl")
include("InverseDistanceSurrogate.jl")
include("SecondOrderPolynomialSurrogate.jl")
include("SthenoKriging.jl")
include("RandomForestSurrogate.jl")
include("NeuralSurrogate.jl")

export AbstractSurrogate, SamplingAlgorithm
export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample, LowDiscrepancySample
export RandomSample, KroneckerSample, GoldenSample
export SRBF,LCBS,EI,DYCORS,SOP,surrogate_optimize
export LobacheskySurrogate, lobachesky_integral, lobachesky_integrate_dimension
export LinearSurrogate
export RandomForestSurrogate
export SVMSurrogate
export NeuralSurrogate
export InverseDistanceSurrogate
export SecondOrderPolynomialSurrogate
export SthenoKriging
end
