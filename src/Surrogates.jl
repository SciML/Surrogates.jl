module Surrogates

using LinearAlgebra
using DataFrames
using GLM
using Distributions
using Sobol
using LatinHypercubeSampling
using XGBoost
using LIBSVM
using Flux

abstract type AbstractSurrogate <: Function end
include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")
include("Optimization.jl")
include("Lobachesky.jl")
include("LinearSurrogate.jl")
include("RandomForestSurrogate.jl")
include("SVMSurrogate.jl")
include("NeuralSurrogate.jl")
include("InverseDistanceSurrogate.jl")
include("SecondOrderPolynomialSurrogate.jl")

export AbstractSurrogate, SamplingAlgorithm
export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample, LowDiscrepancySample
export SRBF,LCBS,EI,DYCORS,surrogate_optimize
export LobacheskySurrogate, lobachesky_integral, lobachesky_integrate_dimension
export LinearSurrogate
export RandomForestSurrogate
export SVMSurrogate
export NeuralSurrogate
export InverseDistanceSurrogate
export SecondOrderPolynomialSurrogate
end
