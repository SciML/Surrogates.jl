module Surrogates
using LinearAlgebra
using GLM
using Distributions
using Sobol
using LatinHypercubeSampling
using Requires

abstract type AbstractSurrogate <: Function end
include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")
include("Optimization.jl")
include("Lobachesky.jl")
include("LinearSurrogate.jl")
include("InverseDistanceSurrogate.jl")
include("SecondOrderPolynomialSurrogate.jl")

remove_tracker(x) = x

function __init__()
    @require XGBoost="009559a3-9522-5dbb-924b-0b6ed2b22bb9" begin
        using XGBoost
        include("RandomForestSurrogate.jl")
    end

    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        using Flux
        using Flux: @epochs
        include("NeuralSurrogate.jl")
    end

    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        remove_tracker(x::TrackedReal) = Tracker.data(x)
        remove_tracker(x::TrackedArray) = Tracker.data(x)
    end

    @require LIBSVM="b1bec4e5-fd48-53fe-b0cb-9723c09d164b" begin
        using LIBSVM
        include("SVMSurrogate.jl")
    end

    @require Stheno = "8188c328-b5d6-583d-959b-9690869a5511" begin
        using Stheno
        include("SthenoKriging.jl")
    end
end



export AbstractSurrogate, SamplingAlgorithm
export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample, LowDiscrepancySample
export RandomSample
export SRBF,LCBS,EI,DYCORS,SOP,surrogate_optimize
export LobacheskySurrogate, lobachesky_integral, lobachesky_integrate_dimension
export LinearSurrogate
export RandomForestSurrogate
export SVMSurrogate
export NeuralSurrogate
export InverseDistanceSurrogate
export SecondOrderPolynomialSurrogate
end
