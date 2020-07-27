module Surrogates
using LinearAlgebra
using Distributions

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
include("Wendland.jl")
include("MOE.jl")
include("VariableFidelity.jl")
include("PolynomialChaos.jl")
include("Earth.jl")
include("GEK.jl")

current_surrogates = ["Kriging","LinearSurrogate","LobacheskySurrogate","NeuralSurrogate",
                      "RadialBasis","RandomForestSurrogate","SecondOrderPolynomialSurrogate",
                      "Wendland","GEK","PolynomialChaosSurrogate"]
export current_surrogates_MOE
export AbstractSurrogate, SamplingAlgorithm
export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
export linearRadial,cubicRadial,multiquadricRadial,thinplateRadial
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
export Wendland
export RadialBasisStructure, KrigingStructure, LinearStructure, InverseDistanceStructure
export LobacheskyStructure, NeuralStructure, RandomForestStructure, SecondOrderPolynomialStructure
export WendlandStructure
export MOE
export VariableFidelitySurrogate
export PolynomialChaosSurrogate
export EarthSurrogate
export GEK
end
