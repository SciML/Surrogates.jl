module Surrogates
using LinearAlgebra
using Distributions

abstract type AbstractSurrogate <: Function end
include("utils.jl")
include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")
include("Optimization.jl")
include("Lobachevsky.jl")
include("LinearSurrogate.jl")
include("InverseDistanceSurrogate.jl")
include("SecondOrderPolynomialSurrogate.jl")
include("Wendland.jl")
include("MOE.jl") #rewrite gaussian mixture with own algorithm to fix deps issue
include("VariableFidelity.jl")
include("Earth.jl")
include("GEK.jl")
include("GEKPLS.jl")

current_surrogates = ["Kriging", "LinearSurrogate", "LobachevskySurrogate",
    "NeuralSurrogate",
    "RadialBasis", "RandomForestSurrogate", "SecondOrderPolynomialSurrogate",
    "Wendland", "GEK", "PolynomialChaosSurrogate"]

#Radial structure:
function RadialBasisStructure(; radial_function, scale_factor, sparse)
    return (name = "RadialBasis", radial_function = radial_function,
            scale_factor = scale_factor, sparse = sparse)
end

#Kriging structure:
function KrigingStructure(; p, theta)
    return (name = "Kriging", p = p, theta = theta)
end

function GEKStructure(; p, theta)
    return (name = "GEK", p = p, theta = theta)
end

#Linear structure
function LinearStructure()
    return (name = "LinearSurrogate")
end

#InverseDistance structure
function InverseDistanceStructure(; p)
    return (name = "InverseDistanceSurrogate", p = p)
end

#Lobachevsky structure
function LobachevskyStructure(; alpha, n, sparse)
    return (name = "LobachevskySurrogate", alpha = alpha, n = n, sparse = sparse)
end

#Neural structure
function NeuralStructure(; model, loss, opt, n_echos)
    return (name = "NeuralSurrogate", model = model, loss = loss, opt = opt,
            n_echos = n_echos)
end

#Random forest structure
function RandomForestStructure(; num_round)
    return (name = "RandomForestSurrogate", num_round = num_round)
end

#Second order poly structure
function SecondOrderPolynomialStructure()
    return (name = "SecondOrderPolynomialSurrogate")
end

#Wendland structure
function WendlandStructure(; eps, maxiters, tol)
    return (name = "Wendland", eps = eps, maxiters = maxiters, tol = tol)
end

#Polychaos structure
function PolyChaosStructure(; op)
    return (name = "PolynomialChaosSurrogate", op = op)
end

export current_surrogates
export GEKPLS
export RadialBasisStructure, KrigingStructure, LinearStructure, InverseDistanceStructure
export LobachevskyStructure, NeuralStructure, RandomForestStructure,
       SecondOrderPolynomialStructure
export WendlandStructure
export AbstractSurrogate, SamplingAlgorithm
export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
# radial basis functions
export linearRadial, cubicRadial, multiquadricRadial, thinplateRadial

# samplers
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample,
       LowDiscrepancySample
export RandomSample, KroneckerSample, GoldenSample, SectionSample

# Optimization algorithms
export SRBF, LCBS, EI, DYCORS, SOP, EGO, RTEA, SMB, surrogate_optimize
export LobachevskySurrogate, lobachevsky_integral, lobachevsky_integrate_dimension
export LinearSurrogate
export SVMSurrogate
export InverseDistanceSurrogate
export SecondOrderPolynomialSurrogate
export Wendland
export RadialBasisStructure, KrigingStructure, LinearStructure, InverseDistanceStructure
export LobachevskyStructure, NeuralStructure, RandomForestStructure,
       SecondOrderPolynomialStructure
export WendlandStructure
#export MOE
export VariableFidelitySurrogate
export EarthSurrogate
export GEK
end
