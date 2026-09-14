module Surrogates

using Distributions: Normal, cdf, pdf, truncated
using ExtendableSparse: ExtendableSparseMatrix
using IterativeSolvers: cg
using CommonSolve: solve
using LinearAlgebra: Cholesky, ColumnNorm, Diagonal, I, LAPACKException,
    PosDefException, SingularException, Symmetric, cholesky, diag, dot, eigvals,
    issuccess, logdet, norm, pinv, qr, rank, ⋅
using OptimizationOptimJL: NelderMead
using SciMLBase: OptimizationProblem
using PrecompileTools: @compile_workload, @setup_workload
using QuasiMonteCarlo: GoldenSample, GridSample, HaltonSample, KroneckerSample,
    LatinHypercubeSample, RandomSample, SamplingAlgorithm, SobolSample
using Statistics: mean, std
using SurrogatesBase: update!, AbstractDeterministicSurrogate,
    AbstractStochasticSurrogate, parameters, hyperparameters,
    update_hyperparameters!

import QuasiMonteCarlo
import SurrogatesBase
import Zygote

"""
    std_error_at_point(surrogate, point)

Return the predictive standard error of `surrogate` at `point`.

Surrogate implementations that expose uncertainty should add a method to this
generic. The method must accept every point representation supported by the
surrogate's call overload, must not mutate the surrogate, and must return a
nonnegative scalar in the same response units as `surrogate(point)`.

# Arguments

  - `surrogate`: fitted surrogate with a predictive uncertainty model.
  - `point`: scalar or multidimensional query point accepted by `surrogate`.

# Returns

A nonnegative scalar predictive standard error. Implementations should throw an
`ArgumentError` for a point with incompatible dimensionality.

# Example

```julia
using Surrogates

x = [0.0, 0.5, 1.0]
y = sin.(x)
surrogate = Kriging(x, y, 0.0, 1.0)
std_error_at_point(surrogate, 0.25)
```
"""
function std_error_at_point end

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
include("VariableFidelity.jl")
include("Earth.jl")
include("GEK.jl")
include("GEKPLS.jl")
include("KPLS.jl")
include("KPLSK.jl")
include("VirtualStrategy.jl")
include("ParameterInterface.jl")

"""
    current_surrogates

Names of surrogate model families currently listed by Surrogates.jl.

This vector is informational. Use the exported constructor names, such as
[`RadialBasis`](@ref), [`Kriging`](@ref), or [`Wendland`](@ref), to build
surrogates programmatically.

# Returns

A mutable vector of strings. The list is intended for display and discovery;
its contents are not a dispatch contract.

# Example

```julia
using Surrogates

filter(contains("Kriging"), current_surrogates)
```
"""
const current_surrogates = [
    "AbstractGPSurrogate", "EarthSurrogate", "GEK", "GEKPLS", "GENNSurrogate",
    "InverseDistanceSurrogate", "KPLS", "KPLSK", "Kriging", "LinearSurrogate",
    "LobachevskySurrogate", "MOE", "NeuralSurrogate", "PolynomialChaosSurrogate",
    "RadialBasis", "SecondOrderPolynomialSurrogate", "SVMSurrogate",
    "VariableFidelitySurrogate", "Wendland", "XGBoostSurrogate",
]

"""
    RadialBasisStructure(; radial_function, scale_factor, sparse)

Create a named-tuple configuration for a [`RadialBasis`](@ref) surrogate.

# Keywords

  - `radial_function`: radial basis function object, for example
    [`linearRadial()`](@ref) or [`cubicRadial()`](@ref).
  - `scale_factor`: scale factor passed to the `RadialBasis` constructor.
  - `sparse`: whether to use the sparse interpolation matrix path.

# Returns

A named tuple with fields `name`, `radial_function`, `scale_factor`, and
`sparse`. Composite constructors such as [`VariableFidelitySurrogate`](@ref)
consume this value to build the requested surrogate internally.
"""
function RadialBasisStructure(; radial_function, scale_factor, sparse)
    return (
        name = "RadialBasis", type = RadialBasis,
        radial_function = radial_function,
        scale_factor = scale_factor, sparse = sparse,
    )
end

"""
    KrigingStructure(; p, theta)

Create a named-tuple configuration for a [`Kriging`](@ref) surrogate.

# Keywords

  - `p`: Kriging correlation exponent.
  - `theta`: Kriging correlation scale parameter.

# Returns

A named tuple with fields `name`, `p`, and `theta`.
"""
function KrigingStructure(; p, theta)
    return (name = "Kriging", type = Kriging, p = p, theta = theta)
end

"""
    GEKStructure(; p, theta)

Create a named-tuple configuration for a [`GEK`](@ref) surrogate.

# Keywords

  - `p`: correlation exponent. [`GEK`](@ref) requires `2`.
  - `theta`: Kriging correlation scale parameter.

# Returns

A named tuple with fields `name`, `p`, and `theta`.
"""
function GEKStructure(; p, theta)
    return (name = "GEK", type = GEK, p = p, theta = theta)
end

"""
    LinearStructure()

Create a named-tuple configuration for a [`LinearSurrogate`](@ref).

# Returns

A named tuple with the field `name = "LinearSurrogate"`.
"""
function LinearStructure()
    return (name = "LinearSurrogate", type = LinearSurrogate)
end

"""
    InverseDistanceStructure(; p)

Create a named-tuple configuration for an
[`InverseDistanceSurrogate`](@ref).

# Keywords

  - `p`: inverse-distance power parameter.

# Returns

A named tuple with fields `name` and `p`.
"""
function InverseDistanceStructure(; p)
    return (name = "InverseDistanceSurrogate", type = InverseDistanceSurrogate, p = p)
end

"""
    LobachevskyStructure(; alpha, n, sparse)

Create a named-tuple configuration for a [`LobachevskySurrogate`](@ref).

# Keywords

  - `alpha`: Lobachevsky basis scale parameter.
  - `n::Int`: Lobachevsky basis order.
  - `sparse`: whether to use the sparse coefficient path.

# Returns

A named tuple with fields `name`, `alpha`, `n`, and `sparse`.
"""
function LobachevskyStructure(; alpha, n, sparse)
    return (
        name = "LobachevskySurrogate", type = LobachevskySurrogate,
        alpha = alpha, n = n, sparse = sparse,
    )
end

"""
    NeuralStructure(; model, loss, opt, n_epochs)

Create a named-tuple configuration for a [`NeuralSurrogate`](@ref).

# Keywords

  - `model`: Flux model used by the neural surrogate.
  - `loss`: training loss.
  - `opt`: optimizer state or optimizer object accepted by the extension.
  - `n_epochs`: number of training epochs.

# Returns

A named tuple with fields `name`, `model`, `loss`, `opt`, and `n_epochs`.
"""
function NeuralStructure(; model, loss, opt, n_epochs)
    return (
        name = "NeuralSurrogate", type = NeuralSurrogate, model = model,
        loss = loss, opt = opt, n_epochs = n_epochs,
    )
end

"""
    GENNStructure(; model, opt, n_epochs, gamma)

Create a named-tuple configuration for a [`GENNSurrogate`](@ref).

# Keywords

  - `model`: Flux model used by the gradient-enhanced neural surrogate.
  - `opt`: optimizer state or optimizer object accepted by the extension.
  - `n_epochs`: number of training epochs.
  - `gamma`: weight applied to derivative information during training.

# Returns

A named tuple with fields `name`, `model`, `opt`, `n_epochs`, and `gamma`.

!!! note
    
    Unlike the other `*Structure` helpers, this one cannot be used as a
    component of [`MOE`](@ref) or [`VariableFidelitySurrogate`](@ref).
    [`GENNSurrogate`](@ref) is gradient-enhanced: it requires a per-sample
    gradient array alongside `x` and `y`, and a composite surrogate carries only
    function values, so there is nothing to supply. Both composites reject the
    descriptor with a message saying so. The helper is retained for symmetry and
    for direct use in code that does have gradients to hand.
"""
function GENNStructure(; model, opt, n_epochs, gamma)
    return (
        name = "GENNSurrogate", type = GENNSurrogate, model = model, opt = opt,
        n_epochs = n_epochs, gamma = gamma,
    )
end

"""
    XGBoostStructure(; num_round)

Create a named-tuple configuration for an [`XGBoostSurrogate`](@ref).

# Keywords

  - `num_round::Integer`: number of boosting rounds.

# Returns

A named tuple with fields `name` and `num_round`.
"""
function XGBoostStructure(; num_round)
    return (name = "XGBoostSurrogate", type = XGBoostSurrogate, num_round = num_round)
end

"""
    SecondOrderPolynomialStructure()

Create a named-tuple configuration for a
[`SecondOrderPolynomialSurrogate`](@ref).

# Returns

A named tuple with the field `name = "SecondOrderPolynomialSurrogate"`.
"""
function SecondOrderPolynomialStructure()
    return (
        name = "SecondOrderPolynomialSurrogate",
        type = SecondOrderPolynomialSurrogate,
    )
end

"""
    WendlandStructure(; eps, maxiters, tol)

Create a named-tuple configuration for a [`Wendland`](@ref) surrogate.

# Keywords

  - `eps`: reciprocal of the kernel support radius.
  - `maxiters::Integer`: maximum number of conjugate-gradient iterations.
  - `tol`: relative tolerance for the coefficient solve.

# Returns

A named tuple with fields `name`, `eps`, `maxiters`, and `tol`.
"""
function WendlandStructure(; eps, maxiters, tol)
    return (
        name = "Wendland", type = Wendland, eps = eps, maxiters = maxiters,
        tol = tol,
    )
end

"""
    PolyChaosStructure(; op)

Create a named-tuple configuration for a [`PolynomialChaosSurrogate`](@ref).

# Keywords

  - `op`: orthogonal-polynomial basis object from PolyChaos.jl.

# Returns

A named tuple with fields `name` and `op`.
"""
function PolyChaosStructure(; op)
    return (name = "PolynomialChaosSurrogate", type = PolynomialChaosSurrogate, op = op)
end

Base.@deprecate_binding surrogate_optimize surrogate_optimize!

export current_surrogates
export GEKPLS
export RadialBasisStructure, KrigingStructure, GEKStructure, LinearStructure,
    InverseDistanceStructure
export LobachevskyStructure,
    NeuralStructure, GENNStructure, XGBoostStructure,
    SecondOrderPolynomialStructure
export WendlandStructure, PolyChaosStructure
export SamplingAlgorithm
export Kriging, RadialBasis, std_error_at_point
# Parallelization Strategies
export potential_optimal_points
export MinimumConstantLiar, MaximumConstantLiar, MeanConstantLiar, KrigingBeliever,
    KrigingBelieverUpperBound, KrigingBelieverLowerBound
export update!
export parameters, hyperparameters, update_hyperparameters!

# radial basis functions
export linearRadial, cubicRadial, multiquadricRadial, thinplateRadial

# samplers
export sample, GridSample, RandomSample, SobolSample, LatinHypercubeSample,
    HaltonSample
export RandomSample, KroneckerSample, GoldenSample, SectionSample

# Optimization algorithms
export SRBF, LCBS, EI, DYCORS, SOP, RTEA, SMB, surrogate_optimize!
export LobachevskySurrogate, lobachevsky_integral, lobachevsky_integrate_dimension
export LinearSurrogate
export InverseDistanceSurrogate
export SecondOrderPolynomialSurrogate
export Wendland
#export MOE
export VariableFidelitySurrogate
export EarthSurrogate
export GEK
export KPLS
export KPLSK
export AbstractSurrogate

# Extensions
include("extensions.jl")

# Component construction for the composite surrogates. Included after
# `extensions.jl` because it dispatches on the extension surrogate types, whose
# stubs are defined there.
include("ComponentSurrogates.jl")

# A surrogate passed as an *argument* to a broadcast must be treated as a scalar:
# `std_error_at_point.(surrogate, points)` and `gradient.(surrogate, points)`
# should evaluate at each point, not iterate the surrogate. Broadcast asks
# `broadcastable` of its arguments, and Base's fallback is `collect(x)`, so
# without a method these fail with `no method matching length(...)`.
#
# `surrogate.(points)` needs nothing: the surrogate is in function position
# there, and broadcast never asks `broadcastable` of the function.
#
# Deterministic surrogates already get this from `Function`. The stochastic ones
# do not, so each needs a method. Declared per concrete type rather than on
# `AbstractStochasticSurrogate`: that type belongs to `SurrogatesBase`, and a
# method on another package's function over another package's types is piracy.
for T in (:Kriging, :GEK, :KPLS, :KPLSK, :GEKPLS, :AbstractGPSurrogate)
    @eval Base.broadcastable(surrogate::$T) = Ref(surrogate)
end

export AbstractGPSurrogate, logpdf_surrogate
export NeuralSurrogate
export GENNSurrogate, predict_derivative
export PolynomialChaosSurrogate
export XGBoostSurrogate
export SVMSurrogate
export MOE

# Precompilation workloads
include("precompilation.jl")

end
