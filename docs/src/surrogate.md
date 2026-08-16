# Surrogate

Surrogates.jl models are callable objects that fit observations and evaluate an
approximation at new points. The abstract type hierarchy comes from
[SurrogatesBase.jl](https://github.com/SciML/SurrogatesBase.jl); the package
provides deterministic and extension-backed stochastic implementations.

## Generic interface

An implementation of [`AbstractSurrogate`](@ref) must provide the following
operations:

  1. `surrogate(point)`: evaluate the fitted approximation at one point. The
     point representation must be the same representation accepted by the
     constructor and should be documented by the implementation.
  2. `update!(surrogate, x_new, y_new)`: incorporate one observation or a batch
     of observations and update the fitted state in place. A batch must contain
     one response for each new point. Implementations may additionally accept
     algorithm-specific keyword arguments, such as gradient observations.
  3. `surrogate.x` and `surrogate.y`: retain the training points and responses
     when the surrogate is used with [`surrogate_optimize!`](@ref) or
     [`potential_optimal_points`](@ref). The optimization routines use these
     fields to identify the current best observation and to avoid duplicate
     candidate points.

Stochastic surrogates additionally implement [`std_error_at_point`](@ref) when
they expose predictive uncertainty and [`logpdf_surrogate`](@ref) when they
expose a log density or marginal likelihood. Implementations should document
the response shape and units returned by these methods.

The interface is deliberately expressed in terms of the generic call syntax
and `update!`; optimization code should not depend on a concrete surrogate
type. A minimal implementation is:

```julia
using SurrogatesBase

mutable struct MySurrogate{T} <: SurrogatesBase.AbstractDeterministicSurrogate
    x::Vector{T}
    y::Vector{T}
end

(s::MySurrogate)(point) = point^2

function SurrogatesBase.update!(s::MySurrogate, x_new, y_new)
    push!(s.x, x_new)
    push!(s.y, y_new)
    return nothing
end
```

The generic contract is tested with a local implementation in the test suite;
concrete surrogate tests then cover each model's fitting and numerical behavior.

```@docs
AbstractSurrogate
current_surrogates
std_error_at_point
```

  - Linear surrogate

```@docs
LinearSurrogate
```

  - Radial basis function surrogate

```@docs
RadialBasis
```

  - Kriging surrogate

```@docs
Kriging
```

  - Lobachevsky surrogate

```@docs
LobachevskySurrogate
lobachevsky_integral(loba::LobachevskySurrogate,lb,ub)
```

  - Support vector machine surrogate, requires `using LIBSVM`.

```
SVMSurrogate(x,y,lb::Number,ub::Number)
```

  - Random forest surrogate, requires `using XGBoost`.

```
XGBoostSurrogate(x,y,lb,ub;num_round::Int = 1)
```

  - Neural network surrogate, requires `using Flux`.

```
NeuralSurrogate(x,y,lb,ub; model = Chain(Dense(length(x[1]),1), first), loss = (x,y) -> Flux.mse(model(x), y),opt = Descent(0.01),n_echos::Int = 1)
```

```@docs
EarthSurrogate
SVMSurrogate
```

## Structure Descriptors

```@docs
RadialBasisStructure
KrigingStructure
LinearStructure
InverseDistanceStructure
LobachevskyStructure
NeuralStructure
GENNStructure
XGBoostStructure
SecondOrderPolynomialStructure
WendlandStructure
```

# Creating another surrogate

It's great that you want to add another surrogate to the library!
You will need to:

 1. Define a new mutable struct and a constructor function
 2. Define update!(your\_surrogate, x\_new, y\_new)
 3. Define your\_surrogate(value) for the approximation

## Example

```julia
mutable struct NewSurrogate{X, Y, L, U, C, A, B} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    alpha::A
    beta::B
end

function NewSurrogate(x, y, lb, ub, parameters)
    ...
    return NewSurrogate(x, y, lb, ub, calculated \ _coeff, alpha, beta)
end

function update!(NewSurrogate, x_new, y_new)
    ...
end

function (s::NewSurrogate)(value)
    return s.coeff * value + s.alpha
end
```
