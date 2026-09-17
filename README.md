## Surrogates.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/Surrogates/stable/)

[![codecov](https://codecov.io/gh/SciML/Surrogates.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/Surrogates.jl)
[![Build Status](https://github.com/SciML/Surrogates.jl/workflows/Tests.yml/badge.svg?branch=master)](https://github.com/SciML/Surrogates.jl/actions/workflows/Tests.yml)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12571718.svg)](https://doi.org/10.5281/zenodo.12571718)

A surrogate model is an approximation method that mimics the behavior of a computationally
expensive simulation. In more mathematical terms: suppose we are attempting to optimize a function
`f(p)`, but each calculation of `f` is very expensive. It may be the case we need to solve a PDE for each point or use advanced numerical linear algebra machinery, which is usually costly. The idea is then to develop a surrogate model `g` which approximates `f` by training on previous data collected from evaluations of `f`.
The construction of a surrogate model can be seen as a three-step process:

 1. Sample selection
 2. Construction of the surrogate model
 3. Surrogate optimization

Sampling can be done through [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl), all the functions available there can be used in Surrogates.jl.

## Getting started

```julia
using Surrogates

f(x) = log(x) * x^2 + x^3          # the expensive function
lb, ub = 1.0, 6.0                  # the box to model it on

x = sample(50, lb, ub, SobolSample())
y = f.(x)

s = Kriging(x, y, lb, ub)          # build the surrogate
s(3.5)                             # evaluate it anywhere in [lb, ub]
```

Every surrogate is a callable object, so `s(x)` is the prediction at `x`. Sampling comes
from [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl): any sampler
there works here.

Two more things you can do with any surrogate:

```julia
update!(s, 4.2, f(4.2))                                    # add a sample in place
surrogate_optimize!(f, SRBF(), lb, ub, s, SobolSample())   # minimize f using s
```

## Surrogate models

### Included in Surrogates.jl

| Model                           | Constructor                                                         | Notes                                                     |
| :------------------------------ | :------------------------------------------------------------------ | :-------------------------------------------------------- |
| Linear                          | `LinearSurrogate(x, y, lb, ub)`                                     | Least-squares affine fit                                  |
| Radial basis                    | `RadialBasis(x, y, lb, ub)`                                         | Choice of kernel                                          |
| Wendland                        | `Wendland(x, y, lb, ub)`                                            | Compactly supported basis, sparse system                  |
| Inverse distance                | `InverseDistanceSurrogate(x, y, lb, ub)`                            | Shepard interpolation                                     |
| Second order polynomial         | `SecondOrderPolynomialSurrogate(x, y, lb, ub)`                      | Full quadratic; needs enough samples to identify          |
| Lobachevsky                     | `LobachevskySurrogate(x, y, lb, ub)`                                | Spline basis; supports exact integration                  |
| Earth                           | `EarthSurrogate(x, y, lb, ub)`                                      | Adaptive regression splines (additive; no interactions)   |
| Kriging                         | `Kriging(x, y, lb, ub)`                                             | Gaussian process; also gives a variance estimate          |
| Gradient enhanced Kriging       | `GEK(x, y, lb, ub)`                                                 | Trains on gradients; `y` holds values then gradients      |
| Kriging + partial least squares | `KPLS(x, y, n_comp, lb, ub, theta)`                                 | For many input dimensions                                 |
| KPLS with a second fit          | `KPLSK(x, y, n_comp, lb, ub, theta)`                                | As `KPLS`, refits in the full space                       |
| Gradient enhanced KPLS          | `GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, theta)` | Gradient enhanced, dimension reduced                      |
| Variable fidelity               | `VariableFidelitySurrogate(x, y, lb, ub)`                           | Cheap model plus a correction fitted to expensive samples |

### Available through package extensions

These models ship with Surrogates.jl but stay dormant until you load their backing
packages, which are weak dependencies and so are **not** installed for you. Add them
yourself, then `using` them alongside Surrogates. The type is always visible, but its
constructor only exists once the extension loads.

| Model                            | Constructor                              | Install and `using`              |
| :------------------------------- | :--------------------------------------- | :------------------------------- |
| Gaussian process                 | `AbstractGPSurrogate(x, y)`              | `AbstractGPs`, `KernelFunctions` |
| Neural network                   | `NeuralSurrogate(x, y, lb, ub)`          | `Flux`, `NNlib`, `Optimisers`    |
| Gradient enhanced neural network | `GENNSurrogate(x, y, lb, ub, dydx)`      | `Flux`, `NNlib`, `Optimisers`    |
| Mixture of experts               | `MOE(x, y, expert_types)`                | `GaussianMixtures`               |
| Polynomial chaos                 | `PolynomialChaosSurrogate(x, y, lb, ub)` | `PolyChaos`                      |
| Support vector machine           | `SVMSurrogate(x, y, lb, ub)`             | `LIBSVM`, `ScikitLearnBase`      |
| Gradient boosted trees           | `XGBoostSurrogate(x, y, lb, ub)`         | `XGBoost`                        |

So a neural surrogate is:

```julia
using Pkg; Pkg.add(["Flux", "NNlib", "Optimisers"])
using Surrogates, Flux, NNlib, Optimisers

s = NeuralSurrogate(x, y, lb, ub)
```

Note that `AbstractGPSurrogate` takes no bounds, and `MOE` fits a separate surrogate per
cluster. If one of its experts is itself extension-backed, you need that extension too.

## Working with a surrogate

In one dimension a sample is a number and the bounds are numbers; in `d` dimensions a
sample is a tuple or vector of length `d` and the bounds are vectors of length `d`:

```julia
s = Kriging(x, y, 0.0, 10.0)                # 1D: x isa Vector{Float64}
s = Kriging(x, y, [0.0, 0.0], [10.0, 10.0]) # 2D: x isa Vector{Tuple{Float64, Float64}}
```

Write a point whichever way suits you: `s((1.0, 2.0))` and `s([1.0, 2.0])` mean the same
thing.

Add observations as they arrive, one at a time or in batches:

```julia
update!(s, (1.0, 2.0), f((1.0, 2.0)))                  # one point
update!(s, [(1.0, 2.0), (2.0, 3.0)], f.(new_points))   # a batch
update!(gek, x_new, y_new, grad_new)                   # gradient-enhanced models
```

### Several outputs at once

A surrogate can model a vector-valued function: give each `y[i]` as a vector and `s(x)`
returns a vector of the same length.

```julia
f(p) = [p[1]^2 + p[2]^2, p[1] - p[2]]
s = RadialBasis(x, f.(x), lb, ub)
s((1.0, 2.0))     # a 2-element vector
```

`LinearSurrogate`, `RadialBasis`, `InverseDistanceSurrogate`,
`SecondOrderPolynomialSurrogate`, `LobachevskySurrogate`, `VariableFidelitySurrogate`,
`NeuralSurrogate` and `GENNSurrogate` support this. The rest model a scalar response.

### Uncertainty

`Kriging`, `GEK`, `KPLS`, `KPLSK`, `GEKPLS` and `AbstractGPSurrogate` model their own
uncertainty as well as the response:

```julia
s(3.5)                        # the prediction
std_error_at_point(s, 3.5)    # how much to trust it
```

This is what the uncertainty-driven acquisition functions need, so `EI()` and `LCBS()`
work with these six.

### What each surrogate takes

| Surrogate                        | 1D  | Multidimensional | Several outputs | Needs gradients |
| :------------------------------- | :-- | :--------------- | :-------------- | :-------------- |
| `LinearSurrogate`                | yes | yes              | yes             | no              |
| `RadialBasis`                    | yes | yes              | yes             | no              |
| `Wendland`                       | yes | yes              | no              | no              |
| `InverseDistanceSurrogate`       | yes | yes              | yes             | no              |
| `SecondOrderPolynomialSurrogate` | yes | yes              | yes             | no              |
| `LobachevskySurrogate`           | yes | yes              | yes             | no              |
| `EarthSurrogate`                 | yes | yes              | no              | no              |
| `Kriging`                        | yes | yes              | no              | no              |
| `GEK`                            | yes | yes              | no              | yes             |
| `KPLS`                           | yes | yes              | no              | no              |
| `KPLSK`                          | yes | yes              | no              | no              |
| `GEKPLS`                         | yes | yes              | no              | yes             |
| `VariableFidelitySurrogate`      | yes | yes              | yes             | no              |
| `AbstractGPSurrogate`            | yes | yes              | no              | no              |
| `NeuralSurrogate`                | yes | yes              | yes             | no              |
| `GENNSurrogate`                  | yes | yes              | yes             | yes             |
| `MOE`                            | yes | yes              | no              | no              |
| `PolynomialChaosSurrogate`       | yes | yes              | no              | no              |
| `SVMSurrogate`                   | yes | yes              | no              | no              |
| `XGBoostSurrogate`               | yes | yes              | no              | no              |

Two exceptions worth knowing: `KPLS`, `KPLSK` and `GEKPLS` always take a component count,
so their bounds are vectors even in one dimension: `KPLS(x, y, 1, [lb], [ub], [1.0])`.
And `SVMSurrogate` wraps a classifier, so its `y` holds class labels rather than a
continuous response.

### Inspecting and refitting

Surrogates implement the [SurrogatesBase.jl](https://github.com/SciML/SurrogatesBase.jl)
interface: `parameters(s)` returns what the fit produced, `hyperparameters(s)` what
governed it.

`Kriging`, `GEK`, `KPLS`, `KPLSK` and `GEKPLS` fit their correlation scales by maximum
likelihood. Calling `update_hyperparameters!(s)` re-estimates those scales in place, using
the design points and observations the model already holds.

## Automatic differentiation

Surrogates are ordinary callable objects, so a fitted surrogate can be differentiated
with respect to its input with either ForwardDiff or Zygote:

```julia
using ForwardDiff
ForwardDiff.derivative(s, 3.5)          # 1D
ForwardDiff.gradient(s, [1.0, 2.0])     # multidimensional
```

This works for every surrogate in Surrogates.jl, with two exceptions.

`XGBoostSurrogate` and `SVMSurrogate` cannot be differentiated: they wrap
gradient-boosted trees and a LIBSVM model, which are piecewise constant. They predict
normally; only differentiation is unavailable.

`GENNSurrogate` is trained on gradients you supply and can report them directly with
`predict_derivative(s, x)`.

## Optimization

Minimize an expensive function through a surrogate with `surrogate_optimize!`:

```julia
surrogate_optimize!(f, SRBF(), lb, ub, s, SobolSample())
```

| Method                                     | Type              | What it does                                             |
| :----------------------------------------- | :---------------- | :------------------------------------------------------- |
| Stochastic RBF                             | `SRBF()`          | Weighs predicted value against distance from past points |
| Lower confidence bound                     | `LCBS()`          | Scores candidates by prediction minus uncertainty        |
| Expected improvement                       | `EI()`            | Scores candidates by expected gain over the best so far  |
| DYCORS                                     | `DYCORS()`        | Perturbs a subset of coordinates; suits many dimensions  |
| Surrogate optimization with Pareto centers | `SOP(p)`          | Keeps `p` search centers at once                         |
| Multi-objective                            | `SMB()`, `RTEA()` | Several objectives at once                               |

`SRBF()`, `DYCORS()` and `SOP()` work with any surrogate. `EI()` and `LCBS()` score
candidates by uncertainty, so they need one of the six that provide it.

`SMB()` and `RTEA()` minimize several objectives at once, so they need a surrogate that
takes a vector-valued response: any of the multi-output models above except
`GENNSurrogate`, which also requires a Jacobian for every new observation.

Gradient-enhanced surrogates need a gradient alongside every new response. Ask the
optimizer to obtain it by AD:

```julia
surrogate_optimize!(f, SRBF(), lb, ub, gek, SobolSample(); needs_gradient = true)
```

### Evaluating a batch in parallel

When `f` can be evaluated several times at once, ask for a batch of points with
`potential_optimal_points`, available for `SRBF()`, `EI()` and `LCBS()`:

```julia
points = potential_optimal_points(EI(), MeanConstantLiar(), lb, ub, s, SobolSample(), 4)
```

It keeps the points apart by assigning *virtual* values to the ones not yet evaluated.
Choose how with `MinimumConstantLiar`, `MeanConstantLiar`, `MaximumConstantLiar`,
`KrigingBeliever`, `KrigingBelieverUpperBound` or `KrigingBelieverLowerBound`: the
liars take their virtual value from the observations already in hand, the believers from
the surrogate's own prediction, so those three need a surrogate that models uncertainty.

## Installing Surrogates package

```julia
using Pkg
Pkg.add("Surrogates")
```
