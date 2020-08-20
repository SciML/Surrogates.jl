![SurrogatesLogo](docs/src/images/Surrogates.png)

## Surrogates.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/Surrogates.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/Surrogates.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fl7hr18apc7lt4of?svg=true)](https://ci.appveyor.com/project/ludoro/surrogates-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiffEq/Surrogates.jl/badge.svg)](https://coveralls.io/github/JuliaDiffEq/Surrogates.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://surrogates.sciml.ai/stable/)
[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://surrogates.sciml.ai/dev/)

A surrogate model is an approximation method that mimics the behavior of a computationally
expensive simulation. In more mathematical terms: suppose we are attempting to optimize a function
`f(p)`, but each calculation of `f` is very expensive. It may be the case we need to solve a PDE for each point or use advanced numerical linear algebra machinery, which is usually costly. The idea is then to develop a surrogate model `g` which approximates `f` by training on previous data collected from evaluations of `f`.
The construction of a surrogate model can be seen as a three-step process:

1. Sample selection
2. Construction of the surrogate model
3. Surrogate optimization

## ALL the currently available sampling methods:

- Grid
- Uniform
- Sobol
- Latin Hypercube
- Low Discrepancy
- Random following different distributions

## ALL the currently available surrogate models:

- Kriging
- Radial Basis
- Wendland
- Linear
- Second Order Polynomial
- Support Vector Machines
- Neural Networks
- Random Forests
- Lobachesky
- Inverse-distance
- Polynomial expansions
- Variable fidelity
- Mixture of experts
- Earth
- Gradient Enhanced Kriging 

## ALL the currently available optimization methods:

- SRBF
- LCBS
- DYCORS
- EI
- SOP

## Installing Surrogates package

```julia
using Pkg
Pkg.add("Surrogates")
```
