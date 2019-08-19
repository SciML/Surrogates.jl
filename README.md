![SurrogatesLogo](docs/src/images/Surrogates.png)
## Surrogates.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/Surrogates.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/Surrogates.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fl7hr18apc7lt4of?svg=true)](https://ci.appveyor.com/project/ludoro/surrogates-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiffEq/Surrogates.jl/badge.svg)](https://coveralls.io/github/JuliaDiffEq/Surrogates.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://surrogates.juliadiffeq.org/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://surrogates.juliadiffeq.org/latest/)

A surrogate model is an approximation method that mimics the behavior of a computationally
expensive simulation. In more mathematical terms: suppose we are attempting to optimize a function
`f(p)`, but each calculation of `f` is very expensive. It may be the case we need to solve a PDE for each point or use advanced numerical linear algebra machinery which is usually costly. The idea is then to develop a surrogate model `g` which approximates `f` by training on previous data collected from evaluations of `f`.
The construction of a surrogate model can be seen as a three steps process:
- Sample selection
- Construction of the surrogate model
- Surrogate optimization

## ALL the currently available sampling methods: 

- Random
- Uniform 
- Sobol
- Latin Hypercube
- Low Discrepancy 

## ALL the currently available surrogate models: 

- Kriging
- Radial Basis Function
- Space Mapping
- Gradient-Enhanced Kriging (GEK)
- Polynomial Response Surfaces
- Support Vector Machines
- Artificial Neural Networks 
- Random Forests

## ALL the currently available optimization methods: 

- SRBF
- LCBS 
- DYCOR
- EI

## Installing Surrogates package

Using Pkg
Pkg.add ("Surrogates")
