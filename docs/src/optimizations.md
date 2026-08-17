# Optimization techniques

Surrogate optimization methods share the call/update contract documented under
[`AbstractSurrogate`](@ref). The algorithm markers below select the search
strategy; the algorithm-specific methods use the same `surrogate_optimize!`
entry point.

```@docs
Surrogates.SurrogateOptimizationAlgorithm
Surrogates.ParallelStrategy
SRBF
LCBS
EI
DYCORS
SOP
SMB
RTEA
```

```@docs
surrogate_optimize!
```

`SurrogateOptimizationAlgorithm` and `ParallelStrategy` are developer
interfaces. A new optimization algorithm should subtype the appropriate marker
and implement the corresponding `surrogate_optimize!` or
`potential_optimal_points` method while using only the generic surrogate
operations described in [`AbstractSurrogate`](@ref).

  - SRBF

```@docs
surrogate_optimize!(obj::Function,::SRBF,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
```

  - LCBS

```@docs
surrogate_optimize!(obj::Function,::LCBS,lb,ub,krig,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
```

  - EI

```@docs
surrogate_optimize!(obj::Function,::EI,lb,ub,krig,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
```

  - DYCORS

```@docs
surrogate_optimize!(obj::Function,::DYCORS,lb,ub,surrn::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
```

  - SOP

```@docs
surrogate_optimize!(obj::Function,sop1::SOP,lb::Number,ub::Number,surrSOP::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=min(500*1,5000))
```

## Adding another optimization method

To add another optimization method, you just need to define a new
SurrogateOptimizationAlgorithm and write its corresponding algorithm, overloading the following:

```
surrogate_optimize!(obj::Function,::NewOptimizationType,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
```
