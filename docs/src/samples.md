# Sampling

Sampling methods are provided by the [QuasiMonteCarlo package](https://docs.sciml.ai/QuasiMonteCarlo/stable/).

The syntax for sampling in an interval or region is the following:

```julia
sample(n, lb, ub, S::SamplingAlgorithm)
```

where lb and ub are, respectively, the lower and upper bounds.
There are many sampling algorithms to choose from:

  - Grid sample

```julia
sample(n, lb, ub, GridSample())
```

  - Uniform sample

```julia
sample(n, lb, ub, RandomSample())
```

  - Sobol sample

```julia
sample(n, lb, ub, SobolSample())
```

  - Latin Hypercube sample

```julia
sample(n, lb, ub, LatinHypercubeSample())
```

  - Low Discrepancy sample

```julia
sample(n, lb, ub, HaltonSample())
```

  - Sample on section

```julia
sample(n, lb, ub, SectionSample())
```

## Adding a new sampling method

Adding a new sampling method is a two-step process:

 1. Adding a new SamplingAlgorithm type
 2. Overloading the sample function with the new type.

**Example**

```julia
struct NewAmazingSamplingAlgorithm{OPTIONAL} <: QuasiMonteCarlo.SamplingAlgorithm end

function sample(n,lb,ub,::NewAmazingSamplingAlgorithm)
    if lb is  Number
        ...
        return x
    else
        ...
        return Tuple.(x)
    end
end
```
