# Samples

The syntax for sampling in an interval or region is the following:
```
sample(n,lb,ub,S::SamplingAlgorithm)
```
where lb and ub are, respectively, the lower and upper bounds.
There are many sampling algorithms to choose from:

* Grid sample
```@docs
GridSample{T}
sample(n,lb,ub,S::GridSample)
```

* Uniform sample
```@docs
sample(n,lb,ub,::UniformSample)
```

* Sobol sample
```@docs
sample(n,lb,ub,::SobolSample)
```

* Latin Hypercube sample
```@docs
sample(n,lb,ub,::LatinHypercubeSample)
```

* Low Discrepancy sample
```@docs
LowDiscrepancySample{T}
sample(n,lb,ub,S::LowDiscrepancySample)
```

## Adding a new sampling method

Adding a new sampling method is a two- step process:

1. Adding a new SamplingAlgorithm type
2. Overloading the sample function with the new type.

**Example**
```
struct NewAmazingSamplingAlgorithm{OPTIONAL} <: SamplingAlgorithm end

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
