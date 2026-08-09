# Public API

Package-owned interfaces are documented with their corresponding topics:

- [Surrogate construction](surrogate.md)
- [Sampling](samples.md)
- [Surrogate optimization](optimizations.md)
- [Parallel optimization](parallel.md)
- [Radial basis functions](radials.md)
- [Gaussian process surrogates](abstractgps.md)
- [Gradient-enhanced Kriging](gek.md)
- [Gradient-enhanced Kriging with partial least squares](gekpls.md)
- [Neural surrogates](neural.md)
- [Mixture-of-experts surrogates](moe.md)
- [Polynomial chaos surrogates](polychaos.md)
- [Variable-fidelity surrogates](variablefidelity.md)
- [Wendland surrogates](wendland.md)
- [XGBoost surrogates](xgboost.md)

## Reexported Sampling Algorithms

The reexported `SamplingAlgorithm`, `GoldenSample`, `GridSample`, `HaltonSample`,
`KroneckerSample`, `LatinHypercubeSample`, `RandomSample`, and `SobolSample`
types are defined and documented by
[QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/).

Surrogates.jl defines its own `sample` wrapper so multidimensional samples retain
the package's vector-of-tuples representation.
