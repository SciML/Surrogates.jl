# Public API

## Surrogate Interfaces

```@docs
Surrogates.AbstractSurrogate
Surrogates.std_error_at_point
```

## Surrogate Types

```@docs
Surrogates.AbstractGPSurrogate
Surrogates.EarthSurrogate
Surrogates.GEK
Surrogates.GEKPLS
Surrogates.GENNSurrogate
Surrogates.InverseDistanceSurrogate
Surrogates.MOE
Surrogates.NeuralSurrogate
Surrogates.PolynomialChaosSurrogate
Surrogates.SVMSurrogate
Surrogates.SecondOrderPolynomialSurrogate
Surrogates.VariableFidelitySurrogate
Surrogates.Wendland
Surrogates.XGBoostSurrogate
```

## Surrogate Structure Helpers

```@docs
Surrogates.GENNStructure
Surrogates.InverseDistanceStructure
Surrogates.KrigingStructure
Surrogates.LinearStructure
Surrogates.LobachevskyStructure
Surrogates.NeuralStructure
Surrogates.RadialBasisStructure
Surrogates.SecondOrderPolynomialStructure
Surrogates.WendlandStructure
Surrogates.XGBoostStructure
```

## Radial Basis Functions

```@docs
Surrogates.cubicRadial
Surrogates.linearRadial
Surrogates.multiquadricRadial
Surrogates.thinplateRadial
```

## Sampling

Sampling algorithms and `sample` are re-exported from
[QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/).

## Optimization

```@docs
Surrogates.DYCORS
Surrogates.EI
Surrogates.KrigingBeliever
Surrogates.KrigingBelieverLowerBound
Surrogates.KrigingBelieverUpperBound
Surrogates.LCBS
Surrogates.MaximumConstantLiar
Surrogates.MeanConstantLiar
Surrogates.MinimumConstantLiar
Surrogates.RTEA
Surrogates.SMB
Surrogates.SOP
Surrogates.SRBF
Surrogates.potential_optimal_points
```

## Extension Hooks

```@docs
Surrogates.logpdf_surrogate
Surrogates.predict_derivative
```

## Metadata

```@docs
Surrogates.current_surrogates
Surrogates.lobachevsky_integrate_dimension
```
