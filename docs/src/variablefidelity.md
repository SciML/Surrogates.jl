# Variable fidelity surrogate tutorial

With the variable fidelity surrogate, we can specify two different surrogates: one for high-fidelity data and one for low-fidelity data. By default, the first half of the samples are considered high-fidelity and the second half low-fidelity.

The model is a sum of two parts. The low-fidelity surrogate is fitted to the cheap samples, and a *correction* surrogate is then fitted to the residuals the low-fidelity surrogate leaves at the expensive samples:

```math
\hat f(x) = \hat f_{\text{low}}(x) + \hat\varepsilon(x), \qquad \hat\varepsilon \text{ fitted to } y_{\text{high}} - \hat f_{\text{low}}(x_{\text{high}})
```

so the sum reproduces the high-fidelity data exactly, whatever the low-fidelity surrogate does with it. **The split is positional**: `x` must be ordered with the high-fidelity samples first, and `num_high_fidel` says how many there are. It must leave at least one sample on each side.

`update!` adds *low-fidelity* samples. It refits the correction surrogate as well, since changing the low-fidelity surrogate changes the residuals the correction was fitted to.

Any of the `*Structure` descriptors may be used for either level — `RadialBasisStructure`, `KrigingStructure`, `LinearStructure`, `InverseDistanceStructure`, `LobachevskyStructure`, `NeuralStructure`, `XGBoostStructure`, `SecondOrderPolynomialStructure` and `WendlandStructure`. `GEKStructure` is not supported: `GEK` needs gradient observations alongside its function values, and a variable-fidelity design carries only function values.

```@docs
VariableFidelitySurrogate
```

```@example variablefid
using Surrogates
using Plots
```

```@example variablefid
n = 100
lower_bound = 1.0
upper_bound = 6.0
x = sample(n, lower_bound, upper_bound, SobolSample())
f = x -> 1 / 3 * x
y = f.(x)
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
```

```@example variablefid
varfid = VariableFidelitySurrogate(x, y, lower_bound, upper_bound)
```

```@example variablefid
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(
    varfid, label = "Surrogate function", xlims = (lower_bound, upper_bound), legend = :top)
```
