# Variable fidelity Surrogates

With the variable fidelity surrogate, we can specify two different surrogates: one for high fidelity data and one for low fidelity data.
By default, the first half samples are considered high fidelity and the second half low fidelity.

```@example variablefid
using Surrogates
using Plots
default()
```

```@example variablefid
n = 20
lower_bound = 1.0
upper_bound = 6.0
x = sample(n,lower_bound,upper_bound,LowDiscrepancySample(2))
f = x -> 1/3*x
y = f.(x)
scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function", scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound), legend=:top)
```

```@example variablefid
varfid = VariableFidelitySurrogate(x,y,lower_bound,upper_bound)
```

```@example RadialBasisSurrogate
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(varfid, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)
```
