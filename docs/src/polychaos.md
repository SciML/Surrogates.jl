# Polynomial Chaos Surrogate Tutorial

We can create a surrogate using a polynomial expansion, with a different polynomial basis depending on the distribution of the data we are trying to fit. Under the hood, PolyChaos.jl has been used. It is possible to specify a type of polynomial for each dimension of the problem.

## Sampling

We choose to sample f in 100 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Low Discrepancy. This can be done by passing `HaltonSample()` to the `sample` function.

```@example polychaos
using Surrogates
using PolyChaos
using Plots

n = 100
lower_bound = 1.0
upper_bound = 6.0
x = sample(n, lower_bound, upper_bound, HaltonSample())
f = x -> log(x) * x + sin(x)
y = f.(x)
scatter(x, y, label = "Sampled points", xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
```

## Building a Surrogate

```@example polychaos
poly1 = PolynomialChaosSurrogate(x, y, lower_bound, upper_bound)
poly2 = PolynomialChaosSurrogate(
    x, y, lower_bound, upper_bound, orthopolys = SurrogatesPolyChaos.GaussOrthoPoly(5))
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(poly1, label = "First polynomial", xlims = (lower_bound, upper_bound), legend = :top)
plot!(poly2, label = "Second polynomial", xlims = (lower_bound, upper_bound), legend = :top)
```
