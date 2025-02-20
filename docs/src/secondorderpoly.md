# Second Order Polynomial Surrogate Tutorial

The square polynomial model can be expressed by:

``y = Xβ + ϵ``

Where X is the matrix of the linear model augmented by adding 2d columns,
containing pair by pair products of variables and variables squared.

```@example second_order_tut
using Surrogates
using Plots
```

## Sampling

```@example second_order_tut
f = x -> 3 * sin(x) + 10 / x
lb = 3.0
ub = 6.0
n = 100
x = sample(n, lb, ub, HaltonSample())
y = f.(x)
scatter(x, y, label = "Sampled points", xlims = (lb, ub))
plot!(f, label = "True function", xlims = (lb, ub))
```

## Building the surrogate

```@example second_order_tut
sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
plot(x, y, seriestype = :scatter, label = "Sampled points", xlims = (lb, ub))
plot!(f, label = "True function", xlims = (lb, ub))
plot!(sec, label = "Surrogate function", xlims = (lb, ub))
```

## Optimizing

```@example second_order_tut
surrogate_optimize(f, SRBF(), lb, ub, sec, SobolSample())
scatter(x, y, label = "Sampled points")
plot!(f, label = "True function", xlims = (lb, ub))
plot!(sec, label = "Surrogate function", xlims = (lb, ub))
```

The optimization method successfully found the minimum.
