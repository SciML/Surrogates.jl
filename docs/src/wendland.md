# Wendland Surrogate Tutorial

The Wendland surrogate is a compact surrogate: it allocates much less memory than other surrogates.
The coefficients are found using an iterative solver.

``f = x -> exp(-x^2)``

```@example wendland
using Surrogates
using Plots
```

```@example wendland
n = 100
lower_bound = 0.0
upper_bound = 1.0
f = x -> exp(-x^2)
x = sample(n, lower_bound, upper_bound, SobolSample())
y = f.(x)
```

We choose to sample f in 100 points between 5 and 25 using `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

## Building Surrogate

The choice of the right parameter is especially important here:
a slight change in ϵ would produce a totally different fit.
Try it yourself with this function!

```@example wendland
wend = Wendland(x, y, lower_bound, upper_bound, eps = 0.45)
```

```@example wendland
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(wend, label = "Surrogate function", xlims = (lower_bound, upper_bound), legend = :top)
```
