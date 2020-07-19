## Radial Surrogates
The Radial Basis Surrogate model represents the interpolating function as a linear combination of basis functions, one for each training point. Let's start with something easy to get our hands dirty. I want to build a surrogate for:

`f(x) = log(x)*x^2+x^3``

Let's choose the Radial Basis Surrogate for 1D. First of all we have to import these two packages: `Surrogates` and `Plots`,

```@RadialBasisSurrogate
using Surrogates
using Plots
default()
```

We choose to sample f in 30 points between 5 to 25 using `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@RadialBasisSurrogate
f(x) = log(x)*x^2 + x^3
n_samples = 30
lower_bound = 5
upper_bound = 25
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound))
```


## Building Surrogate
