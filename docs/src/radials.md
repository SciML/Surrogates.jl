## Radial Surrogates
The Radial Basis Surrogate model represents the interpolating function as a linear combination of basis functions, one for each training point. Let's start with something easy to get our hands dirty. I want to build a surrogate for:

`f(x) = log(x)*x^2+x^3``

Let's choose the Radial Basis Surrogate for 1D. First of all we have to import these two packages: `Surrogates` and `Plots`,

```@example RadialBasisSurrogate
using Surrogates
using Plots
default()
```

We choose to sample f in 30 points between 5 to 25 using `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example RadialBasisSurrogate
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

With our sampled points we can build the **Radial Surrogate** using the `RadialBasis` function.

We can simply calculate `radial_surrogate` for any value.

```@example RadialBasisSurrogate
radial_surrogate = RadialBasis(x, y, lower_bound, upper_bound)
val = radial_surrogate(5.4)
```

Now, we will simply plot `radial_surrogate`:

```@example RadialBasisSurrogate
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```


## Optimizing

Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example RadialBasisSurrogate
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, radial_surrogate, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
