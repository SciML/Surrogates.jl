## Lobachevsky surrogate tutorial

Lobachevsky splines function is a function that used for univariate and multivariate scattered interpolation. Introduced by Lobachevsky in 1842 to investigate errors in astronomical measurements.

We are going to use a Lobachevsky surrogate to optimize $f(x)=sin(x)+sin(10/3 * x)$.

First of all import `Surrogates` and `Plots`.
```@example LobachevskySurrogate_tutorial
using Surrogates
using Plots
default()
```
### Sampling

We choose to sample f in 4 points between 0 and 4 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example LobachevskySurrogate_tutorial
f(x) = sin(x) + sin(10/3 * x)
n_samples = 5
lower_bound = 1.0
upper_bound = 4.0
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", xlims=(lower_bound, upper_bound))
```
### Building a surrogate

With our sampled points we can build the Lobachevsky surrogate using the `LobachevskySurrogate` function.

`lobachevsky_surrogate` behaves like an ordinary function which we can simply plot. Alpha is the shape parameters and n specify how close you want lobachevsky function to radial basis function.

```@example LobachevskySurrogate_tutorial
alpha = 2.0
n = 6
lobachevsky_surrogate = LobacheskySurrogate(x, y, lower_bound, upper_bound, alpha = 2.0, n = 6)
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(lobachevsky_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
### Optimizing
Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example LobachevskySurrogate_tutorial
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, lobachevsky_surrogate, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(lobachevsky_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
