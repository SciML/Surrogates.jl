## Kriging surrogate tutorial (1D)

Kriging or Gaussian process regression is a method of interpolation for which the interpolated values are modeled by a Gaussian process.

We are going to use a Kriging surrogate to optimize $f(x)=(6x-2)^2sin(12x-4)$. (function from Forrester et al. (2008)).

First of all import `Surrogates` and `Plots`.
```@example kriging_tutorial
using Surrogates
using Plots
```
### Sampling

We choose to sample f in 4 points between 0 and 1 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example kriging_tutorial
# https://www.sfu.ca/~ssurjano/forretal08.html
# Forrester et al. (2008) Function
f(x) = (6 * x - 2)^2 * sin(12 * x - 4)

n_samples = 4
lower_bound = 0.0
upper_bound = 1.0

x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)

scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17))
plot!(f, label="True function")
```
### Building a surrogate

With our sampled points we can build the Kriging surrogate using the `Kriging` function.

`kriging_surrogate` behaves like an ordinary function which we can simply plot. A nice statistical property of this surrogate is being able to calculate the error of the function at each point, we plot this as a confidence interval using the `ribbon` argument.

```@example kriging_tutorial
kriging_surrogate = Kriging(x, y, lower_bound, upper_bound, p=1.9);

plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17))
plot!(f, label="True function")
plot!(kriging_surrogate, label="Surrogate function", ribbon=p->std_error_at_point(kriging_surrogate, p))
```
### Optimizing
Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example kriging_tutorial
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, kriging_surrogate, SobolSample())

scatter(x, y, label="Sampled points", ylims=(-7, 7))
plot!(f, label="True function")
plot!(kriging_surrogate, label="Surrogate function", ribbon=p->std_error_at_point(kriging_surrogate, p))
```
