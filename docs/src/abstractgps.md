# Gaussian Process Surrogate Tutorial

Gaussian Process regression in Surrogates.jl is implemented as a simple wrapper around the [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) package. AbstractGPs comes with a variety of covariance functions (kernels). See [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/) for examples.

!!! tip
    
    The examples below demonstrate the use of AbstractGPs with out-of-the-box settings without hyperparameter optimization (i.e. without changing parameters like lengthscale, signal variance, and noise variance). Beyond hyperparameter optimization, careful initialization of hyperparameters and priors on the parameters is required for this surrogate to work properly. For more details on how to fit GPs in practice, check out [A Practical Guide to Gaussian Processes](https://infallible-thompson-49de36.netlify.app/).
    
    Also see this [example](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/#Hyperparameter-Optimization) to understand hyperparameter optimization with AbstractGPs.

## 1D Example

In the example below, the 'gp_surrogate' assignment code can be commented / uncommented to see how the different kernels influence the predictions.

```@example gp_tutorial1d
using Surrogates
using Plots
using AbstractGPs

f(x) = (6 * x - 2)^2 * sin(12 * x - 4)
n_samples = 100
lower_bound = 0.0
upper_bound = 1.0
xs = lower_bound:0.001:upper_bound
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)

gp_surrogate = AbstractGPSurrogate(
    x, y, gp = GP(PolynomialKernel(; c = 2.0, degree = 15)), Σy = 0.25)
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), ylims = (-7, 17), legend = :top)
plot!(xs, f.(xs), label = "True function", legend = :top)
plot!(0:0.001:1, gp_surrogate.gp_posterior; label = "Posterior", ribbon_scale = 2)
```

## Optimization Example

This example shows the use of AbstractGP Surrogates to find the minima of a function:

```@example abstractgps_tutorial_optimization
using Surrogates
using Plots
using AbstractGPs

f(x) = (x - 2)^2
n_samples = 100
lower_bound = 0.0
upper_bound = 4.0
xs = lower_bound:0.1:upper_bound
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
gp_surrogate = AbstractGPSurrogate(x, y)
surrogate_optimize!(f, SRBF(), lower_bound, upper_bound, gp_surrogate, SobolSample())
```

Plotting the function and the sampled points:

```@example abstractgps_tutorial_optimization
scatter(gp_surrogate.x, gp_surrogate.y, label = "Sampled points",
    ylims = (-1.0, 5.0), legend = :top)
plot!(xs, gp_surrogate.(xs), label = "Surrogate function",
    ribbon = p -> Surrogates.std_error_at_point(gp_surrogate, p), legend = :top)
plot!(xs, f.(xs), label = "True function", legend = :top)
```

## ND Example

```@example abstractgps_tutorialnd
using Plots
default(c = :matter, legend = false, xlabel = "x", ylabel = "y")
using Surrogates
using AbstractGPs

hypot_func = z -> 3 * hypot(z...) + 1
n_samples = 100
lower_bound = [-1.0, -1.0]
upper_bound = [1.0, 1.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = hypot_func.(xys)

x, y = -2:2, -2:2
p1 = surface(x, y, (x1, x2) -> hypot_func((x1, x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, zs)
p2 = contour(x, y, (x1, x2) -> hypot_func((x1, x2)))
scatter!(xs, ys)
plot(p1, p2, title = "True function")
```

Now let's see how our surrogate performs:

```@example abstractgps_tutorialnd
gp_surrogate = AbstractGPSurrogate(xys, zs)
p1 = surface(x, y, (x, y) -> gp_surrogate([x y]))
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> gp_surrogate([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2, title = "Surrogate")
```

```@example abstractgps_tutorialnd
gp_surrogate((0.2, 0.2))
```

```@example abstractgps_tutorialnd
hypot_func((0.2, 0.2))
```

And this is our log marginal posterior predictive probability:

```@example abstractgps_tutorialnd
logpdf_surrogate(gp_surrogate)
```
