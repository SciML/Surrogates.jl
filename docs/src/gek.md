# Gradient Enhanced Kriging Surrogate Tutorial

Gradient-enhanced Kriging extends Kriging with derivative observations. Because the
Gaussian kernel is mean-square differentiable, the joint covariance of a process and
its partial derivatives is available in closed form, so a gradient can be treated as
`d` extra observations rather than as a separate model. That is usually more accurate
than Kriging at the same number of sample points, and it is what makes GEK attractive
when gradients come cheaply — from an adjoint solver or from automatic differentiation.

The cost is the size of the system. With `n` points in `d` dimensions the covariance
matrix is `n(1 + d) × n(1 + d)`, so it grows with the number of inputs *and* with the
number of samples, and it is far worse conditioned than the plain correlation matrix
— see the conditioning note in the docstring below. [`GEKPLS`](@ref) is the indirect
alternative for high dimensions.

```@docs
GEK
```

Let's have a look at the following function to use Gradient Enhanced Surrogate:
``f(x) = x^3 - 6x^2 + 4x + 12``

First of all, we will import `Surrogates` and `Plots` packages:

```@example GEK1D
using Surrogates
using Plots
```

## Sampling

We choose to sample f in 100 points between 2 and 10 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example GEK1D
n_samples = 100
lower_bound = 2
upper_bound = 10
xs = lower_bound:0.001:upper_bound
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
f(x) = x^3 - 6x^2 + 4x + 12
der(x) = 3 * x^2 - 12 * x + 4
y1 = f.(x)
y2 = der.(x)
scatter(x, y1, label = "Sampled points", xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
```

`GEK` takes all the observations in one vector: every function value first, then
every derivative. In one dimension that is simply

```@example GEK1D
y = vcat(y1, y2)
length(y) == 2 * n_samples
```

## Building a surrogate

With our sampled points, we can build the Gradient Enhanced Kriging surrogate using the `GEK` function.

```@example GEK1D
my_gek = GEK(x, y, lower_bound, upper_bound, theta = 0.3)

scatter(x, y1, label = "Sampled points", xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(my_gek, label = "Surrogate function", ribbon = p -> std_error_at_point(my_gek, p),
    xlims = (lower_bound, upper_bound), legend = :top)
```

# Gradient Enhanced Kriging Surrogate Tutorial (ND)

First of all, let's define the function we are going to build a surrogate for.

```@example GEK_ND
using Plots
using Surrogates
```

Now, let's define the function:

```@example GEK_ND
function leon(x)
    x1 = x[1]
    x2 = x[2]
    term1 = (x2 - x1^3)^2
    term2 = (1 - x1)^2
    y = term1 + term2
end
```

## Sampling

Let's define our bounds, this time we are working in two dimensions. In particular, we want our first dimension `x` to have bounds `0, 1`, and `0, 1` for the second dimension. We are taking 100 samples of the space using Sobol Sequences. We then evaluate our function on all the sampling points.

```@example GEK_ND
n_samples = 100
lower_bound = [0, 0]
upper_bound = [1, 1]
xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
y1 = leon.(xys)
```

```@example GEK_ND
xgrid, ygrid = 0:0.05:1, 0:0.05:1
p1 = surface(xgrid, ygrid, (x1, x2) -> leon((x1, x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, y1)
p2 = contour(xgrid, ygrid, (x1, x2) -> leon((x1, x2)))
scatter!(xs, ys)
plot(p1, p2, title = "True function")
```

## Building a surrogate

Using the sampled points, we build the surrogate, the steps are analogous to the 1-dimensional case.

In `d` dimensions the observation vector holds all `n` function values, then each
point's `d` partial derivatives in coordinate order:

```math
[\, f(x_1),\ \dots,\ f(x_n),\ \partial_1 f(x_1),\ \dots,\ \partial_d f(x_1),\ \partial_1 f(x_2),\ \dots,\ \partial_d f(x_n) \,]
```

so flattening a vector of gradients in order produces exactly the right layout.

```@example GEK_ND
grad(x) = (2 * (x[2] - x[1]^3) * (-3x[1]^2) - 2 * (1 - x[1]), 2 * (x[2] - x[1]^3))
y2 = reduce(vcat, collect.(grad.(xys)))
y = vcat(y1, y2)
length(y) == n_samples * (1 + 2)
```

Left unset, `theta` is fitted by maximum likelihood over all `n(1 + d)` observations,
exactly as [`Kriging`](@ref) fits it over the `n` function values alone.

```@example GEK_ND
my_GEK = GEK(xys, y, lower_bound, upper_bound)
my_GEK.theta
```

```@example GEK_ND
p1 = surface(xgrid, ygrid, (x1, x2) -> my_GEK((x1, x2)))
scatter!(xs, ys, y1, marker_z = y1)
p2 = contour(xgrid, ygrid, (x1, x2) -> my_GEK((x1, x2)))
scatter!(xs, ys, marker_z = y1)
plot(p1, p2, title = "Surrogate")
```
