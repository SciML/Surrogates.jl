# InverseDistance Surrogate Tutorial

The **Inverse Distance Surrogate** is an interpolating method, and in this method, the unknown points are calculated with a weighted average of the sampling points. This model uses the inverse distance between the unknown and training points to predict the unknown point. We do not need to fit this model because the response of an unknown point x is computed with respect to the distance between x and the training points.

Let's optimize the following function to use Inverse Distance Surrogate:

$f(x) = sin(x) + sin(x)^2 + sin(x)^3$.

First of all, we have to import these two packages: `Surrogates` and `Plots`.

```@example Inverse_Distance1D
using Surrogates
using Plots
```

### Sampling

We choose to sample f in 1000 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Low Discrepancy, this can be done by passing `HaltonSample()` to the `sample` function.

```@example Inverse_Distance1D
f(x) = sin(x) + sin(x)^2 + sin(x)^3

n_samples = 100
lower_bound = 0.0
upper_bound = 10.0
x = sample(n_samples, lower_bound, upper_bound, HaltonSample())
y = f.(x)

scatter(x, y, label = "Sampled points", xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
```

## Building a Surrogate

```@example Inverse_Distance1D
InverseDistance = InverseDistanceSurrogate(x, y, lower_bound, upper_bound)
prediction = InverseDistance(5.0)
```

Now, we will simply plot `InverseDistance`:

```@example Inverse_Distance1D
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(InverseDistance, label = "Surrogate function",
    xlims = (lower_bound, upper_bound), legend = :top)
```

## Optimizing

Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as the optimization technique and again Sobol sampling as the sampling technique.

```@example Inverse_Distance1D
surrogate_optimize(
    f, SRBF(), lower_bound, upper_bound, InverseDistance, SobolSample())
scatter(x, y, label = "Sampled points", legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(InverseDistance, label = "Surrogate function",
    xlims = (lower_bound, upper_bound), legend = :top)
```

## Inverse Distance Surrogate Tutorial (ND):

First of all we will define the `Schaffer` function we are going to build a surrogate for. Notice, how its argument is a vector of numbers, one for each coordinate, and its output is a scalar.

```@example Inverse_DistanceND
using Plots
default(c = :matter, legend = false, xlabel = "x", ylabel = "y")
using Surrogates

function schaffer(x)
    x1 = x[1]
    x2 = x[2]
    fact1 = (sin(x1^2 - x2^2))^2 - 0.5
    fact2 = (1 + 0.001 * (x1^2 + x2^2))^2
    y = 0.5 + fact1 / fact2
end
```

### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-5, 10`, and `0, 15` for the second dimension. We are taking 100 samples of the space using Sobol Sequences. We then evaluate our function on all the sampling points.

```@example Inverse_DistanceND
n_samples = 100
lower_bound = [-5.0, 0.0]
upper_bound = [10.0, 15.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = schaffer.(xys);
```

```@example Inverse_DistanceND
x, y = -5:10, 0:15
p1 = surface(x, y, (x1, x2) -> schaffer((x1, x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, zs)
p2 = contour(x, y, (x1, x2) -> schaffer((x1, x2)))
scatter!(xs, ys)
plot(p1, p2, title = "True function")
```

### Building a surrogate

Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example Inverse_DistanceND
InverseDistance = InverseDistanceSurrogate(xys, zs, lower_bound, upper_bound)
```

```@example Inverse_DistanceND
p1 = surface(x, y, (x, y) -> InverseDistance([x y]))
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> InverseDistance([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2, title = "Surrogate")
```

### Optimizing

With our surrogate, we can now search for the minima of the function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example Inverse_DistanceND
size(xys)
```

```@example Inverse_DistanceND
surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound,
    InverseDistance, SobolSample(), maxiters = 10)
```

```@example Inverse_DistanceND
size(xys)
```

```@example Inverse_DistanceND
p1 = surface(x, y, (x, y) -> InverseDistance([x y]))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
zs = schaffer.(xys)
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> InverseDistance([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2)
```
