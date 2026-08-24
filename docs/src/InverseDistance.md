# InverseDistance Surrogate Tutorial

The **Inverse Distance Surrogate** (Shepard's method) is an interpolating method, and in this method, the unknown points are calculated with a weighted average of the sampling points. This model uses the inverse distance between the unknown and training points to predict the unknown point. We do not need to fit this model because the response of an unknown point x is computed with respect to the distance between x and the training points.

The single hyperparameter is the exponent `p`, which must be positive. Each sample point is weighted by `1 / distance^p`, so a larger `p` concentrates the weight on the nearest samples and pushes the surrogate towards nearest-neighbour interpolation; a smaller `p` spreads the weight out and flattens the surrogate towards the mean of the responses. The prediction is a weighted average of the sampled responses, so it never leaves their range.

```@docs
InverseDistanceSurrogate
```

Let's optimize the following function to use Inverse Distance Surrogate:

$f(x) = sin(x) + sin(x)^2 + sin(x)^3$.

First of all, we have to import these two packages: `Surrogates` and `Plots`.

```@example Inverse_Distance1D
using Surrogates
using Plots
```

### Sampling

We choose to sample f in 100 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Low Discrepancy, this can be done by passing `HaltonSample()` to the `sample` function.

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

### Choosing the exponent

The default `p = 1.0` still gives distant samples enough weight to pull the
surrogate towards the mean of the responses between the sampled points. Raising
`p` concentrates the weight on the nearest samples:

```@example Inverse_Distance1D
plot(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
for p in [1.0, 2.0, 6.0]
    surr = InverseDistanceSurrogate(x, y, lower_bound, upper_bound, p = p)
    plot!(surr, label = "p = $p", xlims = (lower_bound, upper_bound))
end
plot!()
```

The flattening at `p = 1` is easy to quantify. Away from the sampled points,
compare how far the surrogate strays from the true function with how much of the
function's own spread about the mean response it keeps:

```@example Inverse_Distance1D
grid = range(0.02, 9.98, length = 500)
ybar = sum(y) / length(y)
rms(v) = sqrt(sum(abs2, v) / length(v))

for p in [1.0, 2.0, 6.0]
    surr = InverseDistanceSurrogate(x, y, lower_bound, upper_bound, p = p)
    println("p = ", p,
        "  RMSE vs f = ", round(rms([surr(v) - f(v) for v in grid]), digits = 4),
        "  spread kept = ", round(rms([surr(v) - ybar for v in grid]), digits = 4))
end
println("f's own spread about ybar = ",
    round(rms([f(v) - ybar for v in grid]), digits = 4))
```

Whatever `p` is, the surrogate interpolates: at a sampled point it returns the
sampled response exactly.

```@example Inverse_Distance1D
maximum(abs(InverseDistance(x[i]) - y[i]) for i in eachindex(x))
```

## Optimizing

Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize!` method. We choose to use Stochastic RBF as the optimization technique and again Sobol sampling as the sampling technique.

```@example Inverse_Distance1D
surrogate_optimize!(
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

Notice how the new points sampled during the optimization process are added to
the surrogate. The `xys` array we built it from is left untouched, so it is the
surrogate's own sample list whose size changes.

```@example Inverse_DistanceND
length(xys), length(InverseDistance.x)
```

```@example Inverse_DistanceND
surrogate_optimize!(schaffer, SRBF(), lower_bound, upper_bound,
    InverseDistance, SobolSample(), maxiters = 10)
```

```@example Inverse_DistanceND
length(xys), length(InverseDistance.x)
```

```@example Inverse_DistanceND
p1 = surface(x, y, (x, y) -> InverseDistance([x y]))
xs = [xy[1] for xy in InverseDistance.x]
ys = [xy[2] for xy in InverseDistance.x]
zs = schaffer.(InverseDistance.x)
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> InverseDistance([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2)
```
