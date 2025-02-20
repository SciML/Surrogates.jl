## Linear Surrogate

Linear Surrogate is a linear approach to modeling the relationship between a scalar response or dependent variable and one or more explanatory variables. We will use Linear Surrogate to optimize following function:

$f(x) = \sin(x) + \log(x)$

First of all we have to import these two packages: `Surrogates` and `Plots`.

```@example linear_surrogate1D
using Surrogates
using Plots
```

### Sampling

We choose to sample f in 100 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example linear_surrogate1D
f(x) = 2*x+10.0
n_samples = 100
lower_bound = 5.2
upper_bound = 12.5
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label = "Sampled points", xlims = (lower_bound, upper_bound))
plot!(f, label = "True function", xlims = (lower_bound, upper_bound))
```

## Building a Surrogate

With our sampled points, we can build the **Linear Surrogate** using the `LinearSurrogate` function.

We can simply calculate `linear_surrogate` for any value.

```@example linear_surrogate1D
my_linear_surr_1D = LinearSurrogate(x, y, lower_bound, upper_bound)
val = my_linear_surr_1D(5.0)
```

Now, we will simply plot `linear_surrogate`:

```@example linear_surrogate1D
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound))
plot!(f, label = "True function", xlims = (lower_bound, upper_bound))
plot!(my_linear_surr_1D, label = "Surrogate function", xlims = (lower_bound, upper_bound))
```

## Optimizing

Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as the optimization technique and again Sobol sampling as the sampling technique.

```@example linear_surrogate1D
surrogate_optimize(
    f, SRBF(), lower_bound, upper_bound, my_linear_surr_1D, SobolSample())
scatter(x, y, label = "Sampled points")
plot!(f, label = "True function", xlims = (lower_bound, upper_bound))
plot!(my_linear_surr_1D, label = "Surrogate function", xlims = (lower_bound, upper_bound))
```

## Linear Surrogate tutorial (ND)

First of all we will define the `Egg Holder` function we are going to build a surrogate for. Notice, one how its argument is a vector of numbers, one for each coordinate, and its output is a scalar.

```@example linear_surrogateND
using Plots
default(c = :matter, legend = false, xlabel = "x", ylabel = "y")
using Surrogates

function egg(x)
    x1 = x[1]
    x2 = x[2]
    term1 = -(x2 + 47) * sin(sqrt(abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * sin(sqrt(abs(x1 - (x2 + 47))))
    y = term1 + term2
end
```

### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-10, 5`, and `0, 15` for the second dimension. We are taking 100 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example linear_surrogateND
n_samples = 100
lower_bound = [-10.0, 0.0]
upper_bound = [5.0, 15.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = egg.(xys)
```

```@example linear_surrogateND
x, y = -10:5, 0:15
p1 = surface(x, y, (x1, x2) -> egg((x1, x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, zs)
p2 = contour(x, y, (x1, x2) -> egg((x1, x2)))
scatter!(xs, ys)
plot(p1, p2, title = "True function")
```

### Building a surrogate

Using the sampled points, we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example linear_surrogateND
my_linear_ND = LinearSurrogate(xys, zs, lower_bound, upper_bound)
```

```@example linear_surrogateND
p1 = surface(x, y, (x, y) -> my_linear_ND([x y]))
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> my_linear_ND([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2, title = "Surrogate")
```

### Optimizing

With our surrogate, we can now search for the minima of the function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example linear_surrogateND
size(xys)
```

```@example linear_surrogateND
surrogate_optimize(
    egg, SRBF(), lower_bound, upper_bound, my_linear_ND, SobolSample(), maxiters = 10)
```

```@example linear_surrogateND
size(xys)
```

```@example linear_surrogateND
p1 = surface(x, y, (x, y) -> my_linear_ND([x y]))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
zs = egg.(xys)
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> my_linear_ND([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2)
```
