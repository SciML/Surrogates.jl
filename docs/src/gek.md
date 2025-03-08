# Gradient Enhanced Kriging Surrogate Tutorial

Gradient-enhanced Kriging is an extension of kriging which supports gradient information. GEK is usually more accurate than kriging. However, it is not computationally efficient when the number of inputs, the number of sampling points, or both, are high. This is mainly due to the size of the corresponding correlation matrix, which increases proportionally with both the number of inputs and the number of sampling points.

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
y1 = f.(x)
der = x -> 3 * x^2 - 12 * x + 4
y2 = der.(x)
y = vcat(y1, y2)
scatter(x, y1, label = "Sampled points", xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
```

## Building a surrogate

With our sampled points, we can build the Gradient Enhanced Kriging surrogate using the `GEK` function.

```@example GEK1D
my_gek = GEK(x, y, lower_bound, upper_bound, p = 0.03, theta = 0.3)

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
x, y = 0:1, 0:1
p1 = surface(x, y, (x1, x2) -> leon((x1, x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, y1)
p2 = contour(x, y, (x1, x2) -> leon((x1, x2)))
scatter!(xs, ys)
plot(p1, p2, title = "True function")
```

## Building a surrogate

Using the sampled points, we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example GEK_ND
grad1 = x -> 2 * (x[2] - x[1]^3) * (-3x[1]^2) - 2 * (1 - x[1])
grad2 = x -> 2 * (x[2] - x[1]^3)
d = 2
n = 100
function create_grads(n, d, grad1, grad2, y1)
    c = 0
    y2 = zeros(eltype(y1[1]), n * d)
    for i in 1:n
        y2[i + c] = grad1(xys[i])
        y2[i + c + 1] = grad2(xys[i])
        c = c + 1
    end
    return y2
end
y2 = create_grads(n, d, grad1, grad2, y1)
y = vcat(y1, y2)
```

```@example GEK_ND
my_GEK = GEK(xys, y, lower_bound, upper_bound)
```

```@example GEK_ND
p1 = surface(x, y, (x, y) -> my_GEK([x y]))
scatter!(xs, ys, y1, marker_z = y1)
p2 = contour(x, y, (x, y) -> my_GEK([x y]))
scatter!(xs, ys, marker_z = y1)
plot(p1, p2, title = "Surrogate")
```
