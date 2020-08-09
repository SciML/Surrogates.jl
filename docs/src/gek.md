## Gradient Enhanced Kriging

Gradient-enhanced Kriging is an extension of kriging which supports gradient information. GEK is usually more accurate than kriging, however, it is not computationally efficient when the number of inputs, the number of sampling points, or both, are high. This is mainly due to the size of the corresponding correlation matrix that increases proportionally with both the number of inputs and the number of sampling points.

Let's have a look to the following function to use Gradient Enhanced Surrogate:
``f(x) = sin(x) + 2*x^2``

First of all, we will import `Surrogates` and `Plots` packages:

```@example GEK1D
using Surrogates
using Plots
default()
```

### Sampling

We choose to sample f in 8 points between 0 to 1 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example GEK1D
n_samples = 10
lower_bound = 2
upper_bound = 10
xs = lower_bound:0.001:upper_bound
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
f(x) = x^3 - 6x^2 + 4x + 12
y1 = f.(x)
der = x -> 3*x^2 - 12*x + 4
y2 = der.(x)
y = vcat(y1,y2)
scatter(x, y1, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function", xlims=(lower_bound, upper_bound), legend=:top)
```

### Building a surrogate

With our sampled points we can build the Gradient Enhanced Kriging surrogate using the `GEK` function.

```@example GEK1D
my_gek = GEK(x, y1, lower_bound, upper_bound, p = 2.9);
```
```@example @GEK1D
plot(x, y1, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(my_gek, label="Surrogate function", ribbon=p->std_error_at_point(my_gek, p), xlims=(lower_bound, upper_bound), legend=:top)
```


## Optimizing

Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example GEK1D
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, my_gek, SobolSample())
plot(x, y1, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(my_gek, label="Surrogate function", ribbon=p->std_error_at_point(my_gek, p), xlims=(lower_bound, upper_bound), legend=:top)
```
