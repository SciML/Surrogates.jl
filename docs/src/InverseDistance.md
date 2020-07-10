The **Inverse Distance Surrogate** is an interpolating method and in this method the unknown points are calculated with a weighted average of the sampling points. This model uses the inverse distance between the unknown and training points to predict the unknown point. We do not need to fit this model because the response of an unknown point x is computed with respect to the distance between x and the training points.

Let's optimize following function to use Inverse Distance Surrogate:

$f(x) = sin(x) + sin(x)^2 + sin(x)^3$.

First of all, we have to import these two packages: `Surrogates` and `Plots`.

```@example Inverse_Distance1D
using Surrogates
using Plots
default()
```


### Sampling

We choose to sample f in 25 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Low Discrepancy, this can be done by passing `LowDiscrepancySample()` to the `sample` function.

```@example Inverse_Distance1D
f(x) = sin(x) + sin(x)^2 + sin(x)^3

n_samples = 25
lower_bound = 0.0
upper_bound = 10.0
x = sample(n_samples, lower_bound, upper_bound, LowDiscrepancySample(2))
y = f.(x)

scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", xlims=(lower_bound, upper_bound))
```


## Building a Surrogate

With our sampled points we can build the **Inverse Distance Surrogate** using the `InverseDistance` function.

We can simply calculate `InverseDistance` for any value.

```@example Inverse_Distance1D
InverseDistance = InverseDistanceSurrogate(x,y,lb,ub)
add_point!(InverseDistance,5.0,-0.91)
add_point!(InverseDistance,[5.1,5.2],[1.0,2.0])
prediction = InverseDistance(5.0)
```

Now, we will simply plot `InverseDistance`:

```@example Inverse_Distance1D
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(InverseDistance, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```


## Optimizing

Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example Inverse_Distance1D
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, InverseDistance, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(InverseDistance, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
