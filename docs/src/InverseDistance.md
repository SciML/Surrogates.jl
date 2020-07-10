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
