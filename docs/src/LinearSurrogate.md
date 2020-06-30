## Linear Surrogate
Linear Surrogate is a linear approach to modeling the relationship between a scalar response or dependent variable and one or more explanatory variables. We will use Linear Surrogate to optimize following function:

$f(x) = sin(x) + log(x)$.

First of all we have to import these two packages: `Surrogates` and `Plots`.

```@example linear_surrogate1D
using Surrogates
using Plots
default()
```

### Sampling

We choose to sample f in 20 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example linear_surrogate1D
f(x) = sin(x) + log(x)
n_samples = 20
lower_bound = 5.2
upper_bound = 12.5
x = sample(n_samples, lower_bound, upper_bound, sobolSample())
y = f.(x)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", xlims=(lower_bound, upper_bound))
```

## Building a Surrogate

With our sampled points we can build the **Linear Surrogate** using the `LinearSurrogate` function.

We can simply calculate `linear_surrogate` for any value.

```@example linear_surrogate1D
my_linear_surr_1D = LinearSurrogate(x, y, lower_bound, upper_bound)
add_point!(my_linear_surr_1D,4.0,7.2)
add_point!(my_linear_surr_1D,[5.0,6.0],[8.3,9.7])
val = my_linear_surr_1D(5.0)
```

Now, we will simply plot `linear_surrogate`:

```@example linear_surrogate1D
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(my_linear_surr_1D, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```

## Optimizing

Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example linear_surrogate1D
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, my_linear_surr_1D, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(my_linear_surr_1D, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
