## Salustowicz Benchmark Function

The true underlying function HyGP had to approximate is the 1D Salustowicz function. The function can be evaluated in the given domain:
``x \in [0, 10]``.

The Salustowicz benchmark function is as follows:

``f(x) = e^(-x) * x^3 * cos(x) * sin(x) * (cos(x) * sin(x)*sin(x) - 1)``

Let's import these two packages  `Surrogates` and `Plots`:

```@example salustowicz1D
using Surrogates
using Plots
default()
```

Now, let's define our objective function:

```@example salustowicz1D
function salustowicz(x)
    term1 = 2.72^(-x) * x^3 * cos(x) * sin(x);
    term2 = (cos(x) * sin(x)*sin(x) - 1);
    y = term1 * term2;
end
```

Let's sample f in 30 points between 0 and 10 using the `sample` function. The sampling points are chosen using a Sobol Sample, this can be done by passing `SobolSample()` to the `sample` function.

```@example salustowicz1D
n_samples = 30
lower_bound = 0
upper_bound = 10
num_round = 2
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = salustowicz.(x)
xs = lower_bound:0.001:upper_bound
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(xs, salustowicz.(xs), label="True function", legend=:top)
```

Now, let's fit Salustowicz Function with different Surrogates:

```@example salustowicz1D
InverseDistance = InverseDistanceSurrogate(x, y, lower_bound, upper_bound)
randomforest_surrogate = RandomForestSurrogate(x ,y ,lower_bound, upper_bound, num_round = 2)
lobachevsky_surrogate = LobacheskySurrogate(x, y, lower_bound, upper_bound, alpha = 2.0, n = 6)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:topright)
plot!(xs, salustowicz.(xs), label="True function", legend=:topright)
plot!(xs, InverseDistance.(xs), label="InverseDistanceSurrogate", legend=:topright)
plot!(xs, randomforest_surrogate.(xs), label="RandomForest", legend=:topright)
plot!(xs, lobachevsky_surrogate.(xs), label="Lobachesky", legend=:topright)
```

Not's let's see Kriging Surrogate with different hyper parameter:

```@example salustowicz1D
kriging_surrogate1 = Kriging(x, y, lower_bound, upper_bound, p=0.9);
kriging_surrogate2 = Kriging(x, y, lower_bound, upper_bound, p=1.5);
kriging_surrogate3 = Kriging(x, y, lower_bound, upper_bound, p=1.9);
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(xs, salustowicz.(xs), label="True function", legend=:top)
plot!(xs, kriging_surrogate1.(xs), label="kriging_surrogate1", ribbon=p->std_error_at_point(kriging_surrogate1, p), legend=:top)
plot!(xs, kriging_surrogate2.(xs), label="kriging_surrogate2", ribbon=p->std_error_at_point(kriging_surrogate2, p), legend=:top)
plot!(xs, kriging_surrogate3.(xs), label="kriging_surrogate3", ribbon=p->std_error_at_point(kriging_surrogate3, p), legend=:top)
```
