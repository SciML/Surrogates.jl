# Gramacy & Lee Function

The Gramacy & Lee function is a one-dimensional, continuous, non-convex
benchmark. It is **multimodal**: the ``\sin(10\pi x)`` factor puts thirteen local
minima inside the usual evaluation domain
``x \in [-0.5, 2.5]``, which is what makes it a useful test for a surrogate's
ability to resolve fine structure from few samples.

The Gramacy & Lee function is as follows:
``f(x) = \frac{\sin(10\pi x)}{2x} + (x-1)^4``.

Note that it has a removable singularity at ``x = 0``: the limit is
``5\pi \approx 15.7``, but evaluating the expression there divides zero by zero.
The plotting grid below steps around it.

Let's import these two packages `Surrogates` and `Plots`:

```@example gramacylee1D
using Surrogates
using PolyChaos
using Plots
```

Now, let's define our objective function:

```@example gramacylee1D
function gramacylee(x)
    term1 = sin(10 * pi * x) / (2 * x)
    term2 = (x - 1)^4
    return term1 + term2
end
```

Let's sample f in 25 points between -0.5 and 2.5 using the `sample` function. The sampling points are chosen using a Sobol sample, this can be done by passing `SobolSample()` to the `sample` function.

```@example gramacylee1D
n = 25
lower_bound = -0.5
upper_bound = 2.5
x = sample(n, lower_bound, upper_bound, SobolSample())
y = gramacylee.(x)
xs = (lower_bound + 0.0005):0.001:upper_bound
scatter(x, y, label = "Sampled points", xlims = (lower_bound, upper_bound),
    ylims = (-5, 20), legend = :top)
plot!(xs, gramacylee.(xs), label = "True function", legend = :top)
```

Now, let's fit Gramacy & Lee function with different surrogates:

```@example gramacylee1D
my_pol = PolynomialChaosSurrogate(x, y, lower_bound, upper_bound)
loba_1 = LobachevskySurrogate(x, y, lower_bound, upper_bound)
krig = Kriging(x, y, lower_bound, upper_bound)
scatter(x, y, label = "Sampled points", xlims = (lower_bound, upper_bound),
    ylims = (-5, 20), legend = :top)
plot!(xs, gramacylee.(xs), label = "True function", legend = :top)
plot!(xs, my_pol.(xs), label = "Polynomial expansion", legend = :top)
plot!(xs, loba_1.(xs), label = "Lobachevsky", legend = :top)
plot!(xs, krig.(xs), label = "Kriging", legend = :top)
```
