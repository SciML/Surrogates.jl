## Gramacy & Lee Function

Gramacy & Lee Function is a continues function. It is not convex. The function is defined on 1-dimensional space. It is an unimodal. The function can be defined on any input domain but it is usually evaluated on
``x \in [-0.5, 2.5]``.

The Gramacy & Lee is as follows:
``f(x) = \frac{sin(10\pi x)}{2x} + (x-1)^4``.

Let's import these two packages `Surrogates` and `Plots`:

```@example gramacylee1D
using Surrogates
using Plots
default()
```

Now, let's define our objective function:

```@example gramacylee1D
function gramacylee(x)
    term1 = sin(10*pi*x) / 2*x;
    term2 = (x - 1)^4;
    y = term1 + term2;
end
```

Let's sample f in 25 points between -0.5 and 2.5 using the `sample` function. The sampling points are chosen using a Sobol Sample, this can be done by passing `SobolSample()` to the `sample` function.

```@example gramacylee1D
n = 25
lower_bound = -0.5
upper_bound = 2.5
x = sample(n, lower_bound, upper_bound, SobolSample())
y = gramacylee.(x)
xs = lower_bound:0.001:upper_bound
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-5, 20), legend=:top)
plot!(xs, gramacylee.(xs), label="True function", legend=:top)
```

Now, let's fit Gramacy & Lee Function with different Surrogates:

```@example gramacylee1D
my_pol = PolynomialChaosSurrogate(x, y, lower_bound, upper_bound)
loba_1 = LobachevskySurrogate(x, y, lower_bound, upper_bound)
krig = Kriging(x, y, lower_bound, upper_bound)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-5, 20), legend=:top)
plot!(xs, gramacylee.(xs), label="True function", legend=:top)
plot!(xs, my_pol.(xs), label="Polynomial expansion", legend=:top)
plot!(xs, loba_1.(xs), label="Lobachevsky", legend=:top)
plot!(xs, krig.(xs), label="Kriging", legend=:top)
```
