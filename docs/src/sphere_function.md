# Sphere function

The sphere function of dimension d is defined as:
``f(x) = \sum_{i=1}^d x_i^2``
with lower bound -10 and upper bound 10.

Let's import Surrogates and Plots:

```@example sphere_function
using Surrogates
using Plots
default()
```

Define the objective function:

```@example sphere_function
function sphere_function(x)
    return sum(x .^ 2)
end
```

The 1D case is just a simple parabola, let's plot it:

```@example sphere_function
n = 20
lb = -10
ub = 10
x = sample(n, lb, ub, SobolSample())
y = sphere_function.(x)
xs = lb:0.001:ub
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), ylims = (-2, 120), legend = :top)
plot!(xs, sphere_function.(xs), label = "True function", legend = :top)
```

Fitting RadialSurrogate with different radial basis:

```@example sphere_function
rad_1d_linear = RadialBasis(x, y, lb, ub)
rad_1d_cubic = RadialBasis(x, y, lb, ub, rad = cubicRadial())
rad_1d_multiquadric = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), ylims = (-2, 120), legend = :top)
plot!(xs, sphere_function.(xs), label = "True function", legend = :top)
plot!(xs, rad_1d_linear.(xs), label = "Radial surrogate with linear", legend = :top)
plot!(xs, rad_1d_cubic.(xs), label = "Radial surrogate with cubic", legend = :top)
plot!(xs, rad_1d_multiquadric.(xs),
    label = "Radial surrogate with multiquadric", legend = :top)
```

Fitting Lobachevsky Surrogate with different values of hyperparameter alpha:

```@example sphere_function
loba_1 = LobachevskySurrogate(x, y, lb, ub)
loba_2 = LobachevskySurrogate(x, y, lb, ub, alpha = 1.5, n = 6)
loba_3 = LobachevskySurrogate(x, y, lb, ub, alpha = 0.3, n = 6)
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), ylims = (-2, 120), legend = :top)
plot!(xs, sphere_function.(xs), label = "True function", legend = :top)
plot!(xs, loba_1.(xs), label = "Lobachevsky surrogate 1", legend = :top)
plot!(xs, loba_2.(xs), label = "Lobachevsky surrogate 2", legend = :top)
plot!(xs, loba_3.(xs), label = "Lobachevsky surrogate 3", legend = :top)
```
