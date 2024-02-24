# Lp norm function

The Lp norm function is defined as:
``f(x) = \sqrt[p]{ \sum_{i=1}^d \vert x_i \vert ^p}``

Let's import Surrogates and Plots:

```@example lp
using Surrogates
using SurrogatesPolyChaos
using Plots
using LinearAlgebra
default()
```

Define the objective function:

```@example lp
function f(x, p)
    return norm(x, p)
end
```

Let's see a simple 1D case:

```@example lp
n = 30
lb = -5.0
ub = 5.0
p = 1.3
x = sample(n, lb, ub, SobolSample())
y = f.(x, p)
xs = lb:0.001:ub
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), ylims = (0, 5), legend = :top)
plot!(xs, f.(xs, p), label = "True function", legend = :top)
```

Fitting different surrogates:

```@example lp
my_pol = PolynomialChaosSurrogate(x, y, lb, ub)
loba_1 = LobachevskySurrogate(x, y, lb, ub)
krig = Kriging(x, y, lb, ub)
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), ylims = (0, 5), legend = :top)
plot!(xs, f.(xs, p), label = "True function", legend = :top)
plot!(xs, my_pol.(xs), label = "Polynomial expansion", legend = :top)
plot!(xs, loba_1.(xs), label = "Lobachevsky", legend = :top)
plot!(xs, krig.(xs), label = "Kriging", legend = :top)
```
