# Tensor product function
The tensor product function is defined as:
``f(x) = \prod_{i=1}^d \cos(a\pi x_i)``

Let's import Surrogates and Plots:
```@example tensor
using Surrogates
using Plots
default()
```

Define the 1D objective function:
```@example tensor
function f(x)
    a = 0.5;
    return cos(a*pi*x)
end
```

```@example tensor
n = 30
lb = -5.0
ub = 5.0
a = 0.5
x = sample(n, lb, ub, SobolSample())
y = f.(x)
xs = lb:0.001:ub
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(-1, 1), legend=:top)
plot!(xs, f.(xs), label="True function", legend=:top)
```

Fitting and plotting different surrogates:
```@example tensor
loba_1 = LobachevskySurrogate(x, y, lb, ub)
krig = Kriging(x, y, lb, ub)
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(-2.5, 2.5), legend=:bottom)
plot!(xs,f.(xs), label="True function", legend=:top)
plot!(xs, loba_1.(xs), label="Lobachevsky", legend=:top)
plot!(xs, krig.(xs), label="Kriging", legend=:top)
```
