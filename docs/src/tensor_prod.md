# Tensor product function
The tensor product function is defined as:
``f(x) = \prod_{i=1}^d cos(a\pi x_i)``

Let's import Surrogates and Plots:
```@example tensor
using Surrogates
using Plots
default()
```

Define the 1D objective function:
```@example tensor
function f(x,a)
    return cos(a*pi*x)
end
```

```@example tensor
n = 30
lb = -5.0
ub = 5.0
a = 0.5
x = sample(n,lb,ub,SobolSample())
y = f.(x)
xs = lb:0.001:ub
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lb, ub), ylims=(-1, 1), legend=:top)
plot!(xs,f.(xs,a), label="True function", legend=:top)
```

Fitting and plotting different surrogates:
```@example tensor
gek = GEK(x,y,lb,ub)
loba_1 = LobacheskySurrogate(x,y,lb,ub)
krig = Kriging(x,y,lb,ub)
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lb, ub), ylims=(-2.5, 2.5), legend=:bottom)
plot!(xs,f.(xs,a), label="True function", legend=:top)
plot!(xs, loba_1.(xs), label="Lobachesky", legend=:top)
plot!(xs, krig.(xs), label="Kriging", legend=:top)
plot!(xs, gek.(xs), label="GEK", legend=:top)
```
