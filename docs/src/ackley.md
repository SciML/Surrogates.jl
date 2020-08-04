# Ackley function

The Ackley function is defined as:
``f(x) = -a*exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}) - exp(\frac{1}{d} \sum_{i=1}^d cos(cx_i)) + a + exp(1)``
Usually the recommended values are: ``a =  20``, ``b = 0.2`` and ``c =  2\pi``

Let's see the 1D case.

```@example ackley
using Surrogates
using Plots
default()
```


```@example ackley
n = 100
lb = -32.768
ub = 32.768
x = sample(n,lb,ub,SobolSample())
y = f.(x)
xs = lb:0.001:ub
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lb, ub), ylims=(0,30), legend=:top)
plot!(xs,f.(xs), label="True function", legend=:top)
```

```@example ackley
my_rad = RadialBasis(x,y,lb,ub)
my_krig = Kriging(x,y,lb,ub)
my_loba = LobacheskySurrogate(x,y,lb,ub)
```

```@example ackley
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lb, ub), ylims=(0, 30), legend=:top)
plot!(xs,f.(xs), label="True function", legend=:top)
plot!(xs, my_rad.(xs), label="Polynomial expansion", legend=:top)
plot!(xs, my_krig.(xs), label="Lobachesky", legend=:top)
plot!(xs, my_loba.(xs), label="Kriging", legend=:top)
```

The fit looks good. Let's now see if we are able to find the minimum value using
optimization methods:

```@example ackley
surrogate_optimize(f,DYCORS(),lb,ub,my_rad,UniformSample())
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lb, ub), ylims=(0, 30), legend=:top)
plot!(xs,f.(xs), label="True function", legend=:top)
plot!(xs, my_rad.(xs), label="Radial basis optimized", legend=:top)
```

The DYCORS methods successfully finds the minimum. 
