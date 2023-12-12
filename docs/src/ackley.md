# Ackley Function

The Ackley function is defined as:
``f(x) = -a*\exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}) - \exp(\frac{1}{d} \sum_{i=1}^d \cos(cx_i)) + a + \exp(1)``
Usually the recommended values are: ``a =  20``, ``b = 0.2`` and ``c =  2\pi``

Let's see the 1D case.

```@example ackley
using Surrogates
using Plots
default()
```

Now, let's define the `Ackley` function:

```@example ackley
function ackley(x)
    a, b, c = 20.0, 0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += i^2
    end
    return (-a * exp(-b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + 2.71)
end
```


```@example ackley
n = 100
lb = -32.768
ub = 32.768
x = sample(n, lb, ub, SobolSample())
y = ackley.(x)
xs = lb:0.001:ub
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(0,30), legend=:top)
plot!(xs, ackley.(xs), label="True function", legend=:top)
```

```@example ackley
my_rad = RadialBasis(x, y, lb, ub)
my_loba = LobachevskySurrogate(x, y, lb, ub)
```

```@example ackley
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(0, 30), legend=:top)
plot!(xs, ackley.(xs), label="True function", legend=:top)
plot!(xs, my_rad.(xs), label="Polynomial expansion", legend=:top)
plot!(xs, my_loba.(xs), label="Lobachevsky", legend=:top)

```

The fit looks good. Let's now see if we are able to find the minimum value using
optimization methods:

```@example ackley
surrogate_optimize(ackley,DYCORS(),lb,ub,my_rad,RandomSample())
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(0, 30), legend=:top)
plot!(xs, ackley.(xs), label="True function", legend=:top)
plot!(xs, my_rad.(xs), label="Radial basis optimized", legend=:top)
```

The DYCORS methods successfully finds the minimum.
