# Wendland Surrogate Tutorial

The Wendland surrogate uses a compactly supported radial kernel: a sample point
influences predictions only within a finite radius of it. Most kernel pairs
therefore contribute nothing, the interpolation matrix is sparse, and the
surrogate allocates much less memory than a globally supported one. The
coefficients are found with conjugate gradients; if that solve does not
converge within `maxiters`, a warning is emitted and the fit should not be
trusted.

```@docs
Wendland
```

``f = x -> exp(-x^2)``

```@example wendland
using Surrogates
using Plots
```

We sample `f` at 100 points between 0 and 1 using the `sample` function. The
sampling points are chosen using a Sobol sequence, which is done by passing
`SobolSample()` to `sample`.

```@example wendland
n = 100
lower_bound = 0.0
upper_bound = 1.0
f = x -> exp(-x^2)
x = sample(n, lower_bound, upper_bound, SobolSample())
y = f.(x)
```

## Building Surrogate

`eps` is the reciprocal of the support radius: a sample point influences
predictions within distance `1 / eps` of it, and nowhere else. Here `eps = 0.45`
gives a radius of about 2.2, so every sample point reaches across the whole
domain.

```@example wendland
wend = Wendland(x, y, lower_bound, upper_bound, eps = 0.45)
```

```@example wendland
plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lower_bound, upper_bound), legend = :top)
plot!(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
plot!(wend, label = "Surrogate function", xlims = (lower_bound, upper_bound), legend = :top)
```

### Choosing the support radius

Shrinking the support makes each prediction depend on fewer sample points. The
surrogate still interpolates — it reproduces every sampled response whatever
`eps` is — but between the samples it decays towards zero once the radius drops
below the sample spacing:

```@example wendland
grid = range(lower_bound, upper_bound, length = 400)
rms(v) = sqrt(sum(abs2, v) / length(v))

for eps in [0.45, 2.0, 20.0, 50.0]
    w = Wendland(x, y, lower_bound, upper_bound, eps = eps)
    println("eps = ", eps,
        "  radius = ", round(1 / eps, digits = 3),
        "  RMSE off the samples = ", round(rms([w(v) - f(v) for v in grid]), digits = 6),
        "  max error at the samples = ",
        round(maximum(abs(w(x[i]) - y[i]) for i in eachindex(x)), digits = 8))
end
```

```@example wendland
plot(f, label = "True function", xlims = (lower_bound, upper_bound), legend = :top)
for eps in [0.45, 20.0, 50.0]
    w = Wendland(x, y, lower_bound, upper_bound, eps = eps)
    plot!(w, label = "eps = $eps", xlims = (lower_bound, upper_bound))
end
plot!()
```

## Wendland Surrogate Tutorial (ND)

The surrogate works the same way in more dimensions; only the bounds change
shape. Note that the kernel exponent depends on the input dimension, so the
same `eps` gives a slightly different profile.

```@example wendlandND
using Surrogates
using Plots
default(c = :matter, legend = false, xlabel = "x", ylabel = "y")

function branin(x)
    x1, x2 = x[1], x[2]
    a, b, c, r, s, t = 1.0, 5.1 / (4 * pi^2), 5 / pi, 6.0, 10.0, 1 / (8 * pi)
    return a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s
end

lower_bound = [-5.0, 0.0]
upper_bound = [10.0, 15.0]
xys = sample(200, lower_bound, upper_bound, SobolSample())
zs = branin.(xys);
```

The domain is about 15 units across, so a support radius of 5 (`eps = 0.2`)
lets each sample point reach a useful neighbourhood without covering
everything.

```@example wendlandND
wend_ND = Wendland(xys, zs, lower_bound, upper_bound, eps = 0.2)
maximum(abs(wend_ND(xys[i]) - zs[i]) for i in eachindex(xys))
```

```@example wendlandND
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
x, y = -5.0:10.0, 0.0:15.0
p1 = surface(x, y, (x, y) -> wend_ND([x y]))
scatter!(xs, ys, zs, marker_z = zs)
p2 = contour(x, y, (x, y) -> wend_ND([x y]))
scatter!(xs, ys, marker_z = zs)
plot(p1, p2, title = "Wendland surrogate")
```
