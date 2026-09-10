# Second Order Polynomial Surrogate Tutorial

The second-order polynomial model is the least-squares fit

``y = Xβ + ϵ``

where ``X`` is the design matrix of the linear model augmented with the
``d(d-1)/2`` pairwise products of the variables and their ``d`` squares, for
``1 + 2d + d(d-1)/2`` columns in all. Because it is a regression rather than an
interpolation, the surrogate does not pass through the samples once there are
more of them than coefficients; what it guarantees instead is that the residual
is orthogonal to every column of ``X``.

That column count is also the minimum number of samples: fewer leave the
quadratic underdetermined, and so do degenerate designs such as collinear
points, both of which are rejected rather than silently fitted.

```@docs
SecondOrderPolynomialSurrogate
```

```@example second_order_tut
using Surrogates
using Plots
```

## Sampling

```@example second_order_tut
f = x -> 3 * sin(x) + 10 / x
lb = 3.0
ub = 6.0
n = 100
x = sample(n, lb, ub, HaltonSample())
y = f.(x)
scatter(x, y, label = "Sampled points", xlims = (lb, ub))
plot!(f, label = "True function", xlims = (lb, ub))
```

## Building the surrogate

```@example second_order_tut
sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
plot(x, y, seriestype = :scatter, label = "Sampled points", xlims = (lb, ub))
plot!(f, label = "True function", xlims = (lb, ub))
plot!(sec, label = "Surrogate function", xlims = (lb, ub))
```

## Optimizing

```@example second_order_tut
surrogate_optimize!(f, SRBF(), lb, ub, sec, SobolSample())
scatter(x, y, label = "Sampled points")
plot!(f, label = "True function", xlims = (lb, ub))
plot!(sec, label = "Surrogate function", xlims = (lb, ub))
```

The optimization method successfully found the minimum.

## Multi-output responses

Passing a vector response per sample fits one column of coefficients per output
against a single shared factorization, and evaluation returns a vector.

```@example second_order_multi
using Surrogates

lb = [0.0, 0.0]
ub = [10.0, 10.0]
f = p -> [p[1]^2, p[1] * p[2]]
x = sample(30, lb, ub, SobolSample())
y = f.(x)
sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
sec((2.0, 3.0))
```

Both outputs are exactly quadratic, so they are recovered to round-off:

```@example second_order_multi
f((2.0, 3.0))
```

## Reading the coefficients

`β` follows the columns of the design matrix — intercept, coordinates, pairwise
products in lexicographic order, then squares. A target written as
``a + bᵀp + pᵀCp`` with symmetric ``C`` therefore has cross coefficient
``2C₁₂``, since ``C`` contributes that off-diagonal term twice.

```@example second_order_coeffs
using Surrogates

a = 0.3
b = [0.7, 0.1]
C = [0.3 0.4; 0.4 0.1]
g = p -> a + b' * collect(p) + collect(p)' * C * collect(p)

lb = [-5.0, -5.0]
ub = [5.0, 5.0]
x = sample(30, lb, ub, SobolSample())
sec = SecondOrderPolynomialSurrogate(x, g.(x), lb, ub)
sec.β
```

```@example second_order_coeffs
[a, b[1], b[2], 2C[1, 2], C[1, 1], C[2, 2]]
```
