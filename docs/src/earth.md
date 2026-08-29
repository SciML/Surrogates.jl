# Earth (MARS) Surrogate Tutorial

`EarthSurrogate` implements multivariate adaptive regression splines. The model
is a sum of *hinge* functions about the mean response,

``\hat f(p) = \bar y + \sum_t c_t \max(0, \pm(p_{j_t} - k_t))``

so it is piecewise linear with breakpoints — knots — placed at sampled
coordinates. Unlike a polynomial fit, it puts its flexibility only where the
data asks for it, which makes it a good choice for responses that change
character across the domain: a kink, a plateau, a change of slope.

```@docs
EarthSurrogate
Surrogates.HingeTerm
```

The fit runs in two passes. A **forward pass** greedily adds *reflected pairs* —
a hinge and its mirror image about the same knot — choosing at each step the
knot that most reduces the residual sum of squares. This deliberately overfits.
A **backward pass** then removes terms one at a time for as long as doing so
improves the generalized cross-validation score, which charges each retained
term `1 + penalty / 2` effective parameters. What survives is the basis.

```@example earth_tut
using Surrogates
using Plots
```

## Sampling

A response with a genuine change of slope, which is what the hinge basis is for:

```@example earth_tut
f = x -> x < 4 ? 2x : 8 + 6 * (x - 4)
lb = 0.0
ub = 10.0
n = 60
x = sample(n, lb, ub, SobolSample())
y = f.(x)
scatter(x, y, label = "Sampled points", xlims = (lb, ub), legend = :topleft)
plot!(f, label = "True function", xlims = (lb, ub))
```

## Building the surrogate

```@example earth_tut
earth = EarthSurrogate(x, y, lb, ub)
scatter(x, y, label = "Sampled points", xlims = (lb, ub), legend = :topleft)
plot!(f, label = "True function", xlims = (lb, ub))
plot!(earth, label = "Surrogate function", xlims = (lb, ub))
```

The target here is piecewise linear, so it lies in the span of the hinge basis
and is tracked closely — to about 0.1% of the response range. It is not
reproduced exactly, because knots are drawn from the samples and no sample
falls precisely on the kink at ``x = 4``; the surrogate places its knot at the
nearest sampled coordinate instead.

## Reading the basis

The retained terms are available, and say where the surrogate decided the
response changes slope:

```@example earth_tut
earth.basis
```

Each is a `Surrogates.HingeTerm` carrying the coordinate it acts on, its knot, and
whether it is the mirrored half of a pair. The `intercept` is the mean response,
the value taken where every hinge is inactive:

```@example earth_tut
earth.intercept, sum(y) / length(y)
```

## Controlling the fit

`n_max_terms` caps how many reflected pairs the forward pass may add, and
`penalty` sets how hard the backward pass prunes. A heavier penalty buys a
smaller, smoother model:

```@example earth_tut
coarse = EarthSurrogate(x, y, lb, ub; penalty = 50.0, n_min_terms = 1)
length(coarse.basis), length(earth.basis)
```

`rel_res_error` and `rel_GCV` are *relative* thresholds: a pair is added only if
it cuts the residual sum of squares by at least that fraction of the current
residual, and a term is pruned only if dropping it improves the GCV score by at
least that fraction. Both are therefore invariant to the scale of `y`.

## Adding samples

```@example earth_tut
update!(earth, 10.5, f(10.5))
earth(10.5), f(10.5)
```

`update!` refits both passes from scratch, so the knots are re-selected against
the enlarged sample rather than being carried over.

## Multidimensional inputs

```@example earth_tut
f_nd = p -> 2 * p[1] + 3 * max(0, p[2] - 5)
lb_nd = [0.0, 0.0]
ub_nd = [10.0, 10.0]
x_nd = sample(60, lb_nd, ub_nd, SobolSample())
y_nd = f_nd.(x_nd)
earth_nd = EarthSurrogate(x_nd, y_nd, lb_nd, ub_nd)
earth_nd((3.0, 8.0)), f_nd((3.0, 8.0))
```

!!! note "Additive terms only"

    Both passes select hinges in a single coordinate at a time; products of
    hinges across coordinates — the interaction terms of full MARS — are never
    formed. The surrogate is therefore additive in the input coordinates, and a
    response that is genuinely interacting, such as ``p_1 p_2``, is only
    approximated by its additive part. For those, prefer a surrogate with a
    multiplicative basis, such as `SecondOrderPolynomialSurrogate` or `Kriging`.

## Optimizing

```@example earth_tut
surrogate_optimize!(f, SRBF(), lb, ub, earth, SobolSample())
scatter(earth.x, earth.y, label = "Sampled points", legend = :topleft)
plot!(f, label = "True function", xlims = (lb, ub))
plot!(earth, label = "Surrogate function", xlims = (lb, ub))
```
