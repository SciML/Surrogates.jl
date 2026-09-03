# KPLS and KPLSK Surrogate Tutorial

Full anisotropic Kriging fits one correlation scale per input coordinate, so a
`d`-dimensional problem needs a `d`-dimensional likelihood maximization. That search
is the bottleneck in high dimensions, not the linear algebra.

**KPLS** (Kriging combined with Partial Least Squares) reduces the count from `d` to a
small number `h` of PLS components. PLS finds directions in input space along which the
response varies most; collecting their rotation coefficients in `W* ∈ R^(d × h)` gives
the kernel

```math
R(x, x') = \exp\!\left(-\sum_{l=1}^{h} \theta_l \sum_{k=1}^{d} (w^*_{lk})^2 (x_k - x'_k)^2\right)
```

which has `h` hyperparameters rather than `d`. Note that the double sum rearranges into
a plain anisotropic Gaussian kernel with `θ⁰_k = Σ_l θ_l (w*_lk)²` — KPLS is not a
different model, it is a full Kriging model whose `d` scales are constrained to lie in
an `h`-dimensional family.

**KPLSK** exploits exactly that. It fits KPLS first, expands the result through the
identity above to get `d` scales, and then releases the constraint and refines all `d`
of them locally. Starting from a near-optimal point costs far less than optimizing `d`
scales from scratch, and the result is a full anisotropic Kriging model.

```@docs
KPLS
KPLSK
```

## Basic KPLS usage

`theta` is the *starting point* of the likelihood maximization, not the fitted value —
both models optimize it — and it must have exactly `n_comp` entries.

```@example KPLS1D
using Surrogates

f(x) = x[1]^2 + x[2]^2 + x[3]^2
lb = [-5.0, -5.0, -5.0]
ub = [5.0, 5.0, 5.0]
x = sample(50, lb, ub, SobolSample())
y = f.(x)

n_comp = 2
theta = [0.01, 0.01]
kpls_surrogate = KPLS(x, y, n_comp, lb, ub, theta)
kpls_surrogate((1.0, 2.0, 3.0))
```

The fitted scales live in the reduced PLS space, so there are `n_comp` of them:

```@example KPLS1D
kpls_surrogate.theta
```

## Basic KPLSK usage

```@example KPLS1D
kplsk_surrogate = KPLSK(x, y, n_comp, lb, ub, theta)
kplsk_surrogate((1.0, 2.0, 3.0))
```

Here `theta` has been released to full dimension — one scale per input coordinate —
while `theta_pls` keeps the reduced-space fit it was seeded from:

```@example KPLS1D
(kplsk_surrogate.theta, kplsk_surrogate.theta_pls)
```

## Choosing `n_comp`

`n_comp` must lie between `1` and the input dimension; both models reject anything
outside that range, since PLS has no more components to give. In practice a small value
is the point of the method — the reference results use `h = 2` or `3` even for `d` in
the tens. Larger `h` recovers more of the full anisotropic model at the price of the
search it was meant to avoid, and on a response that is close to a function of a few
linear combinations of the inputs it buys very little.
