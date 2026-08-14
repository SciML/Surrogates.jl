# KPLS and KPLSK Surrogate Tutorial

KPLS (Kriging combined with Partial Least Squares) reduces the number of Kriging
hyperparameters from the input dimension `d` down to a small number of PLS components
`h`, which makes Kriging tractable on high-dimensional problems. KPLSK refines a KPLS
fit into a standard, full-dimensional Kriging model by using the KPLS solution as an
informed starting point, rather than optimizing all `d` hyperparameters from scratch.

```@docs
KPLS
KPLSK
```

## Basic KPLS Usage

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

## Basic KPLSK Usage

```@example KPLS1D
kplsk_surrogate = KPLSK(x, y, n_comp, lb, ub, theta)
kplsk_surrogate((1.0, 2.0, 3.0))
```
