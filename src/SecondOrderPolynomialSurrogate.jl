"""
    SecondOrderPolynomialSurrogate(x, y, lb, ub)

Fit a full second-order polynomial to `(x, y)` by least squares.

For `d`-dimensional inputs the model is the complete quadratic

```math
\\hat f(p) = \\beta_0 + \\sum_{j=1}^{d} \\beta_j p_j
            + \\sum_{j<k} \\beta_{jk} p_j p_k
            + \\sum_{j=1}^{d} \\beta_{jj} p_j^2
```

with `1 + 2d + d(d - 1) ÷ 2` coefficients. It is a regression surrogate, not an
interpolant: with more samples than coefficients it does not pass through the
data, and the residual is orthogonal to every column of the design matrix.

# Fields

  - `x`: sampled input points.
  - `y`: scalar or vector responses corresponding to `x`.
  - `β`: fitted coefficients, ordered as described under *Coefficient order*.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.

# Arguments

  - `x`: training points, as numbers for one-dimensional inputs or as
    equal-length tuples or vectors otherwise. At least
    `1 + 2d + d(d - 1) ÷ 2` points are required; fewer throws an
    `ArgumentError`. The count is necessary but not sufficient: a degenerate
    design (points collinear in two dimensions, say) leaves the quadratic
    unidentifiable and is rejected with an `ArgumentError` too, rather than
    silently returning one of its infinitely many minimum-norm fits.
  - `y`: training responses, one per point. Numbers give a scalar surrogate;
    equal-length vectors or tuples give a multi-output one, fitted one output
    per column against a single shared factorization.
  - `lb`: lower domain bound.
  - `ub`: upper domain bound matching `lb`.

# Returns

A callable `SecondOrderPolynomialSurrogate` supporting
`update!(surrogate, x_new, y_new)`, which refits the coefficients after adding
observations.

# Coefficient order

`β` follows the columns of the design matrix: the intercept, then each
coordinate, then the pairwise products in lexicographic order, then the squares.
In two dimensions that is

```
β = [1, p₁, p₂, p₁p₂, p₁², p₂²]
```

so a target written in matrix form, `a + bᵀp + pᵀCp` with symmetric `C`, has
`β = [a, b₁, b₂, 2C₁₂, C₁₁, C₂₂]` — the cross coefficient is `2C₁₂`, since `C`
contributes the off-diagonal term twice.

For vector responses `β` is a matrix with one column per output, and evaluation
returns a vector.

# Element types

The fit is carried out in `float(eltype)` of the samples, so `Float32` and
`BigFloat` inputs keep their precision and integer or rational inputs are
promoted the same way `\\` would promote them. Queries may be given in any type
that promotes against the fit.

# Differentiability

Evaluation is differentiable in the query point with both ForwardDiff and
Zygote — gradients, Hessians, multi-output Jacobians, and nested duals. The
*fit* is differentiable in the training data with ForwardDiff, which is how to
obtain sensitivities of a fitted value to the samples it was built from; Zygote
cannot trace the fit, as the design matrix is built by mutation.

# Example

```julia
using Surrogates

x = [(0.0,), (1.0,), (2.0,)]
y = [0.0, 1.0, 4.0]
surrogate = SecondOrderPolynomialSurrogate(x, y, [0.0], [2.0])
surrogate((1.5,))
```
"""
mutable struct SecondOrderPolynomialSurrogate{X, Y, B, L, U} <:
    AbstractDeterministicSurrogate
    x::X
    y::Y
    β::B
    lb::L
    ub::U
end

function SecondOrderPolynomialSurrogate(x, y, lb, ub)
    return SecondOrderPolynomialSurrogate(x, y, _fit_2nd_order(x, y), lb, ub)
end

function (poly::SecondOrderPolynomialSurrogate)(val)
    _check_dimension(poly, val)
    return _match_container(_2nd_order_eval(poly.β, val), first(poly.y))
end

function SurrogatesBase.update!(poly::SecondOrderPolynomialSurrogate, x_new, y_new)
    poly.x, poly.y = _append_samples(poly.x, poly.y, x_new, y_new)
    poly.β = _fit_2nd_order(poly.x, poly.y)
    return nothing
end

# Column layout of the quadratic basis, shared by the design matrix and the
# evaluation: the intercept, the `d` coordinates, the `d(d-1)÷2` pairwise
# products, then the `d` squares last.
_2nd_order_ncoeffs(d) = 1 + 2 * d + d * (d - 1) ÷ 2
_2nd_order_sq_offset(d) = _2nd_order_ncoeffs(d) - d

"""
    _fit_2nd_order(x, y)

Least-squares coefficients of a full quadratic through `(x, y)`, once the design
is known to determine them.

Enough samples is necessary but not sufficient: a degenerate design leaves the
quadratic unidentifiable, and `\\` answers such a system with one of its
infinitely many minimum-norm solutions — a fit that follows the samples while its
behavior away from them is an artifact of the pivoting.

Both the rank and the coefficients come from a single pivoted QR. That is the
factorization `\\` runs on a tall matrix anyway, and pivoting orders `|R[i, i]|`
non-increasingly, so the rank is a scan of the diagonal rather than the separate
SVD that `rank` would need — which costs as much again as the solve, and is
unavailable outside `Float32` and `Float64`.
"""
function _fit_2nd_order(x, y)
    n = length(x)
    d = length(first(x))
    ncoeffs = _2nd_order_ncoeffs(d)
    n < ncoeffs && throw(
        ArgumentError(
            "SecondOrderPolynomialSurrogate needs at least $ncoeffs samples to \
            determine a full quadratic in $d dimension(s), but got $n."
        )
    )
    F = qr(_2nd_order_design_matrix(x), ColumnNorm())
    r = _qr_rank(F)
    r < ncoeffs && throw(
        ArgumentError(
            "SecondOrderPolynomialSurrogate was given $n degenerate samples — \
            collinear, say: their design matrix has rank $r, short of the \
            $ncoeffs columns a full quadratic in $d dimension(s) needs. Spread \
            the samples across the domain."
        )
    )
    return F \ _construct_y_matrix(y, first(y))
end

# Pivoting leaves the diagonal of `R` in non-increasing magnitude, so the
# numerical rank is the count of entries above LAPACK's tolerance.
function _qr_rank(F)
    r = abs.(diag(F.R))
    isempty(r) && return 0
    return count(>(maximum(size(F.R)) * eps(float(real(eltype(r)))) * first(r)), r)
end

function _2nd_order_design_matrix(x)
    n = length(x)
    d = length(first(x))
    sq = _2nd_order_sq_offset(d)
    # `float` because the solve promotes an integer or rational design anyway,
    # and a float element type is what lets the factorization report a rank for
    # one. `float(BigFloat) === BigFloat`, so precision is never lowered.
    X = ones(float(eltype(first(x))), n, _2nd_order_ncoeffs(d))
    # Sample index innermost: it is the fast axis of `X`, and one read of a
    # coordinate serves both its linear and its squared column.
    for j in 1:d, i in 1:n
        xij = x[i][j]
        X[i, j + 1] = xij
        X[i, sq + j] = xij^2
    end
    col = d + 1
    for j in 1:d, k in (j + 1):d
        col += 1
        for i in 1:n
            X[i, col] = x[i][j] * x[i][k]
        end
    end
    return X
end

"""
    _2nd_order_eval(β, val)

Value of the fitted quadratic at `val`, over the monomials in the column order
of [`_2nd_order_design_matrix`](@ref).

The accumulator is a scalar, so a scalar-response fit evaluates without
allocating, and nothing is mutated, so both forward and reverse mode
differentiate straight through.
"""
function _2nd_order_eval(β::AbstractVector, val)
    d = length(val)
    v = β[1]
    for j in 1:d
        v += val[j] * β[j + 1]
    end
    col = d + 1
    for j in 1:d, k in (j + 1):d
        col += 1
        v += val[j] * val[k] * β[col]
    end
    sq = _2nd_order_sq_offset(d)
    for j in 1:d
        v += val[j]^2 * β[sq + j]
    end
    return v
end

# One coefficient column per output, and the outputs are independent, so each is
# the scalar contraction against its own column. The views are free: a column of
# a dense matrix is already contiguous.
function _2nd_order_eval(β::AbstractMatrix, val)
    return [_2nd_order_eval(view(β, :, m), val) for m in axes(β, 2)]
end
