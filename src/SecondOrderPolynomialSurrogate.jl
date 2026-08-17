"""
    SecondOrderPolynomialSurrogate(x, y, lb, ub)

Fit a full second-order polynomial surrogate by least squares. The design matrix
contains an intercept, each coordinate, every pairwise coordinate product, and
every squared coordinate.

# Fields

  - `x`: sampled input points.
  - `y`: scalar or vector responses corresponding to `x`.
  - `β`: fitted polynomial coefficient matrix.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.

# Arguments

  - `x`: training points represented by equal-length containers. At least
    `1 + 2d + d(d - 1) ÷ 2` points are required for `d`-dimensional inputs (the
    number of coefficients of a full quadratic); fewer throws an `ArgumentError`.
    The count is necessary but not sufficient: a degenerate sample (for example
    points that are collinear in two dimensions) leaves the quadratic
    unidentifiable, and is rejected with an `ArgumentError` as well.
  - `y`: training responses, with one response per point.
  - `lb`: lower domain bound.
  - `ub`: upper domain bound matching `lb`.

# Returns

A callable `SecondOrderPolynomialSurrogate` supporting
`update!(surrogate, x_new, y_new)`, which refits the coefficient matrix after
adding observations.

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
    n = length(x)
    d = length(first(x))
    num_coeffs = 1 + 2 * d + d * (d - 1) ÷ 2
    if n < num_coeffs
        throw(ArgumentError("SecondOrderPolynomialSurrogate requires at least \
            $num_coeffs samples to determine a full quadratic in $d dimension(s), \
            but got $n samples."))
    end
    X = _construct_2nd_order_interp_matrix(x, first(x))
    _check_2nd_order_rank(X, num_coeffs, n, d)
    Y = _construct_y_matrix(y, first(y))
    β = X \ Y
    return SecondOrderPolynomialSurrogate(x, y, β, lb, ub)
end

# Enough samples is necessary but not sufficient: a degenerate design (points
# collinear in two dimensions, say) leaves the quadratic unidentifiable. Whether
# that surfaces depends on the shape of the design matrix — a square one goes
# through LU and throws, a tall one through QR and quietly returns a minimum-norm
# fit — so it is checked here instead.
function _check_2nd_order_rank(X, num_coeffs, n, d)
    r = _matrix_rank(X)
    if r !== nothing && r < num_coeffs
        throw(
            ArgumentError(
                "SecondOrderPolynomialSurrogate needs samples that determine a full " *
                    "quadratic in $d dimension(s): the design matrix has $num_coeffs " *
                    "columns but rank $r from $n samples. The samples are degenerate " *
                    "(for example collinear); spread them across the domain."
            )
        )
    end
    return nothing
end

function _construct_2nd_order_interp_matrix(x, x_el)
    n = length(x)
    d = length(x_el)
    D = 1 + 2 * d + d * (d - 1) ÷ 2
    X = ones(eltype(x_el), n, D)
    for i in 1:n, j in 1:d
        X[i, j + 1] = x[i][j]
    end
    idx = d + 1
    for j in 1:d, k in (j + 1):d
        idx += 1
        for i in 1:n
            X[i, idx] = x[i][j] * x[i][k]
        end
    end
    for i in 1:n, j in 1:d
        X[i, j + 1 + d + d * (d - 1) ÷ 2] = x[i][j]^2
    end
    return X
end

_construct_y_matrix(y, y_el::Number) = y
_construct_y_matrix(y, y_el) = [y[i][j] for i in 1:length(y), j in 1:length(y_el)]

function (my_second_ord::SecondOrderPolynomialSurrogate)(val)

    _check_dimension(my_second_ord, val)

    #just create the val vector as X and multiply
    d = length(val)

    y = my_second_ord.β[1, :]
    for j in 1:d
        y += val[j] * my_second_ord.β[j + 1, :]
    end
    idx = d + 1
    for j in 1:d, k in (j + 1):d
        idx += 1
        y += val[j] * val[k] * my_second_ord.β[idx, :]
    end
    for j in 1:d
        y += val[j]^2 * my_second_ord.β[j + 1 + d + d * (d - 1) ÷ 2, :]
    end
    return _match_container(y, first(my_second_ord.y))
end

function SurrogatesBase.update!(my_second::SecondOrderPolynomialSurrogate, x_new, y_new)
    my_second.x, my_second.y = _append_samples(my_second.x, my_second.y, x_new, y_new)
    X = _construct_2nd_order_interp_matrix(my_second.x, first(my_second.x))
    Y = _construct_y_matrix(my_second.y, first(my_second.y))
    my_second.β = X \ Y
    return nothing
end
