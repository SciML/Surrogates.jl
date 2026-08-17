"""
    LinearSurrogate(x, y, lb, ub)

Fit an affine least-squares surrogate to sampled inputs `x` and responses `y`.
The fitted model is `coeff[1] + coeff[2] * x[1] + … + coeff[end] * x[end]`,
i.e. an intercept plus one slope per input dimension. The returned object is
callable at new points and implements the SurrogatesBase
deterministic-surrogate interface, including `update!`.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: responses corresponding to `x`.
  - `coeff`: fitted coefficients, laid out as `[intercept; slopes]`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.

# Arguments

  - `x`: training inputs. Use a vector of numbers for one dimension or a vector of
    equal-length point containers for multiple dimensions.
  - `y`: training responses, with one response per element of `x`. Responses
    must be scalars; vector-valued responses are not supported.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Returns

A callable `LinearSurrogate`. Calling `surrogate(point)` evaluates the fitted
affine model, while `update!(surrogate, x_new, y_new)` appends observations and
refits its coefficients.

# Example

```julia
using Surrogates

x = [0.0, 1.0, 2.0]
y = 2 .* x .+ 5
surrogate = LinearSurrogate(x, y, 0.0, 2.0)
surrogate(1.5)
```
"""
mutable struct LinearSurrogate{X, Y, C, L, U} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    coeff::C
    lb::L
    ub::U
end

# Least-squares design matrix with an intercept column: one row `[1, point...]`
# per training point.
_linear_design_matrix(x) = _linear_design_matrix(x, first(x))
function _linear_design_matrix(x, x_el::Number)
    X = ones(typeof(x_el), length(x), 2)
    X[:, 2] .= x
    return X
end
function _linear_design_matrix(x, x_el)
    d = length(x_el)
    X = ones(eltype(x_el), length(x), d + 1)
    for i in eachindex(x), j in 1:d
        X[i, j + 1] = x[i][j]
    end
    return X
end

_linear_coeff(x, y) = _linear_design_matrix(x) \ y

function LinearSurrogate(x, y, lb, ub)
    return LinearSurrogate(x, y, _linear_coeff(x, y), lb, ub)
end

function (lin::LinearSurrogate)(val::Number)
    _check_dimension(lin, val)
    return lin.coeff[1] + lin.coeff[2] * val
end

function (lin::LinearSurrogate)(val)
    _check_dimension(lin, val)
    return lin.coeff[1] + sum(lin.coeff[i + 1] * val[i] for i in 1:length(val))
end

function SurrogatesBase.update!(my_linear::LinearSurrogate, new_x, new_y)
    my_linear.x, my_linear.y = _append_samples(my_linear.x, my_linear.y, new_x, new_y)
    my_linear.coeff = _linear_coeff(my_linear.x, my_linear.y)
    return nothing
end
