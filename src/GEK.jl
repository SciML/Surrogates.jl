"""
    GEK(x, y, lb::Number, ub::Number; p = 2.0,
        theta = 0.5 / max(1.0e-6 * abs(ub - lb), std(x))^2)
    GEK(x, y, lb, ub;
        p = 2.0 .* collect(one.(x[1])),
        theta = [0.5 / max(1.0e-6 * norm(ub .- lb),
                           std(x_i[i] for x_i in x))^2
                 for i in eachindex(x[1])])

Gradient-enhanced Kriging surrogate.

`GEK` augments the Kriging covariance system with derivative observations. The
surrogate is callable as `gek(x_new)`, exposes uncertainty through
[`std_error_at_point`](@ref), and supports
`update!(gek, x_new, y_new, grad_new)`.

# Response layout

`y` holds the function values followed by the gradient observations, in
point-major order:

```
[f(x_1), …, f(x_n), ∂f/∂x_1(x_1), …, ∂f/∂x_d(x_1), …, ∂f/∂x_1(x_n), …, ∂f/∂x_d(x_n)]
```

so `length(y) == n * (1 + d)`. In one dimension this is `[f.(x); f'.(x)]`.

# Fields

  - `x`: training inputs.
  - `y`: observed values and derivative data in the layout above. Responses
    must be scalars; vector-valued responses are not supported.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `p`: correlation exponent. Must be `2`; see below.
  - `theta`: scalar or per-dimension correlation scale parameter.
  - `mu`: fitted process mean.
  - `b`: BLUP weight vector over values *and* derivative observations.
  - `sigma`: fitted process variance.
  - `R_fact`: Cholesky factorization of the GEK correlation matrix.

# Keywords

  - `p`: correlation exponent, fixed at `2`. The value/derivative and
    derivative/derivative covariance blocks are the first and second
    derivatives of the squared-exponential kernel, so any other exponent makes
    those blocks inconsistent with the value block. Supplying `p != 2` throws
    an `ArgumentError`.
  - `theta`: positive correlation scale, scalar or one entry per dimension. The
    default follows [`Kriging`](@ref) and scales with the data: a fixed constant
    gives a correlation length unrelated to the sample spacing, so on any domain
    much wider than 1 the samples are effectively uncorrelated and the surrogate
    reverts to its mean between them. A defaulted `theta` is re-derived by
    `update!` as samples are added; an explicitly supplied one is preserved.

# Returns

A `GEK` surrogate satisfying the generic surrogate interface.
"""
mutable struct GEK{X, Y, L, U, P, T, M, B, S, R} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    p::P
    theta::T
    mu::M
    b::B
    sigma::S
    R_fact::R
    # See `Kriging`: `update!` re-derives a defaulted theta and keeps an
    # explicitly supplied one.
    theta_auto::Bool
end

# Deprecated property: materializes R⁻¹ from the stored factorization.
function Base.getproperty(k::GEK, s::Symbol)
    if s === :inverse_of_R
        Base.depwarn(
            "`GEK.inverse_of_R` is deprecated: the Cholesky factorization of the " *
                "correlation matrix is stored in `R_fact`.",
            :getproperty
        )
        return inv(getfield(k, :R_fact))
    end
    return getfield(k, s)
end

# Scalar samples index as length-1 containers (`(5.0)[1] === 5.0`), so one
# implementation serves both the scalar and the multidimensional case.
_gek_dim(x) = length(first(x))

function _gek_check_p(p)
    if !all(isequal(2), p)
        throw(
            ArgumentError(
                "GEK requires p = 2! Got: $p. The value/derivative covariance blocks are " *
                    "derivatives of the squared-exponential kernel, so a different exponent " *
                    "would be inconsistent with the value block."
            )
        )
    end
    return nothing
end

function _gek_check_theta(theta)
    if !all(>(0), theta)
        throw(ArgumentError("All theta must be positive! Got: $theta."))
    end
    return nothing
end

# Squared-exponential correlation R(u, v) = exp(-Σ_l θ_l (u_l - v_l)^2) and the
# derivative blocks GEK is built from:
#
#   ∂R/∂v_l        =  2 θ_l Δ_l R                       (value  × derivative)
#   ∂R/∂u_l        = -2 θ_l Δ_l R                       (derivative × value)
#   ∂²R/∂u_l ∂v_k  = (2 θ_l δ_lk - 4 θ_l θ_k Δ_l Δ_k) R (derivative × derivative)
#
# with Δ_l = u_l - v_l. The δ_lk term is what gives the derivative observations a
# nonzero variance; without it the diagonal of the derivative block vanishes and
# the correlation matrix is singular.
_gek_index(n, i, l, d) = n + (i - 1) * d + l

function _calc_gek_coeffs(x, y, p, theta)
    n = length(x)
    d = _gek_dim(x)
    nd = n * (1 + d)
    length(y) == nd || throw(
        ArgumentError(
            "GEK expects length(y) == n * (1 + d) = $nd for $n samples in $d dimension(s), got $(length(y))."
        )
    )

    T = eltype(first(x))
    R = zeros(T, nd, nd)
    @inbounds for i in 1:n, j in 1:n
        Rij = exp(-sum(theta[l] * (x[i][l] - x[j][l])^2 for l in 1:d))
        R[i, j] = Rij
        for l in 1:d
            Δl = x[i][l] - x[j][l]
            R[i, _gek_index(n, j, l, d)] = 2 * theta[l] * Δl * Rij
            R[_gek_index(n, i, l, d), j] = -2 * theta[l] * Δl * Rij
            for k in 1:d
                Δk = x[i][k] - x[j][k]
                cross = -4 * theta[l] * theta[k] * Δl * Δk * Rij
                if l == k
                    cross += 2 * theta[l] * Rij
                end
                R[_gek_index(n, i, l, d), _gek_index(n, j, k, d)] = cross
            end
        end
    end

    # The constant trend acts on the value observations only: differentiating a
    # constant gives zero, so the derivative rows carry 0.
    one_vec = [i <= n ? one(T) : zero(T) for i in 1:nd]
    F = _kriging_factorize(R)
    mu = dot(one_vec, F \ y) / dot(one_vec, F \ one_vec)
    y_minus_1mu = y - one_vec * mu
    b = F \ y_minus_1mu
    # Normalized by the number of observations, not the number of points.
    sigma = dot(y_minus_1mu, b) / nd
    return mu, b, sigma, F
end

# Cross-covariance between the prediction at `val` and every observation:
# the value entries are R(val, x_i), the derivative entries ∂R(val, x_i)/∂x_i,l.
function _gek_r(k::GEK, val)
    n = length(k.x)
    d = _gek_dim(k.x)
    r_val = [
        exp(-sum(k.theta[l] * (val[l] - k.x[i][l])^2 for l in 1:d))
            for i in 1:n
    ]
    # One flat comprehension over the response layout: a comprehension with two
    # `for` clauses lowers to `push!`, which Zygote cannot differentiate.
    return [
        idx <= n ? r_val[idx] :
            let i = div(idx - n - 1, d) + 1, l = mod(idx - n - 1, d) + 1
            2 * k.theta[l] * (val[l] - k.x[i][l]) * r_val[i]
        end
            for idx in 1:(n * (1 + d))
    ]
end

function _gek_predict(k::GEK, val)
    # The derivative weights are part of the predictor; without them the
    # gradient observations would not influence the prediction.
    return k.mu + dot(_gek_r(k, val), k.b)
end

function _gek_std_error(k::GEK, val)
    n = length(k.x)
    d = _gek_dim(k.x)
    nd = n * (1 + d)
    T = eltype(first(k.x))
    r = _gek_r(k, val)
    one_vec = [i <= n ? one(T) : zero(T) for i in 1:nd]
    a = dot(r, k.R_fact \ r)
    b = dot(one_vec, k.R_fact \ one_vec)
    mean_squared_error = k.sigma * (1 - a + (1 - a)^2 / b)
    # Clamp rather than reflect; see the Kriging methods.
    return sqrt(max(mean_squared_error, zero(mean_squared_error)))
end

function (k::GEK)(val::Number)
    _check_dimension(k, val)
    return _gek_predict(k, val)
end

function (k::GEK)(val)
    _check_dimension(k, val)
    return _gek_predict(k, val)
end

std_error_at_point(k::GEK, val::Number) = (_check_dimension(k, val); _gek_std_error(k, val))
std_error_at_point(k::GEK, val) = (_check_dimension(k, val); _gek_std_error(k, val))

function _gek_check_duplicates(x)
    if length(x) != length(unique(x))
        throw(
            ArgumentError(
                "There is a repetition in the samples, cannot build GEK: duplicate points make the correlation matrix singular."
            )
        )
    end
    return nothing
end

function GEK(x, y, lb::Number, ub::Number; p = 2.0, theta = nothing)
    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, 2) : theta
    _gek_check_duplicates(x)
    _gek_check_p(p)
    _gek_check_theta(theta)
    mu, b, sigma, R_fact = _calc_gek_coeffs(x, y, p, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto)
end

function GEK(x, y, lb, ub; p = 2.0 .* collect(one.(x[1])), theta = nothing)
    theta_auto = theta === nothing
    theta = theta_auto ?
        _kriging_default_theta(x, lb, ub, fill(2, length(x[1]))) : theta
    _gek_check_duplicates(x)
    _gek_check_p(p)
    _gek_check_theta(theta)
    mu, b, sigma, R_fact = _calc_gek_coeffs(x, y, p, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto)
end

"""
    update!(k::GEK, new_x, new_y, new_grad)

Add a sample together with its gradient. GEK's response vector interleaves
values and derivatives, so a value alone does not determine a valid state;
`update!(k, new_x, new_y)` therefore throws.

`new_grad` is the gradient at `new_x`: a scalar in one dimension, or a
`d`-element container otherwise.
"""
function SurrogatesBase.update!(k::GEK, new_x, new_y, new_grad)
    # See `Kriging.update!`: a duplicate is a no-op here, not an error.
    if new_x in k.x
        @warn "Skipping `update!`: this sample already exists in the GEK surrogate, and duplicate points would make the correlation matrix singular."
        return nothing
    end
    n = length(k.x)
    d = _gek_dim(k.x)
    length(new_grad) == d || throw(
        ArgumentError("Expected a $d-element gradient at the new sample, got $(length(new_grad)).")
    )

    push!(k.x, new_x)
    # Values occupy 1:n and derivatives follow in point-major order, so the new
    # value goes directly after the existing values and its gradient at the end.
    y = k.y
    k.y = vcat(y[1:n], [new_y], y[(n + 1):end], collect(new_grad))
    if k.theta_auto
        k.theta = _kriging_default_theta(k.x, k.lb, k.ub, k.p)
    end
    k.mu, k.b, k.sigma, k.R_fact = _calc_gek_coeffs(k.x, k.y, k.p, k.theta)
    return nothing
end

function SurrogatesBase.update!(k::GEK, new_x, new_y)
    throw(
        ArgumentError(
            "GEK needs a gradient for each sample: its response vector is " *
                "[values; derivatives], so adding a value alone would corrupt that layout. " *
                "Use `update!(gek, new_x, new_y, new_grad)`."
        )
    )
end
