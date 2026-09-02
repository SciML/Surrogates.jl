"""
    Kriging(x, y, lb::Number, ub::Number; p = 2.0,
            theta = 0.5 / max(1.0e-6 * abs(ub - lb), std(x))^p)
    Kriging(x, y, lb, ub;
            p = 2.0 .* collect(one.(x[1])),
            theta = [0.5 / max(1.0e-6 * norm(ub .- lb),
                               std(x_i[i] for x_i in x))^p[i]
                     for i in eachindex(x[1])])

Fit a Kriging interpolant with a power-exponential correlation model. The
surrogate is callable for mean predictions, while [`std_error_at_point`](@ref)
evaluates predictive uncertainty.

Based on: Jones, Schonlau and Welch (1998), "Efficient Global Optimization of
Expensive Black-Box Functions", J Glob Optim 13:455-492; and Jones (2001), "A
Taxonomy of Global Optimization Methods Based on Response Surfaces", J Glob
Optim 21:345-383.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: scalar responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `p`: correlation smoothness exponent: a scalar in one dimension, one
    entry per input coordinate otherwise.
  - `theta`: correlation scale, shaped like `p`.
  - `mu`: estimated constant process mean.
  - `b`: BLUP weights `R⁻¹(y - 𝟙μ)`.
  - `sigma`: estimated process variance.
  - `R_fact`: Cholesky factorization of the nugget-regularized correlation
    matrix. The deprecated property `inverse_of_R` still materializes `R⁻¹`
    from it.

# Arguments

  - `x`: training inputs with no repeated points.
  - `y`: scalar training responses, with one response per input.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `p`: correlation smoothness in the half-open interval `(0, 2]`. The default
    is `2.0` in one dimension and a vector filled with `2.0` otherwise. Zero is
    excluded: it makes every off-diagonal correlation `exp(-θ)`, so the
    correlation matrix is singular for more than two samples. Only `p = 2` gives
    a mean-square differentiable process; smaller values give rougher sample
    paths.
  - `theta`: positive correlation scale. When left unset it is **fitted by
    maximum likelihood**, starting from a value derived from the sample spread
    and the domain width as shown above; `update!` then refits it as samples are
    added. An explicitly supplied `theta` is a modelling choice: it is used as
    given and preserved across `update!`.
  - `optimize_theta`: whether to fit `theta` by maximizing the concentrated
    log-likelihood `-n/2 log σ̂²(θ) - 1/2 log|R(θ)|`, as in Jones (2001) §2 and
    DACE. Defaults to `true` exactly when `theta` is not supplied. Fitting
    costs a Nelder-Mead search whose every step factorizes an `n × n` matrix, so
    set it to `false` for a cheap fit on a large design.
  - `n_start`: Latin-hypercube starts for that search, in addition to the
    data-derived one. Ignored when `optimize_theta` is `false`.
  - `maxiters`: Nelder-Mead iteration cap per start.

# Returns

A callable `Kriging` supporting `update!(surrogate, x_new, y_new)` and
[`std_error_at_point`](@ref). Duplicate points make the correlation matrix
singular: the constructors reject them with an `ArgumentError`, while `update!`
warns and leaves the surrogate unchanged, since the observation is already
present and optimizers routinely re-propose points near convergence.

# Example

```julia
using Surrogates

x = [0.0, 0.5, 1.0]
y = sin.(x)
surrogate = Kriging(x, y, 0.0, 1.0)
surrogate(0.25)
std_error_at_point(surrogate, 0.25)
```
"""
mutable struct Kriging{X, Y, L, U, P, T, M, B, S, R} <: AbstractDeterministicSurrogate
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
    # `update!` re-derives `theta` when it came from the data-dependent default,
    # and refits it when it was fitted by maximum likelihood.
    theta_auto::Bool
    optimize_theta::Bool
end

function Base.getproperty(k::Kriging, s::Symbol)
    s === :inverse_of_R && return _deprecated_inverse_of_R(k, :Kriging)
    return getfield(k, s)
end

# Correlation between `val` and each sample. Scalar samples index as length-1
# containers (`(5.0)[1] === 5.0`), so one implementation serves both the scalar
# and the multidimensional case.
function _kriging_r(k::Kriging, val)
    d = length(k.x[1])
    return [
        exp(-sum(k.theta[l] * abs(val[l] - k.x[i][l])^k.p[l] for l in 1:d))
            for i in eachindex(k.x)
    ]
end

"""
Gives the current estimate for 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val)
    _check_dimension(k, val)
    return k.mu + dot(_kriging_r(k, val), k.b)
end

(k::Kriging)(val::Number) = (_check_dimension(k, val); k.mu + dot(_kriging_r(k, val), k.b))

# Ordinary kriging: the trend basis is 𝟙 over every observation. Jones (2001)
# eq. (5); see `_blup_std_error`.
function _kriging_std_error(k::Kriging, val)
    r = _kriging_r(k, val)
    return _blup_std_error(k.sigma, r, ones(eltype(r), length(r)), k.R_fact)
end

function std_error_at_point(k::Kriging, val)
    _check_dimension(k, val)
    return _kriging_std_error(k, val)
end

function std_error_at_point(k::Kriging, val::Number)
    _check_dimension(k, val)
    return _kriging_std_error(k, val)
end

# The sample spread, floored by a fraction of the domain width so a coordinate
# that barely varies cannot send the correlation scale to infinity. `std` is
# undefined for a single sample; the floor then stands alone, since a `NaN`
# scale reaches the solve as a "near-duplicate samples" error naming the wrong
# cause.
_kriging_spread(s, floor_) = isfinite(s) ? max(floor_, s) : floor_

# The default correlation scale, derived from the sample spread and the domain
# width. Kept as a function so `update!` can re-derive it.
function _kriging_default_theta(x, lb::Number, ub::Number, p)
    T = _surrogate_eltype(x)
    return T(0.5) / _kriging_spread(T(std(x)), T(1.0e-6) * abs(ub - lb))^p
end
function _kriging_default_theta(x, lb, ub, p)
    T = _surrogate_eltype(x)
    return [
        T(0.5) / _kriging_spread(
            T(std(x_i[i] for x_i in x)), T(1.0e-6) * T(norm(ub .- lb))
        )^p[i]
            for i in eachindex(x[1])
    ]
end

# The largest condition number the regularized correlation matrix may have.
# A tighter target buys a safer solve with a larger nugget, which relaxes
# interpolation; at 1e12 a Float64 solve still retains about four digits.
const _KRIGING_KAPPA_MAX = 1.0e12

# The nugget that brings `cond(R + δI)` to `_KRIGING_KAPPA_MAX`, letting samples
# lie close together without R going singular, at the cost of relaxing the
# interpolation condition. Mohammadi et al, "An analytic comparison of
# regularization methods for Gaussian Processes", arXiv:1602.00853.
#
# Only defined for BLAS element types; for generic ones the criterion is skipped
# and the factorization below escalates instead.
function _kriging_nugget(R, ::Type{T}) where {T <: Union{Float32, Float64}}
    all(isfinite, R) || return zero(T)
    S = Symmetric(R)
    # At the matrix' own precision, so a Float64 literal does not promote a
    # Float32 solve through `R + Diagonal(nugget)`.
    κ = T(_KRIGING_KAPPA_MAX)
    # One eigensolve: this dominates a likelihood evaluation, and the search
    # makes thousands. It can fail to converge as an extreme scale drives R to
    # the all-ones limit, leaving the escalation below to size the nugget.
    λdiff = try
        λ = eigvals(S)
        λ[end] - κ * λ[1]
    catch e
        e isa LAPACKException || rethrow()
        return zero(T)
    end
    return λdiff ≥ 0 ? λdiff / (κ - 1) : zero(λdiff)
end
_kriging_nugget(R, ::Type) = zero(eltype(R))

# Factorize the regularized correlation matrix, escalating the nugget if
# round-off leaves it marginally indefinite. `condition_nugget = false` skips the
# eigenvalue criterion for callers that only need the factorization to exist; the
# likelihood is not one of them, since the objective has to see the same
# regularization the final fit will use.
function _try_kriging_factorize(R; condition_nugget = true)
    all(isfinite, R) || return nothing
    n = size(R, 1)
    T = eltype(R)
    nugget = condition_nugget ? _kriging_nugget(R, T) : zero(T)
    for _ in 1:8
        F = cholesky(Symmetric(R + Diagonal(fill(nugget, n))), check = false)
        issuccess(F) && return F
        nugget = iszero(nugget) ? n * eps(real(float(T))) : 10 * nugget
    end
    return nothing
end

function _kriging_factorize(R)
    F = _try_kriging_factorize(R)
    F === nothing && throw(
        ArgumentError(
            "Could not factorize the Kriging correlation matrix even after " *
                "regularization. The samples are likely near-duplicates; remove " *
                "them or increase theta."
        )
    )
    return F
end

# Generalized least squares for the constant trend, given a factorization of R.
function _kriging_gls(F, y, n)
    one_vec = ones(eltype(y), n)
    mu = dot(one_vec, F \ y) / dot(one_vec, F \ one_vec)
    y_minus_1mu = y - one_vec * mu
    b = F \ y_minus_1mu
    sigma = dot(y_minus_1mu, b) / n
    return mu, b, sigma
end

# `p` and `theta` are indexed one entry per coordinate, so a scalar — the shape
# the one-dimensional constructors take — would reach a `BoundsError` instead.
# Only length is checked: a tuple indexes as happily as a vector.
function _check_kriging_length(name, v, d)
    if length(v) != d
        throw(
            ArgumentError(
                "For a $d-dimensional design, $name needs $d entries, one per " *
                    "input coordinate! Got $(length(v)): $v."
            )
        )
    end
    return nothing
end

function Kriging(
        x, y, lb::Number, ub::Number; p = _default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_no_duplicate_samples("Kriging", x)

    if p > 2.0 || p <= 0.0
        throw(ArgumentError("Hyperparameter p must be in (0, 2]! Got: $p."))
    end

    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, p) : theta

    if theta ≤ 0
        throw(ArgumentError("Hyperparameter theta must be positive! Got: $theta"))
    end

    optimize_theta && (theta = _fit_kriging_theta(x, y, p, theta; n_start = n_start, maxiters = maxiters))

    mu, b, sigma, R_fact = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(
        x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto, optimize_theta
    )
end

# The power-exponential correlation matrix; indexed like `_kriging_r`.
function _kriging_correlation(x, p, theta)
    n = length(x)
    d = length(x[1])
    return [
        exp(-sum(theta[l] * abs(x[i][l] - x[j][l])^p[l] for l in 1:d))
            for i in 1:n, j in 1:n
    ]
end

function _calc_kriging_coeffs(x, y, p, theta)
    n = length(x)
    R = _kriging_correlation(x, p, theta)
    F = _kriging_factorize(R)
    mu, b, sigma = _kriging_gls(F, y, n)
    return mu, b, sigma, F
end

function Kriging(
        x, y, lb, ub; p = _default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_no_duplicate_samples("Kriging", x)

    d = length(x[1])
    _check_kriging_length("p", p, d)
    for i in eachindex(x[1])
        if p[i] > 2.0 || p[i] <= 0.0
            throw(ArgumentError("All p must be in (0, 2]! Got: $p."))
        end
    end

    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, p) : theta
    _check_kriging_length("theta", theta, d)

    for i in eachindex(x[1])
        if theta[i] ≤ 0.0
            throw(ArgumentError("All theta must be positive! Got: $theta."))
        end
    end

    optimize_theta && (theta = _fit_kriging_theta(x, y, p, theta; n_start = n_start, maxiters = maxiters))

    mu, b, sigma, R_fact = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(
        x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto, optimize_theta
    )
end

"""
    _kriging_loglik(x, y, p, theta)

Concentrated log-likelihood of the correlation scale, up to additive constants:

    l(θ) = -n/2 log σ̂²(θ) - 1/2 log |R(θ)|

This is the objective Jones (2001) §2 and DACE both maximize over `θ`. A
scale whose correlation matrix cannot be factorized scores `-Inf` rather than
raising, so the search simply walks away from it.
"""
function _kriging_loglik(x, y, p, theta)
    n = length(x)
    R = _kriging_correlation(x, p, theta)
    all(isfinite, R) || return -Inf
    # The same nugget the final fit uses. Without it the likelihood is unbounded
    # as the scale shrinks: R goes near-singular and `logdet(R)` dives.
    F = _try_kriging_factorize(R)
    F === nothing && return -Inf
    _, _, sigma = _kriging_gls(F, y, n)
    (isfinite(sigma) && sigma > 0) || return -Inf
    return -n / 2 * log(sigma) - logdet(F) / 2
end

# Fit the correlation scale by maximum likelihood; see `_fit_theta`.
function _fit_kriging_theta(x, y, p, theta0; kwargs...)
    return _fit_theta(theta -> _kriging_loglik(x, y, p, theta), theta0; kwargs...)
end

"""
    update!(k::Kriging, new_x, new_y)

Add new samples and their responses and refit the surrogate.

Every call refits from scratch on the full sample set, which costs `O(n³)`, so
adding points one at a time in a loop is quartic in the number of additions.

# Returns

Returns `nothing`. A `new_x` that already appears in the samples warns and
leaves the surrogate unchanged, since duplicate points make the correlation
matrix singular.
"""
function SurrogatesBase.update!(k::Kriging, new_x, new_y)
    x_new, y_new = _append_samples(k.x, k.y, new_x, new_y)
    # A duplicate is a no-op, not an error: the model already carries that
    # observation, and optimizers re-propose points near convergence. Checked on
    # the merged samples, so a repetition *within* a batch is caught too.

    if length(unique(x_new)) != length(x_new)
        @warn "Skipping `update!`: these samples repeat a point already in the " *
            "Kriging surrogate, and duplicate points would make the correlation " *
            "matrix singular."
        return nothing
    end
    k.x, k.y = x_new, y_new
    if k.optimize_theta
        # A local refinement from the scale in hand: the extended sample set is
        # a small perturbation, and `update!` is called in a loop.
        k.theta = _fit_kriging_theta(k.x, k.y, k.p, k.theta; multistart = false)
    elseif k.theta_auto
        k.theta = _kriging_default_theta(k.x, k.lb, k.ub, k.p)
    end
    k.mu, k.b, k.sigma, k.R_fact = _calc_kriging_coeffs(k.x, k.y, k.p, k.theta)
    return nothing
end
