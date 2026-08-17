#=
One-dimensional Kriging method, following these papers:
"Efficient Global Optimization of Expensive Black Box Functions" and
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
both by DONALD R. JONES
=#
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

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: scalar responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `p`: scalar or per-dimension correlation smoothness exponent.
  - `theta`: scalar or per-dimension correlation scale.
  - `mu`: estimated constant process mean.
  - `b`: BLUP weight vector `R⁻¹(y - 1μ)`, the same quantity GEKPLS calls
    `gamma`.
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
    correlation matrix is singular for more than two samples. Only `p = 2`
    gives a smooth process; smaller values give non-differentiable sample
    paths.
  - `theta`: positive correlation scale. In one dimension the default is based
    on the sampled and bounded domain width; in multiple dimensions each
    coordinate receives its own scale. When left unset it is derived from the
    sample spread and domain width as shown above, and `update!` re-derives it
    so the scale stays calibrated as samples are added. An explicitly supplied
    `theta` is a modelling choice and is preserved across `update!`.

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
    # Whether `theta` came from the data-dependent default. `update!` re-derives
    # it in that case so the correlation scale stays calibrated as samples are
    # added; an explicitly supplied theta is a modelling choice and is kept.
    theta_auto::Bool
end

# Deprecated property: materializes R⁻¹ from the stored factorization. Every
# other field goes straight to `getfield`, which the compiler folds away for a
# literal field name.
function Base.getproperty(k::Kriging, s::Symbol)
    if s === :inverse_of_R
        Base.depwarn(
            "`Kriging.inverse_of_R` is deprecated: the Cholesky factorization of the " *
                "regularized correlation matrix is stored in `R_fact`. Prefer solving " *
                "with `k.R_fact \\ v` over forming the inverse.",
            :getproperty
        )
        return inv(getfield(k, :R_fact))
    end
    return getfield(k, s)
end

"""
Gives the current estimate for array 'val' with respect to the Kriging object k.
"""
# Scalar samples index as length-1 containers (`(5.0)[1] === 5.0`), so one
# implementation serves both the scalar and the multidimensional case.
_kriging_r(k::Kriging, val) = [
    exp(-sum(k.theta[l] * norm(val[l] - k.x[i][l])^k.p[l] for l in 1:length(k.x[1])))
        for i in 1:length(k.x)
]

"""
Gives the current estimate for 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val)
    _check_dimension(k, val)
    return k.mu + dot(_kriging_r(k, val), k.b)
end

(k::Kriging)(val::Number) = (_check_dimension(k, val); k.mu + dot(_kriging_r(k, val), k.b))

function _kriging_std_error(k::Kriging, val)
    r = _kriging_r(k, val)
    one_vec = ones(eltype(r), length(r))
    a = dot(r, k.R_fact \ r)
    b = dot(one_vec, k.R_fact \ one_vec)
    mean_squared_error = k.sigma * (1 - a + (1 - a)^2 / b)
    # Clamp rather than reflect: a negative mean-squared-error estimate signals
    # an ill-conditioned solve, and `abs` would turn it into a plausible-looking
    # positive standard error.
    return sqrt(max(mean_squared_error, zero(mean_squared_error)))
end

function std_error_at_point(k::Kriging, val)
    _check_dimension(k, val)
    return _kriging_std_error(k, val)
end

function std_error_at_point(k::Kriging, val::Number)
    _check_dimension(k, val)
    return _kriging_std_error(k, val)
end







# The default correlation scale, derived from the sample spread and the domain
# width. Kept as a function so `update!` can re-derive it.
_kriging_default_theta(x, lb::Number, ub::Number, p) =
    0.5 / max(1.0e-6 * abs(ub - lb), std(x))^p
_kriging_default_theta(x, lb, ub, p) = [
    0.5 / max(1.0e-6 * norm(ub .- lb), std(x_i[i] for x_i in x))^p[i]
        for i in eachindex(x[1])
]

function Kriging(
        x, y, lb::Number, ub::Number; p = 2.0, theta = nothing
    )
    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, p) : theta
    if length(x) != length(unique(x))
        throw(ArgumentError("There is a repetition in the samples, cannot build Kriging: duplicate points make the correlation matrix singular."))
    end

    if p > 2.0 || p <= 0.0
        throw(ArgumentError("Hyperparameter p must be in (0, 2]! Got: $p."))
    end

    if theta ≤ 0
        throw(ArgumentError("Hyperparameter theta must be positive! Got: $theta"))
    end

    mu, b, sigma, R_fact = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto)
end

# Estimate a nugget from the maximum allowed condition number. This regularizes
# R so that samples may lie close together without R becoming singular, at the
# cost of slightly relaxing the interpolation condition. Derived from "An
# analytic comparison of regularization methods for Gaussian Processes" by
# Mohammadi et al (https://arxiv.org/pdf/1602.00853.pdf).
#
# `eigmin`/`eigmax` ask LAPACK for just the two extremal eigenvalues instead of
# the whole spectrum. They are only available for BLAS element types, so for
# generic ones (BigFloat, for instance) the criterion is skipped and the
# factorization below falls back to escalating the nugget until R is positive
# definite.
const _KRIGING_KAPPA_MAX = 1.0e8

function _kriging_nugget(R, ::Type{T}) where {T <: Union{Float32, Float64}}
    S = Symmetric(R)
    λdiff = eigmax(S) - _KRIGING_KAPPA_MAX * eigmin(S)
    return λdiff ≥ 0 ? λdiff / (_KRIGING_KAPPA_MAX - 1) : zero(λdiff)
end
_kriging_nugget(R, ::Type) = zero(eltype(R))

# Factorize the regularized correlation matrix, escalating the nugget if
# round-off leaves it marginally indefinite. Returns the Cholesky factorization,
# which is both cheaper and more accurate than forming `inv(R)`: every use site
# needs only solves against R.
function _kriging_factorize(R)
    n = size(R, 1)
    T = eltype(R)
    nugget = _kriging_nugget(R, T)
    for _ in 1:8
        F = cholesky(Symmetric(R + Diagonal(fill(nugget, n))), check = false)
        issuccess(F) && return F
        nugget = iszero(nugget) ? n * eps(real(float(T))) : 10 * nugget
    end
    throw(
        ArgumentError(
            "Could not factorize the Kriging correlation matrix even after regularization. " *
                "The samples are likely near-duplicates; remove them or increase theta."
        )
    )
end

# Generalized least squares for the constant trend, given a factorization of R.
function _kriging_gls(F, y, n)
    one = ones(eltype(y), n)
    mu = dot(one, F \ y) / dot(one, F \ one)
    y_minus_1mu = y - one * mu
    b = F \ y_minus_1mu
    sigma = dot(y_minus_1mu, b) / n
    return mu, b, sigma
end

function _calc_kriging_coeffs(x, y, p::Number, theta::Number)
    n = length(x)
    R = [exp(-theta * abs(x[i] - x[j])^p) for i in 1:n, j in 1:n]
    F = _kriging_factorize(R)
    mu, b, sigma = _kriging_gls(F, y, n)
    return mu, b, sigma, F
end

function Kriging(
        x, y, lb, ub; p = 2.0 .* collect(one.(x[1])), theta = nothing
    )
    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, p) : theta
    if length(x) != length(unique(x))
        throw(ArgumentError("There is a repetition in the samples, cannot build Kriging: duplicate points make the correlation matrix singular."))
    end

    for i in 1:length(x[1])
        if p[i] > 2.0 || p[i] <= 0.0
            throw(ArgumentError("All p must be in (0, 2]! Got: $p."))
        end

        if theta[i] ≤ 0.0
            throw(ArgumentError("All theta must be positive! Got: $theta."))
        end
    end

    mu, b, sigma, R_fact = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, theta_auto)
end

function _calc_kriging_coeffs(x, y, p, theta)
    n = length(x)
    d = length(x[1])

    # `i` indexes rows and `j` columns, matching the 1-D method.
    R = [
        let
                s = zero(eltype(x[1]))
                for l in 1:d
                    s = s + theta[l] * norm(x[i][l] - x[j][l])^p[l]
            end
                exp(-s)
        end
            for i in 1:n, j in 1:n
    ]

    F = _kriging_factorize(R)
    mu, b, sigma = _kriging_gls(F, y, n)
    return mu, b, sigma, F
end

"""
    update!(k::Kriging,new_x,new_y)

Adds the new point and its respective value to the sample points.
Warning: If you are just adding a single point, you have to wrap it with [].
Returns the updated Kriging model.

Every call refits the surrogate from scratch on the full sample set, which
costs O(n^3) for the kernel models. Adding points one at a time in a loop is
therefore quadratic in the number of additions.

"""
function SurrogatesBase.update!(k::Kriging, new_x, new_y)
    # A duplicate is a no-op, not an error: the model already carries that
    # observation, and optimizers re-propose points near convergence.
    if new_x in k.x
        @warn "Skipping `update!`: this sample already exists in the Kriging surrogate, and duplicate points would make the correlation matrix singular."
        return nothing
    end
    k.x, k.y = _append_samples(k.x, k.y, new_x, new_y)
    if k.theta_auto
        k.theta = _kriging_default_theta(k.x, k.lb, k.ub, k.p)
    end
    k.mu, k.b, k.sigma, k.R_fact = _calc_kriging_coeffs(k.x, k.y, k.p, k.theta)
    return nothing
end
