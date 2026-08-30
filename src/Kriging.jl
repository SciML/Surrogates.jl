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
    log-likelihood `-n/2 log σ̂²(θ) - 1/2 log|R(θ)|`, as in Jones (2001) §2, DACE
    and SMT. Defaults to `true` exactly when `theta` is not supplied. Fitting
    costs a Nelder-Mead search whose every step factorizes an `n × n` matrix, so
    set it to `false` for a cheap fit on a large design; across a six-problem
    benchmark it lowered prediction error by factors of 1.03 to 25.6.
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
    # Whether `theta` came from the data-dependent default, in which case
    # `update!` re-derives it.
    theta_auto::Bool
    # Whether `theta` is fitted by maximum likelihood, in which case `update!`
    # refits it against the extended sample set.
    optimize_theta::Bool
end

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

# Ordinary-kriging mean squared error, Jones (2001) "A Taxonomy of Global
# Optimization Methods Based on Response Surfaces", eq. (5):
#
#   s²(x) = σ² [1 - rᵀR⁻¹r + (1 - 𝟙ᵀR⁻¹r)² / (𝟙ᵀR⁻¹𝟙)]
#
# The third term is the variance contributed by estimating the trend μ, so its
# numerator is the *trend* residual `1 - 𝟙ᵀR⁻¹r`, not the prediction residual
# `1 - rᵀR⁻¹r` that this used to square.
function _kriging_std_error(k::Kriging, val)
    r = _kriging_r(k, val)
    Rinv_r = k.R_fact \ r
    one_vec = ones(eltype(r), length(r))
    a = dot(r, Rinv_r)
    c = sum(Rinv_r)
    b = dot(one_vec, k.R_fact \ one_vec)
    mean_squared_error = k.sigma * (1 - a + (1 - c)^2 / b)
    # As a variance the expression is non-negative in exact arithmetic, so a
    # negative value is round-off and clamps to zero. Reflecting it with `abs`
    # would turn an ill-conditioned solve into a plausible-looking error bar.
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

# The element type the fit is carried out in: whatever the samples are, made
# floating point. Every default and every constant below is taken to it, so a
# Float32 design is not silently promoted to Float64 by a literal.
_kriging_eltype(x) = float(eltype(first(x)))

# The default correlation scale, derived from the sample spread and the domain
# width. Kept as a function so `update!` can re-derive it.
function _kriging_default_theta(x, lb::Number, ub::Number, p)
    T = _kriging_eltype(x)
    return T(0.5) / max(T(1.0e-6) * abs(ub - lb), T(std(x)))^p
end
function _kriging_default_theta(x, lb, ub, p)
    T = _kriging_eltype(x)
    return [
        T(0.5) / max(T(1.0e-6) * T(norm(ub .- lb)), T(std(x_i[i] for x_i in x)))^p[i]
            for i in eachindex(x[1])
    ]
end

# The default correlation smoothness, at the samples' own precision.
_kriging_default_p(x::Any, ::Number) = 2 * one(_kriging_eltype(x))
_kriging_default_p(x, _) = fill(2 * one(_kriging_eltype(x)), length(x[1]))

# The largest condition number the regularized correlation matrix may have.
#
# This was 1e8, which is conservative by four orders of magnitude: at 1e12 a
# Float64 solve still retains about four significant digits, and the escalating
# factorization below still catches a matrix no jitter can save. The difference
# is not academic — the nugget relaxes interpolation, and a fitted correlation
# scale is precisely where R is most ill-conditioned, so an over-tight target
# throws away most of what fitting the scale buys. Across a six-problem
# benchmark, raising it lowered prediction error by up to a factor of eighteen
# on the three problems where the nugget fires at all, and left the other three
# untouched; the worst training-point reproduction error improved with it.
const _KRIGING_KAPPA_MAX = 1.0e12

# Estimate a nugget from the maximum allowed condition number. This regularizes
# R so that samples may lie close together without R becoming singular, at the
# cost of slightly relaxing the interpolation condition. Derived from "An
# analytic comparison of regularization methods for Gaussian Processes" by
# Mohammadi et al (https://arxiv.org/pdf/1602.00853.pdf).
#
# `eigmin`/`eigmax` ask LAPACK for the two extremal eigenvalues rather than the
# whole spectrum, and are only defined for BLAS element types; for generic ones
# the criterion is skipped and the factorization below escalates instead.
function _kriging_nugget(R, ::Type{T}) where {T <: Union{Float32, Float64}}
    all(isfinite, R) || return zero(T)
    S = Symmetric(R)
    # At the matrix' own precision: as a Float64 literal this alone promoted a
    # Float32 solve, through `R + Diagonal(nugget)`.
    κ = T(_KRIGING_KAPPA_MAX)
    # The symmetric eigensolver can fail to converge on a correlation matrix
    # driven to the all-ones limit by an extreme scale, which the likelihood
    # search visits routinely. Skipping the criterion leaves the escalating
    # factorization below to size the nugget instead.
    # One decomposition, not two: `eigmin` and `eigmax` each run a full
    # symmetric eigensolve, and this is the dominant cost of a likelihood
    # evaluation, which the hyperparameter search makes thousands of.
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
# round-off leaves it marginally indefinite. The Cholesky factorization is both
# cheaper and more accurate than `inv(R)`, and every use site needs only solves.
# `condition_nugget = false` skips the eigenvalue criterion and lets the
# escalation below size the jitter alone. Kept for callers that only need the
# factorization to exist; the likelihood is not one of them, since the objective
# has to see the same regularization the final fit will use.
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

function _check_kriging_samples(x)
    if length(x) != length(unique(x))
        throw(
            ArgumentError(
                "There is a repetition in the samples, cannot build Kriging: " *
                    "duplicate points make the correlation matrix singular."
            )
        )
    end
    return nothing
end

function Kriging(
        x, y, lb::Number, ub::Number; p = _kriging_default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_kriging_samples(x)

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

# The power-exponential correlation matrix. Scalar samples index as length-1
# containers, so one implementation serves the scalar and multidimensional case,
# as it does for `_kriging_r`.
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
        x, y, lb, ub; p = _kriging_default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_kriging_samples(x)

    for i in eachindex(x[1])
        if p[i] > 2.0 || p[i] <= 0.0
            throw(ArgumentError("All p must be in (0, 2]! Got: $p."))
        end
    end

    theta_auto = theta === nothing
    theta = theta_auto ? _kriging_default_theta(x, lb, ub, p) : theta

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

# How far, in decades, the fitted correlation scale may move from the
# data-derived default. The default already carries the right order of magnitude
# — it is the inverse sample spread — so a search centred on it is both better
# conditioned and far cheaper than a fixed box spanning forty decades, which is
# what an absolute bound would need in order to cover designs whose coordinates
# differ by ten orders of magnitude.
const _KRIGING_THETA_DECADES = 5.0

# Nelder-Mead's own default is 1000 iterations per start, which is far more than
# a handful of correlation scales needs and is paid once per start.
const _KRIGING_MAXITERS = 250

# Latin-hypercube starts in addition to the data-derived one. Four rather than
# ten: across a six-problem benchmark the two gave identical fits, and four costs
# half as much. Zero is not enough — on a Levy function the extra starts were
# worth a factor of four, so the sweep is doing real work.
const _KRIGING_N_START = 4

"""
    _kriging_loglik(x, y, p, theta)

Concentrated log-likelihood of the correlation scale, up to additive constants:

    l(θ) = -n/2 log σ̂²(θ) - 1/2 log |R(θ)|

This is the objective Jones (2001) §2, DACE and SMT all maximize over `θ`. A
scale whose correlation matrix cannot be factorized scores `-Inf` rather than
raising, so the search simply walks away from it.
"""
function _kriging_loglik(x, y, p, theta)
    n = length(x)
    R = _kriging_correlation(x, p, theta)
    all(isfinite, R) || return -Inf
    # The conditioning nugget must be the *same* one the final fit uses.
    # Without it the likelihood is unbounded as the scale shrinks: R goes
    # near-singular, `logdet(R)` dives, and the search walks off to a degenerate
    # correlation length that predicts worse than the heuristic it started from.
    F = _try_kriging_factorize(R)
    F === nothing && return -Inf
    _, _, sigma = _kriging_gls(F, y, n)
    (isfinite(sigma) && sigma > 0) || return -Inf
    return -n / 2 * log(sigma) - logdet(F) / 2
end

# Fit the correlation scale by maximum likelihood, in log10 space so that the
# search is scale-free and the positivity constraint is automatic.
function _fit_kriging_theta(x, y, p, theta0; n_start::Integer = _KRIGING_N_START, multistart = true, maxiters = _KRIGING_MAXITERS)
    scalar = theta0 isa Number
    # The search itself always runs in Float64: the Latin-hypercube sampler
    # cannot work in a non-bits element type, and the extra precision would buy
    # nothing for a correlation scale. The fitted value converts back below.
    u0 = log10.(Float64.(scalar ? [theta0] : collect(theta0)))
    lo = u0 .- _KRIGING_THETA_DECADES
    hi = u0 .+ _KRIGING_THETA_DECADES
    negll(u, _) = begin
        theta = 10.0 .^ clamp.(u, lo, hi)
        v = _kriging_loglik(x, y, p, scalar ? theta[1] : theta)
        return isfinite(v) ? -v : Inf
    end
    u, value = _multistart_optimize(
        negll, u0, lo, hi; n_start = n_start, multistart = multistart,
        maxiters = maxiters
    )
    # Every start failed: keep the data-derived default rather than a scale the
    # likelihood never actually endorsed.
    isfinite(value) || return theta0
    # The search runs in Float64 for conditioning; the scale goes back to the
    # design's own element type so a Float32 fit stays a Float32 fit.
    theta = 10.0 .^ u
    return scalar ? oftype(theta0, theta[1]) : convert(typeof(theta0), theta)
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
        # Refit from the scale already in hand, without the multi-start sweep:
        # the extended sample set is a small perturbation of the one that scale
        # was fitted to, so a local refinement is both enough and affordable in
        # the `update!`-in-a-loop pattern optimizers use.
        k.theta = _fit_kriging_theta(k.x, k.y, k.p, k.theta; multistart = false)
    elseif k.theta_auto
        k.theta = _kriging_default_theta(k.x, k.lb, k.ub, k.p)
    end
    k.mu, k.b, k.sigma, k.R_fact = _calc_kriging_coeffs(k.x, k.y, k.p, k.theta)
    return nothing
end
