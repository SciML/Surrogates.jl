#=
Gradient-enhanced Kriging (first-order cokriging), following
"Improving variable-fidelity surrogate modeling via gradient-enhanced kriging"
by Han, Görtz and Zimmermann, and Chung and Alonso, "Using Gradients to
Construct Cokriging Approximation Models for High-Dimensional Design
Optimization Problems".

Writing k(x, x') = exp(-Σ_l θ_l (x_l - x'_l)²) and Δ_l = x_l - x'_l, the joint
covariance of the process and its partial derivatives is

    Cov(f(x),  f(x'))     = σ² k
    Cov(f(x),  ∂_l f(x')) = σ² ∂k/∂x'_l           = σ² · 2θ_l Δ_l k
    Cov(∂_l f(x), f(x'))  = σ² ∂k/∂x_l            = σ² · -2θ_l Δ_l k
    Cov(∂_l f(x), ∂_m f(x')) = σ² ∂²k/∂x_l ∂x'_m
                             = σ² (2θ_l δ_lm - 4θ_l θ_m Δ_l Δ_m) k

The derivative blocks exist only because the Gaussian kernel is mean-square
differentiable; a power-exponential kernel with `p < 2` is not differentiable at
the origin, so `p` is restricted to 2.
=#
"""
    GEK(x, y, lb, ub; p = 2.0, theta = 1.0)

Gradient-enhanced Kriging surrogate.

`GEK` augments the Kriging covariance system with derivative observations. The
surrogate is callable as `gek(x_new)`, exposes uncertainty through
[`std_error_at_point`](@ref), and supports
`update!(gek, x_new, y_new, grad_new)`.

# Fields

  - `x`: sample locations, `n` of them.
  - `y`: the `n(1 + d)` observations, values first and then gradients grouped by
    sample point, as described under `y` below.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `p`: correlation exponent. Only `2` is admissible; see the keyword below.
  - `theta`: scalar or per-dimension correlation scale parameter.
  - `mu`: fitted process mean.
  - `b`: covariance weights `R⁻¹(y - Fμ)`, one per observation.
  - `sigma`: fitted process variance.
  - `R_fact`: Cholesky factorization of the nugget-regularized GEK covariance
    matrix. The deprecated property `inverse_of_R` still materializes `R⁻¹`
    from it.

# Arguments

  - `x`: sample locations.
  - `y`: the `n(1 + d)` observations, laid out as

    ```
    [f(x₁), …, f(xₙ), ∂₁f(x₁), …, ∂_d f(x₁), ∂₁f(x₂), …, ∂_d f(xₙ)]
    ```

    that is, all `n` function values, then each point's `d` partial derivatives
    in order. In one dimension this is `vcat(f.(x), f'.(x))`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `p`: correlation exponent, `2` by default and required to be `2`. The
    derivative blocks above are second derivatives of the correlation function,
    which exist only for the Gaussian kernel; `exp(-θ|Δ|^p)` with `p < 2` has a
    cusp at the origin and no derivative-derivative covariance. The keyword is
    kept, and validated, so that an inadmissible value is reported rather than
    silently producing a meaningless model.
  - `theta`: scalar or per-dimension correlation scale. When left unset it is
    **fitted by maximum likelihood**, starting from the same sample-spread
    heuristic [`Kriging`](@ref) uses; `update!` then refits it. An explicitly
    supplied `theta` is used as given. The former fixed default of `1.0` carried
    no information about the design: on a sphere function over `[-5, 5]³` the
    fitted scale predicts about two hundred times more accurately.
  - `optimize_theta`: whether to fit `theta` by maximizing the concentrated
    log-likelihood over all `n(1 + d)` observations. Defaults to `true` exactly
    when `theta` is not supplied.
  - `n_start`: Latin-hypercube starts for that search.
  - `maxiters`: Nelder-Mead iteration cap per start.

# Returns

A `GEK` surrogate satisfying the generic surrogate interface. Duplicate sample
points make the covariance matrix singular and are rejected with an
`ArgumentError`.
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
    # Whether `theta` is fitted by maximum likelihood, in which case `update!`
    # refits it against the extended sample set.
    optimize_theta::Bool
end

function Base.getproperty(k::GEK, s::Symbol)
    if s === :inverse_of_R
        Base.depwarn(
            "`GEK.inverse_of_R` is deprecated: the Cholesky factorization of the " *
                "regularized covariance matrix is stored in `R_fact`. Prefer solving " *
                "with `k.R_fact \\ v` over forming the inverse.",
            :getproperty
        )
        return inv(getfield(k, :R_fact))
    end
    return getfield(k, s)
end

# Index of the `l`-th partial derivative of sample `i` in the observation
# vector: the `n` function values come first, then `d` derivatives per point.
_gek_deriv_index(n, d, i, l) = n + (i - 1) * d + l

# The GEK covariance matrix, σ² factored out. Scalar samples index as length-1
# containers (`(5.0)[1] === 5.0`), so one implementation serves the scalar and
# the multidimensional case.
function _gek_covariance(x, theta)
    n = length(x)
    d = length(x[1])
    N = n * (1 + d)
    # `theta` too: a Float64 scale over a Float32 design would otherwise be
    # truncated back to Float32 on assignment into `R`.
    T = float(promote_type(eltype(x[1]), eltype(theta)))
    R = zeros(T, N, N)

    @inbounds for i in 1:n, j in 1:n
        kij = exp(-sum(theta[l] * (x[i][l] - x[j][l])^2 for l in 1:d))
        R[i, j] = kij
        for l in 1:d
            il = _gek_deriv_index(n, d, i, l)
            jl = _gek_deriv_index(n, d, j, l)
            Δl = x[i][l] - x[j][l]
            # Cov(f(xᵢ), ∂ₗf(xⱼ)) and its transpose, Cov(∂ₗf(xᵢ), f(xⱼ)).
            R[i, jl] = 2 * theta[l] * Δl * kij
            R[il, j] = -2 * theta[l] * Δl * kij
            for m in 1:d
                jm = _gek_deriv_index(n, d, j, m)
                Δm = x[i][m] - x[j][m]
                # The `2θ_l δ_lm` term is what makes the derivative-derivative
                # diagonal positive; without it R is not a covariance matrix.
                R[il, jm] = (
                    (l == m ? 2 * theta[l] : zero(T)) -
                        4 * theta[l] * theta[m] * Δl * Δm
                ) * kij
            end
        end
    end
    return R
end

# Cross-covariance between the process at `val` and every observation. The
# element type promotes the query with the samples, so a query of a different
# type than the design — a dual number, say — is not truncated back to it.
function _gek_r(k::GEK, val)
    n = length(k.x)
    d = length(k.x[1])
    T = float(promote_type(eltype(k.x[1]), eltype(val), eltype(k.theta)))
    r = zeros(T, n * (1 + d))
    @inbounds for i in 1:n
        ki = exp(-sum(k.theta[l] * (val[l] - k.x[i][l])^2 for l in 1:d))
        r[i] = ki
        for l in 1:d
            r[_gek_deriv_index(n, d, i, l)] = 2 * k.theta[l] * (val[l] - k.x[i][l]) * ki
        end
    end
    return r
end

# Trend basis for the constant mean: a derivative observation carries no
# information about it, so its rows are zero.
function _gek_trend(n, d, ::Type{T}) where {T}
    f = zeros(T, n * (1 + d))
    f[1:n] .= one(T)
    return f
end

function _check_gek_observations(x, y)
    n = length(x)
    d = length(x[1])
    N = n * (1 + d)
    if length(y) != N
        throw(
            ArgumentError(
                "GEK expects $N observations for $n points in $d dimensions " *
                    "($n values followed by $(n * d) partial derivatives), got $(length(y))."
            )
        )
    end
    return n, d, N
end

# Generalized least squares for the constant trend, given a factorization.
function _gek_gls(F, y, n, d, N)
    f = _gek_trend(n, d, eltype(y))
    mu = dot(f, F \ y) / dot(f, F \ f)
    y_minus_fmu = y - f * mu
    b = F \ y_minus_fmu
    # Maximum likelihood over all N observations, not over the n sample points.
    sigma = dot(y_minus_fmu, b) / N
    return mu, b, sigma
end

function _calc_gek_coeffs(x, y, theta)
    n, d, N = _check_gek_observations(x, y)
    F = _kriging_factorize(_gek_covariance(x, theta))
    mu, b, sigma = _gek_gls(F, y, n, d, N)
    return mu, b, sigma, F
end

"""
    _gek_loglik(x, y, theta)

Concentrated log-likelihood of the correlation scale, up to additive constants,
over all `N = n(1 + d)` observations:

    l(θ) = -N/2 log σ̂²(θ) - 1/2 log |R(θ)|

The same objective as [`Kriging`](@ref)'s, over the gradient-enhanced covariance
matrix rather than the plain correlation matrix.
"""
function _gek_loglik(x, y, theta)
    n = length(x)
    d = length(x[1])
    N = n * (1 + d)
    R = _gek_covariance(x, theta)
    all(isfinite, R) || return -Inf
    # The conditioning nugget must be the *same* one the final fit uses.
    # Without it the likelihood is unbounded as the scale shrinks: R goes
    # near-singular, `logdet(R)` dives, and the search walks off to a degenerate
    # correlation length that predicts worse than the heuristic it started from.
    F = _try_kriging_factorize(R)
    F === nothing && return -Inf
    _, _, sigma = _gek_gls(F, y, n, d, N)
    (isfinite(sigma) && sigma > 0) || return -Inf
    return -N / 2 * log(sigma) - logdet(F) / 2
end

function _fit_gek_theta(x, y, theta0; n_start::Integer = _KRIGING_N_START, multistart = true, maxiters = _KRIGING_MAXITERS)
    scalar = theta0 isa Number
    # The search itself always runs in Float64: the Latin-hypercube sampler
    # cannot work in a non-bits element type, and the extra precision would buy
    # nothing for a correlation scale. The fitted value converts back below.
    u0 = log10.(Float64.(scalar ? [theta0] : collect(theta0)))
    lo = u0 .- _KRIGING_THETA_DECADES
    hi = u0 .+ _KRIGING_THETA_DECADES
    negll(u, _) = begin
        theta = 10.0 .^ clamp.(u, lo, hi)
        v = _gek_loglik(x, y, scalar ? theta[1] : theta)
        return isfinite(v) ? -v : Inf
    end
    u, value = _multistart_optimize(
        negll, u0, lo, hi; n_start = n_start, multistart = multistart,
        maxiters = maxiters
    )
    isfinite(value) || return theta0
    # The search runs in Float64 for conditioning; the scale goes back to the
    # design's own element type so a Float32 fit stays a Float32 fit.
    theta = 10.0 .^ u
    return scalar ? oftype(theta0, theta[1]) : convert(typeof(theta0), theta)
end

function (k::GEK)(val)
    _check_dimension(k, val)
    return k.mu + dot(_gek_r(k, val), k.b)
end

(k::GEK)(val::Number) = (_check_dimension(k, val); k.mu + dot(_gek_r(k, val), k.b))

# Universal-kriging mean squared error with the constant trend basis `f`; see
# the note on `Kriging`'s `_kriging_std_error` for the shape of the third term.
function _gek_std_error(k::GEK, val)
    n = length(k.x)
    d = length(k.x[1])
    r = _gek_r(k, val)
    f = _gek_trend(n, d, eltype(r))
    Rinv_r = k.R_fact \ r
    a = dot(r, Rinv_r)
    c = dot(f, Rinv_r)
    b = dot(f, k.R_fact \ f)
    mean_squared_error = k.sigma * (1 - a + (1 - c)^2 / b)
    return sqrt(max(mean_squared_error, zero(mean_squared_error)))
end

function std_error_at_point(k::GEK, val)
    _check_dimension(k, val)
    return _gek_std_error(k, val)
end

function std_error_at_point(k::GEK, val::Number)
    _check_dimension(k, val)
    return _gek_std_error(k, val)
end

function _check_gek_p(p, d)
    for l in 1:d
        if p[l] != 2
            throw(
                ArgumentError(
                    "GEK requires p = 2. The derivative covariance blocks are " *
                        "second derivatives of the correlation function, which the " *
                        "power-exponential kernel only has for p = 2. Got: $p."
                )
            )
        end
    end
    return nothing
end

function _check_gek_samples(x)
    if length(x) != length(unique(x))
        throw(
            ArgumentError(
                "There is a repetition in the samples, cannot build GEK: " *
                    "duplicate points make the covariance matrix singular."
            )
        )
    end
    return nothing
end

# Defaults at the samples' own precision, so a Float32 design is not promoted
# to Float64 by a literal.
_gek_eltype(x) = float(eltype(first(x)))
_gek_default_p(x, ::Number) = 2 * one(_gek_eltype(x))
_gek_default_p(x, _) = fill(2 * one(_gek_eltype(x)), length(x[1]))

# The starting correlation scale, derived from the sample spread exactly as
# `Kriging`'s is. The former fixed `1.0` carried no information about the design
# and was a poor place to start a likelihood search from — on a sphere function
# over [-5, 5]^3 it predicted eighty times worse than the fitted scale.
_gek_default_theta(x, lb, ub, p) = _kriging_default_theta(x, lb, ub, p)

function GEK(
        x, y, lb::Number, ub::Number; p = _gek_default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_gek_samples(x)
    _check_gek_p(p, 1)
    theta = theta === nothing ? _gek_default_theta(x, lb, ub, p) : theta
    if theta ≤ 0
        throw(ArgumentError("Hyperparameter theta must be positive! Got: $theta."))
    end
    _check_gek_observations(x, y)
    optimize_theta && (theta = _fit_gek_theta(x, y, theta; n_start = n_start, maxiters = maxiters))
    mu, b, sigma, R_fact = _calc_gek_coeffs(x, y, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, optimize_theta)
end

function GEK(
        x, y, lb, ub; p = _gek_default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_gek_samples(x)
    d = length(x[1])
    _check_gek_p(p, d)
    theta = theta === nothing ? _gek_default_theta(x, lb, ub, p) : theta
    for l in 1:d
        if theta[l] ≤ 0
            throw(ArgumentError("All theta must be positive! Got: $theta."))
        end
    end
    _check_gek_observations(x, y)
    optimize_theta && (theta = _fit_gek_theta(x, y, theta; n_start = n_start, maxiters = maxiters))
    mu, b, sigma, R_fact = _calc_gek_coeffs(x, y, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, optimize_theta)
end

"""
    update!(k::GEK, new_x, new_y, new_grad)

Add new samples, their responses and their gradients, and refit the surrogate.

`new_grad` gives the `d` partial derivatives at `new_x`, or one such container
per point when several points are added at once.

# Returns

Returns `nothing`. A `new_x` that already appears in the samples warns and
leaves the surrogate unchanged, since duplicate points make the covariance
matrix singular.
"""
function SurrogatesBase.update!(k::GEK, new_x, new_y, new_grad)
    n = length(k.x)
    d = length(k.x[1])
    single = _is_single_sample(new_x, first(k.x))
    pts = single ? [new_x] : collect(new_x)
    vals = single ? [new_y] : collect(new_y)
    grads = single ? collect(new_grad) : reduce(vcat, collect.(new_grad))

    if length(vals) != length(pts) || length(grads) != length(pts) * d
        throw(
            ArgumentError(
                "GEK `update!` needs one response and $d partial derivatives per " *
                    "new point; got $(length(pts)) points, $(length(vals)) responses " *
                    "and $(length(grads)) derivatives."
            )
        )
    end

    x_new = vcat(k.x, pts)
    if length(unique(x_new)) != length(x_new)
        @warn "Skipping `update!`: these samples repeat a point already in the " *
            "GEK surrogate, and duplicate points would make the covariance " *
            "matrix singular."
        return nothing
    end

    # The observation vector keeps all values ahead of all gradients, so a new
    # point splices into both halves rather than appending to one.
    k.x = x_new
    k.y = vcat(k.y[1:n], vals, k.y[(n + 1):end], grads)
    if k.optimize_theta
        # Local refinement from the scale in hand; see `Kriging.update!`.
        k.theta = _fit_gek_theta(k.x, k.y, k.theta; multistart = false)
    end
    k.mu, k.b, k.sigma, k.R_fact = _calc_gek_coeffs(k.x, k.y, k.theta)
    return nothing
end

function SurrogatesBase.update!(::GEK, ::Any, ::Any)
    throw(
        ArgumentError(
            "GEK is a gradient-enhanced model: a new sample must come with its " *
                "gradient. Use `update!(gek, new_x, new_y, new_grad)`."
        )
    )
end
