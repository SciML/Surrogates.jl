"""
    GEK(x, y, lb, ub; p = 2.0, theta = 1.0)

Gradient-enhanced Kriging surrogate.

Based on: Han, Görtz and Zimmermann (2013), "Improving variable-fidelity
surrogate modeling via gradient-enhanced kriging and a generalized hybrid
bridge function", Aerosp Sci Technol 25:177-189; and Chung and Alonso (2002),
"Using Gradients to Construct Cokriging Approximation Models for
High-Dimensional Design Optimization Problems", AIAA 2002-0317.

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
  - `theta`: correlation scale: a scalar in one dimension, one entry per input
    coordinate otherwise.
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
  - `theta`: correlation scale, a scalar in one dimension and one entry per
    input coordinate otherwise. When left unset it is **fitted by maximum
    likelihood**, starting from the same sample-spread heuristic
    [`Kriging`](@ref) uses; `update!` then refits it. An explicitly supplied
    `theta` is used as given.
  - `optimize_theta`: whether to fit `theta` by maximizing the concentrated
    log-likelihood over all `n(1 + d)` observations. Defaults to `true` exactly
    when `theta` is not supplied.
  - `n_start`: Latin-hypercube starts for that search.
  - `maxiters`: Nelder-Mead iteration cap per start.

# Returns

A `GEK` surrogate satisfying the generic surrogate interface. Duplicate sample
points make the covariance matrix singular and are rejected with an
`ArgumentError`.

!!! note "Conditioning"

    Direct GEK is far worse conditioned than plain Kriging. Derivative
    observations at nearby points are nearly redundant once the correlation
    length is long, so a fitted `theta` routinely puts `cond(R)` at `1e16`–`1e18`
    where the value block alone is near `1e9`. The nugget of [`Kriging`](@ref)
    regularizes it to a condition number of `1e12`, which keeps predictions
    accurate — the fitted scale beats the sample-spread heuristic by two to three
    orders of magnitude on smooth problems — at the cost of reproducing training
    points to about `1e-5` rather than to machine precision. Where that matters,
    [`GEKPLS`](@ref) is the indirect alternative: it adds Taylor-extrapolated
    points to an ordinary Kriging system instead of forming derivative covariance
    blocks, and so never builds this matrix.
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
    s === :inverse_of_R && return _deprecated_inverse_of_R(k, :GEK)
    return getfield(k, s)
end

# Index of the `l`-th partial derivative of sample `i` in the observation
# vector: the `n` function values come first, then `d` derivatives per point.
_gek_deriv_index(n, d, i, l) = n + (i - 1) * d + l

# The GEK covariance matrix, σ² factored out. Writing k(x, x') = exp(-Σ_l θ_l Δ_l²)
# with Δ_l = x_l - x'_l, the joint covariance of the process and its partials is
#
#     Cov(f(x),     f(x'))     = k
#     Cov(f(x),     ∂_l f(x')) = ∂k/∂x'_l        =  2θ_l Δ_l k
#     Cov(∂_l f(x), f(x'))     = ∂k/∂x_l         = -2θ_l Δ_l k
#     Cov(∂_l f(x), ∂_m f(x')) = ∂²k/∂x_l ∂x'_m  = (2θ_l δ_lm - 4θ_l θ_m Δ_l Δ_m) k
#
# These exist only because the Gaussian kernel is mean-square differentiable,
# which is why `p` is restricted to 2. Scalar samples index as length-1
# containers, so one implementation serves both input layouts.
function _gek_covariance(x, theta)
    n = length(x)
    d = length(x[1])
    N = n * (1 + d)
    # `theta` too, so a Float64 scale over a Float32 design is not truncated.
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
                # The `2θ_l δ_lm` term makes the derivative-derivative diagonal
                # positive; without it R is not a covariance matrix.
                R[il, jm] = (
                    (l == m ? 2 * theta[l] : zero(T)) -
                        4 * theta[l] * theta[m] * Δl * Δm
                ) * kij
            end
        end
    end
    return R
end

# Cross-covariance between the process at `val` and every observation.
#
# Built by concatenation rather than by writing into a preallocated vector:
# reverse-mode AD cannot differentiate through `setindex!`, and this is on the
# prediction path, so `Zygote.gradient(gek, x)` has to work. The element type
# follows from the arithmetic, so a dual number is not truncated back to the
# design's type.
function _gek_r(k::GEK, val)
    n = length(k.x)
    d = length(k.x[1])
    ks = [exp(-sum(k.theta[l] * (val[l] - k.x[i][l])^2 for l in 1:d)) for i in 1:n]
    # Flattened in `_gek_deriv_index` order — point by point, and within a point,
    # coordinate by coordinate — so the layout matches `_gek_covariance`. The
    # index arithmetic inverts `n + (i - 1) * d + l`; a nested comprehension
    # would read better but lowers to a `Flatten` iterator that reverse-mode AD
    # accumulates with `push!`.
    dks = [
        let i = (j - 1) ÷ d + 1, l = (j - 1) % d + 1
            2 * k.theta[l] * (val[l] - k.x[i][l]) * ks[i]
        end
            for j in 1:(n * d)
    ]
    return vcat(ks, dks)
end

# Trend basis for the constant mean: a derivative observation carries no
# information about it, so its rows are zero.
function _gek_trend(n, d, ::Type{T}) where {T}
    # Concatenated rather than filled in place: this is on the `std_error_at_point`
    # path, and reverse-mode AD cannot differentiate through the assignment.
    return vcat(ones(T, n), zeros(T, n * d))
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
    # Over all N observations, not the n sample points.
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
    # The same nugget the final fit uses; see `_kriging_loglik`.
    F = _try_kriging_factorize(R)
    F === nothing && return -Inf
    _, _, sigma = _gek_gls(F, y, n, d, N)
    (isfinite(sigma) && sigma > 0) || return -Inf
    return -N / 2 * log(sigma) - logdet(F) / 2
end

function _fit_gek_theta(x, y, theta0; kwargs...)
    return _fit_theta(theta -> _gek_loglik(x, y, theta), theta0; kwargs...)
end

function (k::GEK)(val)
    _check_dimension(k, val)
    return k.mu + dot(_gek_r(k, val), k.b)
end

(k::GEK)(val::Number) = (_check_dimension(k, val); k.mu + dot(_gek_r(k, val), k.b))

# The trend basis is zero on the derivative rows; see `_blup_std_error`.
function _gek_std_error(k::GEK, val)
    r = _gek_r(k, val)
    f = _gek_trend(length(k.x), length(k.x[1]), eltype(r))
    return _blup_std_error(k.sigma, r, f, k.R_fact)
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

function GEK(
        x, y, lb::Number, ub::Number; p = _default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_no_duplicate_samples("GEK", x)
    _check_gek_p(p, 1)
    theta = theta === nothing ? _kriging_default_theta(x, lb, ub, p) : theta
    if theta ≤ 0
        throw(ArgumentError("Hyperparameter theta must be positive! Got: $theta."))
    end
    _check_gek_observations(x, y)
    optimize_theta && (theta = _fit_gek_theta(x, y, theta; n_start = n_start, maxiters = maxiters))
    mu, b, sigma, R_fact = _calc_gek_coeffs(x, y, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, R_fact, optimize_theta)
end

function GEK(
        x, y, lb, ub; p = _default_p(x, first(x)),
        theta = nothing, optimize_theta = theta === nothing, n_start::Integer = _KRIGING_N_START,
        maxiters = _KRIGING_MAXITERS
    )
    _check_no_duplicate_samples("GEK", x)
    d = length(x[1])
    _check_kriging_length("p", p, d)
    _check_gek_p(p, d)
    theta = theta === nothing ? _kriging_default_theta(x, lb, ub, p) : theta
    _check_kriging_length("theta", theta, d)
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

    # All values sit ahead of all gradients, so a new point splices into both
    # halves rather than appending to one.
    k.x = x_new
    k.y = vcat(k.y[1:n], vals, k.y[(n + 1):end], grads)
    if k.optimize_theta
        # Local refinement from the scale in hand; see `Kriging`'s `update!`.
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
