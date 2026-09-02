"""
KPLS: Kriging combined with Partial Least Squares for high-dimensional problems.

Based on: Bouhlel et al. (2016), "Improving kriging surrogates of high-dimensional design
models by Partial Least Squares dimension reduction", Struct Multidisc Optim 53:935–952.

KPLS reduces the number of kriging hyperparameters from d (input dimension) to h
(number of PLS components, h << d) by using PLS to define a new covariance kernel:

    R(x, x') = exp(-Σ_{l=1}^h θ_l * Σ_{k=1}^d (w*_lk)^2 * (xk - x'k)^2)

where W* ∈ R^(d × h) are the PLS rotation coefficients.
"""
mutable struct KPLS{T, X, Y} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    x_matrix::Matrix{T}
    y_matrix::Matrix{T}
    xl::Matrix{T}           # xlimits [d, 2]
    n_comp::Int             # number of PLS components h
    beta::Vector{T}         # GLS regression weights
    gamma::Matrix{T}        # GP weights
    theta::Vector{T}        # h hyperparameters (one per PLS component)
    reduced_likelihood_function_value::T
    X_offset::Matrix{T}     # mean of X (for standardization)
    X_scale::Matrix{T}      # std of X (for standardization)
    X_after_std::Matrix{T}  # standardized training X
    pls_mean::Matrix{T}     # PLS rotation matrix W* [d, h]
    y_mean::T               # mean of y
    y_std::T                # std of y
end

# The PLS basis has at most `d` components, and one correlation scale is fitted
# per retained component. Both are checked at the constructor: an oversized
# `n_comp` otherwise leaves `_modified_pls` returning columns of zeros for the
# components that do not exist, and a `theta` of the wrong length dies six
# frames down in `squar_exp`'s `reshape` as an opaque `DimensionMismatch`.
function _check_pls_components(name, n_comp, d, theta)
    if n_comp < 1 || n_comp > d
        throw(
            ArgumentError(
                "$name needs 1 to $d PLS components for a $d-dimensional design! " *
                    "Got n_comp = $(n_comp)."
            )
        )
    end
    if length(theta) != n_comp
        throw(
            ArgumentError(
                "$name needs one correlation scale per PLS component: " *
                    "length(theta) = $(length(theta)) but n_comp = $(n_comp)."
            )
        )
    end
    return nothing
end

# Training points outside `[lb, ub]` cannot be standardized against the bounds
# the model reports, so they are rejected rather than silently returning a
# `nothing` that every later call turns into a `MethodError`.
function _check_pls_bounds(name, X, xlimits)
    if bounds_error(X, xlimits)
        throw(
            ArgumentError("Some training points lie outside [lb, ub]; cannot build $name.")
        )
    end
    return nothing
end

"""
    _compute_pls(X, y, n_comp)

Compute PLS coefficients (W*) from input matrix X [n_obs, d] and output y [n_obs, 1].
Returns (coeff_pls [d, n_comp], X, y).
"""
function _compute_pls(X, y, n_comp)
    # Pass copies: _modified_pls calls _center_scale which modifies its inputs in-place
    coeff_pls = _modified_pls(copy(X), copy(y), n_comp)  # [d, n_comp] = W*
    return abs.(coeff_pls), X, y
end

"""
    _optimize_theta(theta_init, kernel_type, d, nt, ij, y_norma; multistart = true, n_start = 10)

Optimize KPLS hyperparameters `theta` by maximizing the reduced log-likelihood.

Uses Nelder-Mead (via Optimization.jl + OptimizationOptimJL) in log₁₀(theta) space
(bounds [-20, 20]). `theta_init` is always used as a starting point. When
`multistart` is `true` (the default), `n_start` additional Latin Hypercube starts
spread across the log₁₀(theta) bounds are also tried, to improve robustness
against the reduced likelihood surface's local optima. Set `multistart = false` for a cheap
local refinement when `theta_init` is already known to be a good guess (e.g. the
KPLSK full-dimensional refinement stage).
"""
function _optimize_theta(
        theta_init, kernel_type, d, nt, ij, y_norma; multistart = true, n_start = 10,
        nugget = _PLS_NUGGET, noise = 0.0, max_escalations = 0
    )
    n_comp = length(theta_init)
    log10_lb = fill(-20.0, n_comp)
    log10_ub = fill(20.0, n_comp)

    function neg_rlf(log10_theta, _)
        theta = 10 .^ clamp.(log10_theta, log10_lb, log10_ub)
        try
            _, _, val = _reduced_likelihood_function(
                theta, kernel_type, d, nt, ij, y_norma;
                nugget = nugget, noise = noise, max_escalations = max_escalations
            )
            return isfinite(val) ? -val : Inf
        catch e
            # Unqualified: `LinearAlgebra` itself is not in scope here, so the
            # qualified form raised `UndefVarError` from inside the handler.
            e isa Union{SingularException, PosDefException} || rethrow()
            return Inf
        end
    end

    # Multi-start over the log10(theta) box; see `_multistart_optimize`.
    best_log10_theta, _ = _multistart_optimize(
        neg_rlf, log10.(theta_init), log10_lb, log10_ub;
        n_start = n_start, multistart = multistart
    )
    return 10 .^ best_log10_theta
end

"""
    KPLS(x_vec, y_vec, n_comp, lb, ub, theta) -> KPLS

Construct a Kriging-with-partial-least-squares (KPLS) surrogate.

KPLS projects the input coordinates onto `n_comp` partial-least-squares
components, then fits a Kriging model in that reduced space. It is intended for
high-dimensional inputs where a full anisotropic Kriging fit would require too
many correlation parameters.

# Fields

  - `x`: training points in the caller-provided representation.
  - `y`: training responses.
  - `x_matrix`: matrix representation of the training points.
  - `y_matrix`: column-matrix representation of the responses.
  - `xl`: lower and upper bounds stored as a two-column matrix.
  - `n_comp`: number of retained PLS components.
  - `beta`: generalized-least-squares trend coefficients.
  - `gamma`: Kriging residual coefficients.
  - `theta`: optimized correlation scales in the PLS space.
  - `reduced_likelihood_function_value`: fitted reduced likelihood value.
  - `X_offset`: input centering values.
  - `X_scale`: input scaling values.
  - `X_after_std`: standardized projected training inputs.
  - `pls_mean`: PLS projection coefficients.
  - `y_mean`: response centering value.
  - `y_std`: response scaling value.

# Arguments

  - `x_vec`: vector of tuples containing the training points.
  - `y_vec`: vector of responses, with one value for each point in `x_vec`.
  - `n_comp::Integer`: number of PLS components. It must not exceed the input
    dimension.
  - `lb`: lower bounds for the input coordinates.
  - `ub`: upper bounds for the input coordinates, matching `lb`.
  - `theta`: positive initial correlation scales, with one value per component.

# Returns

A callable `KPLS` surrogate supporting the generic call and `update!` interface.
The stored `theta` contains the optimized correlation scales.

# Example

```julia
using Surrogates

objective(x) = sum(abs2, x)
lb, ub = [-1.0, -1.0], [1.0, 1.0]
x = sample(12, lb, ub, SobolSample())
y = objective.(x)
surrogate = KPLS(x, y, 1, lb, ub, [1.0])
surrogate((0.2, -0.1))
```
"""
function KPLS(x_vec, y_vec, n_comp, lb, ub, theta)
    xlimits = hcat(collect(Float64, lb), collect(Float64, ub))
    X = vector_of_tuples_to_matrix(x_vec)
    y = reshape(collect(Float64, y_vec), (size(X, 1), 1))
    _check_pls_components("KPLS", n_comp, size(X, 2), theta)
    _check_pls_bounds("KPLS", X, xlimits)

    pls_mean, X_after_PLS, y_after_PLS = _compute_pls(X, y, n_comp)
    X_after_std, y_after_std, X_offset, y_mean, X_scale, y_std = standardization(
        copy(X_after_PLS), copy(y_after_PLS)
    )
    D, ij = cross_distances(X_after_std)
    d = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean)
    nt = size(X_after_PLS, 1)

    # Optimize theta by maximizing the reduced log-likelihood.
    theta_opt = _optimize_theta(theta, "squar_exp", d, nt, ij, y_after_std)

    beta, gamma, reduced_likelihood_function_value = _reduced_likelihood_function(
        theta_opt, "squar_exp", d, nt, ij, y_after_std
    )

    return KPLS(
        x_vec, y_vec, X, y, xlimits, n_comp, beta, gamma, theta_opt,
        reduced_likelihood_function_value,
        X_offset, X_scale, X_after_std, pls_mean, y_mean, y_std
    )
end

"""
    (k::KPLS)(x_vec)

Predict the output at input point `x_vec` (a tuple or vector).
"""
function (k::KPLS)(x_vec)
    _check_dimension(k, x_vec)
    X_test = prep_data_for_pred([x_vec])
    n_eval = size(X_test, 1)
    X_cont = (X_test .- k.X_offset) ./ k.X_scale
    dx = differences(X_cont, k.X_after_std)
    pred_d = componentwise_distance_PLS(dx, "squar_exp", k.n_comp, k.pls_mean)
    nt = size(k.X_after_std, 1)
    r = transpose(reshape(squar_exp(k.theta, pred_d), (nt, n_eval)))
    f = ones(n_eval, 1)
    y_ = (f * k.beta) + (r * k.gamma)
    y = k.y_mean .+ k.y_std * y_
    return y[1]
end

"""
    (k::KPLS)(val::Number)

Predict at a scalar input (1D case).
"""
function (k::KPLS)(val::Number)
    _check_dimension(k, val)
    return k((val,))
end

"""
    update!(k::KPLS, new_x, new_y)

Add a new sample point and re-train the KPLS model.
"""
function SurrogatesBase.update!(k::KPLS, new_x, new_y)
    new_x_mat = prep_data_for_pred([new_x])
    # A duplicate is a no-op, not an error; see `Kriging`'s `update!`.
    if vec(new_x_mat) in eachrow(k.x_matrix)
        @warn "Skipping `update!`: this sample already exists in the KPLS " *
            "surrogate, and duplicate points would make the correlation matrix singular."
        return nothing
    end

    if bounds_error(new_x_mat, k.xl)
        throw(ArgumentError("The new sample lies outside [lb, ub]; cannot update KPLS."))
    end

    # `vcat` rather than `push!`: the containers are the caller's own, and
    # growing them behind their back would extend a design they still hold.
    k.x = vcat(k.x, [new_x])
    k.y = vcat(k.y, new_y)
    k.x_matrix = vcat(k.x_matrix, new_x_mat)
    k.y_matrix = vcat(k.y_matrix, reshape([Float64(new_y)], (1, 1)))

    pls_mean, X_after_PLS, y_after_PLS = _compute_pls(k.x_matrix, k.y_matrix, k.n_comp)
    k.X_after_std, y_after_std, k.X_offset, k.y_mean, k.X_scale, k.y_std = standardization(
        copy(X_after_PLS), copy(y_after_PLS)
    )
    D, ij = cross_distances(k.X_after_std)
    k.pls_mean = pls_mean
    d = componentwise_distance_PLS(D, "squar_exp", k.n_comp, k.pls_mean)
    nt = size(X_after_PLS, 1)
    k.theta = _optimize_theta(k.theta, "squar_exp", d, nt, ij, y_after_std)
    k.beta, k.gamma, k.reduced_likelihood_function_value = _reduced_likelihood_function(
        k.theta, "squar_exp", d, nt, ij, y_after_std
    )
    return nothing
end
