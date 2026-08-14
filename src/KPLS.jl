using CommonSolve: solve
using SciMLBase: OptimizationProblem
using OptimizationOptimJL: NelderMead

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
spread across the log₁₀(theta) bounds are also tried, following the multistart
strategy used by SMT's Kriging-based optimizer, to improve robustness against the
reduced likelihood surface's local optima. Set `multistart = false` for a cheap
local refinement when `theta_init` is already known to be a good guess (e.g. the
KPLSK full-dimensional refinement stage).
"""
function _optimize_theta(
        theta_init, kernel_type, d, nt, ij, y_norma; multistart = true, n_start = 10
    )
    n_comp = length(theta_init)
    log10_lb = fill(-20.0, n_comp)
    log10_ub = fill(20.0, n_comp)

    function neg_rlf(log10_theta, _)
        theta = 10 .^ clamp.(log10_theta, log10_lb, log10_ub)
        try
            _, _, val = _reduced_likelihood_function(theta, kernel_type, d, nt, ij, y_norma)
            return isfinite(val) ? -val : Inf
        catch e
            e isa Union{LinearAlgebra.SingularException, LinearAlgebra.PosDefException} ||
                rethrow()
            return Inf
        end
    end

    # Multi-start: user-provided theta + n_start Latin Hypercube samples across the
    # log10(theta) bounds, covering the space more evenly than a handful of fixed points.
    starts = if multistart
        lhs_pts = sample(n_start, log10_lb, log10_ub, LatinHypercubeSample())
        # `sample` returns a Vector{Float64} of scalars for n_comp == 1 (1D bounds) and
        # a Vector{NTuple{n_comp, Float64}} otherwise; normalize both to Vector{Float64}.
        lhs_starts = n_comp == 1 ? [[p] for p in lhs_pts] : [
                collect(Float64, p)
                for p in lhs_pts
            ]
        vcat([clamp.(log10.(theta_init), -20.0, 20.0)], lhs_starts)
    else
        [clamp.(log10.(theta_init), -20.0, 20.0)]
    end

    best_val = Inf
    best_log10_theta = starts[1]
    for x0 in starts
        # NelderMead is derivative-free; passing lb/ub here would make
        # OptimizationOptimJL wrap it in Fminbox, which requires gradients
        # and errors. Bounds are instead enforced by clamping in neg_rlf.
        prob = OptimizationProblem(neg_rlf, x0, nothing)
        sol = solve(prob, NelderMead())
        if sol.objective < best_val
            best_val = sol.objective
            best_log10_theta = sol.u
        end
    end

    return 10 .^ clamp.(best_log10_theta, log10_lb, log10_ub)
end

"""
    KPLS(x_vec, y_vec, n_comp, lb, ub, theta)

Constructor for KPLS surrogate.

# Arguments
- `x_vec`: vector of tuples containing input points
- `y_vec`: vector of output values
- `n_comp`: number of PLS components (h), typically 1–4; must be ≤ input dimension
- `lb`: lower bounds (vector)
- `ub`: upper bounds (vector)
- `theta`: initial hyperparameter vector of length n_comp (values > 0), used as
  the starting point for likelihood-based optimization. The stored `theta` field
  will contain the optimized values.
"""
function KPLS(x_vec, y_vec, n_comp, lb, ub, theta)
    xlimits = hcat(collect(Float64, lb), collect(Float64, ub))
    X = vector_of_tuples_to_matrix(x_vec)
    y = reshape(collect(Float64, y_vec), (size(X, 1), 1))

    if bounds_error(X, xlimits)
        println("X values outside bounds")
        return
    end

    pls_mean, X_after_PLS, y_after_PLS = _compute_pls(X, y, n_comp)
    X_after_std, y_after_std, X_offset, y_mean, X_scale, y_std = standardization(
        copy(X_after_PLS), copy(y_after_PLS)
    )
    D, ij = cross_distances(X_after_std)
    d = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean)
    nt = size(X_after_PLS, 1)

    # Optimize theta by maximizing the reduced log-likelihood (matches SMT behaviour)
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
    if vec(new_x_mat) in eachrow(k.x_matrix)
        println("Adding a sample that already exists. Cannot update KPLS.")
        return
    end

    if bounds_error(new_x_mat, k.xl)
        println("x values outside bounds")
        return
    end

    push!(k.x, new_x)
    push!(k.y, new_y)
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
