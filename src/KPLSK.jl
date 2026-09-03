"""
KPLSK: KPLS followed by a full-dimensional Kriging refinement.

Based on: Bouhlel et al. (2016), "Improving kriging surrogates of high-dimensional design
models by Partial Least Squares dimension reduction", Struct Multidisc Optim 53:935–952.

KPLSK first fits a reduced-dimension KPLS model (h << d hyperparameters, see [`KPLS`](@ref))
and uses it only to obtain a good initial guess for a standard, full-dimensional (d
hyperparameters) anisotropic Kriging model:

    θ0_k = Σ_{l=1}^h θ_l * (w*_lk)^2,   k = 1, ..., d

where θ_l are the optimized KPLS hyperparameters and W* ∈ R^(d × h) are the PLS rotation
coefficients. Re-optimizing a d-dimensional Kriging model from this near-optimal starting
point is cheaper than optimizing from scratch, while giving the accuracy of a full
anisotropic Kriging model.
"""
mutable struct KPLSK{T, X, Y} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    x_matrix::Matrix{T}
    y_matrix::Matrix{T}
    xl::Matrix{T}           # xlimits [d, 2]
    n_comp::Int             # number of PLS components h (used only to seed theta)
    beta::Vector{T}         # GLS regression weights
    gamma::Matrix{T}        # GP weights
    theta::Vector{T}        # d hyperparameters (one per input dimension)
    theta_pls::Vector{T}    # h stage-1 KPLS hyperparameters (warm start for update!)
    reduced_likelihood_function_value::T
    X_offset::Matrix{T}     # mean of X (for standardization)
    X_scale::Matrix{T}      # std of X (for standardization)
    X_after_std::Matrix{T}  # standardized training X
    y_mean::T               # mean of y
    y_std::T                # std of y
end

"""
    _expand_kpls_theta(theta_pls, coeff_pls)

Expand KPLS hyperparameters `theta_pls` (length h) into full-dimensional Kriging
hyperparameters `theta0` (length d), using the PLS rotation coefficients
`coeff_pls` [d, h] as `_compute_pls` returns them, that is `abs.(W*)`:

    θ0_k = Σ_{l=1}^h θ_l * coeff_pls[k, l]^2
"""
function _expand_kpls_theta(theta_pls, coeff_pls)
    return vec((coeff_pls .^ 2) * theta_pls)
end

"""
    KPLSK(x_vec, y_vec, n_comp, lb, ub, theta) -> KPLSK

Construct a KPLSK surrogate.

KPLSK first obtains a reduced-dimensional KPLS fit and uses its learned
correlation scales to initialize a full-dimensional anisotropic Kriging fit.
This combines the initialization advantages of KPLS with the predictive model
of full-dimensional Kriging.

# Fields

  - `x`: training points in the caller-provided representation.
  - `y`: training responses.
  - `x_matrix`: matrix representation of the training points.
  - `y_matrix`: column-matrix representation of the responses.
  - `xl`: lower and upper bounds stored as a two-column matrix.
  - `n_comp`: number of PLS components used to initialize the fit.
  - `beta`: generalized-least-squares trend coefficients.
  - `gamma`: Kriging residual coefficients.
  - `theta`: optimized full-dimensional correlation scales.
  - `theta_pls`: optimized reduced-dimensional correlation scales.
  - `reduced_likelihood_function_value`: fitted reduced likelihood value.
  - `X_offset`: input centering values.
  - `X_scale`: input scaling values.
  - `X_after_std`: standardized training inputs.
  - `y_mean`: response centering value.
  - `y_std`: response scaling value.

# Arguments

  - `x_vec`: vector of tuples containing the training points.
  - `y_vec`: vector of responses, with one value for each point in `x_vec`.
  - `n_comp::Integer`: number of PLS components used to initialize the fit. It
    must not exceed the input dimension.
  - `lb`: lower bounds for the input coordinates.
  - `ub`: upper bounds for the input coordinates, matching `lb`.
  - `theta`: positive initial correlation scales for the intermediate KPLS fit.

# Returns

A callable `KPLSK` surrogate supporting the generic call and `update!` interface.

# Example

```julia
using Surrogates

objective(x) = sum(abs2, x)
lb, ub = [-1.0, -1.0], [1.0, 1.0]
x = sample(12, lb, ub, SobolSample())
y = objective.(x)
surrogate = KPLSK(x, y, 1, lb, ub, [1.0])
surrogate((0.2, -0.1))
```
"""
function KPLSK(x_vec, y_vec, n_comp, lb, ub, theta)
    xlimits = hcat(collect(Float64, lb), collect(Float64, ub))
    X = vector_of_tuples_to_matrix(x_vec)
    y = reshape(collect(Float64, y_vec), (size(X, 1), 1))
    _check_pls_components("KPLSK", n_comp, size(X, 2), theta)
    _check_pls_bounds("KPLSK", X, xlimits)

    # Stage 1: KPLS, to get a reduced-dimension theta and the PLS rotation coefficients.
    pls_mean, X_after_PLS, y_after_PLS = _compute_pls(X, y, n_comp)
    X_after_std, y_after_std, X_offset, y_mean, X_scale, y_std = standardization(
        copy(X_after_PLS), copy(y_after_PLS)
    )
    D, ij = cross_distances(X_after_std)
    d_pls = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean)
    nt = size(X_after_PLS, 1)
    theta_pls = _optimize_theta(theta, "squar_exp", d_pls, nt, ij, y_after_std)

    # Stage 2: expand theta into full dimension d and refine it locally by
    # maximizing the reduced log-likelihood of the full-dimensional (non-PLS) kernel.
    theta0 = _expand_kpls_theta(theta_pls, pls_mean)
    d_full = D .^ 2
    theta_opt = _optimize_theta(
        theta0, "squar_exp", d_full, nt, ij, y_after_std; multistart = false
    )

    beta, gamma, reduced_likelihood_function_value = _reduced_likelihood_function(
        theta_opt, "squar_exp", d_full, nt, ij, y_after_std
    )

    return KPLSK(
        x_vec, y_vec, X, y, xlimits, n_comp, beta, gamma, theta_opt, theta_pls,
        reduced_likelihood_function_value,
        X_offset, X_scale, X_after_std, y_mean, y_std
    )
end

"""
    (k::KPLSK)(x_vec)

Predict the output at input point `x_vec` (a tuple or vector).
"""
function (k::KPLSK)(x_vec)
    _check_dimension(k, x_vec)
    X_test = prep_data_for_pred(_single_query_point("KPLSK", x_vec))
    n_eval = size(X_test, 1)
    X_cont = (X_test .- k.X_offset) ./ k.X_scale
    dx = differences(X_cont, k.X_after_std)
    pred_d = dx .^ 2
    nt = size(k.X_after_std, 1)
    r = transpose(reshape(squar_exp(k.theta, pred_d), (nt, n_eval)))
    f = ones(n_eval, 1)
    y_ = (f * k.beta) + (r * k.gamma)
    y = k.y_mean .+ k.y_std * y_
    return y[1]
end

"""
    (k::KPLSK)(val::Number)

Predict at a scalar input (1D case).
"""
function (k::KPLSK)(val::Number)
    _check_dimension(k, val)
    return k((val,))
end

"""
    update!(k::KPLSK, new_x, new_y)

Add a new sample point and re-train the KPLSK model.
"""
function SurrogatesBase.update!(k::KPLSK, new_x, new_y)
    new_x_mat = prep_data_for_pred([new_x])
    # A duplicate is a no-op, not an error; see `Kriging`'s `update!`.
    if vec(new_x_mat) in eachrow(k.x_matrix)
        @warn "Skipping `update!`: this sample already exists in the KPLSK " *
            "surrogate, and duplicate points would make the correlation matrix singular."
        return nothing
    end

    if bounds_error(new_x_mat, k.xl)
        throw(ArgumentError("The new sample lies outside [lb, ub]; cannot update KPLSK."))
    end

    # `vcat` rather than `push!`; see `KPLS`'s `update!`.
    k.x = vcat(k.x, [new_x])
    k.y = vcat(k.y, new_y)
    k.x_matrix = vcat(k.x_matrix, new_x_mat)
    k.y_matrix = vcat(k.y_matrix, reshape([Float64(new_y)], (1, 1)))

    pls_mean, X_after_PLS, y_after_PLS = _compute_pls(k.x_matrix, k.y_matrix, k.n_comp)
    k.X_after_std, y_after_std, k.X_offset, k.y_mean, k.X_scale, k.y_std = standardization(
        copy(X_after_PLS), copy(y_after_PLS)
    )
    D, ij = cross_distances(k.X_after_std)
    d_pls = componentwise_distance_PLS(D, "squar_exp", k.n_comp, pls_mean)
    nt = size(X_after_PLS, 1)
    k.theta_pls = _optimize_theta(k.theta_pls, "squar_exp", d_pls, nt, ij, y_after_std)

    theta0 = _expand_kpls_theta(k.theta_pls, pls_mean)
    d_full = D .^ 2
    k.theta = _optimize_theta(
        theta0, "squar_exp", d_full, nt, ij, y_after_std; multistart = false
    )
    k.beta, k.gamma, k.reduced_likelihood_function_value = _reduced_likelihood_function(
        k.theta, "squar_exp", d_full, nt, ij, y_after_std
    )
    return nothing
end
