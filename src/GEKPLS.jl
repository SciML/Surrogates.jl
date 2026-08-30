"""
    GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, theta;
           nugget = 10.0 * eps(), noise = 0.0)
    GEKPLS(x, y, grads, lb, ub; n_comp = 2, delta_x = 1.0e-4, extra_points = 2,
           theta = fill(1.0e-2, n_comp), nugget = 10.0 * eps(), noise = 0.0)

Fit a gradient-enhanced Kriging model after projecting the input dimensions
with partial least squares (PLS). This model is intended for problems where
function values and input gradients are available at every training point.

# Fields

  - `x`: training points in their caller-provided representation.
  - `y`: training responses.
  - `x_matrix`: matrix form of the training points.
  - `y_matrix`: column-matrix form of the responses.
  - `grads`: gradient matrix associated with `x_matrix`.
  - `xl`: two-column matrix of lower and upper bounds.
  - `delta`: Taylor-expansion step used to create gradient-enhanced points.
  - `extra_points`: number of additional Taylor points per training point.
  - `num_components`: number of retained PLS components.
  - `beta`: generalized least-squares trend coefficients.
  - `gamma`: correlation residual coefficients.
  - `theta`: correlation scales in the reduced PLS space.
  - `reduced_likelihood_function_value`: fitted reduced likelihood value.
  - `X_offset`: input centering values.
  - `X_scale`: input scaling values.
  - `X_after_std`: standardized gradient-enhanced training matrix.
  - `pls_mean`: mean PLS projection matrix.
  - `y_mean`: response centering value.
  - `y_std`: response scaling value.
  - `nugget`: starting jitter added to the correlation diagonal.
  - `noise`: observation-noise term added alongside the nugget.

# Arguments

  - `x`: vector of training points.
  - `y`: scalar response at each point in `x`.
  - `grads`: gradient at each point in `x`, in the same input-coordinate order.
  - `n_comp::Integer`: number of PLS components to retain.
  - `delta_x`: first-order Taylor-expansion step.
  - `lb`: lower bound for each input coordinate.
  - `ub`: upper bound for each input coordinate.
  - `extra_points::Integer`: number of gradient-enhanced points used by PLS.
  - `theta`: correlation scales, one per PLS component. Used as given unless
    `optimize_theta` is set, in which case it is the starting point of the fit.
    The value matters a great deal — on the welded-beam benchmark `theta = 1`
    predicts about thirteen times more accurately than the conventional `0.01`.

# Keywords

  - `nugget`: starting jitter added to the correlation diagonal. It is raised by
    factors of ten only as far as the Cholesky factorization requires, so a
    well-conditioned problem pays the smallest jitter that works. Oversizing it
    is not free: the fixed `1e6 * eps()` this replaced costs more than a factor
    of two in RMSE on the welded-beam problems.
  - `noise`: observation-noise term added alongside the nugget; increasing it
    lowers the reduced likelihood value.
  - `optimize_theta`: whether to fit `theta` by maximizing the reduced
    likelihood. **Defaults to `false`**, unlike [`Kriging`](@ref) and
    [`GEK`](@ref): every evaluation factorizes an `nt × nt` matrix with
    `nt = n(1 + extra_points)`, so the search is far more expensive here than the
    fit itself. Worth setting on a modest design — on welded beam it lowered
    RMSE by a factor of 2.75 — and worth avoiding on a large one.
  - `n_start`: Latin-hypercube starts for that search.

# Returns

A callable `GEKPLS`. Calling it with one point returns a scalar prediction.
Training points outside `[lb, ub]` are rejected with an `ArgumentError`.

# Example

```julia
using Surrogates, Zygote

f(x) = sum(abs2, x)
lb = [-1.0, -1.0]
ub = [1.0, 1.0]
x = sample(20, lb, ub, SobolSample())
y = f.(x)
grads = Zygote.gradient.(f, x)
surrogate = GEKPLS(x, y, grads, 1, 1.0e-4, lb, ub, 1, [0.01])
surrogate((0.25, 0.5))
```
"""
mutable struct GEKPLS{T, X, Y} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    x_matrix::Matrix{T}
    y_matrix::Matrix{T}
    grads::Matrix{T}
    xl::Matrix{T}
    delta::T
    extra_points::Int
    num_components::Int
    beta::Vector{T}
    gamma::Matrix{T}
    theta::Vector{T}
    reduced_likelihood_function_value::T
    X_offset::Matrix{T}
    X_scale::Matrix{T}
    X_after_std::Matrix{T}
    pls_mean::Matrix{T}
    y_mean::T
    y_std::T
    nugget::T
    noise::T
    # Whether `theta` is fitted by maximizing the reduced likelihood, in which
    # case `update!` refits it against the extended sample set.
    optimize_theta::Bool
end

function bounds_error(x, xl)
    num_x_rows = size(x, 1)
    num_dim = size(xl, 1)
    for i in 1:num_x_rows
        for j in 1:num_dim
            if (x[i, j] < xl[j, 1] || x[i, j] > xl[j, 2])
                return true
            end
        end
    end
    return false
end

# Project the inputs with PLS, standardize, and solve for the Kriging
# coefficients. The constructor and `update!` need exactly this sequence and
# differ only in where the inputs come from and where the results go.
function _gekpls_fit(
        X, y, grads, n_comp, delta_x, xlimits, extra_points, theta, nugget, noise;
        optimize_theta = false, n_start::Integer = 10, multistart = true
    )
    pls_mean, X_after_PLS, y_after_PLS = _ge_compute_pls(
        X, y, n_comp, grads, delta_x, xlimits, extra_points
    )
    X_after_std, y_after_std, X_offset, y_mean, X_scale, y_std = standardization(
        X_after_PLS, y_after_PLS
    )
    D, ij = cross_distances(X_after_std)
    pls_mean_reshaped = reshape(pls_mean, (size(X, 2), n_comp))
    d = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean_reshaped)
    nt = size(X_after_PLS, 1)
    if optimize_theta
        theta = _optimize_theta(
            theta, "squar_exp", d, nt, ij, y_after_std;
            multistart = multistart, n_start = n_start,
            nugget = nugget, noise = noise, max_escalations = 8
        )
    end
    beta, gamma, rlf = _reduced_likelihood_function(
        theta, "squar_exp", d, nt, ij, y_after_std;
        nugget = nugget, noise = noise, max_escalations = 8
    )
    return (;
        beta, gamma, theta, reduced_likelihood_function_value = rlf, X_offset,
        X_scale, X_after_std, pls_mean = pls_mean_reshaped, y_mean, y_std,
    )
end

"""
    GEKPLS(x, y, grads, lb, ub; n_comp = 2, delta_x = 1.0e-4, extra_points = 2,
           theta = fill(1.0e-2, n_comp), nugget = 10.0 * eps(), noise = 0.0)

Keyword front-end matching the shape used by every other surrogate in the
package, `(x, y, lb, ub; kwargs...)`. Equivalent to the positional form but with
defaults for the tuning parameters. See [`GEKPLS`](@ref) for the meaning of each
argument.
"""
function GEKPLS(
        x_vec, y_vec, grads_vec, lb, ub; n_comp::Integer = 2, delta_x = 1.0e-4,
        extra_points::Integer = 2, theta = fill(1.0e-2, n_comp),
        nugget = _GEKPLS_NUGGET, noise = 0.0, optimize_theta = false,
        n_start::Integer = 10
    )
    return GEKPLS(
        x_vec, y_vec, grads_vec, n_comp, delta_x, lb, ub, extra_points, theta;
        nugget = nugget, noise = noise, optimize_theta = optimize_theta,
        n_start = n_start
    )
end

function GEKPLS(
        x_vec, y_vec, grads_vec, n_comp, delta_x, lb, ub, extra_points, theta;
        nugget = _GEKPLS_NUGGET, noise = 0.0, optimize_theta = false,
        n_start::Integer = 10
    )
    xlimits = hcat(lb, ub)
    X = vector_of_tuples_to_matrix(x_vec)
    y = reshape(y_vec, (size(X, 1), 1))
    grads = vector_of_tuples_to_matrix2(grads_vec)

    #ensure that X values are within the upper and lower bounds
    if bounds_error(X, xlimits)
        throw(
            ArgumentError(
                "Some training points lie outside [lb, ub]; cannot build GEKPLS."
            )
        )
    end

    fit = _gekpls_fit(
        X, y, grads, n_comp, delta_x, xlimits, extra_points, theta, nugget, noise;
        optimize_theta = optimize_theta, n_start = n_start
    )
    return GEKPLS(
        x_vec, y_vec, X, y, grads, xlimits, delta_x, extra_points, n_comp,
        fit.beta, fit.gamma, fit.theta, fit.reduced_likelihood_function_value,
        fit.X_offset, fit.X_scale, fit.X_after_std, fit.pls_mean,
        fit.y_mean, fit.y_std, nugget, noise, optimize_theta
    )
end

function _gekpls_predict(g::GEKPLS, pts)
    X_test = prep_data_for_pred(pts)
    n_eval = size(X_test, 1)
    X_cont = (X_test .- g.X_offset) ./ g.X_scale
    dx = differences(X_cont, g.X_after_std)
    pred_d = componentwise_distance_PLS(dx, "squar_exp", g.num_components, g.pls_mean)
    nt = size(g.X_after_std, 1)
    r = transpose(reshape(squar_exp(g.theta, pred_d), (nt, n_eval)))
    f = ones(n_eval, 1)
    y_ = (f * g.beta) + (r * g.gamma)
    y = g.y_mean .+ g.y_std * y_
    return y[1]
end

function (g::GEKPLS)(x_vec::Number)
    _check_dimension(g, x_vec)
    # A scalar point is wrapped as a one-tuple; the prediction path is shared.
    return _gekpls_predict(g, [(x_vec,)])
end

function (g::GEKPLS)(x_vec)
    _check_dimension(g, x_vec)
    return _gekpls_predict(g, x_vec)
end

"""
    update!(surrogate::GEKPLS, x_new, y_new, grad_new)

Add one observation and its gradient to a fitted [`GEKPLS`](@ref), then refit
the PLS projection and Kriging coefficients in place.

# Arguments

  - `surrogate`: model to update.
  - `x_new`: new input point in the same representation used for training.
  - `y_new`: scalar response at `x_new`.
  - `grad_new`: gradient at `x_new`, ordered like its input coordinates.

# Returns

Returns `nothing`. A duplicate `x_new` warns and leaves the model unchanged; an
`x_new` outside the model bounds throws an `ArgumentError`.
"""
function SurrogatesBase.update!(g::GEKPLS, x_tup, y_val, grad_tup)
    new_x = prep_data_for_pred(x_tup)
    new_grads = prep_data_for_pred(grad_tup)
    # See `Kriging.update!`: a duplicate is a no-op here, not an error.
    if vec(new_x) in eachrow(g.x_matrix)
        @warn "Skipping `update!`: this sample already exists in the GEKPLS " *
            "surrogate, and duplicate points would make the correlation matrix singular."
        return nothing
    end

    if bounds_error(new_x, g.xl)
        throw(ArgumentError("The new sample lies outside [lb, ub]; cannot update GEKPLS."))
    end

    # `vcat` rather than `push!`: the containers are the caller's own, and
    # growing them behind their back is what the `copy` below used to paper over.
    g.x = vcat(g.x, [x_tup])
    g.y = vcat(g.y, y_val)
    g.x_matrix = vcat(g.x_matrix, new_x)
    g.y_matrix = vcat(g.y_matrix, y_val)
    g.grads = vcat(g.grads, new_grads)

    fit = _gekpls_fit(
        g.x_matrix, g.y_matrix, g.grads, g.num_components, g.delta, g.xl,
        g.extra_points, g.theta, g.nugget, g.noise;
        optimize_theta = g.optimize_theta, multistart = false
    )
    g.theta = fit.theta
    g.beta = fit.beta
    g.gamma = fit.gamma
    g.reduced_likelihood_function_value = fit.reduced_likelihood_function_value
    g.X_offset = fit.X_offset
    g.X_scale = fit.X_scale
    g.X_after_std = fit.X_after_std
    g.pls_mean = fit.pls_mean
    g.y_mean = fit.y_mean
    g.y_std = fit.y_std
    return nothing
end

"""
    _ge_compute_pls(X, y, n_comp, grads, delta_x, xlimits, extra_points)

## Gradient-enhanced PLS-coefficients.

Parameters

  - X: [n_obs,dim] - The input variables.
  - y: [n_obs,ny] - The output variable
  - n_comp: int - Number of principal components used.
  - gradients: - The gradient values. Matrix size (n_obs,dim)
  - delta_x: real - The step used in the First Order Taylor Approximation
  - xlimits: [dim, 2]- The upper and lower var bounds.
  - extra_points: int - The number of extra points per each training point.
    Returns

* * *

  - Coeff_pls: [dim, n_comp] - The PLS-coefficients.
  - X: Concatenation of XX: [extra_points*nt, dim] - Extra points added (when extra_points > 0) and X
  - y: Concatenation of yy[extra_points*nt, 1]- Extra points added (when extra_points > 0) and y
"""
function _ge_compute_pls(X, y, n_comp, grads, delta_x, xlimits, extra_points)

    # this function is equivalent to a combination of
    # https://github.com/SMTorg/smt/blob/f124c01ffa78c04b80221dded278a20123dac742/smt/utils/kriging_utils.py#L1036
    # and https://github.com/SMTorg/smt/blob/f124c01ffa78c04b80221dded278a20123dac742/smt/surrogate_models/gekpls.py#L48

    nt, dim = size(X)
    XX = zeros(0, dim)
    yy = zeros(0, size(y)[2])
    coeff_pls = zeros((dim, n_comp, nt))

    for i in 1:nt
        if dim >= 3
            bb_vals = circshift(boxbehnken(dim, 1), 1)
        elseif dim == 2
            bb_vals = [
                0.0 0.0; #center
                1.0 0.0; #right
                0.0 1.0; #up
                -1.0 0.0; #left
                0.0 -1.0; #down
                1.0 1.0; #right up
                -1.0 1.0; #left up
                -1.0 -1.0; #left down
                1.0 -1.0
            ]
        else # dim == 1
            bb_vals = [
                0.0; #center
                1.0; #right
                -1.0
            ] #left
        end
        _X = zeros((size(bb_vals)[1], dim))
        _y = zeros((size(bb_vals)[1], 1))
        bb_vals = bb_vals .* (delta_x * (xlimits[:, 2] - xlimits[:, 1]))' #smt calls this sign. I've called it bb_vals
        _X = X[i, :]' .+ bb_vals
        bb_vals = bb_vals .* grads[i, :]'
        _y = y[i, :] .+ sum(bb_vals, dims = 2)

        #_pls.fit(_X, _y) # relic from sklearn version; retained for future reference.
        #coeff_pls[:, :, i] = _pls.x_rotations_ #relic from sklearn version; retained for future reference.

        coeff_pls[:, :, i] = _modified_pls(_X, _y, n_comp) #_modified_pls returns the equivalent of SKLearn's _pls.x_rotations_
        if extra_points != 0
            start_index = max(1, length(coeff_pls[:, 1, i]) - extra_points + 1)
            max_coeff = sortperm(broadcast(abs, coeff_pls[:, 1, i]))[start_index:end]
            for ii in max_coeff
                XX = [XX; transpose(X[i, :])]
                XX[end, ii] += delta_x * (xlimits[ii, 2] - xlimits[ii, 1])
                yy = [yy; y[i]]
                yy[end] += grads[i, ii] * delta_x * (xlimits[ii, 2] - xlimits[ii, 1])
            end
        end
    end
    if extra_points != 0
        X = [X; XX]
        y = [y; yy]
    end

    pls_mean = mean(broadcast(abs, coeff_pls), dims = 3)
    return pls_mean, X, y
end

######start of bbdesign######

#
# Adapted from 'ExperimentalDesign.jl: Design of Experiments in Julia'
# https://github.com/phrb/ExperimentalDesign.jl

# MIT License

# ExperimentalDesign.jl: Design of Experiments in Julia
# Copyright (C) 2019 Pedro Bruel <pedro.bruel@gmail.com>

# Permission is hereby granted, free of charge,  to any person obtaining a copy of
# this software  and associated documentation  files (the "Software"), to  deal in
# the Software  without restriction,  including without  limitation the  rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to  whom the Software is furnished to do so,
# subject to the following conditions:

# The  above copyright  notice  and  this permission  notice  (including the  next
# paragraph)  shall be  included  in all  copies or  substantial  portions of  the
# Software.

# THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,  EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR  PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE  LIABLE FOR ANY CLAIM, DAMAGES OR  OTHER LIABILITY, WHETHER
# IN  AN ACTION  OF  CONTRACT, TORT  OR  OTHERWISE,  ARISING FROM,  OUT  OF OR  IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

function boxbehnken(matrix_size::Int)
    return boxbehnken(matrix_size, matrix_size)
end

function boxbehnken(matrix_size::Int, center::Int)
    @assert matrix_size >= 3

    A_fact = explicit_fullfactorial(Tuple([-1, 1] for i in 1:2))

    rows = floor(Int, (0.5 * matrix_size * (matrix_size - 1)) * size(A_fact)[1])

    A = zeros(rows, matrix_size)

    l = 0
    for i in 1:(matrix_size - 1)
        for j in (i + 1):matrix_size
            l = l + 1
            A[(max(0, (l - 1) * size(A_fact)[1]) + 1):(l * size(A_fact)[1]), i] = A_fact[
                :,
                1,
            ]
            A[(max(0, (l - 1) * size(A_fact)[1]) + 1):(l * size(A_fact)[1]), j] = A_fact[
                :,
                2,
            ]
        end
    end

    if center == matrix_size
        if matrix_size <= 16
            points = [0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
            center = points[matrix_size]
        end
    end

    return A = transpose(hcat(transpose(A), transpose(zeros(center, matrix_size))))
end

function explicit_fullfactorial(factors::Tuple)
    return explicit_fullfactorial(fullfactorial(factors))
end

function explicit_fullfactorial(iterator::Base.Iterators.ProductIterator)
    return hcat(vcat.(collect(iterator)...)...)
end

function fullfactorial(factors::Tuple)
    return Base.Iterators.product(factors...)
end

######end of bb design######

"""
We subtract the mean from each variable. Then, we divide the values of each
variable by its standard deviation.

## Parameters

X - The input variables.
y - The output variable.

## Returns

X: [n_obs, dim]
The standardized input matrix.

y: [n_obs, 1]
The standardized output vector.

X_offset: The mean (or the min if scale_X_to_unit=True) of each input variable.

y_mean: The mean of the output variable.

X_scale:  The standard deviation of each input variable.

y_std: The standard deviation of the output variable.
"""
function standardization(X, y)
    #Equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L21
    X_offset = mean(X, dims = 1)
    X_scale = std(X, dims = 1)
    X_scale = map(x -> (x == 0.0) ? x = 1 : x = x, X_scale) #to prevent division by 0 below
    y_mean = mean(y)
    y_std = std(y)
    y_std = map(y -> (y == 0) ? y = 1 : y = y, y_std) #to prevent division by 0 below
    X = (X .- X_offset) ./ X_scale
    y = (y .- y_mean) ./ y_std
    return X, y, X_offset, y_mean, X_scale, y_std
end

"""
Computes the nonzero componentwise cross-distances between the vectors
in X

## Parameters

X: [n_obs, dim]

## Returns

D:  [n_obs * (n_obs - 1) / 2, dim]

  - The cross-distances between the vectors in X.

ij: [n_obs * (n_obs - 1) / 2, 2]

  - The indices i and j of the vectors in X associated to the cross-
    distances in D.
"""
function cross_distances(X)
    # equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L86
    n_samples, n_features = size(X)
    n_nonzero_cross_dist = (n_samples * (n_samples - 1)) ÷ 2
    ij = zeros((n_nonzero_cross_dist, 2))
    D = zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in 1:(n_samples - 1)
        ll_0 = ll_1 + 1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 1] .= k
        ij[ll_0:ll_1, 2] = (k + 1):1:n_samples
        D[ll_0:ll_1, :] = -(X[(k + 1):n_samples, :] .- X[k, :]')
    end
    return D, Int.(ij)
end

"""
        Computes the nonzero componentwise cross-spatial-correlation-distance
        between the vectors in X.

        Equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L1257
        with some simplifications (removed theta and return_derivative as it's not required for GEKPLS)

        Parameters
        ----------

        D: [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.

        corr: str
                - Name of the correlation function used.
                squar_exp or abs_exp.

        n_comp: int
                - Number of principal components used.

        coeff_pls: [dim, n_comp]
                - The PLS-coefficients.

        Returns
        -------

        D_corr: [n_obs * (n_obs - 1) / 2, n_comp]
                - The componentwise cross-spatial-correlation-distance between the
                vectors in X.
"""
function componentwise_distance_PLS(D, corr, n_comp, coeff_pls)

    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L1257
    #todo
    #figure out how to handle this computation in the case of very large matrices
    #similar to what SMT has done
    #at https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L1257
    D_corr = zeros((size(D)[1], n_comp))

    if corr == "squar_exp"
        D_corr = D .^ 2 * coeff_pls .^ 2
    else #abs_exp
        D_corr = abs.(D) * abs.(coeff_pls)
    end

    return D_corr
end

"""
## Squared exponential correlation model.

Equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L604
Parameters:

theta : Hyperparameters of the correlation model
d: componentwise distances from componentwise_distance_PLS

## Returns:

r:  array containing the values of the autocorrelation model
"""
function squar_exp(theta, d)
    n_components = size(d)[2]
    theta = reshape(theta, (1, n_components))
    return exp.(-sum(theta .* d, dims = 2))
end

"""
    differences(X, Y)

return differences between two arrays

given an input like this:

X = [1.0 1.0 1.0; 2.0 2.0 2.0]
Y = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
diff = differences(X,Y)

We get an output (diff) that looks like this:

[ 0. -1. -2.
-3. -4. -5.
-6. -7. -8.

 1.  0. -1.
        -2. -3. -4.
        -5. -6. -7.]
"""
function differences(X, Y)
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L392
    #code credit: Elias Carvalho - https://stackoverflow.com/questions/72392010/row-wise-operations-between-matrices-in-julia
    Rx = repeat(X, inner = (size(Y, 1), 1))
    Ry = repeat(Y, size(X, 1))
    return Rx - Ry
end

"""
    _reduced_likelihood_function(theta, kernel_type, d, nt, ij, y_norma;
                                 nugget = 1.0e6 * eps(), noise = 0.0,
                                 max_escalations = 0)

Compute the reduced likelihood function value and other coefficients necessary for prediction
This function is a loose translation of SMT code from
https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/surrogate_models/krg_based.py#L247
It  determines the BLUP parameters and evaluates the reduced likelihood function for the given theta.

## Parameters

theta: array containing the parameters at which the Gaussian Process model parameters should be determined.
kernel_type: name of the correlation function.
d: The componentwise cross-spatial-correlation-distance between the vectors in X.
nt: number of training points
ij: The indices i and j of the vectors in X associated to the cross-distances in D.
y_norma: Standardized y values
noise: noise hyperparameter - increasing noise reduces reduced_likelihood_function_value

## Returns

reduced_likelihood_function_value: real

  - The value of the reduced likelihood function associated with the given autocorrelation parameters theta.
    beta:  Generalized least-squares regression weights
    gamma: Gaussian Process weights.
"""
# The jitter this function has always used. `KPLS` and `KPLSK` keep it, and keep
# `max_escalations = 0`, so their fits and their likelihood optimization are
# unchanged; `GEKPLS` starts far lower and escalates.
const _PLS_NUGGET = 1.0e6 * eps()
const _GEKPLS_NUGGET = 10.0 * eps()

# Assemble the correlation matrix and factorize it, raising the jitter by
# factors of ten only as far as the factorization requires. Oversizing it is not
# free: on the welded-beam benchmark the fixed `1e6 * eps()` costs more than a
# factor of two in RMSE against the smallest jitter that still factorizes.
function _correlation_cholesky(r, nt, ij, nugget, noise, max_escalations)
    R = Matrix{eltype(r)}(I, nt, nt)
    for k in 1:size(ij, 1)
        R[ij[k, 1], ij[k, 2]] = r[k]
        R[ij[k, 2], ij[k, 1]] = r[k]
    end
    for _ in 1:max_escalations
        F = cholesky(Symmetric(R + (nugget + noise) * I), check = false)
        issuccess(F) && return F.L
        nugget *= 10
    end
    # The last attempt is checked, so a matrix no jitter can save still raises
    # the `PosDefException` that callers like `_optimize_theta` handle.
    return cholesky(Symmetric(R + (nugget + noise) * I)).L
end

function _reduced_likelihood_function(
        theta, kernel_type, d, nt, ij, y_norma;
        nugget = _PLS_NUGGET, noise = 0.0, max_escalations = 0
    )
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/surrogate_models/krg_based.py#L247
    if kernel_type == "squar_exp" #todo - add other kernel type abs_exp etc.
        r = squar_exp(theta, d)
    else
        throw(ArgumentError("unsupported kernel_type $(kernel_type); only \"squar_exp\" is implemented"))
    end

    C = _correlation_cholesky(r, nt, ij, nugget, noise, max_escalations)
    F = ones(nt, 1) #todo - examine if this should be a parameter for this function
    Ft = C \ F
    Q, G = qr(Ft)
    Q = Array(Q)
    Yt = C \ y_norma
    #todo - in smt, they check if the matrix is ill-conditioned using SVD. Verify and include if necessary
    beta = G \ [(transpose(Q) ⋅ Yt)]
    rho = Yt .- (Ft .* beta)
    gamma = transpose(C) \ rho
    sigma2 = sum((rho) .^ 2, dims = 1) / nt
    detR = prod(diag(C) .^ (2.0 / nt))
    reduced_likelihood_function_value = -nt * log10(sum(sigma2)) - nt * log10(detR)
    return beta, gamma, reduced_likelihood_function_value
end

### MODIFIED PLS BELOW ###

# The code below is a simplified version of
# SKLearn's PLS
# https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/cross_decomposition/_pls.py
# It is completely self-contained (no dependencies)

function _center_scale(X, Y)
    x_mean = mean(X, dims = 1)
    X .-= x_mean
    y_mean = mean(Y, dims = 1)
    Y .-= y_mean
    x_std = std(X, dims = 1)
    x_std[x_std .== 0] .= 1.0
    X ./= x_std
    y_std = std(Y, dims = 1)
    y_std[y_std .== 0] .= 1.0
    Y ./= y_std
    return X, Y
end

function _svd_flip_1d(u, v)
    # equivalent of https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/cross_decomposition/_pls.py#L149
    biggest_abs_val_idx = findmax(abs.(vec(u)))[2]
    sign_ = sign(u[biggest_abs_val_idx])
    u .*= sign_
    return v .*= sign_
end

function _get_first_singular_vectors_power_method(X, Y)
    my_eps = eps()
    y_score = vec(Y)
    x_weights = transpose(X)y_score / dot(y_score, y_score)
    x_weights ./= (sqrt(dot(x_weights, x_weights)) + my_eps)
    x_score = X * x_weights
    y_weights = transpose(Y)x_score / dot(x_score, x_score)
    y_score = Y * y_weights / (dot(y_weights, y_weights) + my_eps)
    #Equivalent in intent to https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/cross_decomposition/_pls.py#L66
    if any(isnan.(x_weights)) || any(isnan.(y_weights))
        return false, false
    end
    return x_weights, y_weights
end

function _modified_pls(X, Y, n_components)
    x_weights_ = zeros(size(X, 2), n_components)
    _x_scores = zeros(size(X, 1), n_components)
    x_loadings_ = zeros(size(X, 2), n_components)
    Xk, Yk = _center_scale(X, Y)

    for k in 1:n_components
        x_weights, y_weights = _get_first_singular_vectors_power_method(Xk, Yk)

        if x_weights == false
            break
        end

        _svd_flip_1d(x_weights, y_weights)
        x_scores = Xk * x_weights
        x_loadings = transpose(x_scores)Xk / dot(x_scores, x_scores)
        Xk = Xk - (x_scores * x_loadings)
        y_loadings = transpose(x_scores) * Yk / dot(x_scores, x_scores)
        Yk = Yk - x_scores * y_loadings
        x_weights_[:, k] = x_weights
        _x_scores[:, k] = x_scores
        x_loadings_[:, k] = vec(x_loadings)
    end

    x_rotations_ = x_weights_ * pinv(transpose(x_loadings_)x_weights_)
    return x_rotations_
end

### MODIFIED PLS ABOVE ###

### BELOW ARE HELPER FUNCTIONS TO HELP MODIFY VECTORS INTO ARRAYS

function vector_of_tuples_to_matrix(v)
    num_rows = length(v)
    num_cols = length(first(v))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col] = v[row][col]
        end
    end
    return K
end

function vector_of_tuples_to_matrix2(v)
    #convert gradients into matrix form
    num_rows = length(v)
    num_cols = length(first(first(v)))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col] = v[row][1][col]
        end
    end
    return K
end

function prep_data_for_pred(v)
    l = length(first(v))
    if (l == 1)
        return [tup[k] for k in 1:1, tup in v]
    end
    return [tup[k] for tup in v, k in 1:l]
end
