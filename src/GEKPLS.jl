"""
    GEKPLS

Gradient-Enhanced Kriging with Partial Least Squares surrogate model.

Supports both scalar outputs (`y::Number`) and vector outputs
(`y::AbstractVector{<:Real}` of length `ny`). For vector outputs the model uses
a single set of hyperparameters and PLS rotations shared across all `ny`
output dimensions (PLS2 in the local Taylor-augmented PLS step), and
independent BLUP coefficients per output (`beta`, `gamma`, `sigma2`). See:

  - Bouhlel & Martins, "Gradient-enhanced kriging for high-dimensional
    problems", Engineering with Computers 35 (2019), arXiv:1708.02663.
  - Wold, Sjöström & Eriksson, "PLS-regression: a basic tool of
    chemometrics" (2001), for the NIPALS PLS2 algorithm.

Fields below use the internal canonical shapes; `y_matrix` is `(nt, ny)`,
`grads` is `(nt, dim, ny)` storing `∂y_k/∂x_j`, and `beta`/`gamma`/`y_mean`/
`y_std` carry one column per output dimension.
"""
mutable struct GEKPLS{T, X, Y} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    x_matrix::Matrix{T} #1
    y_matrix::Matrix{T} #2 -- (nt, ny)
    grads::Array{T, 3} #3 -- (nt, dim, ny); grads[i,j,k] = ∂y_k/∂x_j at x^(i)
    xl::Matrix{T} #xlimits #4
    delta::T #5
    extra_points::Int #6
    num_components::Int #7
    beta::Matrix{T} #8 -- (1, ny)
    gamma::Matrix{T} #9 -- (nt_after_PLS, ny)
    theta::Vector{T} #10
    reduced_likelihood_function_value::T #11
    X_offset::Matrix{T} #12
    X_scale::Matrix{T} #13
    X_after_std::Matrix{T} #14 - X after standardization
    pls_mean::Matrix{T} #15
    y_mean::Matrix{T} #16 -- (1, ny)
    y_std::Matrix{T} #17 -- (1, ny)
    scalar_output::Bool #18 -- if true, predict returns a scalar (legacy API)
end

"""
    bounds_error(x, xl)

Checks if input x values are within range of upper and lower bounds
"""
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

"""
    _canonicalize_y_and_grads(y_vec, grads_vec, nt, dim)

Convert the user-facing `(y_vec, grads_vec)` representation into the canonical
internal storage `(y_matrix, grads_3tensor, scalar_output)`.

Scalar output (`eltype(y_vec) <: Number`):
  - `y_matrix`: `(nt, 1)` matrix.
  - `grads_3tensor`: `(nt, dim, 1)` array; populated via
    [`vector_of_tuples_to_matrix2`](@ref) (Zygote-style gradient unpacking).
  - `scalar_output = true`.

Vector output (`eltype(y_vec) <: AbstractVector`):
  - `y_matrix`: `(nt, ny)` matrix, one row per sample point.
  - `grads_3tensor`: `(nt, dim, ny)` array. `grads_vec[i]` must be either a
    `(ny, dim)` Jacobian matrix, or a 1-tuple wrapping one (e.g. as returned
    by `Zygote.jacobian`), or any indexable object with `J[k, j] = ∂y_k/∂x_j`.
  - `scalar_output = false`.
"""
function _canonicalize_y_and_grads(y_vec, grads_vec, nt, dim)
    @assert length(y_vec) == nt "length(y_vec)=$(length(y_vec)) does not match length(x_vec)=$nt"
    @assert length(grads_vec) == nt "length(grads_vec)=$(length(grads_vec)) does not match length(x_vec)=$nt"
    if eltype(y_vec) <: Number
        y_matrix = reshape(collect(float.(y_vec)), nt, 1)
        grads_mat = vector_of_tuples_to_matrix2(grads_vec)
        T = promote_type(eltype(y_matrix), eltype(grads_mat))
        grads_3t = Array{T, 3}(undef, nt, dim, 1)
        @inbounds for j in 1:dim, i in 1:nt
            grads_3t[i, j, 1] = grads_mat[i, j]
        end
        return Matrix{T}(y_matrix), grads_3t, true
    else
        ny = length(first(y_vec))
        T = promote_type(eltype(first(y_vec)), Float64)
        y_matrix = Matrix{T}(undef, nt, ny)
        @inbounds for i in 1:nt
            yi = y_vec[i]
            @assert length(yi) == ny "All entries of y_vec must have the same length (ny=$ny); entry $i has length $(length(yi))"
            for k in 1:ny
                y_matrix[i, k] = yi[k]
            end
        end
        grads_3t = Array{T, 3}(undef, nt, dim, ny)
        @inbounds for i in 1:nt
            J = grads_vec[i]
            if J isa Tuple
                J = J[1]
            end
            @assert size(J, 1) == ny&&size(J, 2) == dim "grads_vec[$i] must be a (ny=$ny, dim=$dim) Jacobian; got size $(size(J))"
            for k in 1:ny, j in 1:dim
                grads_3t[i, j, k] = J[k, j]
            end
        end
        return y_matrix, grads_3t, false
    end
end

"""
    GEKPLS(x_vec, y_vec, grads_vec, n_comp, delta_x, lb, ub, extra_points, theta)

Construct a Gradient-Enhanced Kriging with Partial Least Squares surrogate.

Arguments
=========

  - `x_vec`: vector of tuples giving the sampling locations in `R^dim`.
  - `y_vec`: outputs at each sampling point. Either
    * a `Vector{<:Number}` for scalar-valued targets (legacy API), or
    * a `Vector{<:AbstractVector{<:Number}}` of length `nt`, with each entry a
      length-`ny` vector, for vector-valued targets.
  - `grads_vec`: derivatives at each sampling point. For scalar `y`, the
    Zygote-style representation produced by `Zygote.gradient.(f, x_vec)` is
    accepted. For vector `y`, each entry should be a `(ny, dim)` Jacobian
    matrix (or a 1-tuple wrapping one, as returned by `Zygote.jacobian`).
  - `n_comp`: number of PLS principal components retained.
  - `delta_x`: step size for the first-order Taylor approximation around each
    sample point.
  - `lb`, `ub`: lower / upper bounds on the design space.
  - `extra_points`: number of Taylor-extrapolated approximating points added
    around each sample (in the most-influential PLS directions).
  - `theta`: initial covariance hyperparameters (one per principal component).

The surrogate `g` is then callable as `g(x)`, returning a scalar for the
scalar-output case and a length-`ny` `Vector` for the vector-output case.
"""
function GEKPLS(x_vec, y_vec, grads_vec, n_comp, delta_x, lb, ub, extra_points, theta)
    xlimits = hcat(lb, ub)
    X = vector_of_tuples_to_matrix(x_vec)
    nt, dim = size(X)
    y, grads, scalar_output = _canonicalize_y_and_grads(y_vec, grads_vec, nt, dim)

    #ensure that X values are within the upper and lower bounds
    if bounds_error(X, xlimits)
        println("X values outside bounds")
        return
    end

    pls_mean, X_after_PLS,
        y_after_PLS = _ge_compute_pls(
        X, y, n_comp, grads, delta_x,
        xlimits, extra_points
    )
    X_after_std, y_after_std, X_offset, y_mean,
        X_scale, y_std = standardization(
        X_after_PLS,
        y_after_PLS
    )
    D, ij = cross_distances(X_after_std)
    pls_mean_reshaped = reshape(pls_mean, (size(X, 2), n_comp))
    d = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean_reshaped)
    nt_aug, _ = size(X_after_PLS)
    beta, gamma,
        reduced_likelihood_function_value = _reduced_likelihood_function(
        theta,
        "squar_exp",
        d, nt_aug, ij,
        y_after_std
    )
    return GEKPLS(
        x_vec, y_vec, X, y, grads, xlimits, delta_x, extra_points, n_comp, beta,
        gamma, theta,
        reduced_likelihood_function_value,
        X_offset, X_scale, X_after_std, pls_mean_reshaped, y_mean, y_std,
        scalar_output
    )
end

"""
    (g::GEKPLS)(X_test)

    Take in a set of one or more points and provide predicted approximate outputs (predictor).
"""
function (g::GEKPLS)(x_vec::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(g, x_vec)
    # Convert scalar to tuple format for prep_data_for_pred
    x_vec_tuple = [(x_vec,)]
    return _gekpls_predict(g, x_vec_tuple)
end

function (g::GEKPLS)(x_vec)
    _check_dimension(g, x_vec)
    return _gekpls_predict(g, x_vec)
end

function _gekpls_predict(g::GEKPLS, x_vec)
    X_test = prep_data_for_pred(x_vec)
    n_eval, _ = size(X_test)
    X_cont = (X_test .- g.X_offset) ./ g.X_scale
    dx = differences(X_cont, g.X_after_std)
    pred_d = componentwise_distance_PLS(dx, "squar_exp", g.num_components, g.pls_mean)
    nt = size(g.X_after_std, 1)
    r = transpose(reshape(squar_exp(g.theta, pred_d), (nt, n_eval)))
    f = ones(n_eval, 1)
    y_ = (f * g.beta) + (r * g.gamma)           # (n_eval, ny)
    y = g.y_mean .+ g.y_std .* y_               # (n_eval, ny)
    # Single-point evaluation is the only API contract: return scalar or
    # length-ny vector matching the user-facing `y_vec` shape.
    return g.scalar_output ? y[1] : Vector(view(y, 1, :))
end

"""
    update!(g::GEKPLS, new_x, new_y, new_grads)

add a new point to the dataset.
"""
function SurrogatesBase.update!(g::GEKPLS, x_tup, y_val, grad_val)
    new_x = prep_data_for_pred(x_tup)
    dim = size(g.x_matrix, 2)
    ny = size(g.y_matrix, 2)
    if vec(new_x) in eachrow(g.x_matrix)
        println("Adding a sample that already exists. Cannot build GEKPLS")
        return
    end

    if bounds_error(new_x, g.xl)
        println("x values outside bounds")
        return
    end

    new_y_row, new_grads_3t = if g.scalar_output
        # legacy scalar API: y_val is a Number, grad_val is a tuple of tuples
        new_grads_mat = prep_data_for_pred(grad_val)             # (1, dim)
        (
            reshape([float(y_val)], 1, 1),
            reshape(new_grads_mat, 1, dim, 1),
        )
    else
        # vector API: y_val is a length-ny vector,
        # grad_val is a (ny, dim) Jacobian (or a 1-tuple wrapping one)
        J = grad_val isa Tuple ? grad_val[1] : grad_val
        @assert length(y_val) == ny "y_val must have length ny=$ny; got $(length(y_val))"
        @assert size(J, 1) == ny&&size(J, 2) == dim "grad_val must be (ny=$ny, dim=$dim); got $(size(J))"
        row = reshape(collect(float.(y_val)), 1, ny)
        grads_block = Array{eltype(g.grads), 3}(undef, 1, dim, ny)
        @inbounds for k in 1:ny, j in 1:dim
            grads_block[1, j, k] = J[k, j]
        end
        (row, grads_block)
    end

    temp_y = copy(g.y) #without the copy here, we get error ("cannot resize array with shared data")
    push!(g.x, x_tup)
    push!(temp_y, y_val)
    g.y = temp_y
    g.x_matrix = vcat(g.x_matrix, new_x)
    g.y_matrix = vcat(g.y_matrix, new_y_row)
    g.grads = cat(g.grads, new_grads_3t, dims = 1)
    pls_mean, X_after_PLS,
        y_after_PLS = _ge_compute_pls(
        g.x_matrix, g.y_matrix,
        g.num_components,
        g.grads, g.delta, g.xl,
        g.extra_points
    )
    g.X_after_std, y_after_std, g.X_offset, g.y_mean,
        g.X_scale, g.y_std = standardization(
        X_after_PLS,
        y_after_PLS
    )
    D, ij = cross_distances(g.X_after_std)
    g.pls_mean = reshape(pls_mean, (size(g.x_matrix, 2), g.num_components))
    d = componentwise_distance_PLS(D, "squar_exp", g.num_components, g.pls_mean)
    nt, _ = size(X_after_PLS)
    return g.beta, g.gamma,
        g.reduced_likelihood_function_value = _reduced_likelihood_function(
        g.theta,
        "squar_exp",
        d,
        nt,
        ij,
        y_after_std
    )
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
    #
    # Multivariate-y extension: `y` is `(nt, ny)` and `grads` is `(nt, dim, ny)`
    # with `grads[i,j,k] = ∂y_k/∂x_j` at `x^(i)`. The Taylor augmentation
    # becomes `_y[bb, k] = y[i, k] + Σⱼ bb_dx[bb, j] * grads[i, j, k]`, i.e.
    # `_y = y[i, :]ᵀ .+ bb_dx * grads[i, :, :]`. The PLS step is then PLS2 on
    # the (n_bb, ny) response when ny>1, reducing exactly to PLS1 when ny==1
    # (see `_get_first_singular_vectors_power_method`).

    nt, dim = size(X)
    ny = size(y, 2)
    XX = zeros(0, dim)
    yy = zeros(0, ny)
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
        # bb_dx: scaled perturbation vectors, size (n_bb, dim)
        bb_dx = bb_vals .* (delta_x * (xlimits[:, 2] - xlimits[:, 1]))'
        _X = X[i, :]' .+ bb_dx
        # Multivariate FOTA: directional derivative for every output column.
        # grads[i, :, :] is the (dim, ny) transposed Jacobian at sample i.
        _y = reshape(y[i, :], 1, ny) .+ bb_dx * grads[i, :, :]   # (n_bb, ny)

        coeff_pls[:, :, i] = _modified_pls(_X, _y, n_comp) #_modified_pls returns the equivalent of SKLearn's _pls.x_rotations_
        if extra_points != 0
            start_index = max(1, length(coeff_pls[:, 1, i]) - extra_points + 1)
            max_coeff = sortperm(broadcast(abs, coeff_pls[:, 1, i]))[start_index:end]
            for ii in max_coeff
                dx_ii = delta_x * (xlimits[ii, 2] - xlimits[ii, 1])
                new_X_row = transpose(X[i, :])
                new_X_row = copy(new_X_row)
                new_X_row[1, ii] += dx_ii
                XX = vcat(XX, new_X_row)
                # FOTA per output: y(x + dx_ii e_ii)_k = y(x)_k + grads[i,ii,k]*dx_ii
                new_y_row = reshape(y[i, :] .+ grads[i, ii, :] .* dx_ii, 1, ny)
                yy = vcat(yy, new_y_row)
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
    # Per-output standardization so vector-valued y is normalized columnwise.
    y_mean = mean(y, dims = 1)
    y_std = std(y, dims = 1)
    y_std = map(s -> (s == 0) ? one(s) : s, y_std) #to prevent division by 0 below
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
    _reduced_likelihood_function(theta, kernel_type, d, nt, ij, y_norma, noise = 0.0)

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
function _reduced_likelihood_function(theta, kernel_type, d, nt, ij, y_norma, noise = 0.0)
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/surrogate_models/krg_based.py#L247
    #
    # Multi-output extension: `y_norma` may be `(nt, ny)`. Under shared
    # hyperparameters θ the kriging correlation matrix R, its Cholesky C, and
    # the regression matrices F=1, Ft, Q, G are independent of the output
    # index, so they are computed once. The BLUP coefficients then take one
    # column per output: β is (1, ny), γ is (nt, ny), σ² is (1, ny). The
    # objective preserves SMT's `-nt*log10(sum(σ²)) - nt*log10(detR)` form,
    # where `sum(σ²)` now sums across the ny output dimensions — monotonically
    # consistent with the standard concentrated log-likelihood for shared θ.
    reduced_likelihood_function_value = -Inf
    nugget = 1000000.0 * eps() #a jitter for numerical stability; reducing the multiple from 1000000.0 results in positive definite error for Cholesky decomposition;
    if kernel_type == "squar_exp" #todo - add other kernel type abs_exp etc.
        r = squar_exp(theta, d)
    end
    R = (I + zeros(nt, nt)) .* (1.0 + nugget + noise)

    for k in 1:size(ij)[1]
        R[ij[k, 1], ij[k, 2]] = r[k]
        R[ij[k, 2], ij[k, 1]] = r[k]
    end

    C = cholesky(R).L #todo - #values diverge at this point from SMT code; verify impact
    F = ones(nt, 1) #todo - examine if this should be a parameter for this function
    Ft = C \ F
    Q, G = qr(Ft)
    Q = Array(Q)
    Yt = C \ y_norma                              # (nt, ny)
    #todo - in smt, they check if the matrix is ill-conditioned using SVD. Verify and include if necessary
    beta = G \ (transpose(Q) * Yt)                # (1, ny)
    rho = Yt .- (Ft * beta)                       # (nt, ny)
    gamma = transpose(C) \ rho                    # (nt, ny)
    sigma2 = sum(rho .^ 2, dims = 1) / nt         # (1, ny)
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

function _get_first_singular_vectors_power_method(X, Y; tol = 1.0e-6, max_iter = 500)
    # NIPALS power iteration for the leading PLS components of (X, Y).
    #
    # When Y has a single column (PLS1, scalar response) one iteration is
    # exact, and we reproduce the original implementation bit-for-bit so the
    # scalar surrogate is unchanged.
    # When Y has multiple columns (PLS2, vector response) we iterate the same
    # body to convergence, following Wold's NIPALS algorithm; the initial
    # `y_score` is the column of Y with the largest variance, matching
    # scikit-learn's PLSRegression default.
    my_eps = eps()
    ny = size(Y, 2)
    if ny == 1
        y_score = vec(Y)
    else
        vars = vec(var(Y, dims = 1))
        y_score = Vector(view(Y, :, argmax(vars)))
    end
    x_weights = transpose(X) * y_score / dot(y_score, y_score)
    x_weights ./= (sqrt(dot(x_weights, x_weights)) + my_eps)
    x_score = X * x_weights
    y_weights = transpose(Y) * x_score / dot(x_score, x_score)
    y_score = Y * y_weights / (dot(y_weights, y_weights) + my_eps)
    if ny > 1
        y_score_prev = similar(y_score)
        for _ in 2:max_iter
            y_score_prev .= y_score
            x_weights = transpose(X) * y_score / (dot(y_score, y_score) + my_eps)
            x_weights ./= (sqrt(dot(x_weights, x_weights)) + my_eps)
            x_score = X * x_weights
            y_weights = transpose(Y) * x_score / (dot(x_score, x_score) + my_eps)
            y_score = Y * y_weights / (dot(y_weights, y_weights) + my_eps)
            ref = max(norm(y_score), one(eltype(y_score)))
            if norm(y_score - y_score_prev) < tol * ref
                break
            end
        end
    end
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
