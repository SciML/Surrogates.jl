module SurrogatesMOEExt

import Surrogates
import Surrogates: MOE
using Distributions: MvNormal
using GaussianMixtures: GMM, covars, llpg
using LinearAlgebra: norm

import Distributions
import GaussianMixtures
import SurrogatesBase

"""
    MOE(x, y, expert_types;  ndim=1, n_clusters=2)

constructor for MOE; takes in x, y and expert types and returns an MOE struct
"""
function MOE(x, y, expert_types; ndim = 1, n_clusters = 2, quantile = 10)
    if (ndim > 1)
        X = _vector_of_tuples_to_matrix(x)
        values = hcat(X, y)
    else
        values = hcat(x, y)
    end

    x_and_y_test, x_and_y_train = _extract_part(values, quantile)
    # We get posdef error without jitter; And if values repeat we get NaN vals
    # https://github.com/davidavdav/GaussianMixtures.jl/issues/21
    jitter_vals = ((rand(eltype(x_and_y_train), size(x_and_y_train))) ./ 10000)
    gm_cluster = GMM(
        n_clusters, x_and_y_train + jitter_vals, kind = :full, nInit = 50,
        nIter = 20
    )
    mvn_distributions = _create_clusters_distributions(gm_cluster, ndim, n_clusters)
    cluster_classifier_train = _cluster_predict(gm_cluster, x_and_y_train)
    clusters_train = _cluster_values(x_and_y_train, cluster_classifier_train, n_clusters)
    cluster_classifier_test = _cluster_predict(gm_cluster, x_and_y_test)
    clusters_test = _cluster_values(x_and_y_test, cluster_classifier_test, n_clusters)
    # `Any` deliberately: which expert wins a cluster is not known until the fit
    # runs and may differ on a refit, so the field's element type has to admit
    # any of them. A narrower one makes `update!` fail to assign back.
    best_models = Any[
        _find_best_model(clusters_train[i], clusters_test[i], ndim, expert_types)
            for i in 1:n_clusters
    ]
    return MOE(
        x, y, gm_cluster, mvn_distributions, best_models, expert_types, ndim,
        n_clusters, quantile
    )
end

"""
    (moe::MOE)(val::Number)

predictor for 1D inputs
"""
function (moe::MOE)(val::Number)
    val = [val]
    weights = GaussianMixtures.weights(moe.c)
    rvs = [Distributions.pdf(moe.d[k], val) for k in 1:length(weights)]
    probs = weights .* rvs
    rad = sum(probs)
    if rad > 0
        probs = probs / rad
    end
    max_index = argmax(probs)
    prediction = moe.m[max_index](val[1])
    return prediction
end

"""
    (moe::MOE)(val)

predictor for ndimensional inputs
"""
function (moe::MOE)(val)
    val = collect(val) #to handle inputs that may sometimes be tuples
    weights = GaussianMixtures.weights(moe.c)
    rvs = [Distributions.pdf(moe.d[k], val) for k in 1:length(weights)]
    probs = weights .* rvs
    rad = sum(probs)

    if rad > 0
        probs = probs ./ rad
    end

    max_index = argmax(probs)
    prediction = moe.m[max_index](val)
    return prediction
end

"""
    _cluster_predict(gmm:GMM, X::Matrix)

gmm - a trained Gaussian Mixture Model
X - a matrix of points with dimensions equal to the inputs used for the
training of the model

Return - Clusters to which each of the points belong to (starts at int 1)

Example:
X = [1.0 2; 1 4; 1 0; 10 2; 10 4; 10 0] + rand(Float64, (6, 2))
gm = GMM(2, X)
_cluster_predict(gm,  [0.0 0.0; 12.0 3.0]) #returns [1,2]
"""
function _cluster_predict(gmm::GMM, X::Matrix)
    llpg_X = llpg(gmm, X) #log likelihood probability of X belonging to each of the clusters in the gaussian mixture
    return map(argmax, eachrow(llpg_X))
end

"""
    _extract_part(values, quantile)
    values - a matrix containing all the input values (n test points by d dimensions)
    quantiles - the interval between rows 
    returns a test values matrix and a training values matrix
    Ex: 
    values = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10]
    quantile = 4
    test, train = _extract_part(values, quantile)
    test # [1.0   2.0; 9.0  10.0]
    train #  [3.0  4.0; 5.0  6.0; 7.0  8.0]
"""
function _extract_part(values, quantile)
    num = size(values, 1)
    indices = collect(1:quantile:num)
    mask = falses(num)
    mask[indices] .= true
    return values[mask, :], values[.~mask, :]
end

"""
    _cluster_values(values, cluster_classifier, num_clusters)

values - a concatenation of input and output values
cluster_classifier - a vector of integers representing which cluster each data point belongs to
num_clusters - number of clusters

output
clusters - values grouped by clusters

## Ex:

vals = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
cluster_classifier = [1, 2, 2, 2, 1]
num_clusters = 2
clusters = _cluster_values(vals, cluster_classifier, num_clusters)
@show clusters #prints values below

    [[1.0, 2.0], [9.0, 10.0]]
    [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
"""
function _cluster_values(values, cluster_classifier, num_clusters)
    num = length(cluster_classifier)
    if (size(values, 1) != num)
        error("Number of values don't match number of cluster_classifier points")
    end
    # Typed: an untyped container makes the downstream comprehensions
    # `Vector{Any}`, and `_find_best_model`'s `norm` then needs `zero(Any)`.
    clusters = [Vector{Vector{eltype(values)}}() for _ in 1:num_clusters]
    for i in 1:num
        push!(clusters[cluster_classifier[i]], (values[i, :]))
    end
    return clusters
end

"""
_create_clusters_distributions(gmm::GMM, ndim, n_clusters)
gmm - a gaussian mixture model with concatenated X and y values that have been clustered
ndim - number of dimensions in X
n_clusters - number of clusters

output
distribs - a vector containing frozen multivariate normal distributions for each cluster
"""
function _create_clusters_distributions(gmm::GMM, ndim, n_clusters)
    means = gmm.μ
    cov = covars(gmm)
    # Marginal over the inputs: the mixture is fitted on the joint `(x, y)`
    # design, and prediction weights clusters by the input alone.
    return [
        MvNormal(means[k, 1:ndim], cov[k][1:ndim, 1:ndim]) for k in 1:n_clusters
    ]
end

"""
_find_upper_lower_bounds(m::Matrix)
returns upper and lower bounds in vector form
"""
function _find_upper_lower_bounds(X::Matrix)
    # Column extrema. `push!` into an untyped literal gave `Vector{Any}` bounds,
    # which then reached the component constructors.
    lb = vec(minimum(X, dims = 1))
    ub = vec(maximum(X, dims = 1))
    return size(X, 2) == 1 ? (lb[1], ub[1]) : (lb, ub)
end

"""
_find_best_model(clustered_values, clustered_test_values)
finds best model for each set of clustered values by validating against the clustered_test_values
"""
function _find_best_model(
        clustered_train_values, clustered_test_values, dim,
        enabled_expert_types
    )
    # find upper and lower bounds for clustered_train and test values concatenated

    x_vec = dim == 1 ? [first(a) for a in clustered_train_values] :
        [a[1:dim] for a in clustered_train_values]
    y_vec = [last(a) for a in clustered_train_values]

    x_test_vec = dim == 1 ? [first(a) for a in clustered_test_values] :
        [a[1:dim] for a in clustered_test_values]
    y_test_vec = [last(a) for a in clustered_test_values]

    if (dim == 1)
        xtrain_mat = reshape(x_vec, (size(clustered_train_values, 1), dim))
        xtest_mat = reshape(x_test_vec, (size(clustered_test_values, 1), dim))
    else
        xtrain_mat = _vector_of_tuples_to_matrix(x_vec)
        xtest_mat = _vector_of_tuples_to_matrix(x_test_vec)
    end

    X = !isnothing(xtest_mat) ? vcat(xtrain_mat, xtest_mat) : xtrain_mat
    x_test_vec = !isnothing(xtest_mat) ? x_test_vec : x_vec
    y_test_vec = !isnothing(xtest_mat) ? y_test_vec : y_vec
    lb, ub = _find_upper_lower_bounds(X)

    # call on _surrogate_builder with clustered_train_vals, enabled expert types, lb, ub

    surr_vec = _surrogate_builder(
        enabled_expert_types, length(enabled_expert_types), x_vec,
        y_vec, lb, ub
    )

    # use the models to find best model after validating against test data and return best model
    best_rmse = Inf
    best_model = surr_vec[1] #initial assignment can be any model
    for surr_model in surr_vec
        pred = surr_model.(x_test_vec)
        rmse = norm(pred - y_test_vec, 2) / sqrt(length(y_test_vec))
        if (rmse < best_rmse)
            best_rmse = rmse
            best_model = surr_model
        end
    end
    return best_model
end

"""
    _surrogate_builder(local_kind, k, x, y, lb, ub)

takes in an array of surrogate types, and number of cluster, builds the surrogates and returns
an array of surrogate objects
"""
function _surrogate_builder(local_kind, k, x, y, lb, ub)
    # Dispatch on the descriptor's type; see `src/ComponentSurrogates.jl`.
    return [Surrogates._build_component(local_kind[i], x, y, lb, ub) for i in 1:k]
end

"""
    update!(m::MOE, new_x, new_y)

add a new point to the dataset.
"""
function SurrogatesBase.update!(m::MOE, x, y)
    # `_append_samples`: the caller's containers are left alone, one new point
    # is told from a batch of them, and a point may be written either as a tuple
    # or as a coordinate vector.
    m.x, m.y = Surrogates._append_samples(m.x, m.y, x, y)

    # The split the constructor used, not a fresh one: refitting on a different
    # train/test partition would score the experts against different data.
    quantile = m.q

    if (m.nd > 1) #number of dimensions
        X = _vector_of_tuples_to_matrix(m.x)
        values = hcat(X, m.y)
    else
        values = hcat(m.x, m.y)
    end
    x_and_y_test, x_and_y_train = _extract_part(values, quantile)
    # We get posdef error without jitter; And if values repeat we get NaN vals
    # https://github.com/davidavdav/GaussianMixtures.jl/issues/21
    jitter_vals = ((rand(eltype(x_and_y_train), size(x_and_y_train))) ./ 10000)
    gm_cluster = GMM(
        m.nc, x_and_y_train + jitter_vals, kind = :full, nInit = 50,
        nIter = 20
    )
    mvn_distributions = _create_clusters_distributions(gm_cluster, m.nd, m.nc)
    cluster_classifier_train = _cluster_predict(gm_cluster, x_and_y_train)
    clusters_train = _cluster_values(x_and_y_train, cluster_classifier_train, m.nc)
    cluster_classifier_test = _cluster_predict(gm_cluster, x_and_y_test)
    clusters_test = _cluster_values(x_and_y_test, cluster_classifier_test, m.nc)
    # `Any` for the reason given in the constructor.
    best_models = Any[
        _find_best_model(clusters_train[i], clusters_test[i], m.nd, m.e)
            for i in 1:(m.nc)
    ]
    m.c = gm_cluster
    m.d = mvn_distributions
    return m.m = best_models
end

"""
    _vector_of_tuples_to_matrix(v)

takes in a vector of tuples or vector of vectors and converts it into a matrix
"""
function _vector_of_tuples_to_matrix(v)
    if !isempty(v)
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
    return nothing
end


# ---- SurrogatesBase parameter interface -----------------------------------
#
# The mixture model, its cluster distributions and the selected experts are all
# outputs of the fit; the expert menu and the cluster count are configuration.

SurrogatesBase.parameters(m::MOE) = (;
    cluster_model = m.c, cluster_distributions = m.d, experts = m.m,
)
SurrogatesBase.hyperparameters(m::MOE) = (;
    expert_types = m.e, ndim = m.nd, n_clusters = m.nc, quantile = m.q,
)

end #module
