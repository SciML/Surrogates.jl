module SurrogatesMOE

import Surrogates: AbstractSurrogate, linearRadial, cubicRadial, multiquadricRadial,
                   thinplateRadial, RadialBasisStructure, RadialBasis,
                   InverseDistanceSurrogate, Kriging, LobachevskyStructure,
                   LobachevskySurrogate, NeuralStructure, PolyChaosStructure

export MOE


using GaussianMixtures
using Random
using Distributions
using GaussianMixtures
using LinearAlgebra
using SurrogatesFlux
using SurrogatesPolyChaos
using SurrogatesRandomForest
using XGBoost


mutable struct MOE{X, Y, C, D, M} <: AbstractSurrogate
    x::X
    y::Y
    c::C #clusters (C) - vector of gaussian mixture clusters
    d::D #distributions (D) - vector of frozen multivariate distributions
    m::M # models (M) - vector of trained models correspnoding to clusters (C) and distributions (D)
end

"""
    MOE(x, y, expert_types;  ndim=1, n_clusters=2)
constructor for MOE; takes in x, y and expert types and returns an MOE struct
"""
function MOE(x, y, expert_types;  ndim=1, n_clusters=2)
    quantile = 10
    if(ndim>1)
        x = _vector_of_tuples_to_matrix(x)
    end
    values = hcat(x,y)
    

    x_and_y_test, x_and_y_train = _extract_part(values, quantile)    
    # We get posdef error without jitter; And if values repeat we get NaN vals 
    # https://github.com/davidavdav/GaussianMixtures.jl/issues/21 
    jitter_vals = ((rand(eltype(x_and_y_train), size(x_and_y_train)))./10000)
    gm_cluster = GMM(n_clusters, x_and_y_train+jitter_vals, kind=:full, nInit=50, nIter=20)
    mvn_distributions = _create_clusters_distributions(gm_cluster, ndim, n_clusters) 
    cluster_classifier_train = _cluster_predict(gm_cluster, x_and_y_train)
    clusters_train = _cluster_values(x_and_y_train, cluster_classifier_train, n_clusters)
    cluster_classifier_test = _cluster_predict(gm_cluster, x_and_y_test)
    clusters_test = _cluster_values(x_and_y_test, cluster_classifier_test, n_clusters)
    best_models = []
    for i in 1:n_clusters
        best_model = _find_best_model(clusters_train[i], clusters_test[i], ndim, expert_types)
        push!(best_models, best_model)
    end
    X = values[:, 1:ndim]
    y = values[:, 2] 
    return MOE(X, y, gm_cluster, mvn_distributions, best_models)
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
    val = val
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
    num = size(values,1)
    indices = collect(1:quantile:num)
    mask = BitArray(undef, num)
    mask[indices].= true
    #mask
    return values[mask, :], values[.~mask,:]
end


"""
    _cluster_values(values, cluster_classifier, num_clusters)
values - a concatenation of input and output values
cluster_classifier - a vector of integers representing which cluster each data point belongs to
num_clusters - number of clusters

output
clusters - values grouped by clusters

Ex: 
vals = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
cluster_classifier = [1, 2, 2, 2, 1]
num_clusters = 2
clusters = _cluster_values(vals, cluster_classifier, num_clusters)
@show clusters #prints values below
---
    [[1.0, 2.0], [9.0, 10.0]]
    [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

"""
function _cluster_values(values, cluster_classifier, num_clusters)
    num = length(cluster_classifier)
    if (size(values,1) != num)
        error("Number of values don't match number of cluster_classifier points")
    end
    clusters = [[] for n in 1:num_clusters]
    for i in 1:num
        push!(clusters[cluster_classifier[i]],(values[i, :]))
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
    distribs = []

    for k in 1:n_clusters
        meansk = means[k, 1:ndim]
        covk = cov[k][1:ndim, 1:ndim]
        mvn = MvNormal(meansk, covk) # todo - check if we need allow_singular=True and implement
        push!(distribs, mvn)
    end
    return distribs
end

"""
_find_upper_lower_bounds(m::Matrix)
    returns upper and lower bounds in vector form
"""
function _find_upper_lower_bounds(X::Matrix)
    ub = []
    lb = []
    for col in eachcol(X)
        push!(ub, findmax(col)[1])
        push!(lb, findmin(col)[1])
    end
    if(size(X, 2)==1)
        return lb[1][1], ub[1][1]
    else
        return lb, ub
    end
end

"""
_find_best_model(clustered_values, clustered_test_values)
finds best model for each set of clustered values by validating against the clustered_test_values

"""
function _find_best_model(clustered_train_values, clustered_test_values, dim, enabled_expert_types)
    # find upper and lower bounds for clustered_train and test values concatenated

    x_vec = [a[1:dim] for a in clustered_train_values] 
    y_vec = [last(a) for a in clustered_train_values] 

    x_test_vec = [a[1:dim] for a in clustered_test_values] 
    y_test_vec = [last(a) for a in clustered_test_values]

    if (dim == 1)
        xtrain_mat = reshape(x_vec, (size(clustered_train_values, 1), dim))
        xtest_mat = reshape(x_test_vec, (size(clustered_test_values, 1), dim))
    else 
        xtrain_mat = _vector_of_tuples_to_matrix(x_vec)
        xtest_mat = _vector_of_tuples_to_matrix(x_test_vec)
    end

    X = vcat(xtrain_mat, xtest_mat)
    lb, ub = _find_upper_lower_bounds(X)
    
    # call on _surrogate_builder with clustered_train_vals, enabled expert types, lb, ub 

    surr_vec = _surrogate_builder(enabled_expert_types, length(enabled_expert_types), x_vec, y_vec, lb, ub) 

    # use the models to find best model after validating against test data and return best model 
    best_rmse = Inf
    best_model = surr_vec[1] #initial assignment can be any model
    for surr_model in surr_vec 
        pred = surr_model.(x_test_vec)
        rmse = norm(pred-y_test_vec, 2)
        if(rmse < best_rmse)
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
    local_surr = []
    for i in 1:k
        if local_kind[i][1] == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x, y, lb, ub,
                                     rad = local_kind[i].radial_function,
                                     scale_factor = local_kind[i].scale_factor,
                                     sparse = local_kind[i].sparse)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "Kriging"
            x = [a[1] for a in x] #because Kriging takes abs of two vectors

            my_local_i = Kriging(x, y, lb, ub, p = local_kind[i].p,
                                 theta = local_kind[i].theta)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "GEK"
            my_local_i = GEK(x, y, lb, ub, p = local_kind[i].p,
                             theta = local_kind[i].theta)
            push!(local_surr,  my_local_i)

        elseif local_kind[i] == "LinearSurrogate"
            my_local_i = LinearSurrogate(x, y, lb, ub)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x, y, lb, ub, local_kind[i].p)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "LobachevskySurrogate"
            my_local_i = LobachevskyStructure(x, y, lb, ub,
                                              alpha = local_kind[i].alpha,
                                              n = local_kind[i].n,
                                              sparse = local_kind[i].sparse)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x, y, lb, ub,
                                         model = local_kind[i].model,
                                         loss = local_kind[i].loss, opt = local_kind[i].opt,
                                         n_echos = local_kind[i].n_echos)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x, y, lb, ub,
                                               num_round = local_kind[i].num_round)
            push!(local_surr,  my_local_i)

        elseif local_kind[i] == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "Wendland"
            my_local_i = Wendand(x, y, lb, ub, eps = local_kind[i].eps,
                                 maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            push!(local_surr,  my_local_i)

        elseif local_kind[i][1] == "PolynomialChaosSurrogate"
            my_local_i = PolynomialChaosSurrogate(x, y, lb, ub, op = local_kind[i].op)
            push!(local_surr,  my_local_i)
        else
            throw("A surrogate with name provided does not exist or is not currently supported with MOE.")
        end 
    end 
    return local_surr
end 

"""
    _vector_of_tuples_to_matrix(v)
takes in a vector of tuples or vector of vectors and converts it into a matrix
"""
function _vector_of_tuples_to_matrix(v)
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

end #module
