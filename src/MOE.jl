using Clustering
using GaussianMixtures
using LinearAlgebra
using XGBoost
using SurrogatesRandomForest, SurrogatesFlux, SurrogatesPolyChaos

mutable struct MOE{X, Y, L, U, S, K, M, V, W} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    local_surr::S
    k::K
    means::M
    varcov::V
    weights::W
end

function MOE(x, y, lb::Number, ub::Number; scale_factor::Number = 1.0, k::Int = 2,
             local_kind = [
                 RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
                                      sparse = false),
                 RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0,
                                      sparse = false),
             ])
    if k != length(local_kind)
        throw("Number of mixtures = $k is not equal to length of local surrogates")
    end
    n = length(x)
    x = x ./ scale_factor
    y = y ./ scale_factor
    # find weight, mean and variance for each mixture
    # For GaussianMixtures I need nxd matrix
    X_G = reshape(x, (n, 1))
    moe_gmm = GaussianMixtures.GMM(k, X_G)
    weights = GaussianMixtures.weights(moe_gmm)
    means = GaussianMixtures.means(moe_gmm)
    variances = moe_gmm.Σ

    #cluster the points
    #For clustering I need dxn matrix
    X_C = reshape(x, (1, n))
    KNN = kmeans(X_C, k)
    x_c = [Array{eltype(x)}(undef, 0) for i in 1:k]
    y_c = [Array{eltype(y)}(undef, 0) for i in 1:k]
    a = assignments(KNN)
    @inbounds for i in 1:n
        pos = a[i]
        append!(x_c[pos], x[i])
        append!(y_c[pos], y[i])
    end

    local_surr = Dict()
    for i in 1:k
        if local_kind[i][1] == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x_c[i], y_c[i], lb, ub,
                                     rad = local_kind[i].radial_function,
                                     scale_factor = local_kind[i].scale_factor,
                                     sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Kriging"
            my_local_i = Kriging(x_c[i], y_c[i], lb, ub, p = local_kind[i].p,
                                 theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "GEK"
            my_local_i = GEK(x_c[i], y_c[i], lb, ub, p = local_kind[i].p,
                             theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "LinearSurrogate"
            my_local_i = LinearSurrogate(x_c[i], y_c[i], lb, ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x_c[i], y_c[i], lb, ub, local_kind[i].p)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "LobachevskySurrogate"
            my_local_i = LobachevskySurrogate(x_c[i], y_c[i], lb, ub,
                                              alpha = local_kind[i].alpha,
                                              n = local_kind[i].n,
                                              sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x_c[i], y_c[i], lb, ub,
                                         model = local_kind[i].model,
                                         loss = local_kind[i].loss, opt = local_kind[i].opt,
                                         n_echos = local_kind[i].n_echos)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x_c[i], y_c[i], lb, ub,
                                               num_round = local_kind[i].num_round)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x_c[i], y_c[i], lb, ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Wendland"
            my_local_i = Wendand(x_c[i], y_c[i], lb, ub, eps = local_kind[i].eps,
                                 maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "PolynomialChaosSurrogate"
            my_local_i = PolynomialChaosSurrogate(x, y, lb, ub, op = local_kind[i].op)
            local_surr[i] = my_local_i
        else
            throw("A surrogate with name provided does not exist or is not currently supported with MOE.")
        end
    end
    return MOE(x, y, lb, ub, local_surr, k, means, variances, weights)
end

function MOE(x, y, lb, ub; k::Int = 2, scale_factor::Number = 1.0,
             local_kind = [
                 RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
                                      sparse = false),
                 RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0,
                                      sparse = false),
             ])
    n = length(x)
    d = length(lb)
    for i in 1:n
        x[i] = x[i] ./ scale_factor
    end
    y = y ./ scale_factor
    #GMM parameters:
    X_G = collect(reshape(collect(Base.Iterators.flatten(x)), (d, n))')
    my_gmm = GMM(k, X_G, kind = :full)
    weights = my_gmm.w
    means = my_gmm.μ
    varcov = my_gmm.Σ

    #cluster the points
    X_C = collect(reshape(collect(Base.Iterators.flatten(x)), (d, n)))
    KNN = kmeans(X_C, k)
    x_c = [Array{eltype(x)}(undef, 0) for i in 1:k]
    y_c = [Array{eltype(y)}(undef, 0) for i in 1:k]
    a = assignments(KNN)
    @inbounds for i in 1:n
        pos = a[i]
        x_c[pos] = vcat(x_c[pos], x[i])
        append!(y_c[pos], y[i])
    end

    local_surr = Dict()
    for i in 1:k
        if local_kind[i][1] == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x_c[i], y_c[i], lb, ub,
                                     rad = local_kind[i].radial_function,
                                     scale_factor = local_kind[i].scale_factor,
                                     sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Kriging"
            my_local_i = Kriging(x_c[i], y_c[i], lb, ub, p = local_kind[i].p,
                                 theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "GEK"
            my_local_i = GEK(x_c[i], y_c[i], lb, ub, p = local_kind[i].p,
                             theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "LinearSurrogate"
            my_local_i = LinearSurrogate(x_c[i], y_c[i], lb, ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x_c[i], y_c[i], lb, ub, local_kind[i].p)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "LobachevskySurrogate"
            my_local_i = LobachevskySurrogate(x_c[i], y_c[i], lb, ub,
                                              alpha = local_kind[i].alpha,
                                              n = local_kind[i].n,
                                              sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x_c[i], y_c[i], lb, ub,
                                         model = local_kind[i].model,
                                         loss = local_kind[i].loss, opt = local_kind[i].opt,
                                         n_echos = local_kind[i].n_echos)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x_c[i], y_c[i], lb, ub,
                                               num_round = local_kind[i].num_round)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x_c[i], y_c[i], lb, ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Wendland"
            my_local_i = Wendand(x_c[i], y_c[i], lb, ub, eps = local_kind[i].eps,
                                 maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "PolynomialChaosSurrogate"
            my_local_i = PolynomialChaosSurrogate(x, y, lb, ub, op = local_kind[i].op)
            local_surr[i] = my_local_i
        else
            throw("A surrogate with name " * local_kind[i][1] *
                  " does not exist or is not currently supported with MOE.")
        end
    end
    return MOE(x, y, lb, ub, local_surr, k, means, varcov, weights)
end

function _prob_x_in_i(x::Number, i, mu, varcov, alpha, k)
    num = (1 / sqrt(varcov[i])) * alpha[i] *
          exp(-0.5(x - mu[i]) * (1 / varcov[i]) * (x - mu[i]))
    den = sum([(1 / sqrt(varcov[j])) * alpha[j] *
               exp(-0.5(x - mu[j]) * (1 / varcov[j]) * (x - mu[j])) for j in 1:k])
    return num / den
end

function _prob_x_in_i(x, i, mu, varcov, alpha, k)
    num = (1 / sqrt(det(varcov[i]))) * alpha[i] *
          exp(-0.5 * (x .- mu[i, :])' * (inv(varcov[i])) * (x .- mu[i, :]))
    den = sum([(1 / sqrt(det(varcov[j]))) * alpha[j] *
               exp(-0.5 * (x .- mu[j, :])' * (inv(varcov[j])) * (x .- mu[j, :]))
               for j in 1:k])
    return num / den
end

function (moe::MOE)(val)
    return prod([moe.local_surr[i](val) *
                 _prob_x_in_i(val, i, moe.means, moe.varcov, moe.weights, moe.k)
                 for i in 1:(moe.k)])
end

function add_point!(my_moe::MOE, x_new, y_new)
    if length(my_moe.x[1]) == 1
        #1D
        my_moe.x = vcat(my_moe.x, x_new)
        my_moe.y = vcat(my_moe.y, y_new)
        n = length(my_moe.x)

        #New mixture parameters
        X_G = reshape(my_moe.x, (n, 1))
        moe_gmm = GaussianMixtures.GMM(my_moe.k, X_G)
        my_moe.weights = GaussianMixtures.weights(moe_gmm)
        my_moe.means = GaussianMixtures.means(moe_gmm)
        my_moe.varcov = moe_gmm.Σ

        #Find cluster of new point(s):
        n_added = length(x_new)
        X_C = reshape(my_moe.x, (1, n))
        KNN = kmeans(X_C, my_moe.k)
        a = assignments(KNN)
        #Recalculate only relevant surrogates
        for i in 1:n_added
            pos = a[n - n_added + i]
            add_point!(my_moe.local_surr[i], my_moe.x[n - n_added + i],
                       my_moe.y[n - n_added + i])
        end
    else
        #ND
        my_moe.x = vcat(my_moe.x, x_new)
        my_moe.y = vcat(my_moe.y, y_new)
        n = length(my_moe.x)
        d = length(my_moe.lb)
        #New mixture parameters
        X_G = collect(reshape(collect(Base.Iterators.flatten(my_moe.x)), (d, n))')
        my_gmm = GMM(my_moe.k, X_G, kind = :full)
        my_moe.weights = my_gmm.w
        my_moe.means = my_gmm.μ
        my_moe.varcov = my_gmm.Σ

        #cluster the points
        X_C = collect(reshape(collect(Base.Iterators.flatten(my_moe.x)), (d, n)))
        KNN = kmeans(X_C, my_moe.k)
        a = assignments(KNN)
        n_added = length(x_new)
        for i in 1:n_added
            pos = a[n - n_added + i]
            add_point!(my_moe.local_surr[i], my_moe.x[n - n_added + i],
                       my_moe.y[n - n_added + i])
        end
    end
    nothing
end
