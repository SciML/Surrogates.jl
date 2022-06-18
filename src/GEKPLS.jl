using LinearAlgebra
using Statistics

mutable struct GEKPLS <: AbstractSurrogate
    x::Matrix{Float64} #1
    y::Matrix{Float64} #2
    grads::Matrix{Float64} #3
    xl::Matrix{Float64} #xlimits #4
    delta:: Float64 #5
    extra_points::Int64 #6
    num_components::Int64 #7
    beta::Vector{Float64} #8
    gamma::Matrix{Float64} #9
    theta::Vector{Float64} #10
    reduced_likelihood_function_value:: Float64 #11
    X_offset::Matrix{Float64} #12
    X_scale::Matrix{Float64} #13
    X_after_std::Matrix{Float64} #14 - X after standardization
    pls_mean::Matrix{Float64} #15
    y_mean::Float64 #16
    y_std::Float64 #17
end

#constructor for GEKPLS Struct
function GEKPLS(X, y, grads, n_comp, delta_x, xlimits, extra_points, θ)
    theta = [θ for i in 1:n_comp]
    pls_mean, X_after_PLS, y_after_PLS = _ge_compute_pls(X, y, n_comp, grads, delta_x, xlimits, extra_points)
    X_after_std, y_after_std, X_offset, y_mean, X_scale, y_std = standardization(X_after_PLS, y_after_PLS)
    D, ij = cross_distances(X_after_std)
    pls_mean_reshaped = reshape(pls_mean, (size(X,2),n_comp))
    d = componentwise_distance_PLS(D, "squar_exp", n_comp, pls_mean_reshaped)
    nt, nd = size(X_after_PLS)
    beta, gamma, reduced_likelihood_function_value = _reduced_likelihood_function(theta, "squar_exp", d, nt, ij, y_after_std)
    return GEKPLS(X, y, grads, xlimits, delta_x, extra_points, n_comp, beta, gamma, theta, reduced_likelihood_function_value, 
            X_offset, X_scale, X_after_std, pls_mean_reshaped, y_mean,  y_std)
    println("struct created")
end


# predictor 
function (g::GEKPLS)(X_test)
    n_eval, n_features_x = size(X_test)
    X_cont = (X_test .- g.X_offset) ./ g.X_scale
    dx = differences(X_cont, g.X_after_std)
    pred_d = componentwise_distance_PLS(dx, "squar_exp", g.num_components, g.pls_mean)
    nt = size(g.X_after_std,1)
    r = transpose(reshape(squar_exp(g.theta, pred_d),(nt,n_eval)))
    f = ones(n_eval,1)
    y_ = (f * g.beta) + (r * g.gamma)
    y = g.y_mean .+ g.y_std * y_
    return y
end

function _ge_compute_pls(X, y, n_comp, grads, delta_x, xlimits, extra_points)
    """
    Gradient-enhanced PLS-coefficients.
        Parameters
        ----------
        X: [n_obs,dim] - The input variables.
        y: [n_obs,ny] - The output variable
        n_comp: int - Number of principal components used.
        gradients: - The gradient values. Matrix size (n_obs,dim)
        delta_x: real - The step used in the First Order Taylor Approximation
        xlimits: [dim, 2]- The upper and lower var bounds.
        extra_points: int - The number of extra points per each training point.
        Returns
        -------
        Coeff_pls: [dim, n_comp] - The PLS-coefficients.
        X: Concatenation of XX: [extra_points*nt, dim] - Extra points added (when extra_points > 0) and X
        y: Concatenation of yy[extra_points*nt, 1]- Extra points added (when extra_points > 0) and y
        """
    # this function is equivalent to a combination of 
    # https://github.com/SMTorg/smt/blob/f124c01ffa78c04b80221dded278a20123dac742/smt/utils/kriging_utils.py#L1036
    # and https://github.com/SMTorg/smt/blob/f124c01ffa78c04b80221dded278a20123dac742/smt/surrogate_models/gekpls.py#L48
    nt, dim = size(X)
    XX = zeros(0,dim)
    yy = zeros(0,size(y)[2])
    #_pls = PLSRegression(n_comp) #todo: relic from sklearn versiom. REMOVE.
    coeff_pls = zeros((dim, n_comp, nt))
    
    for i in 1:nt
        if dim >= 3 
            bb_vals = circshift(boxbehnken(dim, 1),1)
            # _X = zeros((size(bb_vals)[1], dim)) 
            # _y = zeros((size(bb_vals)[1], 1)) 
            # bb_vals = bb_vals .* (delta_x * (xlimits[:, 2] - xlimits[:, 1]))' #smt calls this sign. I've called it bb_vals
            # _X = X[i, :]' .+ bb_vals 
            # bb_vals =  bb_vals .* grads[i,:]'  
            # _y = y[i, :] .+ sum(bb_vals, dims=2) 
        else
            bb_vals = [
                0.0 0.0; #center
                1.0 0.0; #right
                0.0 1.0; #up
                -1.0 0.0; #left
                0.0 -1.0; #down
                1.0 1.0; #right up
                -1.0 1.0; #left up
                -1.0 -1.0; #left down
                1.0 -1.0; #right down
            ] 
        end
        _X = zeros((size(bb_vals)[1], dim)) 
        _y = zeros((size(bb_vals)[1], 1)) 
        bb_vals = bb_vals .* (delta_x * (xlimits[:, 2] - xlimits[:, 1]))' #smt calls this sign. I've called it bb_vals
        _X = X[i, :]' .+ bb_vals 
        bb_vals =  bb_vals .* grads[i,:]'  
        _y = y[i, :] .+ sum(bb_vals, dims=2)
        
        #_pls.fit(_X, _y) #todo: relic from sklearn versiom. REMOVE.
        #coeff_pls[:, :, i] = _pls.x_rotations_ #size of _pls.x_rotations_ is (dim, n_comp) #todo: relic from sklearn versiom. REMOVE.

        coeff_pls[:, :, i] = _modified_pls(_X, _y, n_comp)

        if extra_points != 0
            start_index = max(1, length(coeff_pls[:,1,i])-extra_points+1) #todo: evaluate just once
            max_coeff = sortperm(broadcast(abs,coeff_pls[:,1,i]))[start_index:end]
            for ii in max_coeff
                XX = [XX; transpose(X[i, :])]
                XX[end, ii] += delta_x * (xlimits[ii,2]-xlimits[ii,1])
                yy = [yy; y[i]]
                yy[end] += grads[i,ii] * delta_x * (xlimits[ii,2]-xlimits[ii,1])                
            end
        end
    end
    if extra_points != 0
        X = [X; XX]
        y = [y; yy]
    end

    pls_mean = mean(broadcast(abs,coeff_pls),dims=3)
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
        boxbehnken(matrix_size, matrix_size) 
    end
    
    function boxbehnken(matrix_size::Int, center::Int)
        @assert matrix_size>=3
    
        A_fact = explicit_fullfactorial(Tuple([-1,1] for i = 1:2))
        
        rows = floor(Int, (0.5*matrix_size*(matrix_size-1))*size(A_fact)[1])
        
        A = zeros(rows, matrix_size)
    
        l = 0
        for i in 1:matrix_size-1
            for j in i+1:matrix_size
                l = l +1
                A[max(0, (l - 1)*size(A_fact)[1])+1:l*size(A_fact)[1], i] = A_fact[:, 1]
                A[max(0, (l - 1)*size(A_fact)[1])+1:l*size(A_fact)[1], j] = A_fact[:, 2]
            end
        end
    
        if center == matrix_size
            if matrix_size <= 16
                points = [0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
                center = points[matrix_size]
            end
        end
    
        A = transpose(hcat(transpose(A), transpose(zeros(center, matrix_size))))
    end
    
    function explicit_fullfactorial(factors::Tuple)
        explicit_fullfactorial(fullfactorial(factors))
    end
    
    function explicit_fullfactorial(iterator::Base.Iterators.ProductIterator)
        hcat(vcat.(collect(iterator)...)...)
    end
    
    function fullfactorial(factors::Tuple)
        Base.Iterators.product(factors...)
    end
    

######end of bb design######

function standardization(X,y)
    """
    We substract the mean from each variable. Then, we divide the values of each
    variable by its standard deviation. 

    Parameters
    ----------

    X - The input variables.
    y - The output variable.

    Returns
    -------

    X: [n_obs, dim]
       The standardized input matrix.

    y: [n_obs, 1]
       The standardized output vector.

    X_offset: The mean (or the min if scale_X_to_unit=True) of each input variable.

    y_mean: The mean of the output variable.

    X_scale:  The standard deviation of each input variable.

    y_std: The standard deviation of the output variable.

    """
    #Equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L21
    X_offset = mean(X, dims = 1)
    X_scale = std(X, dims = 1)
    X_scale = map(x -> (x==0.0) ? x=1 : x=x, X_scale) #to prevent division by 0 below
    y_mean = mean(y)
    y_std = std(y)
    y_std = map(y -> (y==0) ? y=1 : y=y, y_std) #to prevent division by 0 below
    X = (X.-X_offset) ./ X_scale
    y = (y .- y_mean) ./ y_std
    return X, y, X_offset, y_mean, X_scale, y_std

end

function cross_distances(X)
    """
    Computes the nonzero componentwise cross-distances between the vectors
    in X

    Parameters
    ----------

    X: [n_obs, dim]

    Returns
    -------
    D:  [n_obs * (n_obs - 1) / 2, dim]
        - The cross-distances between the vectors in X.

    ij: [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """
    # equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L86
    n_samples, n_features = size(X)
    n_nonzero_cross_dist = ( n_samples * (n_samples - 1) ) ÷ 2
    ij = zeros((n_nonzero_cross_dist, 2))
    D = zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    
    for k in 1:n_samples - 1 
        ll_0 = ll_1 + 1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 1] .= k
        ij[ll_0:ll_1, 2] = k+1:1:n_samples
        D[ll_0:ll_1, :] = -(X[(k + 1) : n_samples,:] .- X[k,:]')
        
    end
    return D, Int.(ij)
end

function componentwise_distance_PLS(D, corr, n_comp, coeff_pls)
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
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L1257
    #todo
    #figure out how to handle this computation in the case of very large matrices
    #similar to what SMT has done
    #at https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L1257
    D_corr = zeros((size(D)[1], n_comp))

    if corr == "squar_exp"
        D_corr = D.^2 * coeff_pls.^2
    else #abs_exp
        D_corr = abs.(D) * abs.(coeff_pls)
    end

    return D_corr
end

function squar_exp(theta, d)
    """
    Squared exponential correlation model.
    Equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L604
    Parameters:
    -----------
    theta : Hyperparameters of the correlation model
    d: componentwise distances from componentwise_distance_PLS

    Returns:
    --------
    r:  array containing the values of the autocorrelation model
    
    """
    n_components = size(d)[2]
    theta = reshape(theta, (1,n_components))
    return exp.(-sum(theta .* d, dims=2))
end

function differences(X, Y)
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/utils/kriging_utils.py#L392
    #code credit: Elias Carvalho - https://stackoverflow.com/questions/72392010/row-wise-operations-between-matrices-in-julia
    Rx = repeat(X, inner=(size(Y, 1), 1))
    Ry = repeat(Y, size(X, 1))
    return Rx - Ry
end

function _reduced_likelihood_function(theta, kernel_type, d, nt, ij, y_norma, noise=0.0)
    """
    This function is a loose translation of SMT code from 
    https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/surrogate_models/krg_based.py#L247
    It  determines the BLUP parameters and evaluates the reduced likelihood function for the given theta.
    
    Parameters
    ----------
    theta: array containing the parameters at which the Gaussian Process model parameters should be determined.
    kernel_type: name of the correlation function.
    d: The componentwise cross-spatial-correlation-distance between the vectors in X.
    nt: number of training points
    ij: The indices i and j of the vectors in X associated to the cross-distances in D.
    y_norma: Standardized y values
    noise: noise hyperparameter - increasing noise reduces reduced_likelihood_function_value


    Returns
    -------
    reduced_likelihood_function_value: real
        - The value of the reduced likelihood function associated with the given autocorrelation parameters theta. 
    beta:  Generalized least-squares regression weights
    gamma: Gaussian Process weights.

    """
    #equivalent of https://github.com/SMTorg/smt/blob/4a4df255b9259965439120091007f9852f41523e/smt/surrogate_models/krg_based.py#L247
    reduced_likelihood_function_value = -Inf
    nugget = 1000000.0 * eps() #a jitter for numerical stability; reducing the multiple from 1000000.0 results in positive definite error for Cholesky decomposition;
    if kernel_type =="squar_exp" #todo - add other kernel type abs_exp etc.
        r = squar_exp(theta,d)
    end
    R = (I + zeros(nt,nt)) .* (1.0 + nugget + noise)

    for k in 1:size(ij)[1] #todo - speed up using @inbounds / @simd where required
        R[ij[k,1],ij[k,2]]=r[k]
        R[ij[k,2],ij[k,1]]=r[k]
    end

    C = cholesky(R).L #todo - #values diverge at this point from SMT code; verify impact
    F = ones(nt,1) #todo - examine if this should be a parameter for this function
    Ft = C\F 
    Q, G = qr(Ft)
    Q = Array(Q)
    Yt = C\y_norma
    #todo - in smt, they check if the matrix is ill-conditioned using SVD. Verify and include if necessary
    beta = G \ [(transpose(Q) ⋅ Yt)]
    rho = Yt .- (Ft .* beta)
    gamma = transpose(C) \ rho
    sigma2 = sum((rho).^2, dims=1)/nt
    detR = prod(diag(C).^(2.0/nt))
    reduced_likelihood_function_value = - nt * log10(sum(sigma2)) - nt * log10(detR)
    return beta, gamma, reduced_likelihood_function_value
end


### MODIFIED PLS BELOW ###

function _center_scale(X,Y)
    x_mean = mean(X, dims = 1)
    X .-= x_mean
    y_mean = mean(Y, dims = 1)
    Y .-= y_mean

    x_std = std(X, dims = 1) 
    x_std[x_std .== 0] .= 1.0
    X ./= x_std
    y_std = std(Y, dims = 1) 
    y_std[y_std .==0] .= 1.0
    Y ./= y_std
    return X, Y, x_mean, y_mean, x_std, y_std
end

function _svd_flip_1d(u,v)
    # equivalent of https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/cross_decomposition/_pls.py#L149
    biggest_abs_val_idx = findmax(abs.(vec(u)))[2]
    sign_ = sign(u[biggest_abs_val_idx])
    u .*= sign_
    v .*= sign_
    
end

function _get_first_singular_vectors_power_method(X,Y)
    #todo - do a check for Y residual being constant 
    #as in https://github.com/scikit-learn/scikit-learn/blob/80598905e517759b4696c74ecc35c6e2eb508cff/sklearn/cross_decomposition/_pls.py#L66
    x_weights_old = 100
    y_score = vec(Y)
    x_weights = transpose(X)y_score / dot(y_score, y_score)
    my_eps = eps()
    x_weights ./= (sqrt(dot(x_weights,x_weights)) + my_eps)
    x_score = X * x_weights
    y_weights = transpose(Y)x_score / dot(x_score, x_score)
    y_score = Y * y_weights / (dot(y_weights, y_weights) + my_eps)
    x_weights_diff = x_weights .- x_weights_old
    return x_weights, y_weights
end

function _modified_pls(X,Y, n_components)
    x_weights_  = zeros(size(X,2),n_components)
    y_weights_  = zeros(1, n_components)
    _x_scores = zeros(size(X,1), n_components)
    _y_scores = zeros(size(Y,1), n_components)
    x_loadings_ = zeros(size(X,2),n_components)
    y_loadings_ = zeros(1, n_components)

    Xk, Yk, x_mean, y_mean, x_std, y_std = _center_scale(X,Y)
    
    for k in 1:n_components
        x_weights, y_weights = _get_first_singular_vectors_power_method(Xk,Yk)
        _svd_flip_1d(x_weights, y_weights) 
        x_scores = Xk * x_weights
        y_ss = dot(y_weights, y_weights)
        y_scores = (Yk * y_weights) ./ y_ss
        x_loadings = transpose(x_scores)Xk / dot(x_scores, x_scores)
        Xk = Xk - (x_scores * x_loadings)
        y_loadings = transpose(x_scores)*Yk / dot(x_scores, x_scores)
        Yk = Yk - x_scores*y_loadings
        x_weights_[:,k] = x_weights 
        _x_scores[:,k] = x_scores 
        x_loadings_[:,k] = vec(x_loadings)
    end

    x_rotations_ = x_weights_ * pinv(transpose(x_loadings_)x_weights_)
    return x_rotations_
end

### MODIFIED PLS ABOVE ###
