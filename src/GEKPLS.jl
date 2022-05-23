using ForwardDiff
using LinearAlgebra
using ScikitLearn
using Statistics

@sk_import cross_decomposition: PLSRegression

mutable struct GEKPLS{X,Y,XL, T} <: AbstractSurrogate
    x::X
    y::Y
    xl::XL #xlimits
    # lb::L
    # ub::U
    # coeff_pls::C
    # alpha::A
    # beta::B
    # gamma::G
    theta::T
  end
  
function GEKPLS(X,y, gradients, n_comp, delta_x, xlimits, extra_points, theta )
    println("GEKPLS constructor called") 
    # parameters = 25.0 
    # bb = boxbehnken(3, 1)
    # println("printing bb below")
    # println(bb)
    # println("printed bb above")
    println("calling on _ge_compute_pls")
    my_pls_mean, my_X, my_y = _ge_compute_pls(X, y, n_comp, gradients, delta_x, xlimits, extra_points)
    display(my_pls_mean)
    display(my_X)
    display(my_y)
    println("now constructing struct")
    return GEKPLS(X,y,xlimits,theta)
end
  
function add_point!(GEKPLS,x_new,y_new)  
    nothing
end
  
function (s::GEKPLS)(value)
    println("GEKPLS predictor called")
    #println(s.coeff_pls)
    #return s.coeff*value
    #return s.coeff*value + s.alpha
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
    _pls = PLSRegression(n_comp) 
    coeff_pls = zeros((dim, n_comp, nt))
    for i in 1:nt
        if dim >= 3 #to do ... this is consistent with SMT but inefficient. Move outside for loop and evaluate dim >= 3 just once.
            bb_vals = circshift(boxbehnken(dim, 1),1)
            _X = zeros((size(bb_vals)[1], dim)) 
            _y = zeros((size(bb_vals)[1], 1)) 
            bb_vals = bb_vals .* (delta_x * (xlimits[:, 2] - xlimits[:, 1]))' #smt calls this sign. I've called it bb_vals
            _X = X[i, :]' .+ bb_vals 
            bb_vals =  bb_vals .* grads[i,:]'  
            _y = y[i, :] .+ sum(bb_vals, dims=2) 
        else
            println("GEKPLS for less than 3 dimensions is coming soon")
        end
        _pls.fit(_X, _y)
        coeff_pls[:, :, i] = _pls.x_rotations_ #size of _pls.x_rotations_ is (dim, n_comp)
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

function _standardization(X,y)
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


function _reduced_likelihood_function()
    nothing
end

function _check_param()
    nothing
end

function _squar_exp(theta, d)
    n_components = size(d)[2]
    theta = reshape(theta, (1,n_components))
    return exp.(-sum(theta .* d, dims=2))
end
    
function _cross_distances(X)
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

function _componentwise_distance_PLS(D, corr, n_comp, coeff_pls)
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

#bb = boxbehnken(3, 1);
######end of bb design######
