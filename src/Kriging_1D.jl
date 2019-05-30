#=
One dimensional Kriging method, following this paper:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#
export Kriging_1D,evaluate_Kriging,Kriging_ND,evaluate_Kriging_ND

"""
    Kriging_1D(x,y,p)

Returns mu,b,sigma,inverse_of_R which are used to find
estimation at a new point.

#Arguments:

-(x,y): sampled points
-'p': value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity
"""
function Kriging_1D(x,y,p)
    if length(x) != length(y)
        error("Dimension of x and y are not equal")
    end
    n = length(x)
    theta_l = 1
    R = zeros(float(eltype(x)), n, n)
    for i = 1:n
        for j = 1:n
            R[i,j] = exp(-theta_l*abs(x[i]-x[j])^p)
        end
    end
    one = ones(n,1)
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t*inverse_of_R*y)/(one_t*inverse_of_R*one)
    b = inverse_of_R*(y-one*mu)
    sigma = ((y-one*mu)' * inverse_of_R * (y - one*mu))/n
    return mu[1], b, sigma[1],inverse_of_R
end

"""
    evaluate_Kriging(new_point,x,p,mu,b,sigma,inverse_of_R)

Returns the prediction at the new point and the expected mean squared error at
that point.

#Arguments
-'new_point': value at which we want the approximation
-'x': set of points
-'p': value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity
-'mu,b,sigma,inverse_of_R' values returned from Krigin_1D
"""
function evaluate_Kriging_1D(new_point,x,p,mu,b,sigma,inverse_of_R)
    n = length(x)
    phi(z) = exp(-(abs(z))^p)
    prediction = 0
    for i = 1:n
        prediction = prediction + b[i]*phi(new_point-x[i])
    end
    prediction = mu + prediction

    r = zeros(float(eltype(x)),n,1)
    for i = 1:n
        r[i] = phi(new_point - x[i])
    end
    one = ones(n,1)
    one_t = one'
    a = r'*inverse_of_R*r
    a = a[1]
    b = one_t*inverse_of_R*one
    b = b[1]
    mean_squared_error = sigma*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    return prediction,std_error
end

"""
    Kriging_ND(x,y,p)

Returns mu,b,sigma,inverse_of_R which are used to find
estimation at a new point

#Arguments:

-'x': NxL matrix of samples
-'y': vector containing values at samples
-'p': vector containing values 0<p_l<=2 determining the smoothness of
      the function in the l-th direction. Higher p_l higher the smoothness.
-'theta': vector containing values theta_l>=0. Large values of theta_l serve to
          model functions that are highly active in the l-th variable.
"""
function Kringing_ND(x,y,p,theta)
    if size(x,1) != length(y)
        error("Dimension of x and y are not equal")
    end
    n = size(x,1)
    d = size(x,2)

    R = zeros(float(eltype(x)), n, n)
    for i = 1:n
        for j = 1:n
            sum = 0
            for l = 1:d
            sum = sum + theta[l]*norm(x[i][l]-x[j][l])^p[l]
            end
            R[i,j] = exp(-sum)
        end
    end
    one = ones(n,1)
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t*inverse_of_R*y)/(one_t*inverse_of_R*one)
    b = inverse_of_R*(y-one*mu)
    sigma = ((y-one*mu)' * inverse_of_R * (y - one*mu))/n
    return mu[1], b, sigma[1],inverse_of_R

end

"""
    evaluate_Kriging_ND(new_point,x,p,mu,b,sigma,inverse_of_R)

Returns the prediction at the new point and the expected mean squared error at
that point.

#Arguments
-'new_point': value at which we want the approximation
-'x': set of points
-'p': vector containing values 0<p_l<=2 determining the smoothness of
      the function in the l-th direction. Higher p_l higher the smoothness.
-'theta': vector containing values theta_l>=0. Large values of theta_l serve to
          model functions that are highly active in the l-th variable.
-'mu,b,sigma,inverse_of_R' values returned from Krigin_1D
"""
function evaluate_Kriging_ND(new_point,x,p,theta,mu,b,sigma,inverse_of_R)
    n = size(x,1)
    d = size(x,2)
    prediction = 0
    for i = 1:n
        sum = 0
        for l = 1:d
            sum = sum + theta[l]*norm(new_point[l]-x[i][l])^p[l]
        end
        prediction = prediction + b[i]*exp(-sum)
    end
    prediction = mu + prediction

    r = zeros(float(eltype(x)),n,1)
    for i = 1:n
        sum = 0
        for l = 1:d
            sum = sum + theta[l]*norm(new_point[l]-x[i][l])^p[l]
        end
        r[i] = exp(-sum)
    end
    one = ones(n,1)
    one_t = one'
    a = r'*inverse_of_R*r
    a = a[1]
    b = one_t*inverse_of_R*one
    b = b[1]
    mean_squared_error = sigma*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    return prediction,std_error
end
