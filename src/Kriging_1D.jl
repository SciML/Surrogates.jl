"""
Kriging:
(x,y): sampled point
p: value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity
"""
function Kriging_1D(x,y,p)
    if length(x) != length(y)
        error("Dimension of x and y are not equal")
    end
    n = length(x)
    theta_l = 1
    R = zeros(Float32, n, n)
    for i = 1:n
        for j = 1:n
            R[i,j] = exp(-theta_l*abs(x[i]-x[j])^p)
        end
    end

    #Finding coeffcients
    one = ones(n,1)
    one_t = one'
    inverse_of_R = R^(-1)
    mu = (one_t*inverse_of_R*y)/(one_t*inverse_of_R*one)
    b = inverse_of_R*(y-one*mu)
    sigma = ((y-one*mu)' * inverse_of_R * (y - one*mu))/n
    return mu[1], b, sigma[1],inverse_of_R
end

"""
Kriging predictor:
y(x*) = mu + sum(b_i * phi(x* - x[i]))
new_point: value at which we want the approximation
x: set of points
p: value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity
mu,b,sigma,inv values returned from Krigin_1D

Returns the prediction at the new point and the expected mean squared error at
that point
"""
function evaluate(new_point,x,p,mu,b,sigma,inv)
    n = length(x)
    phi(z) = exp(-(abs(z))^p)
    prediction = 0
    for i = 1:n
        prediction = prediction + b[i]*phi(new_point-x[i])
    end
    prediction = mu + prediction

    r = zeros(Float32,n,1)
    for i = 1:n
        r[i] = phi(new_point - x[i])
    end
    one = ones(n,1)
    one_t = one'
    a = r'*inv*r
    a = a[1]
    b = one_t*inv*one
    b = b[1]
    mean_squared_error = sigma*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    return prediction,std_error
end
