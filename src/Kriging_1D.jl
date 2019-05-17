function Kriging_1D(x,y,p)
    #=
    theta_l > 0 higher theta models functions that are highly changing in the
                l-th coordinate, theta = 1 for now seemes the safest option
                since we are in 1D (Tried very different values but it does not change result)
    0 < p <= 2, higher p -> function being modelled is smoother
    =#
    theta_l = 1

    if length(x) != length(y)
        error("Dimension of x and y are not equal")
    end

    n = length(x)

    #Covariance matrix R
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

mu,b,sigma,inverse_cov = Kriging_1D([1,2,3,4], [3,4,5,8],1.5)
function evaluate(new_point,p,mu,b,x,sigma,inv)
    #= Krigin predictor:
    y(x*) = mu + sum(b_i * phi(x* - x[i]))
    =#
    n = length(x)
    #Building the predictor
    phi(z) = exp(-(abs(z))^p)
    prediction = 0
    for i = 1:n
        prediction = prediction + b[i]*phi(new_point-x[i])
    end
    prediction = mu + prediction
    #Building the error at the new point
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
