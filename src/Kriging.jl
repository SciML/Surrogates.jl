using LinearAlgebra
#=
One dimensional Kriging method, following this paper:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#

abstract type AbstractBasisFunction end

export Kriging

struct Kriging <: AbstractBasisFunction
    x
    y
    p
    mu
    b
    sigma
    inverse_of_R
    new_point
    prediction
    std_error
 end


"""
    Kriging(new_point::Number,x::Array,y::Array,p::Number)

Constructor for type Kriging.

#Arguments:
-'new_point': value at which you want to calculate approximation
-(x,y): sampled points
-'p': value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity

"""
function Kriging(new_point::Number,x::Array,y::Array,p::Number)
    n = length(x)
    theta_l = 1
    R = zeros(float(eltype(x)), n, n)
    @inbounds for i = 1:n
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
    #mu[1], b, sigma[1],inverse_of_R

    phi(z) = exp(-(abs(z))^p)
    prediction = zero(eltype(x))
    @inbounds for i = 1:n
        prediction = prediction + b[i]*phi(new_point-x[i])
    end
    prediction = mu[1] + prediction

    r = zeros(float(eltype(x)),n,1)
    @inbounds for i = 1:n
        r[i] = phi(new_point - x[i])
    end
    one = ones(n,1)
    one_t = one'
    a = r'*inverse_of_R*r
    a = a[1]
    b = one_t*inverse_of_R*one
    b = b[1]
    mean_squared_error = sigma[1]*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    Kriging(x,y,p,mu,b,sigma,inverse_of_R,new_point,prediction,std_error)
end



"""
    Kriging(new_point::Array,x::Array,y::Array,p::Array,theta::Array)

Constructor for type Kriging.

#Arguments:
-'new_point': Array at which you want to calculate approximation
-''(x,y)': sampled points
-'p': array of values between 0 and 2 modelling the
   smoothness of the function being approximated in the i-th variable,
    0-> rough  2-> C^infinity
-'theta': array of values bigger than 0 modellig how much the function is
          changing in the i-th variable

"""
function Kriging(new_point::Array,x::Array,y::Array,p::Array,theta::Array)
    n = size(x,1)
    d = size(x,2)
    R = zeros(float(eltype(x)), n, n)
    @inbounds for i = 1:n
        for j = 1:n
            sum = zero(eltype(x))
            for l = 1:d
            sum = sum + theta[l]*norm(x[i,l]-x[j,l])^p[l]
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
    #mu[1], b, sigma[1],inverse_of_R
    prediction = zero(eltype(x))
    @inbounds for i = 1:n
        sum = zero(eltype(x))
        for l = 1:d
            sum = sum + theta[l]*norm(new_point[l]-x[i,l])^p[l]
        end
        prediction = prediction + b[i]*exp(-sum)
    end
    prediction = mu[1] + prediction

    r = zeros(float(eltype(x)),n,1)
    @inbounds for i = 1:n
        sum = zero(eltype(x))
        for l = 1:d
            sum = sum + theta[l]*norm(new_point[l]-x[i,l])^p[l]
        end
        r[i] = exp(-sum)
    end
    one = ones(n,1)
    one_t = one'
    a = r'*inverse_of_R*r
    a = a[1]
    b = one_t*inverse_of_R*one
    b = b[1]
    mean_squared_error = sigma[1]*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    Kriging(x,y,p,mu,b,sigma,inverse_of_R,new_point,prediction,std_error)
end
