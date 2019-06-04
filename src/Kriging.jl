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
    theta
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
    theta = 1
    R = zeros(float(eltype(x)), n, n)
    @inbounds for i = 1:n
        for j = 1:n
            R[i,j] = exp(-theta*abs(x[i]-x[j])^p)
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
    Kriging(x,y,p,theta,mu,b,sigma,inverse_of_R,new_point,prediction,std_error)
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
    Kriging(x,y,p,theta,mu,b,sigma,inverse_of_R,new_point,prediction,std_error)
end


"""
    add_point!(k::AbstractBasisFunction,new_x::Array,new_y::Array)

Adds the new point and its respective value to the sample points.
Returns the updated Kriging model.

"""
function add_point!(k::AbstractBasisFunction,new_x::Array,new_y::Array)
    k.x = vcat(k.x,new_x)
    k.y = vcat(k.y,new_y)
    return Kriging(k.x,k.y,k.p,k.theta,k.mu,k.b,k.sigma,k.inverse_of_R,k.new_point,k.prediction,k.std_error)
end


"""
    current_estimate(k::AbstractBasisFunction,val::Array)

Gives the current estimate for array value 'val' with respect to the Kriging object k.
"""
function current_estimate(k::AbstractBasisFunction,val::Array)
    prediction = zero(eltype(x))
    r = zeros(float(eltype(x)),n,1)
    @inbounds for i = 1:n
        sum = zero(eltype(x))
        for l = 1:d
            sum = sum + k.theta[l]*norm(val[l]-k.x[i,l])^k.p[l]
        end
        r[i] = exp(-sum)
        prediction = prediction + k.b[i]*exp(-sum)
    end
    prediction = k.mu + prediction

    one = ones(n,1)
    one_t = one'
    a = r'*k.inverse_of_R*r
    a = a[1]
    b = one_t*k.inverse_of_R*one
    b = b[1]
    mean_squared_error = k.sigma*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    return prediction, std_error
end

"""
    current_estimate(k::AbstractBasisFunction,val::Array)

Gives the current estimate for 'val' with respect to the Kriging object k.
"""
function current_estimate(k::AbstractBasisFunction,val::Number)
    phi(z) = exp(-(abs(z))^p)
    prediction = zero(eltype(x))
    r = zeros(float(eltype(x)),n,1)
    @inbounds for i = 1:n
        prediction = prediction + k.b[i]*phi(val-k.x[i])
        r[i] = phi(new_point - k.x[i])
    end
    prediction = k.mu + prediction

    one = ones(n,1)
    one_t = one'
    a = r'*k.inverse_of_R*r
    a = a[1]
    b = one_t*k.inverse_of_R*one
    b = b[1]
    mean_squared_error = k.sigma*(1 - a + (1-a)^2/(b))
    std_error = sqrt(mean_squared_error)
    return prediction, std_error
end
