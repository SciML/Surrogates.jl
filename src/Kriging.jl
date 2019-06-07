using LinearAlgebra
#=
One dimensional Kriging method, following this paper:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#

abstract type AbstractBasisFunction end

export Kriging,add_point!,current_estimate

mutable struct Kriging <: AbstractBasisFunction
    x
    y
    p
    theta
    mu
    b
    sigma
    inverse_of_R
 end

 """
 Gives the current estimate for array 'val' with respect to the Kriging object k.
 """
 function (k::Kriging)(val)
     prediction = zero(eltype(k.x))
     n = Base.size(k.x,1)
     d = Base.size(k.x,2)
     r = zeros(float(eltype(k.x)),n,1)
     @inbounds for i = 1:n
         sum = zero(eltype(k.x))
         for l = 1:d
             sum = sum + k.theta[l]*norm(val[l]-k.x[i,l])^(k.p[l])
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
 Gives the current estimate for 'val' with respect to the Kriging object k.
 """
 function (k::Kriging)(val::Number)
     phi(z) = exp(-(abs(z))^k.p)
     n = length(k.x)
     prediction = zero(eltype(k.x))
     r = zeros(float(eltype(k.x)),n,1)
     @inbounds for i = 1:n
         prediction = prediction + k.b[i]*phi(val-k.x[i])
         r[i] = phi(val - k.x[i])
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
    Kriging(x,y,p::Number)

Constructor for type Kriging.

#Arguments:
-(x,y): sampled points
-'p': value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity

"""
function Kriging(x,y,p::Number)

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
    Kriging(x,y,p,theta,mu[1],b,sigma[1],inverse_of_R)
end



"""
    Kriging(x,y,p,theta)

Constructor for type Kriging.

#Arguments:
-''(x,y)': sampled points
-'p': array of values between 0 and 2 modelling the
   smoothness of the function being approximated in the i-th variable,
    0-> rough  2-> C^infinity
-'theta': array of values bigger than 0 modellig how much the function is
          changing in the i-th variable

"""
function Kriging(x,y,p,theta)
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
    Kriging(x,y,p,theta,mu[1],b,sigma[1],inverse_of_R)
end


"""
    add_point!(k::Kriging,new_x,new_y)

Adds the new point and its respective value to the sample points.
Warning: If you are just adding a single point, you have to wrap it with []
Returns the updated Kriging model.

"""
function add_point!(k::Kriging,new_x,new_y)
    if Base.size(k.x,1) == 1
        if length(new_x) > 1
            k.x = hcat(k.x,new_x)
            k.y = vcat(k.y,new_y)
            return Kriging(k.x,k.y,k.p)
        else
            k.x = vcat(vec(k.x),new_x)
            k.y = vcat(vec(k.y),new_y)
            return Kriging(k.x,k.y,k.p)
        end
    else
        k.x = vcat(k.x,new_x)
        k.y = vcat(k.y,new_y)
        return Kriging(k.x,k.y,k.p,k.theta)
    end
end
