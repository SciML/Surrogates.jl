#=
One dimensional Kriging method, following this paper:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#

mutable struct Kriging{X,Y,P,T,M,B,S,R} <: AbstractSurrogate
    x::X
    y::Y
    p::P
    theta::T
    mu::M
    b::B
    sigma::S
    inverse_of_R::R
 end

 """
 Gives the current estimate for array 'val' with respect to the Kriging object k.
 """
 function (k::Kriging)(val)
     #val = tuple(val...)
     return k.mu + sum(k.b[i] * exp(-sum(dot(k.theta, norm.(val .- k.x[i]).^(k.p)))) for i in eachindex(k.x))
 end

 """
     Returns sqrt of expected mean_squared_error errot at the point.
 """
 function std_error_at_point(k::Kriging,val)
     n = length(k.x)
     d = length(k.x[1])
     r = zeros(eltype(k.x[1]),n,1)
     @inbounds for i = 1:n
         sum = zero(eltype(k.x[1]))
         for l = 1:d
             sum = sum + k.theta[l]*norm(val[l]-k.x[i][l])^(k.p[l])
         end
         r[i] = exp(-sum)
     end
     one = ones(eltype(k.x[1]),n,1)
     one_t = one'
     a = r'*k.inverse_of_R*r
     a = a[1]
     b = one_t*k.inverse_of_R*one
     b = b[1]
     mean_squared_error = k.sigma*(1 - a + (1-a)^2/(b))
     return sqrt(abs(mean_squared_error))
 end

 """
 Gives the current estimate for 'val' with respect to the Kriging object k.
 """
 function (k::Kriging)(val::Number)
     phi = z -> exp(-(abs(z))^k.p)
     n = length(k.x)
     prediction = zero(eltype(k.x[1]))
     for i = 1:n
         prediction = prediction + k.b[i]*phi(val-k.x[i])
     end
     prediction = k.mu + prediction
     return prediction
 end

"""
    Returns sqrt of expected mean_squared_error errot at the point.
"""
function std_error_at_point(k::Kriging,val::Number)
    phi(z) = exp(-(abs(z))^k.p)
    n = length(k.x)
    r = zeros(eltype(k.x[1]),n,1)
    @inbounds for i = 1:n
        r[i] = phi(val - k.x[i])
    end
    one = ones(eltype(k.x[1]),n,1)
    one_t = one'
    a = r'*k.inverse_of_R*r
    a = a[1]
    b = one_t*k.inverse_of_R*one
    b = b[1]
    mean_squared_error = k.sigma*(1 - a + (1-a)^2/(b))
    return sqrt(abs(mean_squared_error))
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
    mu,b,sigma,inverse_of_R = _calc_kriging_coeffs(x,y,p)
    theta = 1.0
    Kriging(x,y,p,theta,mu,b,sigma,inverse_of_R)
end

function _calc_kriging_coeffs(x,y,p::Number)
    n = length(x)
    R = zeros(eltype(x[1]), n, n)
    @inbounds for i = 1:n
        for j = 1:n
            R[i,j] = exp(-abs(x[i]-x[j])^p)
        end
    end
    one = ones(eltype(x[1]),n,1)
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t*inverse_of_R*y)/(one_t*inverse_of_R*one)
    b = inverse_of_R*(y-one*mu)
    sigma = ((y-one*mu)' * inverse_of_R * (y - one*mu))/n
    mu[1], b, sigma[1],inverse_of_R
end

"""
    Kriging(x,y,p,theta)

Constructor for Kriging surrogate.

- (x,y): sampled points
- p: array of values 0<=p<2 modelling the
     smoothness of the function being approximated in the i-th variable.
     low p -> rough, high p -> smooth
- theta: array of values > 0 modellig how much the function is
          changing in the i-th variable
"""
function Kriging(x,y,p,theta)
    mu,b,sigma,inverse_of_R = _calc_kriging_coeffs(x,y,p,theta)
    Kriging(x,y,p,theta,mu,b,sigma,inverse_of_R)
end

function _calc_kriging_coeffs(x,y,p,theta)
    n = length(x)
    d = length(x[1])
    R = zeros(float(eltype(x[1])), n, n)
    @inbounds for i = 1:n
        for j = 1:n
            sum = zero(eltype(x[1]))
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
    mu[1], b, sigma[1],inverse_of_R
end


"""
    add_point!(k::Kriging,new_x,new_y)

Adds the new point and its respective value to the sample points.
Warning: If you are just adding a single point, you have to wrap it with []
Returns the updated Kriging model.

"""
function add_point!(k::Kriging,new_x,new_y)
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(k.theta)>1)
        push!(k.x,new_x)
        push!(k.y,new_y)
    else
        append!(k.x,new_x)
        append!(k.y,new_y)
    end
    k.mu,k.b,k.sigma,k.inverse_of_R = _calc_kriging_coeffs(k.x,k.y,k.p,k.theta)
    nothing
end
