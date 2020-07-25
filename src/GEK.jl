
mutable struct GEK{X,Y,L,U,P,T,M,B,S,R,D} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    p::P
    theta::T
    mu::M
    b::B
    sigma::S
    inverse_of_R::R
 end

function _calc_gek_coeffs(x,y,p::Number,theta::Number)
    nd1 = length(x) #2n
    n = Int(nd1/2) #n
    R = zeros(eltype(x[1]), nd, nd)

    #top left
    @inbounds for i = 1:n
        for j = 1:n
            R[i,j] = exp(-theta*abs(x[i]-x[j])^p)
        end
    end
    #top right
    @inbounds for i = 1:n
        for j = n+1:nd1
            R[i,j] = 2*theta*(x[i] - x[j])*exp(-theta*abs(x[i]-x[j])^p)
        end
    end
    #bottom left
    @inbounds for i = n+1:nd1
        for j = 1:n
            R[i,j] = -2*theta*(x[i] - x[j])*exp(-theta*abs(x[i]-x[j])^p)
        end
    end
    #bottom right
    @inbounds for i = n+1:nd1
        for j = 1:n
            R[i,j] = -4*theta*(x[i] - x[j])^2*exp(-theta*abs(x[i]-x[j])^p)
        end
    end
    one = ones(eltype(x[1]),nd1,1)
    for i = n+1:nd1
        one[i] = zero(eltype(x[1]))
    end
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t*inverse_of_R*y)/(one_t*inverse_of_R*one)
    b = inverse_of_R*(y-one*mu)
    sigma = ((y-one*mu)' * inverse_of_R * (y - one*mu))/n
    mu[1], b, sigma[1],inverse_of_R
end

function std_error_at_point(k::GEK,val::Number)
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

function (k::GEK)(val::Number)
    phi = z -> exp(-(abs(z))^k.p)
    nd = length(k.x)
    prediction = zero(eltype(k.x[1]))
    for i = 1:nd
        prediction = prediction + k.b[i]*phi(val-k.x[i])
    end
    prediction = k.mu + prediction
    return prediction
end
function GEK(x,y,lb::number,ub::number; p=1.0,theta=1.0)
    if length(x) != length(unique(x))
        println("There exists a repetion in the samples, cannot build Kriging.")
        return
    end
    mu,b,sigma,inverse_of_R = _calc_gek_coeffs(x,y,p,theta)
    return GEK(x,y,kb,ub,p,theta,mu,b,sigma,inverse_of_R)
end





function _calc_gek_coeffs(x,y,p,theta)
    #todo
end

function std_error_at_point(k::GEK,val)
    #todo
end

function (k::GEK)(val)
    #todp
end
function GEK(x,y,lb,ub; p=collect(one.(x[1])),theta=collect(one.(x[1])))
    #todo
end




function add_point!(k::GEK,new_x,new_y)
    if new_x in k.x
        println("Adding a sample that already exists, cannot build Kriging.")
        return
    end
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(k.theta)>1)
        push!(k.x,new_x)
        push!(k.y,new_y)
    else
        append!(k.x,new_x)
        append!(k.y,new_y)
    end
    k.mu,k.b,k.sigma,k.inverse_of_R = _calc_gek_coeffs(k.x,k.y,k.p,k.theta)
    nothing
end
