#=
One-dimensional Kriging method, following this paper:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#
mutable struct Kriging{X, Y, L, U, P, T, M, B, S, R} <: AbstractSurrogate
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

"""
Gives the current estimate for array 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val)
    d = length(val)
    n = length(k.x)
    return k.mu +
           sum(k.b[i] *
               exp(-sum(k.theta[j] * norm(val[j] - k.x[i][j])^k.p[j] for j in 1:d))
               for i in 1:n)
end

"""
    Returns sqrt of expected mean_squared_error error at the point.
"""
function std_error_at_point(k::Kriging, val)
    n = length(k.x)
    d = length(k.x[1])
    r = zeros(eltype(k.x[1]), n, 1)
    @inbounds for i in 1:n
        sum = zero(eltype(k.x[1]))
        for l in 1:d
            sum = sum + k.theta[l] * norm(val[l] - k.x[i][l])^(k.p[l])
        end
        r[i] = exp(-sum)
    end
    one = ones(eltype(k.x[1]), n, 1)
    one_t = one'
    a = r' * k.inverse_of_R * r
    a = a[1]
    b = one_t * k.inverse_of_R * one
    b = b[1]
    mean_squared_error = k.sigma * (1 - a + (1 - a)^2 / (b))
    return sqrt(abs(mean_squared_error))
end

"""
Gives the current estimate for 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val::Number)
    phi = z -> exp(-(abs(z))^k.p)
    n = length(k.x)
    prediction = zero(eltype(k.x[1]))
    for i in 1:n
        prediction = prediction + k.b[i] * phi(val - k.x[i])
    end
    prediction = k.mu + prediction
    return prediction
end

"""
    Returns sqrt of expected mean_squared_error error at the point.
"""
function std_error_at_point(k::Kriging, val::Number)
    phi(z) = exp(-(abs(z))^k.p)
    n = length(k.x)
    r = zeros(eltype(k.x[1]), n, 1)
    @inbounds for i in 1:n
        r[i] = phi(val - k.x[i])
    end
    one = ones(eltype(k.x[1]), n, 1)
    one_t = one'
    a = r' * k.inverse_of_R * r
    a = a[1]
    b = one_t * k.inverse_of_R * one
    b = b[1]
    mean_squared_error = k.sigma * (1 - a + (1 - a)^2 / (b))
    return sqrt(abs(mean_squared_error))
end

"""
    Kriging(x,y,lb::Number,ub::Number;p::Number=1.0)

Constructor for type Kriging.

#Arguments:
-(x,y): sampled points
-p: value between 0 and 2 modelling the
   smoothness of the function being approximated, 0-> rough  2-> C^infinity
"""
function Kriging(x, y, lb::Number, ub::Number; p = 1.0, theta = 1.0)
    if length(x) != length(unique(x))
        println("There exists a repetion in the samples, cannot build Kriging.")
        return
    end
    mu, b, sigma, inverse_of_R = _calc_kriging_coeffs(x, y, p, theta)
    Kriging(x, y, lb, ub, p, theta, mu, b, sigma, inverse_of_R)
end

function _calc_kriging_coeffs(x, y, p::Number, theta::Number)
    n = length(x)
    R = zeros(eltype(x[1]), n, n)

    #=@inbounds for i in 1:n
        for j in 1:n
            R[i, j] = exp(-theta * abs(x[i] - x[j])^p)
        end
    end=#

    R = [
        exp(-theta * abs(x[i] - x[j])^p)
        for j in 1:n, i in 1:n
    ]

    one = ones(eltype(x[1]), n, 1)
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)
    b = inverse_of_R * (y - one * mu)
    sigma = ((y - one * mu)' * inverse_of_R * (y - one * mu)) / n
    mu[1], b, sigma[1], inverse_of_R
end

"""
    Kriging(x,y,lb,ub;p=collect(one.(x[1])),theta=collect(one.(x[1])))

Constructor for Kriging surrogate.

- (x,y): sampled points
- p: array of values 0<=p<2 modeling the
     smoothness of the function being approximated in the i-th variable.
     low p -> rough, high p -> smooth
- theta: array of values > 0 modeling how much the function is
          changing in the i-th variable.
"""
function Kriging(x, y, lb, ub; p = collect(one.(x[1])), theta = collect(one.(x[1])))
    if length(x) != length(unique(x))
        println("There exists a repetition in the samples, cannot build Kriging.")
        return
    end
    mu, b, sigma, inverse_of_R = _calc_kriging_coeffs(x, y, p, theta)
    Kriging(x, y, lb, ub, p, theta, mu, b, sigma, inverse_of_R)
end

function _calc_kriging_coeffs(x, y, p, theta)
    n = length(x)
    d = length(x[1])
    #=R = zeros(float(eltype(x[1])), n, n)
    @inbounds for i in 1:n
        for j in 1:n
            sum = zero(eltype(x[1]))
            for l in 1:d
                sum = sum + theta[l] * norm(x[i][l] - x[j][l])^p[l]
            end
            R[i, j] = exp(-sum)
        end
    end=#

    R = [
        let
            sum = zero(eltype(x[1]))
            for l in 1:d
                sum = sum + theta[l] * norm(x[i][l] - x[j][l])^p[l]
            end
            exp(-sum)
        end

        for j in 1:n, i in 1:n
    ]

    one = ones(n, 1)
    one_t = one'
    inverse_of_R = inv(R)

    mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)

    y_minus_1μ = y - one * mu

    b = inverse_of_R * y_minus_1μ

    sigma = (y_minus_1μ' * inverse_of_R * y_minus_1μ) / n

    mu[1], b, sigma[1], inverse_of_R
end

"""
    add_point!(k::Kriging,new_x,new_y)

Adds the new point and its respective value to the sample points.
Warning: If you are just adding a single point, you have to wrap it with [].
Returns the updated Kriging model.

"""
function add_point!(k::Kriging, new_x, new_y)
    if new_x in k.x
        println("Adding a sample that already exists, cannot build Kriging.")
        return
    end
    if (length(new_x) == 1 && length(new_x[1]) == 1) ||
       (length(new_x) > 1 && length(new_x[1]) == 1 && length(k.theta) > 1)
        push!(k.x, new_x)
        push!(k.y, new_y)
    else
        append!(k.x, new_x)
        append!(k.y, new_y)
    end
    k.mu, k.b, k.sigma, k.inverse_of_R = _calc_kriging_coeffs(k.x, k.y, k.p, k.theta)
    nothing
end
