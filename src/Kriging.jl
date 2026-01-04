#=
One-dimensional Kriging method, following these papers:
"Efficient Global Optimization of Expensive Black Box Functions" and
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
both by DONALD R. JONES
=#
mutable struct Kriging{X, Y, L, U, P, T, M, B, S, R} <: AbstractDeterministicSurrogate
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

    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(k, val)

    n = length(k.x)
    d = length(val)

    return k.mu +
        sum(
        k.b[i] *
            exp(-sum(k.theta[j] * norm(val[j] - k.x[i][j])^k.p[j] for j in 1:d))
            for i in 1:n
    )
end

"""
    Returns sqrt of expected mean_squared_error error at the point.
"""
function std_error_at_point(k::Kriging, val)

    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(k, val)

    n = length(k.x)
    d = length(k.x[1])
    r = zeros(eltype(k.x[1]), n, 1)
    r = [
        let
                sum = zero(eltype(k.x[1]))
                for l in 1:d
                    sum = sum + k.theta[l] * norm(val[l] - k.x[i][l])^(k.p[l])
            end
                exp(-sum)
        end
            for i in 1:n
    ]

    one = ones(eltype(k.x[1]), n, 1)
    one_t = one'
    a = r' * k.inverse_of_R * r
    b = one_t * k.inverse_of_R * one

    mean_squared_error = k.sigma * (1 - a[1] + (1 - a[1])^2 / b[1])
    return sqrt(abs(mean_squared_error))
end

"""
Gives the current estimate for 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(k, val)
    n = length(k.x)
    return k.mu + sum(k.b[i] * exp(-sum(k.theta * abs(val - k.x[i])^k.p)) for i in 1:n)
end

"""
    Returns sqrt of expected mean_squared_error error at the point.
"""
function std_error_at_point(k::Kriging, val::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(k, val)
    n = length(k.x)
    r = [exp(-k.theta * abs(val - k.x[i])^k.p) for i in 1:n]
    one = ones(eltype(k.x), n)
    one_t = one'
    a = r' * k.inverse_of_R * r
    b = one_t * k.inverse_of_R * one

    mean_squared_error = k.sigma * (1 - a[1] + (1 - a[1])^2 / b[1])
    return sqrt(abs(mean_squared_error))
end

"""
    Kriging(x, y, lb::Number, ub::Number; p::Number=2.0, theta::Number = 0.5/var(x))

Constructor for type Kriging.

#Arguments:
-(x,y): sampled points
-p: value between 0 and 2 modelling the
smoothness of the function being approximated, 0-> rough  2-> C^infinity

  - theta: value > 0 modeling how much the function is changing in the i-th variable.
"""
function Kriging(
        x, y, lb::Number, ub::Number; p = 2.0,
        theta = 0.5 / max(1.0e-6 * abs(ub - lb), std(x))^p
    )
    if length(x) != length(unique(x))
        println("There exists a repetition in the samples, cannot build Kriging.")
        return
    end

    if p > 2.0 || p < 0.0
        throw(ArgumentError("Hyperparameter p must be between 0 and 2! Got: $p."))
    end

    if theta ≤ 0
        throw(ArgumentError("Hyperparameter theta must be positive! Got: $theta"))
    end

    mu, b, sigma, inverse_of_R = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(x, y, lb, ub, p, theta, mu, b, sigma, inverse_of_R)
end

function _calc_kriging_coeffs(x, y, p::Number, theta::Number)
    n = length(x)

    R = [exp(-theta * abs(x[i] - x[j])^p) for i in 1:n, j in 1:n]

    # Estimate nugget based on maximum allowed condition number
    # This regularizes R to allow for points being close to each other without R becoming
    # singular, at the cost of slightly relaxing the interpolation condition
    # Derived from "An analytic comparison of regularization methods for Gaussian Processes"
    # by Mohammadi et al (https://arxiv.org/pdf/1602.00853.pdf)
    # Use Symmetric wrapper to ensure real eigenvalues for the symmetric correlation matrix
    λ = eigvals(Symmetric(R))
    λmax = λ[end]
    λmin = λ[1]

    κmax = 1.0e8
    λdiff = λmax - κmax * λmin
    if λdiff ≥ 0
        nugget = λdiff / (κmax - 1)
    else
        nugget = zero(λdiff)
    end

    one = ones(eltype(x[1]), n)
    one_t = one'

    R = R + Diagonal(nugget * one)

    inverse_of_R = inv(R)

    mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)
    b = inverse_of_R * (y - one * mu)
    sigma = ((y - one * mu)' * b) / n
    return mu[1], b, sigma[1], inverse_of_R
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
function Kriging(
        x, y, lb, ub; p = 2.0 .* collect(one.(x[1])),
        theta = [
            0.5 / max(1.0e-6 * norm(ub .- lb), std(x_i[i] for x_i in x))^p[i]
                for i in 1:length(x[1])
        ]
    )
    if length(x) != length(unique(x))
        println("There exists a repetition in the samples, cannot build Kriging.")
        return
    end

    for i in 1:length(x[1])
        if p[i] > 2.0 || p[i] < 0.0
            throw(ArgumentError("All p must be between 0 and 2! Got: $p."))
        end

        if theta[i] ≤ 0.0
            throw(ArgumentError("All theta must be positive! Got: $theta."))
        end
    end

    mu, b, sigma, inverse_of_R = _calc_kriging_coeffs(x, y, p, theta)
    return Kriging(x, y, lb, ub, p, theta, mu, b, sigma, inverse_of_R)
end

function _calc_kriging_coeffs(x, y, p, theta)
    n = length(x)
    d = length(x[1])

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

    # Estimate nugget based on maximum allowed condition number
    # This regularizes R to allow for points being close to each other without R becoming
    # singular, at the cost of slightly relaxing the interpolation condition
    # Derived from "An analytic comparison of regularization methods for Gaussian Processes"
    # by Mohammadi et al (https://arxiv.org/pdf/1602.00853.pdf)
    # Use Symmetric wrapper to ensure real eigenvalues for the symmetric correlation matrix
    λ = eigvals(Symmetric(R))

    λmax = λ[end]
    λmin = λ[1]

    κmax = 1.0e8
    λdiff = λmax - κmax * λmin
    if λdiff ≥ 0
        nugget = λdiff / (κmax - 1)
    else
        nugget = zero(λdiff)
    end

    one = ones(eltype(x[1]), n)
    one_t = one'

    R = R + Diagonal(nugget * one[:, 1])
    inverse_of_R = inv(R)

    mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)

    y_minus_1μ = y - one * mu

    b = inverse_of_R * y_minus_1μ

    sigma = (y_minus_1μ' * b) / n

    return mu[1], b, sigma[1], inverse_of_R
end

"""
    update!(k::Kriging,new_x,new_y)

Adds the new point and its respective value to the sample points.
Warning: If you are just adding a single point, you have to wrap it with [].
Returns the updated Kriging model.
"""
function SurrogatesBase.update!(k::Kriging, new_x, new_y)
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
    return nothing
end
