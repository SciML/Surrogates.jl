using LinearAlgebra

mutable struct GEK{X, Y, L, U, P, T, M, B, S, R} <: AbstractSurrogate
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

function _calc_gek_coeffs(x, y, p, theta)
    nd = length(y)
    n = length(x)
    d = length(x[1])
    R = similar(zeros(eltype(x[1]), nd, nd))

    @inbounds for i in 1:n, j in 1:n
        R[i, j] = prod(exp.(-theta .* (x[i] .- x[j])))
    end

    jr = 1
    @inbounds for i in 1:n, j in (n + 1):d:nd
        for l in 1:d
            R[i, j + l - 1] = +2 * theta[l] * (x[i][l] - x[jr][l]) * R[i, jr]
        end
        jr += 1
    end

    ir = 1
    @inbounds for j in 1:n, i in (n + 1):d:nd
        for l in 1:d
            R[i + l - 1, j] = -2 * theta[l] * (x[ir][l] - x[j][l]) * R[ir, j]
        end
        ir += 1
    end

    ir = 1
    @inbounds for i in (n + 1):d:nd, j in (n + 1):d:nd
        jr = 1
        for l in 1:d, k in 1:d
            R[i + l - 1, j + k - 1] = -4 * theta[l] * theta[k] *
                                      (x[ir][l] - x[jr][l]) *
                                      (x[ir][k] - x[jr][k]) * R[ir, jr]
        end
        jr += 1
    end

    one = ones(eltype(x[1]), nd, 1)
    @inbounds for i in (n + 1):nd
        one[i] = zero(eltype(x[1]))
    end
    one_t = one'
    inverse_of_R = inv(R)
    mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)
    b = inverse_of_R * (y - one * mu)
    sigma = ((y - one * mu)' * inverse_of_R * (y - one * mu)) / n
    return mu[1], b, sigma[1], inverse_of_R
end

function std_error_at_point(k::GEK, val)
    _check_dimension(k, val)
    nd1 = length(k.y)
    n = length(k.x)
    d = length(k.x[1])
    r = similar(zeros(eltype(k.x[1]), nd1, 1))
    @inbounds for i in 1:n
        sum = zero(eltype(k.x[1]))
        for l in 1:d
            sum += k.theta[l] * norm(val[l] - k.x[i][l])^k.p[l]
        end
        r[i] = exp(-sum)
    end
    one = ones(eltype(k.x[1]), nd1, 1)
    @inbounds for i in (n + 1):nd1
        one[i] = zero(eltype(k.x[1]))
    end
    one_t = one'
    a = r' * k.inverse_of_R * r
    a = a[1]
    b = one_t * k.inverse_of_R * one
    b = b[1]
    mean_squared_error = k.sigma * (1 - a + (1 - a)^2 / (b))
    return sqrt(abs(mean_squared_error))
end

function (k::GEK)(val)
    _check_dimension(k, val)
    d = length(val)
    n = length(k.x)
    phi = z -> exp(-abs(z)^k.p)
    prediction = zero(eltype(k.x[1]))
    @inbounds for i in 1:n
        sum = zero(eltype(k.x[1]))
        for j in 1:d
            sum += k.theta[j] * norm(val[j] - k.x[i][j])^k.p[j]
        end
        prediction += k.b[i] * phi(sum)
    end
    return k.mu + prediction
end

function GEK(x, y, lb, ub; p = collect(one.(x[1])), theta = collect(one.(x[1])))
    if length(x) != length(unique(x))
        println("There exists a repetition in the samples, cannot build Kriging.")
        return
    end
    mu, b, sigma, inverse_of_R = _calc_gek_coeffs(x, y, p, theta)
    return GEK(x, y, lb, ub, p, theta, mu, b, sigma, inverse_of_R)
end

function add_point!(k::GEK, new_x, new_y)
    if new_x in k.x
        println("Adding a sample that already exists, cannot build Kriging.")
        return
    end
    n = length(k.x)
    if length(new_x) == 1 || length(new_x[1]) == 1
        k.x = insert!(k.x, n + 1, new_x)
        k.y = insert!(k.y, n + 1, new_y)
    else
        k.x = insert!(k.x, n + 1, new_x)
        k.y = insert!(k.y, n + 1, new_y)
    end
    k.mu, k.b, k.sigma, k.inverse_of_R = _calc_gek_coeffs(k.x, k.y, k.p, k.theta)
    nothing
end

function _check_dimension(k::GEK, val)
    d = length(val)
    if length(k.x[1]) != d
        throw(DimensionMismatch("Input dimension does not match surrogate dimension"))
    end
end
