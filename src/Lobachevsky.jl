using ExtendableSparse
mutable struct LobachevskySurrogate{X, Y, A, N, L, U, C, S} <: AbstractSurrogate
    x::X
    y::Y
    alpha::A
    n::N
    lb::L
    ub::U
    coeff::C
    sparse::S
end

function phi_nj1D(point, x, alpha, n)
    val = false * x[1]
    for l in 0:n
        a = sqrt(n / 3) * alpha * (point - x) + (n - 2 * l)
        if a > 0
            if l % 2 == 0
                val += binomial(n, l) * a^(n - 1)
            else
                val -= binomial(n, l) * a^(n - 1)
            end
        end
    end
    val *= sqrt(n / 3) / (2^n * factorial(n - 1))
    return val
end

function _calc_loba_coeff1D(x, y, alpha, n, sparse)
    dim = length(x)
    if sparse
        D = ExtendableSparseMatrix{eltype(x), Int}(dim, dim)
    else
        D = zeros(eltype(x[1]), dim, dim)
    end
    for i in 1:dim
        for j in 1:dim
            D[i, j] = phi_nj1D(x[i], x[j], alpha, n)
        end
    end
    Sym = Symmetric(D, :U)
    return Sym \ y
end
"""
Lobachevsky interpolation, suggested parameters: 0 <= alpha <= 4, n must be even.
"""
function LobachevskySurrogate(
        x, y, lb::Number, ub::Number; alpha::Number = 1.0, n::Int = 4,
        sparse = false)
    if alpha > 4 || alpha < 0
        error("Alpha must be between 0 and 4")
    end
    if n % 2 != 0
        error("Parameter n must be even")
    end
    coeff = _calc_loba_coeff1D(x, y, alpha, n, sparse)
    LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(loba, val)

    return sum(loba.coeff[j] * phi_nj1D(val, loba.x[j], loba.alpha, loba.n)
    for j in 1:length(loba.x))
end

function phi_njND(point, x, alpha, n)
    return prod(phi_nj1D(point[h], x[h], alpha[h], n) for h in 1:length(x))
end

function _calc_loba_coeffND(x, y, alpha, n, sparse)
    dim = length(x)
    if sparse
        D = ExtendableSparseMatrix{eltype(x[1]), Int}(dim, dim)
    else
        D = zeros(eltype(x[1]), dim, dim)
    end
    for i in 1:dim
        for j in 1:dim
            D[i, j] = phi_njND(x[i], x[j], alpha, n)
        end
    end
    Sym = Symmetric(D, :U)
    return Sym \ y
end
"""
LobachevskySurrogate(x,y,alpha,n::Int,lb,ub,sparse = false)

Build the Lobachevsky surrogate with parameters alpha and n.
"""
function LobachevskySurrogate(x, y, lb, ub; alpha = collect(one.(x[1])), n::Int = 4,
        sparse = false)
    if n % 2 != 0
        error("Parameter n must be even")
    end
    coeff = _calc_loba_coeffND(x, y, alpha, n, sparse)
    LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(loba, val)
    return sum(loba.coeff[j] * phi_njND(val, loba.x[j], loba.alpha, loba.n)
    for j in 1:length(loba.x))
end

function add_point!(loba::LobachevskySurrogate, x_new, y_new)
    if length(loba.x[1]) == 1
        #1D
        append!(loba.x, x_new)
        append!(loba.y, y_new)
        loba.coeff = _calc_loba_coeff1D(loba.x, loba.y, loba.alpha, loba.n, loba.sparse)
    else
        #ND
        loba.x = vcat(loba.x, x_new)
        loba.y = vcat(loba.y, y_new)
        loba.coeff = _calc_loba_coeffND(loba.x, loba.y, loba.alpha, loba.n, loba.sparse)
    end
    nothing
end

#Lobachevsky integrals
function _phi_int(point, n)
    res = zero(eltype(point))
    for k in 0:n
        c = sqrt(n / 3) * point + (n - 2 * k)
        if c > 0
            res = res + (-1)^k * binomial(n, k) * c^n
        end
    end
    res *= 1 / (2^n * factorial(n))
end

function lobachevsky_integral(loba::LobachevskySurrogate, lb::Number, ub::Number)
    val = zero(eltype(loba.y[1]))
    n = length(loba.x)
    for i in 1:n
        a = loba.alpha * (ub - loba.x[i])
        b = loba.alpha * (lb - loba.x[i])
        int = 1 / loba.alpha * (_phi_int(a, loba.n) - _phi_int(b, loba.n))
        val = val + loba.coeff[i] * int
    end
    return val
end

"""
lobachevsky_integral(loba::LobachevskySurrogate,lb,ub)

Calculates the integral of the Lobachevsky surrogate, which has a closed form.
"""
function lobachevsky_integral(loba::LobachevskySurrogate, lb, ub)
    d = length(lb)
    val = zero(eltype(loba.y[1]))
    for j in 1:length(loba.x)
        I = 1.0
        for i in 1:d
            upper = loba.alpha[i] * (ub[i] - loba.x[j][i])
            lower = loba.alpha[i] * (lb[i] - loba.x[j][i])
            I *= 1 / loba.alpha[i] * (_phi_int(upper, loba.n) - _phi_int(lower, loba.n))
        end
        val = val + loba.coeff[j] * I
    end
    return val
end

"""
lobachevsky_integrate_dimension(loba::LobachevskySurrogate,lb,ub,dimension)

Integrating the surrogate on selected dimension dim
"""
function lobachevsky_integrate_dimension(loba::LobachevskySurrogate, lb, ub, dim::Int)
    gamma_d = zero(loba.coeff[1])
    n = length(loba.x)
    for i in 1:n
        a = loba.alpha[dim] * (ub[dim] - loba.x[i][dim])
        b = loba.alpha[dim] * (lb[dim] - loba.x[i][dim])
        int = 1 / loba.alpha[dim] * (_phi_int(a, loba.n) - _phi_int(b, loba.n))
        gamma_d = gamma_d + loba.coeff[i] * int
    end
    new_coeff = loba.coeff .* gamma_d

    if length(lb) == 2
        # Integrating one dimension -> 1D
        new_x = zeros(eltype(loba.x[1][1]), n)
        for i in 1:n
            new_x[i] = deleteat!(collect(loba.x[i]), dim)[1]
        end
    else
        dummy = loba.x[1]
        dummy = deleteat!(collect(dummy), dim)
        new_x = typeof(Tuple(dummy))[]
        for i in 1:n
            push!(new_x, Tuple(deleteat!(collect(loba.x[i]), dim)))
        end
    end
    new_lb = deleteat!(lb, dim)
    new_ub = deleteat!(ub, dim)
    new_loba = deleteat!(loba.alpha, dim)
    return LobachevskySurrogate(new_x, loba.y, loba.alpha, loba.n, new_lb, new_ub,
        new_coeff, loba.sparse)
end
