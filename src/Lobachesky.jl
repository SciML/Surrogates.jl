mutable struct LobacheskySurrogate{X,Y,A,N,L,U,C} <: AbstractSurrogate
    x::X
    y::Y
    alpha::A
    n::N
    lb::L
    ub::U
    coeff::C
end

function phi_nj1D(point,x,alpha,n)
    val = zero(eltype(x[1]))
    for k = 0:n
        a = sqrt(n/3)*alpha*(point-x) + (n - 2*k)
        if a > 0
            val = val +(-1)^k*binomial(n,k)*a^(n-1)
        end
    end
    val *= sqrt(n/3)/(2^n*factorial(n-1))
    return val
end

function _calc_loba_coeff1D(x,y,alpha,n)
    dim = length(x)
    D = zeros(eltype(x[1]), dim, dim)
    for i = 1:dim
        for j = 1:dim
            D[i,j] = phi_nj1D(x[i],x[j],alpha,n)
        end
    end
    Sym = Symmetric(D,:U)
    return Sym\y
end
"""
Lobachesky interpolation, suggested parameters: 0 <= alpha <= 4, n must be even.
"""
function LobacheskySurrogate(x,y,alpha,n::Int,lb::Number,ub::Number)
    if alpha > 4 || alpha < 0
        error("Alpha must be between 0 and 4")
    end
    if n % 2 != 0
        error("Parameter n must be even")
    end
    coeff = _calc_loba_coeff1D(x,y,alpha,n)
    LobacheskySurrogate(x,y,alpha,n,lb,ub,coeff)
end

function (loba::LobacheskySurrogate)(val::Number)
    res = zero(eltype(loba.y[1]))
    for j = 1:length(loba.x)
        res = res + loba.coeff[j]*phi_nj1D(val,loba.x[j],loba.alpha,loba.n)
    end
    return res
end

function phi_njND(point,x,alpha,n)
    s = 1.0
    d = length(x)
    for h = 1:d
        a = phi_nj1D(point[h],x[h],alpha,n)
        s = s*a
    end
    return s
end

function _calc_loba_coeffND(x,y,alpha,n)
    dim = length(x)
    D = zeros(eltype(x[1]), dim, dim)
    for i = 1:dim
        for j = 1:dim
            D[i,j] = phi_njND(x[i],x[j],alpha,n)
        end
    end
    Sym = Symmetric(D,:U)
    return Sym\y
end
"""
LobacheskySurrogate(x,y,alpha,n::Int,lb,ub)

Build the Lobachesky surrogate with parameters alpha and n.
"""
function LobacheskySurrogate(x,y,alpha,n::Int,lb,ub)
    if alpha > 4 || alpha < 0
        error("Alpha must be between 0 and 4")
    end
    if n % 2 != 0
        error("Parameter n must be even")
    end
    coeff = _calc_loba_coeffND(x,y,alpha,n)
    LobacheskySurrogate(x,y,alpha,n,lb,ub,coeff)
end

function (loba::LobacheskySurrogate)(val)
    val = collect(val)
    res = zero(eltype(loba.y[1]))
    for j = 1:length(loba.x)
        res = res + loba.coeff[j]*phi_njND(val,loba.x[j],loba.alpha,loba.n)
    end
    return res
end

function add_point!(loba::LobacheskySurrogate,x_new,y_new)
    if length(loba.x[1]) == 1
        #1D
        append!(loba.x,x_new)
        append!(loba.y,y_new)
        loba.coeff = _calc_loba_coeff1D(loba.x,loba.y,loba.alpha,loba.n)
    else
        #ND
        loba.x = vcat(loba.x,x_new)
        loba.y = vcat(loba.y,y_new)
        loba.coeff = _calc_loba_coeffND(loba.x,loba.y,loba.alpha,loba.n)
    end
    nothing
end

#Lobachesky integrals
function _phi_int(point,n)
    res = zero(eltype(point))
    for k = 0:n
        c = sqrt(n/3)*point + (n - 2*k)
        if c > 0
            res = res + (-1)^k*binomial(n,k)*c^n
        end
    end
    res *= 1/(2^n*factorial(n))
end

function lobachesky_integral(loba::LobacheskySurrogate,lb::Number,ub::Number)
    val = zero(eltype(loba.y[1]))
    n = length(loba.x)
    for i = 1:n
        a = loba.alpha*(ub - loba.x[i])
        b = loba.alpha*(lb - loba.x[i])
        int = 1/loba.alpha*(_phi_int(a,loba.n) - _phi_int(b,loba.n))
        val = val + loba.coeff[i]*int
    end
    return val
end

"""
lobachesky_integral(loba::LobacheskySurrogate,lb,ub)

Calculates the integral of the Lobachesky surrogate, which has a closed form.

"""
function lobachesky_integral(loba::LobacheskySurrogate,lb,ub)
    d = length(lb)
    val = zero(eltype(loba.y[1]))
    for j = 1:length(loba.x)
        I = 1.0
        for i = 1:d
            upper = loba.alpha*(ub[i] - loba.x[j][i])
            lower = loba.alpha*(lb[i] - loba.x[j][i])
            I *= 1/loba.alpha*(_phi_int(upper,loba.n) - _phi_int(lower,loba.n))
        end
        val = val + loba.coeff[j]*I
    end
    return val
end
