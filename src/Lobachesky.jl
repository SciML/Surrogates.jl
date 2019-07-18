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
    #phi = f_n*(alpha*(x - x_j))
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

function add_point!(loba::LobacheskySurrogate,x_new,y_new)
    if length(loba.x[1]) == 1
        append!(loba.x,x_new)
        append!(loba.y,y_new)
        loba.coeff = _calc_loba_coeff1D(loba.x,loba.y,loba.alpha,loba.n)
        #1D
    else
        #ND
        #TODO
    end
end
