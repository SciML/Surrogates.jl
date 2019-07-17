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

"""
Lobachesky interpolation, suggested parameters: 0 <= alpha <= 4, n even.
"""
function LobacheskySurrogate(x,y,alpha,n::Int,lb::Number,ub::Number)
    if n % 2 != 0
        error("Parameter n must be even")
    end
    dim = length(x)
    D = zeros(eltype(x[1]), dim, dim)
    for i = 1:dim
        for j = 1:dim
            D[i,j] = phi_nj1D(x[i],x[j],alpha,n)
        end
    end
    Sym = Symmetric(D,:U)
    coeff = Sym\y
    #calcolo coeff
    LobacheskySurrogate(x,y,alpha,n,lb,ub,coeff)
end

function (loba::LobacheskySurrogate)(val::Number)
    res = zero(eltype(loba.y[1]))
    for j = 1:length(loba.x)
        res = res + loba.coeff[j]*phi_nj1D(val,loba.x[j],loba.alpha,loba.n)
    end
    return res
end
