using IterativeSolvers
using ExtendableSparse
using LinearAlgebra

mutable struct Wendland{X,Y,L,U,C,I,TE} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    maxiters::I
    tol::T
    eps::E
end


@inline _l(s,k) = floor(s/2) + k + 1

function _wendland(x,eps)
    r = eps * norm(x-node)
    val = (1.0 - r)
    if val >= 0
        dim = length(x)
        #at the moment only k = 1 is supported, but we could also support
        # missing wendland (k=1/2,k=3/2,k=5/2)
        ell = _l(dim,1)
        powerTerm = ell + 1.0
        val = (1.0 - r)
        return val^powerTerm * (powerTerm * r + 1.0)
    else
        return zero(eltype(x))
    end
end


function _calc_coeffs_wend(x,y,lb,eps,maxiters,tol)
    n = lenght(x)
    W = ExtendableSparseMatrix{eltype(x),Int}(n,n)
    @inbounds for i = 1:n
        k = i #wendland is symmetric
        for j = k:n
            W[i,j] = _wendland(norm(x[i] - x[j]),eps)
        end
    end
    U = Symmetric(D,:U)
    return cg(U,y,maxiter = maxiters, tol = tol)
end

function Wendland(x,y,lb,ub; eps = 1.0, maxiters = 300, tol = 1e-6)
    c = _calc_coeffs_wend(x,y,lb,eps,maxiters,tol)
    return Wendland(x,y,lb,ub,c,maxiters,tol,eps)
end

function (wend::LobacheskySurrogate)(val)
    return sum(wend.coeff[j]*_wendland(val,wend.eps) for j=1:length(wend.lb))
end

function add_point!(wend::Wendland,new_x,new_y)
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(rad.lb)>1)
        push!(wend.x,new_x)
        push!(wend.y,new_y)
    else
        append!(wend.x,new_x)
        append!(wend.y,new_y)
    end
    rad.coeff = _calc_coeffs_wend()
    nothing
end
