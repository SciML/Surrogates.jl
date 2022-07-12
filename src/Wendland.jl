using IterativeSolvers
using ExtendableSparse
using LinearAlgebra

mutable struct Wendland{X, Y, L, U, C, I, T, E} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    maxiters::I
    tol::T
    eps::E
end

@inline _l(s, k) = floor(s / 2) + k + 1

function _wendland(x, eps)
    r = eps * norm(x)
    val = (1.0 - r)
    if val >= 0
        dim = length(x)
        #at the moment only k = 1 is supported, but we could also support
        # missing wendland (k=1/2,k=3/2,k=5/2), and different k's.
        ell = _l(dim, 1)
        powerTerm = ell + 1.0
        val = (1.0 - r)
        return val^powerTerm * (powerTerm * r + 1.0)
    else
        return zero(eltype(x[1]))
    end
end

function _calc_coeffs_wend(x, y, eps, maxiters, tol)
    n = length(x)
    W = ExtendableSparseMatrix{eltype(x[1]), Int}(n, n)
    @inbounds for i in 1:n
        k = i #wendland is symmetric
        for j in k:n
            W[i, j] = _wendland(x[i] .- x[j], eps)
        end
    end
    U = Symmetric(W, :U)
    return cg(U, y, maxiter = maxiters, reltol = tol)
end

function Wendland(x, y, lb, ub; eps = 1.0, maxiters = 300, tol = 1e-6)
    c = _calc_coeffs_wend(x, y, eps, maxiters, tol)
    return Wendland(x, y, lb, ub, c, maxiters, tol, eps)
end

function (wend::Wendland)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(wend, val)

    return sum(wend.coeff[j] * _wendland(val, wend.eps) for j in 1:length(wend.coeff))
end

function add_point!(wend::Wendland, new_x, new_y)
    if (length(new_x) == 1 && length(new_x[1]) == 1) ||
       (length(new_x) > 1 && length(new_x[1]) == 1 && length(wend.lb) > 1)
        push!(wend.x, new_x)
        push!(wend.y, new_y)
    else
        append!(wend.x, new_x)
        append!(wend.y, new_y)
    end
    wend.coeff = _calc_coeffs_wend(wend.x, wend.y, wend.eps, wend.maxiters, wend.tol)
    nothing
end
