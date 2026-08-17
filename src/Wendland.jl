"""
    Wendland(x, y, lb, ub; eps = 1.0, maxiters = 300, tol = 1.0e-6)

Compactly supported Wendland radial basis surrogate using the C² Wendland
kernel (smoothness `k = 1`). The kernel of a sample point vanishes at input
distance `1 / eps` and beyond, so `eps` sets the reciprocal of the support
radius.

`Wendland` solves a sparse interpolation system with conjugate gradients; a
warning is emitted if the solve does not converge within `maxiters`
iterations. The fitted surrogate is callable as `wendland(x_new)` and can be
updated with `update!(wendland, x_new, y_new)`.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `coeff`: fitted interpolation coefficients.
  - `maxiters`: maximum number of conjugate-gradient iterations.
  - `tol`: relative tolerance used by conjugate gradients.
  - `eps`: compact-support scaling parameter.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`. Responses must be scalars; vector-valued responses are not supported.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `eps`: reciprocal of the kernel support radius; a sample point influences
    predictions within input distance `1 / eps` of it.
  - `maxiters`: maximum iterations for the coefficient solve.
  - `tol`: relative tolerance for the coefficient solve.

# Returns

A `Wendland` surrogate satisfying the generic surrogate interface.
"""
mutable struct Wendland{X, Y, L, U, C, I, T, E} <: AbstractDeterministicSurrogate
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
    # `cg` builds a `CGIterable` whose numeric parameters must share one type, so
    # a Float64 `tol` against a Float32 system is a MethodError rather than a
    # promotion.
    reltol = convert(real(eltype(U)), tol)
    coeff, hist = cg(U, y, maxiter = maxiters, reltol = reltol, log = true)
    if !hist.isconverged
        # Not `maxlog`-limited: that state is process-wide, which would make the
        # warning depend on what ran earlier in the session.
        @warn "Wendland conjugate-gradient solve did not converge within $maxiters iterations (relative tolerance $tol); the surrogate may be inaccurate. Consider increasing maxiters or decreasing eps."
    end
    return coeff
end

function Wendland(x, y, lb, ub; eps = 1.0, maxiters = 300, tol = 1.0e-6)
    c = _calc_coeffs_wend(x, y, eps, maxiters, tol)
    return Wendland(x, y, lb, ub, c, maxiters, tol, eps)
end

function (wend::Wendland)(val)
    _check_dimension(wend, val)

    return sum(
        wend.coeff[j] * _wendland(val .- wend.x[j], wend.eps)
            for j in 1:length(wend.coeff)
    )
end

function SurrogatesBase.update!(wend::Wendland, new_x, new_y)
    wend.x, wend.y = _append_samples(wend.x, wend.y, new_x, new_y)
    wend.coeff = _calc_coeffs_wend(wend.x, wend.y, wend.eps, wend.maxiters, wend.tol)
    return nothing
end
