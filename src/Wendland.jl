"""
    Wendland(x, y, lb, ub; eps = 1.0, maxiters = 300, tol = 1.0e-6)

Compactly supported Wendland radial basis surrogate, using the C² kernel
(smoothness `k = 1`). A sample point's kernel vanishes at input distance
`1 / eps` and beyond, so `eps` is the reciprocal of the support radius.

`Wendland` solves a sparse interpolation system with conjugate gradients, and
warns if that solve does not converge within `maxiters`. The fitted surrogate
is callable as `wendland(x_new)` and can be updated with
`update!(wendland, x_new, y_new)`.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `coeff`: fitted interpolation coefficients.
  - `maxiters`: maximum number of conjugate-gradient iterations.
  - `tol`: relative tolerance used by conjugate gradients.
  - `eps`: reciprocal of the kernel support radius.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`. Responses must be scalars; vector-valued
    responses are not supported.
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

@inline _l(s, k) = s ÷ 2 + k + 1

function _wendland(x, eps)
    nx = norm(x)
    # `norm` of a zero vector evaluates 0 / 0 when rescaling, so at a sample
    # point it hands back NaN under AD. The kernel peaks at zero distance;
    # without this the point's own term is dropped there.
    r = isnan(nx) ? zero(nx) : eps * nx
    if r > 1
        return zero(r)
    end
    # Only k = 1 is supported at the moment; k = 1/2, 3/2 and 5/2 could be
    # added as further kernels.
    powerTerm = _l(length(x), 1) + 1
    return (one(r) - r)^powerTerm * (powerTerm * r + one(r))
end

function _calc_coeffs_wend(x, y, eps, maxiters, tol)
    n = length(x)
    # `float` so that integer or rational sample coordinates still give a
    # matrix able to hold the kernel values, and a `reltol` that converts.
    W = ExtendableSparseMatrix{float(eltype(x[1])), Int}(n, n)
    @inbounds for i in 1:n
        for j in i:n #wendland is symmetric
            W[i, j] = _wendland(x[i] .- x[j], eps)
        end
    end
    U = Symmetric(W, :U)
    # `cg` holds its numeric parameters at one type, so a Float64 `tol` against
    # a Float32 system is a MethodError rather than a promotion.
    reltol = convert(real(eltype(U)), tol)
    coeff, hist = cg(U, y, maxiter = maxiters, reltol = reltol, log = true)
    if !hist.isconverged
        # Not `maxlog`-limited: that state is process-wide, so the warning
        # would depend on what ran earlier in the session.
        @warn "Wendland conjugate-gradient solve did not converge in $maxiters iterations (relative tolerance $tol); the surrogate may be inaccurate. Consider raising maxiters or lowering eps."
    end
    return coeff
end

function Wendland(x, y, lb, ub; eps = 1.0, maxiters = 300, tol = 1.0e-6)
    c = _calc_coeffs_wend(x, y, eps, maxiters, tol)
    return Wendland(x, y, lb, ub, c, maxiters, tol, eps)
end

function (wend::Wendland)(val)
    _check_dimension(wend, val)
    # A row-matrix query would broadcast into a d x d outer difference.
    point = _as_point(val)
    return sum(
        wend.coeff[j] * _wendland(point .- wend.x[j], wend.eps)
            for j in eachindex(wend.coeff)
    )
end

function SurrogatesBase.update!(wend::Wendland, new_x, new_y)
    wend.x, wend.y = _append_samples(wend.x, wend.y, new_x, new_y)
    wend.coeff = _calc_coeffs_wend(wend.x, wend.y, wend.eps, wend.maxiters, wend.tol)
    return nothing
end
