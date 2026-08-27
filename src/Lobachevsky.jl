"""
    LobachevskySurrogate(x, y, lb, ub; alpha = 1.0, n = 4, sparse = false)

Construct a univariate or multivariate Lobachevsky-spline interpolant. The
kernel order `n` must be even; multidimensional models use one `alpha` scale per
input dimension.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: responses corresponding to `x`.
  - `alpha`: scalar or per-dimension kernel scale.
  - `n`: even Lobachevsky kernel order.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `coeff`: fitted interpolation coefficients, one per sample for scalar
    responses and an `n x m` matrix with one column per output for
    vector-valued ones.
  - `sparse`: whether coefficient construction uses a sparse matrix.

# Arguments

  - `x`: training inputs.
  - `y`: training responses, with one response per input. Responses may be
    scalars or equal-length vectors; vector-valued responses are fitted one
    output at a time and the surrogate then returns a vector.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `alpha = 1.0`: kernel scale, which must lie in `(0, 4]`. A scale of zero
    makes every kernel value identical and the interpolation system singular.
    For multidimensional inputs, supply one scale per input dimension; the
    default is a vector of ones matching one training point.
  - `n::Int = 4`: even, positive kernel order, at most 20 (`factorial(n)`
    overflows `Int64` beyond that).
  - `sparse::Bool = false`: use sparse coefficient construction.

# Returns

A callable `LobachevskySurrogate` supporting
`update!(surrogate, x_new, y_new)` and the closed-form integration helpers
[`lobachevsky_integral`](@ref) and [`lobachevsky_integrate_dimension`](@ref).

# Example

```julia
using Surrogates

x = [0.0, 1.0, 2.0]
y = sin.(x)
surrogate = LobachevskySurrogate(x, y, 0.0, 2.0; alpha = 1.0, n = 4)
surrogate(0.5)
```
"""
mutable struct LobachevskySurrogate{X, Y, A, N, L, U, C, S} <:
    AbstractDeterministicSurrogate
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
    # At the element type, so a Float32 sample is not promoted by the constant.
    c = sqrt(oftype(float(val), n) / 3)
    for l in 0:n
        a = c * alpha * (point - x) + (n - 2 * l)
        if a > 0
            if l % 2 == 0
                val += binomial(n, l) * a^(n - 1)
            else
                val -= binomial(n, l) * a^(n - 1)
            end
        end
    end
    val *= c / (2^n * factorial(n - 1))
    return val
end

# The interpolant is linear in the responses and the kernel matrix does not
# involve them, so vector-valued `y` is one solve per output against the same
# matrix and `coeff` becomes `n x m`. Contraction dispatches on that shape: the
# scalar path stays allocation free, the multi-output one accumulates a row per
# sample. `weight(j)` is the scalar multiplying sample `j`, so the same pair
# serves evaluation and the closed-form integrals.
_loba_combine(coeff::AbstractVector, weight, n) = sum(coeff[j] * weight(j) for j in 1:n)
_loba_combine(coeff::AbstractMatrix, weight, n) = sum(coeff[j, :] * weight(j) for j in 1:n)

function _calc_loba_coeff1D(x, y, alpha, n, sparse)
    dim = length(x)
    # `float` so integer samples still give a matrix able to hold the kernel.
    T = float(eltype(x[1]))
    D = sparse ? ExtendableSparseMatrix{T, Int}(dim, dim) : zeros(T, dim, dim)
    for i in 1:dim
        # The kernel depends only on the difference and is even in it, so only
        # the triangle `Symmetric` reads has to be filled.
        for j in i:dim
            D[i, j] = phi_nj1D(x[i], x[j], alpha, n)
        end
    end
    Sym = Symmetric(D, :U)
    return Sym \ _construct_y_matrix(y, first(y))
end

# The kernel evaluates `factorial(n - 1)` and `_phi_int` `factorial(n)`, and
# `factorial(::Int)` overflows above 20.
function _check_lobachevsky_n(n)
    if n <= 0 || n % 2 != 0
        throw(ArgumentError("Kernel order n must be even and positive! Got: $n."))
    end
    if n > 20
        throw(ArgumentError("Kernel order n must be at most 20, as factorial(n) overflows Int64! Got: $n."))
    end
    return nothing
end

# A scale of zero collapses the kernel to a constant, leaving a rank-one
# system. `any` iterates a scalar as well as a vector.
function _check_lobachevsky_alpha(alpha)
    if any(a -> !(0 < a <= 4), alpha)
        throw(ArgumentError("Kernel scale alpha must be in (0, 4]! Got: $alpha."))
    end
    return nothing
end

function LobachevskySurrogate(
        x, y, lb::Number, ub::Number; alpha::Number = 1.0, n::Int = 4,
        sparse = false
    )
    _check_lobachevsky_alpha(alpha)
    _check_lobachevsky_n(n)
    coeff = _calc_loba_coeff1D(x, y, alpha, n, sparse)
    return LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val::Number)
    _check_dimension(loba, val)
    return _loba_combine(
        loba.coeff,
        j -> phi_nj1D(val, loba.x[j], loba.alpha, loba.n),
        length(loba.x)
    )
end

function phi_njND(point, x, alpha, n)
    return prod(phi_nj1D(point[h], x[h], alpha[h], n) for h in 1:length(x))
end

function _calc_loba_coeffND(x, y, alpha, n, sparse)
    dim = length(x)
    T = float(eltype(x[1]))
    D = sparse ? ExtendableSparseMatrix{T, Int}(dim, dim) : zeros(T, dim, dim)
    for i in 1:dim
        for j in i:dim
            D[i, j] = phi_njND(x[i], x[j], alpha, n)
        end
    end
    Sym = Symmetric(D, :U)
    return Sym \ _construct_y_matrix(y, first(y))
end
function LobachevskySurrogate(
        x, y, lb, ub; alpha = collect(one.(x[1])), n::Int = 4,
        sparse = false
    )
    # A scalar alpha would otherwise reach a BoundsError per dimension.
    d = length(x[1])
    if length(alpha) != d
        throw(ArgumentError("Expected one kernel scale alpha per input dimension, $d of them! Got: $(length(alpha))."))
    end
    _check_lobachevsky_alpha(alpha)
    _check_lobachevsky_n(n)
    coeff = _calc_loba_coeffND(x, y, alpha, n, sparse)
    return LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val)
    _check_dimension(loba, val)
    return _loba_combine(
        loba.coeff,
        j -> phi_njND(val, loba.x[j], loba.alpha, loba.n),
        length(loba.x)
    )
end

function SurrogatesBase.update!(loba::LobachevskySurrogate, x_new, y_new)
    loba.x, loba.y = _append_samples(loba.x, loba.y, x_new, y_new)
    loba.coeff = if first(loba.x) isa Number
        _calc_loba_coeff1D(loba.x, loba.y, loba.alpha, loba.n, loba.sparse)
    else
        _calc_loba_coeffND(loba.x, loba.y, loba.alpha, loba.n, loba.sparse)
    end
    return nothing
end

function _phi_int(point, n)
    res = zero(eltype(point))
    # As in `phi_nj1D`, at the element type.
    s = sqrt(oftype(float(res), n) / 3)
    for k in 0:n
        c = s * point + (n - 2 * k)
        if c > 0
            res = res + (-1)^k * binomial(n, k) * c^n
        end
    end
    return res / (2^n * factorial(n))
end

function lobachevsky_integral(loba::LobachevskySurrogate, lb::Number, ub::Number)
    function int(i)
        a = loba.alpha * (ub - loba.x[i])
        b = loba.alpha * (lb - loba.x[i])
        return 1 / loba.alpha * (_phi_int(a, loba.n) - _phi_int(b, loba.n))
    end
    return _loba_combine(loba.coeff, int, length(loba.x))
end

"""
lobachevsky_integral(loba::LobachevskySurrogate,lb,ub)

Calculates the integral of the Lobachevsky surrogate, which has a closed form.
"""
function lobachevsky_integral(loba::LobachevskySurrogate, lb, ub)
    d = length(lb)
    # The kernel is a tensor product, so the integral over a box factorizes.
    function box(j)
        I = one(float(eltype(loba.x[1])))
        for i in 1:d
            upper = loba.alpha[i] * (ub[i] - loba.x[j][i])
            lower = loba.alpha[i] * (lb[i] - loba.x[j][i])
            I *= 1 / loba.alpha[i] * (_phi_int(upper, loba.n) - _phi_int(lower, loba.n))
        end
        return I
    end
    return _loba_combine(loba.coeff, box, length(loba.x))
end

"""
    lobachevsky_integrate_dimension(loba::LobachevskySurrogate, lb, ub, dim)

Integrate the surrogate over dimension `dim` on `[lb[dim], ub[dim]]`, returning
a surrogate on the remaining `d - 1` coordinates that evaluates to the marginal
of `loba`.

Neither `loba` nor the `lb` and `ub` passed in are mutated. The returned
surrogate carries the marginal values at the reduced nodes as its `y`, so it is
self-consistent: refitting it reproduces its coefficients.
"""
function lobachevsky_integrate_dimension(loba::LobachevskySurrogate, lb, ub, dim::Int)
    d = length(loba.x[1])
    if !(1 <= dim <= d)
        throw(ArgumentError("Cannot integrate dimension $dim of a $d-dimensional surrogate!"))
    end
    n = length(loba.x)
    # The kernel is a tensor product, so integrating out `dim` scales each
    # coefficient by that sample's own one-dimensional integral. Summing the
    # factors instead would apply one global scale to every sample alike.
    function scale(i)
        upper = _phi_int(loba.alpha[dim] * (ub[dim] - loba.x[i][dim]), loba.n)
        lower = _phi_int(loba.alpha[dim] * (lb[dim] - loba.x[i][dim]), loba.n)
        return (upper - lower) / loba.alpha[dim]
    end
    # For an `n x m` multi-output `coeff` this broadcasts down the rows, scaling
    # every output of a sample by that sample's factor.
    new_coeff = loba.coeff .* scale.(1:n)

    if length(lb) == 2
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
    # `collect` before deleting, so neither the surrogate's alpha nor the
    # caller's bounds are mutated; it also accepts tuple bounds.
    new_lb = deleteat!(collect(lb), dim)
    new_ub = deleteat!(collect(ub), dim)
    new_alpha = deleteat!(collect(loba.alpha), dim)
    # 2D -> 1D leaves scalar alpha and bounds
    function build(y)
        if length(lb) == 2
            return LobachevskySurrogate(
                new_x, y, new_alpha[1], loba.n, new_lb[1], new_ub[1],
                new_coeff, loba.sparse
            )
        end
        return LobachevskySurrogate(
            new_x, y, new_alpha, loba.n, new_lb, new_ub, new_coeff, loba.sparse
        )
    end
    # `loba.y` are the full-dimensional responses; the marginal's are its own
    # values at the reduced nodes, so a later refit stays consistent. Built
    # rather than assigned, so an integer `y` need not hold them.
    return build([build(loba.y)(p) for p in new_x])
end
