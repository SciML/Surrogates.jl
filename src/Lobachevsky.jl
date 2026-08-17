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
  - `coeff`: fitted interpolation coefficients.
  - `sparse`: whether coefficient construction uses a sparse matrix.

# Arguments

  - `x`: training inputs.
  - `y`: training responses, with one response per input. Responses must be scalars; vector-valued responses are not supported.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `alpha = 1.0`: kernel scale. The one-dimensional scalar must lie in `[0, 4]`.
    For multidimensional inputs, supply one scale per input dimension; the
    default is a vector of ones matching one training point.
  - `n::Int = 4`: even kernel order, at most 20 (`factorial(n)` overflows
    `Int64` beyond that).
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
# `phi_nj1D` evaluates `factorial(n - 1)` and `_phi_int` evaluates
# `factorial(n)`; `factorial(::Int)` overflows for arguments above 20.
function _check_lobachevsky_n(n)
    if n <= 0 || n % 2 != 0
        throw(ArgumentError("Kernel order n must be even and positive! Got: $n."))
    end
    if n > 20
        throw(ArgumentError("Kernel order n must be at most 20 (factorial(n) overflows Int64)! Got: $n."))
    end
    return nothing
end

function LobachevskySurrogate(
        x, y, lb::Number, ub::Number; alpha::Number = 1.0, n::Int = 4,
        sparse = false
    )
    if alpha > 4 || alpha < 0
        throw(ArgumentError("Kernel scale alpha must be between 0 and 4! Got: $alpha."))
    end
    _check_lobachevsky_n(n)
    coeff = _calc_loba_coeff1D(x, y, alpha, n, sparse)
    return LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val::Number)
    _check_dimension(loba, val)

    return sum(
        loba.coeff[j] * phi_nj1D(val, loba.x[j], loba.alpha, loba.n)
            for j in 1:length(loba.x)
    )
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
function LobachevskySurrogate(
        x, y, lb, ub; alpha = collect(one.(x[1])), n::Int = 4,
        sparse = false
    )
    if any(a -> a > 4 || a < 0, alpha)
        throw(ArgumentError("All kernel scales alpha must be between 0 and 4! Got: $alpha."))
    end
    _check_lobachevsky_n(n)
    coeff = _calc_loba_coeffND(x, y, alpha, n, sparse)
    return LobachevskySurrogate(x, y, alpha, n, lb, ub, coeff, sparse)
end

function (loba::LobachevskySurrogate)(val)
    _check_dimension(loba, val)
    return sum(
        loba.coeff[j] * phi_njND(val, loba.x[j], loba.alpha, loba.n)
            for j in 1:length(loba.x)
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

#Lobachevsky integrals
function _phi_int(point, n)
    res = zero(eltype(point))
    for k in 0:n
        c = sqrt(n / 3) * point + (n - 2 * k)
        if c > 0
            res = res + (-1)^k * binomial(n, k) * c^n
        end
    end
    return res *= 1 / (2^n * factorial(n))
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
    lobachevsky_integrate_dimension(loba::LobachevskySurrogate, lb, ub, dim)

Integrate the surrogate over dimension `dim` on `[lb[dim], ub[dim]]`, returning
a surrogate on the remaining `d - 1` coordinates that evaluates to the marginal
of `loba`.

The returned surrogate carries the marginal values at the reduced nodes as its
`y`, so it is self-consistent: refitting it (for example through `update!`)
reproduces its coefficients. Neither `loba` nor the `lb`/`ub` passed in are
mutated.
"""
function lobachevsky_integrate_dimension(loba::LobachevskySurrogate, lb, ub, dim::Int)
    n = length(loba.x)
    # The kernel is a tensor product, so integrating out dimension `dim` scales
    # each coefficient by that sample's own one-dimensional integral factor.
    new_coeff = copy(loba.coeff)
    for i in 1:n
        a = loba.alpha[dim] * (ub[dim] - loba.x[i][dim])
        b = loba.alpha[dim] * (lb[dim] - loba.x[i][dim])
        int = 1 / loba.alpha[dim] * (_phi_int(a, loba.n) - _phi_int(b, loba.n))
        new_coeff[i] = loba.coeff[i] * int
    end

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
    # `collect` before deleting so neither the input surrogate nor the caller's
    # bounds are mutated; it also accepts tuple bounds.
    new_lb = deleteat!(collect(lb), dim)
    new_ub = deleteat!(collect(ub), dim)
    new_alpha = deleteat!(collect(loba.alpha), dim)
    reduced = if length(lb) == 2
        # 2D -> 1D: the one-dimensional surrogate stores scalar alpha and bounds
        LobachevskySurrogate(
            new_x, loba.y, new_alpha[1], loba.n, new_lb[1], new_ub[1],
            new_coeff, loba.sparse
        )
    else
        LobachevskySurrogate(
            new_x, loba.y, new_alpha, loba.n, new_lb, new_ub,
            new_coeff, loba.sparse
        )
    end
    # `loba.y` are the responses of the full-dimensional model; the marginal's
    # responses are its own values at the reduced nodes. Storing them keeps the
    # surrogate self-consistent, so a later `update!` refit reproduces
    # `new_coeff` instead of silently discarding the marginalization.
    reduced.y = [reduced(p) for p in new_x]
    return reduced
end
