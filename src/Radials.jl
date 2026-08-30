_copy(t::Tuple) = t
_copy(t) = copy(t)

"""
    RadialBasis(x, y, lb, ub; rad = linearRadial(), scale_factor = 0.5,
                sparse = false, regularization = 0.0)

Fit a radial-basis interpolant, optionally augmented by the polynomial term
required by the selected `RadialFunction`. The result is callable at new
points and implements the SurrogatesBase deterministic-surrogate interface.

# Fields

  - `phi`: radial basis function applied to scaled distances.
  - `dim_poly`: degree of the accompanying polynomial basis.
  - `x`: sampled scalar points or multidimensional points.
  - `y`: scalar or vector responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `coeff`: fitted interpolation coefficients.
  - `scale_factor`: divisor applied to distances before evaluating `phi`.
  - `sparse`: whether coefficient construction uses a sparse matrix.
  - `regularization`: diagonal regularization added to the interpolation matrix.

# Arguments

  - `x`: training inputs.
  - `y`: training responses, with one response per input.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `rad::RadialFunction = linearRadial()`: radial basis descriptor. See also
    [`cubicRadial`](@ref), [`multiquadricRadial`](@ref), and
    [`thinplateRadial`](@ref).
  - `scale_factor::Real = 0.5`: distance scale used by the radial function.
  - `sparse::Bool = false`: use sparse coefficient construction.
  - `regularization::Real = 0.0`: diagonal stabilization term.

# Returns

A callable `RadialBasis` supporting `update!(surrogate, x_new, y_new)`.

# Example

```julia
using Surrogates

x = [0.0, 1.0, 2.0]
y = x .^ 2
surrogate = RadialBasis(x, y, 0.0, 2.0; rad = cubicRadial())
surrogate(1.5)
```
"""
mutable struct RadialBasis{F, Q, X, Y, L, U, C, S, D, R} <: AbstractDeterministicSurrogate
    phi::F
    dim_poly::Q
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    scale_factor::S
    sparse::D
    regularization::R
end

mutable struct RadialFunction{Q, P}
    q::Q # degree of polynomial
    phi::P
end

"""
    linearRadial() -> RadialFunction

Construct the linear radial basis function used by [`RadialBasis`](@ref).

# Returns

A `RadialFunction` with polynomial degree `0` and basis
`z -> norm(z)`.
"""
linearRadial() = RadialFunction(0, z -> norm(z))

"""
    cubicRadial() -> RadialFunction

Construct the cubic radial basis function used by [`RadialBasis`](@ref).

# Returns

A `RadialFunction` with polynomial degree `1` and basis
`z -> norm(z)^3`.
"""
cubicRadial() = RadialFunction(1, z -> norm(z)^3)

"""
    multiquadricRadial(c = 1.0) -> RadialFunction

Construct the multiquadric radial basis function used by
[`RadialBasis`](@ref).

# Arguments

  - `c::Real`: shape parameter inside the multiquadric basis.

# Returns

A `RadialFunction` with polynomial degree `1` and basis
`z -> sqrt((c * norm(z))^2 + 1)`.

# Note

This is Hardy's multiquadric `sqrt(r^2 + c0^2)` with `c0 = 1 / c`, times a
constant that the interpolant is invariant to. So `c` runs the *opposite* way to
the usual shape parameter: a larger `c` gives a peakier basis, not a flatter one.

`c` and `scale_factor` both scale the radius, so they are one parameter between
them: `multiquadricRadial(2.0)` with `scale_factor = 1.0` gives the same
interpolant as `multiquadricRadial(1.0)` with `scale_factor = 0.5`.
"""
function multiquadricRadial(c = 1.0)
    return RadialFunction(
        1, z -> begin
            # `c` at the radius' own precision: its `1.0` default is a Float64
            # literal, which would otherwise carry a Float32 design into Float64.
            r = norm(z)
            return sqrt((oftype(r, c) * r)^2 + one(r))
        end
    )
end

"""
    thinplateRadial() -> RadialFunction

Construct the thin-plate radial basis function used by [`RadialBasis`](@ref).

# Returns

A `RadialFunction` with polynomial degree `2` and basis
`z -> norm(z)^2 * log(norm(z))`, with the origin handled by returning zero.
"""
thinplateRadial() = RadialFunction(
    2, z -> begin
        # On the radius: `iszero` of a tuple is a `MethodError`, which made
        # this kernel unusable for multidimensional inputs.
        r = norm(z)
        iszero(r) ? zero(r * r) : r^2 * log(r)
    end
)

# The closure captures nothing, so one instance identifies the kernel — and
# calling `linearRadial()` to compare against put a construction in the hot loop.
const _LINEAR_RADIAL_PHI = linearRadial().phi

function RadialBasis(
        x, y, lb, ub; rad::RadialFunction = linearRadial(),
        scale_factor::Real = 0.5, sparse = false, regularization::Real = 0.0
    )
    q = rad.q
    phi = rad.phi
    # At the samples' own precision: one divides the distance and the other is
    # added to the matrix diagonal, so a Float64 default would carry a Float32
    # design into Float64.
    T = float(eltype(first(x)))
    scale = convert(T, scale_factor)
    reg = convert(T, regularization)
    coeff = _calc_coeffs(x, y, phi, scale, sparse, reg)
    return RadialBasis(phi, q, x, y, lb, ub, coeff, scale, sparse, reg)
end

function _calc_coeffs(x, y, phi, scale_factor, sparse, regularization)
    D = _construct_rbf_interp_matrix(x, phi, scale_factor, sparse)
    # Guarded: `D += r * I` copies the whole matrix, and the default is zero.
    iszero(regularization) || (D += regularization * I)
    # One coefficient row per output; a scalar response gives a 1 x n row.
    return _copy(transpose(D \ _construct_y_matrix(y, first(y))))
end

# Broadcasting reads a scalar and a multidimensional sample the same way, so one
# method serves both.
function _construct_rbf_interp_matrix(x, phi, scale_factor, sparse)
    n = length(x)
    # `float` because `phi` returns a distance: an integer design would
    # otherwise throw `InexactError` for any scale the samples do not divide.
    T = float(eltype(first(x)))
    D = sparse ? ExtendableSparseMatrix{T, Int}(n, n) : zeros(T, n, n)
    @inbounds for i in 1:n
        # The kernel depends only on the difference and is even in it, so only
        # the triangle `Symmetric` reads has to be filled.
        for j in i:n
            D[i, j] = phi((x[i] .- x[j]) ./ scale_factor)
        end
    end
    return Symmetric(D, :U)
end

using Zygote: Buffer
using ChainRulesCore: @non_differentiable

function _make_combination(n, d, ix)
    exponents_combinations = [
        e
            for e
            in
            collect(
                Iterators.product(
                    Iterators.repeated(
                        0:n,
                        d
                    )...
                )
            )[:]
            if sum(e) <= n
    ]

    return exponents_combinations[ix + 1]
end
# TODO: Is this correct? Do we ever want to differentiate w.r.t n, d, or ix?
# By using @non_differentiable we force the gradient to be 1 for n, d, ix
@non_differentiable _make_combination(n, d, ix)

"""
    multivar_poly_basis(x, ix, d, n)

Evaluates in `x` the `ix`-th element of the multivariate polynomial basis of maximum
degree `n` and `d` dimensions.

Time complexity: `(n+1)^d.`

# Example

For n=2, d=2 the multivariate polynomial basis is

```
1,
x,y
x^2,y^2,xy
```

Therefore the 3rd (ix=3) element is `y` .
Therefore when x=(13,43) and ix=3 this function will return 43.
"""
function multivar_poly_basis(x, ix, d, n)
    if n == 0
        return one(eltype(x))
    else
        prod(
            a^d
                for (a, d)
                in zip(x, _make_combination(n, d, ix))
        )
    end
end

"""
Calculates current estimate of value 'val' with respect to the RadialBasis object.
"""
function (rad::RadialBasis)(val)
    _check_dimension(rad, val)

    approx = _approx_rbf(val, rad)
    return _match_container(approx, first(rad.y))
end

# The accumulator holds coefficient times kernel value, so its type is the
# promotion of the two; taking it from the query alone broke integer queries.
_approx_eltype(val, rad) = promote_type(eltype(val), eltype(rad.coeff))

# Zygote cannot trace `setindex!`, so the multi-output accumulator is one of its
# buffers rather than a plain array.
function _make_approx(val, rad::RadialBasis)
    return Buffer(zeros(_approx_eltype(val, rad), size(rad.coeff, 1)), false)
end

function _add_tmp_to_approx!(approx, i, kernel, rad::RadialBasis)
    return @inbounds @simd ivdep for j in 1:size(rad.coeff, 1)
        approx[j] += rad.coeff[j, i] * kernel
    end
end

function _check_coeff_count(rad::RadialBasis)
    n = length(rad.x)
    if n > size(rad.coeff, 2)
        throw(
            ArgumentError(
                "RadialBasis has $n samples but only $(size(rad.coeff, 2)) \
                coefficients. The sample and coefficient containers have fallen \
                out of step; rebuild the surrogate, or use update! to add \
                samples so the coefficients are refitted with them."
            )
        )
    end
    return n
end

function _approx_rbf(val, rad::RadialBasis)
    n = _check_coeff_count(rad)

    approx = _make_approx(val, rad)
    if rad.phi === _LINEAR_RADIAL_PHI
        @inbounds for i in 1:n
            _add_tmp_to_approx!(approx, i, _linear_distance(val, rad, i), rad)
        end
    else
        @inbounds for i in 1:n
            _add_tmp_to_approx!(
                approx, i, rad.phi((val .- rad.x[i]) ./ rad.scale_factor), rad
            )
        end
    end
    return copy(approx)
end

# A scalar response has a single coefficient row, so it accumulates into a local
# rather than a buffer: the total stays in a register, and nothing is mutated, so
# reverse mode differentiates straight through.
function _approx_rbf(
        val, rad::RadialBasis{F, Q, X, <:AbstractArray{<:Number}}
    ) where {F, Q, X}
    n = _check_coeff_count(rad)
    approx = zero(_approx_eltype(val, rad))
    # Hoisted rather than tested per sample, which costs about a third of the
    # evaluation. `@simd` marks a float reduction into a local, over reads alone.
    if rad.phi === _LINEAR_RADIAL_PHI
        @inbounds @simd for i in 1:n
            approx += rad.coeff[1, i] * _linear_distance(val, rad, i)
        end
    else
        @inbounds for i in 1:n
            approx += rad.coeff[1, i] * rad.phi((val .- rad.x[i]) ./ rad.scale_factor)
        end
    end
    return approx
end

# The linear kernel is `norm` itself, so its scaled distance is summed
# coordinate by coordinate rather than through a temporary difference.
@inline function _linear_distance(val, rad::RadialBasis, i)
    tmp = zero(_approx_eltype(val, rad))
    @inbounds @simd ivdep for j in 1:length(val)
        tmp += ((val[j] - rad.x[i][j]) / rad.scale_factor)^2
    end
    return sqrt(tmp)
end

_scaled_chebyshev(x, k, lb, ub) = cos(k * acos(-1 + 2 * (x - lb) / (ub - lb)))
_center_bounds(x::Tuple, lb, ub) = ntuple(i -> (ub[i] - lb[i]) / 2, length(x))
_center_bounds(x, lb, ub) = (ub .- lb) ./ 2

"""
    update!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and update the coefficients. Return the new object radial.
"""
function SurrogatesBase.update!(rad::RadialBasis, new_x, new_y)
    rad.x, rad.y = _append_samples(rad.x, rad.y, new_x, new_y)
    rad.coeff = _calc_coeffs(
        rad.x, rad.y, rad.phi, rad.scale_factor, rad.sparse, rad.regularization
    )
    return nothing
end
