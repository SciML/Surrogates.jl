_copy(t::Tuple) = t
_copy(t) = copy(t)

"""
    RadialBasis(x, y, lb, ub; rad = linearRadial(), scale_factor = 0.5,
                sparse = false, regularization = 0.0)

Fit a radial-basis interpolant augmented by the polynomial term required by the
selected `RadialFunction`. The result is callable at new points and implements
the SurrogatesBase deterministic-surrogate interface.

The fitted model is

```
s(x) = Σ_j coeff_j φ(‖x - x_j‖ / scale_factor) + Σ_k d_k p_k(x)
```

where `p_k` runs over the multivariate monomials of degree at most
`rad.q`. Coefficients solve the augmented system `[Φ P; Pᵀ 0] [c; d] = [y; 0]`.
The kernels offered here are only *conditionally* positive definite, so the
polynomial block is what makes the system uniquely solvable; it also makes the
surrogate reproduce polynomials of degree at most `dim_poly` exactly.

# Fields

  - `phi`: radial basis function applied to scaled distances.
  - `dim_poly`: degree of the accompanying polynomial basis. The surrogate
    reproduces polynomials up to this degree exactly.
  - `x`: sampled scalar points or multidimensional points.
  - `y`: scalar or vector responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `coeff`: fitted coefficients, `[radial weights; polynomial weights]`, with
    one row per output.
  - `scale_factor`: divisor applied to distances before evaluating `phi`.
  - `sparse`: whether coefficient construction uses a sparse matrix.
  - `regularization`: diagonal regularization added to the radial block of the
    interpolation matrix. The polynomial rows are left untouched, so the side
    conditions still hold exactly.

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

  - `c::Real`: scale parameter inside the multiquadric basis.

# Returns

A `RadialFunction` with polynomial degree `1` and basis
`z -> sqrt((c * norm(z))^2 + 1)`.
"""
multiquadricRadial(c = 1.0) = RadialFunction(1, z -> sqrt((c * norm(z))^2 + 1))

"""
    thinplateRadial() -> RadialFunction

Construct the thin-plate radial basis function used by [`RadialBasis`](@ref).

# Returns

A `RadialFunction` with polynomial degree `2` and basis
`z -> norm(z)^2 * log(norm(z))`, with the origin handled by returning zero.
"""
thinplateRadial() = RadialFunction(
    2, z -> begin
        # Branch on the radius, not on `z`: in N dimensions `z` is a tuple, and
        # `iszero(::Tuple)` needs `zero(::Tuple)`, which does not exist. The
        # branch also short-circuits, so `log(0)` is never evaluated — an
        # `ifelse` would compute `0 * -Inf = NaN` and poison forward-mode
        # derivatives.
        r = norm(z)
        iszero(r) ? zero(r) : r^2 * log(r)
    end
)

function RadialBasis(
        x, y, lb, ub; rad::RadialFunction = linearRadial(),
        scale_factor::Real = 0.5, sparse = false, regularization::Real = 0.0
    )
    q = rad.q
    phi = rad.phi
    coeff = _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse, regularization)
    return RadialBasis(phi, q, x, y, lb, ub, coeff, scale_factor, sparse, regularization)
end

# Number of terms in the multivariate polynomial basis of degree ≤ `q` in `nd`
# variables. This is the width of the polynomial tail block `P`.
_num_poly_terms(q, nd) = binomial(q + nd, q)

function _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse, regularization)
    nd = length(first(x))
    m = _num_poly_terms(q, nd)
    D = _construct_rbf_interp_matrix(
        x, first(x), lb, ub, phi, q, scale_factor, sparse, regularization
    )
    # The right-hand side is `[y; 0]`: the trailing zeros are the side conditions
    # `Σ_j c_j p_k(x_j) = 0` that make the augmented system square and pin down
    # the polynomial tail.
    Y = _construct_rbf_y_matrix(y, first(y), length(y) + m)
    return _copy(transpose(D \ Y))
end

# Augmented interpolation matrix
#
#     [ Φ  P ]      Φ_ij = φ(‖x_i - x_j‖ / scale_factor)
#     [ Pᵀ 0 ]      P_ik = p_k(x_i)
#
# The radial kernels used here (linear, cubic, multiquadric, thin plate) are only
# *conditionally* positive definite, so Φ alone is not guaranteed invertible. The
# polynomial block restores unique solvability and lets the surrogate reproduce
# polynomials of degree ≤ q exactly.
function _construct_rbf_interp_matrix(
        x, x_el, lb, ub, phi, q, scale_factor, sparse, regularization
    )
    n = length(x)
    nd = length(x_el)
    m = _num_poly_terms(q, nd)
    T = eltype(x_el)
    if sparse
        D = ExtendableSparseMatrix{T, Int}(n + m, n + m)
    else
        D = zeros(T, n + m, n + m)
    end
    @inbounds for i in 1:n
        for j in i:n
            D[i, j] = phi((x[i] .- x[j]) ./ scale_factor)
        end
        # Polynomial tail; the lower-left Pᵀ follows from the `Symmetric` wrapper,
        # and the trailing m×m block stays zero.
        for k in 1:m
            D[i, n + k] = multivar_poly_basis(x[i], k - 1, nd, q)
        end
    end
    if !iszero(regularization)
        # Only the radial block is regularized: perturbing the tail rows would
        # relax the side conditions rather than stabilize the fit.
        @inbounds for i in 1:n
            D[i, i] += regularization
        end
    end
    return Symmetric(D, :U)
end

function _construct_rbf_y_matrix(y, y_el::Number, m)
    return [i <= length(y) ? y[i] : zero(y_el) for i in 1:m]
end
function _construct_rbf_y_matrix(y, y_el, m)
    return [i <= length(y) ? y[i][j] : zero(first(y_el)) for i in 1:m, j in 1:length(y_el)]
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
# `n`, `d` and `ix` are integer basis indices, never differentiation variables,
# so the rule below simply keeps AD from trying to trace through them.
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

function _approx_rbf(val::Number, rad::RadialBasis)
    n = length(rad.x)
    q = rad.dim_poly
    approx = zero(rad.coeff[:, 1])
    for i in 1:n
        approx += rad.coeff[:, i] * rad.phi((val .- rad.x[i]) / rad.scale_factor)
    end
    for k in 1:_num_poly_terms(q, 1)
        approx += rad.coeff[:, n + k] * multivar_poly_basis(val, k - 1, 1, q)
    end
    return approx
end

function _make_approx(val, rad::RadialBasis)
    l = size(rad.coeff, 1)
    return Buffer(zeros(eltype(val), l), false)
end
function _add_tmp_to_approx!(approx, i, tmp, rad::RadialBasis; f = identity)
    return @inbounds @simd ivdep for j in 1:size(rad.coeff, 1)
        approx[j] += rad.coeff[j, i] * f(tmp)
    end
end
function _make_approx(
        val,
        ::RadialBasis{F, Q, X, <:AbstractArray{<:Number}}
    ) where {F, Q, X}
    return Ref(zero(eltype(val)))
end
function _add_tmp_to_approx!(
        approx::Base.RefValue, i, tmp,
        rad::RadialBasis{F, Q, X, <:AbstractArray{<:Number}};
        f = identity
    ) where {F, Q, X}
    return @inbounds @simd ivdep for j in 1:size(rad.coeff, 1)
        approx[] += rad.coeff[j, i] * f(tmp)
    end
end

_ret_copy(v::Base.RefValue) = v[]
_ret_copy(v) = copy(v)

function _approx_rbf(val_raw, rad::RadialBasis)
    val = _as_point(val_raw)
    n = length(rad.x)
    nd = length(val)
    m = _num_poly_terms(rad.dim_poly, nd)

    if n + m > size(rad.coeff, 2)
        throw(
            ArgumentError(
                "Length of model's x vector exceeds number of calculated coefficients ($(n + m) != $(size(rad.coeff, 2)))."
            )
        )
    end

    approx = _make_approx(val, rad)

    if rad.phi === linearRadial().phi
        for i in 1:n
            tmp = zero(eltype(val))
            @inbounds @simd ivdep for j in 1:length(val)
                tmp += ((val[j] - rad.x[i][j]) / rad.scale_factor)^2
            end
            tmp = sqrt(tmp)
            _add_tmp_to_approx!(approx, i, tmp, rad)
        end
    else
        tmp = collect(val)
        @inbounds for i in 1:n
            tmp = (val .- rad.x[i]) ./ rad.scale_factor
            _add_tmp_to_approx!(approx, i, tmp, rad; f = rad.phi)
        end
    end

    # Polynomial tail, evaluated in the original coordinates so that the side
    # conditions imposed during the fit hold exactly.
    for k in 1:m
        _add_tmp_to_approx!(
            approx, n + k, multivar_poly_basis(val, k - 1, nd, rad.dim_poly), rad
        )
    end

    return _ret_copy(approx)
end

"""
    update!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and update the coefficients. Return the new object radial.

Every call refits the surrogate from scratch on the full sample set, which
costs O(n^3) for the kernel models. Adding points one at a time in a loop is
therefore quadratic in the number of additions.

"""
function SurrogatesBase.update!(rad::RadialBasis, new_x, new_y)
    rad.x, rad.y = _append_samples(rad.x, rad.y, new_x, new_y)
    rad.coeff = _calc_coeffs(
        rad.x, rad.y, rad.lb, rad.ub, rad.phi, rad.dim_poly,
        rad.scale_factor, rad.sparse, rad.regularization
    )
    return nothing
end
