using LinearAlgebra
using ExtendableSparse
using Base.Threads
using Distributed

_copy(t::Tuple) = t
_copy(t) = copy(t)

mutable struct RadialBasis{F, Q, X, Y, L, U, C, S, D} <: AbstractSurrogate
    phi::F
    dim_poly::Q
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    scale_factor::S
    sparse::D
end

mutable struct RadialFunction{Q, P}
    q::Q # degree of polynomial
    phi::P
end

linearRadial() = RadialFunction(0, z -> norm(z))
cubicRadial() = RadialFunction(1, z -> norm(z)^3)
multiquadricRadial(c = 1.0) = RadialFunction(1, z -> sqrt((c * norm(z))^2 + 1))

thinplateRadial() = RadialFunction(2, z -> begin
    result = norm(z)^2 * log(norm(z))
    ifelse(iszero(z), zero(result), result)
end)

"""
RadialBasis(x,y,lb,ub,rad::RadialFunction, scale_factor::Float = 1.0)

Constructor for RadialBasis surrogate, of the form

``f(x) = \\sum_{i=1}^{N} w_i \\phi(|x - \\bold{c}_i|) \\bold{v}^{T} + \\bold{v}^{\\mathrm{T}} [ 0; \\bold{x} ]``

where ``w_i`` are the weights of polyharmonic splines ``\\phi(x)`` and ``\\bold{v}`` are coefficients
of a polynomial term.

References:
https://en.wikipedia.org/wiki/Polyharmonic_spline
"""
function RadialBasis(
        x,
        y,
        lb,
        ub;
        rad::RadialFunction = linearRadial(),
        scale_factor::Real = 0.5,
        sparse = false
)
    q = rad.q
    phi = rad.phi
    coeff = _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse)
    return RadialBasis(phi, q, x, y, lb, ub, coeff, scale_factor, sparse)
end

function _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse)
    nd = length(first(x))
    num_poly_terms = binomial(q + nd, q)
    D = _construct_rbf_interp_matrix(x, first(x), lb, ub, phi, q, scale_factor, sparse)
    Y = _construct_rbf_y_matrix(y, first(y), length(y) + num_poly_terms)
    if (typeof(y) == Vector{Float64}) #single output case
        coeff = _copy(transpose(D \ y))
    else
        coeff = _copy(transpose(D \ Y[1:size(D)[1], :])) #if y is multi output;
    end
    return coeff
end

function _construct_rbf_interp_matrix(x, x_el::Number, lb, ub, phi, q, scale_factor, sparse)
    n = length(x)
    if sparse
        D = ExtendableSparseMatrix{eltype(x_el), Int}(n, n)
    else
        D = zeros(eltype(x_el), n, n)
    end
    @inbounds for i in 1:n
        for j in i:n
            D[i, j] = phi((x[i] .- x[j]) ./ scale_factor)
        end
    end
    D_sym = Symmetric(D, :U)
    return D_sym
end

function _construct_rbf_interp_matrix(x, x_el, lb, ub, phi, q, scale_factor, sparse)
    n = length(x)
    nd = length(x_el)
    if sparse
        D = ExtendableSparseMatrix{eltype(x_el), Int}(n, n)
    else
        D = zeros(eltype(x_el), n, n)
    end
    @inbounds for i in 1:n
        for j in i:n
            D[i, j] = phi((x[i] .- x[j]) ./ scale_factor)
        end
    end
    D_sym = Symmetric(D, :U)
    return D_sym
end

function _construct_rbf_y_matrix(y, y_el::Number, m)
    [i <= length(y) ? y[i] : zero(y_el) for i in 1:m]
end
function _construct_rbf_y_matrix(y, y_el, m)
    [i <= length(y) ? y[i][j] : zero(first(y_el)) for i in 1:m, j in 1:length(y_el)]
end

using Zygote: Buffer
using ChainRulesCore: @non_differentiable

function _make_combination(n, d, ix)
    exponents_combinations = [e
                              for
                              e in collect(Iterators.product(Iterators.repeated(
                                  0:n, d)...))[:] if sum(e) <= n]

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

````
1,
x,y
x^2,y^2,xy
````

Therefore the 3rd (ix=3) element is `y` .
Therefore when x=(13,43) and ix=3 this function will return 43.
"""
function multivar_poly_basis(x, ix, d, n)
    if n == 0
        return one(eltype(x))
    else
        prod(a^d for (a, d) in zip(x, _make_combination(n, d, ix)))
    end
end

"""
Calculates current estimate of value 'val' with respect to the RadialBasis object.
"""
function (rad::RadialBasis)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(rad, val)

    approx = _approx_rbf(val, rad)
    return _match_container(approx, first(rad.y))
end

function _approx_rbf(val::Number, rad::RadialBasis)
    n = length(rad.x)
    approx = zero(rad.coeff[:, 1])
    for i in 1:n
        approx += rad.coeff[:, i] * rad.phi((val .- rad.x[i]) / rad.scale_factor)
    end
    return approx
end

function _make_approx(val, rad::RadialBasis)
    l = size(rad.coeff, 1)
    return Buffer(zeros(eltype(val), l), false)
end
function _add_tmp_to_approx!(approx, i, tmp, rad::RadialBasis; f = identity)
    @inbounds @simd ivdep for j in 1:size(rad.coeff, 1)
        approx[j] += rad.coeff[j, i] * f(tmp)
    end
end
# specialise when only single output dimension
function _make_approx(
        val, ::RadialBasis{F, Q, X, <:AbstractArray{<:Number}}) where {F, Q, X}
    return Ref(zero(eltype(val)))
end
function _add_tmp_to_approx!(
        approx::Base.RefValue,
        i,
        tmp,
        rad::RadialBasis{F, Q, X, <:AbstractArray{<:Number}};
        f = identity
) where {F, Q, X}
    @inbounds @simd ivdep for j in 1:size(rad.coeff, 1)
        approx[] += rad.coeff[j, i] * f(tmp)
    end
end

_ret_copy(v::Base.RefValue) = v[]
_ret_copy(v) = copy(v)

function _approx_rbf(val, rad::RadialBasis)
    n = length(rad.x)
    approx = _make_approx(val, rad)

    # Define a function to compute tmp for a single index i
    function compute_tmp(i)
        tmp = zero(eltype(val))
        @inbounds @simd ivdep for j in eachindex(val)
            tmp += ((val[j] - rad.x[i][j]) / rad.scale_factor)^2
        end
        return sqrt(tmp)
    end

    # Use pmap to parallelize the computation of tmp
    tmp_values = pmap(compute_tmp, 1:n)

    # Update the approx using the computed tmp values
    for (i, tmp) in enumerate(tmp_values)
        _add_tmp_to_approx!(approx, i, tmp, rad)
    end

    return _ret_copy(approx)
end

_scaled_chebyshev(x, k, lb, ub) = cos(k * acos(-1 + 2 * (x - lb) / (ub - lb)))
_center_bounds(x::Tuple, lb, ub) = ntuple(i -> (ub[i] - lb[i]) / 2, length(x))
_center_bounds(x, lb, ub) = (ub .- lb) ./ 2

"""
    add_point!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and update the coefficients. Return the new object radial.
"""
function add_point!(rad::RadialBasis, new_x, new_y)
    if (length(new_x) == 1 && length(new_x[1]) == 1) ||
       (length(new_x) > 1 && length(new_x[1]) == 1 && length(rad.lb) > 1)
        push!(rad.x, new_x)
        push!(rad.y, new_y)
    else
        append!(rad.x, new_x)
        append!(rad.y, new_y)
    end
    rad.coeff = _calc_coeffs(
        rad.x,
        rad.y,
        rad.lb,
        rad.ub,
        rad.phi,
        rad.dim_poly,
        rad.scale_factor,
        rad.sparse
    )
    nothing
end
