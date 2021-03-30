using LinearAlgebra
using ExtendableSparse

mutable struct RadialBasis{F,Q,X,Y,L,U,C,S,D} <: AbstractSurrogate
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

mutable struct RadialFunction{Q,P}
    q::Q # degree of polynomial
    phi::P
end

linearRadial = RadialFunction(0,z->norm(z))

cubicRadial = RadialFunction(1,z->norm(z)^3)
multiquadricRadial = RadialFunction(1,z->sqrt(norm(z)^2+1))

thinplateRadial = RadialFunction(2, z->begin
    result = norm(z)^2 * log(norm(z))
    ifelse(iszero(z), zero(result), result)
end)

"""
    RadialBasis(x,y,lb::Number,ub::Number; rad::RadialFunction = linearRadial,scale::Real=1.0)

Constructor for RadialBasis surrogate.
"""
function RadialBasis(x, y, lb::Number, ub::Number; rad::RadialFunction=linearRadial, scale_factor::Real=1.0, sparse = false)
    q = rad.q
    phi = rad.phi
    coeff = _calc_coeffs(x, y, lb, ub, phi, q,scale_factor, sparse)
    return RadialBasis(phi, q, x, y, lb, ub, coeff,scale_factor,sparse)
end

"""
RadialBasis(x,y,lb,ub,rad::RadialFunction, scale_factor::Float = 1.0)

Constructor for RadialBasis surrogate
"""
function RadialBasis(x, y, lb, ub; rad::RadialFunction = linearRadial, scale_factor::Real=1.0, sparse = false)
    q = rad.q
    phi = rad.phi
    coeff = _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse)
    return RadialBasis(phi, q, x, y, lb, ub, coeff,scale_factor, sparse)
end

function _calc_coeffs(x, y, lb, ub, phi, q, scale_factor, sparse)
    nd = length(first(x))
    num_poly_terms = binomial(q + nd, q)

    D = _construct_rbf_interp_matrix(x, first(x), lb, ub, phi, q, scale_factor, sparse)
    Y = _construct_rbf_y_matrix(y, first(y), length(y) + num_poly_terms)
    coeff = D \ Y
    return coeff
end

function _construct_rbf_interp_matrix(x, x_el::Number, lb, ub, phi, q, scale_factor, sparse)
    n = length(x)

    num_poly_terms = binomial(q + 1, q)
    m = n + num_poly_terms

    if sparse
        D = ExtendableSparseMatrix{eltype(x_el),Int}(m,m)
    else
        D = zeros(eltype(x_el), m, m)
    end
    @inbounds for i = 1:n
        for j = 1:n
            D[i,j] = phi( (x[i] .- x[j]) ./ scale_factor )
        end
        if i <= n
            for k = 1:num_poly_terms
                    D[i,n+k] = _scaled_chebyshev(x[i], k-1, lb, ub)
            end
        end
    end
    D_sym = Symmetric(D, :U)
    return D_sym
end

function _construct_rbf_interp_matrix(x, x_el, lb, ub, phi, q, scale_factor,sparse)
    n = length(x)
    nd = length(x_el)

    num_poly_terms = binomial(q + nd, q)
    m = n + num_poly_terms

    if sparse
        D = ExtendableSparseMatrix{eltype(x_el),Int}(m,m)
    else
        D = zeros(eltype(x_el), m, m)
    end
    @inbounds for i = 1:n
        for j = 1:n
            D[i,j] = phi( (x[i] .- x[j]) ./ scale_factor)
        end
        if i < n + 1
            for k = 1:num_poly_terms
                D[i,n+k] = multivar_poly_basis(x[i], k-1, nd, q)
            end
        end
    end
    D_sym = Symmetric(D, :U)
    return D_sym
end

_construct_rbf_y_matrix(y, y_el::Number, m) = [i <= length(y) ? y[i] : zero(y_el) for i = 1:m]
_construct_rbf_y_matrix(y, y_el, m) = [i <= length(y) ? y[i][j] : zero(first(y_el)) for i=1:m, j=1:length(y_el)]

using Zygote: @nograd

function _make_combination(n, d, ix)
    exponents_combinations = [
        e
        for e
        in collect(
            Iterators.product(
                Iterators.repeated(0:n, d)...
            )
        )[:]
        if sum(e) <= n
    ]

    return exponents_combinations[ix + 1]
end
# TODO: Is this correct? Do we ever want to differentiate w.r.t n, d, or ix?
# By using @nograd we force the gradient to be 1 for n, d, ix
@nograd _make_combination

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
        prod(
        a^d
        for (a, d)
        in zip(x, _make_combination(n, d, ix)))
    end
end

"""
Calculates current estimate of value 'val' with respect to the RadialBasis object.
"""
function (rad::RadialBasis)(val)
    approx = _approx_rbf(val, rad)
    return _match_container(approx, first(rad.y))
end

function _approx_rbf(val::Number, rad)
    n = length(rad.x)
    q = rad.dim_poly
    num_poly_terms = binomial(q + 1, q)
    lb = rad.lb
    ub = rad.ub
    approx = zero(rad.coeff[1, :])
    for i = 1:n
        approx += rad.coeff[i, :] * rad.phi( (val .- rad.x[i]) / rad.scale_factor)
    end
    for k = 1:num_poly_terms
        approx += rad.coeff[n+k, :] * _scaled_chebyshev(val, k-1, lb, ub)
    end
    return approx
end
function _approx_rbf(val, rad)
    n = length(rad.x)
    d = length(rad.x[1])
    q = rad.dim_poly
    num_poly_terms = binomial(q + d, q)
    lb = rad.lb
    ub = rad.ub
    sum_half_diameter = sum((ub[k]-lb[k])/2 for k = 1:d)
    mean_half_diameter = sum_half_diameter/d
    central_point = _center_bounds(first(rad.x), lb, ub)

    approx = zero(rad.coeff[1, :])
    @views approx += sum( rad.coeff[i, :] * rad.phi( (val .- rad.x[i]) ./rad.scale_factor) for i = 1:n)
    for k = 1:num_poly_terms
        @views approx += rad.coeff[n+k, :] .* multivar_poly_basis(val, k-1, d, q)
    end
    return approx
end

_scaled_chebyshev(x, k, lb, ub) = cos(k*acos(-1 + 2*(x-lb)/(ub-lb)))
_center_bounds(x::Tuple, lb, ub) = ntuple(i -> (ub[i] - lb[i])/2, length(x))
_center_bounds(x, lb, ub) = (ub .- lb) ./ 2

"""
    add_point!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and update the coefficients. Return the new object radial.
"""
function add_point!(rad::RadialBasis,new_x,new_y)
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(rad.lb)>1)
        push!(rad.x,new_x)
        push!(rad.y,new_y)
    else
        append!(rad.x,new_x)
        append!(rad.y,new_y)
    end
    rad.coeff = _calc_coeffs(rad.x,rad.y,rad.lb,rad.ub,rad.phi,rad.dim_poly, rad.scale_factor, rad.sparse)
    nothing
end
