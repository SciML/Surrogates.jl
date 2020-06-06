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
    q::Q
    phi::P
end

linearRadial = RadialFunction(0,z->norm(z))
cubicRadial = RadialFunction(1,z->norm(z)^3)
multiquadricRadial = RadialFunction(1,z->sqrt(norm(z)^2+1))
thinplateRadial = RadialFunction(2,z->norm(z)^2*log(norm(z)))

"""
    RadialBasis(x,y,lb::Number,ub::Number; rad::RadialFunction = linearRadial,scale::Real=1.0)

Constructor for Radial basis surrogate.
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
    D = _construct_rbf_interp_matrix(x, first(x), lb, ub, phi, q, scale_factor, sparse)
    Y = _construct_rbf_y_matrix(y, first(y), length(y) + q)
    coeff = D \ Y
    return coeff
end

function _construct_rbf_interp_matrix(x, x_el::Number, lb, ub, phi, q, scale_factor, sparse)
    n = length(x)
    m = n + q
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
            for k = 1:q
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
    central_point = _center_bounds(x_el, lb, ub)
    sum_half_diameter = sum((ub[k]-lb[k])/2 for k = 1:nd)
    mean_half_diameter = sum_half_diameter / nd
    m = n+q
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
            for k = 1:q
                D[i,n+k] = centralized_monomial(x[i], k-1, mean_half_diameter, central_point)
            end
        end
    end
    D_sym = Symmetric(D, :U)
    return D_sym
end

_construct_rbf_y_matrix(y, y_el::Number, m) = [i <= length(y) ? y[i] : zero(y_el) for i = 1:m]
_construct_rbf_y_matrix(y, y_el, m) = [i <= length(y) ? y[i][j] : zero(first(y_el)) for i=1:m, j=1:length(y_el)]

"""
    centralized_monomial(vect,alpha,mean_half_diameter,central_point)

Returns the value at point vect[] of the alpha degree monomial centralized.

#Arguments:
-'vect': vector of points i.e [x,y,...,w]
-'alpha': degree
-'mean_half_diameter': half diameter of the domain
-'central_point': central point in the domain
"""
function centralized_monomial(vect,alpha,mean_half_diameter,central_point)
    if iszero(alpha) return one(eltype(vect)) end
    centralized_product = prod(vect .- central_point)
    return (centralized_product / mean_half_diameter)^alpha
end

"""
Calculates current estimate of value 'val' with respect to RadialBasis object.
"""
function (rad::RadialBasis)(val)
    approx = _approx_rbf(val, rad)
    return _match_container(approx, first(rad.y))
end

function _approx_rbf(val::Number, rad)
    n = length(rad.x)
    q = rad.dim_poly
    lb = rad.lb
    ub = rad.ub
    approx = zero(rad.coeff[1, :])
    for i = 1:n
        approx += rad.coeff[i, :] * rad.phi( (val .- rad.x[i]) / rad.scale_factor)
    end
    for k = 1:q
        approx += rad.coeff[n+k, :] * _scaled_chebyshev(val, k-1, lb, ub)
    end
    return approx
end
function _approx_rbf(val, rad)
    n = length(rad.x)
    d = length(rad.x[1])
    q = rad.dim_poly
    lb = rad.lb
    ub = rad.ub
    sum_half_diameter = sum((ub[k]-lb[k])/2 for k = 1:d)
    mean_half_diameter = sum_half_diameter/d
    central_point = _center_bounds(first(rad.x), lb, ub)

    approx = zero(rad.coeff[1, :])
    @views approx += sum( rad.coeff[i, :] * rad.phi( (val .- rad.x[i]) ./rad.scale_factor) for i = 1:n)
    for k = 1:q
        @views approx += rad.coeff[n+k, :] .* centralized_monomial(val, k-1, mean_half_diameter, central_point)
    end
    return approx
end

_scaled_chebyshev(x, k, lb, ub) = cos(k*acos(-1 + 2*(x-lb)/(ub-lb)))
_center_bounds(x::Tuple, lb, ub) = ntuple(i -> (ub[i] - lb[i])/2, length(x))
_center_bounds(x, lb, ub) = (ub .- lb) ./ 2

"""
    add_point!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and updates the coefficients. Return the new object radial.
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
