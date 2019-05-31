#=
Rsponse surfaces implementantion, following:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#
using LinearAlgebra

export Radial_1D,evaluate_Radial_1D,linear_basis_function,
       cubic_basis_function,thinplate_basis_function,
       multiquadric_basis_function

abstract type AbstractBasisFunction end

struct Basis <: AbstractBasisFunction
    phi::Function
    number_of_elements_in_polynomial_basis::Int
end

linear_basis_function = Basis(z->norm(z), 1)
cubic_basis_function = Basis(z->norm(z)^3, 2)
thinplate_basis_function = Basis(z->norm(z)^2*log(norm(z)),2)
function multiquadric_basis_function(lambda)
    return Basis(z->sqrt(norm(z)^2 + lambda^2),1)
end


"""
    Radial_1D(x,y,a,b,AbstractBasisFunction)

Find coefficients for interpolation using a radial basis function and a low
degree polynomial.

#Arguments:
- (x,y): set of nodes
- (a,b): interval
- 'basisFunc': selected Basis function
"""
function Radial_1D(x,y,a,b,basisFunc::AbstractBasisFunction)
    if length(x) != length(y)
        error("Data length does not match")
    end
    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))

    n = length(x)
    q = basisFunc.number_of_elements_in_polynomial_basis
    size = n+q
    D = zeros(float(eltype(x)), size, size)
    d = zeros(float(eltype(x)),size)

    @inbounds for i = 1:n
        d[i] =  y[i]
        for j = 1:n
            D[i,j] = basisFunc.phi(x[i] - x[j])
        end
        if i < n + 1
            for k = n+1:size
                D[i,k] = Chebyshev(x[i],k)
            end
        end
    end

    Sym = Symmetric(D,:U)

    return Sym\d

end

"""
    evaluate_Radial_1D(value,coeff,x,a,b,AbstractBasisFunction)

Finds the estimation at point "value" given the coefficients previously
calculated with  Radial_1D.

#Arguments:

- 'value' : value at which you want the approximation
- 'coeff' : returned vector from Radial_1D
- 'x' : set of points
- (a,b): interval
- 'basisFunc': same basis function used in Radial_1D
"""
function evaluate_Radial_1D(value,coeff,x,a,b,basisFunc::AbstractBasisFunction)

    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))
    n = length(x)
    q = basisFunc.number_of_elements_in_polynomial_basis
    approx = 0
    for i = 1:n
        approx = approx + coeff[i]*basisFunc.phi(value - x[i])
    end
    for i = n+1:q
        approx = approx + coeff[i]*Chebyshev(value,n+1-i)
    end
    return approx
end

"""
    centralized_monomial(vect,alpha,half_diameter_domain,central_point)

Returns the value at point vect[] of the alpha degree monomial centralized.

#Arguments:
-'vect': vector of points i.e [x,y,...,w]
-'alpha': degree
-'half_diameter_domain': half diameter of the domain
-'central_point': central point in the domain
"""
function centralized_monomial(vect,alpha,half_diameter_domain,central_point)
    mul = 1
    for i = 1:length(vect)
        mul *= vect[i]
    end
    return ((mul-norm(central_point))/(half_diameter_domain))^alpha
end

"""
    Radial_ND(x,y,bounds,basisFunc::AbstractBasisFunction)

Find coefficients for interpolation using a radial basis function and a low
degree polynomial.

#Arguments:
- '(x,y)': set of sample points
- 'bounds': tuple that contains interval of definition for each component
- 'basisFunc': selected Basis function
"""
function Radial_ND(x,y,bounds,basisFunc::AbstractBasisFunction)
    if Base.size(x,1) != length(y)
        error("Dimension of x and y are not equal")
    end

    n = Base.size(x,1)
    d = Base.size(x,2)
    central_point = zeros(float(eltype(x)), d)
    sum = 0
    @inbounds for i = 1:d
        central_point[i] = (bounds[i][1]+bounds[i][2])/2
        sum += (bounds[i][2]-bounds[i][1])/2
    end
    half_diameter_domain = sum/d

    q = basisFunc.number_of_elements_in_polynomial_basis
    size = n+q
    D = zeros(float(eltype(x)), size, size)
    d = zeros(float(eltype(x)),size)

    @inbounds for i = 1:n
        d[i] =  y[i]
        for j = 1:n
            D[i,j] = basisFunc.phi(x[i,:] - x[j,:])
        end
        if i < n + 1
            for k = n+1:size
                D[i,k] = centralized_monomial(x[i,:],k,half_diameter_domain,central_point)
            end
        end
    end

    Sym = Symmetric(D,:U)

    return Sym\d
end

"""
    evaluate_Radial_ND(value,coeff,x,bounds,basisFunc::AbstractBasisFunction)

Finds the estimation at point "value" given the coefficients previously
calculated with  Radial_ND.

#Arguments:
- 'value' : value at which you want the approximation of the form d-element array
- 'coeff' : returned vector from Radial_ND
- 'x': sample points
- 'bounds': tuple contain interval of definition for each component
- 'basisFunc': same basis function used in Radial_ND
"""
function evaluate_Radial_ND(value,coeff,x,bounds,basisFunc::AbstractBasisFunction)

    n = size(x,1)
    d = size(x,2)

    central_point = zeros(float(eltype(x)), d)
    sum = 0
    @inbounds for i = 1:d
        central_point[i] = (bounds[i][1]+bounds[i][2])/2
        sum += (bounds[i][2]-bounds[i][1])/2
    end
    half_diameter_domain = sum/d

    q = basisFunc.number_of_elements_in_polynomial_basis
    approx = 0
    for i = 1:n
        approx = approx + coeff[i]*basisFunc.phi(value - x[i,:])
    end
    for i = n+1:q
        approx = approx + coeff[i]*centralized_monomial(value,n+1-i)
    end
    return approx
end
