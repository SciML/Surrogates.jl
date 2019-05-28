using LinearAlgebra

abstract type AbstractBasisFunction end

struct Basis <: AbstractBasisFunction
    phi::Function
    number_of_elements_in_polynomial_basis::Int
end

linear_basis_function = Basis(z->abs(z), 1)
cubic_basis_function = Basis(z->abs(z)^3, 2)
thinplate_basis_function = Basis(z->z^2*log(abs(z)),2)
function multiquadric_basis_function(lambda)
    return Basis(z->sqrt(abs(z)^2 + lambda^2),1)
end


"""
(x,y): set of nodes
(a,b): interval
basisFunc: selected Basis function
Output: vector of coefficients n+q required by the estimator
"""
function Radial_1D(x,y,a,b,basisFunc::AbstractBasisFunction)
    if length(x) != length(y)
        error("Data length does not match")
    end
    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))

    n = length(x)
    q = basisFunc.number_of_elements_in_polynomial_basis
    #Find coefficients for both radial basis functions and polynomial terms
    size = n+q
    D = zeros(float(eltype(x)), size, size)
    d = zeros(float(eltype(x)),size)
    #In this array I have in the first n entries the coefficient of the radial
    #basis function, in the last q term the coefficients for polynomial terms
    coeff = zeros(float(eltype(x)),size)

    #Matrix made by 4 blocks:
    #=
    A | B
    B^t | 0
    A nxn, B nxq A,B symmetric so the matrix D is symmetric as well.
    =#

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
value: value at which you want the approximation
coeff: returned vector from Radial_1D
x: set of points
(a,b): interval
"""
function evaluate(value,coeff,x,a,b,basisFunc::AbstractBasisFunction)

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
