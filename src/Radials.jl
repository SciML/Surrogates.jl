#=
Response surfaces implementantion, following:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#
using LinearAlgebra

abstract type AbstractBasisFunction end

struct RadialBasis{F} <: AbstractBasisFunction
    phi::F
    dim_poly::Int
    x
    y
    bounds
    coeff
    prediction
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

#=
linear_basis_function = Basis(z->norm(z), 1)
cubic_basis_function = Basis(z->norm(z)^3, 2)
thinplate_basis_function = Basis(z->norm(z)^2*log(norm(z)),2)
function multiquadric_basis_function(lambda)
    return Basis(z->sqrt(norm(z)^2 + lambda^2),1)
end
=#

"""
    RadialBasis(new_value::Number,x::Array,y::Array,a::Number,b::Number,phi::Function,q::Int)

Constructor for RadialBasis type that finds the approximation at point new_value

#Arguments:
- 'new_value': point at which you want to find the approximation
- (x,y): sampled points
- '(a,b)': interval of interest
-'phi': radial basis of choice
-'q': number of polynomial elements
"""
function RadialBasis(new_value::Number,x::Array,y::Array,a::Number,b::Number,phi::Function,q::Int)
    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))
    n = length(x)
    size = n+q
    D = zeros(float(eltype(x)), size, size)
    d = zeros(float(eltype(x)),size)

    @inbounds for i = 1:n
        d[i] =  y[i]
        for j = 1:n
            D[i,j] = phi(x[i] - x[j])
        end
        if i < n + 1
            for k = n+1:size
                D[i,k] = Chebyshev(x[i],k)
            end
        end
    end
    Sym = Symmetric(D,:U)
    coeff = Sym\d
    approx = zero(eltype(x))
    for i = 1:n
        approx = approx + coeff[i]*phi(new_value - x[i])
    end
    for i = n+1:q
        approx = approx + coeff[i]*Chebyshev(new_value,n+1-i)
    end

    RadialBasis(phi,q,x,y,[a,b],coeff,approx)
end

"""
    RadialBasis(new_value::Array,x::Array,y::Array,bounds,phi::Function,q::Int)

    Constructor for RadialBasis type that finds the approximation at array new_value

#Arguments:
- 'new_value': array at which you want to find the approximation
- (x,y): sampled points
- 'bounds': region of interest of the form  [[a,b],[c,d],...,[w,z]]
-'phi': radial basis of choice
-'q': number of polynomial elements
"""
function RadialBasis(new_value::Array,x::Array,y::Array,bounds,phi::Function,q::Int)
    n = Base.size(x,1)
    d = Base.size(x,2)
    central_point = zeros(float(eltype(x)), d)
    sum = zero(eltype(x))
    @inbounds for i = 1:d
        central_point[i] = (bounds[i][1]+bounds[i][2])/2
        sum += (bounds[i][2]-bounds[i][1])/2
    end
    half_diameter_domain = sum/d

    size = n+q
    D = zeros(float(eltype(x)), size, size)
    d = zeros(float(eltype(x)),size)

    @inbounds for i = 1:n
        d[i] =  y[i]
        for j = 1:n
            D[i,j] = phi(x[i,:] - x[j,:])
        end
        if i < n + 1
            for k = n+1:size
                D[i,k] = centralized_monomial(x[i,:],k,half_diameter_domain,central_point)
            end
        end
    end

    Sym = Symmetric(D,:U)
    coeff = Sym\d

    approx = zero(eltype(x))
    for i = 1:n
        approx = approx + coeff[i]*phi(new_value - x[i,:])
    end
    for i = n+1:q
        approx = approx + coeff[i]*centralized_monomial(new_value,n+1-i)
    end
    RadialBasis(phi,q,x,y,bounds,coeff,approx)
end
