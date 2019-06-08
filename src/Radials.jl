#=
Response surfaces implementantion, following:
"A Taxonomy of Global Optimization Methods Based on Response Surfaces"
by DONALD R. JONES
=#
mutable struct RadialBasis{F} <: AbstractSurrogate
    phi::F
    dim_poly::Int
    x
    y
    bounds
    coeff
end

"""
Calculates current estimate of array value 'val' with respect to RadialBasis object.
"""
function (rad::RadialBasis)(val)
    n = Base.size(rad.x,1)
    d = Base.size(rad.x,2)
    q = rad.dim_poly
    central_point = zeros(float(eltype(rad.x)), d)
    sum = zero(eltype(rad.x))
    @inbounds for i = 1:d
        central_point[i] = (rad.bounds[i][1]+rad.bounds[i][2])/2
        sum += (rad.bounds[i][2]-rad.bounds[i][1])/2
    end
    half_diameter_domain = sum/d
    approx = zero(eltype(rad.x))
    for i = 1:n
        approx = approx + rad.coeff[i]*rad.phi(vec(val) - rad.x[i,:])
    end
    for i = n+1:n+q
        approx = approx + rad.coeff[i]*centralized_monomial(val,n+1-i,half_diameter_domain,central_point)
    end
    return approx
end

"""
Calculates current estimate of value 'val' with respect to RadialBasis object.
"""
function (rad::RadialBasis)(val::Number)
    approx = zero(eltype(rad.x))
    n = length(rad.x)
    q = rad.dim_poly
    Chebyshev(x,k) = cos(k*acos(-1 + 2/(rad.bounds[2]-rad.bounds[1])*(x-rad.bounds[1])))
    for i = 1:n
        approx = approx + rad.coeff[i]*rad.phi(val - rad.x[i])
    end
    for i = n+1:n+q
        approx = approx + rad.coeff[i]*Chebyshev(val,n+1-i)
    end
    return approx
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
    RadialBasis(x,y,a::Number,b::Number,phi::Function,q::Int)

Constructor for RadialBasis type

#Arguments:
- (x,y): sampled points
- '(a,b)': interval of interest
-'phi': radial basis of choice
-'q': number of polynomial elements
"""
function RadialBasis(x,y,a::Number,b::Number,phi::Function,q::Int)
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
    RadialBasis(phi,q,x,y,(a,b),coeff)
end

"""
    RadialBasis(x,y,bounds,phi::Function,q::Int)

    Constructor for RadialBasis type.

#Arguments:
- 'new_value': array at which you want to find the approximation
- (x,y): sampled points
- 'bounds': region of interest of the form  [[a,b],[c,d],...,[w,z]]
-'phi': radial basis of choice
-'q': number of polynomial elements
"""
function RadialBasis(x,y,bounds,phi::Function,q::Int)
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
    RadialBasis(phi,q,x,y,bounds,coeff)
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
    @inbounds for i = 1:length(vect)
        mul *= vect[i]
    end
    return ((mul-norm(central_point))/(half_diameter_domain))^alpha
end

"""
    add_point!(rad::RadialBasis,new_x,new_y)

Add new samples x and y and updates the coefficients. Return the new object radial.
"""
function add_point!(rad::RadialBasis,new_x,new_y)
    if Base.size(rad.x,1) == 1
        if length(new_x) > 1
            rad.x = hcat(rad.x, new_x)
            rad.y = vcat(rad.y, new_y)
            return RadialBasis(rad.x,rad.y,rad.bounds[1],rad.bounds[2],rad.phi,rad.dim_poly)
        else
            rad.x = vcat(vec(rad.x),new_x)
            rad.y = vcat(vec(rad.y),new_y)
            return RadialBasis(rad.x,rad.y,rad.bounds[1],rad.bounds[2],rad.phi,rad.dim_poly)
        end
    else
        rad.x = vcat(rad.x,new_x)
        rad.y = vcat(rad.y,new_y)
        return RadialBasis(rad.x,rad.y,rad.bounds,rad.phi,rad.dim_poly)
    end
end
