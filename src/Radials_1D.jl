function Radial_1D(x,y,a,b,kind::String,lambda = 0)
    '''
    (x,y) set of nodes
    (a,b) interval
    kind is type of radial basis function
    lambda is optional parameter with kind == multiquadric
    '''
    if length(x) != length(y)
        error("Data length does not match")
    end

    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))

    #Type of Radial basis function
    #q is the number of polynomials in the basis
    #The numbers are suggested by papers
    if lambda === nothing
        if kind == "linear"
            q = 1
            phi = z -> abs(z)
        elseif kind == "cubic"
            q = 2
            phi = z -> abs(z)^3
        elseif kind == "thinplate"
            q = 2
            phi = z -> z^2*log(abs(z))
        else
            error("Wrong type")
        end
    else
        if kind == "multiquadric"
            q = 1
            phi = z -> sqrt(abs(z)^2 + lambda^2)
        else
            error("Wrong type")
        end
    end
    n = length(x)
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
            D[i,j] = phi(x[i] - x[j])
        end
        if i < n + 1
            for k = n+1:size
                D[i,k] = Chebyshev(x[i],k)
            end
        end
    end

    Sym = Symmetric(D,:U)

    #Vector of size n + q containing in the first n terms the coefficients of
    # the radial basis function and in the last q term the coefficient for the
    # polynomials
    return Sym\d

end

function evaluate(value,coeff,x,a,b,kind::String,lambda = 0)

    Chebyshev(x,k) = cos(k*acos(-1 + 2/(b-a)*(x-a)))
    if lambda === nothing
        if kind == "linear"
            q = 1
            phi = z -> abs(z)
        elseif kind == "cubic"
            q = 2
            phi = z -> abs(z)^3
        elseif kind == "thinplate"
            q = 2
            phi = z -> z^2*log(abs(z))
        else
            error("Wrong type")
        end
    else
        if kind == "multiquadric"
            q = 1
            phi = z -> sqrt(abs(z)^2 + lambda^2)
        else
            error("Wrong type")
        end
    end
    n = length(x)
    q = length(coeff) - n
    approx = 0
    for i = 1:n
        approx = approx + coeff[i]*phi(value - x[i])
    end
    for i = n+1:q
        approx = approx + coeff[i]*Chebyshev(value,n+1-i)
    end
    return approx
end
