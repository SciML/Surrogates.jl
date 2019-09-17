"""
mutable struct InverseDistanceSurrogate{X,Y,P,L,U} <: AbstractSurrogate

The square polynomial model can be expressed by 𝐲 = 𝐗β + ϵ, with β = 𝐗ᵗ𝐗⁻¹𝐗ᵗ𝐲
"""
mutable struct SecondOrderPolynomialSurrogate{X,Y,B,L,U} <: AbstractSurrogate
    x::X
    y::Y
    β::B
    lb::L
    ub::U
end

function SecondOrderPolynomialSurrogate(x,y,lb::Number,ub::Number)
    n = length(x)
    d = 1
    X = ones(eltype(x[1]),n,3*d)
    X[:,2] = x
    X[:,3] = x.^2
    β = X\y
    return SecondOrderPolynomialSurrogate(x,y,β,lb,ub)
end

function (sec_ord::SecondOrderPolynomialSurrogate)(val::Number)
    return sec_ord.β[1] + sec_ord.β[2]*val + sec_ord.β[3]*val^2
end

function SecondOrderPolynomialSurrogate(x,y,lb,ub)
    n = length(x)
    d = length(lb)
    X = ones(eltype(x[1]),n,3*d)
    for j = 1:d
        X[:,j+1] =[x[i][j] for i=1:n]
    end
    for j = 1:d-1
        X[:,j+d+1] = [x[i][j]*x[i][j+1] for i = 1:n]
    end
    for j = 1:d
        X[:,j+2*d] = [x[i][j]^2 for i=1:n]
    end
    β = X\y
    return SecondOrderPolynomialSurrogate(x,y,β,lb,ub)
end

function (my_second_ord::SecondOrderPolynomialSurrogate)(val)
    #just create the val vector as X and multiply
    d = length(val)
    X = [[one(eltype(val[1]))]; [val[j] for j =1:d]; [val[j]*val[j+1] for j = 1:d-1]; [val[j]^2 for j = 1:d]]
    return my_second_ord.β'*X
end

function add_point!(my_second::SecondOrderPolynomialSurrogate,x_new,y_new)
    d = length(my_second.lb)
    if d == 1
        #1D
        my_second.x = vcat(my_second.x,x_new)
        my_second.y = vcat(my_second.y,y_new)
        n = length(my_second.x)
        X = ones(eltype(my_second.x[1]),n,3*d)
        X[:,2] = my_second.x
        X[:,3] = my_second.x.^2
        my_second.β = X\my_second.y
    else
        #ND
        my_second.x = vcat(my_second.x,x_new)
        my_second.y = vcat(my_second.y,y_new)
        n = length(my_second.x)
        X = ones(eltype(my_second.x[1]),n,3*d)
        for j = 1:d
            X[:,j+1] =[my_second.x[i][j] for i=1:n]
        end
        for j = 1:d-1
            X[:,j+d+1] = [my_second.x[i][j]*my_second.x[i][j+1] for i = 1:n]
        end
        for j = 1:d
            X[:,j+2*d] = [my_second.x[i][j]^2 for i=1:n]
        end
        my_second.β = X\my_second.y
    end
    nothing
end
