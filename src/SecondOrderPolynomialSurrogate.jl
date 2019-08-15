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
    X = ones(eltype(x[1]),n,3)
    X[:,2] = x
    X[:,3] = x.^2
    β = pinv(X'*X)*X'*y
    return SecondOrderPolynomialSurrogate(X,y,β,lb,ub)
end

function (sec_ord::SecondOrderPolynomialSurrogate)(val::Number)
    return sec_ord.β[1] + sec_ord.β[2]*val + sec_ord.β[3]*val^2
end

function SecondOrderPolynomialSurrogate(x,y,lb,ub)
    n = length(x)
    d = length(lb)
    X = ones(eltype(x[1]),n,1+d+(d-1)+d)
    for j = 1:d
        X[:,j+1] =[x[i][j] for i=1:n]
    end
    for j = 1:d-1
        X[:,j+d+1] = [x[i][j]*x[i][j+1] for i = 1:n]
    end
    for j = 1:d
        X[:,j+2*d] = [x[i][j]^2 for i=1:n]
    end
    β = pinv(X'*X)*X'*y

    return SecondOrderPolynomialSurrogate(X,y,β,lb,ub)
end

function (my_second_ord::SecondOrderPolynomialSurrogate)(val)
    #just create the val vector as X and multiply
    d = length(val)
    X = ones(eltype(val[1]),1,3*d)
    for j = 1:d
        X[j+1] = val[j]
    end
    for j = 1:d-1
        X[j+d+1] = val[j]*val[j+1]
    end
    for j = 1:d
        X[j+2*d] = val[j]^2
    end
    return X*my_second_ord.β
end

function add_point!(my_second::SecondOrderPolynomialSurrogate,x_new,y_new)
    if length(my_second.lb) == 1
        #1D
        for j = 1:length(x_new)
            new = [1 x_new[j] x_new[j]^2]
            my_second.x = vcat(my_second.x,new)
        end
        my_second.y = vcat(my_second.y,y_new)
        my_second.β = pinv(my_second.x'*my_second.x)*my_second.x'*my_second.y
    else
        #ND
        my_second.y = vcat(my_second.y,y_new)
        my_second.β = pinv(my_second.x'*my_second.x)*my_second.x'*my_second.y
    end
    nothing
end
