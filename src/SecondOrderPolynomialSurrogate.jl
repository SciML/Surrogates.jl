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

function SecondOrderPolynomialSurrogate(x,y,lb,ub)
    X = _construct_2nd_order_interp_matrix(x, first(x))
    Y = _construct_y_matrix(y, first(y))
    β = X\Y
    return SecondOrderPolynomialSurrogate(x, y, β, lb, ub)
end

function _construct_2nd_order_interp_matrix(x, x_el)
    n = length(x)
    d = length(x_el)
    D = 1 + 2*d + d*(d-1)÷2
    X = ones(eltype(x_el), n, D)
    for i = 1:n, j = 1:d
        X[i, j+1] = x[i][j]
    end
    for i = 1:n, j = 1:d, k = 1:j-1
        idx = j + (k*(k-1)÷2)
        X[i, 1+d+idx] = x[i][j]*x[i][end-k+1]
    end
    for i = 1:n, j = 1:d
        X[i, j+1+d+d*(d-1)÷2] = x[i][j]^2
    end
    return X
end
function _construct_y_matrix(y, y_el::Number)
    return y
end
function _construct_y_matrix(y, y_el)
    Y = [y[i][j] for i=1:length(y), j=1:length(y_el)]
    return Y
end

function (my_second_ord::SecondOrderPolynomialSurrogate)(val)
    #just create the val vector as X and multiply
    d = length(val)

    y = my_second_ord.β[1, :]
    for j = 1:d
        #X[j+1] = val[j]
        y += val[j]*my_second_ord.β[j+1, :]
    end
    for j = 1:d, k = 1:j-1
        idx = j + (k*(k-1)÷2)
        y += val[j] * val[end-k+1] * my_second_ord.β[1+d+idx, :]
        #X[1+d+idx] = val[j]*val[end-k+1]
    end
    for j = 1:d
        #X[j + 1 + d + d*(d-1)÷2] = val[j]^2
        y += val[j]^2 * my_second_ord.β[j+1+d+d*(d-1)÷2, :]
    end
    return _match_container(y, first(my_second_ord.y))
end
_match_container(y, y_el::Number) = first(y)
_match_container(y, y_el) = y

function add_point!(my_second::SecondOrderPolynomialSurrogate, x_new, y_new)
    if eltype(x_new) == eltype(my_second.x)
        append!(my_second.x, x_new)
        append!(my_second.y, y_new)
    else
        push!(my_second.x, x_new)
        push!(my_second.y, y_new)
    end
    X = _construct_2nd_order_interp_matrix(my_second.x, first(my_second.x))
    Y = _construct_y_matrix(my_second.y, first(my_second.y))
    β = X\Y
    my_second.β = β
    nothing
end
