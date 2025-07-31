"""
mutable struct InverseDistanceSurrogate{X,Y,P,L,U} <: AbstractSurrogate

The square polynomial model can be expressed by 𝐲 = 𝐗β + ϵ, with β = 𝐗ᵗ𝐗⁻¹𝐗ᵗ𝐲
"""
mutable struct SecondOrderPolynomialSurrogate{X, Y, B, L, U} <:
               AbstractDeterministicSurrogate
    x::X
    y::Y
    β::B
    lb::L
    ub::U
end

function SecondOrderPolynomialSurrogate(x, y, lb, ub)
    X = _construct_2nd_order_interp_matrix(x, first(x))
    Y = _construct_y_matrix(y, first(y))
    β = X \ Y
    return SecondOrderPolynomialSurrogate(x, y, β, lb, ub)
end

function _construct_2nd_order_interp_matrix(x, x_el)
    n = length(x)
    d = length(x_el)
    D = 1 + 2 * d + d * (d - 1) ÷ 2
    X = ones(eltype(x_el), n, D)
    for i in 1:n, j in 1:d

        X[i, j + 1] = x[i][j]
    end
    idx = d + 1
    for j in 1:d, k in (j + 1):d

        idx += 1
        for i in 1:n
            X[i, idx] = x[i][j] * x[i][k]
        end
    end
    for i in 1:n, j in 1:d

        X[i, j + 1 + d + d * (d - 1) ÷ 2] = x[i][j]^2
    end
    return X
end

_construct_y_matrix(y, y_el::Number) = y
_construct_y_matrix(y, y_el) = [y[i][j] for i in 1:length(y), j in 1:length(y_el)]

function (my_second_ord::SecondOrderPolynomialSurrogate)(val)

    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(my_second_ord, val)

    #just create the val vector as X and multiply
    d = length(val)

    y = my_second_ord.β[1, :]
    for j in 1:d
        y += val[j] * my_second_ord.β[j + 1, :]
    end
    idx = d + 1
    for j in 1:d, k in (j + 1):d

        idx += 1
        y += val[j] * val[k] * my_second_ord.β[idx, :]
    end
    for j in 1:d
        y += val[j]^2 * my_second_ord.β[j + 1 + d + d * (d - 1) ÷ 2, :]
    end
    return _match_container(y, first(my_second_ord.y))
end

function SurrogatesBase.update!(my_second::SecondOrderPolynomialSurrogate, x_new, y_new)
    if eltype(x_new) == eltype(my_second.x)
        append!(my_second.x, x_new)
        append!(my_second.y, y_new)
    else
        push!(my_second.x, x_new)
        push!(my_second.y, y_new)
    end
    X = _construct_2nd_order_interp_matrix(my_second.x, first(my_second.x))
    Y = _construct_y_matrix(my_second.y, first(my_second.y))
    β = X \ Y
    my_second.β = β
    nothing
end
