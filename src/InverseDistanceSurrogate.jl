"""
mutable struct InverseDistanceSurrogate{X,Y,P,C,L,U} <: AbstractSurrogate

The inverse distance weighting model is an interpolating method and the
unknown points are calculated with a weighted average of the sampling points.
p is a positive real number called the power parameter.
p > 1 is needed for the derivative to be continuous.
"""
mutable struct InverseDistanceSurrogate{X, Y, L, U, P} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    p::P
end

function InverseDistanceSurrogate(x, y, lb, ub; p::Number = 1.0)
    return InverseDistanceSurrogate(x, y, lb, ub, p)
end

function (inverSurr::InverseDistanceSurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(inverSurr, val)

    if val in inverSurr.x
        return inverSurr.y[findfirst(x -> x == val, inverSurr.x)]
    else
        if length(inverSurr.lb) == 1
            num = sum(
                inverSurr.y[i] * (norm(val .- inverSurr.x[i]))^(-inverSurr.p)
                    for i in 1:length(inverSurr.x)
            )
            den = sum(
                norm(val .- inverSurr.x[i])^(-inverSurr.p)
                    for i in 1:length(inverSurr.x)
            )
            return num / den
        else
            βᵢ = [norm(val .- inverSurr.x[i])^(-inverSurr.p) for i in 1:length(inverSurr.x)]
            num = sum(inverSurr.y[i] * βᵢ[i] for i in 1:length(inverSurr.y))
            den = sum(βᵢ)
            return num / den
        end
    end
end

function SurrogatesBase.update!(inverSurr::InverseDistanceSurrogate, x_new, y_new)
    if eltype(x_new) == eltype(inverSurr.x)
        #1D
        append!(inverSurr.x, x_new)
        append!(inverSurr.y, y_new)
    else
        #ND
        push!(inverSurr.x, x_new)
        push!(inverSurr.y, y_new)
    end
    return nothing
end
