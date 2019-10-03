"""
mutable struct InverseDistanceSurrogate{X,Y,P,C,L,U} <: AbstractSurrogate

The inverse distance weighting model is an interpolating method and the
unknown points are calculated with a weighted average of the sampling points.
p is a positive real number called the power parameter.
p > 1 is needed for the derivative to be continuous

"""
mutable struct InverseDistanceSurrogate{X,Y,P,L,U} <: AbstractSurrogate
    x::X
    y::Y
    p::P
    lb::L
    ub::U
end

function (inverSurr::InverseDistanceSurrogate)(val)
    if val in inverSurr.x
        return inverSurr.y[findall(x->x==val,inverSurr.x)[1]]
    else
        if length(inverSurr.lb) == 1
            num = sum(inverSurr.y[i]*(norm(val .- inverSurr.x[i]))^(-inverSurr.p) for i = 1:length(inverSurr.x))
            den = sum(norm(val .- inverSurr.x[i])^(-inverSurr.p) for i = 1:length(inverSurr.x))
            return num/den
        else
            βᵢ = [norm(val .- inverSurr.x[i]) for i = 1:length(inverSurr.x)]
            num = inverSurr.y'*βᵢ
            den = sum(βᵢ)
            return num/den
        end
    end
end

function add_point!(inverSurr::InverseDistanceSurrogate,x_new,y_new)
    if length(inverSurr.lb) == 1
        #1D
        append!(inverSurr.x,x_new)
        append!(inverSurr.y,y_new)
    else
        #ND
        inverSurr.x = vcat(inverSurr.x,x_new)
        inverSurr.y = vcat(inverSurr.y,y_new)
    end
    nothing
end
