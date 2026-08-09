"""
    InverseDistanceSurrogate(x, y, lb, ub; p = 1.0)

Construct an inverse-distance-weighted interpolating surrogate. At an existing
sample point it returns the recorded response; elsewhere it returns the response
average weighted by inverse distance raised to `p`.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `p`: positive inverse-distance power. Values greater than one provide a
    differentiable interpolant away from coincident points.

# Arguments

  - `x`: training inputs.
  - `y`: training responses, with one response per input.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `p::Number = 1.0`: exponent applied to inverse distances.

# Returns

A callable `InverseDistanceSurrogate` supporting
`update!(surrogate, x_new, y_new)`.

# Example

```julia
using Surrogates

x = [0.0, 1.0, 2.0]
y = x .^ 2
surrogate = InverseDistanceSurrogate(x, y, 0.0, 2.0; p = 2.0)
surrogate(0.5)
```
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
