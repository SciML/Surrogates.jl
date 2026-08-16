"""
    InverseDistanceSurrogate(x, y, lb, ub; p = 1.0)

Construct an inverse-distance-weighted (Shepard) interpolating surrogate. At an
existing sample point it returns the recorded response; elsewhere it returns
the response average weighted by inverse distance raised to `p`.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: responses corresponding to `x`.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.
  - `p`: positive inverse-distance power.

# Arguments

  - `x`: training inputs.
  - `y`: training responses, with one response per input.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Keywords

  - `p::Number = 1.0`: positive exponent applied to inverse distances. Values
    greater than one give an interpolant that is differentiable at the data
    points; `p ≤ 0` is rejected with an `ArgumentError`.

# Returns

A callable `InverseDistanceSurrogate` supporting
`update!(surrogate, x_new, y_new)`. Query points whose inverse-distance weight
overflows (numerically coincident with a training point) return that training
response, or the average over all such points if several coincide.

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
    if p <= 0
        throw(ArgumentError("Inverse-distance exponent p must be positive! Got: $p."))
    end
    return InverseDistanceSurrogate(x, y, lb, ub, p)
end

function (inverSurr::InverseDistanceSurrogate)(val)
    _check_dimension(inverSurr, val)
    w = [norm(val .- inverSurr.x[i])^(-inverSurr.p) for i in eachindex(inverSurr.x)]
    w_max = maximum(w)
    if isinf(w_max)
        # val (numerically) coincides with at least one training point: return
        # the corresponding response instead of evaluating Inf/Inf.
        hits = findall(isinf, w)
        return sum(inverSurr.y[i] for i in hits) / length(hits)
    end
    # Normalize by the largest weight so the sums below cannot overflow even
    # for query points very close to a training point.
    w_scaled = w ./ w_max
    num = sum(w_scaled[i] * inverSurr.y[i] for i in eachindex(w_scaled))
    return num / sum(w_scaled)
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
