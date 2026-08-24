"""
    InverseDistanceSurrogate(x, y, lb, ub; p = 1.0)

Construct an inverse-distance-weighted (Shepard) interpolating surrogate. At an
existing sample point it returns the recorded response; elsewhere it returns the
response average weighted by inverse distance raised to `p`.

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
    greater than one give an interpolant that is differentiable at the sample
    points; `p <= 0` is rejected with an `ArgumentError`.

# Returns

A callable `InverseDistanceSurrogate` supporting
`update!(surrogate, x_new, y_new)`. A query point coinciding with one or more
sample points returns that response, or the mean over the coincident ones.

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
    # A row-matrix query would broadcast into a d x d outer difference.
    point = _as_point(val)
    p = inverSurr.p
    d = [norm(point .- inverSurr.x[i]) for i in eachindex(inverSurr.x)]
    d_min = minimum(d)
    # A query on a sample point returns that response, averaged over ties,
    # instead of Inf / Inf. Tested through the weight rather than `d_min == 0`
    # because `norm` of a zero vector of duals is NaN, and a dual carrying a
    # nonzero derivative is not equal to zero.
    if !isfinite(d_min^(-p))
        hits = findall(dᵢ -> !isfinite(dᵢ^(-p)), d)
        return sum(inverSurr.y[i] for i in hits) / length(hits)
    end
    # (d / d_min)^(-p) is the same convex combination as d^(-p) with the
    # largest weight pinned at one, so nothing overflows or underflows.
    w = (d ./ d_min) .^ (-p)
    return sum(w[i] * inverSurr.y[i] for i in eachindex(w)) / sum(w)
end

function SurrogatesBase.update!(inverSurr::InverseDistanceSurrogate, x_new, y_new)
    inverSurr.x, inverSurr.y = _append_samples(inverSurr.x, inverSurr.y, x_new, y_new)
    return nothing
end
