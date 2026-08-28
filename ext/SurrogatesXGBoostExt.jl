module SurrogatesXGBoostExt

using Surrogates: Surrogates, XGBoostSurrogate
using XGBoost: xgboost, predict

import SurrogatesBase

"""
    XGBoostSurrogate(x, y, lb, ub, num_round)

Build a tree-boosted surrogate. `num_round` is the number of boosting rounds.

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of input data points.

## Keyword Arguments

  - `num_round`: number of boosting rounds.
"""
function Surrogates.XGBoostSurrogate(x, y, lb, ub; num_round::Int = 1)
    X = Array{Float64, 2}(undef, length(x), length(x[1]))
    if length(lb) == 1
        for j in eachindex(x)
            X[j, 1] = x[j]
        end
    else
        for j in eachindex(x)
            X[j, :] .= x[j]
        end
    end
    bst = xgboost((X, y); num_round)
    return XGBoostSurrogate(X, y, bst, lb, ub, num_round)
end

function (xgb::XGBoostSurrogate)(val::Number)
    return xgb([val])
end

function (xgb::XGBoostSurrogate)(val)
    return predict(xgb.bst, reshape(collect(val), length(val), 1))[1]
end

function SurrogatesBase.update!(xgb::XGBoostSurrogate, x_new, y_new)
    if x_new isa Tuple
        x_new = reduce(hcat, x_new)
    elseif x_new isa Vector{<:Tuple}
        x_new = reduce(hcat, collect.(x_new))
    elseif x_new isa Vector
        if size(x_new) == (1,) && size(x_new[1]) == ()
            x_new = hcat(x_new)'
        else
            x_new = reduce(hcat, x_new)'
        end
    end
    xgb.x = vcat(xgb.x, x_new)
    xgb.y = vcat(xgb.y, y_new)
    if length(xgb.lb) == 1
        xgb.bst = xgboost(
            (xgb.x, xgb.y);
            num_round = xgb.num_round
        )
    else
        xgb.bst = xgboost(
            (xgb.x, xgb.y); num_round = xgb.num_round
        )
    end
    return nothing
end

end # module
