module SurrogatesXGBoostExt

using Surrogates: Surrogates, XGBoostSurrogate
using XGBoost: xgboost, predict

import SurrogatesBase

# XGBoost consumes a samples-by-features matrix. The surrogate stores a vector
# of points, as every other surrogate does, and the matrix is built here at the
# boundary so that `length(xgb.x)` counts samples and `xgb.x[i]` is a point.
"""
    XGBoostSurrogate(x, y, lb, ub; num_round = 1)

Build a tree-boosted surrogate. `num_round` is the number of boosting rounds.

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of input data points.

## Keyword Arguments

  - `num_round`: number of boosting rounds.
"""
function _rows_matrix(x)
    n = length(x)
    d = first(x) isa Number ? 1 : length(first(x))
    X = Array{Float64, 2}(undef, n, d)
    if d == 1
        for j in eachindex(x)
            X[j, 1] = first(x[j])
        end
    else
        for j in eachindex(x)
            X[j, :] .= collect(x[j])
        end
    end
    return X
end

function Surrogates.XGBoostSurrogate(x, y, lb, ub; num_round::Int = 1)
    if num_round < 1
        throw(
            ArgumentError(
                "XGBoostSurrogate needs at least one boosting round! Got: " *
                    "num_round = $(num_round). Zero or fewer leaves the booster " *
                    "untrained, so the surrogate would predict a constant."
            )
        )
    end
    bst = xgboost((_rows_matrix(x), y); num_round)
    return XGBoostSurrogate(collect(x), collect(y), bst, lb, ub, num_round)
end

function (xgb::XGBoostSurrogate)(val::Number)
    return xgb([val])
end

# The dimension check is this package's, not XGBoost's: `predict` accepts a
# matrix with the wrong number of features and answers anyway, so a scalar query
# against a multidimensional model returned a number rather than raising.
function (xgb::XGBoostSurrogate)(val)
    Surrogates._check_dimension(xgb, val)
    return predict(xgb.bst, reshape(collect(val), length(val), 1))[1]
end

function SurrogatesBase.update!(xgb::XGBoostSurrogate, x_new, y_new)
    # `_is_single_sample` is the core's rule for telling one new sample from a
    # batch of them.
    added_x, added_y = if Surrogates._is_single_sample(x_new, first(xgb.x))
        ([Surrogates._match_stored(first(xgb.x), x_new)], [y_new])
    else
        ([Surrogates._match_stored(first(xgb.x), p) for p in x_new], collect(y_new))
    end
    xgb.x = vcat(xgb.x, added_x)
    xgb.y = vcat(xgb.y, added_y)
    xgb.bst = xgboost((_rows_matrix(xgb.x), xgb.y); num_round = xgb.num_round)
    return nothing
end


# ---- SurrogatesBase parameter interface -----------------------------------

SurrogatesBase.parameters(x::XGBoostSurrogate) = (; bst = x.bst)
SurrogatesBase.hyperparameters(x::XGBoostSurrogate) = (; num_round = x.num_round)

end # module
