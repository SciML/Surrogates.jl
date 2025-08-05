module SurrogatesXGBoostExt

using Surrogates: XGBoostSurrogate
using SurrogatesBase
using XGBoost: xgboost, predict

"""
    XGBoostSurrogate(x, y, lb, ub, num_round)

Build Random forest surrogate. num_round is the number of trees.

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of output data points.

## Keyword Arguments

  - `num_round`: number of rounds of training.
"""
function XGBoostSurrogate(x, y, lb, ub; num_round::Int = 1)
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
    XGBoostSurrogate(X, y, bst, lb, ub, num_round)
end

function (rndfor::XGBoostSurrogate)(val::Number)
    return rndfor([val])
end

function (rndfor::XGBoostSurrogate)(val)
    return predict(rndfor.bst, reshape(collect(val), length(val), 1))[1]
end

function SurrogatesBase.update!(rndfor::XGBoostSurrogate, x_new, y_new)
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
    rndfor.x = vcat(rndfor.x, x_new)
    rndfor.y = vcat(rndfor.y, y_new)
    if length(rndfor.lb) == 1
        rndfor.bst = xgboost((rndfor.x, rndfor.y);
            num_round = rndfor.num_round)
    else
        rndfor.bst = xgboost(
            (rndfor.x, rndfor.y); num_round = rndfor.num_round)
    end
    nothing
end

end # module
