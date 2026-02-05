module SurrogatesSVMExt

using Surrogates: SVMSurrogate, Surrogates
using SurrogatesBase
using LIBSVM

"""
    SVMSurrogate(x, y, lb, ub)

Builds a SVM Surrogate using [LIBSVM](https://github.com/JuliaML/LIBSVM.jl).

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of output data points.
"""
function Surrogates.SVMSurrogate(x, y, lb, ub)
    X = Array{Float64, 2}(undef, length(x), length(first(x)))
    if length(lb) == 1
        for j in eachindex(x)
            X[j, 1] = x[j]
        end
    else
        for j in eachindex(x)
            X[j, :] .= x[j]
        end
    end
    model = LIBSVM.fit!(SVC(), X, y)
    return SVMSurrogate(x, y, model, lb, ub)
end

function (svmsurr::SVMSurrogate)(val::Number)
    return svmsurr([val])
end

function (svmsurr::SVMSurrogate)(val)
    n = length(val)
    return LIBSVM.predict(svmsurr.model, reshape(collect(val), 1, n))[1]
end

"""
    update!(svmsurr::SVMSurrogate, x_new, y_new)

## Arguments

  - `svmsurr`: Surrogate of type [`SVMSurrogate`](@ref).
  - `x_new`: Vector of new data points to be added to the training set of SVMSurrogate.
  - `y_new`: Vector of new output points to be added to the training set of SVMSurrogate.
"""
function SurrogatesBase.update!(svmsurr::SVMSurrogate, x_new, y_new)
    svmsurr.x = vcat(svmsurr.x, x_new)
    svmsurr.y = vcat(svmsurr.y, y_new)
    return if length(svmsurr.lb) == 1
        svmsurr.model = LIBSVM.fit!(
            SVC(), reshape(svmsurr.x, length(svmsurr.x), 1), svmsurr.y
        )
    else
        svmsurr.model = LIBSVM.fit!(
            SVC(), transpose(reduce(hcat, collect.(svmsurr.x))), svmsurr.y
        )
    end
end

end # module
