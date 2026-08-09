"""
    LinearSurrogate(x, y, lb, ub)

Fit a linear least-squares surrogate to sampled inputs `x` and responses `y`.
The returned object is callable at new points and implements the SurrogatesBase
deterministic-surrogate interface, including `update!`.

# Fields

  - `x`: sampled scalar points or multidimensional points.
  - `y`: responses corresponding to `x`.
  - `coeff`: fitted linear coefficients.
  - `lb`: lower bound of the modeled domain.
  - `ub`: upper bound of the modeled domain.

# Arguments

  - `x`: training inputs. Use a vector of numbers for one dimension or a vector of
    equal-length point containers for multiple dimensions.
  - `y`: training responses, with one response per element of `x`.
  - `lb`: scalar or vector lower domain bound.
  - `ub`: scalar or vector upper domain bound matching `lb`.

# Returns

A callable `LinearSurrogate`. Calling `surrogate(point)` evaluates the fitted
linear model, while `update!(surrogate, x_new, y_new)` appends observations and
refits its coefficients.

# Example

```julia
using Surrogates

x = [0.0, 1.0, 2.0]
y = 2 .* x
surrogate = LinearSurrogate(x, y, 0.0, 2.0)
surrogate(1.5)
```
"""
mutable struct LinearSurrogate{X, Y, C, L, U} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    coeff::C
    lb::L
    ub::U
end

function LinearSurrogate(x, y, lb::Number, ub::Number)
    ols = lm(reshape(x, length(x), 1), y)
    return LinearSurrogate(x, y, coef(ols), lb, ub)
end

function SurrogatesBase.update!(my_linear::LinearSurrogate, new_x, new_y)
    if length(my_linear.lb) == 1
        #1D
        my_linear.x = vcat(my_linear.x, new_x)
        my_linear.y = vcat(my_linear.y, new_y)
        md = lm(reshape(my_linear.x, length(my_linear.x), 1), my_linear.y)
        my_linear.coeff = coef(md)
    else
        #ND
        n_previous = length(my_linear.x)
        a = vcat(my_linear.x, new_x)
        n_after = length(a)
        dim_new = n_after - n_previous
        n = length(my_linear.x)
        d = length(my_linear.x[1])
        tot_dim = n + dim_new
        X = Array{Float64, 2}(undef, tot_dim, d)
        for j in 1:n
            X[j, :] = vec(collect(my_linear.x[j]))
        end
        if dim_new == 1
            X[n + 1, :] = vec(collect(new_x))
        else
            i = 1
            for j in (n + 1):tot_dim
                X[j, :] = vec(collect(new_x[i]))
                i = i + 1
            end
        end
        my_linear.x = vcat(my_linear.x, new_x)
        my_linear.y = vcat(my_linear.y, new_y)
        md = lm(X, my_linear.y)
        my_linear.coeff = coef(md)
    end
    return nothing
end

function (lin::LinearSurrogate)(val::Number)
    _check_dimension(lin, val)
    return lin.coeff[1] * val
end

function (lin::LinearSurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(lin, val)
    return lin.coeff' * [val...]
end

function LinearSurrogate(x, y, lb, ub)
    #X = Array{eltype(x[1]),2}(undef,length(x),length(x[1]))
    #=
    X = Array{eltype(x[1]),2}(undef,length(x),length(x[1]))
    for j = 1:length(x)
        X[j,:] = vec(collect(x[j]))
    end
    =#
    T = collect(reshape(collect(Base.Iterators.flatten(x)), (length(x[1]), length(x)))')
    #T = transpose(reshape(reinterpret(eltype(x[1]), x), length(x[1]), length(x)))
    X = Array{eltype(x[1]), 2}(undef, length(x), length(x[1]))
    X = copy(T)
    ols = lm(X, y)
    return LinearSurrogate(x, y, coef(ols), lb, ub)
end
