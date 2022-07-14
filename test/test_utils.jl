#=
Properties that surrogates should typically satisfy:

1) Interpolate the data

=#

# Check that surrogate correctly interpolates data
function _check_interpolation(surr; tol = sqrt(eps(Float64)))
    N = length(surr.y)
    pred = surr.(surr.x)
    errs = abs.(pred .- surr.y)
    mean_abs_err = sum(errs) / N
    return mean_abs_err ≤ tol
end

# Generate a surrogate of the provided type with a random dimension and number of points
# By default, the function to be modeled is the l2-norm function
function _random_surrogate(surr_type, func = norm, sampler = SobolSample(); kwargs...)
    d = rand(1:10)
    n = rand(1:3) * 10 * d + 1

    if d == 1
        lb = -2.0
        ub = 2.0
    else
        lb = -2 * ones(d)
        ub = 2 * ones(d)
    end

    x = Surrogates.sample(n, lb, ub, sampler)
    y = func.(x)

    surr = surr_type(x, y, lb, ub; kwargs...)
    return surr
end
