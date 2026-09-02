_match_container(y, y_el::Number) = first(y)
_match_container(y, y_el) = y

"""
    _construct_y_matrix(y, y_el)

Responses arranged for a least-squares solve.

Scalar responses are already a vector, so they pass through and the resulting
coefficients stay a vector. Vector-valued responses become an `n x m` matrix,
one column per output, so a single `\\` solves all outputs at once.
"""
_construct_y_matrix(y, y_el::Number) = y
_construct_y_matrix(y, y_el) = [y[i][j] for i in 1:length(y), j in 1:length(y_el)]

_expected_dimension(x) = length(x[1])

function _check_dimension(surr, input)
    expected_dim = _expected_dimension(surr.x)
    input_dim = length(input)

    if input_dim != expected_dim
        throw(ArgumentError("This surrogate expects $expected_dim-dimensional inputs, but the input had dimension $input_dim."))
    end
    return nothing
end

"""
    _is_single_sample(new_x, x_el) -> Bool

Whether `new_x` is one sample rather than a collection of them, judged against
`x_el`, an existing sample.

For one-dimensional inputs a sample is a number, so anything else is a
collection. For `d`-dimensional inputs a sample is a `d`-element container of
numbers; a collection of samples either has a different length or holds
containers rather than numbers, which is what distinguishes `(1.0, 2.0)` from
`[(1.0, 2.0), (3.0, 4.0)]` when `d == 2`.
"""
_is_single_sample(new_x, x_el::Number) = new_x isa Number
function _is_single_sample(new_x, x_el)
    return length(new_x) == length(x_el) && first(new_x) isa Number
end

"""
    _append_samples(x, y, new_x, new_y) -> (x, y)

Append new observations to a surrogate's sample containers and return the
extended pair.

Plain `vcat` will not do: for multi-output responses a *single* `new_y` is
itself a vector, and `vcat` would splat it into the response list.

The caller's arrays are not mutated, so a surrogate built from a user's vector
will not grow that vector behind their back.
"""
function _append_samples(x, y, new_x, new_y)
    if _is_single_sample(new_x, first(x))
        return vcat(x, [new_x]), vcat(y, [new_y])
    end
    return vcat(x, new_x), vcat(y, new_y)
end

"""
    _as_point(val)

Flatten an array-shaped query point.

A point may arrive as a tuple, a vector, or — when derived from bounds written
as row matrices, e.g. `(lb .+ ub) ./ 2` — as a 1×d matrix. The matrix form has
to be flattened before use: it carries a second dimension that leaks into
broadcasts (`Matrix(1×d) .- Tuple(d)` gives a d×d outer difference) and into
comprehensions. `vec` is a no-op view for an ordinary vector.
"""
_as_point(val::AbstractArray) = vec(val)
_as_point(val) = val

"""
    _matrix_rank(A)

Rank of `A`, or `nothing` when it cannot be computed.

`rank` needs an SVD, which is only available for BLAS element types. Callers use
this to give a clear error on a rank-deficient design; for generic element types
the check is skipped and the deficiency surfaces as a singular factorization
instead.
"""
_matrix_rank(A::AbstractMatrix{<:Union{Float32, Float64}}) = rank(A)
_matrix_rank(A) = nothing

"""
    _multistart_optimize(f, u0, lo, hi; n_start = 10, multistart = true) -> (u, value)

Minimize `f(u, _)` over the box `[lo, hi]` with Nelder-Mead, from `u0` and, when
`multistart` is set, from `n_start` Latin-hypercube points as well.

Nelder-Mead is derivative-free; passing bounds to `OptimizationOptimJL` would
make it wrap the solver in `Fminbox`, which needs gradients and errors. The box
is therefore enforced by clamping, both on the starts and on the result, and
`f` is expected to clamp as well.

The whole kriging family fits its correlation scales this way, so the search
lives here rather than beside any one of them.
"""
function _multistart_optimize(
        f, u0, lo, hi; n_start::Integer = 10, multistart = true, maxiters = nothing
    )
    n = length(u0)
    starts = if multistart && n_start > 0
        pts = sample(n_start, lo, hi, LatinHypercubeSample())
        # `sample` gives a Vector{Float64} for one-dimensional bounds and a
        # Vector{NTuple} otherwise; normalize both to Vector{Vector{Float64}}.
        lhs = n == 1 ? [[p] for p in pts] : [collect(Float64, p) for p in pts]
        vcat([clamp.(collect(Float64, u0), lo, hi)], lhs)
    else
        [clamp.(collect(Float64, u0), lo, hi)]
    end

    best_value = Inf
    best_u = starts[1]
    for x0 in starts
        sol = try
            prob = OptimizationProblem(f, collect(x0), nothing)
            maxiters === nothing ? solve(prob, NelderMead()) :
                solve(prob, NelderMead(); maxiters = maxiters)
        catch e
            e isa Union{SingularException, PosDefException} || rethrow()
            continue
        end
        if isfinite(sol.objective) && sol.objective < best_value
            best_value = sol.objective
            best_u = sol.u
        end
    end
    return clamp.(best_u, lo, hi), best_value
end

"""
    _surrogate_eltype(x)

The floating-point element type a fit should be carried out in.

Every default and constant in the kriging family is taken to this before use, so
that a `Float64` literal cannot silently promote a `Float32` design.
"""
_surrogate_eltype(x) = float(eltype(first(x)))

"""
    _default_p(x, x_el)

The default correlation smoothness, `2`, at the samples' own precision. Scalar
for a one-dimensional design, one entry per coordinate otherwise.
"""
_default_p(x, ::Number) = 2 * one(_surrogate_eltype(x))
_default_p(x, _) = fill(2 * one(_surrogate_eltype(x)), length(x[1]))

"""
    _check_no_duplicate_samples(name, x)

Reject repeated sample points, which make the correlation matrix singular.
"""
function _check_no_duplicate_samples(name, x)
    if length(x) != length(unique(x))
        throw(
            ArgumentError(
                "There is a repetition in the samples, cannot build $name: " *
                    "duplicate points make the correlation matrix singular."
            )
        )
    end
    return nothing
end

"""
    _deprecated_inverse_of_R(k, kind)

Materialize `R⁻¹` from a stored Cholesky factorization, warning that the field it
replaced is gone. Shared by the surrogates that used to store the inverse.
"""
function _deprecated_inverse_of_R(k, kind)
    Base.depwarn(
        "`$kind.inverse_of_R` is deprecated: the Cholesky factorization of the " *
            "regularized $(kind === :Kriging ? "correlation" : "covariance") matrix is " *
            "stored in `R_fact`. Prefer solving with `k.R_fact \\ v` over forming the " *
            "inverse.",
        :getproperty
    )
    return inv(getfield(k, :R_fact))
end

"""
    _blup_std_error(sigma, r, f, F)

Standard error of the best linear unbiased predictor, given the process variance
`sigma`, the cross-covariance `r` between the query point and the observations,
the trend basis `f` over the observations, and a factorization `F` of the
regularized covariance matrix:

    s²(x) = σ² [ 1 - rᵀR⁻¹r + (1 - fᵀR⁻¹r)² / (fᵀR⁻¹f) ]

The third term is the variance contributed by estimating the trend, so its
numerator is the *trend* residual `1 - fᵀR⁻¹r`, not the prediction residual
`1 - rᵀR⁻¹r`. For ordinary kriging `f` is `𝟙`; `GEK` zeroes it on the derivative
rows, since a gradient carries no information about the process mean.

As a variance the bracket is non-negative in exact arithmetic, so a negative
value is round-off and clamps to zero. Reflecting it with `abs` would turn an
ill-conditioned solve into a plausible-looking error bar.
"""
function _blup_std_error(sigma, r, f, F)
    Rinv_r = F \ r
    a = dot(r, Rinv_r)
    c = dot(f, Rinv_r)
    b = dot(f, F \ f)
    mean_squared_error = sigma * (1 - a + (1 - c)^2 / b)
    return sqrt(max(mean_squared_error, zero(mean_squared_error)))
end

# How far, in decades, the fitted correlation scale may move from the
# data-derived default. Relative rather than absolute: the default is the inverse
# sample spread, so it already carries the right order of magnitude, where a box
# in absolute units would have to span forty decades to cover designs whose
# coordinates differ by ten orders of magnitude.
const _KRIGING_THETA_DECADES = 5.0

# Nelder-Mead defaults to 1000 iterations per start, far more than a handful of
# correlation scales needs.
const _KRIGING_MAXITERS = 250

# Latin-hypercube starts in addition to the data-derived one.
const _KRIGING_N_START = 4

"""
    _fit_theta(loglik, theta0; n_start, multistart, maxiters)

Maximize a concentrated log-likelihood over the correlation scale.

The search runs in `log₁₀ θ`, which makes it scale-free and enforces positivity
automatically, over a box of `_KRIGING_THETA_DECADES` decades either side of
`theta0`. The box is relative rather than absolute because `theta0` already
carries the right order of magnitude, where a box in absolute units would have to
span forty decades to cover designs whose coordinates differ by ten orders of
magnitude.

`loglik` takes a correlation scale shaped like `theta0` and returns a number;
`-Inf` marks a scale whose matrix cannot be factorized, so the search walks away
from it rather than raising. If every start fails, `theta0` is returned rather
than an unendorsed scale.
"""
function _fit_theta(
        loglik, theta0; n_start::Integer = _KRIGING_N_START, multistart = true,
        maxiters = _KRIGING_MAXITERS
    )
    scalar = theta0 isa Number
    # Always in Float64: the Latin-hypercube sampler needs a bits type, and the
    # extra precision buys nothing for a correlation scale. Converted back below.
    u0 = log10.(Float64.(scalar ? [theta0] : collect(theta0)))
    lo = u0 .- _KRIGING_THETA_DECADES
    hi = u0 .+ _KRIGING_THETA_DECADES
    negll(u, _) = begin
        theta = 10.0 .^ clamp.(u, lo, hi)
        v = loglik(scalar ? theta[1] : theta)
        return isfinite(v) ? -v : Inf
    end
    u, value = _multistart_optimize(
        negll, u0, lo, hi; n_start = n_start, multistart = multistart,
        maxiters = maxiters
    )
    isfinite(value) || return theta0
    # Back to the design's own element type, so a Float32 fit stays Float32.
    theta = 10.0 .^ u
    return scalar ? oftype(theta0, theta[1]) : convert(typeof(theta0), theta)
end
