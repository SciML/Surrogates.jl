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
