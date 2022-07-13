remove_tracker(x) = x

_match_container(y, y_el::Number) = first(y)
_match_container(y, y_el) = y

_expected_dimension(x) = length(x[1])

function _check_dimension(surr, input)
    expected_dim = _expected_dimension(surr.x)
    input_dim = length(input)

    if input_dim != expected_dim
        throw(
            ArgumentError("This surrogate expects $expected_dim-dimensional inputs, but the input had dimension $input_dim.")
        )
    end
    return nothing
end
